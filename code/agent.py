import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
import re
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ValidationError
from pathlib import Path

import openai
import requests

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional assistant specializing in orchestrating MCP tool invocations via LLM. "
    "Your responsibilities include:\n\n"
    "- Receiving a natural language query from the user.\n"
    "- Using LLM tool invocation to call the appropriate MCP tool hosted on the MCP server.\n"
    "- Sending the user's query and relevant parameters to the MCP tool.\n"
    "- Returning the MCP tool's response to the user in a clear, concise, and professional manner.\n"
    "- If the MCP tool or server is unavailable, provide a helpful and courteous fallback response.\n"
    "- Always maintain a formal and informative tone.\n\n"
    "Output format:\n"
    "- Clearly present the MCP tool's response.\n"
    "- If an error occurs, explain the issue and suggest next steps.\n"
    "- If the requested information or tool is not available, inform the user and offer alternatives."
)
OUTPUT_FORMAT = (
    "- Begin with a brief summary of the action taken.\n"
    "- Present the MCP tool's response in a readable format.\n"
    "- If an error or issue occurs, provide a clear explanation and guidance."
)
FALLBACK_RESPONSE = (
    "I'm unable to process your request at the moment due to a technical issue with the MCP tool or server. "
    "Please try again later or contact support for assistance."
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LOGGING CONFIG
# =========================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# =========================
# INPUT/OUTPUT MODELS
# =========================

class MCPToolInvocationRequest(BaseModel):
    user_query: str = Field(..., description="Natural language query for the MCP tool")
    mcp_tool_id: str = Field(..., description="ID of the MCP tool to invoke")
    mcp_server_url: str = Field(..., description="URL of the MCP server hosting the tool")

    @field_validator("user_query")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_user_query(cls, v):
        if not v or not v.strip():
            raise ValueError("user_query must not be empty")
        if len(v) > 50000:
            raise ValueError("user_query exceeds maximum length (50000 characters)")
        return v.strip()

    @field_validator("mcp_tool_id")
    @classmethod
    def validate_tool_id(cls, v):
        if not v or not v.strip():
            raise ValueError("mcp_tool_id must not be empty")
        return v.strip()

    @field_validator("mcp_server_url")
    @classmethod
    def validate_server_url(cls, v):
        if not v or not v.strip():
            raise ValueError("mcp_server_url must not be empty")
        # Basic URL validation
        if not re.match(r"^https?://", v.strip()):
            raise ValueError("mcp_server_url must start with http:// or https://")
        return v.strip()

class MCPToolInvocationResponse(BaseModel):
    success: bool = Field(..., description="Whether the invocation was successful")
    result: Optional[str] = Field(None, description="The MCP tool's response or error message")
    error: Optional[str] = Field(None, description="Error message if failed")

# =========================
# UTILITY: LLM OUTPUT SANITIZER
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# TOOL VALIDATOR
# =========================

class ToolValidator:
    """
    Validates MCP tool registration and MCP server connectivity.
    """

    def __init__(self, registered_tools: Optional[Dict[str, Any]] = None):
        # In a real system, this would be loaded from a registry or config.
        # Here, we assume all tool_ids are valid except for a hardcoded example.
        self.registered_tools = registered_tools or {}

    def validate_tool(self, tool_id: str) -> bool:
        """
        Checks if MCP tool is registered and available.
        """
        # Simulate registry check: all tools except "invalid_tool" are valid.
        if tool_id in self.registered_tools:
            return True
        if tool_id.lower() == "invalid_tool":
            logger.warning(f"MCP tool not found: {tool_id}")
            return False
        return True

    def validate_server(self, server_url: str) -> bool:
        """
        Checks MCP server connectivity. Retries up to 3 times with exponential backoff.
        """
        max_retries = 3
        delay = 0.5
        for attempt in range(max_retries):
            try:
                _t0 = _time.time()
                resp = requests.get(server_url, timeout=2)
                try:
                    trace_tool_call(
                        tool_name="MCPServerPing",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=f"status_code={resp.status_code}",
                        status="success" if resp.status_code == 200 else "error",
                    )
                except Exception:
                    pass
                if resp.status_code == 200:
                    return True
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: MCP server unreachable: {server_url} ({e})")
                try:
                    trace_tool_call(
                        tool_name="MCPServerPing",
                        latency_ms=0,
                        output=str(e),
                        status="error",
                    )
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    _time.sleep(delay)
                    delay *= 2
        logger.error(f"MCP server unavailable after {max_retries} attempts: {server_url}")
        return False

# =========================
# AUDIT LOGGER
# =========================

class AuditLogger:
    """
    Logs tool invocations, errors, and responses for compliance and monitoring.
    """

    def log_event(self, event_type: str, details: dict) -> None:
        """
        Logs tool invocations, errors, and responses for compliance.
        """
        try:
            logger.info(f"[AUDIT] {event_type}: {json.dumps(details, default=str)[:500]}")
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")

# =========================
# ERROR HANDLER
# =========================

class ErrorHandler:
    """
    Handles errors, retries, fallback logic, and user-facing error responses.
    """

    def __init__(self, fallback_response: str = FALLBACK_RESPONSE):
        self.fallback_response = fallback_response

    def handle_error(self, error_code: str, context: dict) -> str:
        """
        Processes errors, determines retry/fallback, and generates user-facing error messages.
        """
        if error_code == "MCP_TOOL_NOT_FOUND":
            return (
                "The requested MCP tool could not be found or is not registered. "
                "Please verify the tool ID and try again, or contact your administrator."
            )
        elif error_code == "MCP_SERVER_UNAVAILABLE":
            return (
                "The MCP server is currently unavailable or unreachable. "
                "Please check the server URL, ensure network connectivity, and try again later."
            )
        elif error_code == "INVALID_INPUT":
            return (
                "Your request could not be processed due to invalid input. "
                "Please check your query and parameters and try again."
            )
        else:
            return self.fallback_response

# =========================
# LLM SERVICE
# =========================

class LLMService:
    """
    Handles LLM prompt construction, tool call formatting, and manages LLM responses.
    """

    def __init__(self):
        self.client = None

    def get_client(self):
        if self.client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def invoke_tool_call(self, prompt: str, tool_params: dict) -> dict:
        """
        Formats prompt and invokes LLM tool call.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
            {"role": "user", "content": prompt},
        ]
        # Optionally, include tool_params as a user message or in the prompt.
        if tool_params:
            tool_params_str = "\n".join(f"{k}: {v}" for k, v in tool_params.items())
            messages.append({"role": "user", "content": f"Tool Parameters:\n{tool_params_str}"})

        _t0 = _time.time()
        try:
            response = await self.get_client().chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **Config.get_llm_kwargs()
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return {"success": True, "result": content}
        except Exception as e:
            logger.error(f"LLM tool call failed: {e}")
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    status="error",
                    error=e,
                    response_summary=str(e)[:200],
                )
            except Exception:
                pass
            return {"success": False, "error": str(e)}

# =========================
# MCP TOOL INVOKER
# =========================

class MCPToolInvoker:
    """
    Executes tool invocation via MCP server and returns processed output.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def invoke(self, tool_id: str, server_url: str, query: str) -> dict:
        """
        Executes MCP tool invocation via MCP server.
        """
        # Simulate MCP server API call (POST /invoke)
        payload = {
            "tool_id": tool_id,
            "query": query
        }
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        delay = 0.5
        for attempt in range(max_retries):
            try:
                _t0 = _time.time()
                resp = requests.post(
                    f"{server_url.rstrip('/')}/invoke",
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=10
                )
                try:
                    trace_tool_call(
                        tool_name="MCPToolInvoke",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        args=payload,
                        output=f"status_code={resp.status_code}",
                        status="success" if resp.status_code == 200 else "error",
                    )
                except Exception:
                    pass
                if resp.status_code == 200:
                    try:
                        result = resp.json()
                        return {"success": True, "result": result.get("output", str(result))}
                    except Exception:
                        return {"success": True, "result": resp.text}
                else:
                    logger.warning(f"MCP tool invocation failed (status {resp.status_code}): {resp.text}")
                    if attempt < max_retries - 1:
                        _time.sleep(delay)
                        delay *= 2
                    else:
                        return {"success": False, "error": f"MCP tool invocation failed: {resp.text}"}
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: MCP tool invocation error: {e}")
                try:
                    trace_tool_call(
                        tool_name="MCPToolInvoke",
                        latency_ms=0,
                        args=payload,
                        output=str(e),
                        status="error",
                    )
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    _time.sleep(delay)
                    delay *= 2
                else:
                    return {"success": False, "error": str(e)}
        return {"success": False, "error": "MCP tool invocation failed after retries."}

# =========================
# MAIN AGENT CLASS
# =========================

class BaseAgent:
    """
    Abstract base class for agent orchestration.
    """
    pass

class MCPToolInvocationAgent(BaseAgent):
    """
    Coordinates the end-to-end flow: receives user query, validates input, orchestrates LLM tool call, invokes MCP tool, returns response.
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.tool_validator = ToolValidator()
        self.mcp_tool_invoker = MCPToolInvoker(self.llm_service)
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()
        self.guardrails_config = GUARDRAILS_CONFIG

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self, user_query: str, mcp_tool_id: str, mcp_server_url: str) -> dict:
        """
        Receives user query, validates input, orchestrates tool invocation, and returns formatted response.
        """
        async with trace_step(
            "input_validation",
            step_type="parse",
            decision_summary="Validate user query and MCP tool/server parameters",
            output_fn=lambda r: f"valid={r.get('validation_status','?')}",
        ) as step:
            # Input validation
            validation_status = "valid"
            connectivity_status = "connected"
            try:
                # Validate tool
                if not self.tool_validator.validate_tool(mcp_tool_id):
                    self.audit_logger.log_event("tool_validation_failed", {
                        "tool_id": mcp_tool_id,
                        "user_query": user_query
                    })
                    validation_status = "invalid"
                    error_msg = self.error_handler.handle_error("MCP_TOOL_NOT_FOUND", {"tool_id": mcp_tool_id})
                    return {
                        "success": False,
                        "result": None,
                        "error": error_msg
                    }
                # Validate server
                if not self.tool_validator.validate_server(mcp_server_url):
                    self.audit_logger.log_event("server_connectivity_failed", {
                        "server_url": mcp_server_url,
                        "user_query": user_query
                    })
                    connectivity_status = "disconnected"
                    error_msg = self.error_handler.handle_error("MCP_SERVER_UNAVAILABLE", {"server_url": mcp_server_url})
                    return {
                        "success": False,
                        "result": None,
                        "error": error_msg
                    }
            except Exception as e:
                self.audit_logger.log_event("input_validation_error", {
                    "error": str(e),
                    "user_query": user_query,
                    "tool_id": mcp_tool_id,
                    "server_url": mcp_server_url
                })
                error_msg = self.error_handler.handle_error("INVALID_INPUT", {"error": str(e)})
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg
                }
            step.capture({"validation_status": validation_status, "connectivity_status": connectivity_status})

        # Decision table: only proceed if both valid
        if validation_status != "valid" or connectivity_status != "connected":
            self.audit_logger.log_event("invocation_blocked", {
                "validation_status": validation_status,
                "connectivity_status": connectivity_status
            })
            return {
                "success": False,
                "result": None,
                "error": self.error_handler.handle_error("INVALID_INPUT", {})
            }

        # Orchestrate LLM tool call
        async with trace_step(
            "llm_tool_call",
            step_type="llm_call",
            decision_summary="Format prompt and invoke LLM tool call",
            output_fn=lambda r: f"llm_success={r.get('success')}",
        ) as step:
            tool_params = {
                "mcp_tool_id": mcp_tool_id,
                "mcp_server_url": mcp_server_url
            }
            llm_result = await self.llm_service.invoke_tool_call(user_query, tool_params)
            step.capture(llm_result)
            if not llm_result.get("success"):
                self.audit_logger.log_event("llm_tool_call_failed", {
                    "error": llm_result.get("error"),
                    "user_query": user_query,
                    "tool_id": mcp_tool_id,
                    "server_url": mcp_server_url
                })
                error_msg = self.error_handler.handle_error("LLM_TOOL_CALL_FAILED", {"error": llm_result.get("error")})
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg
                }

        # Invoke MCP tool via MCP server
        async with trace_step(
            "mcp_tool_invocation",
            step_type="tool_call",
            decision_summary="Invoke MCP tool via MCP server",
            output_fn=lambda r: f"mcp_success={r.get('success')}",
        ) as step:
            mcp_result = await self.mcp_tool_invoker.invoke(mcp_tool_id, mcp_server_url, user_query)
            step.capture(mcp_result)
            if not mcp_result.get("success"):
                self.audit_logger.log_event("mcp_tool_invocation_failed", {
                    "error": mcp_result.get("error"),
                    "user_query": user_query,
                    "tool_id": mcp_tool_id,
                    "server_url": mcp_server_url
                })
                error_msg = self.error_handler.handle_error("MCP_TOOL_INVOCATION_FAILED", {"error": mcp_result.get("error")})
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg
                }

        # Format and return response
        async with trace_step(
            "output_formatting",
            step_type="format",
            decision_summary="Format and sanitize LLM/MCP tool output",
            output_fn=lambda r: f"output_length={len(r.get('result','')) if r.get('result') else 0}",
        ) as step:
            # Compose output as per OUTPUT_FORMAT
            summary = f"Action: Invoked MCP tool '{mcp_tool_id}' on server '{mcp_server_url}'."
            tool_output = sanitize_llm_output(str(mcp_result.get("result", "")), content_type="text")
            formatted = f"{summary}\n\nMCP Tool Response:\n{tool_output}"
            step.capture({"result": formatted})
            self.audit_logger.log_event("mcp_tool_invocation_success", {
                "user_query": user_query,
                "tool_id": mcp_tool_id,
                "server_url": mcp_server_url,
                "output": tool_output
            })
            return {
                "success": True,
                "result": formatted,
                "error": None
            }

# =========================
# FASTAPI APP & ENDPOINTS
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="MCP Tool Invocation Assistant",
    description="Professional assistant for orchestrating MCP tool invocations via LLM.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is well-formed (check for missing quotes, commas, or brackets).",
                "Field names and types must match the API schema.",
                "If sending large text, keep it under 50,000 characters."
            ]
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is well-formed (check for missing quotes, commas, or brackets).",
                "Field names and types must match the API schema.",
                "If sending large text, keep it under 50,000 characters."
            ]
        }
    )

@app.post("/invoke", response_model=MCPToolInvocationResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def invoke_tool(req: MCPToolInvocationRequest):
    """
    Endpoint to invoke an MCP tool via LLM orchestration.
    """
    agent = MCPToolInvocationAgent()
    try:
        async with trace_step(
            "process_query",
            step_type="process",
            decision_summary="Main agent orchestration",
            output_fn=lambda r: f"success={r.get('success')}",
        ) as step:
            result = await agent.process_query(
                user_query=req.user_query,
                mcp_tool_id=req.mcp_tool_id,
                mcp_server_url=req.mcp_server_url
            )
            step.capture(result)
            # Sanitize output
            if result.get("result"):
                result["result"] = sanitize_llm_output(result["result"], content_type="text")
            return MCPToolInvocationResponse(
                success=result.get("success", False),
                result=result.get("result"),
                error=result.get("error")
            )
    except Exception as e:
        logger.error(f"Unhandled error in /invoke: {e}", exc_info=True)
        return MCPToolInvocationResponse(
            success=False,
            result=None,
            error=FALLBACK_RESPONSE
        )

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())