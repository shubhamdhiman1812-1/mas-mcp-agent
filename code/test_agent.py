
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import agent
from agent import MCPToolInvocationAgent, MCPToolInvocationRequest, ToolValidator, AuditLogger, ErrorHandler, LLMService, MCPToolInvoker, FALLBACK_RESPONSE
from pydantic import ValidationError

@pytest.mark.asyncio
async def test_valid_mcp_tool_invocation_end_to_end():
    """Functional: Full workflow with valid tool, server, and LLM (happy path)."""
    agent_instance = MCPToolInvocationAgent()
    user_query = "Summarize logs"
    mcp_tool_id = "tool1"
    mcp_server_url = "https://server"
    # Patch dependencies for happy path
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": True, "result": "LLM output"})), \
         patch.object(agent_instance.mcp_tool_invoker, "invoke", new=AsyncMock(return_value={"success": True, "result": "MCP output"})):
        result = await agent_instance.process_query(user_query, mcp_tool_id, mcp_server_url)
        assert result["success"] is True
        assert "Action: Invoked MCP tool" in result["result"]
        assert result["error"] is None

@pytest.mark.asyncio
async def test_invalid_tool_id_returns_error():
    """Functional: Invalid tool_id returns error."""
    agent_instance = MCPToolInvocationAgent()
    user_query = "Summarize logs"
    mcp_tool_id = "invalid_tool"
    mcp_server_url = "https://server"
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=False):
        result = await agent_instance.process_query(user_query, mcp_tool_id, mcp_server_url)
        assert result["success"] is False
        assert result["result"] is None
        assert "could not be found or is not registered" in result["error"]

@pytest.mark.asyncio
async def test_server_unavailable_returns_error():
    """Functional: Server unreachable returns error."""
    agent_instance = MCPToolInvocationAgent()
    user_query = "Summarize logs"
    mcp_tool_id = "tool1"
    mcp_server_url = "https://unreachable-server"
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=False):
        result = await agent_instance.process_query(user_query, mcp_tool_id, mcp_server_url)
        assert result["success"] is False
        assert result["result"] is None
        assert "server is currently unavailable or unreachable" in result["error"]

@pytest.mark.asyncio
async def test_llm_tool_call_failure_fallback():
    """Functional: LLMService.invoke_tool_call returns error, fallback error returned."""
    agent_instance = MCPToolInvocationAgent()
    user_query = "Summarize logs"
    mcp_tool_id = "tool1"
    mcp_server_url = "https://server"
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": False, "error": "LLM error"})):
        result = await agent_instance.process_query(user_query, mcp_tool_id, mcp_server_url)
        assert result["success"] is False
        assert result["result"] is None
        assert result["error"] is not None

@pytest.mark.asyncio
async def test_mcp_tool_invocation_failure_fallback():
    """Functional: MCPToolInvoker.invoke returns error, fallback error returned."""
    agent_instance = MCPToolInvocationAgent()
    user_query = "Summarize logs"
    mcp_tool_id = "tool1"
    mcp_server_url = "https://server"
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": True, "result": "LLM output"})), \
         patch.object(agent_instance.mcp_tool_invoker, "invoke", new=AsyncMock(return_value={"success": False, "error": "MCP error"})):
        result = await agent_instance.process_query(user_query, mcp_tool_id, mcp_server_url)
        assert result["success"] is False
        assert result["result"] is None
        assert result["error"] is not None

def test_input_validation_model_empty_query():
    """Unit: MCPToolInvocationRequest validation for empty user_query."""
    with pytest.raises(ValidationError) as excinfo:
        MCPToolInvocationRequest(user_query="", mcp_tool_id="tool1", mcp_server_url="https://server")
    assert "user_query must not be empty" in str(excinfo.value)

def test_input_validation_model_invalid_server_url():
    """Unit: MCPToolInvocationRequest validation for invalid mcp_server_url."""
    with pytest.raises(ValidationError) as excinfo:
        MCPToolInvocationRequest(user_query="query", mcp_tool_id="tool1", mcp_server_url="ftp://server")
    assert "mcp_server_url must start with http:// or https://" in str(excinfo.value)

def test_toolvalidator_validates_registered_tool():
    """Unit: ToolValidator.validate_tool returns True for registered tool."""
    validator = ToolValidator(registered_tools={"tool1": {}})
    assert validator.validate_tool("tool1") is True

def test_toolvalidator_rejects_invalid_tool():
    """Unit: ToolValidator.validate_tool returns False for 'invalid_tool'."""
    validator = ToolValidator()
    assert validator.validate_tool("invalid_tool") is False

def test_auditlogger_handles_logging_error_gracefully(caplog):
    """Unit: AuditLogger.log_event handles logging errors gracefully."""
    logger = AuditLogger()
    # object() is not JSON serializable, should trigger except block
    logger.log_event("test_event", {"key": object()})
    # No exception should be raised, warning should be logged
    assert any("Audit log failed" in record.message for record in caplog.records)

def test_errorhandler_returns_fallback_for_unknown_error_code():
    """Unit: ErrorHandler.handle_error returns fallback for unknown error code."""
    handler = ErrorHandler()
    result = handler.handle_error("SOME_UNKNOWN_ERROR", {})
    assert result == FALLBACK_RESPONSE

def test_llmservice_get_client_raises_without_api_key(monkeypatch):
    """Unit: LLMService.get_client raises ValueError if AZURE_OPENAI_API_KEY missing."""
    monkeypatch.setattr(agent.Config, "AZURE_OPENAI_API_KEY", None)
    llm = LLMService()
    with pytest.raises(ValueError) as excinfo:
        llm.get_client()
    assert "AZURE_OPENAI_API_KEY not configured" in str(excinfo.value)

@pytest.mark.asyncio
async def test_mcptoolinvoker_invoke_retries_on_failure():
    """Integration: MCPToolInvoker.invoke retries up to 3 times on failure."""
    llm_service = LLMService()
    invoker = MCPToolInvoker(llm_service)
    with patch("requests.post", side_effect=Exception("fail")) as mock_post:
        result = await invoker.invoke("tool1", "https://server", "query")
        assert mock_post.call_count == 3
        assert result["success"] is False
        assert "fail" in result["error"]

@pytest.mark.asyncio
async def test_process_query_integration_full_success_path():
    """Integration: process_query returns success when all dependencies succeed."""
    agent_instance = MCPToolInvocationAgent()
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": True, "result": "LLM output"})), \
         patch.object(agent_instance.mcp_tool_invoker, "invoke", new=AsyncMock(return_value={"success": True, "result": "MCP output"})):
        result = await agent_instance.process_query("query", "tool1", "https://server")
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "Action: Invoked MCP tool" in result["result"]
        assert result["error"] is None

@pytest.mark.asyncio
async def test_process_query_integration_tool_validation_fails():
    """Integration: process_query returns error if tool validation fails."""
    agent_instance = MCPToolInvocationAgent()
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=False):
        result = await agent_instance.process_query("query", "invalid_tool", "https://server")
        assert result["success"] is False
        assert "could not be found or is not registered" in result["error"]

@pytest.mark.asyncio
async def test_process_query_integration_server_validation_fails():
    """Integration: process_query returns error if server validation fails."""
    agent_instance = MCPToolInvocationAgent()
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=False):
        result = await agent_instance.process_query("query", "tool1", "https://server")
        assert result["success"] is False
        assert "server is currently unavailable or unreachable" in result["error"]

@pytest.mark.asyncio
async def test_process_query_integration_llmservice_failure():
    """Integration: process_query returns error if LLMService.invoke_tool_call returns error."""
    agent_instance = MCPToolInvocationAgent()
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": False, "error": "LLM error"})):
        result = await agent_instance.process_query("query", "tool1", "https://server")
        assert result["success"] is False
        assert result["error"] is not None

@pytest.mark.asyncio
async def test_process_query_integration_mcptoolinvoker_failure():
    """Integration: process_query returns error if MCPToolInvoker.invoke returns error."""
    agent_instance = MCPToolInvocationAgent()
    with patch.object(agent_instance.tool_validator, "validate_tool", return_value=True), \
         patch.object(agent_instance.tool_validator, "validate_server", return_value=True), \
         patch.object(agent_instance.llm_service, "invoke_tool_call", new=AsyncMock(return_value={"success": True, "result": "LLM output"})), \
         patch.object(agent_instance.mcp_tool_invoker, "invoke", new=AsyncMock(return_value={"success": False, "error": "MCP error"})):
        result = await agent_instance.process_query("query", "tool1", "https://server")
        assert result["success"] is False
        assert result["error"] is not None