# MCP Tool Invocation Assistant

A professional Python agent that orchestrates MCP tool invocations via LLM (Azure OpenAI GPT-4.1). It validates user queries, checks tool/server availability, invokes the MCP tool through an LLM, and returns a clear, concise response. Includes robust error handling, observability, and runtime guardrails.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

- **Direct execution:**
  ```
  python code/agent.py
  ```

- **As a FastAPI server:**
  ```
  uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
  ```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME` — Agent name (pre-configured)
- `AGENT_ID` — Agent unique identifier (pre-configured)
- `PROJECT_NAME` — Project name (pre-configured)
- `PROJECT_ID` — Project unique identifier (pre-configured)

**General**
- `ENVIRONMENT` — Deployment environment (e.g., development, production)

**Azure Key Vault**
- `USE_KEY_VAULT` — Enable Azure Key Vault integration (`true`/`false`)
- `KEY_VAULT_URI` — Azure Key Vault URI
- `AZURE_USE_DEFAULT_CREDENTIAL` — Use DefaultAzureCredential (`true`/`false`)
- `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` — Azure AD credentials (if not using managed identity)

**LLM Configuration**
- `MODEL_PROVIDER` — LLM provider (`openai`, `azure`, `anthropic`, `google`)
- `LLM_MODEL` — LLM model name (e.g., `gpt-4.1`)
- `LLM_TEMPERATURE` — Model temperature (float)
- `LLM_MAX_TOKENS` — Maximum tokens in model response

**API Keys / Secrets**
- `OPENAI_API_KEY` — OpenAI API key
- `AZURE_OPENAI_API_KEY` — Azure OpenAI API key
- `ANTHROPIC_API_KEY` — Anthropic API key
- `GOOGLE_API_KEY` — Google API key
- `AZURE_CONTENT_SAFETY_KEY` — Azure Content Safety API key
- `OBS_AZURE_SQL_PASSWORD` — Azure SQL password for observability

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI endpoint URL
- `AZURE_CONTENT_SAFETY_ENDPOINT` — Azure Content Safety endpoint URL

**Observability (Azure SQL)**
- `SERVICE_NAME` — Service name for observability/logging
- `SERVICE_VERSION` — Service version
- `OBS_DATABASE_TYPE` — Observability database type (e.g., `azure_sql`)
- `OBS_AZURE_SQL_SERVER` — Azure SQL server
- `OBS_AZURE_SQL_DATABASE` — Azure SQL database name
- `OBS_AZURE_SQL_PORT` — Azure SQL port
- `OBS_AZURE_SQL_USERNAME` — Azure SQL username
- `OBS_AZURE_SQL_SCHEMA` — Azure SQL schema
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE` — Trust SQL Server certificate (`yes`)

**Agent-Specific**
- `VALIDATION_CONFIG_PATH` — Path to validation config file (optional)

See `.env.example` for all variables and descriptions.

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/invoke`
- **Description:** Invoke an MCP tool via LLM orchestration.
- **Request body:**
  ```
  {
    "user_query": "string (required)",
    "mcp_tool_id": "string (required)",
    "mcp_server_url": "string (required)"
  }
  ```
- **Response:**
  ```
  {
    "success": true|false,
    "result": "string|null",
    "error": "string|null"
  }
  ```

### **422 Validation Error Responses**
- **Response:**
  ```
  {
    "success": false,
    "error": "Malformed JSON or invalid request parameters.",
    "details": [...],
    "tips": [
      "Ensure your JSON is well-formed (check for missing quotes, commas, or brackets).",
      "Field names and types must match the API schema.",
      "If sending large text, keep it under 50,000 characters."
    ]
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t MCP-Tool-Invocation-Assistant -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name MCP-Tool-Invocation-Assistant MCP-Tool-Invocation-Assistant
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs MCP-Tool-Invocation-Assistant
```

### 7. Stop the container:
```
docker stop MCP-Tool-Invocation-Assistant
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**MCP Tool Invocation Assistant** — LLM-powered, compliant, and observable MCP tool orchestration for enterprise automation.
