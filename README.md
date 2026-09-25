# RAGForge

RAGForge is a production-oriented Agentic Retrieval-Augmented Generation (RAG) platform designed to bridge the gap between simple proof-of-concept chatbots and resilient enterprise information retrieval systems. The platform delivers end-to-end multi-format document ingestion, deterministic chunking, vector indexing in Qdrant, hybrid retrieval, provider-agnostic LLM synthesis with automatic failover, and application-managed citation tracking.

## Architecture & RAG Query Flow

```
User Question
      ↓
RetrievalService (Query Embedding + Similarity Search)
      ↓
Qdrant Vector Database
      ↓
Retrieved Chunks & Metadata Provenance
      ↓
Deterministic Context & Prompt Construction (PromptBuilder)
      ↓
RAG Generation Service
      ↓
LLM Provider Router (LLMRouter)
      ↓
┌───────────────────────────────────────┐
│ Groq Primary (llama-3.3-70b-versatile)│
└───────────────────┬───────────────────┘
                    │ Fallback-eligible failure (timeout, 5xx, rate-limit)
                    ↓
┌───────────────────────────────────────┐
│ OpenAI Backup (gpt-4o-mini)           │
└───────────────────┬───────────────────┘
                    ↓
Grounded Answer + Application-Managed Source References
```

## Key Capabilities (Stage 8)

1. **Provider-Agnostic LLM Port (`BaseLLMProvider`)**:
   - Complete abstraction of LLM generation behind a clean async port interface (`src/ragforge/ports/llm.py`).
   - Domain models: `GenerationRequest`, `GenerationResponse`, and `LLMUsage`.
   - Zero vendor-specific SDK imports leak into domain, ports, or generation service layers.

2. **Groq Primary Adapter (`GroqLLMProvider`)**:
   - Ultra-fast inference powered by the official async Groq SDK (`AsyncGroq`).
   - Default flagship model: `llama-3.3-70b-versatile`.
   - Full support for configurable timeout, sampling temperature, max output tokens, and granular exception mapping.

3. **OpenAI Fallback Adapter (`OpenAILLMProvider`)**:
   - Enterprise-grade backup provider using official async OpenAI SDK (`AsyncOpenAI`).
   - Default model: `gpt-4o-mini`.
   - Configurable generation parameters and complete isolation within the adapter layer.

4. **Resilient Provider Router & Controlled Fallback (`LLMRouter`)**:
   - Automatically attempts generation with the primary provider (Groq by default).
   - If a transient or capacity error occurs (e.g. `LLMRateLimitError`, `LLMTimeoutError`, `LLMConnectionError`, or HTTP 503 `LLMProviderUnavailableError`), the router seamlessly invokes the configured fallback provider (OpenAI).
   - **Fast-Fail on Non-Transient Errors**: Programming errors, configuration errors (`LLMConfigurationError`), and authentication failures (`LLMAuthenticationError`) never trigger fallback and fail immediately with sanitized logs.
   - Comprehensive error reporting via `AllLLMProvidersFailedError` when both providers fail.

5. **Grounded Prompt Construction (`PromptBuilder`)**:
   - Strict system instructions instructing the LLM to rely solely on provided context.
   - Prevents prompt injection by treating retrieved document text strictly as evidence, never directives.
   - Explicit instructions to acknowledge uncertainty if context is insufficient.

6. **Application-Managed Source / Citation Mapping**:
   - **The LLM never generates citation IDs.**
   - Provenance is maintained deterministically by RAGForge:
     - Document UUID
     - Chunk UUID
     - Source file path & section header
     - Character offsets (`start_char_idx`, `end_char_idx`)
     - Relevance similarity score
     - Verifiable content snippet

7. **FastAPI Query Endpoint (`POST /api/v1/query`)**:
   - Production HTTP API endpoint accepting natural language questions and returning grounded answers with structured source references.

---

## Configuration & Environment Variables

Configure RAGForge via `.env` or system environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RAGFORGE_LLM_PROVIDER` | `groq` | Primary LLM provider (`groq` or `openai`) |
| `RAGFORGE_LLM_MODEL` | `llama-3.3-70b-versatile` | Model identifier for primary provider |
| `RAGFORGE_LLM_FALLBACK_PROVIDER` | `openai` | Optional fallback provider (`openai` or `groq`) |
| `RAGFORGE_LLM_FALLBACK_MODEL` | `gpt-4o-mini` | Model identifier for fallback provider |
| `GROQ_API_KEY` | None | API key for Groq Cloud (or `RAGFORGE_GROQ_API_KEY`) |
| `OPENAI_API_KEY` | None | API key for OpenAI (or `RAGFORGE_OPENAI_API_KEY`) |
| `RAGFORGE_LLM_TEMPERATURE` | `0.0` | Sampling temperature for generation (0.0 = deterministic) |
| `RAGFORGE_LLM_MAX_TOKENS` | `1024` | Maximum output tokens generated per answer |
| `RAGFORGE_LLM_TIMEOUT` | `30.0` | Request timeout in seconds for upstream LLM calls |
| `RAGFORGE_QDRANT_URL` | `http://localhost:6333` | Endpoint for Qdrant vector database |
| `RAGFORGE_QDRANT_COLLECTION` | `ragforge_chunks` | Collection name for chunk vector index |

---

## Local Setup

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- Docker & Docker Compose (for local Qdrant)

### Installation

1. Clone repository and install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your GROQ_API_KEY and optional OPENAI_API_KEY
   ```

3. Start Qdrant vector database:
   ```bash
   docker compose up -d
   ```

4. Index documents into Qdrant:
   ```bash
   uv run python -m ragforge.cli index path/to/docs/
   ```

5. Launch the FastAPI server:
   ```bash
   uv run uvicorn ragforge.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
   ```

---

## API Usage Example

### POST /api/v1/query

Execute a grounded question-answering query against indexed documentation:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does RAGForge handle fallback between LLM providers?",
    "top_k": 3
  }'
```

#### Example Response:

```json
{
  "answer": "RAGForge uses an LLMRouter that dispatches requests to a primary provider (Groq) and automatically fails over to a backup provider (OpenAI) when transient errors, rate limits, or timeouts occur. Non-transient errors fail immediately.",
  "sources": [
    {
      "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "chunk_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "source_path": "docs/architecture.md",
      "document_title": "Architecture Overview",
      "section_header": "LLM Provider Router",
      "page_number": null,
      "start_char_idx": 1420,
      "end_char_idx": 1850,
      "score": 0.8954,
      "content_snippet": "The LLMRouter routes requests to the primary provider and seamlessly invokes fallback..."
    }
  ],
  "provider": "groq",
  "model": "llama-3.3-70b-versatile"
}
```

---

## Testing & Validation

The test suite runs 100% offline without requiring external network connectivity or real API keys.

```bash
# Run complete test suite (unit tests with mocked SDKs)
uv run pytest -v

# Run lint checks
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Run static type checking (strict mode)
uv run mypy
```

---

## Security Guidance

- **No Hardcoded Secrets**: Real API keys are never embedded in code, committed to Git, or included in default settings.
- **Log & Error Sanitization**: LLM adapters sanitize all error messages and representations. API keys are masked before logging or surfacing in exception details.
- **Evidence Isolation**: The grounded prompt builder enforces that retrieved document contents are treated strictly as reference data, preventing prompt injection attacks from overriding system guidelines.
