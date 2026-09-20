# RAGForge

RAGForge is a production-oriented Agentic Retrieval-Augmented Generation (RAG) platform designed to bridge the gap between simple proof-of-concept chatbots and resilient enterprise information retrieval systems. The platform delivers end-to-end multi-format document ingestion, configurable chunking, hybrid semantic and keyword retrieval, cross-encoder reranking, adaptive LangGraph-driven routing, grounded synthesis with citation attribution, and integrated evaluation and observability.

## Development Status

- **Current Stage**: Day 1 - Project Foundation
- **Active Branch**: `feat/day1-foundation`
- **Implemented**: Clean architecture package layout, core domain models, abstract port definitions, typed Pydantic configuration, structured application logging, FastAPI application factory with health check endpoint, and unit test suite.

## Local Setup

### Prerequisites

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`

### Installation

1. Clone the repository and navigate into the workspace:
   ```bash
   git clone <repo-url>
   cd ragforge
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install runtime and development dependencies in editable mode:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. Create local environment configuration:
   ```bash
   cp .env.example .env
   ```

## Running the Tests

Execute the automated test suite with `pytest`:

```bash
pytest
```

Run code formatting and linting:

```bash
ruff check .
ruff format --check .
```

Run static type checking:

```bash
mypy
```

## Running the FastAPI Development Server

Launch the development server with hot-reloading:

```bash
uvicorn ragforge.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Once running, access:
- Health check: `http://localhost:8000/health`
- Interactive API documentation: `http://localhost:8000/docs`
