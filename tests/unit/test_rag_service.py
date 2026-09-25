"""Unit tests for RAG generation service, context formatting, and prompt builder."""

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from ragforge.domain.models import (
    Chunk,
    ChunkMetadata,
    GenerationRequest,
    GenerationResponse,
    LLMUsage,
    RetrievedChunk,
    SourceReference,
)
from ragforge.ports.llm import BaseLLMProvider
from ragforge.services.llm_router import LLMRouter
from ragforge.services.prompt import (
    PromptBuilder,
    format_rag_context,
)
from ragforge.services.rag import RAGGenerationService
from ragforge.services.retrieval import RetrievalService


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider returning predetermined generation responses."""

    def __init__(self, answer_text: str = "Test answer", provider: str = "groq") -> None:
        self.answer_text = answer_text
        self._provider = provider
        self.last_request: GenerationRequest | None = None

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def default_model(self) -> str:
        return "test-model-v1"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.last_request = request
        return GenerationResponse(
            text=self.answer_text,
            provider=self._provider,
            model="test-model-v1",
            usage=LLMUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )


def make_retrieved_chunk(
    doc_id: UUID | None = None,
    chunk_id: UUID | None = None,
    content: str = "Sample chunk content.",
    score: float = 0.88,
    section: str | None = "Overview",
    source_path: str = "/data/docs/doc1.txt",
    start_char: int = 0,
    end_char: int = 100,
    rank: int = 1,
) -> RetrievedChunk:
    """Helper creating a sample RetrievedChunk."""
    doc_uuid = doc_id or uuid4()
    chunk_uuid = chunk_id or uuid4()
    metadata = ChunkMetadata(
        page_number=1,
        section_header=section,
        start_char_idx=start_char,
        end_char_idx=end_char,
        extra={
            "source_path": source_path,
            "file_path": source_path,
            "document_title": "Sample Doc",
        },
    )
    chunk = Chunk(
        id=chunk_uuid,
        document_id=doc_uuid,
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        content_hash="mockhash",
        metadata=metadata,
    )
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        retrieval_type="dense",
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Prompt Builder & Context Construction Tests (Requirements L, N)
# ---------------------------------------------------------------------------


def test_rag_context_construction_with_chunks() -> None:
    """Verify deterministic formatting of multiple retrieved chunks."""
    c1 = make_retrieved_chunk(
        content="RAG combines search with generation.",
        score=0.92,
        section="Architecture",
        source_path="docs/rag.md",
        rank=1,
    )
    c2 = make_retrieved_chunk(
        content="Embeddings capture semantic meaning.",
        score=0.85,
        section=None,
        source_path="docs/embeddings.md",
        rank=2,
    )

    formatted = format_rag_context([c1, c2])

    assert "[Document 1]" in formatted
    assert "Source: docs/rag.md" in formatted
    assert "Section: Architecture" in formatted
    assert "RAG combines search with generation." in formatted
    assert "[Document 2]" in formatted
    assert "Source: docs/embeddings.md" in formatted
    assert "Section: General" in formatted
    assert "Embeddings capture semantic meaning." in formatted


def test_rag_context_construction_empty() -> None:
    """Verify context formatting when zero chunks are retrieved."""
    formatted = format_rag_context([])
    assert formatted == "No relevant context documents were retrieved."


def test_grounded_prompt_builder() -> None:
    """Verify PromptBuilder includes critical grounding and safety constraints."""
    builder = PromptBuilder()
    sys_prompt = builder.build_system_prompt()

    # Grounding & safety rules
    assert "factual" in sys_prompt.lower()
    assert "do not" in sys_prompt.lower()
    assert "insufficient" in sys_prompt.lower()
    assert "evidence" in sys_prompt.lower()
    assert "instructions" in sys_prompt.lower()

    # User prompt formatting
    c1 = make_retrieved_chunk(
        content="Qdrant is a vector database.",
        score=0.95,
        source_path="docs/qdrant.md",
    )
    user_prompt = builder.build_user_prompt(
        question="What is Qdrant?",
        chunks=[c1],
    )
    assert "Provided Context:" in user_prompt
    assert "Qdrant is a vector database." in user_prompt
    assert "Question: What is Qdrant?" in user_prompt


# ---------------------------------------------------------------------------
# RAG Generation Service Tests (Requirements M, O, P)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_service_end_to_end_success() -> None:
    """Verify end-to-end RAG workflow returns structured RAGResponse with mapped sources."""
    doc_id = uuid4()
    chunk_id = uuid4()
    chunk1 = make_retrieved_chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        content="Antigravity accelerates AI pair programming.",
        score=0.95,
        section="Intro",
        source_path="specs/agent.md",
        start_char=10,
        end_char=54,
    )

    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = [chunk1]

    llm_provider = MockLLMProvider(
        answer_text="Antigravity accelerates programming.",
        provider="groq",
    )
    router = LLMRouter(primary_provider=llm_provider)

    rag_service = RAGGenerationService(
        retrieval_service=mock_retrieval,
        llm_router=router,
    )

    result = await rag_service.generate_answer(
        question="What does Antigravity do?",
        top_k=3,
        score_threshold=0.5,
    )

    # Retrieval assertions
    mock_retrieval.retrieve.assert_awaited_once_with(
        query="What does Antigravity do?",
        top_k=3,
        filters=None,
        score_threshold=0.5,
    )

    # RAGResponse assertions (Requirement P)
    assert result.question == "What does Antigravity do?"
    assert result.answer == "Antigravity accelerates programming."
    assert result.provider == "groq"
    assert result.model == "test-model-v1"
    assert result.usage is not None
    assert result.usage.total_tokens == 75

    # Source metadata preservation (Requirement M)
    assert len(result.sources) == 1
    src = result.sources[0]
    assert isinstance(src, SourceReference)
    assert src.document_id == doc_id
    assert src.chunk_id == chunk_id
    assert src.source_path == "specs/agent.md"
    assert src.section_header == "Intro"
    assert src.start_char == 10
    assert src.end_char == 54
    assert src.score == 0.95
    assert src.content_snippet == "Antigravity accelerates AI pair programming."


@pytest.mark.asyncio
async def test_rag_service_empty_retrieval_handling() -> None:
    """Verify handling when retrieval yields no matches (Requirement O)."""
    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    llm_provider = MockLLMProvider(
        answer_text="I do not have enough information to answer this question.",
        provider="groq",
    )
    router = LLMRouter(primary_provider=llm_provider)

    rag_service = RAGGenerationService(
        retrieval_service=mock_retrieval,
        llm_router=router,
    )

    result = await rag_service.generate_answer(question="Tell me about Martian geology.")

    assert result.answer == "I do not have enough information to answer this question."
    assert result.sources == []
    assert llm_provider.last_request is not None
    assert "No relevant context documents were retrieved." in llm_provider.last_request.prompt


@pytest.mark.asyncio
async def test_rag_service_rejects_empty_question() -> None:
    """Verify RAG service validates question input."""
    mock_retrieval = AsyncMock(spec=RetrievalService)
    router = LLMRouter(primary_provider=MockLLMProvider())
    rag_service = RAGGenerationService(retrieval_service=mock_retrieval, llm_router=router)

    with pytest.raises(ValueError) as exc_info:
        await rag_service.generate_answer(question="   ")

    assert "empty" in str(exc_info.value).lower()
