from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from ragforge.domain.enums import DocumentStatus, IndexingStatus, MimeType, SearchStrategy


class Document(BaseModel):
    """Represents an ingested document entity."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    source_type: MimeType
    content_hash: str
    raw_content: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChunkMetadata(BaseModel):
    """Granular positional and structural metadata for a chunk."""

    page_number: int | None = None
    section_header: str | None = None
    start_char_idx: int | None = None
    end_char_idx: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Represents a discrete text chunk extracted from a Document."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int
    content_hash: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None


class Query(BaseModel):
    """Represents an incoming retrieval query with execution parameters."""

    id: UUID = Field(default_factory=uuid4)
    raw_query: str
    transformed_queries: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None
    top_k: int = 10
    score_threshold: float | None = None
    search_strategy: SearchStrategy = SearchStrategy.HYBRID


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from a vector store or ranking service."""

    chunk: Chunk
    score: float
    retrieval_type: str = "dense"
    rank: int
    dense_score: float | None = None
    lexical_score: float | None = None


class Citation(BaseModel):
    """Direct attribution mapping from an answer back to a source chunk."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    page_number: int | None = None
    matched_text_snippet: str


class DocumentIndexRecord(BaseModel):
    """Record tracking the indexed state of a document for incremental indexing."""

    file_path: str
    document_id: UUID
    content_hash: str
    chunk_count: int = 0
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIndexingResult(BaseModel):
    """Outcome for an individual document during an indexing run."""

    file_path: str
    status: IndexingStatus
    document_id: UUID | None = None
    chunk_count: int = 0
    content_hash: str | None = None
    error: str | None = None


class IndexingResult(BaseModel):
    """Aggregate statistics and document outcomes for an indexing execution."""

    discovered_documents: int = 0
    indexed_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    duration_seconds: float = 0.0
    documents: list[DocumentIndexingResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of documents evaluated during the run."""
        return (
            self.indexed_documents
            + self.updated_documents
            + self.skipped_documents
            + self.failed_documents
        )

    @property
    def is_success(self) -> bool:
        """True if no document-level failures occurred during indexing."""
        return self.failed_documents == 0


class LLMUsage(BaseModel):
    """Token consumption metadata for an LLM generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerationRequest(BaseModel):
    """Provider-agnostic request parameters for text generation."""

    prompt: str
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        effective_prompt = prompt if prompt is not None else user_prompt
        if effective_prompt is None:
            effective_prompt = kwargs.pop("prompt", None) or kwargs.pop("user_prompt", "")
        effective_system = system_prompt if system_prompt is not None else system_instruction
        if effective_system is None:
            effective_system = kwargs.pop("system_prompt", None) or kwargs.pop(
                "system_instruction", None
            )
        super().__init__(
            prompt=effective_prompt,
            system_prompt=effective_system,
            **kwargs,
        )

    @property
    def user_prompt(self) -> str:
        """Alias for prompt."""
        return self.prompt

    @property
    def system_instruction(self) -> str | None:
        """Alias for system_prompt."""
        return self.system_prompt


class GenerationResponse(BaseModel):
    """Provider-agnostic response payload containing generated text and execution provenance."""

    text: str
    provider: str
    model: str
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content(self) -> str:
        """Convenience alias for generated text."""
        return self.text


class SourceReference(BaseModel):
    """Granular provenance attribution linking a generated answer back to a source chunk."""

    document_id: UUID | str
    chunk_id: UUID | str
    source_path: str | None = None
    document_title: str | None = None
    section_header: str | None = None
    page_number: int | None = None
    start_char_idx: int | None = None
    end_char_idx: int | None = None
    score: float = 0.0
    content_snippet: str = ""

    def __init__(
        self,
        start_char: int | None = None,
        end_char: int | None = None,
        **kwargs: Any,
    ) -> None:
        if start_char is not None and "start_char_idx" not in kwargs:
            kwargs["start_char_idx"] = start_char
        if end_char is not None and "end_char_idx" not in kwargs:
            kwargs["end_char_idx"] = end_char
        super().__init__(**kwargs)

    @property
    def start_char(self) -> int | None:
        """Alias for start_char_idx."""
        return self.start_char_idx

    @property
    def end_char(self) -> int | None:
        """Alias for end_char_idx."""
        return self.end_char_idx


class RAGResponse(BaseModel):
    """End-to-end grounded response produced by RAGGenerationService."""

    question: str = ""
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    provider: str
    model: str
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryRequest(BaseModel):
    """User query payload for RAG retrieval and answer synthesis."""

    question: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    score_threshold: float | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class EvaluationCase(BaseModel):
    """A single evaluation query and its ground-truth relevance criteria."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_case_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize sources/documents (supports expected_source, expected_document)
            raw_sources = data.get("expected_sources") or data.get("expected_documents") or []
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            elif isinstance(raw_sources, (list, tuple)):
                raw_sources = list(raw_sources)
            else:
                raw_sources = []

            single_source = data.get("expected_source") or data.get("expected_document")
            if single_source and single_source not in raw_sources:
                raw_sources.append(single_source)
            data["expected_sources"] = raw_sources

            # Normalize chunks (supports expected_chunk, expected_chunks)
            raw_chunks = data.get("expected_chunks") or []
            if isinstance(raw_chunks, str):
                raw_chunks = [raw_chunks]
            elif isinstance(raw_chunks, (list, tuple)):
                raw_chunks = list(raw_chunks)
            else:
                raw_chunks = []

            single_chunk = data.get("expected_chunk")
            if single_chunk and single_chunk not in raw_chunks:
                raw_chunks.append(single_chunk)
            data["expected_chunks"] = raw_chunks

            # Normalize facts (supports expected_fact, expected_facts)
            raw_facts = data.get("expected_facts") or []
            if isinstance(raw_facts, str):
                raw_facts = [raw_facts]
            elif isinstance(raw_facts, (list, tuple)):
                raw_facts = list(raw_facts)
            else:
                raw_facts = []

            single_fact = data.get("expected_fact")
            if single_fact and single_fact not in raw_facts:
                raw_facts.append(single_fact)
            data["expected_facts"] = raw_facts

            # Normalize expected_answer (supports expected_answers)
            if not data.get("expected_answer") and data.get("expected_answers"):
                ans = data.get("expected_answers")
                data["expected_answer"] = ans[0] if isinstance(ans, list) and ans else str(ans)
        return data


class EvaluationDataset(BaseModel):
    """Collection of evaluation cases for benchmark execution."""

    name: str = "default"
    description: str | None = None
    cases: list[EvaluationCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> EvaluationCase:
        return self.cases[index]


class CaseEvaluationResult(BaseModel):
    """Evaluation metrics and details for a single evaluation case."""

    case_id: str
    question: str
    retrieved_count: int
    relevant_retrieved_count: int
    total_expected_relevant: int
    first_relevant_rank: int | None = None
    reciprocal_rank: float = 0.0
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)


class RetrievalEvaluationReport(BaseModel):
    """Aggregated retrieval evaluation report across a benchmark dataset."""

    dataset_name: str = "default"
    total_cases: int = 0
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    mean_recall_at_k: dict[int, float] = Field(default_factory=dict)
    mean_precision_at_k: dict[int, float] = Field(default_factory=dict)
    mean_reciprocal_rank: float = 0.0
    duration_seconds: float = 0.0
    case_results: list[CaseEvaluationResult] = Field(default_factory=list)

    @property
    def mrr(self) -> float:
        """Convenience alias for mean_reciprocal_rank."""
        return self.mean_reciprocal_rank

    def recall_at(self, k: int) -> float:
        """Return Mean Recall@K for the specified K."""
        return self.mean_recall_at_k.get(k, 0.0)

    def precision_at(self, k: int) -> float:
        """Return Mean Precision@K for the specified K."""
        return self.mean_precision_at_k.get(k, 0.0)

    def format_text_report(self) -> str:
        """Format a human-readable evaluation summary matching the CLI specification."""
        lines = [
            "Evaluation Report",
            "-----------------",
            f"Cases: {self.total_cases}",
        ]
        for k in self.k_values:
            lines.append(f"Recall@{k}: {self.recall_at(k):.4f}")
        for k in self.k_values:
            lines.append(f"Precision@{k}: {self.precision_at(k):.4f}")
        lines.append(f"MRR: {self.mean_reciprocal_rank:.4f}")
        return "\n".join(lines)
