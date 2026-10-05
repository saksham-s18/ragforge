"""Unit tests for EvaluationRunner and end-to-end retrieval evaluation workflows."""

from uuid import UUID

import pytest

from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.adapters.rerankers import DeterministicReranker
from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.domain.exceptions import DatasetValidationError
from ragforge.domain.models import (
    Chunk,
    ChunkMetadata,
    EvaluationCase,
    EvaluationDataset,
    RetrievalEvaluationReport,
)
from ragforge.evaluation import EvaluationRunner
from ragforge.services.retrieval import RetrievalService


def _create_mock_chunks() -> list[Chunk]:
    """Helper to generate indexed domain chunks with realistic metadata."""
    doc_id_1 = UUID("11111111-1111-1111-1111-111111111111")
    doc_id_2 = UUID("22222222-2222-2222-2222-222222222222")

    return [
        Chunk(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            document_id=doc_id_1,
            chunk_index=0,
            content="FastAPI is an asynchronous Python web framework for building APIs.",
            token_count=10,
            content_hash="hash_fastapi",
            metadata=ChunkMetadata(
                section_header="Frameworks",
                extra={
                    "file_path": "/docs/fastapi.md",
                    "document_title": "FastAPI Guide",
                },
            ),
        ),
        Chunk(
            id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            document_id=doc_id_1,
            chunk_index=1,
            content="Qdrant is an open-source vector similarity search engine.",
            token_count=8,
            content_hash="hash_qdrant",
            metadata=ChunkMetadata(
                section_header="Vector DB",
                extra={
                    "file_path": "/docs/fastapi.md",
                    "document_title": "FastAPI Guide",
                },
            ),
        ),
        Chunk(
            id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            document_id=doc_id_2,
            chunk_index=0,
            content="Astrophysics studies stars, exoplanets, and gravitational waves.",
            token_count=9,
            content_hash="hash_astro",
            metadata=ChunkMetadata(
                section_header="Cosmology",
                extra={
                    "file_path": "/science/astro.txt",
                    "document_title": "Astrophysics Primer",
                },
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_runner_end_to_end_in_memory_retrieval() -> None:
    """Verify EvaluationRunner executes retrieval against RetrievalService and scores metrics."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    runner = EvaluationRunner(retrieval_service, k_values=[1, 2, 3])

    case1 = EvaluationCase(
        question="asynchronous Python web framework FastAPI",
        expected_sources=["fastapi.md"],
        expected_chunks=["FastAPI is an asynchronous Python web framework"],
    )
    case2 = EvaluationCase(
        question="open-source vector search engine Qdrant",
        expected_sources=["fastapi.md"],
        expected_chunks=["Qdrant is an open-source vector similarity search engine"],
    )

    dataset = EvaluationDataset(name="e2e_test", cases=[case1, case2])
    report = await runner.evaluate_dataset(dataset)

    assert report.total_cases == 2
    assert report.recall_at(1) == pytest.approx(1.0)
    assert report.recall_at(3) == pytest.approx(1.0)
    assert report.precision_at(1) == pytest.approx(1.0)
    assert report.precision_at(3) == pytest.approx(1 / 3)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)
    assert report.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_runner_empty_results_handling() -> None:
    """Verify runner handles empty vector store / zero retrieved results gracefully."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    # Note: store is left empty!
    retrieval_service = RetrievalService(provider, store)

    runner = EvaluationRunner(retrieval_service, k_values=[1, 3, 5])
    case = EvaluationCase(
        question="Search query on empty store",
        expected_sources=["doc.md"],
        expected_chunks=["relevant text snippet"],
    )
    dataset = EvaluationDataset(name="empty_store_test", cases=[case])
    report = await runner.evaluate_dataset(dataset)

    assert report.total_cases == 1
    assert report.recall_at(1) == 0.0
    assert report.recall_at(5) == 0.0
    assert report.precision_at(1) == 0.0
    assert report.precision_at(5) == 0.0
    assert report.mean_reciprocal_rank == 0.0

    case_result = report.case_results[0]
    assert case_result.retrieved_count == 0
    assert case_result.relevant_retrieved_count == 0
    assert case_result.first_relevant_rank is None
    assert case_result.reciprocal_rank == 0.0


@pytest.mark.asyncio
async def test_runner_missing_expected_sources() -> None:
    """Verify evaluation when expected sources do not exist in the indexed database."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    runner = EvaluationRunner(retrieval_service, k_values=[1, 3])

    # Expected source does not exist
    case = EvaluationCase(
        question="FastAPI framework",
        expected_sources=["nonexistent_source_document.pdf"],
        expected_chunks=["FastAPI is an asynchronous Python web framework"],
    )
    dataset = EvaluationDataset(name="missing_source_test", cases=[case])
    report = await runner.evaluate_dataset(dataset)

    # Because expected source wasn't matched, relevance evaluates to False
    assert report.recall_at(1) == 0.0
    assert report.precision_at(1) == 0.0
    assert report.mean_reciprocal_rank == 0.0


@pytest.mark.asyncio
async def test_runner_k_larger_than_available_chunks() -> None:
    """Verify evaluation behavior when K cutoff is strictly larger than retrieved chunks."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    retrieval_service = RetrievalService(provider, store)

    # Index only 2 chunks total in the entire database
    chunks = _create_mock_chunks()[:2]
    await retrieval_service.index_chunks(chunks)

    # Request evaluation up to K=10, when only 2 chunks exist
    runner = EvaluationRunner(retrieval_service, k_values=[1, 5, 10])

    case = EvaluationCase(
        question="FastAPI web framework",
        expected_sources=["fastapi.md"],
        expected_chunks=["FastAPI is an asynchronous Python web framework"],
    )
    dataset = EvaluationDataset(name="large_k_test", cases=[case])
    report = await runner.evaluate_dataset(dataset)

    # 1 of 1 expected target was retrieved
    assert report.recall_at(1) == pytest.approx(1.0)
    assert report.recall_at(5) == pytest.approx(1.0)
    assert report.recall_at(10) == pytest.approx(1.0)

    # Precision denominators are strictly K (standard IR)
    assert report.precision_at(1) == pytest.approx(1.0)
    assert report.precision_at(5) == pytest.approx(1 / 5)
    assert report.precision_at(10) == pytest.approx(1 / 10)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_runner_chunk_matching_by_id_and_hash() -> None:
    """Verify chunk matching by UUID and content hash."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    runner = EvaluationRunner(retrieval_service, k_values=[1])

    # Case 1: Matching by UUID string
    case_by_uuid = EvaluationCase(
        question="FastAPI framework",
        expected_chunks=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
    )
    res_uuid = await runner.evaluate_case(case_by_uuid)
    assert res_uuid.recall_at_k[1] == pytest.approx(1.0)

    # Case 2: Matching by content hash
    case_by_hash = EvaluationCase(
        question="FastAPI framework",
        expected_chunks=["hash_fastapi"],
    )
    res_hash = await runner.evaluate_case(case_by_hash)
    assert res_hash.recall_at_k[1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_runner_source_level_only_relevance() -> None:
    """Verify case specifying only expected sources (no specific chunks)."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    runner = EvaluationRunner(retrieval_service, k_values=[1, 2])

    # Case with document title as expected source
    case_by_title = EvaluationCase(
        question="Cosmology stars and exoplanets",
        expected_sources=["Astrophysics Primer"],
    )
    res = await runner.evaluate_case(case_by_title)
    assert res.first_relevant_rank == 1
    assert res.recall_at_k[1] == pytest.approx(1.0)


def test_runner_invalid_k_values_raises() -> None:
    """Verify ValueError is raised for invalid k_values."""
    provider = DeterministicEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    retrieval_service = RetrievalService(provider, store)

    with pytest.raises(ValueError, match="k_values must be non-empty positive integers"):
        EvaluationRunner(retrieval_service, k_values=[])

    with pytest.raises(ValueError, match="k_values must be non-empty positive integers"):
        EvaluationRunner(retrieval_service, k_values=[0, 3])

    with pytest.raises(ValueError, match="k_values must be non-empty positive integers"):
        EvaluationRunner(retrieval_service, k_values=[-2, 5])


@pytest.mark.asyncio
async def test_runner_empty_dataset_raises() -> None:
    """Verify evaluate_dataset raises DatasetValidationError if dataset has no cases."""
    provider = DeterministicEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    retrieval_service = RetrievalService(provider, store)

    runner = EvaluationRunner(retrieval_service)
    empty_dataset = EvaluationDataset(name="empty", cases=[])

    with pytest.raises(DatasetValidationError, match="must contain at least one evaluation case"):
        await runner.evaluate_dataset(empty_dataset)


def test_retrieval_evaluation_report_formatting() -> None:
    """Verify report formatting method contains the specified CLI summary block."""
    report = RetrievalEvaluationReport(
        dataset_name="demo_report",
        total_cases=5,
        k_values=[1, 3, 5],
        mean_recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        mean_precision_at_k={1: 1.0, 3: 0.3333, 5: 0.2},
        mean_reciprocal_rank=1.0,
    )

    formatted = report.format_text_report()
    assert "Evaluation Report" in formatted
    assert "Cases: 5" in formatted
    assert "Recall@1: 1.0000" in formatted
    assert "Recall@3: 1.0000" in formatted
    assert "Recall@5: 1.0000" in formatted
    assert "Precision@5: 0.2000" in formatted
    assert "MRR: 1.0000" in formatted
    assert report.mrr == 1.0


# ---------------------------------------------------------------------------
# Evaluation Runner Reranker Integration (Stage 10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_with_deterministic_reranker() -> None:
    """Verify EvaluationRunner executes second-stage reranking when reranker is configured."""
    provider = DeterministicEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    reranker = DeterministicReranker()
    runner = EvaluationRunner(
        retrieval_service=retrieval_service,
        k_values=[1, 2],
        reranker=reranker,
        candidate_k=10,
    )

    case = EvaluationCase(
        question="Which open-source vector similarity search engine is used?",
        expected_sources=["/docs/fastapi.md"],
        expected_chunks=["Qdrant is an open-source vector similarity search engine."],
    )

    res = await runner.evaluate_case(case)
    # The chunk mentioning Qdrant should be rank 1 due to high lexical match with question
    assert res.first_relevant_rank == 1
    assert res.recall_at_k[1] == 1.0
    assert res.precision_at_k[1] == 1.0


@pytest.mark.asyncio
async def test_runner_evaluates_dataset_with_reranker() -> None:
    """Verify full dataset batch evaluation with reranker producing valid report."""
    provider = DeterministicEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    retrieval_service = RetrievalService(provider, store)

    chunks = _create_mock_chunks()
    await retrieval_service.index_chunks(chunks)

    runner = EvaluationRunner(
        retrieval_service=retrieval_service,
        k_values=[1, 3],
        reranker=DeterministicReranker(),
    )

    dataset = EvaluationDataset(
        name="rerank_test_suite",
        cases=[
            EvaluationCase(
                question="What is FastAPI?",
                expected_sources=["/docs/fastapi.md"],
            ),
            EvaluationCase(
                question="Vector search engine",
                expected_sources=["/docs/fastapi.md"],
            ),
        ],
    )

    report = await runner.evaluate_dataset(dataset)
    assert report.total_cases == 2
    assert report.mean_reciprocal_rank > 0.0
    assert 1 in report.mean_recall_at_k
