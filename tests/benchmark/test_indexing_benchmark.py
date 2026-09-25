import time
from pathlib import Path

import pytest

from ragforge.adapters.chunkers import DeterministicChunker
from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.adapters.state import InMemoryIndexStateStore
from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.services.indexing import IndexingService


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_indexing_pipeline_synthetic_corpus_throughput(tmp_path: Path) -> None:
    """Benchmark indexing performance and throughput on a synthetic corpus of ~100-200 chunks."""
    corpus_dir = tmp_path / "synthetic_corpus"
    corpus_dir.mkdir()

    # Generate 10 documents, each with 10-15 paragraphs (~120-150 chunks total)
    doc_count = 10
    paragraphs_per_doc = 12
    for doc_idx in range(doc_count):
        sections = []
        for p_idx in range(paragraphs_per_doc):
            sections.append(
                f"### Document {doc_idx} - Section {p_idx}\n"
                f"This is a synthetic paragraph designed to evaluate indexing throughput. "
                f"It contains multiple sentences with vocabulary {doc_idx} and {p_idx}. "
                f"Vector generation, tokenization, hierarchical chunking, and similarity indexing "
                f"must operate efficiently in production pipelines.\n"
            )
        doc_file = corpus_dir / f"doc_{doc_idx:02d}.md"
        doc_file.write_text("\n\n".join(sections), encoding="utf-8")

    dim = 64
    provider = DeterministicEmbeddingProvider(dimension=dim)
    store = InMemoryVectorStore(dimension=dim)
    state = InMemoryIndexStateStore()
    chunker = DeterministicChunker(chunk_size=100, chunk_overlap=15)
    batch_size = 32

    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
        chunker=chunker,
        state_store=state,
        batch_size=batch_size,
    )

    # 1. Measure initial indexing throughput
    start_time = time.perf_counter()
    result = await service.index_path(corpus_dir)
    initial_duration = time.perf_counter() - start_time

    assert result.discovered_documents == doc_count
    assert result.indexed_documents == doc_count
    assert result.failed_documents == 0
    assert result.chunks_created >= 100
    assert result.vectors_upserted == result.chunks_created

    chunks_per_second = result.chunks_created / max(initial_duration, 0.001)

    print(
        f"\n[BENCHMARK] Indexed {result.chunks_created} chunks across {doc_count} documents "
        f"in {initial_duration:.4f}s ({chunks_per_second:.1f} chunks/sec, batch_size={batch_size})"
    )

    # 2. Measure incremental skip performance on unchanged corpus
    start_skip = time.perf_counter()
    skip_result = await service.index_path(corpus_dir)
    skip_duration = time.perf_counter() - start_skip

    assert skip_result.skipped_documents == doc_count
    assert skip_result.indexed_documents == 0
    assert skip_result.vectors_upserted == 0

    print(
        f"[BENCHMARK] Incremental check for {doc_count} unchanged documents completed in "
        f"{skip_duration:.4f}s (speedup: {initial_duration / max(skip_duration, 0.0001):.1f}x)"
    )
