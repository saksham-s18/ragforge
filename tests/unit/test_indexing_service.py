from pathlib import Path

import pytest

from ragforge.adapters.chunkers import DeterministicChunker
from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.adapters.state import InMemoryIndexStateStore
from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.domain.enums import IndexingStatus
from ragforge.domain.exceptions import (
    DocumentNotFoundError,
    VectorDimensionMismatchError,
    VectorStoreError,
)
from ragforge.services.indexing import IndexingService, discover_documents


@pytest.fixture
def test_pipeline() -> tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore]:
    """Fixture providing an IndexingService wired with in-memory adapters."""
    dim = 16
    provider = DeterministicEmbeddingProvider(dimension=dim)
    store = InMemoryVectorStore(dimension=dim)
    state = InMemoryIndexStateStore()
    chunker = DeterministicChunker(chunk_size=100, chunk_overlap=20)
    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
        chunker=chunker,
        state_store=state,
        batch_size=4,
    )
    return service, store, state


# 1. Empty directory
@pytest.mark.asyncio
async def test_indexing_empty_directory(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, _ = test_pipeline
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    result = await service.index_path(empty_dir)

    assert result.discovered_documents == 0
    assert result.indexed_documents == 0
    assert result.updated_documents == 0
    assert result.skipped_documents == 0
    assert result.failed_documents == 0
    assert result.chunks_created == 0
    assert result.vectors_upserted == 0
    assert result.duration_seconds >= 0.0
    assert result.is_success is True
    assert store.count() == 0


# 2. Missing path
@pytest.mark.asyncio
async def test_indexing_missing_path(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, _, _ = test_pipeline
    missing_path = tmp_path / "does_not_exist"

    with pytest.raises(DocumentNotFoundError, match="Indexing target path does not exist"):
        discover_documents(missing_path)

    with pytest.raises(DocumentNotFoundError, match="Indexing target path does not exist"):
        await service.index_path(missing_path)


# 3. Single document
@pytest.mark.asyncio
async def test_indexing_single_document(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, _ = test_pipeline
    doc_file = tmp_path / "single.txt"
    doc_file.write_text("This is a single document for testing.", encoding="utf-8")

    result = await service.index_path(doc_file)

    assert result.discovered_documents == 1
    assert result.indexed_documents == 1
    assert result.failed_documents == 0
    assert result.chunks_created >= 1
    assert result.vectors_upserted == result.chunks_created
    assert store.count() == result.chunks_created

    # Test index_document helper directly
    doc2 = tmp_path / "single2.md"
    doc2.write_text("# Header\nMarkdown single document test.", encoding="utf-8")
    single_res = await service.index_document(doc2)
    assert single_res.status == IndexingStatus.INDEXED
    assert single_res.chunk_count >= 1


# 4. Recursive directory discovery
def test_recursive_directory_discovery(tmp_path: Path) -> None:
    (tmp_path / "dir_a" / "sub").mkdir(parents=True)
    (tmp_path / "dir_b").mkdir()

    file1 = tmp_path / "dir_a" / "doc1.txt"
    file2 = tmp_path / "dir_a" / "sub" / "doc2.md"
    file3 = tmp_path / "dir_b" / "doc3.markdown"
    file4 = tmp_path / "root.text"

    file1.write_text("content 1", encoding="utf-8")
    file2.write_text("content 2", encoding="utf-8")
    file3.write_text("content 3", encoding="utf-8")
    file4.write_text("content 4", encoding="utf-8")

    discovered = discover_documents(tmp_path)
    assert len(discovered) == 4
    resolved_discovered = [p.resolve() for p in discovered]
    assert file1.resolve() in resolved_discovered
    assert file2.resolve() in resolved_discovered
    assert file3.resolve() in resolved_discovered
    assert file4.resolve() in resolved_discovered


# 5. Unsupported file types ignored
def test_unsupported_file_types_ignored(tmp_path: Path) -> None:
    valid_txt = tmp_path / "doc.txt"
    valid_md = tmp_path / "doc.md"
    ignored_img = tmp_path / "image.png"
    ignored_json = tmp_path / "data.json"
    ignored_py = tmp_path / "script.py"

    valid_txt.write_text("valid text", encoding="utf-8")
    valid_md.write_text("# valid md", encoding="utf-8")
    ignored_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    ignored_json.write_text('{"key": "value"}', encoding="utf-8")
    ignored_py.write_text("print('hello')", encoding="utf-8")

    discovered = discover_documents(tmp_path)
    assert len(discovered) == 2
    assert valid_txt.resolve() in discovered
    assert valid_md.resolve() in discovered


# 6. New document indexing
@pytest.mark.asyncio
async def test_new_document_indexing(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "new_doc.txt"
    doc_file.write_text("Indexing brand new document content.", encoding="utf-8")

    result = await service.index_path(doc_file)

    assert result.indexed_documents == 1
    assert result.updated_documents == 0
    assert result.skipped_documents == 0
    assert result.documents[0].status == IndexingStatus.INDEXED
    assert state.count() == 1
    assert store.count() > 0


# 7. Unchanged document skipped
@pytest.mark.asyncio
async def test_unchanged_document_skipped(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "doc_skip.txt"
    doc_file.write_text("Fixed content that will not change.", encoding="utf-8")

    # First run indexes
    first_res = await service.index_path(doc_file)
    assert first_res.indexed_documents == 1
    initial_chunks = store.count()

    # Second run should skip
    second_res = await service.index_path(doc_file)
    assert second_res.discovered_documents == 1
    assert second_res.indexed_documents == 0
    assert second_res.updated_documents == 0
    assert second_res.skipped_documents == 1
    assert second_res.failed_documents == 0
    assert second_res.chunks_created == 0
    assert second_res.vectors_upserted == 0
    assert second_res.documents[0].status == IndexingStatus.SKIPPED

    # Ensure store count was completely untouched
    assert store.count() == initial_chunks


# 8. Changed document re-indexed
@pytest.mark.asyncio
async def test_changed_document_reindexed(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "doc_mutate.txt"
    doc_file.write_text("Version 1 content before any changes.", encoding="utf-8")

    # Initial indexing
    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    assert res1.updated_documents == 0

    # Modify content
    doc_file.write_text("Version 2 radically different updated content.", encoding="utf-8")

    # Re-indexing
    res2 = await service.index_path(doc_file)
    assert res2.discovered_documents == 1
    assert res2.indexed_documents == 0
    assert res2.updated_documents == 1
    assert res2.skipped_documents == 0
    assert res2.documents[0].status == IndexingStatus.UPDATED


# 9. Stale chunks removed
@pytest.mark.asyncio
async def test_stale_chunks_removed_on_update(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "stale_test.txt"

    # Write long text generating multiple chunks
    paragraph = "Sentence testing chunking in RAGForge production pipeline. " * 10
    multi_chunk_text = f"Paragraph One:\n{paragraph}\n\nParagraph Two:\n{paragraph}"
    doc_file.write_text(multi_chunk_text, encoding="utf-8")

    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    old_chunk_count = res1.chunks_created
    assert old_chunk_count >= 2
    assert store.count() == old_chunk_count

    # Replace with very short text generating only 1 chunk
    short_text = "Just a short single sentence."
    doc_file.write_text(short_text, encoding="utf-8")

    res2 = await service.index_path(doc_file)
    assert res2.updated_documents == 1
    assert res2.chunks_created == 1
    assert res2.vectors_upserted == 1

    # Stale chunks from version 1 must be gone; only the 1 new chunk remains
    assert store.count() == 1


# 10. Batch embedding
@pytest.mark.asyncio
async def test_batch_embedding_chunk_batches(tmp_path: Path) -> None:
    dim = 16
    mock_provider = DeterministicEmbeddingProvider(dimension=dim)
    original_embed_texts = mock_provider.embed_texts

    recorded_batch_sizes: list[int] = []

    async def spy_embed_texts(texts: list[str]) -> list[list[float]]:
        recorded_batch_sizes.append(len(texts))
        return await original_embed_texts(texts)

    mock_provider.embed_texts = spy_embed_texts  # type: ignore[method-assign]

    store = InMemoryVectorStore(dimension=dim)
    chunker = DeterministicChunker(chunk_size=50, chunk_overlap=10)
    service = IndexingService(
        embedding_provider=mock_provider,
        vector_store=store,
        chunker=chunker,
        batch_size=3,  # small batch size to test batching
    )

    doc_file = tmp_path / "batch_test.txt"
    doc_file.write_text(
        "Section 1\nAlpha beta gamma delta.\n\n"
        "Section 2\nEpsilon zeta eta theta.\n\n"
        "Section 3\nIota kappa lambda mu.\n\n"
        "Section 4\nNu xi omicron pi rho.\n\n"
        "Section 5\nSigma tau upsilon phi chi psi omega.\n\n",
        encoding="utf-8",
    )

    result = await service.index_path(doc_file)
    assert result.chunks_created >= 4

    # All recorded batch sizes must be <= configured batch_size (3)
    assert len(recorded_batch_sizes) >= 2
    for b_size in recorded_batch_sizes:
        assert 0 < b_size <= 3


# 11. Multiple documents
@pytest.mark.asyncio
async def test_multiple_documents_indexing(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, _ = test_pipeline
    for i in range(5):
        (tmp_path / f"doc_{i}.txt").write_text(f"Content for document {i}", encoding="utf-8")

    result = await service.index_path(tmp_path)
    assert result.discovered_documents == 5
    assert result.indexed_documents == 5
    assert result.updated_documents == 0
    assert result.skipped_documents == 0
    assert result.failed_documents == 0
    assert result.chunks_created >= 5
    assert store.count() == result.chunks_created


# 12. One document failure does not corrupt successful results
@pytest.mark.asyncio
async def test_one_document_failure_isolation(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, _ = test_pipeline
    good_doc = tmp_path / "good.txt"
    good_doc.write_text("Valid text that parses cleanly.", encoding="utf-8")

    empty_doc = tmp_path / "bad_empty.txt"
    empty_doc.write_text("   \n\t  ", encoding="utf-8")  # Whitespace only raises EmptyDocumentError

    result = await service.index_path(tmp_path)

    assert result.discovered_documents == 2
    assert result.indexed_documents == 1
    assert result.failed_documents == 1
    assert len(result.errors) == 1
    assert "empty or contains only whitespace" in result.errors[0]
    assert store.count() >= 1


# 13. Deterministic processing order
def test_deterministic_processing_order(tmp_path: Path) -> None:
    names = ["zeta.txt", "alpha.txt", "beta.txt", "gamma.txt"]
    for name in names:
        (tmp_path / name).write_text(f"content {name}", encoding="utf-8")

    discovered = discover_documents(tmp_path)
    discovered_names = [p.name for p in discovered]
    assert discovered_names == sorted(names)


# 14. Provider abstraction works with deterministic embeddings
@pytest.mark.asyncio
async def test_provider_abstraction_with_deterministic_embeddings(tmp_path: Path) -> None:
    dim = 32
    provider = DeterministicEmbeddingProvider(dimension=dim)
    store = InMemoryVectorStore(dimension=dim)
    service = IndexingService(embedding_provider=provider, vector_store=store)

    doc = tmp_path / "det.txt"
    doc.write_text("Deterministic embedding test document.", encoding="utf-8")

    result = await service.index_path(doc)
    assert result.indexed_documents == 1

    # Search the vector store to verify embeddings are valid and searchable
    query_vec = await provider.embed_query("Deterministic embedding")
    search_res = await store.search(query_vec, top_k=1)
    assert len(search_res) == 1
    assert search_res[0].score > 0.0


# 15. Vector dimension mismatch is surfaced correctly
@pytest.mark.asyncio
async def test_vector_dimension_mismatch_surfaced(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=64)  # Mismatch: 64 != 32
    service = IndexingService(embedding_provider=provider, vector_store=store)

    doc = tmp_path / "mismatch.txt"
    doc.write_text("Mismatch testing content.", encoding="utf-8")

    with pytest.raises(VectorDimensionMismatchError, match="does not match store"):
        await service.index_path(doc)


# 16. Indexing statistics are correct
@pytest.mark.asyncio
async def test_indexing_statistics_correctness(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, _, _ = test_pipeline
    doc1 = tmp_path / "d1.txt"
    doc1.write_text("Doc 1 content", encoding="utf-8")
    doc2 = tmp_path / "d2.txt"
    doc2.write_text("Doc 2 content", encoding="utf-8")

    # Initial run: 2 new docs
    res1 = await service.index_path(tmp_path)
    assert res1.discovered_documents == 2
    assert res1.indexed_documents == 2
    assert res1.updated_documents == 0
    assert res1.skipped_documents == 0
    assert res1.failed_documents == 0
    assert res1.total_processed == 2
    assert res1.is_success is True
    assert res1.duration_seconds >= 0.0

    # Mutate 1 doc, leave 1 unchanged, add 1 bad doc
    doc1.write_text("Doc 1 MUTATED content", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("", encoding="utf-8")

    res2 = await service.index_path(tmp_path)
    assert res2.discovered_documents == 3
    assert res2.indexed_documents == 0
    assert res2.updated_documents == 1
    assert res2.skipped_documents == 1
    assert res2.failed_documents == 1
    assert res2.total_processed == 3
    assert res2.is_success is False
    assert len(res2.documents) == 3


# 17. Force reindex unchanged document
@pytest.mark.asyncio
async def test_force_reindex_unchanged_document(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "force_doc.txt"
    doc_file.write_text("Stable content for testing force reindex.", encoding="utf-8")

    # Initial indexing: newly indexed
    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    assert res1.updated_documents == 0
    assert res1.skipped_documents == 0
    assert res1.documents[0].status == IndexingStatus.INDEXED
    doc_id_1 = res1.documents[0].document_id

    # Second indexing without force: skipped
    res2 = await service.index_path(doc_file, force=False)
    assert res2.indexed_documents == 0
    assert res2.updated_documents == 0
    assert res2.skipped_documents == 1
    assert res2.documents[0].status == IndexingStatus.SKIPPED

    # Third indexing with force=True: reindexed as UPDATED, bypassing skip
    res3 = await service.index_path(doc_file, force=True)
    assert res3.indexed_documents == 0
    assert res3.updated_documents == 1
    assert res3.skipped_documents == 0
    assert res3.documents[0].status == IndexingStatus.UPDATED
    assert res3.documents[0].document_id == doc_id_1
    assert store.count() > 0


# 18. Force reindex causes embedding and upsert work
@pytest.mark.asyncio
async def test_force_reindex_causes_embedding_and_upsert_work(
    tmp_path: Path,
) -> None:
    dim = 16
    mock_provider = DeterministicEmbeddingProvider(dimension=dim)
    store = InMemoryVectorStore(dimension=dim)
    state = InMemoryIndexStateStore()

    embed_call_count = 0
    original_embed = mock_provider.embed_texts

    async def spy_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_call_count
        embed_call_count += 1
        return await original_embed(texts)

    mock_provider.embed_texts = spy_embed  # type: ignore[method-assign]

    service = IndexingService(
        embedding_provider=mock_provider,
        vector_store=store,
        state_store=state,
    )

    doc = tmp_path / "work_doc.txt"
    doc.write_text("Content to monitor embedding invocations.", encoding="utf-8")

    # First run: should embed
    await service.index_path(doc)
    assert embed_call_count == 1

    # Second run without force: should skip, no new embed call
    await service.index_path(doc, force=False)
    assert embed_call_count == 1

    # Third run with force=True: must trigger embed call
    await service.index_path(doc, force=True)
    assert embed_call_count == 2


# 19. Force reindex restores vectors when vector store was cleared
@pytest.mark.asyncio
async def test_force_reindex_restores_cleared_vector_store(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "recovery_doc.txt"
    doc_file.write_text("Content that survives state cache drift.", encoding="utf-8")

    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    original_chunks = store.count()
    assert original_chunks > 0
    assert state.count() == 1

    # Simulate Qdrant data loss / clear out-of-band while state store file remains
    await store.clear()
    assert store.count() == 0

    # Normal run skips because state store still has record
    res2 = await service.index_path(doc_file, force=False)
    assert res2.skipped_documents == 1
    assert store.count() == 0  # Still empty!

    # Run with force=True: repopulates vector store
    res3 = await service.index_path(doc_file, force=True)
    assert res3.updated_documents == 1
    assert res3.skipped_documents == 0
    assert store.count() == original_chunks
    assert state.count() == 1


# 20. Safe update order: new upsert failure leaves old vectors and state intact
@pytest.mark.asyncio
async def test_safe_update_order_upsert_failure_preserves_old_vectors(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "upsert_fail_doc.txt"
    doc_file.write_text("Version 1 original content.", encoding="utf-8")

    # Initial indexing: version 1
    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    v1_doc_id = res1.documents[0].document_id
    v1_chunk_count = store.count()
    assert v1_chunk_count > 0

    # Modify content: version 2
    doc_file.write_text("Version 2 changed content that will fail during upsert.", encoding="utf-8")

    # Monkeypatch store.upsert to fail
    async def failing_upsert(chunks: object) -> None:
        raise VectorStoreError("Simulated network failure during vector upsert.")

    store.upsert = failing_upsert  # type: ignore[method-assign]

    # Run indexing: must raise VectorStoreError
    with pytest.raises(VectorStoreError, match="Simulated network failure"):
        await service.index_path(doc_file)

    # CRITICAL: Old vectors for v1 must STILL BE IN STORE (safe ordering guarantees no prior delete)
    assert store.count() == v1_chunk_count
    for chunk in store._storage.values():
        assert chunk.document_id == v1_doc_id

    # CRITICAL: State store must still reflect version 1
    record = await state.get(str(doc_file.resolve()))
    assert record is not None
    assert record.document_id == v1_doc_id
    assert record.content_hash == res1.documents[0].content_hash


# 21. Safe update order: delete failure does not falsely persist state
@pytest.mark.asyncio
async def test_safe_update_order_delete_failure_preserves_recoverable_state(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "delete_fail_doc.txt"
    doc_file.write_text("Version 1 content.", encoding="utf-8")

    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    v1_doc_id = res1.documents[0].document_id

    doc_file.write_text("Version 2 updated content.", encoding="utf-8")

    # Monkeypatch delete_by_document_id to fail after upsert
    async def failing_delete(document_id: object) -> None:
        raise VectorStoreError("Simulated failure deleting old document vectors.")

    store.delete_by_document_id = failing_delete  # type: ignore[method-assign]

    with pytest.raises(VectorStoreError, match="Simulated failure deleting old document"):
        await service.index_path(doc_file)

    # State store must NOT have been updated to Version 2 (no false success)
    record = await state.get(str(doc_file.resolve()))
    assert record is not None
    assert record.document_id == v1_doc_id
    assert record.content_hash == res1.documents[0].content_hash


# 22. Safe update order: successful update replaces old vectors with new vectors
@pytest.mark.asyncio
async def test_safe_update_order_successful_update_replaces_vectors(
    test_pipeline: tuple[IndexingService, InMemoryVectorStore, InMemoryIndexStateStore],
    tmp_path: Path,
) -> None:
    service, store, state = test_pipeline
    doc_file = tmp_path / "clean_update_doc.txt"
    doc_file.write_text("Version 1 initial text.", encoding="utf-8")

    res1 = await service.index_path(doc_file)
    v1_doc_id = res1.documents[0].document_id

    doc_file.write_text(
        "Version 2 updated text with completely different content.",
        encoding="utf-8",
    )
    res2 = await service.index_path(doc_file)
    assert res2.updated_documents == 1
    v2_doc_id = res2.documents[0].document_id
    assert v1_doc_id != v2_doc_id

    # Verify only new document chunks exist in store
    assert store.count() > 0
    for chunk in store._storage.values():
        assert chunk.document_id == v2_doc_id
        assert chunk.document_id != v1_doc_id

    # Verify state reflects new document
    record = await state.get(str(doc_file.resolve()))
    assert record is not None
    assert record.document_id == v2_doc_id
