import hashlib
from uuid import uuid4

import pytest

from ragforge.adapters.chunkers import (
    DeterministicChunker,
    approximate_token_count,
)
from ragforge.domain.enums import DocumentStatus, MimeType
from ragforge.domain.exceptions import ChunkingError
from ragforge.domain.models import Document


def _make_document(
    content: str | None,
    title: str = "Test Doc",
    source_type: MimeType = MimeType.TXT,
) -> Document:
    """Helper to instantiate a domain Document for chunker testing."""
    raw = content or ""
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return Document(
        id=uuid4(),
        title=title,
        source_type=source_type,
        content_hash=content_hash,
        raw_content=content,
        file_path="/path/to/test.txt",
        metadata={"file_name": "test.txt"},
        status=DocumentStatus.PENDING,
    )


def test_approximate_token_count() -> None:
    """Verify default heuristic token counter estimation and empty behavior."""
    assert approximate_token_count("") == 0
    assert approximate_token_count("Hi") == 1
    # ~4 chars per token: 16 chars -> ~4 tokens
    assert approximate_token_count("1234567890123456") == 4


def test_chunker_invalid_configuration() -> None:
    """Verify invalid chunk_size or overlap parameters raise ChunkingError."""
    with pytest.raises(ChunkingError, match="chunk_size must be greater than 0"):
        DeterministicChunker(chunk_size=0)

    with pytest.raises(ChunkingError, match="chunk_size must be greater than 0"):
        DeterministicChunker(chunk_size=-10)

    with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative"):
        DeterministicChunker(chunk_size=500, chunk_overlap=-1)

    with pytest.raises(ChunkingError, match="strictly less than chunk_size"):
        DeterministicChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ChunkingError, match="strictly less than chunk_size"):
        DeterministicChunker(chunk_size=500, chunk_overlap=600)


def test_chunker_short_document() -> None:
    """Verify document shorter than chunk_size produces a single, well-formed Chunk."""
    text = "RAGForge provides deterministic ingestion and chunking."
    doc = _make_document(text)

    chunker = DeterministicChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.document_id == doc.id
    assert chunk.chunk_index == 0
    assert chunk.content == text
    assert chunk.token_count > 0
    assert chunk.content_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert chunk.metadata.start_char_idx == 0
    assert chunk.metadata.end_char_idx == len(text)
    assert chunk.metadata.extra["document_title"] == "Test Doc"


def test_chunker_multi_paragraph_document() -> None:
    """Verify multi-paragraph document is partitioned cleanly along natural boundaries."""
    p1 = "First paragraph introduces the high-level architecture and principles."
    p2 = "Second paragraph discusses clean ports, domain boundaries, and adapters."
    p3 = "Third paragraph wraps up with deterministic testing and verification steps."
    text = f"{p1}\n\n{p2}\n\n{p3}"
    doc = _make_document(text)

    # Chunk size tailored to fit single paragraphs but not all together
    chunker = DeterministicChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    assert all(len(c.content.strip()) > 0 for c in chunks)

    # Verify all chunks maintain sequential index and valid document ID
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc.id
        assert c.content_hash == hashlib.sha256(c.content.encode("utf-8")).hexdigest()


def test_chunker_long_document() -> None:
    """Verify very long document is segmented into multiple sequential chunks without gaps."""
    paragraphs = [
        f"Paragraph {i}: " + ("Enterprise RAG architecture requires resilient chunking. " * 8)
        for i in range(25)
    ]
    text = "\n\n".join(paragraphs)
    doc = _make_document(text)

    chunker = DeterministicChunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 10
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc.id
        assert len(c.content) > 0
        assert c.token_count > 0


def test_chunker_configured_chunk_size_and_overlap() -> None:
    """Verify varying chunk sizes and zero-overlap produce expected segmentations."""
    text = "Word1 " * 200  # 1200 chars
    doc = _make_document(text)

    chunker_zero_overlap = DeterministicChunker(chunk_size=300, chunk_overlap=0)
    chunks_no_overlap = chunker_zero_overlap.chunk(doc)

    chunker_with_overlap = DeterministicChunker(chunk_size=300, chunk_overlap=100)
    chunks_with_overlap = chunker_with_overlap.chunk(doc)

    assert len(chunks_with_overlap) > len(chunks_no_overlap)


def test_chunker_no_empty_chunks() -> None:
    """Verify documents with irregular whitespace or blank lines never produce empty chunks."""
    text = (
        "Heading text\n\n\n\n\n\n\n\n"
        "   Second paragraph with lots of surrounding whitespace.   \n\n\n"
        "Third paragraph.   \t\t\n\n"
    )
    doc = _make_document(text)

    chunker = DeterministicChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for c in chunks:
        assert len(c.content.strip()) > 0
        assert c.token_count >= 1


def test_chunker_deterministic_output() -> None:
    """Verify repeated chunking of identical document produces identical chunks and IDs."""
    text = (
        "# Overview\n\n"
        "RAGForge is an enterprise AI information retrieval platform.\n\n"
        "## Components\n\n"
        "Includes loaders, chunkers, embedding providers, and vector stores.\n\n"
        "## Observability\n\n"
        "Full telemetry, structured logging, and evaluation metrics are integrated."
    )
    doc = _make_document(text, source_type=MimeType.MARKDOWN)

    chunker = DeterministicChunker(chunk_size=120, chunk_overlap=25)
    run_1 = chunker.chunk(doc)
    run_2 = chunker.chunk(doc)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2, strict=True):
        assert c1.id == c2.id
        assert c1.chunk_index == c2.chunk_index
        assert c1.content == c2.content
        assert c1.content_hash == c2.content_hash
        assert c1.token_count == c2.token_count
        assert c1.metadata.section_header == c2.metadata.section_header
        assert c1.metadata.start_char_idx == c2.metadata.start_char_idx
        assert c1.metadata.end_char_idx == c2.metadata.end_char_idx


def test_chunker_deterministic_chunk_hashes() -> None:
    """Verify every chunk content_hash matches its exact SHA-256 digest."""
    text = "Some test text for hash calculation across multiple chunks.\n\n" * 5
    doc = _make_document(text)

    chunker = DeterministicChunker(chunk_size=80, chunk_overlap=15)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for c in chunks:
        expected = hashlib.sha256(c.content.encode("utf-8")).hexdigest()
        assert c.content_hash == expected


def test_chunker_document_id_propagation() -> None:
    """Verify chunk document_id matches the parent document ID."""
    doc = _make_document("Arbitrary content for ID verification." * 10)
    chunker = DeterministicChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(c.document_id == doc.id for c in chunks)


def test_chunker_chunk_indexes() -> None:
    """Verify chunk indexes form a strictly sequential 0-indexed integer series."""
    doc = _make_document("Sequential index verification. " * 30)
    chunker = DeterministicChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk(doc)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunker_metadata_provenance_preservation() -> None:
    """Verify Markdown headers and character offsets are preserved in ChunkMetadata."""
    markdown_text = (
        "# Introduction\n\n"
        "Welcome to the platform documentation.\n\n"
        "## Architecture Design\n\n"
        "The architecture is decoupled into ports and adapters.\n\n"
        "## Deployment Guidelines\n\n"
        "Production deployments use containerization and observability."
    )
    doc = _make_document(markdown_text, source_type=MimeType.MARKDOWN)

    chunker = DeterministicChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3

    # Verify character offsets correspond to exact substrings in the original document
    for c in chunks:
        assert c.metadata.start_char_idx is not None
        assert c.metadata.end_char_idx is not None
        original_span = markdown_text[c.metadata.start_char_idx : c.metadata.end_char_idx]
        assert c.content == original_span

    # Check that section headers are identified
    headers = [c.metadata.section_header for c in chunks if c.metadata.section_header]
    assert "Introduction" in headers
    assert "Architecture Design" in headers or "Deployment Guidelines" in headers


def test_chunker_empty_document_behavior() -> None:
    """Verify empty document or None content produces an empty chunk sequence."""
    chunker = DeterministicChunker(chunk_size=200, chunk_overlap=50)

    doc_empty = _make_document("")
    assert chunker.chunk(doc_empty) == []

    doc_none = _make_document(None)
    assert chunker.chunk(doc_none) == []


def test_chunker_whitespace_only_document_behavior() -> None:
    """Verify whitespace-only document produces an empty chunk sequence without error."""
    chunker = DeterministicChunker(chunk_size=200, chunk_overlap=50)

    doc_spaces = _make_document("     \n\n\t   \r\n   ")
    assert chunker.chunk(doc_spaces) == []


def test_chunker_custom_token_counter() -> None:
    """Verify pluggable token_counter callable is used by the chunker."""
    custom_counter_called = False

    def custom_counter(text: str) -> int:
        nonlocal custom_counter_called
        custom_counter_called = True
        return len(text.split())

    doc = _make_document("One two three four five.")
    chunker = DeterministicChunker(
        chunk_size=200,
        chunk_overlap=20,
        token_counter=custom_counter,
    )
    chunks = chunker.chunk(doc)

    assert custom_counter_called is True
    assert len(chunks) == 1
    assert chunks[0].token_count == 5
