import hashlib
from pathlib import Path

import pytest

from ragforge.adapters.loaders import MarkdownDocumentLoader, TextDocumentLoader
from ragforge.domain.enums import DocumentStatus, MimeType
from ragforge.domain.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestionError,
    UnsupportedFileTypeError,
)


@pytest.mark.asyncio
async def test_text_loader_valid_file(tmp_path: Path) -> None:
    """Verify loading a valid plain text file populates all expected Document fields."""
    file_path = tmp_path / "sample.txt"
    content = "Hello, RAGForge! This is a test plain text document."
    file_path.write_text(content, encoding="utf-8")

    loader = TextDocumentLoader()
    doc = await loader.load(file_path)

    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert doc.id is not None
    assert doc.title == "sample.txt"
    assert doc.source_type == MimeType.TXT
    assert doc.raw_content == content
    assert doc.content_hash == expected_hash
    assert doc.file_path == str(file_path.resolve())
    assert doc.status == DocumentStatus.PENDING
    assert doc.metadata["file_name"] == "sample.txt"
    assert doc.metadata["file_extension"] == ".txt"
    assert doc.metadata["file_size"] == len(content.encode("utf-8"))


@pytest.mark.asyncio
async def test_text_loader_text_extension(tmp_path: Path) -> None:
    """Verify TextDocumentLoader accepts .text files."""
    file_path = tmp_path / "document.text"
    file_path.write_text("Text extension content.", encoding="utf-8")

    loader = TextDocumentLoader()
    doc = await loader.load(file_path)

    assert doc.source_type == MimeType.TXT
    assert doc.metadata["file_extension"] == ".text"


@pytest.mark.asyncio
async def test_markdown_loader_valid_file_with_heading(tmp_path: Path) -> None:
    """Verify MarkdownDocumentLoader extracts title from first Markdown heading."""
    file_path = tmp_path / "guide.md"
    content = "# Enterprise Retrieval\n\nThis guide explains hybrid search and reranking."
    file_path.write_text(content, encoding="utf-8")

    loader = MarkdownDocumentLoader()
    doc = await loader.load(file_path)

    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert doc.id is not None
    assert doc.title == "Enterprise Retrieval"
    assert doc.source_type == MimeType.MARKDOWN
    assert doc.raw_content == content
    assert doc.content_hash == expected_hash
    assert doc.metadata["file_extension"] == ".md"


@pytest.mark.asyncio
async def test_markdown_loader_fallback_title(tmp_path: Path) -> None:
    """Verify MarkdownDocumentLoader falls back to filename when no heading exists."""
    file_path = tmp_path / "notes.markdown"
    content = "Just plain text without any markdown headings."
    file_path.write_text(content, encoding="utf-8")

    loader = MarkdownDocumentLoader()
    doc = await loader.load(file_path)

    assert doc.title == "notes.markdown"
    assert doc.source_type == MimeType.MARKDOWN
    assert doc.metadata["file_extension"] == ".markdown"


@pytest.mark.asyncio
async def test_loader_custom_title_and_metadata(tmp_path: Path) -> None:
    """Verify caller-provided title and extra metadata are merged cleanly."""
    file_path = tmp_path / "data.txt"
    file_path.write_text("Some text.", encoding="utf-8")

    loader = TextDocumentLoader()
    doc = await loader.load(
        file_path,
        title="Custom Title",
        metadata={"author": "Team RAGForge", "environment": "test"},
    )

    assert doc.title == "Custom Title"
    assert doc.metadata["author"] == "Team RAGForge"
    assert doc.metadata["environment"] == "test"
    assert doc.metadata["file_name"] == "data.txt"


@pytest.mark.asyncio
async def test_loader_unsupported_extension(tmp_path: Path) -> None:
    """Verify unsupported file extension raises UnsupportedFileTypeError."""
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_text("Dummy binary content", encoding="utf-8")

    text_loader = TextDocumentLoader()
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file extension '.pdf'"):
        await text_loader.load(pdf_file)

    md_loader = MarkdownDocumentLoader()
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file extension '.pdf'"):
        await md_loader.load(pdf_file)


@pytest.mark.asyncio
async def test_loader_missing_file(tmp_path: Path) -> None:
    """Verify non-existent file path raises DocumentNotFoundError."""
    missing_path = tmp_path / "does_not_exist.txt"
    loader = TextDocumentLoader()

    with pytest.raises(DocumentNotFoundError, match="Document file not found"):
        await loader.load(missing_path)


@pytest.mark.asyncio
async def test_loader_directory_source(tmp_path: Path) -> None:
    """Verify specifying a directory instead of a file raises IngestionError."""
    dir_path = tmp_path / "subfolder"
    dir_path.mkdir()
    loader = TextDocumentLoader()

    with pytest.raises(IngestionError, match="Specified source path is not a file"):
        await loader.load(dir_path)


@pytest.mark.asyncio
async def test_loader_empty_file_raises_by_default(tmp_path: Path) -> None:
    """Verify loading an empty file raises EmptyDocumentError by default."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = TextDocumentLoader()
    with pytest.raises(EmptyDocumentError, match="is empty or contains only whitespace"):
        await loader.load(empty_file)


@pytest.mark.asyncio
async def test_loader_whitespace_only_file_raises(tmp_path: Path) -> None:
    """Verify loading a whitespace-only file raises EmptyDocumentError by default."""
    ws_file = tmp_path / "spaces.txt"
    ws_file.write_text("   \n\t  \r\n   ", encoding="utf-8")

    loader = TextDocumentLoader()
    with pytest.raises(EmptyDocumentError, match="is empty or contains only whitespace"):
        await loader.load(ws_file)


@pytest.mark.asyncio
async def test_loader_empty_file_allowed(tmp_path: Path) -> None:
    """Verify loading an empty file with allow_empty=True produces an empty Document."""
    empty_file = tmp_path / "allowed_empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = TextDocumentLoader()
    doc = await loader.load(empty_file, allow_empty=True)

    assert doc.raw_content == ""
    assert doc.metadata["file_size"] == 0
    assert doc.content_hash == hashlib.sha256(b"").hexdigest()
    assert doc.id is not None


@pytest.mark.asyncio
async def test_loader_content_hash_determinism(tmp_path: Path) -> None:
    """Verify identical content across different files generates identical hash and ID."""
    content = "Identical content across two different source files."

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    file_a = dir_a / "file1.txt"
    file_b = dir_b / "file2.txt"
    file_a.write_text(content, encoding="utf-8")
    file_b.write_text(content, encoding="utf-8")

    loader = TextDocumentLoader()
    doc_a = await loader.load(file_a)
    doc_b = await loader.load(file_b)

    assert doc_a.content_hash == doc_b.content_hash
    assert doc_a.id == doc_b.id


@pytest.mark.asyncio
async def test_loader_repeated_loading_consistency(tmp_path: Path) -> None:
    """Verify repeatedly loading the exact same file produces consistent identity and hash."""
    file_path = tmp_path / "repeatable.md"
    file_path.write_text("# Repeated Loading\n\nTesting idempotency.", encoding="utf-8")

    loader = MarkdownDocumentLoader()
    doc1 = await loader.load(file_path)
    doc2 = await loader.load(file_path)

    assert doc1.id == doc2.id
    assert doc1.content_hash == doc2.content_hash
    assert doc1.title == doc2.title
    assert doc1.raw_content == doc2.raw_content
