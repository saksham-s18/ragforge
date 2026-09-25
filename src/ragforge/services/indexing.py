import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from ragforge.adapters.chunkers import DeterministicChunker
from ragforge.adapters.loaders import (
    MarkdownDocumentLoader,
    TextDocumentLoader,
)
from ragforge.adapters.state import InMemoryIndexStateStore
from ragforge.domain.enums import IndexingStatus
from ragforge.domain.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestionError,
)
from ragforge.domain.models import (
    Chunk,
    Document,
    DocumentIndexingResult,
    DocumentIndexRecord,
    IndexingResult,
)
from ragforge.ports.chunkers import BaseChunker
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.loaders import BaseDocumentLoader
from ragforge.ports.state import BaseIndexStateStore
from ragforge.ports.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_EXTENSIONS = {".txt", ".text", ".md", ".markdown"}


def discover_documents(
    path: str | Path,
    supported_extensions: set[str] | None = None,
) -> list[Path]:
    """Recursively discover supported document files under a given path.

    Args:
        path: Path to a file or directory.
        supported_extensions: Optional set of allowed lowercase file extensions.
            Defaults to {'.txt', '.text', '.md', '.markdown'}.

    Returns:
        Deterministically sorted list of unique, resolved Path objects.

    Raises:
        DocumentNotFoundError: If the supplied path does not exist.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise DocumentNotFoundError(f"Indexing target path does not exist: {target}")

    extensions = (
        {ext.lower() for ext in supported_extensions}
        if supported_extensions is not None
        else DEFAULT_SUPPORTED_EXTENSIONS
    )

    if target.is_file():
        if target.suffix.lower() in extensions:
            return [target]
        return []

    discovered: list[Path] = []
    for candidate in target.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in extensions:
            discovered.append(candidate.resolve())

    # Safely deduplicate preserving order, then sort stably using POSIX path string
    unique_paths = list(dict.fromkeys(discovered))
    unique_paths.sort(key=lambda p: p.as_posix())
    return unique_paths


@dataclass
class _PreparedDoc:
    path_str: str
    doc: Document
    chunks: list[Chunk]
    is_update: bool
    old_document_id: UUID | None
    content_hash: str


class IndexingService:
    """Production document indexing pipeline orchestrator.

    Coordinates document discovery, loading, chunking, content hashing,
    incremental change detection, batch embedding generation, stale chunk cleanup,
    and vector store upserting.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        chunker: BaseChunker | None = None,
        loaders: dict[str, BaseDocumentLoader] | None = None,
        state_store: BaseIndexStateStore | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the indexing service.

        Args:
            embedding_provider: Dense embedding provider adapter.
            vector_store: Target vector store adapter.
            chunker: Chunker adapter (defaults to DeterministicChunker).
            loaders: Mapping of extension to loader adapters.
            state_store: State persistence adapter (defaults to InMemoryIndexStateStore).
            batch_size: Batch size for embedding and upsert operations (must be > 0).

        Raises:
            ValueError: If batch_size <= 0.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunker = chunker or DeterministicChunker()
        self._state_store = state_store or InMemoryIndexStateStore()
        self._batch_size = batch_size

        if loaders is not None:
            self._loaders = loaders
        else:
            text_loader = TextDocumentLoader()
            md_loader = MarkdownDocumentLoader()
            self._loaders = {
                ".txt": text_loader,
                ".text": text_loader,
                ".md": md_loader,
                ".markdown": md_loader,
            }

    @property
    def supported_extensions(self) -> set[str]:
        """Set of supported lowercase file extensions."""
        return set(self._loaders.keys())

    async def index_path(
        self,
        path: str | Path,
        force: bool = False,
        **kwargs: Any,
    ) -> IndexingResult:
        """Discover and index documents from a directory or single file path.

        Args:
            path: Path to single document file or directory to crawl recursively.
            force: If True, bypass unchanged-content skip checks and reindex documents.
            **kwargs: Extra parameters passed to index_files.

        Returns:
            Structured IndexingResult report.
        """
        files = discover_documents(path, supported_extensions=self.supported_extensions)
        return await self.index_files(files, force=force, **kwargs)

    async def index_document(
        self,
        path: str | Path,
        force: bool = False,
        **kwargs: Any,
    ) -> DocumentIndexingResult:
        """Index a single document file.

        Args:
            path: Path to the document.
            force: If True, bypass unchanged-content skip checks and reindex document.
            **kwargs: Extra parameters.

        Returns:
            DocumentIndexingResult for the single document.
        """
        result = await self.index_files([path], force=force, **kwargs)
        if result.documents:
            return result.documents[0]
        return DocumentIndexingResult(
            file_path=str(Path(path).resolve()),
            status=IndexingStatus.FAILED,
            error="No indexing result produced for document.",
        )

    async def index_files(
        self,
        paths: Sequence[str | Path],
        force: bool = False,
        **kwargs: Any,
    ) -> IndexingResult:
        """Execute incremental indexing over a sequence of document file paths.

        Args:
            paths: Sequence of file paths to process.
            force: If True, bypass unchanged-content skip checks and reindex documents.
            **kwargs: Additional parameters.

        Returns:
            Aggregated IndexingResult with full statistics and document outcomes.
        """
        start_time = time.perf_counter()
        logger.info("Indexing started for %d file(s)", len(paths))

        discovered_count = len(paths)
        indexed_count = 0
        updated_count = 0
        skipped_count = 0
        failed_count = 0
        chunks_created_count = 0
        vectors_upserted_count = 0

        doc_results: list[DocumentIndexingResult] = []
        error_messages: list[str] = []
        prepared_docs: list[_PreparedDoc] = []

        # Phase 1: Discovery, Change Detection, Loader & Chunker
        for item_path in paths:
            resolved_path = Path(item_path).resolve()
            path_str = str(resolved_path)
            ext = resolved_path.suffix.lower()
            loader = self._loaders.get(ext)

            if loader is None:
                err_msg = f"Unsupported file extension '{ext}' for file: {path_str}"
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Failed to index document %s: %s", path_str, err_msg)
                continue

            if not resolved_path.exists():
                err_msg = f"Document file not found at path: {path_str}"
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Failed to index document %s: %s", path_str, err_msg)
                continue

            # Read content and compute SHA-256 content hash
            try:
                raw_content = resolved_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    raw_content = resolved_path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    err_msg = f"Failed to read file '{path_str}': {exc}"
                    failed_count += 1
                    doc_results.append(
                        DocumentIndexingResult(
                            file_path=path_str,
                            status=IndexingStatus.FAILED,
                            error=err_msg,
                        )
                    )
                    error_messages.append(err_msg)
                    logger.error("Failed to index document %s: %s", path_str, err_msg)
                    continue
            except Exception as exc:
                err_msg = f"Operating system error reading file '{path_str}': {exc}"
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Failed to index document %s: %s", path_str, err_msg)
                continue

            if not raw_content.strip():
                err_msg = (
                    f"File '{resolved_path.name}' is empty or contains only whitespace characters."
                )
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Document failure: %s", err_msg)
                continue

            content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

            # Incremental change detection via state store
            existing_record = await self._state_store.get(path_str)
            if (
                not force
                and existing_record is not None
                and existing_record.content_hash == content_hash
            ):
                skipped_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        document_id=existing_record.document_id,
                        status=IndexingStatus.SKIPPED,
                        chunk_count=existing_record.chunk_count,
                        content_hash=content_hash,
                    )
                )
                logger.info(
                    "Document skipped (unchanged): %s (hash: %s)",
                    path_str,
                    content_hash,
                )
                continue

            is_update = existing_record is not None
            old_doc_id = existing_record.document_id if existing_record is not None else None

            # Parse and chunk document
            try:
                doc = await loader.load(resolved_path)
                chunks = list(self._chunker.chunk(doc))
            except (EmptyDocumentError, IngestionError) as exc:
                err_msg = str(exc)
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Document failure for %s: %s", path_str, err_msg)
                continue
            except Exception as exc:
                err_msg = f"Unexpected error processing '{path_str}': {exc}"
                failed_count += 1
                doc_results.append(
                    DocumentIndexingResult(
                        file_path=path_str,
                        status=IndexingStatus.FAILED,
                        error=err_msg,
                    )
                )
                error_messages.append(err_msg)
                logger.error("Document failure for %s: %s", path_str, err_msg)
                continue

            prepared_docs.append(
                _PreparedDoc(
                    path_str=path_str,
                    doc=doc,
                    chunks=chunks,
                    is_update=is_update,
                    old_document_id=old_doc_id,
                    content_hash=content_hash,
                )
            )

        # Phase 2: Batch Embedding Generation
        all_chunks: list[Chunk] = [c for p in prepared_docs for c in p.chunks]
        chunks_created_count = len(all_chunks)

        if all_chunks:
            for i in range(0, len(all_chunks), self._batch_size):
                batch_chunks = all_chunks[i : i + self._batch_size]
                texts = [c.content for c in batch_chunks]
                vectors = await self._embedding_provider.embed_texts(texts)
                for chunk, vec in zip(batch_chunks, vectors, strict=True):
                    chunk.dense_vector = vec

        # Phase 3: Vector Store Upsert & Safe Stale Chunk Removal
        # Upsert chunks into vector store FIRST in batches
        if all_chunks:
            for i in range(0, len(all_chunks), self._batch_size):
                batch_chunks = all_chunks[i : i + self._batch_size]
                await self._vector_store.upsert(batch_chunks)
            vectors_upserted_count = len(all_chunks)

        # Remove stale chunks for updated documents ONLY after new upsert succeeds
        for p in prepared_docs:
            if p.is_update and p.old_document_id is not None and p.old_document_id != p.doc.id:
                await self._vector_store.delete_by_document_id(p.old_document_id)

        # Phase 4: State Store Updates and Reporting
        for p in prepared_docs:
            new_record = DocumentIndexRecord(
                file_path=p.path_str,
                document_id=p.doc.id,
                content_hash=p.content_hash,
                chunk_count=len(p.chunks),
                metadata={
                    "title": p.doc.title,
                    "source_type": p.doc.source_type,
                    **p.doc.metadata,
                },
            )
            await self._state_store.set(new_record)

            if p.is_update:
                updated_count += 1
                status = IndexingStatus.UPDATED
                logger.info(
                    "Document updated: %s (id: %s, chunks: %d)",
                    p.path_str,
                    p.doc.id,
                    len(p.chunks),
                )
            else:
                indexed_count += 1
                status = IndexingStatus.INDEXED
                logger.info(
                    "Document indexed: %s (id: %s, chunks: %d)",
                    p.path_str,
                    p.doc.id,
                    len(p.chunks),
                )

            doc_results.append(
                DocumentIndexingResult(
                    file_path=p.path_str,
                    document_id=p.doc.id,
                    status=status,
                    chunk_count=len(p.chunks),
                    content_hash=p.content_hash,
                )
            )

        duration = round(time.perf_counter() - start_time, 4)
        logger.info(
            "Indexing completed: %d indexed, %d updated, %d skipped, %d failed in %.2fs",
            indexed_count,
            updated_count,
            skipped_count,
            failed_count,
            duration,
        )

        return IndexingResult(
            discovered_documents=discovered_count,
            indexed_documents=indexed_count,
            updated_documents=updated_count,
            skipped_documents=skipped_count,
            failed_documents=failed_count,
            chunks_created=chunks_created_count,
            vectors_upserted=vectors_upserted_count,
            duration_seconds=duration,
            documents=doc_results,
            errors=error_messages,
        )
