import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID

from ragforge.domain.models import Chunk, RetrievedChunk
from ragforge.ports.lexical import BaseLexicalIndex

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Deterministically tokenize text into lowercased alphanumeric terms.

    Splits text on non-alphanumeric boundaries while preserving words and numbers,
    normalizing whitespace, stripping punctuation, and converting to lowercase.
    Stopwords are intentionally retained for deterministic retrieval.

    Args:
        text: Raw document or query text.

    Returns:
        List of lowercased token strings.
    """
    return re.findall(r"\w+", text.lower())


class BM25LexicalIndex(BaseLexicalIndex):
    """Deterministic, pure-Python Okapi BM25 lexical index implementing BaseLexicalIndex.

    Maintains an inverted index of tokenized terms to chunk postings with term
    frequencies, document lengths, and document frequencies. Implements standard Okapi
    BM25 with non-negative Robertson-Spärck Jones IDF and length normalization.
    Optionally persists state to disk via an atomic JSON file for process lifecycle survival.

    Concurrency Model:
        - Safe for concurrent readers across multiple processes or workers (each reader
          detects file mtime changes and reloads updated index snapshots atomically).
        - Safe for single-writer + multiple-reader usage (e.g. single CLI ingestion
          process updating the index file while API workers serve queries).
        - Concurrent writers are NOT serialized without external locking; multiple
          simultaneous writers may overwrite each other's snapshots.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_file: str | Path | None = None,
    ) -> None:
        """Initialize the BM25 lexical index.

        Args:
            k1: Term frequency saturation parameter (k1 >= 0, default 1.5).
            b: Document length normalization parameter (0.0 <= b <= 1.0, default 0.75).
            index_file: Optional path to JSON file for durable state persistence.

        Raises:
            ValueError: If k1 < 0 or b is outside [0.0, 1.0].
        """
        if k1 < 0:
            raise ValueError(f"BM25 parameter k1 cannot be negative, got {k1}.")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"BM25 parameter b must be between 0.0 and 1.0, got {b}.")

        self.k1 = k1
        self.b = b
        self.index_file = Path(index_file).resolve() if index_file is not None else None
        self._loaded_mtime_ns: int | None = None

        # Primary storage: chunk_id -> Chunk
        self._chunks: dict[UUID, Chunk] = {}
        # Document length: chunk_id -> total token count in chunk content
        self._doc_lengths: dict[UUID, int] = {}
        # Forward index: chunk_id -> {term -> count} for clean removals/updates
        self._chunk_term_counts: dict[UUID, dict[str, int]] = {}
        # Inverted index: term -> {chunk_id -> count}
        self._inverted_index: dict[str, dict[UUID, int]] = {}
        # Document ID mapping: document_id -> set of chunk_ids
        self._doc_chunk_ids: dict[UUID, set[UUID]] = {}
        # Sum of all document lengths
        self._total_doc_len: int = 0

        if self.index_file is not None:
            self._ensure_loaded()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Expose tokenization as a class/instance helper."""
        return tokenize(text)

    def _reset_in_memory_state(self) -> None:
        """Clear all in-memory index structures."""
        self._chunks.clear()
        self._doc_lengths.clear()
        self._chunk_term_counts.clear()
        self._inverted_index.clear()
        self._doc_chunk_ids.clear()
        self._total_doc_len = 0

    def _ensure_loaded(self) -> None:
        """Ensure in-memory index is up to date with disk state."""
        if self.index_file is None:
            return
        if not self.index_file.exists():
            if self._loaded_mtime_ns is not None:
                self._reset_in_memory_state()
                self._loaded_mtime_ns = None
            return
        try:
            mtime_ns = self.index_file.stat().st_mtime_ns
        except OSError:
            return
        if self._loaded_mtime_ns is not None and mtime_ns <= self._loaded_mtime_ns:
            return
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Read serialized chunks from disk and populate internal indices."""
        if self.index_file is None or not self.index_file.exists():
            return

        try:
            mtime_ns = self.index_file.stat().st_mtime_ns
            content = self.index_file.read_text(encoding="utf-8").strip()
            if not content:
                self._reset_in_memory_state()
                self._loaded_mtime_ns = mtime_ns
                return

            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected JSON object in '{self.index_file}', got {type(data).__name__}"
                )

            chunks_data = data.get("chunks", [])
            chunks = [Chunk.model_validate(c) for c in chunks_data]
            self._reset_in_memory_state()
            self._index_chunks_internal(chunks)
            self._loaded_mtime_ns = mtime_ns
        except Exception as exc:
            logger.warning(
                "Failed to read lexical index from '%s': %s. Initializing empty state.",
                self.index_file,
                exc,
            )
            self._reset_in_memory_state()
            try:
                self._loaded_mtime_ns = self.index_file.stat().st_mtime_ns
            except OSError:
                pass

    def _save_to_disk(self) -> None:
        """Atomically persist lexical index state and chunks to disk."""
        if self.index_file is None:
            return

        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        serialized_chunks = [chunk.model_dump(mode="json") for chunk in self._chunks.values()]
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "chunks": serialized_chunks,
        }
        json_text = json.dumps(payload, indent=2)

        temp_file = NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.index_file.parent,
            delete=False,
        )
        try:
            temp_file.write(json_text)
            temp_file.flush()
            temp_file.close()
            os.replace(temp_file.name, self.index_file)
            self._loaded_mtime_ns = self.index_file.stat().st_mtime_ns
        except Exception:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass
            raise

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        self._ensure_loaded()
        return len(self._chunks)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    async def get(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve an indexed chunk by its ID."""
        self._ensure_loaded()
        return self._chunks.get(chunk_id)

    async def has_document(self, document_id: UUID) -> bool:
        """Check whether chunks for a document ID are present in the lexical index."""
        self._ensure_loaded()
        return bool(self._doc_chunk_ids.get(document_id))

    async def get_by_document_id(self, document_id: UUID) -> list[Chunk]:
        """Retrieve all indexed chunks associated with a document ID."""
        self._ensure_loaded()
        matching_chunk_ids = list(self._doc_chunk_ids.get(document_id, set()))
        return [self._chunks[cid] for cid in matching_chunk_ids if cid in self._chunks]

    async def clear(self) -> None:
        """Clear all indexed data from the lexical index."""
        self._reset_in_memory_state()
        self._save_to_disk()

    def _remove_chunk(self, chunk_id: UUID) -> None:
        """Internal helper to atomically purge a chunk from all internal indices."""
        if chunk_id not in self._chunks:
            return

        chunk = self._chunks[chunk_id]
        if chunk.document_id in self._doc_chunk_ids:
            self._doc_chunk_ids[chunk.document_id].discard(chunk_id)
            if not self._doc_chunk_ids[chunk.document_id]:
                del self._doc_chunk_ids[chunk.document_id]

        term_counts = self._chunk_term_counts.pop(chunk_id, {})
        for term in term_counts:
            postings = self._inverted_index.get(term)
            if postings is not None:
                postings.pop(chunk_id, None)
                if not postings:
                    del self._inverted_index[term]

        doc_len = self._doc_lengths.pop(chunk_id, 0)
        self._total_doc_len -= doc_len
        self._chunks.pop(chunk_id, None)

    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        """Delete a single chunk by its ID. Returns True if found and deleted."""
        self._ensure_loaded()
        if chunk_id not in self._chunks:
            return False
        self._remove_chunk(chunk_id)
        self._save_to_disk()
        return True

    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete all indexed chunks associated with a document ID."""
        self._ensure_loaded()
        matching_chunk_ids = list(self._doc_chunk_ids.get(document_id, set()))
        if not matching_chunk_ids:
            return
        for cid in matching_chunk_ids:
            self._remove_chunk(cid)
        self._save_to_disk()

    def _index_chunks_internal(self, chunks: Sequence[Chunk]) -> None:
        """Internal helper to index chunks into memory structures."""
        for chunk in chunks:
            if chunk.id in self._chunks:
                self._remove_chunk(chunk.id)

            tokens = tokenize(chunk.content)
            doc_len = len(tokens)
            term_counts = dict(Counter(tokens))

            self._chunks[chunk.id] = chunk
            self._doc_lengths[chunk.id] = doc_len
            self._total_doc_len += doc_len
            self._chunk_term_counts[chunk.id] = term_counts

            if chunk.document_id not in self._doc_chunk_ids:
                self._doc_chunk_ids[chunk.document_id] = set()
            self._doc_chunk_ids[chunk.document_id].add(chunk.id)

            for term, count in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][chunk.id] = count

    async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index or update chunks in the lexical search index.

        If a chunk with an existing chunk ID is indexed, its prior term contributions,
        postings, and length metadata are purged before re-indexing to prevent stale postings.

        Args:
            chunks: Sequence of domain Chunk objects to index.
        """
        if not chunks:
            return
        self._ensure_loaded()
        self._index_chunks_internal(chunks)
        self._save_to_disk()

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Evaluate whether a chunk satisfies the provided filter dictionary."""
        for key, expected_val in filters.items():
            if key == "document_id":
                if str(chunk.document_id) != str(expected_val):
                    return False
            elif key in chunk.metadata.extra:
                if chunk.metadata.extra[key] != expected_val:
                    return False
            elif hasattr(chunk.metadata, key):
                if getattr(chunk.metadata, key) != expected_val:
                    return False
            elif hasattr(chunk, key):
                if getattr(chunk, key) != expected_val:
                    return False
            else:
                return False
        return True

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute lexical BM25 search for a query with optional filters.

        Args:
            query: Query string to match against indexed chunks.
            top_k: Maximum number of ranked results to return. Must be > 0.
            filters: Optional key-value criteria to filter candidate chunks.

        Returns:
            List of RetrievedChunk domain models ordered descending by BM25 score,
            with deterministic tie-breaking by chunk ID.

        Raises:
            ValueError: If top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not query or not query.strip():
            return []

        self._ensure_loaded()

        if not self._chunks or self._total_doc_len == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Deduplicate query terms while preserving first appearance order
        unique_terms = list(dict.fromkeys(query_tokens))

        # Collect candidate chunk IDs containing at least one query term
        candidate_ids: set[UUID] = set()
        active_terms: list[str] = []
        for term in unique_terms:
            if term in self._inverted_index:
                candidate_ids.update(self._inverted_index[term].keys())
                active_terms.append(term)

        if not candidate_ids:
            return []

        n = len(self._chunks)
        avgdl = self._total_doc_len / n
        if avgdl <= 0:
            return []

        # Precompute non-negative Robertson-Spärck Jones IDF for active terms
        idf: dict[str, float] = {}
        for term in active_terms:
            df = len(self._inverted_index[term])
            idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        scored_candidates: list[tuple[float, Chunk]] = []
        for cid in candidate_ids:
            chunk = self._chunks[cid]
            if filters and not self._matches_filters(chunk, filters):
                continue

            doc_len = self._doc_lengths.get(cid, 0)
            score = 0.0
            term_counts = self._chunk_term_counts.get(cid, {})

            for term in active_terms:
                tf = term_counts.get(term, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                    score += idf[term] * (numerator / denominator)

            if score > 0.0:
                scored_candidates.append((score, chunk))

        if not scored_candidates:
            return []

        # Sort descending by BM25 score, tie-break deterministically by string chunk ID
        scored_candidates.sort(key=lambda item: (-item[0], str(item[1].id)))

        results: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retrieval_type="lexical",
                    rank=rank,
                    dense_score=None,
                    lexical_score=score,
                )
            )

        return results
