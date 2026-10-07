import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from ragforge.domain.models import Chunk, RetrievedChunk
from ragforge.ports.lexical import BaseLexicalIndex


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

    Maintains an in-memory inverted index of tokenized terms to chunk postings with term
    frequencies, document lengths, and document frequencies. Implements standard Okapi
    BM25 with non-negative Robertson-Spärck Jones IDF and length normalization.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize the BM25 lexical index.

        Args:
            k1: Term frequency saturation parameter (k1 >= 0, default 1.5).
            b: Document length normalization parameter (0.0 <= b <= 1.0, default 0.75).

        Raises:
            ValueError: If k1 < 0 or b is outside [0.0, 1.0].
        """
        if k1 < 0:
            raise ValueError(f"BM25 parameter k1 cannot be negative, got {k1}.")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"BM25 parameter b must be between 0.0 and 1.0, got {b}.")

        self.k1 = k1
        self.b = b

        # Primary storage: chunk_id -> Chunk
        self._chunks: dict[UUID, Chunk] = {}
        # Document length: chunk_id -> total token count in chunk content
        self._doc_lengths: dict[UUID, int] = {}
        # Forward index: chunk_id -> {term -> count} for clean removals/updates
        self._chunk_term_counts: dict[UUID, dict[str, int]] = {}
        # Inverted index: term -> {chunk_id -> count}
        self._inverted_index: dict[str, dict[UUID, int]] = {}
        # Sum of all document lengths
        self._total_doc_len: int = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Expose tokenization as a class/instance helper."""
        return tokenize(text)

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return len(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    async def get(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve an indexed chunk by its ID."""
        return self._chunks.get(chunk_id)

    async def clear(self) -> None:
        """Clear all indexed data from the lexical index."""
        self._chunks.clear()
        self._doc_lengths.clear()
        self._chunk_term_counts.clear()
        self._inverted_index.clear()
        self._total_doc_len = 0

    def _remove_chunk(self, chunk_id: UUID) -> None:
        """Internal helper to atomically purge a chunk from all internal indices."""
        if chunk_id not in self._chunks:
            return

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
        if chunk_id not in self._chunks:
            return False
        self._remove_chunk(chunk_id)
        return True

    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete all indexed chunks associated with a document ID."""
        matching_chunk_ids = [
            cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id
        ]
        for cid in matching_chunk_ids:
            self._remove_chunk(cid)

    async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index or update chunks in the lexical search index.

        If a chunk with an existing chunk ID is indexed, its prior term contributions,
        postings, and length metadata are purged before re-indexing to prevent stale postings.

        Args:
            chunks: Sequence of domain Chunk objects to index.
        """
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

            for term, count in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][chunk.id] = count

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
