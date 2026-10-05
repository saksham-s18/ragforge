import re
from collections.abc import Sequence

from ragforge.domain.models import RetrievedChunk
from ragforge.ports.reranker import BaseReranker


class DeterministicReranker(BaseReranker):
    """Deterministic lexical reranker computing token overlap, frequency, and exact phrase matching.

    Designed for 100% reproducible execution without external ML models, API keys,
    or GPU dependencies. Provides second-stage relevance re-scoring to reward lexical
    precision and exact keyword matches while preserving original vector search scores
    and provenance.
    """

    def __init__(
        self,
        query_term_weight: float = 0.6,
        term_frequency_weight: float = 0.2,
        phrase_match_weight: float = 0.2,
    ) -> None:
        """Initialize the deterministic reranker with scoring weights.

        Args:
            query_term_weight: Weight assigned to the fraction of query terms present (0.0 - 1.0).
            term_frequency_weight: Weight assigned to capped query term repetition in the chunk.
            phrase_match_weight: Weight assigned to exact query substring matching.

        Raises:
            ValueError: If weights are negative or do not sum to positive value.
        """
        if query_term_weight < 0 or term_frequency_weight < 0 or phrase_match_weight < 0:
            raise ValueError("Reranker scoring weights must be non-negative.")
        total_weight = query_term_weight + term_frequency_weight + phrase_match_weight
        if total_weight <= 0:
            raise ValueError("Total reranker scoring weight must be greater than 0.")

        self.query_term_weight = query_term_weight
        self.term_frequency_weight = term_frequency_weight
        self.phrase_match_weight = phrase_match_weight
        self._total_weight = total_weight

    def _tokenize(self, text: str) -> list[str]:
        """Normalize and tokenize text into lowercase alphanumeric tokens."""
        return [token for token in re.findall(r"\w+", text.lower()) if token]

    def _score_chunk(self, query_tokens: list[str], raw_query: str, content: str) -> float:
        """Calculate normalized lexical relevance score between query and chunk content."""
        if not query_tokens:
            return 0.0

        chunk_tokens = self._tokenize(content)
        if not chunk_tokens:
            return 0.0

        unique_query = sorted(set(query_tokens))

        # 1. Term presence / overlap ratio
        matched_terms = sum(1 for term in unique_query if term in chunk_tokens)
        overlap_score = matched_terms / len(unique_query)

        # 2. Term frequency (capped per term to avoid keyword-stuffing skew)
        tf_sum = sum(min(chunk_tokens.count(term), 3) for term in unique_query)
        max_tf = len(unique_query) * 3
        tf_score = tf_sum / max_tf if max_tf > 0 else 0.0

        # 3. Exact phrase match bonus
        clean_raw_query = raw_query.strip().lower()
        phrase_score = 1.0 if (clean_raw_query and clean_raw_query in content.lower()) else 0.0

        raw_score = (
            (overlap_score * self.query_term_weight)
            + (tf_score * self.term_frequency_weight)
            + (phrase_score * self.phrase_match_weight)
        )
        normalized = raw_score / self._total_weight
        return round(normalized, 4)

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks according to lexical relevance to the query.

        Args:
            query: User query string.
            chunks: Candidate RetrievedChunks from initial retrieval.
            top_k: Optional maximum number of top reranked chunks to return.

        Returns:
            List of reranked RetrievedChunks sorted by descending relevance with updated
            ranks (1 to N) and rerank_score populated. Original vector retrieval scores
            and chunk provenance are strictly preserved.

        Raises:
            ValueError: If top_k is non-None and <= 0.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not chunks:
            return []

        cleaned_query = query.strip()
        query_tokens = self._tokenize(cleaned_query)

        scored_candidates: list[tuple[RetrievedChunk, float]] = []
        for item in chunks:
            if not query_tokens:
                rerank_score = 0.0
            else:
                rerank_score = self._score_chunk(
                    query_tokens=query_tokens,
                    raw_query=cleaned_query,
                    content=item.chunk.content,
                )
            scored_candidates.append((item, rerank_score))

        # Deterministic sort order:
        # 1. Primary: descending rerank score
        # 2. Secondary: descending original vector retrieval score
        # 3. Tertiary: ascending initial rank
        # 4. Quaternary: ascending chunk ID string (strict tie-breaker)
        scored_candidates.sort(
            key=lambda entry: (
                -entry[1],
                -entry[0].score,
                entry[0].rank,
                str(entry[0].chunk.id),
            )
        )

        selected = scored_candidates[:top_k] if top_k is not None else scored_candidates

        return [
            RetrievedChunk(
                chunk=item.chunk,
                score=item.score,  # original vector score strictly preserved
                retrieval_type=item.retrieval_type,
                rank=new_rank,
                rerank_score=rerank_score,
            )
            for new_rank, (item, rerank_score) in enumerate(selected, start=1)
        ]
