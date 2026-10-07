from collections.abc import Sequence
from typing import Any
from uuid import UUID

from ragforge.domain.models import Chunk, RetrievedChunk
from ragforge.ports.fusion import BaseScoreFusion


class ReciprocalRankFusion(BaseScoreFusion):
    """Reciprocal Rank Fusion (RRF) algorithm for fusing multi-channel retrieval results.

    Combines ranked candidate lists from separate retrieval mechanisms (such as dense
    vector search and lexical BM25 search) using reciprocal rank scoring:

        RRF(d) = dense_weight / (k + dense_rank(d)) + lexical_weight / (k + lexical_rank(d))

    where ranks are 1-based indices within each retrieved channel. Results appearing in
    only one channel receive only that channel's score contribution.
    """

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        """Initialize the ReciprocalRankFusion adapter.

        Args:
            k: Ranking smoothing constant (must be > 0, default 60).
            dense_weight: Multiplicative importance weight for dense channel (>= 0.0).
            lexical_weight: Multiplicative importance weight for lexical channel (>= 0.0).

        Raises:
            ValueError: If k <= 0 or weights are negative.
        """
        if k <= 0:
            raise ValueError(f"RRF constant k must be a positive integer, got {k}.")
        if dense_weight < 0:
            raise ValueError(f"dense_weight cannot be negative, got {dense_weight}.")
        if lexical_weight < 0:
            raise ValueError(f"lexical_weight cannot be negative, got {lexical_weight}.")

        self.k = k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def fuse(
        self,
        dense_results: Sequence[RetrievedChunk],
        lexical_results: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse and rank candidate results from dense and lexical retrieval runs.

        Args:
            dense_results: Ranked chunks retrieved via dense vector similarity.
            lexical_results: Ranked chunks retrieved via lexical BM25 search.
            top_k: Optional maximum number of fused results to return. If None,
                all fused candidates are returned.

        Returns:
            Ranked list of fused RetrievedChunk models ordered by descending fusion score.

        Raises:
            ValueError: If top_k is specified and <= 0.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not dense_results and not lexical_results:
            return []

        # Map chunk_id to candidate metadata
        candidates: dict[UUID, dict[str, Any]] = {}

        # 1. Ingest dense results preserving channel scores and 1-based ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item.chunk.id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": item.chunk,
                    "dense_score": (
                        item.dense_score if item.dense_score is not None else item.score
                    ),
                    "lexical_score": item.lexical_score,
                    "dense_rank": rank,
                    "lexical_rank": None,
                }
            else:
                # Keep first/best rank if duplicates exist in dense list
                if candidates[cid]["dense_rank"] is None:
                    candidates[cid]["dense_rank"] = rank
                    if candidates[cid]["dense_score"] is None:
                        candidates[cid]["dense_score"] = (
                            item.dense_score if item.dense_score is not None else item.score
                        )

        # 2. Ingest lexical results preserving channel scores and 1-based ranks
        for rank, item in enumerate(lexical_results, start=1):
            cid = item.chunk.id
            if cid not in candidates:
                candidates[cid] = {
                    "chunk": item.chunk,
                    "dense_score": item.dense_score,
                    "lexical_score": (
                        item.lexical_score if item.lexical_score is not None else item.score
                    ),
                    "dense_rank": None,
                    "lexical_rank": rank,
                }
            else:
                if candidates[cid]["lexical_rank"] is None:
                    candidates[cid]["lexical_rank"] = rank
                    if candidates[cid]["lexical_score"] is None:
                        candidates[cid]["lexical_score"] = (
                            item.lexical_score if item.lexical_score is not None else item.score
                        )

        # 3. Compute fused RRF scores
        scored_candidates: list[tuple[float, Chunk, float | None, float | None]] = []
        for candidate in candidates.values():
            rrf_score = 0.0
            if candidate["dense_rank"] is not None:
                rrf_score += self.dense_weight / (self.k + candidate["dense_rank"])
            if candidate["lexical_rank"] is not None:
                rrf_score += self.lexical_weight / (self.k + candidate["lexical_rank"])

            scored_candidates.append(
                (
                    rrf_score,
                    candidate["chunk"],
                    candidate["dense_score"],
                    candidate["lexical_score"],
                )
            )

        # 4. Sort deterministically: descending RRF score, then ascending chunk ID string
        scored_candidates.sort(key=lambda item: (-item[0], str(item[1].id)))

        # 5. Apply top_k selection
        selected = scored_candidates[:top_k] if top_k is not None else scored_candidates

        # 6. Construct unified RetrievedChunk domain models
        results: list[RetrievedChunk] = []
        for new_rank, (fused_score, chunk, dense_score, lexical_score) in enumerate(
            selected, start=1
        ):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=fused_score,
                    retrieval_type="hybrid",
                    rank=new_rank,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                )
            )

        return results
