"""Matching and relevance evaluation engine for RAG retrieval results.

Determines whether retrieved chunks satisfy evaluation case criteria based on:
- Granular chunk identity (UUID, content hash)
- Structural section headers
- Substring content and factual snippet matches
- Source document provenance (document ID, file path, filename, title)
"""

from collections.abc import Sequence
from pathlib import Path

from ragforge.domain.models import CaseEvaluationResult, EvaluationCase, RetrievedChunk
from ragforge.evaluation.metrics import (
    precision_at_k_binary,
    reciprocal_rank_binary,
)


def is_source_match(retrieved_chunk: RetrievedChunk, expected_source: str) -> bool:
    """Determine if a retrieved chunk originates from an expected source or document.

    Checks against:
    - Document UUID / identifier string
    - Source file path (exact, posix suffix, or filename)
    - Document title metadata

    Args:
        retrieved_chunk: Retrieved chunk domain model with provenance metadata.
        expected_source: Ground-truth source identifier, filename, or document title.

    Returns:
        True if the chunk matches the expected source, False otherwise.
    """
    if not expected_source or not expected_source.strip():
        return False
    source = expected_source.strip()

    # 1. Document ID match
    if str(retrieved_chunk.chunk.document_id).lower() == source.lower():
        return True

    # 2. File path match
    file_path = str(retrieved_chunk.chunk.metadata.extra.get("file_path", "")).strip()
    if file_path:
        norm_fp = file_path.replace("\\", "/").lower()
        norm_src = source.replace("\\", "/").lower()
        if (
            norm_fp == norm_src
            or norm_fp.endswith("/" + norm_src)
            or norm_fp.endswith(norm_src)
            or Path(file_path).name.lower() == Path(source).name.lower()
        ):
            return True

    # 3. Document title match
    doc_title = str(retrieved_chunk.chunk.metadata.extra.get("document_title", "")).strip()
    if doc_title and doc_title.lower() == source.lower():
        return True

    return False


def is_chunk_match(retrieved_chunk: RetrievedChunk, expected_chunk: str) -> bool:
    """Determine if a retrieved chunk matches an expected chunk target.

    Checks against:
    - Unique chunk UUID
    - Chunk content SHA-256 hash
    - Chunk section header
    - Text substring / factual snippet match (case-insensitive, normalized whitespace)

    Args:
        retrieved_chunk: Retrieved chunk candidate.
        expected_chunk: Expected chunk UUID, hash, section header, or text snippet.

    Returns:
        True if the chunk matches the target criteria, False otherwise.
    """
    if not expected_chunk or not expected_chunk.strip():
        return False
    target = expected_chunk.strip()

    # 1. Chunk ID match
    if str(retrieved_chunk.chunk.id).lower() == target.lower():
        return True

    # 2. Content hash match
    if retrieved_chunk.chunk.content_hash.lower() == target.lower():
        return True

    # 3. Section header match
    section = retrieved_chunk.chunk.metadata.section_header
    if section and section.strip().lower() == target.lower():
        return True

    # 4. Normalized text snippet / substring match
    norm_content = " ".join(retrieved_chunk.chunk.content.split()).lower()
    norm_target = " ".join(target.split()).lower()
    if norm_target in norm_content:
        return True

    return False


def is_chunk_relevant(retrieved_chunk: RetrievedChunk, case: EvaluationCase) -> bool:
    """Evaluate whether a retrieved chunk is relevant to the given evaluation case.

    Rules:
    - If `case.expected_chunks` is populated, the chunk must match at least one expected chunk.
      If `case.expected_sources` is also specified, it must additionally match an expected source.
    - If `case.expected_chunks` is empty but `case.expected_sources` is populated, the chunk
      is relevant if it originates from any of the expected sources.
    - If neither is provided, relevance cannot be established (returns False).

    Args:
        retrieved_chunk: Candidate retrieved chunk.
        case: EvaluationCase domain model defining expected targets.

    Returns:
        True if chunk is evaluated as relevant, False otherwise.
    """
    if case.expected_chunks:
        chunk_matched = any(is_chunk_match(retrieved_chunk, c) for c in case.expected_chunks)
        if not chunk_matched:
            return False
        if case.expected_sources:
            return any(is_source_match(retrieved_chunk, s) for s in case.expected_sources)
        return True

    if case.expected_sources:
        return any(is_source_match(retrieved_chunk, s) for s in case.expected_sources)

    return False


def evaluate_case_relevance(
    retrieved_chunks: Sequence[RetrievedChunk],
    case: EvaluationCase,
    k_values: Sequence[int],
) -> CaseEvaluationResult:
    """Evaluate retrieval results for a single query against its evaluation ground truth.

    Computes:
    - Per-rank binary relevance
    - Reciprocal Rank (RR)
    - Precision@K for each cutoff in k_values
    - Recall@K for each cutoff in k_values

    Args:
        retrieved_chunks: Ordered list of chunks retrieved by RetrievalService.
        case: EvaluationCase defining questions and ground-truth targets.
        k_values: Collection of K cutoff values.

    Returns:
        CaseEvaluationResult with metrics and attribution diagnostics.
    """
    sorted_k = sorted(k_values)
    binary_relevance: list[bool] = [is_chunk_relevant(c, case) for c in retrieved_chunks]

    first_rank: int | None = None
    for idx, is_rel in enumerate(binary_relevance, start=1):
        if is_rel:
            first_rank = idx
            break

    rr = reciprocal_rank_binary(binary_relevance)

    # Determine ground-truth target denominator for Recall
    if case.expected_chunks:
        expected_targets = case.expected_chunks
        target_type = "chunk"
    elif case.expected_sources:
        expected_targets = case.expected_sources
        target_type = "source"
    else:
        expected_targets = []
        target_type = "none"

    total_expected = len(expected_targets)

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}

    for k in sorted_k:
        precision_at_k[k] = precision_at_k_binary(binary_relevance, k=k)

        if total_expected <= 0 or not retrieved_chunks:
            recall_at_k[k] = 0.0
            continue

        top_k_chunks = retrieved_chunks[:k]
        if target_type == "chunk":
            # Count unique expected chunk targets hit in top K
            hits = 0
            for t in expected_targets:
                hit = any(
                    is_chunk_match(c, t)
                    and (
                        not case.expected_sources
                        or any(is_source_match(c, s) for s in case.expected_sources)
                    )
                    for c in top_k_chunks
                )
                if hit:
                    hits += 1
            recall_at_k[k] = min(1.0, hits / float(total_expected))
        elif target_type == "source":
            # Count unique expected sources hit in top K
            hits = 0
            for s in expected_targets:
                hit = any(is_source_match(c, s) for c in top_k_chunks)
                if hit:
                    hits += 1
            recall_at_k[k] = min(1.0, hits / float(total_expected))
        else:
            recall_at_k[k] = 0.0

    retrieved_chunk_ids = [str(c.chunk.id) for c in retrieved_chunks]
    relevant_chunk_ids = [
        str(c.chunk.id) for c, rel in zip(retrieved_chunks, binary_relevance, strict=True) if rel
    ]

    return CaseEvaluationResult(
        case_id=case.id,
        question=case.question,
        retrieved_count=len(retrieved_chunks),
        relevant_retrieved_count=len(relevant_chunk_ids),
        total_expected_relevant=total_expected,
        first_relevant_rank=first_rank,
        reciprocal_rank=rr,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )
