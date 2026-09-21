import hashlib
import math
import re
from collections.abc import Sequence

from ragforge.ports.embeddings import BaseEmbeddingProvider


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local embedding adapter for development, testing, and CI pipelines.

    Generates reproducible dense vector representations using token and n-gram feature hashing
    with L2-normalization. Requires no external model weights, no network connectivity, no GPU,
    and no API credentials.

    NOTE:
        This provider is intended strictly for local development, reproducible integration testing,
        and CI environments to decouple retrieval logic from external infrastructure. In later
        stages, production embedding providers (e.g. OpenAI, Cohere, HuggingFace) will implement
        the same BaseEmbeddingProvider port.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be greater than 0, got {dimension}")
        self.dimension = dimension

    def _embed_single(self, text: str) -> list[float]:
        """Compute a deterministic, L2-normalized dense vector for a single text."""
        vec = [0.0] * self.dimension
        clean_text = text.strip()
        if not clean_text:
            return vec

        tokens = re.findall(r"\w+", clean_text.lower())
        if not tokens:
            return vec

        for token in tokens:
            # Word-level feature hashing
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

            # Character 3-gram feature hashing for subword/morphological similarity
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngram = token[i : i + 3]
                    ngram_digest = hashlib.md5(ngram.encode("utf-8")).digest()
                    ngram_idx = int.from_bytes(ngram_digest[:4], "big") % self.dimension
                    ngram_sign = 1.0 if (ngram_digest[4] % 2 == 0) else -1.0
                    vec[ngram_idx] += 0.5 * ngram_sign

        # L2-normalization
        norm_sq = sum(x * x for x in vec)
        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            vec = [x / norm for x in vec]

        return vec

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of text strings into deterministic dense vectors."""
        return [self._embed_single(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a deterministic dense vector."""
        return self._embed_single(query)
