import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding

from ragforge.domain.exceptions import EmbeddingError, EmbeddingModelNotFoundError
from ragforge.ports.embeddings import BaseEmbeddingProvider

KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "snowflake/snowflake-arctic-embed-xs": 384,
    "snowflake/snowflake-arctic-embed-s": 384,
    "snowflake/snowflake-arctic-embed-m": 768,
    "snowflake/snowflake-arctic-embed-l": 1024,
}


class FastEmbedProvider(BaseEmbeddingProvider):
    """Production semantic embedding adapter using FastEmbed and ONNX Runtime.

    Generates dense vector embeddings using local, quantized ONNX transformer models
    such as 'BAAI/bge-small-en-v1.5' without requiring PyTorch, CUDA, or external API keys.
    All embedding generation is offloaded to a worker thread via asyncio.to_thread to maintain
    non-blocking asynchronous operation.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        batch_size: int = 32,
        dimension: int | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize the FastEmbed provider.

        Args:
            model_name: Name of the FastEmbed supported model.
            cache_dir: Optional custom directory to store downloaded ONNX models.
            threads: Optional number of threads for ONNX Runtime inference.
            batch_size: Batch size for embedding computation.
            dimension: Optional explicit vector dimension override.
            model: Optional injected TextEmbedding instance (primarily for mocking/testing).

        Raises:
            ValueError: If batch_size <= 0 or dimension is invalid.
            EmbeddingError: If model initialization fails.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension must be greater than 0, got {dimension}")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.threads = threads
        self.batch_size = batch_size

        if model is not None:
            self._model = model
        else:
            try:
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    threads=self.threads,
                )
            except ValueError as exc:
                raise EmbeddingModelNotFoundError(
                    f"Model '{self.model_name}' not supported or not found: {exc}"
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to initialize FastEmbed model '{self.model_name}': {exc}"
                ) from exc

        self._dimension = dimension or self._resolve_dimension()

    def _resolve_dimension(self) -> int:
        """Resolve vector dimensionality from model metadata or known defaults."""
        if self.model_name in KNOWN_MODEL_DIMENSIONS:
            return KNOWN_MODEL_DIMENSIONS[self.model_name]

        try:
            supported = TextEmbedding.list_supported_models()
            for entry in supported:
                if entry.get("model") == self.model_name and "dim" in entry:
                    return int(entry["dim"])
        except Exception:
            pass

        # Fallback to default 384
        return 384

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this provider."""
        return self._dimension

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronously execute passage embedding generation via FastEmbed."""
        try:
            embeddings_iter = self._model.passage_embed(texts, batch_size=self.batch_size)
            return [vec.tolist() for vec in embeddings_iter]
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate text embeddings: {exc}") from exc

    def _embed_query_sync(self, query: str) -> list[float]:
        """Synchronously execute query embedding generation via FastEmbed."""
        try:
            embeddings_iter = self._model.query_embed(query)
            results = list(embeddings_iter)
            if not results:
                raise EmbeddingError("FastEmbed query_embed produced no output vector.")
            return results[0].tolist()  # type: ignore[no-any-return]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate query embedding: {exc}") from exc

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a collection of text strings asynchronously.

        Args:
            texts: Sequence of strings representing document chunks / passages.

        Returns:
            List of dense float vectors with dimension matching self.dimension.
        """
        if not texts:
            return []

        text_list = list(texts)
        return await asyncio.to_thread(self._embed_texts_sync, text_list)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string asynchronously.

        Args:
            query: Query string to embed.

        Returns:
            Dense float vector with dimension matching self.dimension.
        """
        clean_query = query.strip()
        if not clean_query:
            return [0.0] * self._dimension

        return await asyncio.to_thread(self._embed_query_sync, clean_query)
