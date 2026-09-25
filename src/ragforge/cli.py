import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from ragforge.adapters.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from ragforge.adapters.state import JsonFileIndexStateStore
from ragforge.adapters.vector_stores import (
    InMemoryVectorStore,
    QdrantVectorStore,
)
from ragforge.core.config import Settings, get_settings
from ragforge.core.logging import setup_logging
from ragforge.domain.exceptions import DocumentNotFoundError, RAGForgeError
from ragforge.domain.models import IndexingResult
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.vector_store import BaseVectorStore
from ragforge.services.indexing import IndexingService

logger = logging.getLogger("ragforge.cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for RAGForge CLI."""
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="RAGForge CLI — Production-oriented Agentic RAG Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'index' subcommand
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents from a file or directory into the vector store.",
    )
    index_parser.add_argument(
        "path",
        type=str,
        help="Path to a document file or directory containing documents to index.",
    )
    index_parser.add_argument(
        "--provider",
        type=str,
        choices=["deterministic", "fastembed"],
        default=None,
        help="Embedding provider identifier (default: from configuration or deterministic).",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for chunk embedding and vector upsert (default: 32).",
    )
    index_parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Target Qdrant collection name (default: ragforge_chunks).",
    )
    index_parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server endpoint URL (default: http://localhost:6333).",
    )
    index_parser.add_argument(
        "--state-file",
        type=str,
        help=(
            "Path to JSON file tracking document indexing state "
            "(default: .ragforge/index_state.json)."
        ),
    )
    index_parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an in-memory vector store instead of Qdrant (useful for offline tests).",
    )
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex all discovered documents, ignoring saved index state.",
    )
    index_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Application logging verbosity level.",
    )

    return parser


def print_indexing_report(path: str, result: IndexingResult) -> None:
    """Format and print a user-friendly report of the indexing run to stdout."""
    status_label = "SUCCESS" if result.is_success else "PARTIAL_SUCCESS"
    if result.total_processed == 0 and result.failed_documents > 0:
        status_label = "FAILED"

    print("\n" + "=" * 54)
    print(" RAGFORGE INDEXING REPORT")
    print("=" * 54)
    print(f" Target path:           {path}")
    print(f" Status:                {status_label}")
    print(f" Discovered documents:  {result.discovered_documents}")
    print(f" Indexed (new):         {result.indexed_documents}")
    print(f" Updated (changed):     {result.updated_documents}")
    print(f" Skipped (unchanged):   {result.skipped_documents}")
    print(f" Failed:                {result.failed_documents}")
    print(f" Chunks created:        {result.chunks_created}")
    print(f" Vectors upserted:      {result.vectors_upserted}")
    print(f" Duration:              {result.duration_seconds:.4f}s")
    print("=" * 54)

    if result.errors:
        print("\nFailures:")
        for err in result.errors:
            print(f" - {err}")
    print()


async def execute_indexing(args: argparse.Namespace, settings: Settings) -> IndexingResult:
    """Set up dependencies and execute the indexing pipeline."""
    batch_size = args.batch_size or settings.indexing_batch_size
    provider_name = args.provider or settings.embedding_provider
    collection_name = args.collection or settings.qdrant_collection
    qdrant_url = args.qdrant_url or settings.qdrant_url
    state_file = args.state_file or settings.index_state_file

    # Initialize Embedding Provider
    provider: BaseEmbeddingProvider
    dimension: int
    if provider_name == "fastembed":
        fastembed_provider = FastEmbedProvider(
            model_name=settings.embedding_model,
            batch_size=batch_size,
            cache_dir=settings.embedding_cache_dir,
        )
        provider = fastembed_provider
        dimension = fastembed_provider.dimension
    else:
        dimension = settings.qdrant_vector_dimension
        provider = DeterministicEmbeddingProvider(dimension=dimension)

    # Initialize Vector Store
    vector_store: BaseVectorStore
    qdrant_client: AsyncQdrantClient | None = None
    try:
        if args.in_memory:
            vector_store = InMemoryVectorStore(dimension=dimension)
        else:
            qdrant_client = AsyncQdrantClient(
                url=qdrant_url,
                api_key=settings.qdrant_api_key,
                check_compatibility=False,
            )
            store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                dimension=dimension,
            )
            await store.init_collection()
            vector_store = store

        # Initialize State Store
        state_store = JsonFileIndexStateStore(file_path=Path(state_file))

        # Instantiate Indexing Service
        service = IndexingService(
            embedding_provider=provider,
            vector_store=vector_store,
            state_store=state_store,
            batch_size=batch_size,
        )

        return await service.index_path(args.path, force=args.force)

    finally:
        if qdrant_client is not None:
            await qdrant_client.close()


def run_index_command(args: argparse.Namespace) -> int:
    """Handler for 'index' subcommand."""
    settings = get_settings()
    if args.log_level:
        settings.log_level = args.log_level
    setup_logging(settings)

    try:
        result = asyncio.run(execute_indexing(args, settings))
        print_indexing_report(args.path, result)
        return 0 if result.is_success else 0  # Completed run returns 0 even with partial doc errors
    except DocumentNotFoundError as exc:
        print(f"\n[ERROR] Document path not found: {exc}", file=sys.stderr)
        return 1
    except RAGForgeError as exc:
        print(f"\n[ERROR] RAGForge error during indexing: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[FATAL] Unexpected error during indexing: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return run_index_command(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
