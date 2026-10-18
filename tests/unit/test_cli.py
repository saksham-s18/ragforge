import asyncio
from pathlib import Path

import pytest

from ragforge.cli import build_parser, main


def test_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["index", "./docs"])
    assert args.command == "index"
    assert args.path == "./docs"
    assert args.provider is None
    assert args.batch_size is None
    assert args.collection is None
    assert args.lexical_index_file is None
    assert args.in_memory is False
    assert args.force is False


def test_cli_parser_custom_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "index",
            "./custom_docs",
            "--provider",
            "fastembed",
            "--batch-size",
            "16",
            "--collection",
            "my_col",
            "--qdrant-url",
            "http://localhost:6334",
            "--state-file",
            "./state.json",
            "--lexical-index-file",
            "./lexical.json",
            "--in-memory",
            "--force",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.command == "index"
    assert args.path == "./custom_docs"
    assert args.provider == "fastembed"
    assert args.batch_size == 16
    assert args.collection == "my_col"
    assert args.qdrant_url == "http://localhost:6334"
    assert args.state_file == "./state.json"
    assert args.lexical_index_file == "./lexical.json"
    assert args.in_memory is True
    assert args.force is True
    assert args.log_level == "DEBUG"


def test_cli_missing_path_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nonexistent_directory"
    exit_code = main(["index", str(missing), "--in-memory"])
    assert exit_code == 1
    _, err = capsys.readouterr()
    assert "Document path not found" in err


def test_cli_index_directory_in_memory_and_incremental_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    doc1 = doc_dir / "sample.txt"
    doc1.write_text("CLI test document 1 content.", encoding="utf-8")
    doc2 = doc_dir / "sample2.md"
    doc2.write_text("# Markdown\nCLI test document 2 content.", encoding="utf-8")

    state_file = tmp_path / "cli_state.json"

    # First run: should index 2 documents
    exit_code1 = main(
        [
            "index",
            str(doc_dir),
            "--in-memory",
            "--state-file",
            str(state_file),
        ]
    )
    assert exit_code1 == 0
    out1, _ = capsys.readouterr()
    assert "RAGFORGE INDEXING REPORT" in out1
    assert "Indexed (new):         2" in out1

    # Second run: should skip both unchanged documents
    exit_code2 = main(
        [
            "index",
            str(doc_dir),
            "--in-memory",
            "--state-file",
            str(state_file),
        ]
    )
    assert exit_code2 == 0
    out2, _ = capsys.readouterr()
    assert "Skipped (unchanged):   2" in out2
    assert "Indexed (new):         0" in out2

    # Third run: with --force should reindex even though content is unchanged
    exit_code3 = main(
        [
            "index",
            str(doc_dir),
            "--in-memory",
            "--state-file",
            str(state_file),
            "--force",
        ]
    )
    assert exit_code3 == 0
    out3, _ = capsys.readouterr()
    assert "Skipped (unchanged):   0" in out3
    assert "Updated (changed):     2" in out3


def test_cli_index_with_hybrid_enabled_persists_lexical_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that running CLI indexing with hybrid_enabled=True persists lexical index to disk."""
    from ragforge.adapters.lexical import BM25LexicalIndex
    from ragforge.core.config import get_settings

    doc_dir = tmp_path / "hybrid_docs"
    doc_dir.mkdir()
    doc1 = doc_dir / "hybrid_sample.txt"
    doc1.write_text(
        "Kubernetes container orchestration with lexical retrieval terms.",
        encoding="utf-8",
    )

    state_file = tmp_path / "cli_hybrid_state.json"
    lexical_file = tmp_path / "cli_lexical_index.json"

    monkeypatch.setenv("RAGFORGE_HYBRID_ENABLED", "true")
    get_settings.cache_clear()

    try:
        exit_code = main(
            [
                "index",
                str(doc_dir),
                "--in-memory",
                "--state-file",
                str(state_file),
                "--lexical-index-file",
                str(lexical_file),
            ]
        )
        assert exit_code == 0
        out, _ = capsys.readouterr()
        assert "Indexed (new):         1" in out
        assert lexical_file.exists()

        # Independent API process loading from persisted file has the indexed chunks
        api_lexical = BM25LexicalIndex(index_file=lexical_file)
        assert api_lexical.count() >= 1
        results = asyncio.run(api_lexical.search("orchestration", top_k=5))
        assert len(results) >= 1
        assert "Kubernetes" in results[0].chunk.content
    finally:
        get_settings.cache_clear()


def test_cli_api_lifecycle_live_reload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify running API instance detects when separate CLI process updates lexical index."""
    from ragforge.adapters.lexical import BM25LexicalIndex
    from ragforge.core.config import get_settings

    doc_dir = tmp_path / "live_docs"
    doc_dir.mkdir()
    doc1 = doc_dir / "service_mesh.txt"
    doc1.write_text("Istio service mesh sidecar proxy routing.", encoding="utf-8")

    state_file = tmp_path / "live_state.json"
    lexical_file = tmp_path / "live_lexical.json"

    # API process starts first with empty lexical index file
    api_lexical = BM25LexicalIndex(index_file=lexical_file)
    assert api_lexical.count() == 0

    monkeypatch.setenv("RAGFORGE_HYBRID_ENABLED", "true")
    get_settings.cache_clear()

    try:
        # Separate CLI process performs indexing
        exit_code = main(
            [
                "index",
                str(doc_dir),
                "--in-memory",
                "--state-file",
                str(state_file),
                "--lexical-index-file",
                str(lexical_file),
            ]
        )
        assert exit_code == 0

        # Running API process queries again without restart; it detects updated mtime
        results = asyncio.run(api_lexical.search("sidecar", top_k=5))
        assert len(results) == 1
        assert "Istio" in results[0].chunk.content
        assert api_lexical.count() == 1
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_cli_index_in_memory_uses_non_persistent_lexical_when_unspecified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify execute_indexing with --in-memory uses index_file=None
    when lexical file is not specified.
    """
    from unittest.mock import AsyncMock, patch

    from ragforge.cli import execute_indexing

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "sample.txt").write_text("sample content", encoding="utf-8")

    monkeypatch.setenv("RAGFORGE_HYBRID_ENABLED", "true")
    from ragforge.core.config import get_settings

    get_settings.cache_clear()

    try:
        parser = build_parser()
        args = parser.parse_args(["index", str(doc_dir), "--in-memory"])
        settings = get_settings()

        with (
            patch("ragforge.cli.IndexingService") as mock_idx_cls,
            patch("ragforge.cli.FastEmbedProvider"),
            patch("ragforge.cli.DeterministicEmbeddingProvider"),
        ):
            mock_service = mock_idx_cls.return_value
            mock_service.index_path = AsyncMock()
            await execute_indexing(args, settings)

            _, kwargs = mock_idx_cls.call_args
            lexical = kwargs["lexical_index"]
            assert lexical is not None
            assert lexical.index_file is None
    finally:
        get_settings.cache_clear()


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    out, _ = capsys.readouterr()
    assert "usage:" in out.lower() or "ragforge" in out
