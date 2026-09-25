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


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    out, _ = capsys.readouterr()
    assert "usage:" in out.lower() or "ragforge" in out
