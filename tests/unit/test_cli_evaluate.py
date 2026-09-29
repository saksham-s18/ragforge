"""Unit tests for RAGForge CLI evaluate retrieval command."""

import json
from pathlib import Path

import pytest

from ragforge.cli import build_parser, main


def test_cli_evaluate_parser_defaults() -> None:
    """Verify default parser settings for 'evaluate retrieval' command."""
    parser = build_parser()
    args = parser.parse_args(["evaluate", "retrieval", "eval_data.json"])

    assert args.command == "evaluate"
    assert args.eval_target == "retrieval"
    assert args.dataset == "eval_data.json"
    assert args.index_path is None
    assert args.provider is None
    assert args.collection is None
    assert args.qdrant_url is None
    assert args.in_memory is False
    assert args.k_values == "1,3,5"
    assert args.top_k is None
    assert args.json is False


def test_cli_evaluate_parser_custom_options() -> None:
    """Verify custom option parsing for 'evaluate retrieval' command."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "retrieval",
            "eval_data.json",
            "--index-path",
            "./test_docs",
            "--provider",
            "fastembed",
            "--collection",
            "eval_col",
            "--qdrant-url",
            "http://localhost:6334",
            "--in-memory",
            "-k",
            "1,5,10",
            "--top-k",
            "10",
            "--json",
            "--log-level",
            "DEBUG",
        ]
    )

    assert args.command == "evaluate"
    assert args.eval_target == "retrieval"
    assert args.dataset == "eval_data.json"
    assert args.index_path == "./test_docs"
    assert args.provider == "fastembed"
    assert args.collection == "eval_col"
    assert args.qdrant_url == "http://localhost:6334"
    assert args.in_memory is True
    assert args.k_values == "1,5,10"
    assert args.top_k == 10
    assert args.json is True
    assert args.log_level == "DEBUG"


def test_cli_evaluate_missing_dataset_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI returns error code 1 when dataset file does not exist."""
    missing = tmp_path / "nonexistent_eval.json"
    exit_code = main(["evaluate", "retrieval", str(missing), "--in-memory"])
    assert exit_code == 1

    _, err = capsys.readouterr()
    assert "File not found" in err


def test_cli_evaluate_invalid_k_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI returns error code 1 when k-values cannot be parsed."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "cases": [
                    {"question": "Q1", "expected_sources": ["doc.md"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate",
            "retrieval",
            str(dataset_file),
            "--in-memory",
            "-k",
            "invalid,non,ints",
        ]
    )
    assert exit_code == 1

    _, err = capsys.readouterr()
    assert "Invalid parameter" in err or "Invalid --k-values" in err


def test_cli_evaluate_with_in_memory_and_index_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify end-to-end CLI execution with index-path and in-memory store."""
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text(
        "# FastEmbed Guide\nFastEmbed generates fast local dense embeddings without GPUs.\n",
        encoding="utf-8",
    )

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "cli_test_dataset",
                "cases": [
                    {
                        "question": "What is FastEmbed?",
                        "expected_sources": ["test_doc.md"],
                        "expected_chunks": ["FastEmbed generates fast local dense embeddings"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate",
            "retrieval",
            str(dataset_path),
            "--index-path",
            str(doc_path),
            "--in-memory",
            "-k",
            "1,3,5",
        ]
    )
    assert exit_code == 0

    out, _ = capsys.readouterr()
    assert "Evaluation Report" in out
    assert "Cases: 1" in out
    assert "Recall@1: 1.0000" in out
    assert "Recall@3: 1.0000" in out
    assert "Recall@5: 1.0000" in out
    assert "Precision@5: 0.2000" in out
    assert "MRR: 1.0000" in out


def test_cli_evaluate_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --json flag outputs parsable JSON report."""
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text(
        "# FastEmbed Guide\nFastEmbed generates fast local dense embeddings without GPUs.\n",
        encoding="utf-8",
    )

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "cli_json_dataset",
                "cases": [
                    {
                        "question": "What is FastEmbed?",
                        "expected_sources": ["test_doc.md"],
                        "expected_chunks": ["FastEmbed generates fast local dense embeddings"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate",
            "retrieval",
            str(dataset_path),
            "--index-path",
            str(doc_path),
            "--in-memory",
            "--json",
        ]
    )
    assert exit_code == 0

    out, _ = capsys.readouterr()
    # Find JSON payload in stdout (accounting for log lines)
    json_start = out.find("{")
    assert json_start != -1

    # Extract JSON string: find matching last brace
    json_end = out.rfind("}")
    assert json_end != -1
    parsed_json = json.loads(out[json_start : json_end + 1])
    assert parsed_json["dataset_name"] == "cli_json_dataset"
    assert parsed_json["total_cases"] == 1
    assert parsed_json["mean_reciprocal_rank"] == 1.0
