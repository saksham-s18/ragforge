"""Unit tests for RAGForge CLI evaluate retrieval command."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ragforge.adapters.fusion import ReciprocalRankFusion
from ragforge.adapters.lexical import BM25LexicalIndex
from ragforge.cli import build_parser, execute_retrieval_evaluation, main
from ragforge.core.config import get_settings
from ragforge.domain.enums import SearchStrategy


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
    assert args.rerank is False
    assert args.candidate_k is None


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
            "--rerank",
            "--candidate-k",
            "25",
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
    assert args.rerank is True
    assert args.candidate_k == 25
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


@pytest.mark.asyncio
async def test_cli_evaluate_default_uses_dense_strategy(tmp_path: Path) -> None:
    """Verify that default evaluation uses SearchStrategy.DENSE without lexical or fusion."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "cases": [
                    {"question": "What is Python?", "expected_sources": ["doc.txt"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["evaluate", "retrieval", str(dataset_file), "--in-memory"])
    settings = get_settings()

    with (
        patch("ragforge.cli.EvaluationRunner") as mock_runner_cls,
        patch("ragforge.cli.RetrievalService") as mock_retrieval_cls,
    ):
        mock_runner = mock_runner_cls.return_value
        mock_runner.evaluate_dataset = AsyncMock()
        await execute_retrieval_evaluation(args, settings)

        # RetrievalService check
        _, r_kwargs = mock_retrieval_cls.call_args
        assert r_kwargs["default_search_strategy"] == SearchStrategy.DENSE
        assert r_kwargs["lexical_index"] is None
        assert r_kwargs["fusion"] is None

        # EvaluationRunner check
        _, runner_kwargs = mock_runner_cls.call_args
        assert runner_kwargs["search_strategy"] == SearchStrategy.DENSE


@pytest.mark.asyncio
async def test_cli_evaluate_hybrid_wires_hybrid_strategy_and_fusion(tmp_path: Path) -> None:
    """Verify that --hybrid wires HYBRID strategy, BM25LexicalIndex, and RRF fusion."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "cases": [
                    {"question": "What is Python?", "expected_sources": ["doc.txt"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    lexical_file = tmp_path / "custom_lexical.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "retrieval",
            str(dataset_file),
            "--in-memory",
            "--hybrid",
            "--lexical-index-file",
            str(lexical_file),
        ]
    )
    settings = get_settings()

    with (
        patch("ragforge.cli.EvaluationRunner") as mock_runner_cls,
        patch("ragforge.cli.RetrievalService") as mock_retrieval_cls,
    ):
        mock_runner = mock_runner_cls.return_value
        mock_runner.evaluate_dataset = AsyncMock()
        await execute_retrieval_evaluation(args, settings)

        # RetrievalService check
        _, r_kwargs = mock_retrieval_cls.call_args
        assert r_kwargs["default_search_strategy"] == SearchStrategy.HYBRID
        assert isinstance(r_kwargs["lexical_index"], BM25LexicalIndex)
        assert isinstance(r_kwargs["fusion"], ReciprocalRankFusion)

        # EvaluationRunner check
        _, runner_kwargs = mock_runner_cls.call_args
        assert runner_kwargs["search_strategy"] == SearchStrategy.HYBRID


def test_cli_evaluate_hybrid_differs_from_dense_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that --hybrid changes evaluation results when lexical match alters ranking."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc_a.txt").write_text(
        "golang programming language compiler syntax runtime",
        encoding="utf-8",
    )
    (doc_dir / "doc_b.txt").write_text(
        "guide to using golang and elixir together for backend services",
        encoding="utf-8",
    )

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps(
            {
                "name": "comparison",
                "cases": [
                    {
                        "question": "golang elixir",
                        "expected_sources": ["doc_b.txt"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # 1. Run dense evaluation: doc_a ranks #1 in dense, doc_b ranks #2 -> Recall@1=0, MRR=0.5
    exit_dense = main(
        [
            "evaluate",
            "retrieval",
            str(dataset_file),
            "--index-path",
            str(doc_dir),
            "--in-memory",
            "-k",
            "1,2",
            "--json",
        ]
    )
    assert exit_dense == 0
    out_dense, _ = capsys.readouterr()
    s_idx = out_dense.find("{")
    e_idx = out_dense.rfind("}")
    parsed_dense = json.loads(out_dense[s_idx : e_idx + 1])

    assert parsed_dense["mean_recall_at_k"]["1"] == 0.0
    assert parsed_dense["mean_reciprocal_rank"] == 0.5

    # 2. Run hybrid evaluation: BM25 matches both terms in doc_b, boosting it to rank 1
    monkeypatch.setenv("RAGFORGE_RRF_LEXICAL_WEIGHT", "2.0")
    get_settings.cache_clear()
    try:
        exit_hybrid = main(
            [
                "evaluate",
                "retrieval",
                str(dataset_file),
                "--index-path",
                str(doc_dir),
                "--in-memory",
                "--hybrid",
                "-k",
                "1,2",
                "--json",
            ]
        )
        assert exit_hybrid == 0
        out_hybrid, _ = capsys.readouterr()
        s_h = out_hybrid.find("{")
        e_h = out_hybrid.rfind("}")
        parsed_hybrid = json.loads(out_hybrid[s_h : e_h + 1])

        assert parsed_hybrid["mean_recall_at_k"]["1"] == 1.0
        assert parsed_hybrid["mean_reciprocal_rank"] == 1.0
    finally:
        get_settings.cache_clear()
