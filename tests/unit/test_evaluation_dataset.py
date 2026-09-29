"""Unit tests for evaluation dataset loading, parsing, and validation."""

import json
from pathlib import Path

import pytest

from ragforge.domain.exceptions import DatasetValidationError, DocumentNotFoundError
from ragforge.domain.models import EvaluationCase, EvaluationDataset
from ragforge.evaluation.dataset import (
    load_evaluation_dataset,
    save_evaluation_dataset,
    validate_evaluation_dataset,
)


def test_evaluation_case_singular_plural_normalization() -> None:
    """Verify EvaluationCase normalizes single strings to lists and handles aliases."""
    data = {
        "question": "What is RAGForge?",
        "expected_document": "ragforge_test.md",
        "expected_chunk": "FastAPI framework",
        "expected_fact": "Production RAG",
        "expected_answers": ["A production RAG platform."],
    }
    case = EvaluationCase.model_validate(data)

    assert case.question == "What is RAGForge?"
    assert case.expected_sources == ["ragforge_test.md"]
    assert case.expected_chunks == ["FastAPI framework"]
    assert case.expected_facts == ["Production RAG"]
    assert case.expected_answer == "A production RAG platform."


def test_load_dataset_valid_json_dict(tmp_path: Path) -> None:
    """Verify loading dataset from standard JSON dictionary with envelope."""
    payload = {
        "name": "test_kb",
        "description": "Sample test suite",
        "cases": [
            {
                "id": "c1",
                "question": "What is FastAPI?",
                "expected_sources": ["doc.md"],
                "expected_chunks": ["asynchronous routing"],
            },
            {
                "id": "c2",
                "question": "What is Qdrant?",
                "expected_sources": ["doc.md"],
                "expected_chunks": ["vector database"],
            },
        ],
    }
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_evaluation_dataset(file_path)
    assert dataset.name == "test_kb"
    assert len(dataset) == 2
    assert dataset[0].id == "c1"
    assert dataset[1].question == "What is Qdrant?"


def test_load_dataset_valid_json_list(tmp_path: Path) -> None:
    """Verify loading dataset from a raw JSON list of cases."""
    cases = [
        {
            "id": "c1",
            "question": "Question 1",
            "expected_sources": ["source1.md"],
        },
        {
            "id": "c2",
            "question": "Question 2",
            "expected_chunks": ["target chunk"],
        },
    ]
    file_path = tmp_path / "raw_cases.json"
    file_path.write_text(json.dumps(cases), encoding="utf-8")

    dataset = load_evaluation_dataset(file_path)
    assert dataset.name == "raw_cases"
    assert len(dataset) == 2
    assert dataset[0].question == "Question 1"


def test_load_dataset_valid_jsonl(tmp_path: Path) -> None:
    """Verify loading dataset from JSON Lines (.jsonl) format."""
    lines = [
        json.dumps({"question": "Q1", "expected_sources": ["doc1.md"]}),
        "",  # Empty line to verify whitespace handling
        json.dumps({"question": "Q2", "expected_chunks": ["chunk text"]}),
    ]
    file_path = tmp_path / "dataset.jsonl"
    file_path.write_text("\n".join(lines), encoding="utf-8")

    dataset = load_evaluation_dataset(file_path)
    assert len(dataset) == 2
    assert dataset[0].question == "Q1"
    assert dataset[1].question == "Q2"


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    """Verify DocumentNotFoundError is raised if dataset file does not exist."""
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(DocumentNotFoundError, match="file not found"):
        load_evaluation_dataset(missing)


def test_load_dataset_directory_raises(tmp_path: Path) -> None:
    """Verify DatasetValidationError is raised if dataset path is a directory."""
    with pytest.raises(DatasetValidationError, match="path is not a file"):
        load_evaluation_dataset(tmp_path)


def test_load_dataset_invalid_json_syntax_raises(tmp_path: Path) -> None:
    """Verify DatasetValidationError on syntax errors in JSON."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Failed to parse JSON"):
        load_evaluation_dataset(bad_json)


def test_load_dataset_invalid_jsonl_line_raises(tmp_path: Path) -> None:
    """Verify DatasetValidationError on syntax error in JSON Lines."""
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"question": "Q1"}\n{not a json}\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Invalid JSON at line 2"):
        load_evaluation_dataset(bad_jsonl)


def test_load_dataset_empty_cases_raises(tmp_path: Path) -> None:
    """Verify DatasetValidationError when dataset contains no cases."""
    empty_dataset = tmp_path / "empty.json"
    empty_dataset.write_text(json.dumps({"cases": []}), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="must contain at least one evaluation case"):
        load_evaluation_dataset(empty_dataset)


def test_load_dataset_empty_file_raises(tmp_path: Path) -> None:
    """Verify DatasetValidationError when dataset file is empty."""
    empty_file = tmp_path / "empty_file.json"
    empty_file.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="is empty"):
        load_evaluation_dataset(empty_file)


def test_validate_evaluation_dataset_empty_question_raises() -> None:
    """Verify validation catches empty questions."""
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(question="   ", expected_sources=["doc.md"]),
        ]
    )
    with pytest.raises(DatasetValidationError, match="empty question"):
        validate_evaluation_dataset(dataset)


def test_save_and_reload_dataset(tmp_path: Path) -> None:
    """Verify round-trip dataset serialization and loading."""
    cases = [
        EvaluationCase(
            question="What is Qdrant?",
            expected_answer="Vector database",
            expected_sources=["ragforge_test.md"],
            expected_chunks=["vector database"],
            expected_facts=["Qdrant"],
        )
    ]
    ds = EvaluationDataset(name="roundtrip", cases=cases)
    target_path = tmp_path / "saved.json"

    save_evaluation_dataset(ds, target_path)
    loaded = load_evaluation_dataset(target_path)

    assert loaded.name == "roundtrip"
    assert len(loaded) == 1
    assert loaded[0].question == "What is Qdrant?"
    assert loaded[0].expected_answer == "Vector database"
    assert loaded[0].expected_sources == ["ragforge_test.md"]
