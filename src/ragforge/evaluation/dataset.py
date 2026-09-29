"""Dataset loader, validator, and serializer for RAG retrieval evaluation benchmarks.

Supports:
- JSON files with top-level dataset envelope: {"name": "...", "cases": [...]}
- JSON files with raw list of case objects: [{...}, {...}]
- JSON Lines (JSONL) files with one case per line
"""

import json
import logging
from pathlib import Path
from typing import Any

from ragforge.domain.exceptions import DatasetValidationError, DocumentNotFoundError
from ragforge.domain.models import EvaluationCase, EvaluationDataset

logger = logging.getLogger(__name__)


def validate_evaluation_dataset(dataset: EvaluationDataset) -> None:
    """Validate structural and semantic integrity of an evaluation dataset.

    Args:
        dataset: EvaluationDataset instance to validate.

    Raises:
        DatasetValidationError: If the dataset contains no cases or if any case
            has an invalid/empty question.
    """
    if not dataset.cases:
        raise DatasetValidationError(
            f"Evaluation dataset '{dataset.name}' must contain at least one evaluation case."
        )

    for idx, case in enumerate(dataset.cases, start=1):
        if not case.question or not case.question.strip():
            raise DatasetValidationError(
                f"Evaluation case #{idx} (id='{case.id}') has an empty question."
            )

        if not case.expected_chunks and not case.expected_sources:
            logger.warning(
                "Evaluation case #{idx} (id='%s') has neither expected_chunks nor "
                "expected_sources; retrieval relevance cannot be scored positively.",
                case.id,
            )


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate an EvaluationDataset from a JSON or JSONL file on disk.

    Args:
        path: Path to the dataset file.

    Returns:
        Validated EvaluationDataset instance.

    Raises:
        DocumentNotFoundError: If the file path does not exist.
        DatasetValidationError: If parsing fails or the dataset schema is invalid.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise DocumentNotFoundError(f"Evaluation dataset file not found: {target}")

    if not target.is_file():
        raise DatasetValidationError(f"Evaluation dataset path is not a file: {target}")

    try:
        raw_text = target.read_text(encoding="utf-8")
    except Exception as exc:
        raise DatasetValidationError(
            f"Failed to read evaluation dataset file '{target}': {exc}"
        ) from exc

    if not raw_text.strip():
        raise DatasetValidationError(f"Evaluation dataset file '{target}' is empty.")

    # Parse JSON Lines format
    if target.suffix.lower() == ".jsonl":
        cases: list[EvaluationCase] = []
        for line_idx, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item_dict = json.loads(stripped)
                if not isinstance(item_dict, dict):
                    raise DatasetValidationError(
                        f"Line {line_idx} in '{target.name}' is not a JSON object."
                    )
                cases.append(EvaluationCase.model_validate(item_dict))
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Invalid JSON at line {line_idx} in '{target.name}': {exc}"
                ) from exc
            except Exception as exc:
                raise DatasetValidationError(
                    f"Invalid case definition at line {line_idx} in '{target.name}': {exc}"
                ) from exc

        dataset = EvaluationDataset(name=target.stem, cases=cases)
        validate_evaluation_dataset(dataset)
        return dataset

    # Standard JSON format
    try:
        parsed: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Failed to parse JSON in evaluation dataset '{target.name}': {exc}"
        ) from exc

    try:
        if isinstance(parsed, list):
            # Top-level array of cases
            cases = [EvaluationCase.model_validate(c) for c in parsed]
            dataset = EvaluationDataset(name=target.stem, cases=cases)
        elif isinstance(parsed, dict):
            if "cases" in parsed:
                dataset = EvaluationDataset.model_validate(parsed)
            else:
                # Single case dictionary or envelope without 'cases'
                raise DatasetValidationError(
                    "Evaluation dataset JSON root object must contain a 'cases' list: "
                    f"{target.name}"
                )
        else:
            raise DatasetValidationError(
                f"Unexpected JSON root structure (expected object or list): {type(parsed).__name__}"
            )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"Validation error loading evaluation dataset '{target.name}': {exc}"
        ) from exc

    validate_evaluation_dataset(dataset)
    return dataset


def save_evaluation_dataset(dataset: EvaluationDataset, path: str | Path) -> None:
    """Serialize an EvaluationDataset to formatted JSON file.

    Args:
        dataset: EvaluationDataset instance to persist.
        path: Target file path.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
