"""Validated submission generation aligned to an official sample template."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import SUPPORTED_IMAGE_EXTENSIONS
from .rle import encode_rle


class SubmissionError(ValueError):
    """Raised when prediction IDs or submission schema are inconsistent."""


def sample_id_from_test_path(path: Path | str) -> str:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise SubmissionError(f"Unsupported test image extension: {path.suffix}")
    if not path.stem:
        raise SubmissionError(f"Empty test sample ID: {path}")
    return path.stem


def infer_submission_id_suffix(
    test_paths: Sequence[Path | str],
    sample_submission: pd.DataFrame,
    *,
    id_column: str,
) -> str:
    """Infer one constant ID suffix only when the official template proves it.

    The function accepts either a direct stem mapping (empty suffix) or a
    single constant suffix such as ``_segmentation``. Any non-uniform mapping
    is rejected rather than guessed.
    """

    if id_column not in sample_submission.columns:
        raise SubmissionError(f"Sample submission is missing ID column '{id_column}'")
    stems = [sample_id_from_test_path(path) for path in test_paths]
    if len({stem.casefold() for stem in stems}) != len(stems):
        raise SubmissionError("Test images contain duplicate IDs")
    expected_ids = sample_submission[id_column].astype(str).tolist()
    if len(expected_ids) != len(stems):
        raise SubmissionError(
            f"Cannot infer ID suffix: {len(stems)} test images but {len(expected_ids)} template rows"
        )
    expected_set = set(expected_ids)
    first_stem = stems[0] if stems else ""
    candidates = {
        expected_id[len(first_stem) :]
        for expected_id in expected_ids
        if expected_id.startswith(first_stem)
    }
    valid = [suffix for suffix in candidates if {f"{stem}{suffix}" for stem in stems} == expected_set]
    if len(valid) != 1:
        raise SubmissionError(
            "Official sample submission IDs do not map to test image stems through one constant suffix"
        )
    return valid[0]


def build_submission(
    test_paths: Sequence[Path | str],
    predictions: Sequence[np.ndarray],
    *,
    sample_submission: pd.DataFrame | None = None,
    id_column: str = "id",
    mask_column: str = "segmentation",
    submission_id_suffix: str = "",
    rle_order: str = "C",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build and validate a submission; the sample template controls row order."""

    if len(test_paths) != len(predictions):
        raise SubmissionError(
            f"Prediction count mismatch: {len(test_paths)} test images, {len(predictions)} predictions"
        )
    encoded_by_id: dict[str, str] = {}
    canonical_ids: dict[str, str] = {}
    for path, prediction in zip(test_paths, predictions, strict=True):
        sample_id = f"{sample_id_from_test_path(path)}{submission_id_suffix}"
        canonical = sample_id.casefold()
        if canonical in canonical_ids:
            raise SubmissionError(
                f"duplicate test ID '{sample_id}' conflicts with '{canonical_ids[canonical]}'"
            )
        canonical_ids[canonical] = sample_id
        encoded_by_id[sample_id] = encode_rle(np.asarray(prediction), order=rle_order)

    validated = sample_submission is not None
    if sample_submission is not None:
        expected_columns = [id_column, mask_column]
        if sample_submission.columns.tolist() != expected_columns:
            raise SubmissionError(
                f"Sample submission columns must be {expected_columns}, got {sample_submission.columns.tolist()}"
            )
        template_ids = sample_submission[id_column].astype(str)
        if template_ids.str.casefold().duplicated().any():
            raise SubmissionError("Sample submission contains duplicate IDs")
        expected_ids = sample_submission[id_column].astype(str).tolist()
        actual_ids = set(encoded_by_id)
        if set(expected_ids) != actual_ids:
            missing = sorted(set(expected_ids) - actual_ids)
            extra = sorted(actual_ids - set(expected_ids))
            raise SubmissionError(f"ID mismatch: missing={missing}, extra={extra}")
        result = sample_submission.copy()
        result[mask_column] = result[id_column].astype(str).map(encoded_by_id)
        result = result[expected_columns]
    else:
        result = pd.DataFrame(
            [{id_column: sample_id, mask_column: encoded_by_id[sample_id]} for sample_id in sorted(encoded_by_id)]
        )

    if len(result) != len(test_paths):
        raise SubmissionError(f"Row count mismatch: expected {len(test_paths)}, got {len(result)}")
    if result[id_column].duplicated().any():
        raise SubmissionError("Submission contains duplicate IDs")
    if result[mask_column].isna().any():
        raise SubmissionError("Submission contains missing encodings")
    summary: dict[str, object] = {
        "rows": int(len(result)),
        "columns": result.columns.tolist(),
        "ids_unique": True,
        "sample_submission_validated": validated,
        "rle_order": rle_order.upper(),
        "submission_id_suffix": submission_id_suffix,
    }
    return result, summary
