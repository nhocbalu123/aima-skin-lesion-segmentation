from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.metrics import (
    batch_global_thresholded_dice,
    evaluate_predictions,
    macro_thresholded_dice,
    macro_thresholded_iou,
)


def arr(values: list[list[int]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[None, ..., None]


def test_perfect_prediction_scores_one() -> None:
    y = arr([[1, 0], [0, 1]])
    result = evaluate_predictions(y, y, threshold=0.5)
    assert result["macro_dice"] == 1.0
    assert result["macro_iou"] == 1.0


def test_complete_miss_scores_zero() -> None:
    y_true = arr([[1, 0], [0, 0]])
    y_pred = arr([[0, 0], [0, 0]])
    assert macro_thresholded_dice(y_true, y_pred) == 0.0
    assert macro_thresholded_iou(y_true, y_pred) == 0.0


def test_false_positive_and_false_negative_reduce_scores() -> None:
    y_true = arr([[1, 0], [0, 0]])
    false_positive = arr([[1, 1], [0, 0]])
    false_negative = arr([[0, 0], [0, 0]])
    assert 0.0 < macro_thresholded_dice(y_true, false_positive) < 1.0
    assert macro_thresholded_dice(y_true, false_negative) == 0.0


def test_macro_per_image_differs_from_batch_global() -> None:
    y_true = np.concatenate([
        arr([[1, 0], [0, 0]]),
        arr([[1, 1], [1, 1]]),
    ])
    y_pred = np.concatenate([
        arr([[0, 0], [0, 0]]),
        arr([[1, 1], [1, 1]]),
    ])
    macro = macro_thresholded_dice(y_true, y_pred)
    global_score = batch_global_thresholded_dice(y_true, y_pred)
    assert macro == 0.5
    assert global_score != macro


def test_empty_mask_convention() -> None:
    empty = arr([[0, 0], [0, 0]])
    one_fp = arr([[1, 0], [0, 0]])
    assert macro_thresholded_dice(empty, empty) == 1.0
    assert macro_thresholded_iou(empty, empty) == 1.0
    assert macro_thresholded_dice(empty, one_fp) == 0.0
    assert macro_thresholded_iou(empty, one_fp) == 0.0


def test_empty_mask_metrics_do_not_emit_runtime_warnings() -> None:
    import warnings

    empty = arr([[0, 0], [0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert macro_thresholded_dice(empty, empty) == 1.0
        assert macro_thresholded_iou(empty, empty) == 1.0


def test_saved_prediction_evaluator_reloads_source_of_truth_npz(tmp_path) -> None:
    from skin_lesion_segmentation.evaluation import evaluate_saved_predictions

    y_true = np.zeros((2, 4, 4, 1), dtype=np.uint8)
    y_true[0, 1:3, 1:3, 0] = 1
    probabilities = y_true.astype(np.float32)
    path = tmp_path / "validation_predictions.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["a", "b"]),
        y_true=y_true,
        probabilities=probabilities,
    )

    result = evaluate_saved_predictions(path, threshold=0.5)

    assert result["macro_dice"] == 1.0
    assert result["sample_ids"] == ["a", "b"]
    assert result["predictions_path"] == str(path)


def test_saved_prediction_evaluator_rejects_duplicate_ids(tmp_path) -> None:
    from skin_lesion_segmentation.evaluation import evaluate_saved_predictions

    path = tmp_path / "validation_predictions.npz"
    values = np.zeros((2, 4, 4, 1), dtype=np.float32)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["A", "a"]),
        y_true=values,
        probabilities=values,
    )

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        evaluate_saved_predictions(path)
