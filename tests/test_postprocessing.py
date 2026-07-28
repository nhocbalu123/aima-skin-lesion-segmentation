from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.postprocessing import select_postprocessing


def test_selection_reports_raw_and_selected_metrics() -> None:
    y_true = np.zeros((2, 8, 8, 1), dtype=np.float32)
    y_true[:, 2:6, 2:6, 0] = 1.0
    probabilities = y_true * 0.9
    probabilities[0, 0, 0, 0] = 0.8

    result = select_postprocessing(
        y_true,
        probabilities,
        thresholds=[0.5],
        min_component_sizes=[0, 2],
        morphology_kernels=[0],
    )

    assert "raw_metrics" in result
    assert "selected_metrics" in result
    assert result["selection_source"] == "validation_predictions_only"
    assert result["selected_params"]["min_component_size"] == 2


def test_raw_metrics_always_use_threshold_point_five() -> None:
    y_true = np.zeros((1, 4, 4, 1), dtype=np.float32)
    probabilities = np.full_like(y_true, 0.4)
    result = select_postprocessing(
        y_true,
        probabilities,
        thresholds=[0.3, 0.5],
        min_component_sizes=[0],
        morphology_kernels=[0],
    )
    assert result["raw_params"]["threshold"] == 0.5
    assert result["raw_metrics"]["macro_dice"] == 1.0


def test_equal_scores_prefer_no_postprocessing() -> None:
    y_true = np.zeros((1, 4, 4, 1), dtype=np.float32)
    probabilities = np.zeros_like(y_true)
    result = select_postprocessing(
        y_true,
        probabilities,
        thresholds=[0.5],
        min_component_sizes=[0, 2],
        morphology_kernels=[0, 3],
    )
    assert result["selected_params"] == {
        "threshold": 0.5,
        "min_component_size": 0,
        "morphology_kernel": 0,
    }


def test_selection_never_replaces_raw_baseline_with_worse_candidate() -> None:
    y_true = np.zeros((1, 4, 4, 1), dtype=np.float32)
    y_true[:, 1:3, 1:3, 0] = 1.0
    probabilities = y_true * 0.6

    result = select_postprocessing(
        y_true,
        probabilities,
        thresholds=[0.9],
        min_component_sizes=[0],
        morphology_kernels=[0],
    )

    assert result["raw_metrics"]["macro_dice"] == 1.0
    assert result["selected_params"] == result["raw_params"]
    assert result["selected_metrics"] == result["raw_metrics"]
    assert result["selected_improves_raw"] is False


@pytest.mark.parametrize(
    ("min_component_size", "morphology_kernel"),
    [(-1, 0), (0, -1), (0, 2)],
)
def test_postprocessing_rejects_invalid_parameters(
    min_component_size: int,
    morphology_kernel: int,
) -> None:
    from skin_lesion_segmentation.postprocessing import apply_postprocessing

    with pytest.raises(ValueError):
        apply_postprocessing(
            np.zeros((4, 4), dtype=np.uint8),
            min_component_size,
            morphology_kernel,
        )
