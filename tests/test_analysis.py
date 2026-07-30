from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.analysis import (
    paired_bootstrap_interval,
    per_image_comparison,
    summarize_slices,
)


def test_per_image_comparison_reports_paired_metrics_and_mask_properties() -> None:
    truth = np.zeros((2, 4, 4, 1), dtype=np.float32)
    truth[0, 1:3, 1:3, 0] = 1.0
    truth[1, 0:2, 0:2, 0] = 1.0
    unet = np.zeros_like(truth)
    unet[0, 1:3, 1:3, 0] = 1.0
    unet[1, 0, 0, 0] = 1.0
    attention = np.zeros_like(truth)
    attention[0, 1, 1, 0] = 1.0
    attention[1, 0:2, 0:2, 0] = 1.0

    rows = per_image_comparison(
        ["interior", "border"],
        truth,
        unet,
        attention,
        threshold=0.5,
    )

    assert rows[0]["unet_dice"] == 1.0
    assert rows[0]["attention_dice"] == 0.4
    assert rows[0]["dice_difference"] == -0.6
    assert rows[0]["lesion_fraction"] == 0.25
    assert rows[0]["border_touching"] is False
    assert rows[0]["attention_false_negative_fraction"] == 3 / 16
    assert rows[1]["unet_dice"] == 0.4
    assert rows[1]["attention_dice"] == 1.0
    assert rows[1]["dice_difference"] == 0.6
    assert rows[1]["border_touching"] is True


def test_per_image_comparison_uses_fixed_lesion_size_slices() -> None:
    truth = np.zeros((4, 20, 20, 1), dtype=np.float32)
    truth[1, 0, 0, 0] = 1.0
    truth[2, :4, :5, 0] = 1.0
    truth[3, :10, :10, 0] = 1.0

    rows = per_image_comparison(
        ["empty", "small", "medium", "large"],
        truth,
        truth,
        truth,
    )

    assert [row["lesion_size"] for row in rows] == [
        "empty",
        "small",
        "medium",
        "large",
    ]


def test_per_image_comparison_rejects_duplicate_ids() -> None:
    values = np.zeros((2, 2, 2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate"):
        per_image_comparison(["same", "same"], values, values, values)


def test_paired_bootstrap_interval_uses_paired_candidate_minus_baseline() -> None:
    baseline = np.asarray([0.1, 0.2, 0.3, 0.4])
    candidate = baseline + 0.05

    result = paired_bootstrap_interval(
        baseline,
        candidate,
        seed=7,
        n_resamples=500,
    )

    assert result["difference_definition"] == "candidate_minus_baseline"
    assert result["difference_mean"] == pytest.approx(0.05)
    assert result["ci_low"] == pytest.approx(0.05)
    assert result["ci_high"] == pytest.approx(0.05)
    assert result["n_pairs"] == 4
    assert result["n_resamples"] == 500


def test_paired_bootstrap_is_deterministic_and_contains_observed_mean() -> None:
    baseline = np.asarray([0.2, 0.8, 0.4, 0.6, 0.5])
    candidate = np.asarray([0.3, 0.7, 0.6, 0.65, 0.55])

    first = paired_bootstrap_interval(baseline, candidate, seed=91, n_resamples=1000)
    second = paired_bootstrap_interval(baseline, candidate, seed=91, n_resamples=1000)

    assert first == second
    assert first["ci_low"] <= first["difference_mean"] <= first["ci_high"]


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([], [], "non-empty"),
        ([1.0], [1.0, 2.0], "same length"),
        ([[1.0]], [[1.0]], "one-dimensional"),
    ],
)
def test_paired_bootstrap_rejects_invalid_pairs(
    baseline: object,
    candidate: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_interval(baseline, candidate, seed=1, n_resamples=10)


def test_slice_summary_keeps_empty_predeclared_slices_visible() -> None:
    rows = [
        {
            "lesion_size": "small",
            "border_touching": False,
            "unet_dice": 0.6,
            "attention_dice": 0.7,
            "unet_iou": 0.5,
            "attention_iou": 0.6,
        },
        {
            "lesion_size": "large",
            "border_touching": True,
            "unet_dice": 0.8,
            "attention_dice": 0.75,
            "unet_iou": 0.7,
            "attention_iou": 0.65,
        },
    ]

    summary = summarize_slices(rows)

    assert summary["lesion_size"]["small"]["n_images"] == 1
    assert summary["lesion_size"]["medium"]["n_images"] == 0
    assert summary["lesion_size"]["large"]["dice_difference_mean"] == pytest.approx(-0.05)
    assert summary["border_touching"]["true"]["n_images"] == 1
    assert summary["border_touching"]["false"]["dice_difference_mean"] == pytest.approx(0.1)
