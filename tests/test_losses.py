from __future__ import annotations

import numpy as np

from skin_lesion_segmentation.losses import (
    combined_segmentation_loss_numpy,
    focal_tversky_loss_numpy,
    weighted_dice_loss_numpy,
)


def test_losses_are_near_zero_for_perfect_prediction() -> None:
    y = np.asarray([[[[1.0], [0.0]], [[0.0], [1.0]]]], dtype=np.float32)
    assert weighted_dice_loss_numpy(y, y) < 1e-6
    assert focal_tversky_loss_numpy(y, y) < 1e-6


def test_false_negatives_cost_more_when_beta_exceeds_alpha() -> None:
    y_true = np.asarray([[[[1.0], [1.0]], [[0.0], [0.0]]]], dtype=np.float32)
    fp_heavy = np.asarray([[[[1.0], [1.0]], [[1.0], [1.0]]]], dtype=np.float32)
    fn_heavy = np.asarray([[[[0.0], [0.0]], [[0.0], [0.0]]]], dtype=np.float32)

    fp_loss = focal_tversky_loss_numpy(y_true, fp_heavy, alpha=0.3, beta=0.7)
    fn_loss = focal_tversky_loss_numpy(y_true, fn_heavy, alpha=0.3, beta=0.7)
    assert fn_loss > fp_loss


def test_losses_are_finite_for_extreme_predictions() -> None:
    y_true = np.zeros((2, 4, 4, 1), dtype=np.float32)
    y_pred = np.ones((2, 4, 4, 1), dtype=np.float32)
    assert np.isfinite(weighted_dice_loss_numpy(y_true, y_pred))
    assert np.isfinite(focal_tversky_loss_numpy(y_true, y_pred))


def test_gamma_point_seven_five_is_the_direct_focal_exponent() -> None:
    y_true = np.asarray([[[[1.0], [0.0]]]], dtype=np.float32)
    y_pred = np.asarray([[[[0.5], [0.5]]]], dtype=np.float32)
    expected = 0.5 ** 0.75
    np.testing.assert_allclose(
        focal_tversky_loss_numpy(y_true, y_pred, gamma=0.75), expected, rtol=1e-5
    )


def test_weighted_dice_penalises_false_negative_heavy_prediction_more() -> None:
    y_true = np.asarray([[[[1.0], [1.0]], [[0.0], [0.0]]]], dtype=np.float32)
    fp_heavy = np.asarray([[[[1.0], [1.0]], [[1.0], [1.0]]]], dtype=np.float32)
    fn_heavy = np.zeros_like(y_true)
    assert weighted_dice_loss_numpy(y_true, fn_heavy) > weighted_dice_loss_numpy(y_true, fp_heavy)


def test_combined_loss_numpy_is_finite_and_preserves_fn_weighting() -> None:
    y_true = np.asarray([[[[1.0], [1.0]], [[0.0], [0.0]]]], dtype=np.float32)
    fp_heavy = np.asarray([[[[1.0], [1.0]], [[1.0], [1.0]]]], dtype=np.float32)
    fn_heavy = np.zeros_like(y_true)

    fp_loss = combined_segmentation_loss_numpy(y_true, fp_heavy)
    fn_loss = combined_segmentation_loss_numpy(y_true, fn_heavy)

    assert np.isfinite(fp_loss)
    assert np.isfinite(fn_loss)
    assert fn_loss > fp_loss


def test_weighted_dice_matches_documented_pixel_weight_equation() -> None:
    y_true = np.asarray([[[[1.0], [0.0]]]], dtype=np.float32)
    y_pred = np.asarray([[[[0.5], [0.5]]]], dtype=np.float32)

    loss = weighted_dice_loss_numpy(y_true, y_pred, foreground_weight=2.0)

    expected_dice = 2.0 / 3.5
    np.testing.assert_allclose(loss, 1.0 - expected_dice, rtol=1e-6)
