"""Numerically safe binary segmentation losses.

Weighted Dice applies a larger pixel weight to foreground targets before
computing Dice per image. Focal Tversky uses
``TP / (TP + alpha*FP + beta*FN + eps)`` and
``(1 - Tversky) ** gamma``. Therefore alpha weights false positives and beta
weights false negatives; beta > alpha penalises misses more strongly. Here
``gamma=0.75`` is the direct exponent. This is algebraically equivalent to the
paper's ``(1 - Tversky) ** (1 / gamma_paper)`` with ``gamma_paper=4/3``.
"""

from __future__ import annotations

import numpy as np


def _numpy_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {truth.shape} vs {pred.shape}")
    return truth, np.clip(pred, 0.0, 1.0)


def weighted_dice_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    foreground_weight: float = 2.0,
    smooth: float = 1e-6,
) -> float:
    """Per-image pixel-weighted soft Dice loss.

    ``weight = foreground_weight`` where the ground truth is foreground and
    ``1`` otherwise. False negatives therefore carry the foreground weight,
    while false positives on background pixels carry weight 1.
    """

    if foreground_weight <= 0.0:
        raise ValueError("foreground_weight must be positive")
    truth, pred = _numpy_arrays(y_true, y_pred)
    axes = tuple(range(1, truth.ndim)) if truth.ndim > 1 else (0,)
    weight = np.float32(1.0) + np.float32(foreground_weight - 1.0) * truth
    intersection = np.sum(weight * truth * pred, axis=axes, dtype=np.float32)
    denominator = np.sum(weight * truth, axis=axes, dtype=np.float32) + np.sum(
        weight * pred,
        axis=axes,
        dtype=np.float32,
    )
    dice = (np.float32(2.0) * intersection + np.float32(smooth)) / (
        denominator + np.float32(smooth)
    )
    return float(np.mean(np.float32(1.0) - dice, dtype=np.float32))


def focal_tversky_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1e-6,
) -> float:
    if alpha < 0 or beta < 0 or gamma <= 0:
        raise ValueError("alpha and beta must be non-negative; gamma must be positive")
    truth, pred = _numpy_arrays(y_true, y_pred)
    axes = tuple(range(1, truth.ndim)) if truth.ndim > 1 else (0,)
    tp = np.sum(truth * pred, axis=axes, dtype=np.float32)
    fp = np.sum((1.0 - truth) * pred, axis=axes, dtype=np.float32)
    fn = np.sum(truth * (1.0 - pred), axis=axes, dtype=np.float32)
    tversky = (tp + np.float32(smooth)) / (
        tp + np.float32(alpha) * fp + np.float32(beta) * fn + np.float32(smooth)
    )
    exponent = np.float32(gamma)
    return float(np.mean(np.power(1.0 - tversky, exponent), dtype=np.float32))


def combined_segmentation_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    foreground_weight: float = 2.0,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
) -> float:
    """NumPy reference for the exact compound loss used by TensorFlow training."""

    dice = weighted_dice_loss_numpy(
        y_true,
        y_pred,
        foreground_weight=foreground_weight,
    )
    tversky = focal_tversky_loss_numpy(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return float(np.float32(0.5) * np.float32(dice) + np.float32(0.5) * np.float32(tversky))


def combined_segmentation_loss(y_true: object, y_pred: object) -> object:
    """TensorFlow weighted-Dice plus focal-Tversky loss with float32 reductions."""

    import tensorflow as tf

    truth = tf.cast(y_true, tf.float32)
    pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    axes = tf.range(1, tf.rank(truth))
    eps = tf.constant(1e-6, tf.float32)

    foreground_weight = tf.constant(2.0, tf.float32)
    pixel_weight = 1.0 + (foreground_weight - 1.0) * truth
    weighted_intersection = tf.reduce_sum(pixel_weight * truth * pred, axis=axes)
    weighted_denominator = tf.reduce_sum(pixel_weight * truth, axis=axes) + tf.reduce_sum(
        pixel_weight * pred,
        axis=axes,
    )
    weighted_dice = (2.0 * weighted_intersection + eps) / (weighted_denominator + eps)
    dice_loss = tf.reduce_mean(1.0 - weighted_dice)

    tp = tf.reduce_sum(truth * pred, axis=axes)
    fp = tf.reduce_sum((1.0 - truth) * pred, axis=axes)
    fn = tf.reduce_sum(truth * (1.0 - pred), axis=axes)
    tversky = (tp + eps) / (tp + 0.3 * fp + 0.7 * fn + eps)
    focal = tf.reduce_mean(tf.pow(1.0 - tversky, tf.constant(0.75, tf.float32)))
    return tf.cast(0.5 * dice_loss + 0.5 * focal, tf.float32)
