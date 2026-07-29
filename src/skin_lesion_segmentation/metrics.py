"""Unambiguous training and final segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalise_batch(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 2:
        result = result[None, ..., None]
    elif result.ndim == 3:
        result = result[..., None]
    if result.ndim != 4:
        raise ValueError(f"Expected [N,H,W,C] data, got shape {result.shape}")
    return result


def _binary_pair(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    truth = _normalise_batch(y_true) > 0.5
    pred = _normalise_batch(y_pred) >= threshold
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    return truth, pred


def per_image_thresholded_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return thresholded Dice per image.

    Empty-mask convention: an empty ground truth and empty prediction scores 1;
    an empty ground truth with any predicted foreground scores 0.
    """

    truth, pred = _binary_pair(y_true, y_pred, threshold)
    axes = tuple(range(1, truth.ndim))
    intersection = np.sum(truth & pred, axis=axes, dtype=np.float64)
    truth_sum = np.sum(truth, axis=axes, dtype=np.float64)
    pred_sum = np.sum(pred, axis=axes, dtype=np.float64)
    denominator = truth_sum + pred_sum
    scores = np.ones_like(denominator, dtype=np.float64)
    np.divide(2.0 * intersection, denominator, out=scores, where=denominator != 0)
    return scores


def per_image_thresholded_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return thresholded intersection-over-union per image with the same empty convention as Dice."""

    truth, pred = _binary_pair(y_true, y_pred, threshold)
    axes = tuple(range(1, truth.ndim))
    intersection = np.sum(truth & pred, axis=axes, dtype=np.float64)
    union = np.sum(truth | pred, axis=axes, dtype=np.float64)
    scores = np.ones_like(union, dtype=np.float64)
    np.divide(intersection, union, out=scores, where=union != 0)
    return scores


def macro_thresholded_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean(per_image_thresholded_dice(y_true, y_pred, threshold), dtype=np.float64))


def macro_thresholded_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean(per_image_thresholded_iou(y_true, y_pred, threshold), dtype=np.float64))


def batch_global_thresholded_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Legacy-style batch-global thresholded Dice, exposed only for comparison."""

    truth, pred = _binary_pair(y_true, y_pred, threshold)
    intersection = float(np.sum(truth & pred, dtype=np.float64))
    denominator = float(np.sum(truth, dtype=np.float64) + np.sum(pred, dtype=np.float64))
    return 1.0 if denominator == 0.0 else 2.0 * intersection / denominator


def soft_dice_batch_global(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Soft batch-global Dice for training diagnostics, not final evaluation."""

    truth = _normalise_batch(y_true).astype(np.float32)
    pred = _normalise_batch(y_pred).astype(np.float32)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {truth.shape} vs {pred.shape}")
    intersection = np.sum(truth * pred, dtype=np.float32)
    denominator = np.sum(truth, dtype=np.float32) + np.sum(pred, dtype=np.float32)
    return float((2.0 * intersection + np.float32(smooth)) / (denominator + np.float32(smooth)))


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Source-of-truth offline evaluation for saved validation probabilities."""

    per_dice = per_image_thresholded_dice(y_true, probabilities, threshold)
    per_iou = per_image_thresholded_iou(y_true, probabilities, threshold)
    return {
        "threshold": float(threshold),
        "macro_dice": float(np.mean(per_dice, dtype=np.float64)),
        "macro_iou": float(np.mean(per_iou, dtype=np.float64)),
        "per_image_dice": [float(value) for value in per_dice],
        "per_image_iou": [float(value) for value in per_iou],
        "empty_ground_truth_convention": "empty prediction scores 1; non-empty prediction scores 0",
        "n_images": int(len(per_dice)),
    }


def soft_dice_batch_global_tf(y_true: object, y_pred: object, smooth: float = 1e-6) -> object:
    """TensorFlow soft batch-global Dice for training diagnostics only."""

    import tensorflow as tf

    truth = tf.cast(y_true, tf.float32)
    pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(truth * pred)
    denominator = tf.reduce_sum(truth) + tf.reduce_sum(pred)
    eps = tf.constant(smooth, dtype=tf.float32)
    return tf.cast((2.0 * intersection + eps) / (denominator + eps), tf.float32)
