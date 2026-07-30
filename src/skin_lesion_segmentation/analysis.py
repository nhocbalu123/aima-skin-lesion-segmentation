"""TensorFlow-free paired comparison and robustness slicing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .metrics import per_image_thresholded_dice, per_image_thresholded_iou

LESION_SIZE_SLICES = ("empty", "small", "medium", "large")


def _normalise_segmentation_batch(value: object, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError(f"{field} must have shape [N,H,W,1]")
    return array


def _lesion_size(fraction: float) -> str:
    if fraction == 0.0:
        return "empty"
    if fraction <= 0.01:
        return "small"
    if fraction <= 0.10:
        return "medium"
    return "large"


def per_image_comparison(
    sample_ids: Sequence[str],
    y_true: object,
    unet_probabilities: object,
    attention_probabilities: object,
    *,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Build paired image-level evidence with fixed mask-derived slices."""

    truth = _normalise_segmentation_batch(y_true, "y_true")
    unet = _normalise_segmentation_batch(unet_probabilities, "unet_probabilities")
    attention = _normalise_segmentation_batch(
        attention_probabilities,
        "attention_probabilities",
    )
    if truth.shape != unet.shape or truth.shape != attention.shape:
        raise ValueError("Paired truth and probability shapes must match")
    ids = [str(value).strip() for value in sample_ids]
    if len(ids) != len(truth):
        raise ValueError("sample_ids length must match prediction batches")
    canonical = [value.casefold() for value in ids]
    if any(not value for value in canonical):
        raise ValueError("sample_ids must not contain blanks")
    if len(set(canonical)) != len(canonical):
        raise ValueError("sample_ids must not contain duplicate values")

    unet_dice = per_image_thresholded_dice(truth, unet, threshold)
    attention_dice = per_image_thresholded_dice(truth, attention, threshold)
    unet_iou = per_image_thresholded_iou(truth, unet, threshold)
    attention_iou = per_image_thresholded_iou(truth, attention, threshold)
    truth_binary = truth > 0.5
    unet_binary = unet >= threshold
    attention_binary = attention >= threshold
    pixel_axes = (1, 2, 3)
    lesion_fractions = np.mean(truth_binary, axis=pixel_axes, dtype=np.float64)
    unet_fp = np.mean(
        unet_binary & ~truth_binary,
        axis=pixel_axes,
        dtype=np.float64,
    )
    unet_fn = np.mean(
        truth_binary & ~unet_binary,
        axis=pixel_axes,
        dtype=np.float64,
    )
    attention_fp = np.mean(
        attention_binary & ~truth_binary,
        axis=pixel_axes,
        dtype=np.float64,
    )
    attention_fn = np.mean(
        truth_binary & ~attention_binary,
        axis=pixel_axes,
        dtype=np.float64,
    )

    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        mask = truth_binary[index, ..., 0]
        border_touching = bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        )
        rows.append(
            {
                "sample_id": sample_id,
                "threshold": float(threshold),
                "unet_dice": float(unet_dice[index]),
                "attention_dice": float(attention_dice[index]),
                "dice_difference": float(attention_dice[index] - unet_dice[index]),
                "unet_iou": float(unet_iou[index]),
                "attention_iou": float(attention_iou[index]),
                "iou_difference": float(attention_iou[index] - unet_iou[index]),
                "lesion_fraction": float(lesion_fractions[index]),
                "lesion_size": _lesion_size(float(lesion_fractions[index])),
                "border_touching": border_touching,
                "unet_false_positive_fraction": float(unet_fp[index]),
                "unet_false_negative_fraction": float(unet_fn[index]),
                "attention_false_positive_fraction": float(attention_fp[index]),
                "attention_false_negative_fraction": float(attention_fn[index]),
            }
        )
    return rows


def paired_bootstrap_interval(
    baseline: object,
    candidate: object,
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile paired-bootstrap interval for candidate minus baseline."""

    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("Paired values must be one-dimensional")
    if left.size == 0 or right.size == 0:
        raise ValueError("Paired values must be non-empty")
    if left.size != right.size:
        raise ValueError("Paired values must have the same length")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Paired values must be finite")
    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    differences = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(n_resamples, len(differences)),
    )
    bootstrap_means = np.mean(differences[indices], axis=1, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        bootstrap_means,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    return {
        "difference_definition": "candidate_minus_baseline",
        "difference_mean": float(np.mean(differences, dtype=np.float64)),
        "confidence": float(confidence),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": int(len(differences)),
        "n_resamples": n_resamples,
        "seed": int(seed),
    }


def _slice_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_images": 0,
            "unet_macro_dice": None,
            "attention_macro_dice": None,
            "dice_difference_mean": None,
            "unet_macro_iou": None,
            "attention_macro_iou": None,
            "iou_difference_mean": None,
        }

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows], dtype=np.float64))

    return {
        "n_images": len(rows),
        "unet_macro_dice": mean("unet_dice"),
        "attention_macro_dice": mean("attention_dice"),
        "dice_difference_mean": mean("attention_dice") - mean("unet_dice"),
        "unet_macro_iou": mean("unet_iou"),
        "attention_macro_iou": mean("attention_iou"),
        "iou_difference_mean": mean("attention_iou") - mean("unet_iou"),
    }


def summarize_slices(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize fixed lesion-size and border-touching slices."""

    materialised = list(rows)
    return {
        "lesion_size": {
            label: _slice_metrics(
                [row for row in materialised if row.get("lesion_size") == label]
            )
            for label in LESION_SIZE_SLICES
        },
        "border_touching": {
            "true": _slice_metrics(
                [row for row in materialised if row.get("border_touching") is True]
            ),
            "false": _slice_metrics(
                [row for row in materialised if row.get("border_touching") is False]
            ),
        },
    }
