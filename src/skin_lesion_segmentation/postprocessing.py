"""Validation-only threshold and post-processing selection."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import cv2
import numpy as np

from .metrics import evaluate_predictions


def remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if min_component_size <= 1:
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_size:
            output[labels == label] = 1
    return output


def apply_postprocessing(mask: np.ndarray, min_component_size: int, morphology_kernel: int) -> np.ndarray:
    if min_component_size < 0:
        raise ValueError("min_component_size must be non-negative")
    if morphology_kernel < 0 or (morphology_kernel > 0 and morphology_kernel % 2 == 0):
        raise ValueError("morphology_kernel must be zero or a positive odd integer")
    output = remove_small_components(mask, min_component_size)
    if morphology_kernel > 0:
        kernel = np.ones((morphology_kernel, morphology_kernel), dtype=np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    return (output > 0).astype(np.uint8)


def _postprocess_batch(
    probabilities: np.ndarray,
    threshold: float,
    min_component_size: int,
    morphology_kernel: int,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[-1] != 1:
        raise ValueError(f"Expected [N,H,W,1] probabilities, got {probs.shape}")
    masks = []
    for probability in probs[..., 0]:
        raw = (probability >= threshold).astype(np.uint8)
        masks.append(apply_postprocessing(raw, min_component_size, morphology_kernel)[..., None])
    return np.stack(masks).astype(np.float32)


def select_postprocessing(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Iterable[float],
    min_component_sizes: Iterable[int],
    morphology_kernels: Iterable[int],
) -> dict[str, Any]:
    """Select parameters using validation predictions only and report raw plus selected metrics."""

    thresholds = list(thresholds)
    sizes = list(min_component_sizes)
    kernels = list(morphology_kernels)
    if not thresholds or not sizes or not kernels:
        raise ValueError("All parameter grids must be non-empty")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be in [0, 1]")
    if any(size < 0 for size in sizes):
        raise ValueError("min_component_sizes must be non-negative")
    if any(kernel < 0 or (kernel > 0 and kernel % 2 == 0) for kernel in kernels):
        raise ValueError("morphology_kernels must be zero or positive odd integers")

    raw_threshold = 0.5
    raw_metrics = evaluate_predictions(y_true, probabilities, threshold=raw_threshold)
    raw_params = {"threshold": float(raw_threshold), "min_component_size": 0, "morphology_kernel": 0}
    candidates: list[dict[str, Any]] = []
    raw_ranking = (
        float(raw_metrics["macro_dice"]),
        float(raw_metrics["macro_iou"]),
        0,
        0,
        0.0,
    )
    best: dict[str, Any] = {
        **raw_params,
        "_ranking": raw_ranking,
        "metrics": raw_metrics,
    }
    for threshold, size, kernel in product(thresholds, sizes, kernels):
        processed = _postprocess_batch(probabilities, threshold, size, kernel)
        metrics = evaluate_predictions(y_true, processed, threshold=0.5)
        candidate = {
            "threshold": float(threshold),
            "min_component_size": int(size),
            "morphology_kernel": int(kernel),
            "macro_dice": metrics["macro_dice"],
            "macro_iou": metrics["macro_iou"],
        }
        candidates.append(candidate)
        ranking = (
            float(candidate["macro_dice"]),
            float(candidate["macro_iou"]),
            -int(candidate["min_component_size"]),
            -int(candidate["morphology_kernel"]),
            -abs(float(candidate["threshold"]) - 0.5),
        )
        if ranking > best["_ranking"]:
            best = {**candidate, "_ranking": ranking, "metrics": metrics}
    selected_metrics = best.pop("metrics")
    best.pop("_ranking")
    selected_params = {
        "threshold": best["threshold"],
        "min_component_size": best["min_component_size"],
        "morphology_kernel": best["morphology_kernel"],
    }
    selected_improves_raw = (
        float(selected_metrics["macro_dice"]),
        float(selected_metrics["macro_iou"]),
    ) > (
        float(raw_metrics["macro_dice"]),
        float(raw_metrics["macro_iou"]),
    )
    return {
        "selection_source": "validation_predictions_only",
        "raw_params": raw_params,
        "raw_metrics": raw_metrics,
        "selected_params": selected_params,
        "selected_metrics": selected_metrics,
        "selected_improves_raw": selected_improves_raw,
        "candidates": candidates,
    }
