"""Fixed-rule qualitative validation grids."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .metrics import per_image_thresholded_dice
from .postprocessing import apply_postprocessing


def representative_indices(scores: Sequence[float], count: int = 5) -> list[int]:
    """Choose evenly spaced score quantiles, including weak and strong cases."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional sequence")
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(int(count), values.size)
    ordered = np.argsort(values, kind="stable")
    positions = np.linspace(0, values.size - 1, num=count)
    selected_positions = np.rint(positions).astype(int)
    return [int(ordered[position]) for position in selected_positions]


def save_qualitative_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path | str,
    *,
    threshold: float,
    min_component_size: int,
    morphology_kernel: int,
    sample_ids: Sequence[str] | None = None,
    count: int = 5,
) -> list[str]:
    """Save input, truth, probability, raw mask, and processed mask columns.

    Cases are selected by evenly spaced validation-Dice quantiles rather than
    manual cherry-picking.
    """

    images = np.asarray(images)
    truth = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if images.ndim != 4 or truth.ndim != 4 or probs.ndim != 4:
        raise ValueError("images, y_true, and probabilities must be rank-4 batches")
    if not (len(images) == len(truth) == len(probs)):
        raise ValueError("Batch lengths do not match")
    if sample_ids is None:
        sample_ids = [str(index) for index in range(len(images))]
    if len(sample_ids) != len(images):
        raise ValueError("sample_ids length does not match batch")

    scores = per_image_thresholded_dice(truth, probs, threshold)
    indices = representative_indices(scores.tolist(), count=count)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(indices), 5, figsize=(15, 3 * len(indices)), squeeze=False)
    titles = ["Input", "Ground truth", "Probability", "Raw mask", "Post-processed"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    selected_ids: list[str] = []
    for row, index in enumerate(indices):
        image = images[index]
        gt = truth[index, ..., 0]
        probability = probs[index, ..., 0]
        raw = (probability >= threshold).astype(np.uint8)
        processed = apply_postprocessing(raw, min_component_size, morphology_kernel)
        panels = [image, gt, probability, raw, processed]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, cmap=None if column == 0 else "gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        selected_id = str(sample_ids[index])
        selected_ids.append(selected_id)
        axes[row, 0].set_ylabel(f"{selected_id}\nDice={scores[index]:.3f}")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return selected_ids


def _error_overlay(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*truth.shape, 3), dtype=np.float32)
    overlay[..., 1] = truth & prediction
    overlay[..., 0] = ~truth & prediction
    overlay[..., 2] = truth & ~prediction
    return overlay


def save_comparison_failure_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    unet_probabilities: np.ndarray,
    attention_probabilities: np.ndarray,
    sample_ids: Sequence[str],
    selected_ids: Sequence[str],
    output_path: Path | str,
    *,
    threshold: float = 0.5,
) -> list[str]:
    """Save paired predictions and FP/FN overlays for fixed selected cases."""

    image_batch = np.asarray(images, dtype=np.float32)
    truth_batch = np.asarray(y_true) > 0.5
    unet_batch = np.asarray(unet_probabilities) >= threshold
    attention_batch = np.asarray(attention_probabilities) >= threshold
    if (
        image_batch.ndim != 4
        or image_batch.shape[-1] != 3
        or truth_batch.ndim != 4
        or truth_batch.shape[-1] != 1
        or unet_batch.shape != truth_batch.shape
        or attention_batch.shape != truth_batch.shape
    ):
        raise ValueError("Comparison grid arrays have incompatible shapes")
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(image_batch):
        raise ValueError("sample_ids length does not match comparison batches")
    index_by_id = {sample_id: index for index, sample_id in enumerate(ids)}
    ordered: list[str] = []
    for sample_id in selected_ids:
        value = str(sample_id)
        if value not in index_by_id:
            raise ValueError(f"Unknown selected sample ID '{value}'")
        if value not in ordered:
            ordered.append(value)
    if not ordered:
        raise ValueError("selected_ids must not be empty")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(ordered),
        6,
        figsize=(18, 3 * len(ordered)),
        squeeze=False,
    )
    titles = [
        "Image",
        "Ground truth",
        "U-Net prediction",
        "U-Net error",
        "Attention prediction",
        "Attention error",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, sample_id in enumerate(ordered):
        index = index_by_id[sample_id]
        truth = truth_batch[index, ..., 0]
        unet = unet_batch[index, ..., 0]
        attention = attention_batch[index, ..., 0]
        panels = [
            image_batch[index],
            truth,
            unet,
            _error_overlay(truth, unet),
            attention,
            _error_overlay(truth, attention),
        ]
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(
                panel,
                cmap="gray" if column in {1, 2, 4} else None,
                vmin=0,
                vmax=1,
            )
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(sample_id)
    figure.suptitle("Error overlay: green=TP, red=FP, blue=FN", y=1.0)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return ordered
