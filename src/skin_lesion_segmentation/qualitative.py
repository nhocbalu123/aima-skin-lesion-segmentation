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
