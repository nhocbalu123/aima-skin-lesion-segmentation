"""Test inference and validated submission generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import RunArtifacts
from .config import ExperimentConfig
from .data import SUPPORTED_IMAGE_EXTENSIONS
from .loading import PathImageSequence, restore_binary_mask_to_image_shape
from .model import load_model
from .postprocessing import apply_postprocessing
from .submission import build_submission
from .tta import predict_with_tta


def discover_test_images(directory: Path | str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda path: (path.stem.casefold(), path.name),
    )
    if not paths:
        raise ValueError(f"No supported test images found in {root}")
    return paths


def generate_submission(
    config: ExperimentConfig,
    checkpoint_path: Path | str,
    postprocessing_path: Path | str,
) -> dict[str, Any]:
    """Reload a checkpoint, infer test masks, and validate against the sample submission."""

    config.validate()
    if config.sample_submission_path is None:
        raise ValueError("sample_submission_path is required for validated submission generation")
    if not config.rle_order_confirmed:
        raise ValueError(
            "RLE order must be confirmed from the official competition contract before submission generation"
        )
    params = json.loads(Path(postprocessing_path).read_text(encoding="utf-8"))
    required = {"threshold", "min_component_size", "morphology_kernel"}
    if set(params) != required:
        raise ValueError(f"Post-processing JSON must contain exactly {sorted(required)}")

    artifacts = RunArtifacts(config.output_dir)
    model = load_model(checkpoint_path, compile=False)
    test_paths = discover_test_images(config.test_image_dir)
    sequence = PathImageSequence(
        test_paths,
        (config.image_height, config.image_width),
        config.batch_size,
    )
    predictions: list[np.ndarray] = []
    for batch_index in range(len(sequence)):
        images = sequence[batch_index]
        probabilities = predict_with_tta(
            lambda transformed: model.predict(transformed, verbose=0),
            images,
            n_augmentations=config.n_tta,
        )
        start = batch_index * config.batch_size
        batch_paths = test_paths[start : start + len(probabilities)]
        for probability, image_path in zip(probabilities[..., 0], batch_paths, strict=True):
            raw = probability >= float(params["threshold"])
            model_space_mask = apply_postprocessing(
                raw,
                int(params["min_component_size"]),
                int(params["morphology_kernel"]),
            )
            predictions.append(
                restore_binary_mask_to_image_shape(model_space_mask, image_path)
            )
    sample_submission = pd.read_csv(config.sample_submission_path, keep_default_na=False)
    submission, summary = build_submission(
        test_paths,
        predictions,
        sample_submission=sample_submission,
        id_column=config.submission_id_column,
        mask_column=config.submission_mask_column,
        submission_id_suffix=config.submission_id_suffix,
        rle_order=config.rle_order,
    )
    output_path = artifacts.path("submission.csv")
    submission.to_csv(output_path, index=False)
    result = {
        **summary,
        "path": str(output_path),
        "checkpoint": str(Path(checkpoint_path)),
        "postprocessing": params,
        "rle_order_confirmed": True,
    }
    artifacts.write_json("submission_validation_summary.json", result)
    return result
