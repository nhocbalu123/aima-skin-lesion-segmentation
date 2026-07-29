"""Checkpoint evaluation, validation-only tuning, and qualitative artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import RunArtifacts
from .config import ExperimentConfig
from .data import SamplePair
from .loading import PathBatchSequence
from .metrics import evaluate_predictions
from .model import load_model
from .postprocessing import select_postprocessing
from .qualitative import save_qualitative_grid
from .splitting import audit_split_integrity
from .tta import predict_with_tta


def pairs_from_manifest(
    rows: Sequence[Mapping[str, object]],
    split: str,
) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    for row in rows:
        if str(row.get("split")) != split:
            continue
        pairs.append(
            SamplePair(
                str(row["sample_id"]),
                Path(str(row["image_path"])),
                Path(str(row["mask_path"])),
            )
        )
    if not pairs:
        raise ValueError(f"No samples found for split '{split}'")
    return pairs


def load_split_manifest(path: Path | str) -> list[dict[str, object]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Split manifest must be a JSON list of objects")
    required = {"sample_id", "image_path", "mask_path", "split", "group_id", "image_sha256"}
    for index, row in enumerate(value):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Split manifest row {index} is missing required fields: {', '.join(missing)}"
            )
    audit_split_integrity(value)
    return value


def predict_validation(
    model: Any,
    pairs: Sequence[SamplePair],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Predict a validation set batch-by-batch while retaining evaluation inputs."""

    sequence = PathBatchSequence(
        pairs,
        (config.image_height, config.image_width),
        config.batch_size,
        shuffle=False,
        augmentation=None,
    )
    images_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    ordered_ids = sequence.ordered_sample_ids
    offset = 0
    for batch_index in range(len(sequence)):
        images, masks = sequence[batch_index]
        probabilities = predict_with_tta(
            lambda transformed: model.predict(transformed, verbose=0),
            images,
            n_augmentations=config.n_tta,
        )
        images_all.append(images)
        masks_all.append(masks)
        probabilities_all.append(np.asarray(probabilities, dtype=np.float32))
        sample_ids.extend(ordered_ids[offset : offset + len(images)])
        offset += len(images)
    return (
        np.concatenate(images_all, axis=0),
        np.concatenate(masks_all, axis=0),
        np.concatenate(probabilities_all, axis=0),
        sample_ids,
    )


def evaluate_saved_predictions(
    predictions_path: Path | str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Reload and evaluate a saved validation-prediction NPZ artifact."""

    source = Path(predictions_path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"sample_ids", "y_true", "probabilities"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                "Saved predictions are missing required arrays: " + ", ".join(missing)
            )
        sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
        y_true = np.asarray(archive["y_true"])
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    if len(sample_ids) != len(y_true) or len(sample_ids) != len(probabilities):
        raise ValueError("Saved prediction array lengths do not match sample_ids")
    canonical_ids: set[str] = set()
    for sample_id in sample_ids:
        canonical = sample_id.strip().casefold()
        if not canonical:
            raise ValueError("Saved predictions contain a blank sample ID")
        if canonical in canonical_ids:
            raise ValueError("Saved predictions contain duplicate sample IDs")
        canonical_ids.add(canonical)
    metrics = evaluate_predictions(y_true, probabilities, threshold=threshold)
    return {
        **metrics,
        "sample_ids": sample_ids,
        "predictions_path": str(source),
    }


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path | str,
    split_manifest_path: Path | str,
    *,
    artifact_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Reload a selected checkpoint and run source-of-truth offline validation."""

    config.validate()
    artifacts = RunArtifacts(artifact_dir or config.output_dir)
    rows = load_split_manifest(split_manifest_path)
    validation_pairs = pairs_from_manifest(rows, "validation")
    model = load_model(checkpoint_path, compile=False)
    images, masks, probabilities, sample_ids = predict_validation(model, validation_pairs, config)

    raw_metrics = evaluate_predictions(masks, probabilities, threshold=0.5)
    selection = select_postprocessing(
        masks,
        probabilities,
        thresholds=config.threshold_grid,
        min_component_sizes=config.min_component_sizes,
        morphology_kernels=config.morphology_kernels,
    )
    selected_params = selection["selected_params"]
    selected_ids = save_qualitative_grid(
        images,
        masks,
        probabilities,
        artifacts.path("qualitative_validation.png"),
        threshold=float(selected_params["threshold"]),
        min_component_size=int(selected_params["min_component_size"]),
        morphology_kernel=int(selected_params["morphology_kernel"]),
        sample_ids=sample_ids,
        count=min(5, len(sample_ids)),
    )
    np.savez_compressed(
        artifacts.path("validation_predictions.npz"),
        sample_ids=np.asarray(sample_ids),
        y_true=masks.astype(np.uint8),
        probabilities=probabilities.astype(np.float32),
    )
    result: dict[str, Any] = {
        "checkpoint_reloaded": str(Path(checkpoint_path)),
        "n_validation_images": len(sample_ids),
        "raw_threshold_0_5": raw_metrics,
        "postprocessing_selection": selection,
        "qualitative_selection_rule": "evenly spaced validation-Dice quantiles including weak and strong cases",
        "qualitative_sample_ids": selected_ids,
    }
    artifacts.write_json("final_validation_metrics.json", result)
    artifacts.write_json("chosen_postprocessing.json", selected_params)
    return result
