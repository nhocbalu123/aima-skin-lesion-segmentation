"""Sealed, provenance-checked U-Net versus Attention U-Net comparison."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    paired_bootstrap_interval,
    per_image_comparison,
    summarize_slices,
)
from .artifacts import RunArtifacts, sha256_file
from .config import ExperimentConfig
from .data import SamplePair
from .evaluation import load_split_manifest, pairs_from_manifest
from .loading import PathBatchSequence
from .metrics import evaluate_predictions
from .model import load_model
from .qualitative import save_comparison_failure_grid
from .reproducibility import collect_environment, collect_git_provenance
from .robustness import PERTURBATION_CONDITIONS, apply_perturbation
from .schemas import ResultsSchemaError, validate_run_summary
from .splitting import manifest_sha256, validate_manifest_against_pairs


class ComparisonIntegrityError(ValueError):
    """Raised before test inference when experiment evidence is incompatible."""


def _software_signature(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python_version": environment.get("python_version"),
        "packages": environment.get("packages"),
    }


def _hardware_signature(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tensorflow_devices": environment.get("tensorflow_devices"),
        "tensorflow_device_details": environment.get("tensorflow_device_details"),
    }


def _comparable_config(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("model", None)
    result.pop("output_dir", None)
    return result


def _check_checkpoint(
    summary: Mapping[str, Any],
    path: Path | str,
    *,
    label: str,
) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise ComparisonIntegrityError(f"{label} checkpoint does not exist")
    evidence = summary["checkpoint"]
    if not isinstance(evidence, Mapping):
        raise ComparisonIntegrityError(f"{label} checkpoint evidence is invalid")
    if (
        int(evidence["size_bytes"]) != checkpoint.stat().st_size
        or str(evidence["sha256"]) != sha256_file(checkpoint)
    ):
        raise ComparisonIntegrityError(
            f"{label} checkpoint content does not match its run summary"
        )


def validate_run_pair(
    config: ExperimentConfig,
    manifest_rows: Sequence[Mapping[str, object]],
    unet_summary: Mapping[str, Any],
    attention_summary: Mapping[str, Any],
    unet_checkpoint: Path | str,
    attention_checkpoint: Path | str,
    current_environment: Mapping[str, Any],
    current_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject incompatible run evidence before internal-test inference."""

    config.validate()
    if config.n_tta != 1:
        raise ComparisonIntegrityError(
            "Primary comparison requires no TTA (n_tta must equal 1)"
        )
    try:
        unet = validate_run_summary(unet_summary)
        attention = validate_run_summary(attention_summary)
    except ResultsSchemaError as error:
        raise ComparisonIntegrityError(str(error)) from error
    if unet["model_name"] != "unet" or attention["model_name"] != "attention_unet":
        raise ComparisonIntegrityError(
            "Run roles must be plain U-Net then Attention U-Net"
        )
    if unet["seed"] != attention["seed"]:
        raise ComparisonIntegrityError("Runs use different training seeds")
    actual_manifest_hash = manifest_sha256(manifest_rows)
    if (
        unet["manifest_sha256"] != attention["manifest_sha256"]
        or unet["manifest_sha256"] != actual_manifest_hash
    ):
        raise ComparisonIntegrityError("Runs do not share the supplied manifest")
    if _comparable_config(unet["effective_config"]) != _comparable_config(
        attention["effective_config"]
    ):
        raise ComparisonIntegrityError(
            "Runs use different effective training configurations"
        )
    if unet["source"]["commit"] != attention["source"]["commit"]:
        raise ComparisonIntegrityError("Runs use different source commits")
    if _software_signature(unet["environment"]) != _software_signature(
        attention["environment"]
    ):
        raise ComparisonIntegrityError("Runs use different software environments")
    if _hardware_signature(unet["environment"]) != _hardware_signature(
        attention["environment"]
    ):
        raise ComparisonIntegrityError("Runs use different hardware environments")
    if _comparable_config(config.to_dict()) != _comparable_config(
        unet["effective_config"]
    ):
        raise ComparisonIntegrityError(
            "The evaluation config does not match the training config"
        )
    if current_source.get("commit") != unet["source"]["commit"]:
        raise ComparisonIntegrityError(
            "The current source commit does not match the training runs"
        )
    if _software_signature(current_environment) != _software_signature(
        unet["environment"]
    ):
        raise ComparisonIntegrityError(
            "The current software environment does not match the training runs"
        )
    if _hardware_signature(current_environment) != _hardware_signature(
        unet["environment"]
    ):
        raise ComparisonIntegrityError(
            "The current hardware environment does not match the training runs"
        )
    _check_checkpoint(unet, unet_checkpoint, label="U-Net")
    _check_checkpoint(attention, attention_checkpoint, label="Attention U-Net")
    return unet, attention


def _read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonIntegrityError(f"Expected a JSON object in {path}")
    return value


def _load_split_arrays(
    pairs: Sequence[SamplePair],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sequence = PathBatchSequence(
        pairs,
        (config.image_height, config.image_width),
        config.batch_size,
        shuffle=False,
        augmentation=None,
    )
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch_index in range(len(sequence)):
        image_batch, mask_batch = sequence[batch_index]
        images.append(image_batch)
        masks.append(mask_batch)
    return (
        np.concatenate(images, axis=0),
        np.concatenate(masks, axis=0),
        sequence.ordered_sample_ids,
    )


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows], dtype=np.float64))

    return {
        "n_images": len(rows),
        "unet_macro_dice": mean("unet_dice"),
        "attention_macro_dice": mean("attention_dice"),
        "dice_difference_mean": mean("dice_difference"),
        "unet_macro_iou": mean("unet_iou"),
        "attention_macro_iou": mean("attention_iou"),
        "iou_difference_mean": mean("iou_difference"),
    }


def _failure_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 3,
) -> dict[str, list[str]]:
    materialised = list(rows)

    def lowest(field: str) -> list[str]:
        ordered = sorted(
            materialised,
            key=lambda row: (float(row[field]), str(row["sample_id"])),
        )
        return [str(row["sample_id"]) for row in ordered[:count]]

    def highest(fields: tuple[str, str]) -> list[str]:
        ordered = sorted(
            materialised,
            key=lambda row: (
                -max(float(row[fields[0]]), float(row[fields[1]])),
                str(row["sample_id"]),
            ),
        )
        return [str(row["sample_id"]) for row in ordered[:count]]

    return {
        "worst_unet": lowest("unet_dice"),
        "worst_attention_unet": lowest("attention_dice"),
        "representative_false_positive": highest(
            (
                "unet_false_positive_fraction",
                "attention_false_positive_fraction",
            )
        ),
        "representative_false_negative": highest(
            (
                "unet_false_negative_fraction",
                "attention_false_negative_fraction",
            )
        ),
    }


def compare_runs(
    config: ExperimentConfig,
    manifest_path: Path | str,
    unet_run_summary_path: Path | str,
    attention_run_summary_path: Path | str,
    unet_checkpoint: Path | str,
    attention_checkpoint: Path | str,
    output_dir: Path | str,
    *,
    bootstrap_seed: int = 20260730,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Run one matched-seed internal-test comparison after integrity checks."""

    started_at = time.perf_counter()
    rows = load_split_manifest(manifest_path)
    mounted_pairs = [
        SamplePair(
            str(row["sample_id"]),
            Path(str(row["image_path"])),
            Path(str(row["mask_path"])),
        )
        for row in rows
    ]
    rows = validate_manifest_against_pairs(rows, mounted_pairs)
    unet_raw = _read_json(unet_run_summary_path)
    attention_raw = _read_json(attention_run_summary_path)
    current_environment = collect_environment()
    current_source = collect_git_provenance(Path(__file__).resolve().parents[2])
    unet_summary, attention_summary = validate_run_pair(
        config,
        rows,
        unet_raw,
        attention_raw,
        unet_checkpoint,
        attention_checkpoint,
        current_environment,
        current_source,
    )
    internal_pairs = pairs_from_manifest(rows, "internal_test")
    images, masks, sample_ids = _load_split_arrays(internal_pairs, config)
    unet_model = load_model(unet_checkpoint, compile=False)
    attention_model = load_model(attention_checkpoint, compile=False)
    if int(unet_model.count_params()) != unet_summary["parameter_count"]:
        raise ComparisonIntegrityError(
            "Loaded U-Net parameter count does not match its run summary"
        )
    if int(attention_model.count_params()) != attention_summary["parameter_count"]:
        raise ComparisonIntegrityError(
            "Loaded Attention U-Net parameter count does not match its run summary"
        )

    robustness: dict[str, Any] = {}
    clean_unet: np.ndarray | None = None
    clean_attention: np.ndarray | None = None
    for condition in PERTURBATION_CONDITIONS:
        perturbed = apply_perturbation(images, condition)
        unet_probabilities = np.asarray(
            unet_model.predict(
                perturbed,
                batch_size=config.batch_size,
                verbose=0,
            ),
            dtype=np.float32,
        )
        attention_probabilities = np.asarray(
            attention_model.predict(
                perturbed,
                batch_size=config.batch_size,
                verbose=0,
            ),
            dtype=np.float32,
        )
        unet_metrics = evaluate_predictions(masks, unet_probabilities, threshold=0.5)
        attention_metrics = evaluate_predictions(
            masks,
            attention_probabilities,
            threshold=0.5,
        )
        robustness[condition] = {
            "n_images": int(unet_metrics["n_images"]),
            "unet_macro_dice": float(unet_metrics["macro_dice"]),
            "attention_macro_dice": float(attention_metrics["macro_dice"]),
            "dice_difference": float(
                attention_metrics["macro_dice"] - unet_metrics["macro_dice"]
            ),
            "unet_macro_iou": float(unet_metrics["macro_iou"]),
            "attention_macro_iou": float(attention_metrics["macro_iou"]),
            "iou_difference": float(
                attention_metrics["macro_iou"] - unet_metrics["macro_iou"]
            ),
        }
        if condition == "clean":
            clean_unet = unet_probabilities
            clean_attention = attention_probabilities
    if clean_unet is None or clean_attention is None:
        raise AssertionError("Clean robustness condition was not evaluated")

    per_image_rows = per_image_comparison(
        sample_ids,
        masks,
        clean_unet,
        clean_attention,
        threshold=0.5,
    )
    dice_interval = paired_bootstrap_interval(
        [row["unet_dice"] for row in per_image_rows],
        [row["attention_dice"] for row in per_image_rows],
        seed=bootstrap_seed,
        n_resamples=n_resamples,
    )
    iou_interval = paired_bootstrap_interval(
        [row["unet_iou"] for row in per_image_rows],
        [row["attention_iou"] for row in per_image_rows],
        seed=bootstrap_seed,
        n_resamples=n_resamples,
    )
    slices = summarize_slices(per_image_rows)
    failures = _failure_cases(per_image_rows)
    selected_ids: list[str] = []
    for category in failures.values():
        for sample_id in category:
            if sample_id not in selected_ids:
                selected_ids.append(sample_id)

    artifacts = RunArtifacts(output_dir)
    artifacts.write_csv("per_image_metrics.csv", per_image_rows)
    artifacts.write_json("slice_summary.json", slices)
    artifacts.write_json(
        "robustness_summary.json",
        {
            "description": "predeclared synthetic robustness checks; not clinical validation",
            "conditions": robustness,
        },
    )
    artifacts.write_json("failure_cases.json", failures)
    artifacts.write_json("unet_run_summary.json", unet_summary)
    artifacts.write_json(
        "attention_unet_run_summary.json",
        attention_summary,
    )
    save_comparison_failure_grid(
        images,
        masks,
        clean_unet,
        clean_attention,
        sample_ids,
        selected_ids,
        artifacts.path("qualitative_failures.png"),
        threshold=0.5,
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "seed": unet_summary["seed"],
        "manifest_sha256": manifest_sha256(rows),
        "source_commit": current_source["commit"],
        "primary_policy": {
            "threshold": 0.5,
            "tta": "none",
            "postprocessing": "none",
        },
        "models": {
            "unet": {"parameter_count": unet_summary["parameter_count"]},
            "attention_unet": {
                "parameter_count": attention_summary["parameter_count"]
            },
        },
        "aggregate_metrics": _aggregate_rows(per_image_rows),
        "paired_bootstrap": {
            "dice": dice_interval,
            "iou": iou_interval,
        },
        "runtime_seconds": float(time.perf_counter() - started_at),
        "environment": current_environment,
        "artifacts": {
            "per_image_metrics": "per_image_metrics.csv",
            "slices": "slice_summary.json",
            "robustness": "robustness_summary.json",
            "failure_cases": "failure_cases.json",
            "qualitative": "qualitative_failures.png",
            "unet_run_summary": "unet_run_summary.json",
            "attention_unet_run_summary": "attention_unet_run_summary.json",
        },
    }
    artifacts.write_json("comparison_summary.json", summary)
    return summary
