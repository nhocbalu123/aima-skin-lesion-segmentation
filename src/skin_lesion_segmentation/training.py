"""Training orchestration with path streaming and reproducibility artifacts."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import RunArtifacts
from .augmentations import build_train_augmentation
from .config import ExperimentConfig
from .data import SamplePair, build_pair_manifest
from .evaluation import evaluate_checkpoint, load_split_manifest
from .loading import PathBatchSequence
from .losses import combined_segmentation_loss
from .metrics import soft_dice_batch_global_tf
from .model import build_model
from .reproducibility import (
    collect_environment,
    collect_git_provenance,
    command_record,
    set_global_determinism,
)
from .schemas import validate_run_summary
from .splitting import (
    build_comparison_manifest,
    build_split_manifest,
    manifest_sha256,
    validate_manifest_against_pairs,
)


def load_group_mapping(path: Path | str | None) -> dict[str, str] | None:
    """Load explicit sample-to-group metadata from JSON or CSV.

    The file must provide actual metadata; no filename-derived patient IDs are
    inferred. CSV columns are ``sample_id`` and ``group_id``.
    """

    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Group JSON must be an object mapping sample_id to group_id")
        mapping: dict[str, str] = {}
        for key, group in value.items():
            sample_id = str(key)
            if group is None or not str(group).strip():
                raise ValueError(f"blank group ID for sample '{sample_id}'")
            mapping[sample_id] = str(group).strip()
        return mapping
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"sample_id", "group_id"}.issubset(reader.fieldnames):
                raise ValueError("Group CSV must contain sample_id and group_id columns")
            mapping: dict[str, str] = {}
            for row in reader:
                sample_id = str(row["sample_id"])
                group = row["group_id"]
                if group is None or not str(group).strip():
                    raise ValueError(f"blank group ID for sample '{sample_id}'")
                if sample_id in mapping:
                    raise ValueError(f"Duplicate group mapping for sample ID '{sample_id}'")
                mapping[sample_id] = str(group).strip()
            return mapping
    raise ValueError("group_mapping_path must be a .json or .csv file")


def prepare_manifests(
    config: ExperimentConfig,
    *,
    split_manifest_path: Path | str | None = None,
) -> tuple[list[SamplePair], list[dict[str, object]]]:
    """Validate pairing, audit duplicates, split paths, and save manifests."""

    config.validate()
    artifacts = RunArtifacts(config.output_dir)
    pairs = build_pair_manifest(config.image_dir, config.mask_dir, config.mask_suffix)
    if split_manifest_path is None:
        groups = load_group_mapping(config.group_mapping_path)
        split_rows = build_split_manifest(
            pairs,
            config.validation_fraction,
            config.seed,
            group_mapping=groups,
            output_path=artifacts.path("split_manifest.json"),
        )
    else:
        supplied_rows = load_split_manifest(split_manifest_path)
        split_rows = validate_manifest_against_pairs(supplied_rows, pairs)
        artifacts.write_json("split_manifest.json", split_rows)
    artifacts.write_csv(
        "pair_manifest.csv",
        [
            {
                "sample_id": pair.sample_id,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
            }
            for pair in pairs
        ],
    )
    artifacts.write_csv("split_manifest.csv", split_rows)
    return pairs, split_rows


def prepare_comparison_manifest(
    config: ExperimentConfig,
    output_manifest: Path | str,
) -> tuple[list[SamplePair], list[dict[str, object]]]:
    """Write the single immutable three-way manifest used by every run."""

    config.validate()
    pairs = build_pair_manifest(config.image_dir, config.mask_dir, config.mask_suffix)
    groups = load_group_mapping(config.group_mapping_path)
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=config.validation_fraction,
        internal_test_fraction=config.internal_test_fraction,
        split_seed=config.split_seed,
        group_mapping=groups,
        output_path=output_manifest,
    )
    return pairs, rows


def training_pairs_from_manifest(
    pairs: list[SamplePair],
    rows: list[Mapping[str, object]],
) -> tuple[list[SamplePair], list[SamplePair]]:
    """Return only fitting and checkpoint-selection samples.

    Rows assigned to ``internal_test`` are deliberately ignored so training
    code cannot construct a test sequence accidentally.
    """

    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in rows}
    pair_ids = {pair.sample_id for pair in pairs}
    if pair_ids != set(split_by_id):
        raise ValueError("Pair IDs and split-manifest IDs must match exactly")
    train = [pair for pair in pairs if split_by_id[pair.sample_id] == "train"]
    validation = [pair for pair in pairs if split_by_id[pair.sample_id] == "validation"]
    if not train or not validation:
        raise ValueError("Both train and validation splits must be non-empty")
    return train, validation


def train(
    config: ExperimentConfig,
    *,
    split_manifest_path: Path | str | None = None,
    command: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Train, reload the selected checkpoint, and evaluate it offline.

    Checkpoint selection uses validation loss. Keras loss values may include L2
    regularisation penalties. With ``mixed_float16``, ``Model.fit`` uses the
    framework-supported automatic loss-scaling path; no manual scaling is added.
    """

    import tensorflow as tf

    started_at = time.perf_counter()
    config.validate()
    artifacts = RunArtifacts(config.output_dir)
    determinism = set_global_determinism(config.seed)
    effective_config = config.to_dict()
    environment = collect_environment()
    artifacts.write_json("effective_config.json", effective_config)
    artifacts.write_json("random_seed.json", determinism)
    artifacts.write_json("environment.json", environment)
    pairs, split_rows = prepare_manifests(
        config,
        split_manifest_path=split_manifest_path,
    )
    train_pairs, validation_pairs = training_pairs_from_manifest(pairs, split_rows)

    train_sequence = PathBatchSequence(
        train_pairs,
        (config.image_height, config.image_width),
        config.batch_size,
        shuffle=True,
        seed=config.seed,
        augmentation=build_train_augmentation(config.seed),
    )
    validation_sequence = PathBatchSequence(
        validation_pairs,
        (config.image_height, config.image_width),
        config.batch_size,
        shuffle=False,
        augmentation=None,
    )
    model = build_model(
        config.model,
        (config.image_height, config.image_width, 3),
        base_filters=config.base_filters,
        l2_coefficient=config.l2_coefficient,
        mixed_precision=config.mixed_precision,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=combined_segmentation_loss,
        metrics=[soft_dice_batch_global_tf],
    )
    checkpoint_path = artifacts.path("best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(artifacts.path("training_history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if not checkpoint_path.exists():
        raise RuntimeError("Training completed without producing the selected .keras checkpoint")
    checkpoint_record = artifacts.record_checkpoint(checkpoint_path)
    evaluation = evaluate_checkpoint(
        config,
        checkpoint_path,
        artifacts.path("split_manifest.json"),
        artifact_dir=artifacts.root,
    )
    validation_metrics = evaluation["raw_threshold_0_5"]
    validation_losses = [
        float(value)
        for value in history.history.get("val_loss", [])
    ]
    if not validation_losses:
        raise RuntimeError("Training history did not record validation loss")
    best_index = min(
        range(len(validation_losses)),
        key=validation_losses.__getitem__,
    )
    summary = validate_run_summary({
        "schema_version": 1,
        "model_name": config.model,
        "seed": config.seed,
        "manifest_sha256": manifest_sha256(split_rows),
        "parameter_count": int(model.count_params()),
        "effective_config": effective_config,
        "checkpoint_selection": {
            "monitor": "val_loss",
            "mode": "min",
            "rule": "minimum validation loss",
            "early_stopping_patience": config.early_stopping_patience,
        },
        "checkpoint": checkpoint_record,
        "epochs_completed": len(history.epoch),
        "best_epoch": best_index + 1,
        "best_validation_loss": validation_losses[best_index],
        "validation_metrics": {
            "threshold": float(validation_metrics["threshold"]),
            "macro_dice": float(validation_metrics["macro_dice"]),
            "macro_iou": float(validation_metrics["macro_iou"]),
            "n_images": int(validation_metrics["n_images"]),
        },
        "runtime_seconds": float(time.perf_counter() - started_at),
        "command": command_record(command or sys.argv),
        "source": collect_git_provenance(Path(__file__).resolve().parents[2]),
        "environment": environment,
    })
    artifacts.write_json("run_summary.json", summary)
    return {"summary": summary, "evaluation": evaluation}
