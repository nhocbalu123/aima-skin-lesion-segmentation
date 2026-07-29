"""Training orchestration with path streaming and reproducibility artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import RunArtifacts
from .augmentations import build_train_augmentation
from .config import ExperimentConfig
from .data import SamplePair, build_pair_manifest
from .evaluation import evaluate_checkpoint
from .loading import PathBatchSequence
from .losses import combined_segmentation_loss
from .metrics import soft_dice_batch_global_tf
from .model import build_attention_unet
from .reproducibility import collect_environment, set_global_determinism
from .splitting import build_split_manifest


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


def prepare_manifests(config: ExperimentConfig) -> tuple[list[SamplePair], list[dict[str, object]]]:
    """Validate pairing, audit duplicates, split paths, and save manifests."""

    config.validate()
    artifacts = RunArtifacts(config.output_dir)
    pairs = build_pair_manifest(config.image_dir, config.mask_dir, config.mask_suffix)
    groups = load_group_mapping(config.group_mapping_path)
    split_rows = build_split_manifest(
        pairs,
        config.validation_fraction,
        config.seed,
        group_mapping=groups,
        output_path=artifacts.path("split_manifest.json"),
    )
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


def _pairs_by_split(
    pairs: list[SamplePair],
    rows: list[Mapping[str, object]],
) -> tuple[list[SamplePair], list[SamplePair]]:
    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in rows}
    train = [pair for pair in pairs if split_by_id[pair.sample_id] == "train"]
    validation = [pair for pair in pairs if split_by_id[pair.sample_id] == "validation"]
    if not train or not validation:
        raise ValueError("Both train and validation splits must be non-empty")
    return train, validation


def train(config: ExperimentConfig) -> dict[str, Any]:
    """Train, reload the selected checkpoint, and evaluate it offline.

    Checkpoint selection uses validation loss. Keras loss values may include L2
    regularisation penalties. With ``mixed_float16``, ``Model.fit`` uses the
    framework-supported automatic loss-scaling path; no manual scaling is added.
    """

    import tensorflow as tf

    config.validate()
    artifacts = RunArtifacts(config.output_dir)
    determinism = set_global_determinism(config.seed)
    artifacts.write_json("effective_config.json", config.to_dict())
    artifacts.write_json("random_seed.json", determinism)
    artifacts.write_json("environment.json", collect_environment())
    pairs, split_rows = prepare_manifests(config)
    train_pairs, validation_pairs = _pairs_by_split(pairs, split_rows)

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
    model = build_attention_unet(
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
    summary: dict[str, Any] = {
        "epochs_completed": len(history.epoch),
        "checkpoint": checkpoint_record,
        "checkpoint_selection": "minimum validation loss; final metrics computed offline after reload",
        "evaluation_artifact": str(artifacts.path("final_validation_metrics.json")),
    }
    artifacts.write_json("run_summary.json", summary)
    return {**summary, "evaluation": evaluation}
