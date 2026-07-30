from __future__ import annotations

from copy import deepcopy

import pytest

from skin_lesion_segmentation.schemas import ResultsSchemaError, validate_run_summary


def valid_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_name": "unet",
        "seed": 42,
        "manifest_sha256": "a" * 64,
        "parameter_count": 31_119,
        "effective_config": {
            "model": "unet",
            "image_height": 32,
            "image_width": 32,
            "batch_size": 2,
            "epochs": 1,
        },
        "checkpoint_selection": {
            "monitor": "val_loss",
            "mode": "min",
            "rule": "minimum validation loss",
            "early_stopping_patience": 8,
        },
        "checkpoint": {
            "path": "artifacts/unet-42/best_model.keras",
            "size_bytes": 100,
            "sha256": "b" * 64,
        },
        "epochs_completed": 1,
        "best_epoch": 1,
        "best_validation_loss": 0.5,
        "validation_metrics": {
            "threshold": 0.5,
            "macro_dice": 0.7,
            "macro_iou": 0.6,
            "n_images": 2,
        },
        "runtime_seconds": 3.2,
        "command": {
            "argv": ["python", "-m", "skin_lesion_segmentation.cli", "train"],
            "shell": "python -m skin_lesion_segmentation.cli train",
        },
        "source": {"commit": "c" * 40, "dirty": False},
        "environment": {
            "python_version": "3.11.15",
            "packages": {"tensorflow": "2.16.1"},
            "tensorflow_devices": ["/physical_device:GPU:0"],
            "tensorflow_device_details": [{"device_type": "GPU", "device_name": "T4"}],
        },
    }


def test_run_summary_schema_accepts_complete_training_evidence() -> None:
    summary = valid_summary()

    validated = validate_run_summary(summary)

    assert validated == summary
    assert validated is not summary


@pytest.mark.parametrize(
    "missing",
    [
        "model_name",
        "seed",
        "manifest_sha256",
        "parameter_count",
        "effective_config",
        "checkpoint_selection",
        "checkpoint",
        "validation_metrics",
        "runtime_seconds",
        "command",
        "source",
        "environment",
    ],
)
def test_run_summary_schema_rejects_missing_evidence(missing: str) -> None:
    summary = valid_summary()
    del summary[missing]

    with pytest.raises(ResultsSchemaError, match=missing):
        validate_run_summary(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "resunet"),
        ("seed", True),
        ("manifest_sha256", "not-a-digest"),
        ("parameter_count", 0),
        ("runtime_seconds", -1.0),
    ],
)
def test_run_summary_schema_rejects_invalid_core_values(
    field: str,
    value: object,
) -> None:
    summary = valid_summary()
    summary[field] = value

    with pytest.raises(ResultsSchemaError, match=field):
        validate_run_summary(summary)


def test_run_summary_schema_rejects_internal_test_results_from_training() -> None:
    summary = valid_summary()
    summary["internal_test_metrics"] = {"macro_dice": 0.99}

    with pytest.raises(ResultsSchemaError, match="internal-test"):
        validate_run_summary(summary)


def test_run_summary_schema_rejects_model_config_drift() -> None:
    summary = deepcopy(valid_summary())
    config = summary["effective_config"]
    assert isinstance(config, dict)
    config["model"] = "attention_unet"

    with pytest.raises(ResultsSchemaError, match="effective_config.model"):
        validate_run_summary(summary)
