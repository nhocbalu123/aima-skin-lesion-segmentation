from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from skin_lesion_segmentation.artifacts import sha256_file
from skin_lesion_segmentation.comparison import (
    ComparisonIntegrityError,
    validate_run_pair,
)
from skin_lesion_segmentation.config import ExperimentConfig
from skin_lesion_segmentation.splitting import manifest_sha256


def environment() -> dict[str, object]:
    return {
        "python_version": "3.11.15",
        "packages": {
            "tensorflow": "2.16.1",
            "keras": "3.0.5",
            "numpy": "1.26.4",
        },
        "tensorflow_devices": ["/physical_device:GPU:0"],
        "tensorflow_device_details": [
            {
                "device_name": "/physical_device:GPU:0",
                "device_type": "GPU",
                "details": {"device_name": "Tesla T4", "compute_capability": [7, 5]},
            }
        ],
    }


def manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "train",
            "image_path": "/data/train.png",
            "mask_path": "/data/train_mask.png",
            "split": "train",
            "group_id": None,
            "image_sha256": "1" * 64,
            "mask_sha256": "2" * 64,
        },
        {
            "sample_id": "validation",
            "image_path": "/data/validation.png",
            "mask_path": "/data/validation_mask.png",
            "split": "validation",
            "group_id": None,
            "image_sha256": "3" * 64,
            "mask_sha256": "4" * 64,
        },
        {
            "sample_id": "test",
            "image_path": "/data/test.png",
            "mask_path": "/data/test_mask.png",
            "split": "internal_test",
            "group_id": None,
            "image_sha256": "5" * 64,
            "mask_sha256": "6" * 64,
        },
    ]


def run_summary(
    config: ExperimentConfig,
    model_name: str,
    checkpoint: Path,
) -> dict[str, object]:
    effective = config.to_dict()
    effective["model"] = model_name
    effective["output_dir"] = f"artifacts/{model_name}-42"
    return {
        "schema_version": 1,
        "model_name": model_name,
        "seed": 42,
        "manifest_sha256": manifest_sha256(manifest_rows()),
        "parameter_count": 31_119 if model_name == "unet" else 31_508,
        "effective_config": effective,
        "checkpoint_selection": {
            "monitor": "val_loss",
            "mode": "min",
            "rule": "minimum validation loss",
            "early_stopping_patience": 8,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "epochs_completed": 1,
        "best_epoch": 1,
        "best_validation_loss": 0.5,
        "validation_metrics": {
            "threshold": 0.5,
            "macro_dice": 0.7,
            "macro_iou": 0.6,
            "n_images": 1,
        },
        "runtime_seconds": 1.0,
        "command": {"argv": ["train"], "shell": "train"},
        "source": {"commit": "a" * 40, "dirty": False},
        "environment": environment(),
    }


def valid_inputs(tmp_path: Path):
    config = ExperimentConfig(
        model="attention_unet",
        image_height=32,
        image_width=32,
        base_filters=2,
        n_tta=1,
    )
    unet_checkpoint = tmp_path / "unet.keras"
    attention_checkpoint = tmp_path / "attention.keras"
    unet_checkpoint.write_bytes(b"unet-checkpoint")
    attention_checkpoint.write_bytes(b"attention-checkpoint")
    unet = run_summary(config, "unet", unet_checkpoint)
    attention = run_summary(config, "attention_unet", attention_checkpoint)
    source = {"commit": "a" * 40, "dirty": False}
    return (
        config,
        manifest_rows(),
        unet,
        attention,
        unet_checkpoint,
        attention_checkpoint,
        environment(),
        source,
    )


def test_valid_run_pair_passes_all_integrity_checks(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)

    unet, attention = validate_run_pair(*inputs)

    assert unet["model_name"] == "unet"
    assert attention["model_name"] == "attention_unet"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("seed", "training seed"),
        ("manifest", "manifest"),
        ("config", "effective training configuration"),
        ("commit", "source commit"),
        ("software", "software environment"),
        ("hardware", "hardware environment"),
    ],
)
def test_run_pair_rejects_provenance_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    values = list(valid_inputs(tmp_path))
    attention = deepcopy(values[3])
    values[3] = attention
    if mutation == "seed":
        attention["seed"] = 43
    elif mutation == "manifest":
        attention["manifest_sha256"] = "f" * 64
    elif mutation == "config":
        config = attention["effective_config"]
        assert isinstance(config, dict)
        config["batch_size"] = 99
    elif mutation == "commit":
        source = attention["source"]
        assert isinstance(source, dict)
        source["commit"] = "b" * 40
    elif mutation == "software":
        run_environment = attention["environment"]
        assert isinstance(run_environment, dict)
        packages = run_environment["packages"]
        assert isinstance(packages, dict)
        packages["tensorflow"] = "2.17.0"
    else:
        run_environment = attention["environment"]
        assert isinstance(run_environment, dict)
        run_environment["tensorflow_device_details"] = [
            {
                "device_name": "/physical_device:GPU:0",
                "device_type": "GPU",
                "details": {"device_name": "P100"},
            }
        ]

    with pytest.raises(ComparisonIntegrityError, match=message):
        validate_run_pair(*values)


def test_run_pair_rejects_evaluation_config_drift(tmp_path: Path) -> None:
    values = list(valid_inputs(tmp_path))
    config = values[0]
    assert isinstance(config, ExperimentConfig)
    config.batch_size = 99

    with pytest.raises(ComparisonIntegrityError, match="evaluation config"):
        validate_run_pair(*values)


def test_run_pair_rejects_current_source_or_environment_drift(
    tmp_path: Path,
) -> None:
    values = list(valid_inputs(tmp_path))
    values[7] = {"commit": "b" * 40, "dirty": False}
    with pytest.raises(ComparisonIntegrityError, match="current source"):
        validate_run_pair(*values)

    values = list(valid_inputs(tmp_path))
    current_environment = deepcopy(values[6])
    assert isinstance(current_environment, dict)
    packages = current_environment["packages"]
    assert isinstance(packages, dict)
    packages["numpy"] = "2.0.0"
    values[6] = current_environment
    with pytest.raises(ComparisonIntegrityError, match="current software"):
        validate_run_pair(*values)


def test_run_pair_rejects_checkpoint_content_drift(tmp_path: Path) -> None:
    values = list(valid_inputs(tmp_path))
    attention_checkpoint = values[5]
    assert isinstance(attention_checkpoint, Path)
    attention_checkpoint.write_bytes(b"tampered")

    with pytest.raises(ComparisonIntegrityError, match="checkpoint"):
        validate_run_pair(*values)


def test_run_pair_requires_fixed_primary_evaluation_policy(tmp_path: Path) -> None:
    values = list(valid_inputs(tmp_path))
    config = values[0]
    assert isinstance(config, ExperimentConfig)
    config.n_tta = 8

    with pytest.raises(ComparisonIntegrityError, match="no TTA"):
        validate_run_pair(*values)
