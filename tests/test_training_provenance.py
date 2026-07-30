from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from skin_lesion_segmentation.config import ExperimentConfig
from skin_lesion_segmentation.data import SamplePair
from skin_lesion_segmentation.reproducibility import (
    collect_environment,
    collect_git_provenance,
    command_record,
)
from skin_lesion_segmentation.schemas import validate_run_summary
from skin_lesion_segmentation.splitting import (
    build_comparison_manifest,
    manifest_sha256,
)
from skin_lesion_segmentation.training import train


def test_command_record_preserves_arguments_and_shell_representation() -> None:
    result = command_record(
        ["python", "-m", "skin_lesion_segmentation.cli", "train", "--model", "unet"]
    )

    assert result == {
        "argv": [
            "python",
            "-m",
            "skin_lesion_segmentation.cli",
            "train",
            "--model",
            "unet",
        ],
        "shell": "python -m skin_lesion_segmentation.cli train --model unet",
    }


def test_git_provenance_reports_real_commit_and_dirty_boolean() -> None:
    root = Path(__file__).resolve().parents[1]

    result = collect_git_provenance(root)

    assert len(result["commit"]) == 40
    int(result["commit"], 16)
    assert isinstance(result["dirty"], bool)


def test_git_provenance_uses_exported_commit_when_git_metadata_is_absent(
    tmp_path: Path,
) -> None:
    commit = "d" * 40
    (tmp_path / "SOURCE_COMMIT").write_text(commit + "\n", encoding="ascii")

    result = collect_git_provenance(tmp_path)

    assert result == {"commit": commit, "dirty": False}


def test_environment_records_visible_device_details() -> None:
    environment = collect_environment()

    assert isinstance(environment["tensorflow_devices"], list)
    assert isinstance(environment["tensorflow_device_details"], list)
    assert all("device_type" in item and "device_name" in item for item in environment["tensorflow_device_details"])


@pytest.mark.parametrize("model_name", ["unet", "attention_unet"])
def test_experiment_config_accepts_supported_model(model_name: str) -> None:
    config = ExperimentConfig(model=model_name)

    config.validate()

    assert config.to_dict()["model"] == model_name


def test_experiment_config_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model"):
        ExperimentConfig(model="resunet").validate()


def test_training_writes_complete_summary_without_internal_test_metrics(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    pairs: list[SamplePair] = []
    for index in range(7):
        sample_id = f"case-{index}"
        image_path = image_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_segmentation.png"
        image = np.full((32, 32, 3), index * 20, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        pairs.append(SamplePair(sample_id, image_path, mask_path))
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=101,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    config = ExperimentConfig(
        model="unet",
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        output_dir=str(output_dir),
        image_height=32,
        image_width=32,
        batch_size=2,
        epochs=1,
        base_filters=2,
        early_stopping_patience=0,
        n_tta=1,
        threshold_grid=[0.5],
        min_component_sizes=[0],
        morphology_kernels=[0],
        l2_coefficient=0.0,
    )
    command = ["python", "-m", "skin_lesion_segmentation.cli", "train", "--model", "unet"]

    result = train(
        config,
        split_manifest_path=manifest_path,
        command=command,
    )
    summary = validate_run_summary(result["summary"])

    assert summary["model_name"] == "unet"
    assert summary["parameter_count"] == 31_119
    assert summary["manifest_sha256"] == manifest_sha256(rows)
    assert summary["command"]["argv"] == command
    assert summary["checkpoint_selection"]["monitor"] == "val_loss"
    assert "internal_test_metrics" not in summary
    assert (output_dir / "run_summary.json").is_file()
