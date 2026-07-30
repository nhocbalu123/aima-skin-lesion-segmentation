from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from skin_lesion_segmentation.artifacts import sha256_file
from skin_lesion_segmentation.comparison import compare_runs
from skin_lesion_segmentation.config import ExperimentConfig
from skin_lesion_segmentation.data import SamplePair
from skin_lesion_segmentation.model import build_model
from skin_lesion_segmentation.reproducibility import (
    collect_environment,
    collect_git_provenance,
)
from skin_lesion_segmentation.splitting import (
    build_comparison_manifest,
    manifest_sha256,
)


def make_summary(
    config: ExperimentConfig,
    model_name: str,
    checkpoint: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    effective = config.to_dict()
    effective["model"] = model_name
    effective["output_dir"] = str(checkpoint.parent)
    model = build_model(
        model_name,
        (32, 32, 3),
        base_filters=2,
        l2_coefficient=0.0,
    )
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "model_name": model_name,
        "seed": config.seed,
        "manifest_sha256": manifest_sha256(rows),
        "parameter_count": model.count_params(),
        "effective_config": effective,
        "checkpoint_selection": {
            "monitor": "val_loss",
            "mode": "min",
            "rule": "minimum validation loss",
            "early_stopping_patience": config.early_stopping_patience,
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
            "macro_dice": 0.5,
            "macro_iou": 0.4,
            "n_images": 1,
        },
        "runtime_seconds": 1.0,
        "command": {"argv": ["synthetic-test"], "shell": "synthetic-test"},
        "source": collect_git_provenance(root),
        "environment": collect_environment(),
    }


def test_synthetic_models_produce_portable_comparison_artifacts(
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
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[4 + index : 12 + index, 8:20] = 40 + index * 20
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[6 + index : 14 + index, 10:18] = 255
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        pairs.append(SamplePair(sample_id, image_path, mask_path))
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=303,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(rows), encoding="utf-8")
    config = ExperimentConfig(
        model="attention_unet",
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        output_dir=str(tmp_path / "unused"),
        image_height=32,
        image_width=32,
        batch_size=2,
        epochs=1,
        base_filters=2,
        n_tta=1,
        l2_coefficient=0.0,
    )
    unet_checkpoint = tmp_path / "unet.keras"
    attention_checkpoint = tmp_path / "attention.keras"
    build_model(
        "unet",
        (32, 32, 3),
        base_filters=2,
        l2_coefficient=0.0,
    ).save(unet_checkpoint)
    build_model(
        "attention_unet",
        (32, 32, 3),
        base_filters=2,
        l2_coefficient=0.0,
    ).save(attention_checkpoint)
    unet_summary = tmp_path / "unet-summary.json"
    attention_summary = tmp_path / "attention-summary.json"
    unet_summary.write_text(
        json.dumps(make_summary(config, "unet", unet_checkpoint, rows)),
        encoding="utf-8",
    )
    attention_summary.write_text(
        json.dumps(
            make_summary(
                config,
                "attention_unet",
                attention_checkpoint,
                rows,
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    result = compare_runs(
        config,
        manifest_path,
        unet_summary,
        attention_summary,
        unet_checkpoint,
        attention_checkpoint,
        output,
        bootstrap_seed=19,
        n_resamples=200,
    )

    assert result["status"] == "completed"
    assert result["manifest_sha256"] == manifest_sha256(rows)
    assert result["primary_policy"] == {
        "threshold": 0.5,
        "tta": "none",
        "postprocessing": "none",
    }
    expected = {
        "comparison_summary.json",
        "per_image_metrics.csv",
        "slice_summary.json",
        "robustness_summary.json",
        "failure_cases.json",
        "qualitative_failures.png",
        "unet_run_summary.json",
        "attention_unet_run_summary.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with (output / "per_image_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    expected_test = sum(row["split"] == "internal_test" for row in rows)
    assert len(metric_rows) == expected_test
    copied = json.loads((output / "unet_run_summary.json").read_text(encoding="utf-8"))
    assert copied["checkpoint"]["sha256"] == sha256_file(unet_checkpoint)
