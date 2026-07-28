from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from skin_lesion_segmentation.cli import main
from skin_lesion_segmentation.config import ExperimentConfig


def test_prepare_command_writes_validated_manifests(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    output = tmp_path / "artifacts"
    images.mkdir()
    masks.mkdir()
    for index in range(4):
        assert cv2.imwrite(str(images / f"case{index}.png"), np.full((8, 8, 3), index, np.uint8))
        assert cv2.imwrite(str(masks / f"case{index}_segmentation.png"), np.zeros((8, 8), np.uint8))
    config = ExperimentConfig(
        image_dir=str(images),
        mask_dir=str(masks),
        output_dir=str(output),
        validation_fraction=0.25,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert (output / "pair_manifest.csv").exists()
    assert (output / "split_manifest.json").exists()


def test_evaluate_predictions_command_does_not_require_tensorflow(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "validation_predictions.npz"
    values = np.zeros((1, 4, 4, 1), dtype=np.float32)
    np.savez_compressed(
        predictions,
        sample_ids=np.asarray(["case"]),
        y_true=values,
        probabilities=values,
    )

    assert main(["evaluate-predictions", "--predictions", str(predictions)]) == 0
    assert '"macro_dice": 1.0' in capsys.readouterr().out
