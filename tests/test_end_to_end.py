from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skin_lesion_segmentation.data import build_pair_manifest
from skin_lesion_segmentation.loading import PathBatchSequence
from skin_lesion_segmentation.metrics import evaluate_predictions
from skin_lesion_segmentation.rle import decode_rle
from skin_lesion_segmentation.splitting import build_split_manifest
from skin_lesion_segmentation.submission import build_submission


def test_synthetic_paths_to_evaluation_and_submission(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for i in range(4):
        image = np.full((8, 8, 3), i * 20, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        assert cv2.imwrite(str(image_dir / f"case{i}.PNG"), image)
        assert cv2.imwrite(str(mask_dir / f"case{i}_segmentation.png"), mask)

    pairs = build_pair_manifest(image_dir, mask_dir, "_segmentation")
    split = build_split_manifest(pairs, 0.25, seed=3)
    val_ids = {str(row["sample_id"]) for row in split if row["split"] == "validation"}
    val_pairs = [pair for pair in pairs if pair.sample_id in val_ids]
    sequence = PathBatchSequence(val_pairs, (8, 8), batch_size=3, shuffle=False)
    _, masks = sequence[0]
    metrics = evaluate_predictions(masks, masks, threshold=0.5)
    assert metrics["macro_dice"] == 1.0

    test_paths = [tmp_path / "test_b.JPG", tmp_path / "test_a.png"]
    sample = pd.DataFrame({"id": ["test_a", "test_b"], "segmentation": ["", ""]})
    submission, _ = build_submission(test_paths, [masks[0, ..., 0], masks[0, ..., 0]], sample_submission=sample)
    decoded = decode_rle(submission.loc[0, "segmentation"], (8, 8), order="C")
    assert decoded.sum() == 16
