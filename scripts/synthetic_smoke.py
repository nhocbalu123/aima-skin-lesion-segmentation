#!/usr/bin/env python3
"""Small CPU smoke flow from paths through evaluation and submission."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skin_lesion_segmentation.data import build_pair_manifest
from skin_lesion_segmentation.loading import PathBatchSequence
from skin_lesion_segmentation.metrics import evaluate_predictions
from skin_lesion_segmentation.splitting import build_split_manifest
from skin_lesion_segmentation.submission import build_submission


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        images = root / "images"
        masks = root / "masks"
        images.mkdir()
        masks.mkdir()
        for index in range(5):
            image = np.full((8, 8, 3), index * 20, np.uint8)
            mask = np.zeros((8, 8), np.uint8)
            mask[2:6, 2:6] = 255
            if not cv2.imwrite(str(images / f"case{index}.png"), image):
                raise RuntimeError("Could not write synthetic image")
            if not cv2.imwrite(str(masks / f"case{index}_segmentation.png"), mask):
                raise RuntimeError("Could not write synthetic mask")
        pairs = build_pair_manifest(images, masks)
        manifest = build_split_manifest(pairs, 0.2, seed=42)
        validation_ids = {str(row["sample_id"]) for row in manifest if row["split"] == "validation"}
        validation_pairs = [pair for pair in pairs if pair.sample_id in validation_ids]
        sequence = PathBatchSequence(validation_pairs, (8, 8), batch_size=3, shuffle=False)
        _, y_true = sequence[0]
        metrics = evaluate_predictions(y_true, y_true)
        sample = pd.DataFrame({"id": ["test_a"], "segmentation": [""]})
        submission, summary = build_submission(
            [root / "test_a.png"],
            [y_true[0, ..., 0]],
            sample_submission=sample,
            rle_order="C",
        )
        assert metrics["macro_dice"] == 1.0
        assert summary["rows"] == 1
        assert len(submission) == 1
    print("Synthetic smoke flow passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
