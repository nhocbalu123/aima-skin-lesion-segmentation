from __future__ import annotations

import json
from pathlib import Path

import pytest

from skin_lesion_segmentation.evaluation import load_split_manifest
from skin_lesion_segmentation.splitting import LeakageError


def test_loaded_manifest_is_reaudited_for_duplicate_cross_split(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            [
                {"sample_id": "a", "image_path": "a.png", "mask_path": "a_mask.png", "split": "train", "group_id": None, "image_sha256": "same"},
                {"sample_id": "b", "image_path": "b.png", "mask_path": "b_mask.png", "split": "validation", "group_id": None, "image_sha256": "same"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(LeakageError, match="exact duplicate"):
        load_split_manifest(path)


def test_loaded_manifest_requires_evaluation_paths(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps([{"sample_id": "a", "split": "validation", "image_sha256": "x"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_split_manifest(path)
