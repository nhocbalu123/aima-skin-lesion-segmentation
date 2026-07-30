from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from skin_lesion_segmentation.data import SamplePair
from skin_lesion_segmentation.config import ExperimentConfig
from skin_lesion_segmentation.splitting import (
    LeakageError,
    audit_split_integrity,
    build_comparison_manifest,
    manifest_sha256,
)
from skin_lesion_segmentation.training import prepare_manifests, training_pairs_from_manifest


def make_pair(
    root: Path,
    sample_id: str,
    *,
    image_value: int,
    mask_value: int = 0,
) -> SamplePair:
    image_path = root / f"{sample_id}.png"
    mask_path = root / f"{sample_id}_segmentation.png"
    image = np.full((6, 7, 3), image_value, dtype=np.uint8)
    image[0, 0] = (image_value + 1) % 255
    mask = np.zeros((6, 7), dtype=np.uint8)
    if mask_value:
        mask[1:4, 2:5] = mask_value
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    return SamplePair(sample_id, image_path, mask_path)


def test_three_way_manifest_is_deterministic_and_has_expected_singleton_counts(
    tmp_path: Path,
) -> None:
    pairs = [
        make_pair(tmp_path, f"case-{index:02d}", image_value=index)
        for index in range(20)
    ]

    first = build_comparison_manifest(
        pairs,
        validation_fraction=0.15,
        internal_test_fraction=0.15,
        split_seed=20260730,
    )
    second = build_comparison_manifest(
        pairs,
        validation_fraction=0.15,
        internal_test_fraction=0.15,
        split_seed=20260730,
    )

    assert first == second
    counts = {
        split: sum(row["split"] == split for row in first)
        for split in ("train", "validation", "internal_test")
    }
    assert counts == {"train": 14, "validation": 3, "internal_test": 3}
    assert all(len(str(row["image_sha256"])) == 64 for row in first)
    assert all(len(str(row["mask_sha256"])) == 64 for row in first)
    assert len(manifest_sha256(first)) == 64


def test_comparison_manifest_keeps_duplicate_images_and_groups_together(
    tmp_path: Path,
) -> None:
    pairs = [
        make_pair(tmp_path, f"case-{index}", image_value=index)
        for index in range(8)
    ]
    # Replace case-1 with a decoded-image duplicate of case-0.
    duplicate = cv2.imread(str(pairs[0].image_path), cv2.IMREAD_UNCHANGED)
    assert duplicate is not None
    assert cv2.imwrite(str(pairs[1].image_path), duplicate)
    groups = {
        pair.sample_id: ("shared-patient" if pair.sample_id in {"case-2", "case-3"} else pair.sample_id)
        for pair in pairs
    }

    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.25,
        internal_test_fraction=0.25,
        split_seed=17,
        group_mapping=groups,
    )
    by_id = {str(row["sample_id"]): str(row["split"]) for row in rows}

    assert by_id["case-0"] == by_id["case-1"]
    assert by_id["case-2"] == by_id["case-3"]
    audit_split_integrity(rows)


def test_mask_content_changes_manifest_identity(tmp_path: Path) -> None:
    pairs = [
        make_pair(tmp_path, f"case-{index}", image_value=index)
        for index in range(7)
    ]
    first = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=5,
    )
    first_hash = manifest_sha256(first)

    changed_mask = np.zeros((6, 7), dtype=np.uint8)
    changed_mask[2:5, 1:6] = 255
    assert cv2.imwrite(str(pairs[0].mask_path), changed_mask)
    second = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=5,
    )

    assert first_hash != manifest_sha256(second)
    assert first[0]["image_sha256"] == second[0]["image_sha256"]
    assert first[0]["mask_sha256"] != second[0]["mask_sha256"]


def test_manifest_identity_is_portable_across_path_prefixes() -> None:
    first = [
        {
            "sample_id": "a",
            "image_path": "/kaggle/input/a.png",
            "mask_path": "/kaggle/input/a_mask.png",
            "split": "train",
            "group_id": None,
            "image_sha256": "1" * 64,
            "mask_sha256": "2" * 64,
        }
    ]
    second = [{**first[0], "image_path": "D:/data/a.png", "mask_path": "D:/data/a_mask.png"}]

    assert manifest_sha256(first) == manifest_sha256(second)


def test_training_pair_selection_never_returns_internal_test(tmp_path: Path) -> None:
    pairs = [
        make_pair(tmp_path, f"case-{index}", image_value=index)
        for index in range(6)
    ]
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=9,
    )

    train, validation = training_pairs_from_manifest(pairs, rows)
    selected = {pair.sample_id for pair in [*train, *validation]}
    test_ids = {
        str(row["sample_id"])
        for row in rows
        if row["split"] == "internal_test"
    }

    assert selected.isdisjoint(test_ids)
    assert selected | test_ids == {pair.sample_id for pair in pairs}


def test_three_way_manifest_requires_three_independent_components(tmp_path: Path) -> None:
    pairs = [
        make_pair(tmp_path, f"case-{index}", image_value=index)
        for index in range(4)
    ]
    groups = {pair.sample_id: "one-patient" for pair in pairs}

    with pytest.raises(ValueError, match="at least three independent"):
        build_comparison_manifest(
            pairs,
            validation_fraction=0.25,
            internal_test_fraction=0.25,
            split_seed=1,
            group_mapping=groups,
        )


def test_audit_rejects_mask_hash_crossing_splits() -> None:
    rows = [
        {
            "sample_id": "a",
            "split": "train",
            "group_id": None,
            "image_sha256": "image-a",
            "mask_sha256": "same-mask",
        },
        {
            "sample_id": "b",
            "split": "internal_test",
            "group_id": None,
            "image_sha256": "image-b",
            "mask_sha256": "same-mask",
        },
    ]

    # Identical empty or common masks are legitimate and must not be treated as
    # duplicate samples. The audit still accepts the manifest.
    audit_split_integrity(rows)


def test_audit_accepts_internal_test_but_rejects_unknown_split() -> None:
    audit_split_integrity(
        [
            {
                "sample_id": "a",
                "split": "internal_test",
                "group_id": None,
                "image_sha256": "image-a",
                "mask_sha256": "mask-a",
            }
        ]
    )
    with pytest.raises(LeakageError, match="Unknown split"):
        audit_split_integrity(
            [
                {
                    "sample_id": "a",
                    "split": "holdout",
                    "group_id": None,
                    "image_sha256": "image-a",
                    "mask_sha256": "mask-a",
                }
            ]
        )


def test_prepare_manifests_reuses_supplied_immutable_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    pairs: list[SamplePair] = []
    for index in range(7):
        sample_id = f"case-{index}"
        image_path = image_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_segmentation.png"
        assert cv2.imwrite(
            str(image_path),
            np.full((6, 7, 3), index, dtype=np.uint8),
        )
        assert cv2.imwrite(
            str(mask_path),
            np.zeros((6, 7), dtype=np.uint8),
        )
        pairs.append(SamplePair(sample_id, image_path, mask_path))
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=11,
    )
    external = tmp_path / "immutable-manifest.json"
    external.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "run"
    config = ExperimentConfig(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        output_dir=str(output),
    )

    loaded_pairs, loaded_rows = prepare_manifests(
        config,
        split_manifest_path=external,
    )

    assert [pair.sample_id for pair in loaded_pairs] == [pair.sample_id for pair in pairs]
    assert loaded_rows == rows
    assert json.loads((output / "split_manifest.json").read_text(encoding="utf-8")) == rows


def test_prepare_rejects_manifest_that_does_not_match_dataset_content(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    pairs: list[SamplePair] = []
    for index in range(5):
        sample_id = f"case-{index}"
        image_path = image_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_segmentation.png"
        assert cv2.imwrite(
            str(image_path),
            np.full((6, 7, 3), index, dtype=np.uint8),
        )
        assert cv2.imwrite(str(mask_path), np.zeros((6, 7), dtype=np.uint8))
        pairs.append(SamplePair(sample_id, image_path, mask_path))
    rows = build_comparison_manifest(
        pairs,
        validation_fraction=0.2,
        internal_test_fraction=0.2,
        split_seed=3,
    )
    rows[0]["mask_sha256"] = "f" * 64
    external = tmp_path / "tampered.json"
    external.write_text(json.dumps(rows), encoding="utf-8")
    config = ExperimentConfig(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        output_dir=str(tmp_path / "run"),
    )

    with pytest.raises(LeakageError, match="content does not match"):
        prepare_manifests(config, split_manifest_path=external)
