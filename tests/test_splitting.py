from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import numpy as np
import pytest

from skin_lesion_segmentation.data import SamplePair
from skin_lesion_segmentation.splitting import LeakageError, audit_split_integrity, build_split_manifest


def make_pair(root: Path, sample_id: str, image_bytes: bytes) -> SamplePair:
    image_path = root / f"{sample_id}.png"
    mask_path = root / f"{sample_id}_segmentation.png"
    digest = hashlib.sha256(image_bytes).digest()
    pixels = np.frombuffer((digest * 2)[:48], dtype=np.uint8).reshape(4, 4, 3)
    assert cv2.imwrite(str(image_path), pixels)
    assert cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    return SamplePair(sample_id, image_path, mask_path)


def test_split_is_deterministic(tmp_path: Path) -> None:
    pairs = [make_pair(tmp_path, str(i), f"image-{i}".encode()) for i in range(10)]

    first = build_split_manifest(pairs, val_fraction=0.2, seed=42)
    second = build_split_manifest(pairs, val_fraction=0.2, seed=42)

    assert first == second
    assert {row["split"] for row in first} == {"train", "validation"}


def test_explicit_groups_do_not_cross_splits(tmp_path: Path) -> None:
    pairs = [make_pair(tmp_path, str(i), f"image-{i}".encode()) for i in range(6)]
    groups = {"0": "p0", "1": "p0", "2": "p1", "3": "p1", "4": "p2", "5": "p2"}

    manifest = build_split_manifest(pairs, 0.34, 9, group_mapping=groups)
    audit_split_integrity(manifest)

    by_group: dict[str, set[str]] = {}
    for row in manifest:
        by_group.setdefault(str(row["group_id"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in by_group.values())


def test_missing_explicit_group_is_rejected(tmp_path: Path) -> None:
    pairs = [make_pair(tmp_path, "a", b"a"), make_pair(tmp_path, "b", b"b")]

    with pytest.raises(ValueError, match="missing group IDs"):
        build_split_manifest(pairs, 0.5, 1, group_mapping={"a": "p0"})


def test_exact_duplicate_images_are_kept_in_same_split(tmp_path: Path) -> None:
    pairs = [
        make_pair(tmp_path, "a", b"same"),
        make_pair(tmp_path, "b", b"same"),
        make_pair(tmp_path, "c", b"different"),
        make_pair(tmp_path, "d", b"other"),
    ]

    manifest = build_split_manifest(pairs, 0.5, 3)
    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in manifest}
    assert split_by_id["a"] == split_by_id["b"]
    audit_split_integrity(manifest)


def test_audit_detects_duplicate_crossing_splits() -> None:
    rows = [
        {"sample_id": "a", "split": "train", "group_id": None, "image_sha256": "same"},
        {"sample_id": "b", "split": "validation", "group_id": None, "image_sha256": "same"},
    ]

    with pytest.raises(LeakageError, match="exact duplicate"):
        audit_split_integrity(rows)


def test_audit_detects_group_crossing_splits() -> None:
    rows = [
        {"sample_id": "a", "split": "train", "group_id": "p1", "image_sha256": "x"},
        {"sample_id": "b", "split": "validation", "group_id": "p1", "image_sha256": "y"},
    ]

    with pytest.raises(LeakageError, match="group"):
        audit_split_integrity(rows)


def test_pixel_identical_images_with_different_encodings_stay_together(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = 255
    pairs = []
    for sample_id, compression in [("a", 0), ("b", 9)]:
        image_path = tmp_path / f"{sample_id}.png"
        mask_path = tmp_path / f"{sample_id}_segmentation.png"
        assert cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        assert cv2.imwrite(str(mask_path), np.zeros((8, 8), dtype=np.uint8))
        pairs.append(SamplePair(sample_id, image_path, mask_path))
    assert pairs[0].image_path.read_bytes() != pairs[1].image_path.read_bytes()
    pairs.extend([
        make_pair(tmp_path, "c", b"other-c"),
        make_pair(tmp_path, "d", b"other-d"),
    ])
    manifest = build_split_manifest(pairs, 0.5, 7)
    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in manifest}
    assert split_by_id["a"] == split_by_id["b"]


def test_duplicate_audit_rejects_undecodable_images(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    mask_path = tmp_path / "broken_segmentation.png"
    image_path.write_bytes(b"not-an-image")
    assert cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    pair = SamplePair("broken", image_path, mask_path)

    with pytest.raises(IOError, match="Could not decode image for duplicate audit"):
        build_split_manifest([pair, make_pair(tmp_path, "valid", b"valid")], 0.5, 1)


def test_extra_explicit_group_id_is_rejected(tmp_path: Path) -> None:
    pairs = [make_pair(tmp_path, "a", b"a"), make_pair(tmp_path, "b", b"b")]

    with pytest.raises(ValueError, match="extra group IDs"):
        build_split_manifest(
            pairs,
            0.5,
            1,
            group_mapping={"a": "p0", "b": "p1", "ghost": "p2"},
        )


@pytest.mark.parametrize("blank_group", ["", None])
def test_blank_explicit_group_id_is_rejected(tmp_path: Path, blank_group: object) -> None:
    pairs = [make_pair(tmp_path, "a", b"a"), make_pair(tmp_path, "b", b"b")]

    with pytest.raises(ValueError, match="blank group ID"):
        build_split_manifest(
            pairs,
            0.5,
            1,
            group_mapping={"a": blank_group, "b": "p1"},  # type: ignore[dict-item]
        )


def test_group_mapping_sample_ids_are_case_insensitive(tmp_path: Path) -> None:
    pairs = [make_pair(tmp_path, "a", b"a"), make_pair(tmp_path, "b", b"b")]

    manifest = build_split_manifest(
        pairs,
        0.5,
        1,
        group_mapping={"A": "p0", "B": "p1"},
    )

    assert {str(row["group_id"]) for row in manifest} == {"p0", "p1"}


def test_audit_rejects_duplicate_sample_rows() -> None:
    rows = [
        {"sample_id": "a", "split": "train", "group_id": None, "image_sha256": "x"},
        {"sample_id": "a", "split": "train", "group_id": None, "image_sha256": "x"},
    ]

    with pytest.raises(LeakageError, match="duplicate sample IDs"):
        audit_split_integrity(rows)


def test_audit_rejects_blank_sample_id() -> None:
    rows = [{"sample_id": "", "split": "train", "group_id": None, "image_sha256": "x"}]

    with pytest.raises(LeakageError, match="blank sample ID"):
        audit_split_integrity(rows)
