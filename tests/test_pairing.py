from __future__ import annotations

from pathlib import Path

import pytest

from skin_lesion_segmentation.data import PairingError, build_pair_manifest, canonical_sample_id


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_pairing_matches_ids_independent_of_input_order(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    touch(images / "b.JPEG")
    touch(images / "a.png")
    touch(masks / "a_segmentation.TIF")
    touch(masks / "b_segmentation.bmp")

    pairs = build_pair_manifest(images, masks, mask_suffix="_segmentation")

    assert [pair.sample_id for pair in pairs] == ["a", "b"]
    assert pairs[0].image_path.name == "a.png"
    assert pairs[0].mask_path.name == "a_segmentation.TIF"


def test_missing_mask_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path / "images" / "a.jpg")
    touch(tmp_path / "images" / "b.jpg")
    touch(tmp_path / "masks" / "a_segmentation.png")

    with pytest.raises(PairingError, match="missing masks.*b"):
        build_pair_manifest(tmp_path / "images", tmp_path / "masks", "_segmentation")


def test_extra_mask_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path / "images" / "a.jpg")
    touch(tmp_path / "masks" / "a_segmentation.png")
    touch(tmp_path / "masks" / "b_segmentation.png")

    with pytest.raises(PairingError, match="extra masks.*b"):
        build_pair_manifest(tmp_path / "images", tmp_path / "masks", "_segmentation")


def test_unequal_counts_are_never_truncated(tmp_path: Path) -> None:
    touch(tmp_path / "images" / "a.jpg")
    touch(tmp_path / "images" / "b.jpg")
    touch(tmp_path / "masks" / "a_segmentation.png")

    with pytest.raises(PairingError):
        build_pair_manifest(tmp_path / "images", tmp_path / "masks", "_segmentation")


def test_duplicate_image_ids_are_rejected(tmp_path: Path) -> None:
    touch(tmp_path / "images" / "a.jpg")
    touch(tmp_path / "images" / "a.PNG")
    touch(tmp_path / "masks" / "a_segmentation.png")

    with pytest.raises(PairingError, match="duplicate image ID 'a'"):
        build_pair_manifest(tmp_path / "images", tmp_path / "masks", "_segmentation")


def test_duplicate_mask_ids_are_rejected(tmp_path: Path) -> None:
    touch(tmp_path / "images" / "a.jpg")
    touch(tmp_path / "masks" / "a_segmentation.png")
    touch(tmp_path / "masks" / "a_segmentation.BMP")

    with pytest.raises(PairingError, match="duplicate mask ID 'a'"):
        build_pair_manifest(tmp_path / "images", tmp_path / "masks", "_segmentation")


@pytest.mark.parametrize("name", ["case.JPG", "case.JpEg", "case.PNG", "case.BMP", "case.TIF", "case.TIFF"])
def test_supported_extensions_are_case_insensitive(name: str) -> None:
    assert canonical_sample_id(Path(name)) == "case"


def test_mask_suffix_is_removed_only_from_end() -> None:
    assert canonical_sample_id(Path("x_segmentation.PNG"), mask_suffix="_segmentation") == "x"
    assert canonical_sample_id(Path("segmentation_x.PNG"), mask_suffix="_segmentation") == "segmentation_x"
