from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from skin_lesion_segmentation.data import SamplePair
from skin_lesion_segmentation.loading import PathBatchSequence, load_image_mask


def write_pair(root: Path, sample_id: str, value: int = 64) -> SamplePair:
    image = np.full((5, 7, 3), value, dtype=np.uint8)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 255
    image_path = root / f"{sample_id}.png"
    mask_path = root / f"{sample_id}_segmentation.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    return SamplePair(sample_id, image_path, mask_path)


def test_final_partial_batch_is_included(tmp_path: Path) -> None:
    pairs = [write_pair(tmp_path, str(i), i) for i in range(5)]
    seq = PathBatchSequence(pairs, image_size=(8, 8), batch_size=2, shuffle=False)

    assert len(seq) == 3
    images, masks = seq[2]
    assert images.shape[0] == 1
    assert masks.shape[0] == 1


def test_mask_resize_preserves_binary_values_and_small_region(tmp_path: Path) -> None:
    pair = write_pair(tmp_path, "small")

    _, mask = load_image_mask(pair.image_path, pair.mask_path, image_size=(15, 21))

    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0


def test_read_failure_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(IOError, match="Could not read image"):
        load_image_mask(tmp_path / "missing.jpg", tmp_path / "missing.png", (8, 8))


def test_shuffle_is_deterministic_for_seed(tmp_path: Path) -> None:
    pairs = [write_pair(tmp_path, str(i), i) for i in range(6)]
    first = PathBatchSequence(pairs, (8, 8), 2, shuffle=True, seed=7)
    second = PathBatchSequence(pairs, (8, 8), 2, shuffle=True, seed=7)

    assert first.ordered_sample_ids == second.ordered_sample_ids
    first.on_epoch_end()
    second.on_epoch_end()
    assert first.ordered_sample_ids == second.ordered_sample_ids


def test_image_only_sequence_includes_remainder(tmp_path: Path) -> None:
    from skin_lesion_segmentation.loading import PathImageSequence

    paths = []
    for index in range(5):
        path = tmp_path / f"test_{index}.png"
        assert cv2.imwrite(str(path), np.full((4, 6, 3), index, dtype=np.uint8))
        paths.append(path)
    sequence = PathImageSequence(paths, (8, 8), batch_size=2)
    assert len(sequence) == 3
    assert sequence[2].shape == (1, 8, 8, 3)


def test_mask_loader_preserves_zero_one_encoded_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
    source_mask = np.zeros((4, 4), dtype=np.uint8)
    source_mask[1, 2] = 1
    assert cv2.imwrite(str(mask_path), source_mask)

    _, mask = load_image_mask(image_path, mask_path, image_size=(4, 4))

    assert mask.sum() == 1
    assert set(np.unique(mask).tolist()) == {0.0, 1.0}




def test_mask_binarisation_rejects_low_intensity_background_noise(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    mask = np.full((8, 8), 5, dtype=np.uint8)
    mask[3:5, 3:5] = 255
    assert cv2.imwrite(str(mask_path), mask)

    _, loaded = load_image_mask(image_path, mask_path, (8, 8))

    assert loaded[..., 0].sum() == 4
    assert loaded[0, 0, 0] == 0.0


def test_resize_binary_mask_to_explicit_submission_shape() -> None:
    from skin_lesion_segmentation.loading import resize_binary_mask

    source = np.zeros((3, 5), dtype=np.uint8)
    source[1, 2] = 1
    resized = resize_binary_mask(source, (8, 6))

    assert resized.shape == (8, 6)
    assert set(np.unique(resized).tolist()) <= {0, 1}
    assert resized.sum() > 0


def test_resize_binary_mask_rejects_invalid_shape() -> None:
    from skin_lesion_segmentation.loading import resize_binary_mask

    with pytest.raises(ValueError, match="target shape"):
        resize_binary_mask(np.zeros((2, 2), dtype=np.uint8), (0, 8))

