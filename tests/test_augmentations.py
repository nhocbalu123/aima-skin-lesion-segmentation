from __future__ import annotations

import warnings

import numpy as np
import pytest

albumentations = pytest.importorskip("albumentations")

from skin_lesion_segmentation.augmentations import build_train_augmentation


def test_augmentation_constructs_without_warnings_and_keeps_mask_binary() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        augmentation = build_train_augmentation(seed=11)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:8, 6:9] = 1
    output = augmentation(image=image, mask=mask)
    assert set(np.unique(output["mask"]).tolist()) <= {0, 1}
