"""Albumentations pipeline pinned to the documented 1.4.24 API."""

from __future__ import annotations

import os

import cv2


def build_train_augmentation(seed: int):
    """Construct training-only image/mask augmentation.

    Geometric transforms use linear interpolation for images and nearest-neighbour
    interpolation for masks. The pipeline deliberately avoids deprecated
    ``alpha_affine``, ``shift_limit``, and ``var_limit`` arguments.
    """

    # Albumentations 1.4.24 otherwise performs an import-time PyPI version
    # check. Training and CI must be deterministic and network-independent.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15.0, 15.0),
                shear=(-5.0, 5.0),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=0.2,
            ),
        ],
        seed=seed,
    )
