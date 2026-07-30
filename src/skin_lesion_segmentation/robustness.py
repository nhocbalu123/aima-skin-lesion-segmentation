"""Predeclared deterministic synthetic image perturbations."""

from __future__ import annotations

import cv2
import numpy as np

PERTURBATION_CONDITIONS = (
    "clean",
    "brightness_plus_10",
    "contrast_1_10",
    "gaussian_blur_3",
    "jpeg_quality_85",
)


def _validate_images(images: object) -> np.ndarray:
    array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("images must be a rank-4 RGB batch")
    if not np.isfinite(array).all():
        raise ValueError("images must contain finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("images must be normalised to [0, 1]")
    return array


def _jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(
        np.rint(image * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode the JPEG perturbation")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV could not decode the JPEG perturbation")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def apply_perturbation(images: object, condition: str) -> np.ndarray:
    """Apply one fixed synthetic robustness condition to normalised RGB images."""

    batch = _validate_images(images)
    if condition not in PERTURBATION_CONDITIONS:
        raise ValueError(
            f"Unknown perturbation '{condition}'. Expected one of: "
            + ", ".join(PERTURBATION_CONDITIONS)
        )
    if condition == "clean":
        result = batch.copy()
    elif condition == "brightness_plus_10":
        result = np.clip(batch + np.float32(0.10), 0.0, 1.0)
    elif condition == "contrast_1_10":
        result = np.clip(
            (batch - np.float32(0.5)) * np.float32(1.10) + np.float32(0.5),
            0.0,
            1.0,
        )
    elif condition == "gaussian_blur_3":
        result = np.stack(
            [cv2.GaussianBlur(image, (3, 3), 0) for image in batch],
        )
    else:
        result = np.stack(
            [_jpeg_round_trip(image, quality=85) for image in batch],
        )
    return np.asarray(result, dtype=np.float32)
