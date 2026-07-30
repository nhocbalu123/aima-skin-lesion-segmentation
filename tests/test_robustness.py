from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.robustness import (
    PERTURBATION_CONDITIONS,
    apply_perturbation,
)


def sample_images() -> np.ndarray:
    values = np.linspace(0.0, 1.0, num=2 * 8 * 8 * 3, dtype=np.float32)
    return values.reshape(2, 8, 8, 3)


def test_clean_perturbation_returns_equal_independent_float32_batch() -> None:
    images = sample_images()

    result = apply_perturbation(images, "clean")

    np.testing.assert_array_equal(result, images)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, images)


def test_brightness_and_contrast_are_fixed_and_bounded() -> None:
    images = np.full((1, 4, 4, 3), 0.5, dtype=np.float32)

    brighter = apply_perturbation(images, "brightness_plus_10")
    contrasted = apply_perturbation(images, "contrast_1_10")

    np.testing.assert_allclose(brighter, 0.6, atol=1e-7)
    np.testing.assert_allclose(contrasted, 0.5, atol=1e-7)
    assert 0.0 <= float(brighter.min()) <= float(brighter.max()) <= 1.0


@pytest.mark.parametrize("condition", ["gaussian_blur_3", "jpeg_quality_85"])
def test_image_degradation_is_deterministic_and_shape_preserving(
    condition: str,
) -> None:
    images = sample_images()

    first = apply_perturbation(images, condition)
    second = apply_perturbation(images, condition)

    np.testing.assert_array_equal(first, second)
    assert first.shape == images.shape
    assert first.dtype == np.float32
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0


def test_predeclared_robustness_conditions_are_complete() -> None:
    assert PERTURBATION_CONDITIONS == (
        "clean",
        "brightness_plus_10",
        "contrast_1_10",
        "gaussian_blur_3",
        "jpeg_quality_85",
    )


def test_perturbation_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError, match="Unknown perturbation"):
        apply_perturbation(sample_images(), "rotate_45")


def test_perturbation_requires_normalised_rgb_batches() -> None:
    with pytest.raises(ValueError, match="rank-4 RGB"):
        apply_perturbation(np.zeros((8, 8, 3), dtype=np.float32), "clean")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_perturbation(np.full((1, 8, 8, 3), 2.0, dtype=np.float32), "clean")
