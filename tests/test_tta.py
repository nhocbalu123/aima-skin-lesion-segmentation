from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.tta import DIHEDRAL_TRANSFORMS, predict_with_tta


def test_transforms_are_unique_on_asymmetric_input() -> None:
    x = np.arange(9).reshape(1, 3, 3, 1)
    outputs = [transform.forward(x).tobytes() for transform in DIHEDRAL_TRANSFORMS]
    assert len(set(outputs)) == 8


def test_each_inverse_restores_input() -> None:
    x = np.arange(12).reshape(1, 3, 4, 1)
    for transform in DIHEDRAL_TRANSFORMS:
        assert np.array_equal(transform.inverse(transform.forward(x)), x), transform.name


def test_n_augmentations_controls_call_count_and_shape() -> None:
    calls = 0

    def predict(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return x[..., :1].astype(np.float32)

    x = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    result = predict_with_tta(predict, x, n_augmentations=5)
    assert calls == 5
    assert result.shape == x.shape
    assert np.allclose(result, x)


@pytest.mark.parametrize("value", [0, 9, -1, 1.5])
def test_invalid_n_augmentations_is_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        predict_with_tta(lambda x: x, np.zeros((1, 2, 2, 1)), value)  # type: ignore[arg-type]


def test_rotational_tta_rejects_rectangular_inputs() -> None:
    rectangular = np.zeros((1, 3, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="square"):
        predict_with_tta(lambda x: x, rectangular, n_augmentations=2)


def test_identity_only_tta_allows_rectangular_inputs() -> None:
    rectangular = np.zeros((1, 3, 4, 1), dtype=np.float32)
    result = predict_with_tta(lambda x: x, rectangular, n_augmentations=1)
    assert result.shape == rectangular.shape


def test_tta_passes_contiguous_non_negative_stride_arrays_to_model() -> None:
    calls = 0

    def predict(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert x.flags.c_contiguous
        assert all(stride >= 0 for stride in x.strides)
        return x[..., :1]

    images = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
    predict_with_tta(predict, images, n_augmentations=8)
    assert calls == 8


def test_tta_rejects_non_single_channel_predictions() -> None:
    images = np.zeros((1, 4, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="single-channel"):
        predict_with_tta(
            lambda x: np.zeros((len(x), x.shape[1], x.shape[2], 2), dtype=np.float32),
            images,
            n_augmentations=1,
        )
