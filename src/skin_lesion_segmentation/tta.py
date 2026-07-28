"""Eight unique dihedral test-time augmentations and their inverses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class DihedralTransform:
    name: str
    forward: ArrayTransform
    inverse: ArrayTransform


def _rot(k: int) -> ArrayTransform:
    return lambda x: np.rot90(x, k=k, axes=(1, 2))


def _flip_horizontal(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=2)


def _flip_vertical(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=1)


def _main_diagonal(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, 1, 2)


def _anti_diagonal(x: np.ndarray) -> np.ndarray:
    return np.flip(np.swapaxes(x, 1, 2), axis=(1, 2))


DIHEDRAL_TRANSFORMS: tuple[DihedralTransform, ...] = (
    DihedralTransform("identity", lambda x: x, lambda x: x),
    DihedralTransform("rotate_90", _rot(1), _rot(3)),
    DihedralTransform("rotate_180", _rot(2), _rot(2)),
    DihedralTransform("rotate_270", _rot(3), _rot(1)),
    DihedralTransform("reflect_horizontal", _flip_horizontal, _flip_horizontal),
    DihedralTransform("reflect_vertical", _flip_vertical, _flip_vertical),
    DihedralTransform("reflect_main_diagonal", _main_diagonal, _main_diagonal),
    DihedralTransform("reflect_anti_diagonal", _anti_diagonal, _anti_diagonal),
)


def predict_with_tta(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images: np.ndarray,
    n_augmentations: int = 8,
) -> np.ndarray:
    """Average predictions from exactly the first ``n_augmentations`` transforms."""

    if not isinstance(n_augmentations, int) or isinstance(n_augmentations, bool):
        raise TypeError("n_augmentations must be an integer")
    if not 1 <= n_augmentations <= len(DIHEDRAL_TRANSFORMS):
        raise ValueError(f"n_augmentations must be in [1, {len(DIHEDRAL_TRANSFORMS)}]")
    array = np.asarray(images)
    if array.ndim != 4:
        raise ValueError(f"Expected [N,H,W,C] images, got {array.shape}")
    if n_augmentations > 1 and array.shape[1] != array.shape[2]:
        raise ValueError("Rotational and diagonal dihedral TTA requires square inputs")
    restored: list[np.ndarray] = []
    for transform in DIHEDRAL_TRANSFORMS[:n_augmentations]:
        transformed = np.ascontiguousarray(transform.forward(array))
        prediction = np.asarray(predict_fn(transformed), dtype=np.float32)
        if prediction.ndim != 4 or prediction.shape[-1] != 1:
            raise ValueError(
                f"Prediction for {transform.name} must be a rank-4 single-channel batch, "
                f"got {prediction.shape}"
            )
        restored_prediction = np.ascontiguousarray(transform.inverse(prediction))
        if restored_prediction.shape[:3] != array.shape[:3]:
            raise ValueError(
                f"Prediction shape after inverse for {transform.name} is {restored_prediction.shape}; "
                f"expected spatial shape {array.shape[:3]}"
            )
        restored.append(restored_prediction)
    return np.mean(np.stack(restored, axis=0), axis=0, dtype=np.float32)
