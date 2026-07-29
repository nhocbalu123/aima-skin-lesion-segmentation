"""Path-based batch loading with separate image and mask interpolation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .data import SamplePair

try:
    from tensorflow.keras.utils import Sequence as _KerasSequence
except ImportError:
    class _KerasSequence:  # type: ignore[no-redef]
        def __init__(self, **_: object) -> None:
            pass


Augmentation = Callable[..., dict[str, np.ndarray]]


def load_image_mask(
    image_path: Path | str,
    mask_path: Path | str,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one RGB image and one binary mask as float32 arrays.

    ``image_size`` is ``(height, width)``. Images use area/linear interpolation;
    masks use nearest-neighbour interpolation and are explicitly binarised.
    Fixed-size resizing can distort aspect ratio when source dimensions differ.
    """

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise IOError(f"Could not read image: {image_path}")
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise IOError(f"Could not read mask: {mask_path}")

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if image_rgb.shape[0] >= height and image_rgb.shape[1] >= width else cv2.INTER_LINEAR
    image = cv2.resize(image_rgb, (width, height), interpolation=interpolation).astype(np.float32) / 255.0
    mask = cv2.resize(mask_raw, (width, height), interpolation=cv2.INTER_NEAREST)
    maximum = float(np.max(mask))
    threshold = maximum / 2.0 if maximum > 0.0 else 0.0
    mask = (mask > threshold).astype(np.float32)[..., None]
    unique = np.unique(mask)
    if not set(unique.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Mask is not binary after preprocessing: {unique.tolist()}")
    return image, mask


class PathBatchSequence(_KerasSequence):
    """A lightweight path sequence compatible with Keras ``fit`` semantics.

    It stores paths only, loads each batch on demand, includes the final partial
    batch, and shuffles deterministically through a persistent seeded RNG.
    """

    def __init__(
        self,
        pairs: Sequence[SamplePair],
        image_size: tuple[int, int],
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        augmentation: Augmentation | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not pairs:
            raise ValueError("pairs must not be empty")
        self._pairs = tuple(pairs)
        self.image_size = image_size
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augmentation = augmentation
        self._rng = np.random.default_rng(seed)
        self._indices = np.arange(len(self._pairs), dtype=np.int64)
        if self.shuffle:
            self._rng.shuffle(self._indices)

    def __len__(self) -> int:
        return math.ceil(len(self._pairs) / self.batch_size)

    @property
    def ordered_sample_ids(self) -> list[str]:
        return [self._pairs[int(index)].sample_id for index in self._indices]

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(batch_index, int):
            raise TypeError("batch_index must be an integer")
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self._pairs))
        indices = self._indices[start:stop]
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in indices:
            pair = self._pairs[int(index)]
            image, mask = load_image_mask(pair.image_path, pair.mask_path, self.image_size)
            if self.augmentation is not None:
                transformed = self.augmentation(image=image, mask=mask[..., 0])
                image = np.asarray(transformed["image"], dtype=np.float32)
                mask = (np.asarray(transformed["mask"]) > 0.5).astype(np.float32)[..., None]
            images.append(image)
            masks.append(mask)
        return np.stack(images).astype(np.float32), np.stack(masks).astype(np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._indices)



def load_image(image_path: Path | str, image_size: tuple[int, int]) -> np.ndarray:
    """Load one RGB test image as a normalised float32 array."""

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise IOError(f"Could not read image: {image_path}")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if image_rgb.shape[0] >= height and image_rgb.shape[1] >= width else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (width, height), interpolation=interpolation).astype(np.float32) / 255.0



def resize_binary_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a binary mask to an explicit ``(height, width)`` contract.

    Nearest-neighbour interpolation preserves categorical mask values. The
    output is always a two-dimensional uint8 array containing only 0 and 1.
    """

    height, width = target_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"target shape must be positive, got {target_shape}")
    binary = np.asarray(mask)
    if binary.ndim == 3 and binary.shape[-1] == 1:
        binary = binary[..., 0]
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D or [H,W,1], got {binary.shape}")
    resized = cv2.resize(
        (binary > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


class PathImageSequence(_KerasSequence):
    """Path-only inference sequence that includes the final partial batch."""

    def __init__(
        self,
        paths: Sequence[Path | str],
        image_size: tuple[int, int],
        batch_size: int,
    ) -> None:
        super().__init__()
        if not paths:
            raise ValueError("paths must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.paths = tuple(Path(path) for path in paths)
        self.image_size = image_size
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, batch_index: int) -> np.ndarray:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.paths))
        return np.stack([load_image(path, self.image_size) for path in self.paths[start:stop]]).astype(np.float32)
