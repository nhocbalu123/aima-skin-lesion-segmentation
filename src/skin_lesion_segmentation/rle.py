"""Explicit, configurable run-length encoding."""

from __future__ import annotations

import numpy as np


def _validate_order(order: str) -> str:
    normalised = str(order).upper()
    if normalised not in {"C", "F"}:
        raise ValueError("RLE order must be 'C' or 'F'")
    return normalised


def encode_rle(mask: np.ndarray, order: str) -> str:
    """Encode a binary mask using one-based starts and explicit flatten order."""

    order = _validate_order(order)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D, got {binary.shape}")
    pixels = binary.ravel(order=order)
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(int(value)) for value in changes)


def decode_rle(rle: str, shape: tuple[int, int], order: str) -> np.ndarray:
    """Decode one-based start/length RLE into a binary mask."""

    order = _validate_order(order)
    total = int(shape[0]) * int(shape[1])
    flat = np.zeros(total, dtype=np.uint8)
    text = str(rle).strip()
    if text:
        tokens = [int(token) for token in text.split()]
        if len(tokens) % 2:
            raise ValueError("RLE must contain start/length pairs")
        starts = np.asarray(tokens[0::2], dtype=np.int64) - 1
        lengths = np.asarray(tokens[1::2], dtype=np.int64)
        if np.any(starts < 0) or np.any(lengths < 0) or np.any(starts + lengths > total):
            raise ValueError("RLE run exceeds target mask bounds")
        for start, length in zip(starts, lengths, strict=True):
            flat[start : start + length] = 1
    return flat.reshape(shape, order=order)
