from __future__ import annotations

import numpy as np
import pytest

from skin_lesion_segmentation.rle import decode_rle, encode_rle


@pytest.mark.parametrize("order", ["C", "F"])
def test_asymmetric_mask_round_trip(order: str) -> None:
    mask = np.zeros((3, 5), dtype=np.uint8)
    mask[0, 1] = 1
    mask[2, 3:5] = 1
    encoded = encode_rle(mask, order=order)
    decoded = decode_rle(encoded, shape=mask.shape, order=order)
    assert np.array_equal(decoded, mask)


def test_configured_orders_are_not_silently_equivalent() -> None:
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 2] = 1
    assert encode_rle(mask, order="C") != encode_rle(mask, order="F")


def test_invalid_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="order"):
        encode_rle(np.zeros((2, 2)), order="A")


def test_default_configured_order_round_trip() -> None:
    from pathlib import Path
    from skin_lesion_segmentation.config import ExperimentConfig

    config = ExperimentConfig.from_json(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 2] = 1
    assert np.array_equal(
        decode_rle(encode_rle(mask, config.rle_order), mask.shape, config.rle_order),
        mask,
    )
