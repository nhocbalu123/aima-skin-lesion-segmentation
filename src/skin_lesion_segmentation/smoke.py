"""Small CPU model-construction smoke checks."""

from __future__ import annotations

from typing import Any


def run_model_smoke(model_name: str, *, image_size: int = 32, base_filters: int = 2) -> dict[str, Any]:
    """Build one model and execute a finite float32 forward pass."""

    import numpy as np

    from .model import build_model

    model = build_model(
        model_name,
        (image_size, image_size, 3),
        base_filters=base_filters,
        l2_coefficient=0.0,
        mixed_precision=False,
    )
    values = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
    output = model(values, training=False).numpy()
    return {
        "model": model.name,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output).all()),
        "parameters": int(model.count_params()),
    }
