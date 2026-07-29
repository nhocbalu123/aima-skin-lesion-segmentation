from __future__ import annotations

import random

import numpy as np

from skin_lesion_segmentation.reproducibility import collect_environment, set_global_determinism


def test_seed_repeats_python_and_numpy_sequences() -> None:
    set_global_determinism(17)
    first = (random.random(), np.random.random())
    set_global_determinism(17)
    second = (random.random(), np.random.random())
    assert first == second


def test_environment_capture_contains_core_fields() -> None:
    environment = collect_environment()
    assert environment["python_version"]
    assert environment["platform"]
    assert "numpy" in environment["packages"]
