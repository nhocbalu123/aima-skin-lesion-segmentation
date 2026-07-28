"""Randomness controls and environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from typing import Any

import numpy as np

TRACKED_PACKAGES = (
    "numpy",
    "tensorflow",
    "keras",
    "opencv-python-headless",
    "opencv-python",
    "albumentations",
    "pandas",
    "scikit-learn",
    "matplotlib",
)


def set_global_determinism(seed: int, *, enable_tensorflow_determinism: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, TensorFlow, and deterministic TF operations when available."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tensorflow_status = "not_installed"
    deterministic_ops = False
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tensorflow_status = str(tf.__version__)
        if enable_tensorflow_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
                deterministic_ops = True
            except (AttributeError, RuntimeError):
                deterministic_ops = False
    except ImportError:
        pass
    return {
        "seed": seed,
        "tensorflow": tensorflow_status,
        "tensorflow_deterministic_ops_enabled": deterministic_ops,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect software, operating-system, CPU, and TensorFlow device information."""

    packages: dict[str, str | None] = {name: _package_version(name) for name in TRACKED_PACKAGES}
    # Ensure NumPy is always represented even in unusual editable environments.
    packages["numpy"] = packages.get("numpy") or np.__version__
    devices: list[str] = []
    try:
        import tensorflow as tf

        devices = [device.name for device in tf.config.list_physical_devices()]
    except ImportError:
        pass
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "tensorflow_devices": devices,
    }
