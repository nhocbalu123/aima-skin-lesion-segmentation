"""Randomness controls and environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence
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
    device_details: list[dict[str, Any]] = []
    try:
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices()
        devices = [device.name for device in physical_devices]
        for device in physical_devices:
            details: dict[str, Any] = {}
            try:
                raw_details = tf.config.experimental.get_device_details(device)
                details = {
                    str(key): (
                        list(value)
                        if isinstance(value, tuple)
                        else value
                        if isinstance(value, (str, int, float, bool, type(None)))
                        else str(value)
                    )
                    for key, value in raw_details.items()
                }
            except (AttributeError, RuntimeError, ValueError):
                details = {}
            device_details.append(
                {
                    "device_name": device.name,
                    "device_type": device.device_type,
                    "details": details,
                }
            )
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
        "tensorflow_device_details": device_details,
    }


def command_record(argv: Sequence[object]) -> dict[str, Any]:
    """Record an argument vector without losing shell-relevant boundaries."""

    values = [str(value) for value in argv]
    if not values:
        raise ValueError("argv must not be empty")
    return {"argv": values, "shell": shlex.join(values)}


def collect_git_provenance(root: Path | str) -> dict[str, Any]:
    """Capture the exact Git commit and whether tracked/untracked files differ."""

    repository = Path(root).resolve()
    try:
        commit_process = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_process = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "commit": commit_process.stdout.strip().lower(),
            "dirty": bool(status_process.stdout.strip()),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        exported = repository / "SOURCE_COMMIT"
        if not exported.is_file():
            raise RuntimeError(
                "Git metadata is unavailable and SOURCE_COMMIT is missing"
            ) from None
        commit = exported.read_text(encoding="ascii").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("SOURCE_COMMIT is not a full Git commit hash")
        return {"commit": commit, "dirty": False}
