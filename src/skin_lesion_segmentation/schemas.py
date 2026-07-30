"""Validation for lightweight experiment evidence."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_MODELS = {"unet", "attention_unet"}


class ResultsSchemaError(ValueError):
    """Raised when an experiment artifact is incomplete or inconsistent."""


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultsSchemaError(f"{field} must be an object")
    return value


def _require_hex(value: object, field: str, pattern: re.Pattern[str]) -> str:
    text = str(value)
    if pattern.fullmatch(text) is None:
        raise ResultsSchemaError(f"{field} must be a lowercase hexadecimal digest")
    return text


def validate_run_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one training-run summary and return a defensive copy."""

    summary = _require_mapping(value, "run summary")
    required = {
        "schema_version",
        "model_name",
        "seed",
        "manifest_sha256",
        "parameter_count",
        "effective_config",
        "checkpoint_selection",
        "checkpoint",
        "epochs_completed",
        "best_epoch",
        "best_validation_loss",
        "validation_metrics",
        "runtime_seconds",
        "command",
        "source",
        "environment",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ResultsSchemaError("Missing run-summary fields: " + ", ".join(missing))
    if "internal_test_metrics" in summary or "test_metrics" in summary:
        raise ResultsSchemaError(
            "Training summaries must not contain internal-test metrics"
        )
    if summary["schema_version"] != 1:
        raise ResultsSchemaError("schema_version must be 1")
    model_name = str(summary["model_name"])
    if model_name not in SUPPORTED_MODELS:
        raise ResultsSchemaError("model_name is not supported")
    seed = summary["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ResultsSchemaError("seed must be an integer")
    _require_hex(summary["manifest_sha256"], "manifest_sha256", HEX_64)
    parameters = summary["parameter_count"]
    if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters <= 0:
        raise ResultsSchemaError("parameter_count must be a positive integer")
    runtime = summary["runtime_seconds"]
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or float(runtime) < 0.0
    ):
        raise ResultsSchemaError("runtime_seconds must be non-negative")

    config = _require_mapping(summary["effective_config"], "effective_config")
    if config.get("model") != model_name:
        raise ResultsSchemaError(
            "effective_config.model must match model_name"
        )
    selection = _require_mapping(
        summary["checkpoint_selection"],
        "checkpoint_selection",
    )
    for field in ("monitor", "mode", "rule", "early_stopping_patience"):
        if field not in selection:
            raise ResultsSchemaError(f"checkpoint_selection.{field} is required")
    checkpoint = _require_mapping(summary["checkpoint"], "checkpoint")
    for field in ("path", "size_bytes", "sha256"):
        if field not in checkpoint:
            raise ResultsSchemaError(f"checkpoint.{field} is required")
    _require_hex(checkpoint["sha256"], "checkpoint.sha256", HEX_64)
    if not str(checkpoint["path"]):
        raise ResultsSchemaError("checkpoint.path must not be blank")
    if not isinstance(checkpoint["size_bytes"], int) or checkpoint["size_bytes"] <= 0:
        raise ResultsSchemaError("checkpoint.size_bytes must be positive")

    validation = _require_mapping(
        summary["validation_metrics"],
        "validation_metrics",
    )
    for field in ("threshold", "macro_dice", "macro_iou", "n_images"):
        if field not in validation:
            raise ResultsSchemaError(f"validation_metrics.{field} is required")
    command = _require_mapping(summary["command"], "command")
    if not isinstance(command.get("argv"), list) or not command.get("argv"):
        raise ResultsSchemaError("command.argv must be a non-empty list")
    if not isinstance(command.get("shell"), str) or not command.get("shell"):
        raise ResultsSchemaError("command.shell must not be blank")
    source = _require_mapping(summary["source"], "source")
    _require_hex(source.get("commit"), "source.commit", HEX_40)
    if not isinstance(source.get("dirty"), bool):
        raise ResultsSchemaError("source.dirty must be boolean")
    environment = _require_mapping(summary["environment"], "environment")
    for field in (
        "python_version",
        "packages",
        "tensorflow_devices",
        "tensorflow_device_details",
    ):
        if field not in environment:
            raise ResultsSchemaError(f"environment.{field} is required")
    return copy.deepcopy(dict(summary))
