"""Typed experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    image_dir: str = "data/train/images"
    mask_dir: str = "data/train/masks"
    test_image_dir: str = "data/test/images"
    sample_submission_path: str | None = None
    group_mapping_path: str | None = None
    output_dir: str = "artifacts/run"
    mask_suffix: str = "_segmentation"
    image_height: int = 256
    image_width: int = 256
    batch_size: int = 8
    epochs: int = 50
    base_filters: int = 32
    early_stopping_patience: int = 8
    validation_fraction: float = 0.2
    seed: int = 42
    learning_rate: float = 1e-3
    l2_coefficient: float = 1e-4
    mixed_precision: bool = False
    rle_order: str = "C"
    rle_order_confirmed: bool = False
    submission_id_column: str = "id"
    submission_mask_column: str = "segmentation"
    submission_id_suffix: str = ""
    submission_mask_height: int = 512
    submission_mask_width: int = 512
    n_tta: int = 8
    threshold_grid: list[float] = field(default_factory=lambda: [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    min_component_sizes: list[int] = field(default_factory=lambda: [0, 16, 32, 64])
    morphology_kernels: list[int] = field(default_factory=lambda: [0, 3])

    @classmethod
    def from_json(cls, path: Path | str) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def validate(self) -> None:
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("image dimensions must be divisible by 16 for Attention U-Net")
        if self.batch_size <= 0 or self.epochs <= 0 or self.base_filters <= 0:
            raise ValueError("batch_size, epochs, and base_filters must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.learning_rate <= 0.0 or self.l2_coefficient < 0.0:
            raise ValueError("learning_rate must be positive and l2_coefficient non-negative")
        if self.rle_order.upper() not in {"C", "F"}:
            raise ValueError("rle_order must be explicitly set to 'C' or 'F'")
        if not 1 <= self.n_tta <= 8:
            raise ValueError("n_tta must be in [1, 8]")
        if self.n_tta > 1 and self.image_height != self.image_width:
            raise ValueError("Rotational and diagonal dihedral TTA requires square image dimensions")
        if not self.submission_id_column or not self.submission_mask_column:
            raise ValueError("submission column names must be non-empty")
        if self.submission_mask_height <= 0 or self.submission_mask_width <= 0:
            raise ValueError("submission mask dimensions must be positive")
        if not self.threshold_grid or not self.min_component_sizes or not self.morphology_kernels:
            raise ValueError("threshold and post-processing parameter grids must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.threshold_grid):
            raise ValueError("threshold_grid values must be in [0, 1]")
        if any(value < 0 for value in self.min_component_sizes):
            raise ValueError("min_component_sizes must be non-negative")
        if any(value < 0 or (value > 0 and value % 2 == 0) for value in self.morphology_kernels):
            raise ValueError("morphology_kernels must be zero or positive odd integers")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
