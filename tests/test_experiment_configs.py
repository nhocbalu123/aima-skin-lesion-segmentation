from __future__ import annotations

from pathlib import Path

from skin_lesion_segmentation.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_rerun_config_preserves_historical_protocol() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs" / "controlled_rerun.json")

    config.validate()
    assert config.model == "attention_unet"
    assert config.validation_fraction == 0.2
    assert config.n_tta == 8
    assert config.threshold_grid == [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]


def test_model_comparison_config_freezes_primary_policy() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs" / "model_comparison.json")

    config.validate()
    assert config.validation_fraction == 0.15
    assert config.internal_test_fraction == 0.15
    assert config.split_seed == 20260730
    assert config.seed == 42
    assert config.n_tta == 1
    assert config.threshold_grid == [0.5]
    assert config.min_component_sizes == [0]
    assert config.morphology_kernels == [0]


def test_default_config_contains_every_typed_field() -> None:
    config = ExperimentConfig.from_json(ROOT / "configs" / "default.json")

    config.validate()
    assert set(config.to_dict()) == set(ExperimentConfig().to_dict())
