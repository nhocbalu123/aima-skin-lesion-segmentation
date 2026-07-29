from __future__ import annotations

import pytest

from skin_lesion_segmentation.config import ExperimentConfig


def test_config_uses_l2_name_and_requires_explicit_rle_order() -> None:
    config = ExperimentConfig()
    assert config.l2_coefficient == 1e-4
    assert not hasattr(config, "weight_decay")
    config.validate()


def test_invalid_rle_order_is_rejected() -> None:
    config = ExperimentConfig(rle_order="A")
    with pytest.raises(ValueError, match="rle_order"):
        config.validate()


def test_rotational_tta_configuration_requires_square_images() -> None:
    with pytest.raises(ValueError, match="square"):
        ExperimentConfig(image_height=192, image_width=256, n_tta=8).validate()


def test_identity_only_tta_allows_rectangular_images() -> None:
    ExperimentConfig(image_height=192, image_width=256, n_tta=1).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_grid", []),
        ("min_component_sizes", []),
        ("morphology_kernels", []),
    ],
)
def test_parameter_grids_must_not_be_empty(field: str, value: list[object]) -> None:
    config = ExperimentConfig(**{field: value})
    with pytest.raises(ValueError, match="must not be empty"):
        config.validate()


def test_model_dimensions_must_be_divisible_by_sixteen() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        ExperimentConfig(image_height=250, image_width=250).validate()


def test_submission_mask_dimensions_are_explicit_and_positive() -> None:
    config = ExperimentConfig()
    assert config.submission_mask_height == 512
    assert config.submission_mask_width == 512
    config.validate()

    with pytest.raises(ValueError, match="submission mask dimensions"):
        ExperimentConfig(submission_mask_height=0).validate()

