from __future__ import annotations

import json

import pytest

from skin_lesion_segmentation.cli import main
from skin_lesion_segmentation.model import build_model


@pytest.mark.parametrize(
    ("model_name", "expected_parameters"),
    [("unet", 31_119), ("attention_unet", 31_508)],
)
def test_model_factory_builds_finite_float32_segmentation(
    model_name: str,
    expected_parameters: int,
) -> None:
    import numpy as np

    model = build_model(
        model_name,
        (32, 32, 3),
        base_filters=2,
        l2_coefficient=0.0,
        mixed_precision=False,
    )

    output = model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False).numpy()
    assert model.name == model_name
    assert output.shape == (1, 32, 32, 1)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert model.count_params() == expected_parameters


def test_attention_is_the_only_skip_path_difference() -> None:
    common = {
        "input_shape": (32, 32, 3),
        "base_filters": 2,
        "l2_coefficient": 0.0,
        "mixed_precision": False,
    }

    unet = build_model("unet", **common)
    attention = build_model("attention_unet", **common)
    unet_names = {layer.name for layer in unet.layers}
    attention_names = {layer.name for layer in attention.layers}

    assert not any(name.startswith("attention") for name in unet_names)
    assert {"attention1_multiply", "attention4_multiply"} <= attention_names
    for required in ("encoder1_conv1", "bottleneck_conv2", "decoder4_conv2", "segmentation"):
        assert required in unet_names
        assert required in attention_names
    assert attention.count_params() > unet.count_params()


def test_model_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resunet", (32, 32, 3))


@pytest.mark.parametrize("model_name", ["unet", "attention_unet"])
def test_smoke_cli_executes_requested_architecture(
    model_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke", "--model", model_name]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == model_name
    assert result["shape"] == [1, 32, 32, 1]
    assert result["dtype"] == "float32"
    assert result["finite"] is True
