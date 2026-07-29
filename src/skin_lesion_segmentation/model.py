"""Compact Attention U-Net using one consistent ``tf.keras`` stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .losses import combined_segmentation_loss
from .metrics import soft_dice_batch_global_tf


def _register_custom_objects(tf: Any) -> None:
    registry = tf.keras.utils.get_custom_objects()
    package = "skin_lesion_segmentation"
    objects = {
        "combined_segmentation_loss": combined_segmentation_loss,
        "soft_dice_batch_global_tf": soft_dice_batch_global_tf,
    }
    for name, obj in objects.items():
        tf.keras.utils.register_keras_serializable(package=package, name=name)(obj)
        registry[name] = obj
        registry[f"{package}>{name}"] = obj


def build_attention_unet(
    input_shape: tuple[int, int, int],
    *,
    base_filters: int = 32,
    l2_coefficient: float = 1e-4,
    mixed_precision: bool = False,
):
    """Build a standard Attention U-Net with a float32 sigmoid output.

    This is an implementation of published methods, not a claim of an original
    architecture. Spatial dimensions must be divisible by 16.
    """

    import tensorflow as tf

    if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError("input_shape must be (height, width, 3)")
    if input_shape[0] % 16 or input_shape[1] % 16:
        raise ValueError("input height and width must be divisible by 16")
    if base_filters <= 0 or l2_coefficient < 0:
        raise ValueError("base_filters must be positive and l2_coefficient non-negative")

    tf.keras.mixed_precision.set_global_policy("mixed_float16" if mixed_precision else "float32")
    regularizer = tf.keras.regularizers.l2(l2_coefficient) if l2_coefficient else None

    def conv_block(x: Any, filters: int, name: str) -> Any:
        for index in (1, 2):
            x = tf.keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name=f"{name}_conv{index}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{index}")(x)
            x = tf.keras.layers.Activation("relu", name=f"{name}_relu{index}")(x)
        return x

    def attention_gate(skip: Any, gating: Any, filters: int, name: str) -> Any:
        theta = tf.keras.layers.Conv2D(filters, 1, padding="same", name=f"{name}_theta")(skip)
        phi = tf.keras.layers.Conv2D(filters, 1, padding="same", name=f"{name}_phi")(gating)
        merged = tf.keras.layers.Add(name=f"{name}_add")([theta, phi])
        merged = tf.keras.layers.Activation("relu", name=f"{name}_relu")(merged)
        coefficient = tf.keras.layers.Conv2D(1, 1, padding="same", name=f"{name}_psi")(merged)
        coefficient = tf.keras.layers.Activation("sigmoid", name=f"{name}_sigmoid")(coefficient)
        return tf.keras.layers.Multiply(name=f"{name}_multiply")([skip, coefficient])

    inputs = tf.keras.Input(shape=input_shape, name="image")
    skips: list[Any] = []
    x = inputs
    for level, filters in enumerate(
        (base_filters, base_filters * 2, base_filters * 4, base_filters * 8),
        start=1,
    ):
        x = conv_block(x, filters, f"encoder{level}")
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"pool{level}")(x)

    x = conv_block(x, base_filters * 16, "bottleneck")
    for level, (skip, filters) in enumerate(
        zip(reversed(skips), (base_filters * 8, base_filters * 4, base_filters * 2, base_filters), strict=True),
        start=1,
    ):
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"up{level}")(x)
        attended = attention_gate(skip, x, max(filters // 2, 1), f"attention{level}")
        x = tf.keras.layers.Concatenate(name=f"concat{level}")([x, attended])
        x = conv_block(x, filters, f"decoder{level}")

    outputs = tf.keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        dtype="float32",
        name="segmentation",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attention_unet")
    _register_custom_objects(tf)
    return model


def load_model(path: Path | str, *, compile: bool = True):
    """Reload a modern ``.keras`` checkpoint with tested custom objects."""

    import tensorflow as tf

    _register_custom_objects(tf)
    return tf.keras.models.load_model(
        Path(path),
        custom_objects={
            "combined_segmentation_loss": combined_segmentation_loss,
            "soft_dice_batch_global_tf": soft_dice_batch_global_tf,
        },
        compile=compile,
    )
