from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from skin_lesion_segmentation.losses import (
    combined_segmentation_loss,
    combined_segmentation_loss_numpy,
)
from skin_lesion_segmentation.model import build_attention_unet, load_model


def test_model_output_and_loss_are_float32() -> None:
    model = build_attention_unet((32, 32, 3), base_filters=4, mixed_precision=True)
    x = tf.zeros((1, 32, 32, 3), dtype=tf.float32)
    y = tf.zeros((1, 32, 32, 1), dtype=tf.float32)
    prediction = model(x, training=False)
    loss = combined_segmentation_loss(y, prediction)
    assert prediction.dtype == tf.float32
    assert loss.dtype == tf.float32


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    from skin_lesion_segmentation.metrics import soft_dice_batch_global_tf

    model = build_attention_unet((32, 32, 3), base_filters=4, mixed_precision=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=combined_segmentation_loss,
        metrics=[soft_dice_batch_global_tf],
    )
    x = np.random.default_rng(1).random((1, 32, 32, 3), dtype=np.float32)
    y = np.zeros((1, 32, 32, 1), dtype=np.float32)

    # Build Adam's slot variables before saving. A compiled-but-untrained model
    # only has the optimizer iteration and learning-rate variables, which cannot
    # be restored into the fully built optimizer created by Keras at load time.
    model.train_on_batch(x, y)
    before = model.predict(x, verbose=0)
    path = tmp_path / "model.keras"
    model.save(path)
    restored = load_model(path, compile=True)
    after = restored.predict(x, verbose=0)
    assert np.allclose(before, after, atol=1e-6, rtol=1e-6)
    assert restored.loss is not None


def test_soft_training_metric_is_explicitly_batch_global_and_float32() -> None:
    from skin_lesion_segmentation.metrics import soft_dice_batch_global_tf

    y = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]], dtype=tf.float32)
    score = soft_dice_batch_global_tf(y, y)
    assert score.dtype == tf.float32
    assert float(score.numpy()) == pytest.approx(1.0)


def test_tensorflow_combined_loss_matches_numpy_reference() -> None:
    y_true = np.asarray([[[[1.0], [0.0]], [[1.0], [0.0]]]], dtype=np.float32)
    y_pred = np.asarray([[[[0.8], [0.2]], [[0.3], [0.1]]]], dtype=np.float32)

    expected = combined_segmentation_loss_numpy(y_true, y_pred)
    actual = float(combined_segmentation_loss(tf.constant(y_true), tf.constant(y_pred)).numpy())

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
