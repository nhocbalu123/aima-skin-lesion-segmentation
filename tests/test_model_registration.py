from __future__ import annotations

from types import SimpleNamespace

from skin_lesion_segmentation.model import _register_custom_objects


class _FakeUtils:
    def __init__(self) -> None:
        self.registry: dict[str, object] = {}
        self.registered: list[tuple[str, str, object]] = []

    def get_custom_objects(self) -> dict[str, object]:
        return self.registry

    def register_keras_serializable(self, *, package: str, name: str):
        def decorator(obj: object) -> object:
            self.registered.append((package, name, obj))
            return obj

        return decorator


def test_custom_objects_are_available_by_plain_and_serialized_names() -> None:
    utils = _FakeUtils()
    fake_tf = SimpleNamespace(keras=SimpleNamespace(utils=utils))

    _register_custom_objects(fake_tf)

    assert "combined_segmentation_loss" in utils.registry
    assert "soft_dice_batch_global_tf" in utils.registry
    assert "skin_lesion_segmentation>combined_segmentation_loss" in utils.registry
    assert "skin_lesion_segmentation>soft_dice_batch_global_tf" in utils.registry
    assert {(package, name) for package, name, _ in utils.registered} == {
        ("skin_lesion_segmentation", "combined_segmentation_loss"),
        ("skin_lesion_segmentation", "soft_dice_batch_global_tf"),
    }
