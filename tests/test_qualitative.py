from __future__ import annotations

from pathlib import Path

import numpy as np

from skin_lesion_segmentation.qualitative import representative_indices, save_qualitative_grid


def test_representative_rule_includes_weak_median_and_strong() -> None:
    indices = representative_indices([0.8, 0.1, 0.5, 0.9, 0.4], count=3)
    assert indices == [1, 2, 3]


def test_qualitative_grid_is_saved(tmp_path: Path) -> None:
    images = np.zeros((3, 8, 8, 3), dtype=np.float32)
    truth = np.zeros((3, 8, 8, 1), dtype=np.float32)
    truth[:, 2:6, 2:6, 0] = 1.0
    probabilities = truth * 0.9
    output = tmp_path / "grid.png"

    selected = save_qualitative_grid(
        images,
        truth,
        probabilities,
        output,
        threshold=0.5,
        min_component_size=0,
        morphology_kernel=0,
        sample_ids=["a", "b", "c"],
        count=3,
    )

    assert output.exists() and output.stat().st_size > 0
    assert selected == ["a", "b", "c"]
