from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_ascii_only_and_has_no_unclosed_fence() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text.encode("ascii")
    assert text.count("```") % 2 == 0


def test_notebook_is_valid_thin_json_without_saved_outputs() -> None:
    notebook_path = ROOT / "notebooks" / "skin_lesion_segmentation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) <= 6
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)


def test_kaggle_controlled_rerun_notebook_is_complete_and_clean() -> None:
    notebook_path = ROOT / "notebooks" / "kaggle_controlled_rerun.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in [
        "Kaggle controlled rerun",
        "RUN_TRAINING",
        "RUN_SUBMISSION",
        "TRAIN_IMAGE_DIR",
        "MASK_DIR",
        "TEST_IMAGE_DIR",
        "SAMPLE_SUBMISSION_PATH",
        "submission_id_suffix",
        "rle_order_confirmed",
        "pytest -q -W error",
        "prepare --config",
        "train --config",
        "submit --config",
        "final_validation_metrics.json",
        "qualitative_validation.png",
    ]:
        assert required in source
