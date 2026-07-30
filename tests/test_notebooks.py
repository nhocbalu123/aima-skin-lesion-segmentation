from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_repository_notebook_is_valid_and_output_free() -> None:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    relative = {path.relative_to(ROOT).as_posix() for path in notebooks}
    assert relative == {
        "notebooks/kaggle_controlled_rerun.ipynb",
        "notebooks/original_pipeline.ipynb",
        "notebooks/skin_lesion_segmentation.ipynb",
    }
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4, path
        assert isinstance(notebook.get("cells"), list), path
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, path
            assert cell.get("outputs") == [], path


def test_canonical_kaggle_notebook_exposes_sealed_workflow() -> None:
    path = ROOT / "notebooks" / "kaggle_controlled_rerun.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    for required in (
        "/kaggle/working",
        "SOURCE_COMMIT",
        "RUN_VERIFY",
        "RUN_PREPARE",
        "RUN_TRAINING",
        "RUN_COMPARISON",
        "prepare-comparison",
        "--split-manifest",
        "unet",
        "attention_unet",
        "compare",
    ):
        assert required in source
