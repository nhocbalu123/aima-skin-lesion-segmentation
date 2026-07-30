#!/usr/bin/env python3
"""Static repository checks that do not require private data or a GPU."""

from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml"}
FORBIDDEN_README_PHRASES = (
    "Expected improvement",
    "Fixed NaN",
    "Leakage-free",
    "Clinically validated",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        fail(f"README is not ASCII-only: {error}")
    if text.count("```") % 2:
        fail("README contains an unclosed fenced code block")
    for phrase in FORBIDDEN_README_PHRASES:
        if phrase in text:
            fail(f"README contains forbidden claim: {phrase}")


def check_notebooks() -> None:
    expected = {
        "notebooks/kaggle_controlled_rerun.ipynb",
        "notebooks/original_pipeline.ipynb",
        "notebooks/skin_lesion_segmentation.ipynb",
    }
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.ipynb")
        if ".venv" not in path.parts
    }
    if actual != expected:
        fail(f"Unexpected notebook set: expected={sorted(expected)}, actual={sorted(actual)}")
    for relative in sorted(expected):
        path = ROOT / relative
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            fail(f"Notebook is not nbformat 4: {relative}")
        code_cells = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        if relative.endswith("skin_lesion_segmentation.ipynb") and len(code_cells) > 6:
            fail("Local notebook is no longer a thin orchestrator")
        for cell in code_cells:
            if cell.get("outputs"):
                fail(f"Notebook contains saved outputs: {relative}")
            if cell.get("execution_count") is not None:
                fail(f"Notebook contains execution counts: {relative}")

    kaggle = json.loads(
        (ROOT / "notebooks" / "kaggle_controlled_rerun.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in kaggle.get("cells", [])
    )
    for required in (
        "/kaggle/working",
        "SOURCE_COMMIT",
        "RUN_VERIFY",
        "RUN_PREPARE",
        "RUN_TRAINING",
        "RUN_COMPARISON",
        "RUN_SUBMISSION",
        "RLE_ORDER_CONFIRMED",
        "prepare-comparison",
        "--split-manifest",
        "attention_unet",
    ):
        if required not in source:
            fail(f"Kaggle notebook is missing workflow marker: {required}")


def check_experiment_configs() -> None:
    for name in ("default.json", "controlled_rerun.json", "model_comparison.json"):
        path = ROOT / "configs" / name
        if not path.is_file():
            fail(f"Missing experiment config: configs/{name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        for field in ("model", "validation_fraction", "internal_test_fraction", "split_seed"):
            if field not in value:
                fail(f"configs/{name} is missing '{field}'")
    comparison = json.loads(
        (ROOT / "configs" / "model_comparison.json").read_text(encoding="utf-8")
    )
    if (
        comparison.get("validation_fraction"),
        comparison.get("internal_test_fraction"),
        comparison.get("n_tta"),
        comparison.get("threshold_grid"),
    ) != (0.15, 0.15, 1, [0.5]):
        fail("Model-comparison config does not preserve the sealed primary policy")



def check_controlled_rerun_evidence() -> None:
    evidence = ROOT / "results" / "controlled_rerun"
    required = {
        "README.md",
        "validation_summary.json",
        "leaderboard_results.json",
        "final_submission_parameters.json",
        "training_history.csv",
        "environment.json",
        "checkpoint.json",
        "external_artifacts.json",
    }
    missing = sorted(name for name in required if not (evidence / name).is_file())
    if missing:
        fail(f"Controlled-rerun evidence is incomplete: {missing}")
    prohibited = {
        "best_model.keras",
        "validation_predictions.npz",
        "qualitative_validation.png",
        "validation_metrics_full.json",
    }
    present = sorted(name for name in prohibited if (evidence / name).exists())
    if present:
        fail(f"Large or restricted generated artifacts must not be committed: {present}")

    validation = json.loads((evidence / "validation_summary.json").read_text(encoding="utf-8"))
    raw = validation["offline_validation"]["raw_threshold_0_5"]
    if abs(float(raw["macro_dice"]) - 0.805922296461998) > 1e-12:
        fail("Controlled-rerun raw validation Dice does not match the recorded run")

    config = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    if (config.get("submission_mask_height"), config.get("submission_mask_width")) != (512, 512):
        fail("Default submission geometry must be explicitly fixed at 512 x 512")


def check_whitespace() -> None:
    ignored = {".pytest_cache", "__pycache__", ".git", ".venv"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in ignored for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.endswith((" ", "\t")):
                fail(f"Trailing whitespace: {path.relative_to(ROOT)}:{line_number}")


def main() -> int:
    if not compileall.compile_dir(ROOT / "src", quiet=1):
        fail("Source compilation failed")
    if not compileall.compile_dir(ROOT / "tests", quiet=1):
        fail("Test compilation failed")
    check_readme()
    check_notebooks()
    check_experiment_configs()
    check_controlled_rerun_evidence()
    check_whitespace()
    print("Repository static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
