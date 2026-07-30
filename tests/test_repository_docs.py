from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_ascii_only_and_has_no_unclosed_fence() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text.encode("ascii")
    assert text.count("```") % 2 == 0


def test_readme_separates_historical_controlled_and_unexecuted_comparison() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "Author-reported historical results",
        "Controlled rerun results",
        "Scientific model-comparison status",
        "0.805922",
        "0.80225",
        "0.76879",
        "No real U-Net versus Attention U-Net comparison has been executed",
        "Attention's incremental value remains unknown",
        "image-level",
    ):
        assert required in text
    assert "kaggle_submission_only.ipynb" not in text
    assert "AIMA_Skin_Lesion_Kaggle_Raw_Mask_Submission_v6.ipynb" not in text


def test_comparison_docs_define_exact_six_run_protocol_without_results() -> None:
    path = ROOT / "docs" / "MODEL_COMPARISON.md"
    text = path.read_text(encoding="utf-8")
    for model in ("unet", "attention_unet"):
        for seed in (42, 43, 44):
            assert f"--model {model}" in text
            assert f"--seed {seed}" in text
    for required in (
        "70/15/15",
        "minimum validation loss",
        "internal test",
        "paired bootstrap",
        "No real-data comparison result is reported",
        "python -m skin_lesion_segmentation.cli compare",
    ):
        assert required in text


def test_result_artifact_contract_lists_portable_lightweight_evidence() -> None:
    text = (ROOT / "docs" / "RESULT_ARTIFACTS.md").read_text(encoding="utf-8")
    for required in (
        "run_summary.json",
        "comparison_summary.json",
        "per_image_metrics.csv",
        "slice_summary.json",
        "robustness_summary.json",
        "best_model.keras",
        "Do not commit",
    ):
        assert required in text


def test_controlled_rerun_evidence_is_compact_and_complete() -> None:
    evidence = ROOT / "results" / "controlled_rerun"
    for name in (
        "README.md",
        "validation_summary.json",
        "leaderboard_results.json",
        "final_submission_parameters.json",
        "training_history.csv",
        "environment.json",
        "checkpoint.json",
        "external_artifacts.json",
    ):
        assert (evidence / name).is_file(), name
    for prohibited in (
        "best_model.keras",
        "validation_predictions.npz",
        "qualitative_validation.png",
        "validation_metrics_full.json",
    ):
        assert not (evidence / prohibited).exists()


def test_canonical_notebook_retains_separate_submission_gate() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "kaggle_controlled_rerun.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    for required in (
        "RUN_SUBMISSION",
        "RLE_ORDER_CONFIRMED",
        "Submission is not part of model selection",
    ):
        assert required in source


def test_ci_runs_strict_tests_and_both_model_smokes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )
    assert "pytest -q -W error" in workflow
    assert "smoke --model unet" in workflow
    assert "smoke --model attention_unet" in workflow
