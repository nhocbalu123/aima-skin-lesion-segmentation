from __future__ import annotations

from pathlib import Path

from skin_lesion_segmentation.cli import apply_training_overrides, build_parser
from skin_lesion_segmentation.config import ExperimentConfig


def test_train_cli_accepts_architecture_seed_output_and_shared_manifest() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            "configs/model_comparison.json",
            "--split-manifest",
            "artifacts/comparison_manifest.json",
            "--model",
            "unet",
            "--seed",
            "43",
            "--output-dir",
            "artifacts/unet-43",
        ]
    )
    config = apply_training_overrides(ExperimentConfig(), args)

    assert args.split_manifest == Path("artifacts/comparison_manifest.json")
    assert config.model == "unet"
    assert config.seed == 43
    assert config.output_dir == "artifacts/unet-43"


def test_prepare_comparison_cli_has_one_explicit_manifest_output() -> None:
    args = build_parser().parse_args(
        [
            "prepare-comparison",
            "--config",
            "configs/model_comparison.json",
            "--output-manifest",
            "artifacts/comparison_manifest.json",
        ]
    )

    assert args.output_manifest == Path("artifacts/comparison_manifest.json")


def test_compare_cli_requires_both_run_summaries_and_checkpoints() -> None:
    args = build_parser().parse_args(
        [
            "compare",
            "--config",
            "configs/model_comparison.json",
            "--split-manifest",
            "artifacts/comparison_manifest.json",
            "--unet-run-summary",
            "runs/unet/run_summary.json",
            "--attention-run-summary",
            "runs/attention/run_summary.json",
            "--unet-checkpoint",
            "runs/unet/best_model.keras",
            "--attention-checkpoint",
            "runs/attention/best_model.keras",
            "--output-dir",
            "results/seed-42",
            "--bootstrap-seed",
            "99",
            "--bootstrap-resamples",
            "2000",
        ]
    )

    assert args.unet_run_summary == Path("runs/unet/run_summary.json")
    assert args.attention_run_summary == Path("runs/attention/run_summary.json")
    assert args.unet_checkpoint == Path("runs/unet/best_model.keras")
    assert args.attention_checkpoint == Path("runs/attention/best_model.keras")
    assert args.bootstrap_seed == 99
    assert args.bootstrap_resamples == 2000
