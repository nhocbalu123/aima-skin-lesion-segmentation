"""Command-line entry point for prepare, train, evaluate, and submit workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible binary skin-lesion segmentation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Build and execute a small CPU model")
    smoke.add_argument("--model", required=True, choices=("unet", "attention_unet"))

    prepare = subparsers.add_parser("prepare", help="Validate pairing and write split manifests")
    prepare.add_argument("--config", required=True, type=Path)

    comparison_prepare = subparsers.add_parser(
        "prepare-comparison",
        help="Write one immutable train/validation/internal-test manifest",
    )
    comparison_prepare.add_argument("--config", required=True, type=Path)
    comparison_prepare.add_argument("--output-manifest", required=True, type=Path)

    train_parser = subparsers.add_parser("train", help="Train and evaluate a selected checkpoint")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--split-manifest", type=Path)
    train_parser.add_argument("--model", choices=("unet", "attention_unet"))
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--output-dir", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Offline evaluation of a saved checkpoint")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--split-manifest", type=Path)

    saved = subparsers.add_parser(
        "evaluate-predictions",
        help="Evaluate a saved validation_predictions.npz artifact",
    )
    saved.add_argument("--predictions", required=True, type=Path)
    saved.add_argument("--threshold", type=float, default=0.5)

    submit = subparsers.add_parser("submit", help="Generate and validate submission.csv")
    submit.add_argument("--config", required=True, type=Path)
    submit.add_argument("--checkpoint", required=True, type=Path)
    submit.add_argument("--postprocessing", type=Path)

    compare = subparsers.add_parser(
        "compare",
        help="Run one provenance-checked matched-seed internal-test comparison",
    )
    compare.add_argument("--config", required=True, type=Path)
    compare.add_argument("--split-manifest", required=True, type=Path)
    compare.add_argument("--unet-run-summary", required=True, type=Path)
    compare.add_argument("--attention-run-summary", required=True, type=Path)
    compare.add_argument("--unet-checkpoint", required=True, type=Path)
    compare.add_argument("--attention-checkpoint", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    compare.add_argument("--bootstrap-seed", type=int, default=20260730)
    compare.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def apply_training_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Apply only explicit run-matrix overrides to a base configuration."""

    changes: dict[str, object] = {}
    if args.model is not None:
        changes["model"] = args.model
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.output_dir is not None:
        changes["output_dir"] = str(args.output_dir)
    return replace(config, **changes)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        from .smoke import run_model_smoke

        result = run_model_smoke(args.model)
    elif args.command == "evaluate-predictions":
        from .evaluation import evaluate_saved_predictions

        result = evaluate_saved_predictions(args.predictions, threshold=args.threshold)
    else:
        config = ExperimentConfig.from_json(args.config)
        if args.command == "train":
            config = apply_training_overrides(config, args)
        config.validate()
        if args.command == "prepare":
            from .training import prepare_manifests

            pairs, rows = prepare_manifests(config)
            result = {"pairs": len(pairs), "split_rows": len(rows), "output_dir": config.output_dir}
        elif args.command == "prepare-comparison":
            from .splitting import manifest_sha256
            from .training import prepare_comparison_manifest

            pairs, rows = prepare_comparison_manifest(
                config,
                args.output_manifest,
            )
            result = {
                "pairs": len(pairs),
                "split_rows": len(rows),
                "manifest_sha256": manifest_sha256(rows),
                "output_manifest": str(args.output_manifest),
            }
        elif args.command == "train":
            from .training import train

            invoked_command = (
                list(sys.argv)
                if argv is None
                else ["skin-lesion-segmentation", *[str(value) for value in argv]]
            )
            result = train(
                config,
                split_manifest_path=args.split_manifest,
                command=invoked_command,
            )
        elif args.command == "evaluate":
            from .evaluation import evaluate_checkpoint

            split_manifest = args.split_manifest or Path(config.output_dir) / "split_manifest.json"
            result = evaluate_checkpoint(config, args.checkpoint, split_manifest)
        elif args.command == "submit":
            from .inference import generate_submission

            postprocessing = args.postprocessing or Path(config.output_dir) / "chosen_postprocessing.json"
            result = generate_submission(config, args.checkpoint, postprocessing)
        elif args.command == "compare":
            from .comparison import compare_runs

            result = compare_runs(
                config,
                args.split_manifest,
                args.unet_run_summary,
                args.attention_run_summary,
                args.unet_checkpoint,
                args.attention_checkpoint,
                args.output_dir,
                bootstrap_seed=args.bootstrap_seed,
                n_resamples=args.bootstrap_resamples,
            )
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
