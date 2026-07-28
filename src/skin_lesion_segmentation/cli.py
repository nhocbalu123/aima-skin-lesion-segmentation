"""Command-line entry point for prepare, train, evaluate, and submit workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible binary skin-lesion segmentation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate pairing and write split manifests")
    prepare.add_argument("--config", required=True, type=Path)

    train_parser = subparsers.add_parser("train", help="Train and evaluate a selected checkpoint")
    train_parser.add_argument("--config", required=True, type=Path)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate-predictions":
        from .evaluation import evaluate_saved_predictions

        result = evaluate_saved_predictions(args.predictions, threshold=args.threshold)
    else:
        config = ExperimentConfig.from_json(args.config)
        config.validate()
        if args.command == "prepare":
            from .training import prepare_manifests

            pairs, rows = prepare_manifests(config)
            result = {"pairs": len(pairs), "split_rows": len(rows), "output_dir": config.output_dir}
        elif args.command == "train":
            from .training import train

            result = train(config)
        elif args.command == "evaluate":
            from .evaluation import evaluate_checkpoint

            split_manifest = args.split_manifest or Path(config.output_dir) / "split_manifest.json"
            result = evaluate_checkpoint(config, args.checkpoint, split_manifest)
        elif args.command == "submit":
            from .inference import generate_submission

            postprocessing = args.postprocessing or Path(config.output_dir) / "chosen_postprocessing.json"
            result = generate_submission(config, args.checkpoint, postprocessing)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
