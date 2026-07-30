# Result artifact contract

This document separates portable evidence from large, private, or disposable
experiment files.

## Training-run evidence

Each run directory contains:

- `run_summary.json`: model, seed, effective configuration, manifest SHA-256,
  source commit and dirty state, environment, hardware, exact command,
  parameter count, runtime, checkpoint rule, validation evidence, and
  checkpoint checksum;
- `history.csv`: per-epoch training and validation history;
- `best_model.keras`: selected model checkpoint.

The summary is the portable provenance record. The checkpoint is required for
comparison but is not repository evidence.

## Matched comparison evidence

Each seed-specific comparison directory contains:

- `comparison_summary.json`: aggregate Dice/IoU, paired-bootstrap intervals,
  provenance links, failure-case IDs, runtime, and output inventory;
- `per_image_metrics.csv`: paired image-level Dice, IoU, foreground fraction,
  border status, and false-positive/false-negative fractions;
- `slice_summary.json`: fixed lesion-size and border-touching summaries;
- `robustness_summary.json`: clean and predeclared synthetic perturbation
  summaries;
- `failure_cases.png`: qualitative predictions and error overlays;
- `unet_run_summary.json` and `attention_unet_run_summary.json`: validated
  copies of both run summaries.

Review restricted identifiers and images before publishing any per-image CSV or
figure.

## What may be committed

After the real experiment is complete and privacy/licensing checks pass, commit
small JSON/CSV summaries, effective configurations, histories, confidence
intervals, approved plots, and documentation.

## Do not commit

- `best_model.keras` or other checkpoints;
- datasets, images, masks, or private metadata;
- prediction arrays such as `.npy` or `.npz`;
- competition submissions;
- credentials, API tokens, or Kaggle configuration files;
- generated caches or copied runtime source trees;
- figures containing restricted imagery.

For excluded artifacts, preserve a checksum, byte size, origin, and retention
location outside Git when policy allows.
