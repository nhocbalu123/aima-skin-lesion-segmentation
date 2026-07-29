# Controlled rerun evidence

This directory contains compact evidence from the repaired single-split Kaggle rerun completed on 2026-07-29. Large generated artifacts are intentionally excluded from Git.

## Final reported pipeline

The final default uses the reloaded checkpoint, eight-transform TTA, raw threshold 0.5 predictions, fixed 512 x 512 submission masks, and C-order RLE.

| Evaluation | Score |
| --- | ---: |
| Offline validation macro Dice | 0.805922 |
| Offline validation macro IoU | 0.713368 |
| Public leaderboard | 0.80225 |
| Private leaderboard | 0.76879 |

The leaderboard values are author-reported because a Kaggle leaderboard export or screenshot is not committed.

## Post-processing ablation

Validation-selected post-processing used threshold 0.35, minimum component size 64, and a 3 x 3 morphology kernel. It raised offline macro Dice to 0.808482 but produced public/private scores of 0.80166/0.76888. The leaderboard difference was negligible, so the simpler raw threshold-0.5 path is the maintained default.

An earlier 0.12522 public / 0.12470 private submission is excluded from model comparison because it encoded masks at source-image dimensions instead of the required 512 x 512 geometry.

## Included evidence

- `validation_summary.json`: compact split, training, and offline metric summary.
- `training_history.csv`: 31 completed epochs; minimum validation loss at epoch 23.
- `training_effective_config.json`: configuration saved by the training run.
- `environment.json`: Kaggle runtime and package versions.
- `random_seed.json`: seed and deterministic-operation status.
- `checkpoint.json`: external checkpoint path, size, and SHA-256 checksum.
- `run_summary.json`: checkpoint-selection and run-completion summary.
- `final_submission_parameters.json`: maintained raw threshold-0.5 submission parameters.
- `leaderboard_results.json`: author-reported leaderboard results and the excluded geometry-error submission.
- `external_artifacts.json`: checksums and sizes for large or restricted generated artifacts retained outside Git.

## Deliberately excluded

The following remain outside Git because of size, restricted-data concerns, or both:

- `best_model.keras` (about 151 MB)
- `validation_predictions.npz` (about 102 MB)
- pair and split manifests containing Kaggle dataset paths
- qualitative grids containing dataset images and masks
- full per-image validation outputs containing dataset identifiers
- generated competition submission CSV files
- private images and masks

The checkpoint can be matched to this run using the SHA-256 value in `checkpoint.json`.
