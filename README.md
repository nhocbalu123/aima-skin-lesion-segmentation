# AIMA Skin Lesion Segmentation

A compact TensorFlow/Keras pipeline for binary skin-lesion segmentation with Attention U-Net. The work was completed through AIMA Research Lab's guided Research Warm-up Program and is presented as a learner healthcare-AI portfolio project.

This is not a publication, internship, independent research study, original architecture, medical device, or clinically validated system.

## Status

The repository has been repaired for pairing correctness, evaluation integrity, reproducibility, testing, checkpoint reloadability, and submission validation. A controlled Kaggle rerun has been completed and compact evidence is stored in `results/controlled_rerun/`.

The original three-cell notebook is preserved at `notebooks/original_pipeline.ipynb`. The modular package and thin orchestration notebooks are the maintained implementation.

## Controlled rerun results

The final maintained submission uses the reloaded checkpoint, eight-transform TTA, raw threshold 0.5 predictions, fixed 512 x 512 submission masks, and C-order RLE.

| Evaluation | Score |
| --- | ---: |
| Offline validation macro Dice | 0.805922 |
| Offline validation macro IoU | 0.713368 |
| Public leaderboard | 0.80225 |
| Private leaderboard | 0.76879 |

The offline scores are reproduced from committed evaluator artifacts. The leaderboard values are author-reported because a Kaggle leaderboard export or screenshot is not committed.

Validation-only tuning selected threshold 0.35, minimum component size 64, and a 3 x 3 morphology kernel. That configuration reached validation macro Dice 0.808482, public 0.80166, and private 0.76888. The leaderboard difference was negligible, so the simpler raw threshold-0.5 path is the final default.

An earlier public 0.12522 / private 0.12470 submission is excluded from model comparison because masks were incorrectly encoded at source-image dimensions instead of the competition's 512 x 512 geometry. This was a submission-contract failure, not a model result.

## Author-reported historical results

The repository does not contain the original training logs, checkpoint, or leaderboard export needed to independently verify these historical values.

| Evaluation | Score |
| --- | ---: |
| Best original training-time validation soft Dice | 0.858775 |
| Public leaderboard | 0.82545 |
| Private leaderboard | 0.80996 |

The original validation value used a random 80/20 image-level split and a custom soft, batch-global Dice metric. It is not directly comparable with the repaired pipeline's thresholded per-image macro metrics or the hidden-test leaderboard scores.

## What was repaired

The maintained pipeline now:

- Matches images and masks by validated canonical sample IDs instead of separately sorting and zipping file lists.
- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff` case-insensitively.
- Splits paths before loading and streams batches from disk, including the final partial batch.
- Resizes masks with nearest-neighbour interpolation and explicitly binarises them.
- Uses patient or lesion groups only when supplied through explicit metadata; filenames are never treated as patient IDs.
- Detects exact duplicate decoded image content and rejects group or duplicate leakage across splits.
- Separates batch-global soft Dice used during training from thresholded per-image macro Dice and IoU used for final reporting.
- Selects checkpoints by validation loss and reloads the `.keras` checkpoint before offline validation and test inference.
- Keeps mixed precision optional, uses supported automatic loss scaling, and forces the final sigmoid output and loss reductions to float32.
- Implements eight unique dihedral test-time transformations with tested inverses.
- Makes RLE order, submission columns, ID suffix, and output mask geometry explicit.
- Validates submission IDs, row count, ordering, duplicates, and RLE bounds against the official sample submission.
- Records raw and validation-selected post-processing metrics separately.

## Controlled-run facts

- Validated image-mask pairs: 2,594
- Split: 2,075 training / 519 validation
- Split type: deterministic image-level split
- Reliable patient or lesion grouping metadata: unavailable
- Exact duplicate images detected: 0
- Epochs completed: 31
- Best checkpoint: epoch 23 by validation loss 0.260925
- Checkpoint SHA-256: `668b03c90e8cbb3e9119d649ee65290f3ba785f41f7cb08c2dbd9bef53e0dab8`

Patient or lesion independence cannot be guaranteed, so the split is not described as leakage-free.

## Model and loss

The model is a standard Attention U-Net with four encoder and decoder levels. Attention U-Net and focal-Tversky loss are published methods and are not claimed as original contributions.

The combined objective is:

```text
w = 2.0 where y = 1, otherwise 1.0
WeightedDice = (2 * sum(w * y * p) + eps) /
               (sum(w * y) + sum(w * p) + eps)
WeightedDiceLoss = 1 - WeightedDice

TP = sum(y * p)
FP = sum((1-y) * p)
FN = sum(y * (1-p))
Tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
FocalTverskyLoss = (1 - Tversky) ** gamma

CombinedLoss = 0.5 * WeightedDiceLoss + 0.5 * FocalTverskyLoss
```

Defaults are foreground weight 2.0, `alpha=0.3`, `beta=0.7`, and direct exponent `gamma=0.75`. Alpha weights false positives and beta weights false negatives. Logged Keras loss can also include L2 kernel-regularisation penalties.

## Repository structure

```text
configs/                                Run configuration
notebooks/
  original_pipeline.ipynb               Preserved historical notebook
  skin_lesion_segmentation.ipynb        Thin local orchestrator
  kaggle_controlled_rerun.ipynb         Gated verify/train/submit workflow
  kaggle_submission_only.ipynb          Checkpoint-based final submission workflow
results/controlled_rerun/               Compact controlled-run evidence
src/skin_lesion_segmentation/           Maintained implementation
scripts/                                Verification and checksum utilities
tests/                                  Synthetic CPU tests
```

## Supported environment

Pinned development target:

- Python 3.11
- TensorFlow 2.16.1
- Keras 3.0.5 through `tf.keras`
- NumPy 1.26.4
- OpenCV 4.10.0.84
- Albumentations 1.4.24
- pandas 2.2.2
- Matplotlib 3.9.2

The controlled Kaggle rerun recorded Python 3.11.13, TensorFlow 2.18.0, Keras 3.8.0, NumPy 1.26.4, and two visible GPUs. See `results/controlled_rerun/environment.json`. This difference is recorded rather than hidden.

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Dataset access

Obtain the dataset through its authorised competition or program channel. Do not commit or redistribute restricted images, masks, metadata, sample submissions, or generated competition submissions.

Default local layout:

```text
data/
  train/
    images/
    masks/
  test/
    images/
  sample_submission.csv
```

Fixed-size model resizing can distort aspect ratio. The competition submission contract used by this project requires masks encoded at 512 x 512, independent of the original source-image dimensions.

## Test and prepare

Edit `configs/default.json` for the authorised dataset paths. A group mapping can be a JSON object or a CSV with `sample_id,group_id` columns.

```bash
python -m skin_lesion_segmentation.cli prepare --config configs/default.json
pytest -q -W error
python scripts/verify_repository.py
python scripts/synthetic_smoke.py
```

## Train

```bash
python -m skin_lesion_segmentation.cli train --config configs/default.json
```

Training streams paths batch by batch. Validation and test data are not augmented. TensorFlow deterministic operations are enabled where supported, but exact repeatability can still vary across hardware, drivers, and library implementations.

## Evaluate

```bash
python -m skin_lesion_segmentation.cli evaluate \
  --config configs/default.json \
  --checkpoint artifacts/run/best_model.keras \
  --split-manifest artifacts/run/split_manifest.json
```

Saved predictions can be evaluated without another model pass:

```bash
python -m skin_lesion_segmentation.cli evaluate-predictions \
  --predictions artifacts/run/validation_predictions.npz \
  --threshold 0.5
```

Empty-mask convention:

- Empty ground truth and empty prediction: score 1.
- Empty ground truth and any predicted foreground: score 0.

## Generate the final submission

Confirm the official sample-submission path and columns. For this competition, the historical successful encoder confirms C-order RLE and fixed 512 x 512 masks.

Set these configuration fields:

```json
{
  "rle_order": "C",
  "rle_order_confirmed": true,
  "submission_mask_height": 512,
  "submission_mask_width": 512
}
```

Then use the maintained raw parameters:

```bash
python -m skin_lesion_segmentation.cli submit \
  --config configs/default.json \
  --checkpoint artifacts/run/best_model.keras \
  --postprocessing results/controlled_rerun/final_submission_parameters.json
```

The code validates columns, IDs, row order, duplicates, and output geometry against the official template.

## Reproducibility evidence

`results/controlled_rerun/` contains the effective training configuration, environment, seed, training history, checkpoint checksum, final submission parameters, compact metric summaries, and checksums for external artifacts.

Large or restricted artifacts are intentionally excluded from Git:

- `best_model.keras` (about 151 MB)
- `validation_predictions.npz` (about 102 MB)
- private images and masks
- pair and split manifests containing Kaggle paths
- full per-image evaluator outputs and qualitative grids containing dataset identifiers or imagery
- generated submission CSV files

## Limitations

- The split is image-level because reliable patient or lesion grouping metadata was unavailable.
- Exact duplicate detection does not replace a domain-specific near-duplicate or acquisition audit.
- Validation-selected post-processing did not produce a meaningful hidden-test benefit.
- The private leaderboard score was lower than validation and public performance, suggesting distribution shift or unstable generalisation.
- No architecture comparison, cross-validation, external validation, lesion-size study, or clinical validation has been completed.

## Method references

- Ozan Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas." https://arxiv.org/abs/1804.03999
- Nabila Abraham and Naimul Mefraz Khan. "A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation." https://arxiv.org/abs/1810.07842

## Non-clinical disclaimer

This learner pipeline is for education and portfolio review. It must not be used for diagnosis, treatment, triage, or patient care.
