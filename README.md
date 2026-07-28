# AIMA Skin Lesion Segmentation

A compact TensorFlow/Keras pipeline for binary skin-lesion segmentation with Attention U-Net. The project was completed through AIMA Research Lab's guided Research Warm-up Program and is presented as learner portfolio work in healthcare AI.

This is not a publication, internship, independent research study, original architecture, medical device, or clinically validated system.

## Current status

The repository code has been repaired for correctness, evaluation integrity, reproducibility, testing, and educational clarity. The repaired pipeline has not yet completed a controlled dataset rerun, so no repaired-pipeline score is reported.

The original three-cell notebook is preserved at `notebooks/original_pipeline.ipynb` for historical traceability. The tested package and thin orchestration notebook are the maintained implementation.

## Author-reported original-run results

The repository does not contain the original training logs, checkpoint, leaderboard export, or screenshot needed to independently verify these values. They are therefore labelled as author-reported original-run results.

| Evaluation | Score |
| --- | ---: |
| Best original training-time validation soft Dice | 0.858775 |
| Public leaderboard score | 0.82545 |
| Private leaderboard score | 0.80996 |

The value `0.858775` came from a random 80/20 image-level split and the original custom soft, batch-global Dice metric. It is not directly comparable with the public or private hidden-test leaderboard scores. These historical values must not be attributed to the repaired pipeline until a controlled rerun produces new evidence.

## Student scope and attribution

The repository demonstrates the application of established segmentation methods in a guided learner setting. Attention U-Net and focal-Tversky loss are published methods and are not claimed as original contributions.

The exact provenance of any starter code, guided snippets, or externally derived code in the historical notebook has not been documented. This repository therefore does not make a broader code-ownership claim. No licence has been added or changed pending provenance confirmation.

## Corrected engineering approach

The maintained pipeline now:

- Matches images and masks through validated canonical sample IDs instead of separately sorting and zipping file lists.
- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff` case-insensitively.
- Splits paths before loading data and streams batches from disk, including the final partial batch.
- Resizes masks with nearest-neighbour interpolation and explicitly binarises them.
- Uses explicit patient or lesion group metadata when available; it never infers patient IDs from filenames.
- Detects exact duplicate image content before splitting and rejects group or duplicate leakage across splits.
- Separates the soft batch-global training diagnostic from thresholded per-image macro Dice and IoU used for final validation reporting.
- Uses validation loss for checkpoint selection, then reloads the `.keras` checkpoint for offline evaluation.
- Keeps mixed precision optional, uses framework-supported automatic loss scaling, and forces the final sigmoid output and loss reductions to float32.
- Implements eight unique dihedral test-time transformations with tested inverses.
- Makes RLE flattening order explicit and blocks submission generation until the official order is confirmed.
- Restores each predicted test mask to the source image dimensions before RLE encoding.
- Selects threshold and post-processing parameters using validation predictions only, while retaining both raw and selected metrics.

## Model and loss

The model is a standard Attention U-Net with four encoder and decoder levels. The default repaired configuration uses a smaller base width than the historical notebook so the pipeline remains practical for learner hardware. This is a configuration change, not a reproduced comparison.

The combined training loss is:

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

The weighted-Dice foreground weight is `2.0`. In that term, a false negative lies on a foreground pixel and receives weight 2.0, while a false positive lies on a background pixel and receives weight 1.0. Focal-Tversky defaults are `alpha=0.3`, `beta=0.7`, and `gamma=0.75`: alpha weights false positives, beta weights false negatives, and the larger beta places more cost on missed lesion pixels. Here `gamma=0.75` is the direct exponent shown above. Logged Keras loss can also include L2 kernel-regularisation penalties.

## Repository structure

```text
configs/
  default.json                         Default run configuration
notebooks/
  original_pipeline.ipynb              Historical three-cell notebook
  skin_lesion_segmentation.ipynb       Thin maintained orchestrator
  kaggle_controlled_rerun.ipynb         Complete gated Kaggle rerun workflow
scripts/
  checkpoint_checksum.py               External checkpoint checksum utility
  synthetic_smoke.py                   Synthetic end-to-end smoke flow
  verify_repository.py                 Static repository checks
src/skin_lesion_segmentation/
  config.py                             Typed configuration and validation
  data.py                               Discovery and canonical pairing
  splitting.py                          Split manifest and leakage audit
  loading.py                            Path-based batch loading
  augmentations.py                      Training-only Albumentations pipeline
  model.py                              Attention U-Net and checkpoint loading
  losses.py                             Weighted Dice and focal-Tversky losses
  metrics.py                            Training diagnostic and offline metrics
  evaluation.py                         Validation evaluation and tuning
  postprocessing.py                     Component and morphology operations
  qualitative.py                        Fixed-rule qualitative grids
  tta.py                                Eight dihedral transformations
  rle.py                                Explicit C-order or F-order RLE
  submission.py                         Submission schema and ID validation
  inference.py                          Test inference and submission generation
  training.py                           Training orchestration and artifacts
  cli.py                                Command-line entry point
tests/                                  Synthetic CPU tests
```

## Supported environment

The pinned target is Python 3.11 with:

- TensorFlow 2.16.1
- Keras 3.0.5 through the `tf.keras` API
- NumPy 1.26.4
- OpenCV 4.10.0.84
- Albumentations 1.4.24
- pandas 2.2.2
- Matplotlib 3.9.2

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Dataset access and layout

Obtain the dataset through its authorised competition or program channel. Do not commit or redistribute restricted images, masks, metadata, or sample submissions.

Default layout:

```text
data/
  train/
    images/
    masks/
  test/
    images/
  sample_submission.csv
```

The default mask suffix is `_segmentation`. Change it only when the dataset naming contract confirms another suffix. Fixed-size resizing can distort aspect ratio when source dimensions differ from the configured height-to-width ratio.

## Prepare and test

Edit `configs/default.json` for the authorised dataset paths. A group mapping can be either a JSON object or a CSV with `sample_id,group_id` columns.

```bash
python -m skin_lesion_segmentation.cli prepare --config configs/default.json
pytest -q
python scripts/verify_repository.py
python scripts/synthetic_smoke.py
```

Without reliable group metadata, the split is image-level. Patient or lesion independence cannot be guaranteed, and the split must not be described as leakage-free.

For the recommended Kaggle workflow, upload the repaired repository archive as
a private Kaggle Dataset, attach the authorised competition data, and run
`notebooks/kaggle_controlled_rerun.ipynb`. Training and submission are gated so
the pairing and split manifests can be reviewed before GPU work begins.
Detailed steps are in `docs/KAGGLE_NOTEBOOK.md`.

## Train

```bash
python -m skin_lesion_segmentation.cli train --config configs/default.json
```

Training streams paths batch-by-batch. Validation and test data are not augmented. TensorFlow deterministic operations are enabled where supported, but exact repeatability can still vary across hardware, drivers, and library implementations.

## Evaluate a checkpoint

```bash
python -m skin_lesion_segmentation.cli evaluate \
  --config configs/default.json \
  --checkpoint artifacts/run/best_model.keras \
  --split-manifest artifacts/run/split_manifest.json
```

Saved validation probabilities can also be re-evaluated without TensorFlow or another model pass:

```bash
python -m skin_lesion_segmentation.cli evaluate-predictions \
  --predictions artifacts/run/validation_predictions.npz \
  --threshold 0.5
```

Final metrics are thresholded per image and macro-averaged. Empty-mask convention:

- Empty ground truth and empty prediction: score 1.
- Empty ground truth and any predicted foreground: score 0.

The evaluator reports raw threshold-0.5 results and validation-selected post-processing results separately.

## Generate a submission

Confirm the official sample-submission columns and RLE order first. Then set `sample_submission_path`, `submission_id_column`, `submission_mask_column`, `rle_order`, and `rle_order_confirmed=true` in the configuration.

Set `submission_id_suffix` only when the official sample submission proves a
constant suffix mapping from each test image stem. The Kaggle notebook infers
and validates this field rather than assuming `_segmentation`.

```bash
python -m skin_lesion_segmentation.cli submit \
  --config configs/default.json \
  --checkpoint artifacts/run/best_model.keras \
  --postprocessing artifacts/run/chosen_postprocessing.json
```

The output is `submission.csv`. The code validates column names, row count, ordering, duplicate IDs, and the exact ID set against the sample submission.

## Reproducibility artifacts

A real controlled run writes:

- Effective configuration and seed metadata.
- Package, operating-system, CPU, and TensorFlow device information.
- Pair and split manifests.
- Training history CSV.
- Selected `.keras` checkpoint and SHA-256 checksum.
- Saved validation probabilities and masks.
- Raw and selected validation metrics.
- Chosen threshold and post-processing parameters.
- A representative qualitative grid selected by a fixed score-quantile rule.
- Submission validation summary when submission generation is run.

No generated scores or placeholder result files are committed before a controlled rerun produces them.

## Evaluation limitations

- The historical validation result used an image-level split and a batch-global soft metric.
- The repaired pipeline has not yet been rerun on the private dataset.
- Patient or lesion independence requires trustworthy external metadata.
- Exact duplicate detection does not replace a domain-specific near-duplicate or acquisition audit.
- The official competition RLE order must be confirmed from the specification, evaluator, starter code, or a known asymmetric example.
- Threshold and post-processing selection must use validation predictions only, never public or private leaderboard feedback.
- No external clinical validation is provided.

## Method references

- Ozan Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas." https://arxiv.org/abs/1804.03999
- Nabila Abraham and Naimul Mefraz Khan. "A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation." https://arxiv.org/abs/1810.07842

## Non-clinical disclaimer

This learner pipeline is for education and portfolio review. It must not be used for diagnosis, treatment, triage, or patient care.
