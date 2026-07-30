# AIMA Skin Lesion Segmentation

A tested TensorFlow/Keras pipeline for binary skin-lesion segmentation with
plain U-Net and Attention U-Net. This work was completed through AIMA Research
Lab's guided Research Warm-up Program and is presented as a learner
healthcare-AI portfolio project.

This is not a publication, internship, independent research study, original
architecture, medical device, or clinically validated system.

## Status

The repository preserves the original notebook and an earlier controlled
Attention U-Net rerun. It now also contains the infrastructure for a fair,
sealed U-Net versus Attention U-Net experiment.

The comparison infrastructure has been exercised only with synthetic data.
No real U-Net versus Attention U-Net comparison has been executed. There are
therefore no new comparative Dice, IoU, confidence-interval, failure-analysis,
or robustness results.

## Controlled rerun results

The earlier maintained submission used the reloaded Attention U-Net checkpoint,
eight-transform TTA, raw threshold 0.5 predictions, fixed 512 x 512 submission
masks, and C-order RLE.

| Evaluation | Score |
| --- | ---: |
| Offline validation macro Dice | 0.805922 |
| Offline validation macro IoU | 0.713368 |
| Public leaderboard | 0.80225 |
| Private leaderboard | 0.76879 |

The offline scores are reproduced from committed evaluator artifacts. The
leaderboard values are author-reported because no leaderboard export or
screenshot is committed.

Validation-only tuning selected threshold 0.35, minimum component size 64, and
a 3 x 3 morphology kernel. That configuration reached validation macro Dice
0.808482, public 0.80166, and private 0.76888. The leaderboard difference was
negligible, so the simpler raw threshold-0.5 path remained the default.

The reported validation set was also used for checkpoint selection. It is
historical evidence, not an untouched internal-test result.

## Author-reported historical results

The repository does not contain the original training logs, checkpoint, or
leaderboard export required to independently verify these values.

| Evaluation | Score |
| --- | ---: |
| Best original training-time validation soft Dice | 0.858775 |
| Public leaderboard | 0.82545 |
| Private leaderboard | 0.80996 |

The original validation value used a random 80/20 image-level split and a
custom soft, batch-global Dice metric. It is not directly comparable with the
controlled rerun's thresholded per-image metrics or hidden-test scores.

An earlier public 0.12522 / private 0.12470 submission is excluded from model
comparison because masks were encoded at source-image dimensions instead of
the competition's 512 x 512 geometry. This was a submission-contract failure,
not a model result.

## Scientific model-comparison status

The new protocol predeclares:

- one immutable 70/15/15 train/validation/internal-test manifest;
- training seeds 42, 43, and 44 for both architectures;
- shared preprocessing, augmentation, filter schedule, optimizer, loss,
  learning-rate policy, epoch budget, early stopping, and checkpoint rule;
- minimum validation loss as the checkpoint-selection criterion;
- threshold 0.5, no TTA, and no post-processing for the primary comparison;
- paired per-image Dice and IoU with paired-bootstrap intervals;
- fixed lesion-size and border-touching slices;
- deterministic brightness, contrast, blur, and JPEG checks labelled as
  synthetic robustness tests;
- worst-case, false-positive, false-negative, and error-overlay outputs.

The primary architectural difference is the presence of attention gates.

| Model | Parameters |
| --- | ---: |
| U-Net | 7,768,929 |
| Attention U-Net | 7,856,693 |
| Attention overhead | 87,764 (1.13%) |

These counts are construction facts, not performance evidence.
Attention's incremental value remains unknown.

When reliable patient or lesion identifiers are unavailable, the split is
explicitly image-level. Exact decoded-image duplicate groups remain within one
split, but that does not prove patient independence or eliminate near-duplicate
leakage.

See [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) for the exact six-run
protocol and [docs/RESULT_ARTIFACTS.md](docs/RESULT_ARTIFACTS.md) for the
evidence contract. No apples-to-apples result table will be added until both
architectures have actually been rerun.

## What the maintained pipeline enforces

- Image-mask pairing by validated canonical sample IDs.
- Case-insensitive support for common image extensions.
- Path-level streaming, including final partial batches.
- Nearest-neighbour mask resizing and explicit binarisation.
- Explicit group metadata only; filenames are not treated as patient IDs.
- Decoded image and mask hashes in comparison manifests.
- Exact duplicate isolation across train, validation, and internal test.
- Shared model construction through `--model unet` and
  `--model attention_unet`.
- Separate training-time soft Dice and final thresholded per-image metrics.
- Checkpoint selection by validation loss and checkpoint reload before
  evaluation.
- Run summaries with model configuration, seed, manifest hash, parameter count,
  environment, hardware, command, runtime, checkpoint checksum, and source
  commit.
- Provenance rejection before internal-test inference when matched runs differ.
- Explicit TTA, RLE, submission-column, ID, and output-geometry contracts.

## Repository structure

```text
configs/
  controlled_rerun.json                 Earlier Attention U-Net rerun
  model_comparison.json                 New sealed comparison protocol
notebooks/
  original_pipeline.ipynb               Preserved historical notebook
  skin_lesion_segmentation.ipynb        Thin local orchestrator
  kaggle_controlled_rerun.ipynb         Canonical gated Kaggle workflow
results/controlled_rerun/               Compact earlier-rerun evidence
src/skin_lesion_segmentation/           Maintained package
scripts/                                Verification and smoke utilities
tests/                                  Synthetic CPU-compatible tests
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

Obtain the dataset through its authorised competition or program channel. Do
not commit or redistribute restricted images, masks, metadata, sample
submissions, generated submissions, or checkpoints.

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

Fixed 256 x 256 model resizing can distort aspect ratio. The historical
competition submission contract used fixed 512 x 512 output masks.

## Verify locally

```bash
python scripts/verify_repository.py
pytest -q -W error
python scripts/synthetic_smoke.py
python -m skin_lesion_segmentation.cli smoke --model unet
python -m skin_lesion_segmentation.cli smoke --model attention_unet
```

These commands do not require the private dataset or full training.

## Earlier controlled-rerun path

The earlier 80/20 Attention U-Net evidence is preserved without changing its
meaning:

```bash
python -m skin_lesion_segmentation.cli prepare \
  --config configs/controlled_rerun.json

python -m skin_lesion_segmentation.cli train \
  --config configs/controlled_rerun.json
```

See [docs/CONTROLLED_RERUN.md](docs/CONTROLLED_RERUN.md) for the historical
evidence and limitations.

## New comparison path

Prepare the manifest once:

```bash
python -m skin_lesion_segmentation.cli prepare-comparison \
  --config configs/model_comparison.json \
  --output-manifest artifacts/model_comparison/comparison_manifest.json
```

Then execute the six predeclared training runs and three matched-seed
comparisons exactly as documented in
[docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md).

The internal test must not be used for checkpoint selection, tuning, model
selection, seed selection, slice definition, or perturbation selection.

## Submission path

Submission generation is separate from model selection:

```bash
python -m skin_lesion_segmentation.cli submit \
  --config configs/controlled_rerun.json \
  --checkpoint artifacts/controlled_rerun/best_model.keras \
  --postprocessing results/controlled_rerun/final_submission_parameters.json
```

The implementation validates columns, IDs, row order, duplicates, RLE bounds,
and output geometry against the official template.

## Reproducibility evidence

`results/controlled_rerun/` contains the effective configuration, environment,
seed, training history, checkpoint checksum, final submission parameters,
compact metric summaries, and checksums for excluded external artifacts.

Large or restricted artifacts are intentionally excluded from Git:

- model checkpoints;
- saved prediction tensors;
- private images, masks, and metadata;
- manifests containing restricted paths or identifiers;
- generated submission CSV files;
- qualitative figures containing restricted imagery.

## Limitations

- No real-data U-Net versus Attention U-Net comparison has been completed.
- The new protocol remains image-level unless reliable patient or lesion IDs
  are supplied.
- Exact hashing does not detect all near-duplicates or repeated acquisitions.
- Three seeds offer limited evidence about training variability.
- Accelerator-level determinism can differ across hardware and libraries.
- Fixed-size resizing can distort anatomy.
- Synthetic perturbations are not clinical robustness evidence.
- There is no external-site, prospective, patient-disjoint, calibration, or
  clinical evaluation.
- Historical leaderboard values remain author-reported evidence.

## Method references

- Ozan Oktay et al. "Attention U-Net: Learning Where to Look for the
  Pancreas." https://arxiv.org/abs/1804.03999
- Nabila Abraham and Naimul Mefraz Khan. "A Novel Focal Tversky Loss Function
  with Improved Attention U-Net for Lesion Segmentation."
  https://arxiv.org/abs/1810.07842

## Non-clinical disclaimer

This learner pipeline is for education and portfolio review. It must not be
used for diagnosis, treatment, triage, or patient care.
