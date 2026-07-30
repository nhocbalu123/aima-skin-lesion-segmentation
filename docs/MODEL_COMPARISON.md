# U-Net versus Attention U-Net comparison

## Evidence status

The code and synthetic tests implement this protocol.
No real-data comparison result is reported. Do not insert scores, confidence
intervals, or qualitative claims until every required run has completed on the
authorised dataset.

The earlier 80/20 controlled rerun remains separate historical evidence. Its
validation set was used for checkpoint selection and must not be relabelled as
the new internal test.

## Frozen design

The primary comparison uses:

- one deterministic 70/15/15 train/validation/internal-test manifest;
- training seeds 42, 43, and 44 for each architecture;
- image size 256 x 256 and base filter count 32;
- batch size 8 and an epoch budget of 50;
- identical preprocessing, augmentation, optimizer, loss, learning-rate
  policy, early stopping, and filter schedule;
- minimum validation loss for checkpoint selection;
- early-stopping patience 8;
- threshold 0.5, one inference pass, no TTA, and no post-processing;
- an internal test that remains sealed until all six training runs finish.

The intended architecture-level difference is the presence or absence of
attention gates. With the production configuration, plain U-Net has 7,768,929
parameters and Attention U-Net has 7,856,693, an overhead of 87,764 parameters
(1.13%).

Use genuine patient or lesion identifiers as group IDs only when authorised,
reliable metadata supplies them. Never derive patient identity from filenames.
Without those identifiers, label the study image-level. Exact decoded-image
duplicates are grouped within one split; image and mask hashes are recorded in
the manifest.

## 1. Verify the source

From one frozen source commit:

```bash
python scripts/verify_repository.py
pytest -q -W error
python scripts/synthetic_smoke.py
git rev-parse HEAD
```

Record the returned commit. All six runs and all three comparisons must use
that exact source.

## 2. Create the immutable manifest

Set the authorised dataset paths in `configs/model_comparison.json`. If real
group metadata exists, set `group_mapping_path` before continuing.

```bash
python -m skin_lesion_segmentation.cli prepare-comparison \
  --config configs/model_comparison.json \
  --output-manifest artifacts/model_comparison/comparison_manifest.json
```

Confirm that all expected image-mask pairs occur once, split IDs do not
overlap, duplicate groups do not cross splits, and a stable manifest SHA-256 is
reported. Preserve this file and use it unchanged in every run.

## 3. Train exactly six runs

```bash
python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model unet --seed 42 \
  --output-dir artifacts/model_comparison/unet-42

python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model attention_unet --seed 42 \
  --output-dir artifacts/model_comparison/attention-unet-42

python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model unet --seed 43 \
  --output-dir artifacts/model_comparison/unet-43

python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model attention_unet --seed 43 \
  --output-dir artifacts/model_comparison/attention-unet-43

python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model unet --seed 44 \
  --output-dir artifacts/model_comparison/unet-44

python -m skin_lesion_segmentation.cli train \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --model attention_unet --seed 44 \
  --output-dir artifacts/model_comparison/attention-unet-44
```

Do not inspect the internal test between runs. Do not select the best seed,
replace an unfavourable run, or change settings for one architecture. If a
shared setting such as batch size must change, update it before test unsealing
and rerun the entire matrix.

Each `run_summary.json` must record the model, seed, manifest hash, source
commit, effective configuration, parameter count, environment and visible
hardware, exact command, runtime, best epoch, validation evidence, and
checkpoint checksum. It must not contain internal-test metrics.

## 4. Unseal three matched-seed comparisons

Run these only after all six training runs are complete:

```bash
python -m skin_lesion_segmentation.cli compare \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --unet-run-summary artifacts/model_comparison/unet-42/run_summary.json \
  --attention-run-summary artifacts/model_comparison/attention-unet-42/run_summary.json \
  --unet-checkpoint artifacts/model_comparison/unet-42/best_model.keras \
  --attention-checkpoint artifacts/model_comparison/attention-unet-42/best_model.keras \
  --output-dir results/model_comparison/seed-42 \
  --bootstrap-seed 20260772 --bootstrap-resamples 10000

python -m skin_lesion_segmentation.cli compare \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --unet-run-summary artifacts/model_comparison/unet-43/run_summary.json \
  --attention-run-summary artifacts/model_comparison/attention-unet-43/run_summary.json \
  --unet-checkpoint artifacts/model_comparison/unet-43/best_model.keras \
  --attention-checkpoint artifacts/model_comparison/attention-unet-43/best_model.keras \
  --output-dir results/model_comparison/seed-43 \
  --bootstrap-seed 20260773 --bootstrap-resamples 10000

python -m skin_lesion_segmentation.cli compare \
  --config configs/model_comparison.json \
  --split-manifest artifacts/model_comparison/comparison_manifest.json \
  --unet-run-summary artifacts/model_comparison/unet-44/run_summary.json \
  --attention-run-summary artifacts/model_comparison/attention-unet-44/run_summary.json \
  --unet-checkpoint artifacts/model_comparison/unet-44/best_model.keras \
  --attention-checkpoint artifacts/model_comparison/attention-unet-44/best_model.keras \
  --output-dir results/model_comparison/seed-44 \
  --bootstrap-seed 20260774 --bootstrap-resamples 10000
```

The comparison rejects mismatched manifests, training seeds, effective
settings, source commits, software, hardware, or checkpoint checksums before
loading the internal test.

For each seed, report macro Dice and IoU for both models, the paired mean
difference, and its paired bootstrap 95% confidence interval. Report mean and
standard deviation across the three seeds separately; do not average confidence
interval endpoints.

## 5. Read the failure and robustness evidence

Each comparison produces:

- per-image Dice and IoU for both models and paired differences;
- fixed empty, small, medium, and large lesion slices;
- border-touching and non-border-touching slices;
- deterministic worst, false-positive, and false-negative examples;
- image, ground truth, prediction, and error-overlay figures;
- clean, brightness, contrast, blur, and JPEG summaries.

The perturbations are fixed synthetic robustness checks, not clinical
validation. Do not redefine slices, tune thresholds, or select favourable
conditions after seeing the internal-test results.

## 6. Interpret the result

- If attention improves consistently and paired intervals exclude zero without
  material slice harm, the data support the added gates.
- If effects are small or intervals include zero, prefer the simpler U-Net.
- If seed effects change sign, call the architecture comparison inconclusive.
- If attention is worse, prefer U-Net.

A null or negative result is acceptable. The purpose is to make a defensible
model-selection decision, not to make Attention U-Net win.

The workload is six training runs, three comparisons, and 30 model-condition
inference passes across five declared conditions. The repository has no
recorded reference hardware/runtime from which to make a defensible time or
cost estimate.
