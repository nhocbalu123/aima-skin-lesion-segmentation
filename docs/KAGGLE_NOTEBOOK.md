# Kaggle notebook guide

Use `notebooks/kaggle_controlled_rerun.ipynb` for verification, training, or a same-workflow submission. Use `notebooks/kaggle_submission_only.ipynb` when the checkpoint already exists as an attached Kaggle Dataset or notebook output.

## Inputs

Attach:

1. The repaired repository as either a ZIP or an extracted Kaggle Dataset.
2. The authorised AIMA skin-lesion dataset.
3. For submission-only mode, the controlled-run artifact ZIP or prior notebook output containing `best_model.keras`.

The historical dataset root was:

```text
/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation
```

## Workflow modes

In `kaggle_controlled_rerun.ipynb`, choose exactly one:

```python
WORKFLOW_MODE = "verify"  # tests, pairing, split audit, model smoke check
WORKFLOW_MODE = "train"   # verify plus controlled training and evaluation
WORKFLOW_MODE = "submit"  # restore checkpoint and generate final submission
```

The notebook starts in `verify` mode. Review the manifests before changing to `train`.

## Final submission contract

The successful historical encoder used C-order and fixed 512 x 512 masks. The maintained final parameters are:

```text
threshold: 0.5
minimum component size: 0
morphology kernel: 0
TTA transforms: 8
submission mask: 512 x 512
RLE order: C
```

The final parameters are stored in `results/controlled_rerun/final_submission_parameters.json`. Do not resize masks to original source-image dimensions.

Before uploading a CSV, confirm that `submission_validation_summary.json` reports:

```text
sample_submission_validated: true
rle_order: C
submission_mask_height: 512
submission_mask_width: 512
maximum_valid_rle_position: 262144
```
