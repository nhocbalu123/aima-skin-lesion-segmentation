# Kaggle notebook run guide

Use `notebooks/kaggle_controlled_rerun.ipynb` for the first controlled rerun.

## Kaggle inputs

Attach two private inputs to the notebook:

1. The repaired repository archive named
   `aima-skin-lesion-segmentation-repaired.zip`.
2. The authorised AIMA skin-lesion dataset.

The historical notebook used this dataset root:

```text
/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation
```

The complete notebook verifies the actual paths before training.

## First pass: preparation only

1. Enable a GPU in the Kaggle notebook settings.
2. Open the first code cell.
3. Keep `RUN_TRAINING = False`.
4. Keep `RUN_SUBMISSION = False`.
5. Run the notebook through the pairing and split-manifest section.
6. Check that image and mask counts match and both splits are non-empty.
7. Review whether reliable group metadata exists. Without it, the split is
   image-level and patient or lesion independence cannot be guaranteed.

## Training pass

After the preparation output is correct:

1. Change `RUN_TRAINING = True` in the first code cell.
2. Rerun the configuration cell.
3. Run the controlled-training cell.
4. Inspect `training_history.csv`, `final_validation_metrics.json`,
   `chosen_postprocessing.json`, and `qualitative_validation.png`.
5. Save a Kaggle notebook version so the `/kaggle/working` outputs persist.

## Submission pass

Do not generate a submission until the official competition contract confirms
RLE flattening order.

1. Confirm whether the order is `C` or `F` from an official source.
2. Set `RLE_ORDER` accordingly.
3. Set `RLE_ORDER_CONFIRMED = True`.
4. Set `RUN_SUBMISSION = True`.
5. Rerun the configuration and submission cells.
6. Check `submission_validation_summary.json` before uploading
   `submission.csv`.

The notebook reads the official sample submission, derives its column names,
and accepts an ID suffix only when the template proves one constant mapping
from test image stems. It does not blindly assume `_segmentation`.
