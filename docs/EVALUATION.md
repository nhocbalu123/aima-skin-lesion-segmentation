# Evaluation contract

## Final metrics

Final validation metrics are thresholded per image and macro-averaged:

```text
Dice_i = 2 * TP_i / (2 * TP_i + FP_i + FN_i)
IoU_i  = TP_i / (TP_i + FP_i + FN_i)
Macro metric = mean over validation images
```

For an empty ground-truth mask:

- Empty prediction: score 1.
- Non-empty prediction: score 0.

`soft_dice_batch_global_tf` is a training diagnostic only. It reduces all pixels in a batch together and must not be labelled as ordinary validation Dice.

## Checkpoint selection

Training selects the checkpoint with minimum validation loss. Logged Keras loss may include L2 regularisation. The selected `.keras` checkpoint is reloaded before offline validation or test inference.

The controlled rerun completed 31 epochs and selected epoch 23 with validation loss 0.260925.


## Split integrity

- Patient or lesion groups are used only when supplied through explicit metadata.
- Filenames are not interpreted as patient IDs.
- Exact duplicate decoded pixel content is SHA-256 hashed and assigned together.
- The audit rejects a group or decoded-pixel duplicate hash that crosses splits.
- Without reliable group metadata, patient or lesion independence cannot be guaranteed.

The controlled rerun used 2,075 training images and 519 validation images. No reliable group metadata was available and no exact duplicate image hashes were detected.

## Controlled validation result

| Configuration | Macro Dice | Macro IoU |
| --- | ---: | ---: |
| Raw threshold 0.5 | 0.805922 | 0.713368 |
| Validation-selected post-processing | 0.808482 | 0.714909 |

The selected post-processing parameters were threshold 0.35, minimum component size 64, and morphology kernel 3. The small validation gain did not produce a meaningful leaderboard benefit, so raw threshold 0.5 is the maintained final default.

## Submission contract

The historical competition encoder used `mask.flatten()`, which is C-order. The historical pipeline and successful corrected submissions used fixed 512 x 512 masks.

The final submission path therefore:

1. Predicts in model space.
2. Applies the configured threshold and post-processing.
3. Resizes the binary mask to the explicit configured submission shape with nearest-neighbour interpolation.
4. Encodes the 512 x 512 mask using C-order RLE.
5. Validates IDs, ordering, row count, duplicates, and RLE bounds against the official sample submission.

Resizing predictions to source-image dimensions is incorrect for this competition and is protected by tests.

