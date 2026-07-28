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

Training selects the checkpoint with minimum validation loss. Logged Keras loss may include L2 regularisation. The selected checkpoint is reloaded from `.keras` before offline validation or test inference.

## Split integrity

- Patient or lesion groups are used only when supplied through explicit metadata.
- Filenames are not interpreted as patient IDs.
- Exact duplicate decoded pixel content is SHA-256 hashed and assigned together.
- The audit rejects a group or decoded-pixel duplicate hash that crosses splits.
- Without reliable group metadata, patient or lesion independence cannot be guaranteed.

## Post-processing selection

Threshold, minimum component size, and morphology kernel are selected using saved validation probabilities only. Both raw threshold-0.5 metrics and selected metrics are written so a non-beneficial post-processing choice remains visible.

## RLE

RLE flattening order is mandatory configuration. C-order and F-order have separate asymmetric round-trip tests. The repository does not claim which order the competition uses until an official contract or known asymmetric example is available.
Test predictions are resized back to each source image's dimensions with nearest-neighbour interpolation before RLE encoding.
