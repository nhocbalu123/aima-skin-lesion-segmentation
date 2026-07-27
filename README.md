# Skin Lesion Segmentation with Attention U-Net

An end-to-end TensorFlow/Keras pipeline for binary skin-lesion segmentation. The project trains an Attention U-Net on paired dermoscopic images and masks, then produces run-length encoded predictions for a competition submission.

This project was completed through AIMA Research Lab's guided Research Warm-up Program. It represents guided student engineering work and the application of established segmentation methods—not independent research, a novel architecture, or a clinically validated system.

## Results

| Evaluation | Score |
|---|---:|
| Best training-time validation soft Dice | `0.858775` |
| Public leaderboard | `0.82545` |
| Private leaderboard | `0.80996` |

The validation result was recorded at epoch 92 using a random 80/20 image-level split. The notebook's validation metric is a soft, batch-global Dice score, so it is not directly comparable with the hidden-test leaderboard scores.

### Original run snapshot

| Item | Value |
|---|---:|
| Labelled image-mask pairs | 2,594 |
| Training images | 2,075 |
| Validation images | 519 |
| Test images | 1,100 |
| Input resolution | 512 × 512 |
| Training epochs | 100 |
| Model parameters | 31,404,901 |

## What I implemented

- Image and binary-mask preprocessing with OpenCV, CLAHE contrast enhancement, resizing, and normalization
- Joint image-mask augmentation with Albumentations
- Attention U-Net with gated skip connections, batch normalization, dropout, and L2 regularization
- A combined weighted Dice and focal Tversky training objective
- Adam optimization with learning-rate warm-up and cosine decay
- Mixed-precision training with `float32` loss and metric reductions
- Eight-pass test-time inference, probability averaging, and binary-mask post-processing
- Connected-component filtering and run-length encoding for submission generation

## Pipeline

1. Load paired training images and masks.
2. Split the data into training and validation sets.
3. Apply joint spatial and appearance augmentation to training batches.
4. Train the Attention U-Net and retain the checkpoint with the highest validation Dice.
5. Average predictions from eight test-time forward passes.
6. Threshold and post-process each predicted mask.
7. Encode masks as RLE strings and export the submission CSV.

## Model and training configuration

| Component | Configuration |
|---|---|
| Framework | TensorFlow / Keras |
| Architecture | Attention U-Net |
| Encoder filters | 64 → 128 → 256 → 512 |
| Bottleneck filters | 1,024 |
| Decoder filters | 512 → 256 → 128 → 64 |
| Output | Single-channel sigmoid mask |
| Batch size | 8 |
| Optimizer | Adam |
| Peak learning rate | `1e-4` |
| Learning-rate schedule | 5-epoch warm-up followed by cosine decay |
| Loss | `0.5 × weighted Dice + 0.5 × focal Tversky` |
| Regularization | Dropout and L2 kernel regularization |
| Checkpoint criterion | Highest validation soft Dice |

## Preprocessing and augmentation

Images are converted from BGR to RGB, resized to `512 × 512`, contrast-enhanced with CLAHE in LAB colour space, and normalized to `[0, 1]`. Masks are loaded in grayscale, resized, binarized, and expanded to a single channel.

The training pipeline applies joint image-mask transformations, including flips, rotations, affine transformations, geometric distortions, blur/noise, and colour or contrast adjustments.

## Inference and submission

For each test image, the notebook:

1. Generates eight predictions from the original image and transformed variants.
2. Inverts each transformation and averages the probability maps.
3. Applies a threshold of `0.5`.
4. Uses morphological closing, median filtering, and removal of connected components smaller than 150 pixels.
5. Converts the final binary mask to RLE.
6. Writes `submission_improved.csv` with `ID` and `Predicted_Mask` columns.

## Repository contents

```text
.
├── README.md
└── code-for-ai-vietnam-skin-lesion-segmentation.ipynb
```

The notebook contains the complete training and inference workflow.

## Running the notebook

The code was designed for a Kaggle GPU environment. Attach the competition dataset using this layout:

```text
/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation/
├── Train/Train/Image/
├── Train/Train/Mask/
└── Test/Test/Image/
```

Then run all three notebook cells. A completed run is designed to produce:

```text
best_attention_unet.h5
submission_improved.csv
```

Core dependencies include TensorFlow/Keras, NumPy, pandas, OpenCV, scikit-learn, Matplotlib, and Albumentations. Exact package versions from the original run were not retained, so a current environment may require small compatibility changes.

## Evaluation scope and limitations

- The 80/20 validation split is image-level, not patient- or lesion-grouped. It therefore does not establish patient-level generalization.
- The validation score uses the notebook's custom soft, batch-global Dice implementation rather than thresholded per-image Dice.
- The training notebook assumes independently sorted image and mask filenames remain aligned.
- The original training history, checkpoint, leaderboard screenshot, and qualitative predictions are not committed to this repository.
- Dependency versions and global random seeds are not fully pinned, so exact reruns may differ.
- Some augmentation arguments depend on the Albumentations version and may need updating with newer releases.
- The notebook is a learning and competition artifact, not a medical device or a basis for clinical decisions.

The highest-value future improvements are explicit ID-based image-mask pairing, a patient- or lesion-aware split where metadata permits, thresholded per-image evaluation, pinned dependencies, and committed run artifacts.

## Method references

- Ozan Oktay et al., [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- Nabila Abraham and Naimul Mefraz Khan, [A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation](https://arxiv.org/abs/1810.07842)
