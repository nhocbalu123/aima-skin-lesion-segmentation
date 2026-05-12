# AI Vietnam Skin Lesion Segmentation

TensorFlow/Keras pipeline for binary skin lesion segmentation. The notebook trains an Attention U-Net on paired lesion images and masks, then generates a Kaggle-style submission file with run-length encoded masks.

The implementation is built around practical competition improvements: attention gates, strong augmentation, mixed precision training, a Dice/Tversky-based loss, cosine learning-rate scheduling, post-processing, and test-time augmentation.

## Notebook

- `code-for-ai-vietnam-skin-lesion-segmentation(1).ipynb`

## Problem

Given a dermoscopic skin image, predict a binary segmentation mask identifying the lesion region. The final predictions are exported as RLE strings in a CSV file with these columns:

```text
ID, Predicted_Mask
```

## Dataset layout

The notebook expects the Kaggle dataset to be available at:

```text
/kaggle/input/warm-up-program-ai-vietnam-skin-segmentation/
├── Train/Train/Image/
├── Train/Train/Mask/
└── Test/Test/Image/
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and `.tif`.

## Key metrics from the completed run

| Metric | Value |
|---|---:|
| Labeled image-mask pairs found | 2,594 |
| Training samples | 2,075 |
| Validation samples | 519 |
| Train/validation split | 79.99% / 20.01% |
| Test images | 1,100 |
| Input image size | 512 × 512 × 3 |
| Mask size | 512 × 512 × 1 |
| Batch size | 8 |
| Epochs run | 100 |
| Steps per epoch | 259 |
| Effective training samples per epoch | 2,072 / 2,075 |
| Samples skipped per epoch due to floor-divided generator length | 3 |
| Total optimizer update steps | 25,900 |
| Model parameters | 31,404,901 |
| Best validation Dice | 0.858775 |
| Best validation IoU | 0.753756 |
| Best validation Dice epoch | 92 |
| Lowest validation loss | 0.1997 |
| Final training Dice | 0.8520 |
| Final training IoU | 0.7468 |
| Final training loss | 0.1836 |
| Final validation Dice | 0.8565 |
| Final validation IoU | 0.7504 |
| Final validation loss | 0.1997 |
| Data loading time | 429.66 seconds, about 7 min 10 sec |
| Training time from epoch logs | 24,218 seconds, about 6 hr 43 min 38 sec |
| Average epoch time | 242.18 seconds |
| Median epoch time | 239 seconds |
| TTA forward passes per test image | 8 |
| Total test-time forward passes | 8,800 |
| Submission rows generated | 1,100 |

### Validation progress

| Checkpoint | Validation Dice | Validation IoU | Validation Loss |
|---|---:|---:|---:|
| Epoch 1 | 0.1353 | 0.0726 | 0.9362 |
| Best Dice epoch, epoch 92 | 0.8588 | 0.7538 | 0.2002 |
| Final epoch, epoch 100 | 0.8565 | 0.7504 | 0.1997 |

From epoch 1 to the best checkpoint, validation Dice improved by **+0.7235** and validation IoU improved by **+0.6812**.

## Model architecture

The model is an Attention U-Net with encoder-decoder skip connections and attention gates on decoder skip paths.

Architecture summary:

- Input: `512 × 512 × 3`
- Encoder filters: `64 → 128 → 256 → 512`
- Bottleneck filters: `1024`
- Decoder filters: `512 → 256 → 128 → 64`
- Skip connections: gated with attention blocks
- Output: single-channel sigmoid mask
- Regularization: dropout and L2 weight decay
- Parameter count: `31,404,901`

## Training configuration

| Setting | Value |
|---|---:|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Scheduler | Cosine decay with 5-epoch warmup |
| Warmup learning rates | `2e-5`, `4e-5`, `6e-5`, `8e-5`, `1e-4` |
| Weight decay | `1e-5` |
| Mixed precision | Enabled |
| Loss precision fix | Cast loss/metric tensors to `float32` |
| Checkpoint monitor | `val_dice_coefficient` |
| Checkpoint mode | Maximize |
| Early stopping patience | 20 epochs |
| Saved best model | `best_attention_unet.h5` |

The combined loss is:

```text
0.5 × weighted_dice_loss + 0.5 × focal_tversky_loss
```

Loss details:

- Weighted Dice foreground weight: `2.0`
- Focal Tversky parameters: `alpha=0.7`, `beta=0.3`, `gamma=0.75`
- Numerical stability smoothing: `1e-6`

## Data preprocessing

Images are loaded with OpenCV, converted from BGR to RGB, resized to `512 × 512`, contrast-enhanced with CLAHE in LAB color space, and normalized to `[0, 1]`.

Masks are loaded as grayscale, resized to `512 × 512`, thresholded at `127`, and expanded to shape `512 × 512 × 1`.

## Augmentation

Training augmentation uses Albumentations and includes:

- Horizontal flip
- Vertical flip
- Random 90-degree rotation
- Shift-scale-rotate
- Elastic, grid, and optical distortions
- Gaussian noise, Gaussian blur, and median blur
- Random brightness/contrast
- Random gamma
- CLAHE
- Hue/saturation/value shift

## Inference pipeline

For each test image, the notebook:

1. Loads and preprocesses the image.
2. Runs prediction with test-time augmentation.
3. Averages predictions across augmentations.
4. Applies a `0.5` threshold.
5. Post-processes the binary mask.
6. Converts the mask to RLE.
7. Writes the result to `submission_improved.csv`.

### Test-time augmentation

The TTA function averages 8 predictions:

1. Original image
2. Horizontal flip
3. Vertical flip
4. Horizontal + vertical flip
5. 90-degree rotation
6. 180-degree rotation
7. 270-degree rotation
8. Transpose

For 1,100 test images, this produces **8,800 model forward passes**.

## Post-processing

Predicted masks are refined with:

- Morphological closing using an elliptical kernel of size `5 × 5`
- Median blur with kernel size `5`
- Connected-component filtering
- Minimum retained component area: `150` pixels

## Outputs

Running the notebook produces:

```text
best_attention_unet.h5
submission_improved.csv
```

The completed run generated:

- `1,100` predictions
- output file: `submission_improved.csv`

## How to run

1. Open the notebook in Kaggle or another GPU-enabled environment.
2. Attach the dataset at the expected Kaggle input path.
3. Run all cells.
4. Download `submission_improved.csv`.
5. Submit the CSV to the competition platform.

## Dependencies

The notebook imports:

```text
numpy
pandas
opencv-python
scikit-learn
tensorflow
keras
matplotlib
albumentations
```

A minimal local installation command is:

```bash
pip install numpy pandas opencv-python scikit-learn tensorflow matplotlib albumentations
```

For Kaggle, most dependencies are usually preinstalled. GPU acceleration is strongly recommended because the logged run used a Tesla P100 GPU and took about 6 hours 44 minutes for training.

## Reproducibility notes

- The train/validation split uses `random_state=42`.
- The data generator shuffles indices each epoch without setting a global NumPy/TensorFlow seed, so exact training results may vary.
- Mixed precision is enabled to reduce memory usage, while losses and metrics cast tensors to `float32` to reduce NaN risk.
- The generator length is `len(images) // batch_size`, so 3 of 2,075 training samples are skipped in each epoch with the current batch size.

## Known warnings and improvement opportunities

The logged run completed successfully, but it emitted a few Albumentations warnings:

- `ShiftScaleRotate` is treated as a special case of `Affine`.
- `alpha_affine` is not valid for the installed `ElasticTransform` version.
- `shift_limit` is not valid for the installed `OpticalDistortion` version.
- `var_limit` is not valid for the installed `GaussNoise` version.

Suggested improvements:

- Update augmentation arguments to match the installed Albumentations API.
- Change `DataGenerator.__len__` from floor division to ceiling division if every training sample should be used each epoch.
- Set global seeds for NumPy and TensorFlow for stronger reproducibility.
- Save the training history to CSV or JSON for easier experiment tracking.
- Consider exporting a `requirements.txt` with pinned package versions.
