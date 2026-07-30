# Scientific U-Net Comparison Design

## Goal

Add a fair, reproducible plain U-Net versus Attention U-Net experiment to the
existing learner skin-lesion segmentation project without fabricating results
or replacing the preserved historical controlled rerun.

## Selected approach

Use one tested package and CLI for both architectures. A shared model factory
builds the same encoder, decoder, filter schedule, loss, optimizer, data
pipeline, checkpoint rule, and training policy; attention gates are the only
intended architectural difference.

This is preferred over:

1. Two independent training scripts, which would make fair configuration drift
   difficult to detect.
2. A notebook-only comparison, which would be hard to test and easy to edit
   after seeing results.

The maintained notebook is therefore a thin Kaggle orchestrator around package
commands.

## Split and training protocol

- Preserve the historical 80/20 controlled rerun unchanged.
- Introduce a separate deterministic 70/15/15 comparison manifest.
- Use explicit patient or lesion groups only when real metadata is supplied.
- Otherwise label the comparison image-level and never infer groups from names.
- Keep exact decoded-image duplicates in one split.
- Include decoded image and mask hashes in the manifest identity.
- Use split seed 20260730 and training seeds 42, 43, and 44.
- Train on `train`; select checkpoints only by minimum `validation` loss.
- Do not construct an internal-test training sequence or evaluate the internal
  test during training.
- Use threshold 0.5, no TTA, and no post-processing for the primary comparison.
- Treat a one-seed run as exploratory and three seeds as the intended study.

## Evidence and provenance

Each run summary records model name, seed, manifest hash, effective
configuration, parameter count, checkpoint checksum, checkpoint-selection
rule, validation metrics, runtime, command, source commit and dirty state,
software versions, and visible hardware.

The comparison rejects model pairs when manifest, seed, core configuration,
source commit, software, hardware, or checkpoint evidence is incompatible. It
copies the two lightweight run summaries into the comparison output but never
copies checkpoints.

## Internal-test analysis

For each matched seed, the comparison produces:

- per-image Dice and IoU for both models and paired differences;
- paired bootstrap 95% confidence intervals;
- fixed lesion-size slices based on foreground proportion;
- border-touching versus non-border-touching slices;
- deterministic worst, false-positive, and false-negative case selections;
- six-panel image, truth, predictions, and error-overlay figures;
- predeclared clean, brightness, contrast, blur, and JPEG conditions.

Perturbations are synthetic robustness checks, not clinical validation. Test
results cannot change slice boundaries, perturbation strengths, thresholds,
TTA, or post-processing.

## Repository boundaries

- Keep reusable logic in focused package modules.
- Keep the preserved original notebook.
- Replace duplicate maintained notebooks with one canonical Kaggle workflow.
- Add the missing controlled-rerun configuration.
- Commit only lightweight code, configuration, summaries, CSV files, and plots.
- Never commit datasets, checkpoints, saved probability tensors, credentials,
  submissions, or generated caches.

## Testing and completion criteria

CPU-compatible tests cover model construction, deterministic manifests,
duplicate isolation, mask-aware manifest identity, evidence schemas, per-image
analysis, slices, bootstrap intervals, perturbations, comparison provenance,
notebook JSON, and CLI smoke behavior.

Completion requires strict full tests, repository verification, JSON validation
of every tracked notebook, CPU smoke execution for both models, documentation
path checks, and archive-content inspection. Real comparative metrics remain
explicitly unavailable until the six authorised-data training runs are
executed.
