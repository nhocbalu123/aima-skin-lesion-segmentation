# Scientific U-Net Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a fair, sealed U-Net versus Attention U-Net experiment and export a clean source ZIP.

**Architecture:** Extend the existing package rather than adding a second
pipeline. Separate manifest construction, training evidence, paired analysis,
robustness transforms, qualitative output, and result-schema validation into
focused modules with one CLI surface.

**Tech Stack:** Python 3.11, TensorFlow 2.16.1, Keras 3.0.5, NumPy 1.26.4,
OpenCV 4.10.0.84, pandas 2.2.2, Matplotlib 3.9.2, pytest 8.3.3.

## Global Constraints

- Preserve the historical controlled-rerun evidence and meaning.
- Never invent experiment results or claim real-data execution.
- Keep the internal test inaccessible to checkpoint selection.
- Keep CI CPU-compatible and private-data-free.
- Use tests before production behavior changes.
- Do not push, merge, or open a PR.

---

### Task 1: Strict dependency baseline and model factory

**Files:**
- Modify: `requirements.txt`
- Modify: `src/skin_lesion_segmentation/model.py`
- Create: `tests/test_model_factory.py`
- Modify: `src/skin_lesion_segmentation/cli.py`

**Interfaces:**
- Produces: `build_model(model_name, input_shape, *, base_filters, l2_coefficient, mixed_precision)`
- Produces: CLI `smoke --model {unet,attention_unet}`

- [ ] Write tests requiring both model names, matched output shapes, float32
  output, direct U-Net skips, attention-gated skips, and invalid-name rejection.
- [ ] Run the focused tests and confirm they fail because the factory is absent.
- [ ] Refactor the existing builder into one shared body and add the smoke CLI.
- [ ] Run focused tests and both CPU model smokes.
- [ ] Pin `pyparsing==3.1.4` and prove strict pytest collection succeeds.

### Task 2: Three-way immutable manifests

**Files:**
- Modify: `src/skin_lesion_segmentation/splitting.py`
- Modify: `src/skin_lesion_segmentation/evaluation.py`
- Modify: `src/skin_lesion_segmentation/training.py`
- Create: `tests/test_comparison_manifest.py`

**Interfaces:**
- Produces: `build_comparison_manifest(..., validation_fraction, internal_test_fraction, split_seed)`
- Produces: `manifest_sha256(rows)`
- Produces: `pairs_for_splits(pairs, rows)` returning train and validation only

- [ ] Add failing tests for deterministic 70/15/15 assignment, nonempty splits,
  group/duplicate isolation, mask hashes, stable manifest identity, and absence
  of internal-test samples from training sequences.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement component-level deterministic assignment and mask-aware hashing.
- [ ] Add external-manifest loading to training without changing the legacy
  two-way default.
- [ ] Run focused and existing split/training tests.

### Task 3: Run evidence schema and provenance

**Files:**
- Create: `src/skin_lesion_segmentation/schemas.py`
- Modify: `src/skin_lesion_segmentation/reproducibility.py`
- Modify: `src/skin_lesion_segmentation/training.py`
- Create: `tests/test_results_schema.py`
- Create: `tests/test_training_provenance.py`

**Interfaces:**
- Produces: `validate_run_summary(value)`
- Produces: `collect_git_provenance(root)`
- Produces: `command_record(argv)`

- [ ] Write failing schema and provenance tests for all required fields.
- [ ] Implement validation, Git commit/dirty capture, exact command capture,
  hardware details, runtime, model/parameter evidence, and manifest identity.
- [ ] Ensure summaries contain validation evidence but no internal-test metrics.
- [ ] Run focused tests.

### Task 4: Paired statistics, slices, and robustness

**Files:**
- Create: `src/skin_lesion_segmentation/analysis.py`
- Create: `src/skin_lesion_segmentation/robustness.py`
- Create: `tests/test_analysis.py`
- Create: `tests/test_robustness.py`

**Interfaces:**
- Produces: `per_image_comparison(...)`
- Produces: `paired_bootstrap_interval(left, right, *, seed, n_resamples)`
- Produces: `summarize_slices(rows)`
- Produces: `apply_perturbation(images, condition)`

- [ ] Write failing tests for exact per-image values, deterministic paired
  intervals, empty-input rejection, fixed size slices, border slices, and
  deterministic bounded perturbations.
- [ ] Implement the TensorFlow-free functions with fixed predeclared settings.
- [ ] Run focused tests and verify no warning output.

### Task 5: Sealed comparison and qualitative evidence

**Files:**
- Create: `src/skin_lesion_segmentation/comparison.py`
- Modify: `src/skin_lesion_segmentation/qualitative.py`
- Modify: `src/skin_lesion_segmentation/cli.py`
- Create: `tests/test_comparison.py`
- Create: `tests/test_comparison_end_to_end.py`

**Interfaces:**
- Produces: `compare_runs(config, manifest_path, unet_run, attention_run, output_dir)`
- Produces: CLI `compare`

- [ ] Write failing tests for mismatched model, seed, manifest, config, commit,
  software, hardware, checkpoint hash, and evaluation config.
- [ ] Write a synthetic end-to-end test using tiny saved models and a three-way
  manifest.
- [ ] Implement clean and perturbation inference, CSV/JSON artifacts, copied run
  summaries, and deterministic six-panel figures.
- [ ] Run focused comparison tests.

### Task 6: Configuration, canonical notebook, and cleanup

**Files:**
- Create: `configs/controlled_rerun.json`
- Create: `configs/model_comparison.json`
- Replace: `notebooks/kaggle_controlled_rerun.ipynb`
- Delete: duplicate maintained notebooks confirmed by audit
- Modify: `notebooks/README.md`
- Modify: `scripts/verify_repository.py`
- Create: `tests/test_notebooks.py`

**Interfaces:**
- Produces: one thin Kaggle workflow that copies attached source into
  `/kaggle/working` before editable installation.

- [ ] Add failing tests that parse every notebook and enforce the canonical
  workflow markers and no stored outputs.
- [ ] Add the two configurations and canonical notebook.
- [ ] Remove only confirmed duplicate/obsolete maintained notebooks while
  retaining `original_pipeline.ipynb`.
- [ ] Strengthen static verification and run notebook tests.

### Task 7: Documentation and CI

**Files:**
- Modify: `README.md`
- Modify: `docs/CONTROLLED_RERUN.md`
- Create: `docs/MODEL_COMPARISON.md`
- Create: `docs/RESULT_ARTIFACTS.md`
- Modify: `docs/KAGGLE_NOTEBOOK.md`
- Modify: `.github/workflows/tests.yml`
- Modify: `scripts/synthetic_smoke.py`
- Modify: `tests/test_repository_docs.py`

**Interfaces:**
- Documents exact prepare, six-train, and three-compare commands.

- [ ] Add failing documentation tests for required paths, commands, evidence
  separation, and unexecuted-study language.
- [ ] Update docs without inserting result placeholders.
- [ ] Make CI run strict pytest and dual-model CPU smoke.
- [ ] Run documentation tests and static verification.

### Task 8: Full verification and ZIP handoff

**Files:**
- Create outside Git tracking: `aima-skin-lesion-scientific-comparison.zip`

- [ ] Run `.venv/bin/python scripts/verify_repository.py`.
- [ ] Run `.venv/bin/python -m pytest -q -W error`.
- [ ] Run both CLI model smokes and `scripts/synthetic_smoke.py`.
- [ ] Parse every tracked notebook independently as JSON.
- [ ] Inspect `git diff --check`, status, tracked large files, cache/checkpoint
  patterns, and high-confidence credential patterns.
- [ ] Commit the verified source state locally.
- [ ] Build the ZIP from that exact commit with `git archive`.
- [ ] Verify archive integrity, top-level layout, commit marker, and SHA-256.
- [ ] Save the ZIP and provide the user a download link plus concise extraction
  and upload instructions.
