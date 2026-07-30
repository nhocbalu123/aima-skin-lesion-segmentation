# Canonical Kaggle workflow

Use only `notebooks/kaggle_controlled_rerun.ipynb`. It is a thin orchestrator
around the tested package and starts with every mutation gate disabled.

## Attach inputs

Attach:

1. a ZIP or Kaggle Dataset created from one frozen repository commit;
2. the authorised AIMA skin-lesion dataset;
3. the immutable comparison manifest after its one-time creation;
4. for comparison, the two matched training outputs for one seed.

Kaggle inputs are read-only. The notebook copies source into a dedicated
`/kaggle/working` runtime directory before editable installation and
verification. Set `SOURCE_COMMIT` to the exact commit used to make the source
archive so provenance remains available without `.git`.

Do not attach credentials, unapproved datasets, or a substitute skin-lesion
dataset.

## Gates

The notebook separates:

- `RUN_VERIFY`: install, repository checks, tests, pairing, and model smoke;
- `RUN_PREPARE`: one-time immutable 70/15/15 manifest creation;
- `RUN_TRAINING`: exactly one architecture/seed run per Kaggle session;
- `RUN_COMPARISON`: one matched-seed internal-test comparison;
- `RUN_SUBMISSION`: optional competition submission, explicitly outside model
  selection.

Verification must pass before enabling another gate. Review the resolved
dataset path, pair count, manifest hash, source commit, environment, device,
and effective configuration in every session.

## Six training sessions

Run this matrix with the same source, dataset, manifest, accelerator class, and
configuration:

| Session | Model | Seed |
| --- | --- | ---: |
| 1 | `unet` | 42 |
| 2 | `attention_unet` | 42 |
| 3 | `unet` | 43 |
| 4 | `attention_unet` | 43 |
| 5 | `unet` | 44 |
| 6 | `attention_unet` | 44 |

Download each run output before the Kaggle session expires. Do not inspect the
internal test until all six training sessions finish.

## Three comparison sessions

Pair outputs only by identical seed. The comparison code rejects mismatched
source, manifest, settings, software, hardware, or checkpoint checksum.

Preserve the lightweight comparison directory described in
`docs/RESULT_ARTIFACTS.md`. Do not return checkpoints, prediction arrays,
private images, masks, generated submissions, or caches to Git.

Full commands and the interpretation rules are in
`docs/MODEL_COMPARISON.md`.
