# Controlled rerun

For Kaggle, upload the repaired repository ZIP as a private Kaggle Dataset and
open `notebooks/kaggle_controlled_rerun.ipynb`. Attach both the repository
dataset and the authorised AIMA skin-lesion dataset. The notebook contains
explicit gates for training and submission and is the recommended execution
path for the first repaired-pipeline rerun.

1. Obtain the authorised dataset and official sample submission.
2. Confirm image and mask directories, mask suffix, sample-submission columns, and RLE order.
3. Add reliable patient or lesion group metadata when available. Otherwise retain the image-level limitation.
4. Copy `configs/default.json` to a run-specific file and edit paths only.
5. Preserve the run-specific configuration with the output artifacts.

Exact commands:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m skin_lesion_segmentation.cli prepare --config configs/controlled_rerun.json
pytest -q
python scripts/verify_repository.py
python -m skin_lesion_segmentation.cli train --config configs/controlled_rerun.json
python -m skin_lesion_segmentation.cli submit \
  --config configs/controlled_rerun.json \
  --checkpoint artifacts/controlled_rerun/best_model.keras \
  --postprocessing artifacts/controlled_rerun/chosen_postprocessing.json
```

Recommended run-specific changes:

```json
{
  "output_dir": "artifacts/controlled_rerun",
  "mixed_precision": true,
  "seed": 42,
  "rle_order": "C",
  "rle_order_confirmed": false
}
```

Training and validation can run with `rle_order_confirmed=false` because RLE is not used there. Before submission, replace `rle_order` with the verified competition order and set `rle_order_confirmed=true`. Do not tune threshold or post-processing against leaderboard scores.

When the official sample submission is available, the Kaggle notebook derives
the two column names and accepts a test-ID suffix only when the template proves
one constant mapping from test image stems, for example `_segmentation`.
