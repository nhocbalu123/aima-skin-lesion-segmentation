# Controlled rerun

A controlled Kaggle rerun was completed on 2026-07-29. Compact evidence is stored in `results/controlled_rerun/`.

## Reproduced workflow

1. Validate canonical image-mask pairing.
2. Create a deterministic image-level split.
3. Audit exact duplicate image content and split leakage.
4. Run the synthetic test suite and model smoke checks.
5. Train with checkpoint selection by minimum validation loss.
6. Reload the selected `.keras` checkpoint.
7. Save validation predictions and compute per-image macro Dice and IoU offline.
8. Compare raw threshold 0.5 against validation-only post-processing.
9. Generate a fixed 512 x 512 C-order RLE submission validated against the sample template.

## Results

- 2,594 validated pairs
- 2,075 training / 519 validation
- 31 completed epochs
- best checkpoint at epoch 23, validation loss 0.260925
- raw validation macro Dice 0.805922
- validation-selected macro Dice 0.808482
- final raw submission public/private 0.80225 / 0.76879
- post-processed ablation public/private 0.80166 / 0.76888

The leaderboard values are author-reported because a leaderboard export is not committed.

## Commands


```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m skin_lesion_segmentation.cli prepare --config configs/controlled_rerun.json
pytest -q -W error

python scripts/verify_repository.py
python -m skin_lesion_segmentation.cli train --config configs/controlled_rerun.json
python -m skin_lesion_segmentation.cli submit \
  --config configs/controlled_rerun.json \
  --checkpoint artifacts/controlled_rerun/best_model.keras \
  --postprocessing results/controlled_rerun/final_submission_parameters.json
```

Required submission configuration:

```json
{
  "rle_order": "C",
  "rle_order_confirmed": true,
  "submission_mask_height": 512,
  "submission_mask_width": 512
}
```

Do not tune additional thresholds or post-processing choices against leaderboard results.

