# Notebooks

- `original_pipeline.ipynb` is the historical three-cell notebook preserved for traceability. It contains the original monolithic implementation, obsolete dependency arguments, and claims that are not used by the maintained pipeline. Do not use it as the source of truth for reruns.
- `skin_lesion_segmentation.ipynb` is the thin maintained orchestrator. The implementation lives under `src/skin_lesion_segmentation/` and is covered by synthetic tests.
- `kaggle_controlled_rerun.ipynb` is the complete Kaggle workflow. It extracts the repaired repository, checks the runtime, discovers the authorised data, runs tests, prepares manifests, gates training, reviews generated evidence, and gates submission until the official RLE order is confirmed.
