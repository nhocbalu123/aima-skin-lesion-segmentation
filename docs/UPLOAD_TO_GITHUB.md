# Upload this source package to GitHub

The ZIP containing this file is a source archive, not a patch. Do not upload
the ZIP itself as one repository file.

## Recommended local workflow

1. Extract the ZIP.
2. Clone your public repository.
3. Copy the extracted source over the clone while preserving the clone's
   `.git` directory.
4. Remove these obsolete duplicate notebooks if they still exist:
   - `AIMA_Skin_Lesion_Kaggle_Raw_Mask_Submission.ipynb`
   - `notebooks/AIMA_Skin_Lesion_Kaggle_Raw_Mask_Submission_v6.ipynb`
   - `notebooks/kaggle_submission_only.ipynb`
5. Review and publish:

```bash
git status --short
git add -A
git diff --cached --stat
git commit -m "Add controlled U-Net comparison infrastructure"
git push
```

Confirm GitHub Actions passes after the push. Never add the dataset,
checkpoints, Kaggle credentials, generated submissions, or caches.

If using only GitHub's web interface, delete the three obsolete notebook paths
first, then upload the extracted files in their directory structure.
