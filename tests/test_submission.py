from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skin_lesion_segmentation.submission import SubmissionError, build_submission, sample_id_from_test_path


@pytest.mark.parametrize("name", ["abc.jpg", "abc.JPEG", "abc.PnG", "abc.bmp", "abc.TIF", "abc.tiff"])
def test_submission_id_supports_all_extensions(name: str) -> None:
    assert sample_id_from_test_path(Path(name)) == "abc"


def test_sample_submission_controls_order_and_columns(tmp_path: Path) -> None:
    paths = [tmp_path / "b.JPG", tmp_path / "a.png"]
    predictions = [np.zeros((2, 2)), np.ones((2, 2))]
    sample = pd.DataFrame({"id": ["a", "b"], "segmentation": ["", ""]})

    result, summary = build_submission(paths, predictions, sample_submission=sample, rle_order="C")

    assert result.columns.tolist() == ["id", "segmentation"]
    assert result["id"].tolist() == ["a", "b"]
    assert summary["rows"] == 2
    assert summary["sample_submission_validated"] is True


def test_sample_submission_id_mismatch_fails(tmp_path: Path) -> None:
    paths = [tmp_path / "a.png"]
    sample = pd.DataFrame({"id": ["b"], "segmentation": [""]})
    with pytest.raises(SubmissionError, match="ID mismatch"):
        build_submission(paths, [np.zeros((2, 2))], sample_submission=sample)


def test_duplicate_test_ids_fail(tmp_path: Path) -> None:
    paths = [tmp_path / "a.jpg", tmp_path / "a.PNG"]
    with pytest.raises(SubmissionError, match="duplicate"):
        build_submission(paths, [np.zeros((2, 2)), np.zeros((2, 2))])


def test_case_only_duplicate_test_ids_fail(tmp_path: Path) -> None:
    paths = [tmp_path / "A.jpg", tmp_path / "a.PNG"]
    with pytest.raises(SubmissionError, match="duplicate"):
        build_submission(paths, [np.zeros((2, 2)), np.zeros((2, 2))])


def test_submission_requires_explicit_rle_contract_confirmation(tmp_path: Path) -> None:
    from skin_lesion_segmentation.config import ExperimentConfig
    from skin_lesion_segmentation.inference import generate_submission

    config = ExperimentConfig(
        sample_submission_path=str(tmp_path / "sample.csv"),
        rle_order="C",
        rle_order_confirmed=False,
    )
    with pytest.raises(ValueError, match="confirmed"):
        generate_submission(config, tmp_path / "model.keras", tmp_path / "post.json")


def test_infer_submission_id_suffix_from_official_template(tmp_path: Path) -> None:
    from skin_lesion_segmentation.submission import infer_submission_id_suffix

    paths = [tmp_path / "case_a.jpg", tmp_path / "case_b.PNG"]
    sample = pd.DataFrame({"ID": ["case_b_segmentation", "case_a_segmentation"], "Predicted_Mask": ["", ""]})

    assert infer_submission_id_suffix(paths, sample, id_column="ID") == "_segmentation"


def test_infer_submission_id_suffix_returns_empty_for_direct_stems(tmp_path: Path) -> None:
    from skin_lesion_segmentation.submission import infer_submission_id_suffix

    paths = [tmp_path / "case_a.jpg", tmp_path / "case_b.PNG"]
    sample = pd.DataFrame({"ID": ["case_b", "case_a"], "Predicted_Mask": ["", ""]})

    assert infer_submission_id_suffix(paths, sample, id_column="ID") == ""


def test_infer_submission_id_suffix_rejects_nonuniform_mapping(tmp_path: Path) -> None:
    from skin_lesion_segmentation.submission import infer_submission_id_suffix

    paths = [tmp_path / "case_a.jpg", tmp_path / "case_b.PNG"]
    sample = pd.DataFrame({"ID": ["case_a_segmentation", "different"], "Predicted_Mask": ["", ""]})

    with pytest.raises(SubmissionError, match="constant suffix"):
        infer_submission_id_suffix(paths, sample, id_column="ID")


def test_build_submission_applies_explicit_template_confirmed_suffix(tmp_path: Path) -> None:
    paths = [tmp_path / "case_b.JPG", tmp_path / "case_a.png"]
    predictions = [np.zeros((2, 2)), np.ones((2, 2))]
    sample = pd.DataFrame({"ID": ["case_a_segmentation", "case_b_segmentation"], "Predicted_Mask": ["", ""]})

    result, _ = build_submission(
        paths,
        predictions,
        sample_submission=sample,
        id_column="ID",
        mask_column="Predicted_Mask",
        submission_id_suffix="_segmentation",
        rle_order="C",
    )

    assert result["ID"].tolist() == ["case_a_segmentation", "case_b_segmentation"]
