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


def test_generate_submission_uses_explicit_competition_mask_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2
    import json

    from skin_lesion_segmentation.config import ExperimentConfig
    from skin_lesion_segmentation import inference
    from skin_lesion_segmentation.rle import decode_rle

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    image_path = test_dir / "case.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))

    sample_path = tmp_path / "sample.csv"
    pd.DataFrame({"ID": ["case_segmentation"], "Predicted_Mask": [""]}).to_csv(
        sample_path, index=False
    )
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps({"threshold": 0.5, "min_component_size": 0, "morphology_kernel": 0})
    )

    class DummyModel:
        def predict(self, images: np.ndarray, verbose: int = 0) -> np.ndarray:
            result = np.zeros((len(images), 16, 16, 1), dtype=np.float32)
            result[:, 4:12, 4:12, 0] = 1.0
            return result

    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: DummyModel())

    config = ExperimentConfig(
        test_image_dir=str(test_dir),
        sample_submission_path=str(sample_path),
        output_dir=str(tmp_path / "artifacts"),
        image_height=16,
        image_width=16,
        batch_size=1,
        n_tta=1,
        submission_id_column="ID",
        submission_mask_column="Predicted_Mask",
        submission_id_suffix="_segmentation",
        submission_mask_height=8,
        submission_mask_width=6,
        rle_order="C",
        rle_order_confirmed=True,
    )

    result = inference.generate_submission(config, tmp_path / "model.keras", params_path)
    submission = pd.read_csv(result["path"], keep_default_na=False)
    decoded = decode_rle(submission.loc[0, "Predicted_Mask"], (8, 6), order="C")

    assert decoded.shape == (8, 6)
    assert decoded.sum() > 0
    assert result["submission_mask_height"] == 8
    assert result["submission_mask_width"] == 6
    assert result["maximum_valid_rle_position"] == 48

