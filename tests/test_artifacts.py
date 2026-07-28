from __future__ import annotations

import json
from pathlib import Path

from skin_lesion_segmentation.artifacts import RunArtifacts


def test_run_artifacts_write_real_metadata_and_checksum(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"checkpoint")
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.write_json("effective_config.json", {"seed": 42})
    checkpoint_record = artifacts.record_checkpoint(checkpoint)

    assert (tmp_path / "run" / "effective_config.json").exists()
    assert checkpoint_record["sha256"]
    saved = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert saved["path"] == str(checkpoint)
