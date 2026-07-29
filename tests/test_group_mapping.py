from __future__ import annotations

import json
from pathlib import Path

import pytest

from skin_lesion_segmentation.training import load_group_mapping


def test_json_null_group_id_is_not_coerced_to_text(tmp_path: Path) -> None:
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"case": None}), encoding="utf-8")

    with pytest.raises(ValueError, match="blank group ID"):
        load_group_mapping(path)
