"""Run artifact writers that never manufacture experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunArtifacts:
    """Write actual run outputs atomically inside one artifact directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, name: str, text: str) -> Path:
        destination = self.path(name)
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    def write_json(self, name: str, value: Any) -> Path:
        return self.write_text(name, json.dumps(value, indent=2, sort_keys=True) + "\n")

    def write_csv(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        materialised = list(rows)
        if not materialised:
            raise ValueError("Cannot write an empty CSV artifact")
        destination = self.path(name)
        fieldnames = list(materialised[0])
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(materialised)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    def record_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        record = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        }
        self.write_json("checkpoint.json", record)
        return record
