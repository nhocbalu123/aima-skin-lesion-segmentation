"""Reproducible image/group splitting with exact-duplicate leakage audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .data import SamplePair


class LeakageError(ValueError):
    """Raised when a group or exact duplicate crosses dataset splits."""



def _decoded_content_sha256(path: Path, *, kind: str) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        audit = "duplicate audit" if kind == "image" else "content audit"
        raise IOError(f"Could not decode {kind} for {audit}: {path}")
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def image_content_sha256(path: Path) -> str:
    """Hash decoded image content so encoding differences cannot hide duplicates."""

    return _decoded_content_sha256(path, kind="image")


def mask_content_sha256(path: Path) -> str:
    """Hash decoded annotation content for manifest identity."""

    return _decoded_content_sha256(path, kind="mask")




def _normalise_group_mapping(
    group_mapping: Mapping[str, str],
    sample_ids: Sequence[str],
) -> dict[str, str]:
    normalised: dict[str, str] = {}
    source_keys: dict[str, str] = {}
    for raw_sample_id, raw_group_id in group_mapping.items():
        sample_id = str(raw_sample_id).casefold()
        if raw_group_id is None:
            raise ValueError(f"blank group ID for sample '{raw_sample_id}'")
        group_id = str(raw_group_id).strip()
        if not sample_id:
            raise ValueError("Group mapping contains a blank sample ID")
        if not group_id:
            raise ValueError(f"blank group ID for sample '{raw_sample_id}'")
        if sample_id in normalised:
            raise ValueError(
                f"duplicate group mapping IDs after case normalisation: "
                f"'{source_keys[sample_id]}' and '{raw_sample_id}'"
            )
        normalised[sample_id] = group_id
        source_keys[sample_id] = str(raw_sample_id)

    expected = set(sample_ids)
    actual = set(normalised)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError("missing group IDs for: " + ", ".join(missing))
    if extra:
        raise ValueError("extra group IDs for: " + ", ".join(extra))
    return normalised


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_split_manifest(
    pairs: Sequence[SamplePair],
    val_fraction: float,
    seed: int,
    *,
    group_mapping: Mapping[str, str] | None = None,
    output_path: Path | str | None = None,
) -> list[dict[str, object]]:
    """Create an image-level or explicit group-aware split manifest.

    Patient/lesion groups are used only when supplied explicitly. Exact image
    duplicates are always grouped before assignment. With no metadata this is
    an image-level split; patient/lesion independence cannot be guaranteed.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(pairs) < 2:
        raise ValueError("At least two samples are required")
    sample_ids = [pair.sample_id for pair in pairs]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique")
    normalised_groups = (
        _normalise_group_mapping(group_mapping, sample_ids)
        if group_mapping is not None
        else None
    )

    hashes = {pair.sample_id: image_content_sha256(pair.image_path) for pair in pairs}
    mask_hashes = {
        pair.sample_id: mask_content_sha256(pair.mask_path)
        for pair in pairs
    }
    union_find = _UnionFind(sample_ids)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for sample_id, digest in hashes.items():
        by_hash[digest].append(sample_id)
    for ids in by_hash.values():
        for sample_id in ids[1:]:
            union_find.union(ids[0], sample_id)
    if normalised_groups is not None:
        by_group: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids:
            by_group[normalised_groups[sample_id]].append(sample_id)
        for ids in by_group.values():
            for sample_id in ids[1:]:
                union_find.union(ids[0], sample_id)

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        components[union_find.find(sample_id)].append(sample_id)
    component_list = [sorted(ids) for _, ids in sorted(components.items())]
    rng = np.random.default_rng(seed)
    rng.shuffle(component_list)
    target_validation = max(1, int(round(len(pairs) * val_fraction)))
    validation_ids: set[str] = set()
    for component in component_list:
        if len(validation_ids) >= target_validation and validation_ids:
            break
        if len(validation_ids) + len(component) == len(pairs):
            continue
        validation_ids.update(component)
    if not validation_ids:
        validation_ids.update(component_list[0])
    if len(validation_ids) == len(pairs):
        validation_ids.difference_update(component_list[-1])
    if not validation_ids or len(validation_ids) == len(pairs):
        raise ValueError("Could not create non-empty train and validation splits with current groups")

    rows: list[dict[str, object]] = []
    for pair in sorted(pairs, key=lambda item: item.sample_id):
        rows.append(
            {
                "sample_id": pair.sample_id,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "split": "validation" if pair.sample_id in validation_ids else "train",
                "group_id": normalised_groups[pair.sample_id] if normalised_groups is not None else None,
                "image_sha256": hashes[pair.sample_id],
                "mask_sha256": mask_hashes[pair.sample_id],
            }
        )
    audit_split_integrity(rows)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def build_comparison_manifest(
    pairs: Sequence[SamplePair],
    *,
    validation_fraction: float,
    internal_test_fraction: float,
    split_seed: int,
    group_mapping: Mapping[str, str] | None = None,
    output_path: Path | str | None = None,
) -> list[dict[str, object]]:
    """Create a deterministic train/validation/internal-test manifest.

    Exact decoded-image duplicates and any explicitly supplied real groups are
    assigned as indivisible components. Mask hashes are evidence inputs, not
    grouping keys, because many legitimate samples can share an empty mask.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < internal_test_fraction < 1.0:
        raise ValueError("internal_test_fraction must be between 0 and 1")
    if validation_fraction + internal_test_fraction >= 1.0:
        raise ValueError("validation and internal-test fractions must sum to less than 1")
    if len(pairs) < 3:
        raise ValueError("At least three samples are required")
    sample_ids = [pair.sample_id for pair in pairs]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique")
    normalised_groups = (
        _normalise_group_mapping(group_mapping, sample_ids)
        if group_mapping is not None
        else None
    )

    image_hashes = {
        pair.sample_id: image_content_sha256(pair.image_path)
        for pair in pairs
    }
    mask_hashes = {
        pair.sample_id: mask_content_sha256(pair.mask_path)
        for pair in pairs
    }
    union_find = _UnionFind(sample_ids)
    by_image_hash: dict[str, list[str]] = defaultdict(list)
    for sample_id, digest in image_hashes.items():
        by_image_hash[digest].append(sample_id)
    for ids in by_image_hash.values():
        for sample_id in ids[1:]:
            union_find.union(ids[0], sample_id)
    if normalised_groups is not None:
        by_group: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids:
            by_group[normalised_groups[sample_id]].append(sample_id)
        for ids in by_group.values():
            for sample_id in ids[1:]:
                union_find.union(ids[0], sample_id)

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        components[union_find.find(sample_id)].append(sample_id)
    component_list = [sorted(ids) for _, ids in sorted(components.items())]
    if len(component_list) < 3:
        raise ValueError(
            "A three-way comparison requires at least three independent groups "
            "after duplicate and metadata grouping"
        )
    rng = np.random.default_rng(split_seed)
    rng.shuffle(component_list)

    n_samples = len(pairs)
    target_validation = max(1, int(round(n_samples * validation_fraction)))
    target_internal_test = max(1, int(round(n_samples * internal_test_fraction)))
    target_train = n_samples - target_validation - target_internal_test
    if target_train < 1:
        raise ValueError("Fractions leave no samples for training")
    targets = {
        "train": target_train,
        "validation": target_validation,
        "internal_test": target_internal_test,
    }
    counts = {split: 0 for split in targets}
    split_by_id: dict[str, str] = {}
    split_order = ("train", "validation", "internal_test")
    for component in component_list:
        selected_split = min(
            split_order,
            key=lambda split: counts[split] / targets[split],
        )
        for sample_id in component:
            split_by_id[sample_id] = selected_split
        counts[selected_split] += len(component)
    empty = [split for split, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            "Could not create non-empty comparison splits with current groups: "
            + ", ".join(empty)
        )

    rows: list[dict[str, object]] = []
    for pair in sorted(pairs, key=lambda item: item.sample_id):
        rows.append(
            {
                "sample_id": pair.sample_id,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "split": split_by_id[pair.sample_id],
                "group_id": (
                    normalised_groups[pair.sample_id]
                    if normalised_groups is not None
                    else None
                ),
                "image_sha256": image_hashes[pair.sample_id],
                "mask_sha256": mask_hashes[pair.sample_id],
            }
        )
    audit_split_integrity(rows)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return rows


def manifest_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash portable manifest identity fields, excluding mount-specific paths."""

    if not rows:
        raise ValueError("Cannot hash an empty manifest")
    audit_split_integrity(rows)
    identity_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item.get("sample_id", "")).casefold()):
        required = ("sample_id", "split", "image_sha256", "mask_sha256")
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise ValueError(
                "Manifest identity fields are missing: " + ", ".join(missing)
            )
        identity_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "group_id": (
                    None if row.get("group_id") in (None, "") else str(row["group_id"])
                ),
                "image_sha256": str(row["image_sha256"]),
                "mask_sha256": str(row["mask_sha256"]),
            }
        )
    canonical = json.dumps(
        identity_rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest_against_pairs(
    rows: Sequence[Mapping[str, object]],
    pairs: Sequence[SamplePair],
) -> list[dict[str, object]]:
    """Verify an immutable manifest against the currently mounted dataset.

    Paths are rebased to the validated local pairs after both decoded image and
    mask hashes match, making the manifest portable without weakening identity.
    """

    audit_split_integrity(rows)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    pairs_by_id = {pair.sample_id: pair for pair in pairs}
    if set(rows_by_id) != set(pairs_by_id):
        raise LeakageError("Manifest IDs and discovered dataset IDs do not match exactly")
    validated: list[dict[str, object]] = []
    for sample_id in sorted(rows_by_id):
        row = rows_by_id[sample_id]
        pair = pairs_by_id[sample_id]
        expected_image = str(row.get("image_sha256", ""))
        expected_mask = str(row.get("mask_sha256", ""))
        if not expected_image or not expected_mask:
            raise LeakageError(
                f"Manifest content hashes are incomplete for sample '{sample_id}'"
            )
        actual_image = image_content_sha256(pair.image_path)
        actual_mask = mask_content_sha256(pair.mask_path)
        if expected_image != actual_image or expected_mask != actual_mask:
            raise LeakageError(
                f"Manifest content does not match the mounted dataset for sample '{sample_id}'"
            )
        validated.append(
            {
                **dict(row),
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
            }
        )
    audit_split_integrity(validated)
    return validated


def audit_split_integrity(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise LeakageError("Split manifest is empty")
    valid_splits = {"train", "validation", "test", "internal_test"}
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, set[str]] = defaultdict(set)
    seen_sample_ids: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise LeakageError("Split manifest contains a blank sample ID")
        canonical_id = sample_id.casefold()
        if canonical_id in seen_sample_ids:
            raise LeakageError(
                "duplicate sample IDs in split manifest: "
                f"'{seen_sample_ids[canonical_id]}' and '{sample_id}'"
            )
        seen_sample_ids[canonical_id] = sample_id
        split = str(row.get("split"))
        if split not in valid_splits:
            raise LeakageError(f"Unknown split '{split}'")
        digest = row.get("image_sha256")
        if digest:
            by_hash[str(digest)].add(split)
        group = row.get("group_id")
        if group is not None and str(group) != "":
            by_group[str(group)].add(split)
    crossed_hashes = sorted(key for key, splits in by_hash.items() if len(splits) > 1)
    if crossed_hashes:
        raise LeakageError("exact duplicate images cross splits: " + ", ".join(crossed_hashes))
    crossed_groups = sorted(key for key, splits in by_group.items() if len(splits) > 1)
    if crossed_groups:
        raise LeakageError("group IDs cross splits: " + ", ".join(crossed_groups))
