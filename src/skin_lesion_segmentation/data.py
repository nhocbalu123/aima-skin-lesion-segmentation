"""Dataset discovery and exact image-mask pairing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


class PairingError(ValueError):
    """Raised when image-mask files cannot be paired unambiguously."""


@dataclass(frozen=True, slots=True)
class SamplePair:
    """A validated image-mask pair identified by a canonical sample ID."""

    sample_id: str
    image_path: Path
    mask_path: Path


def canonical_sample_id(path: Path, mask_suffix: str | None = None) -> str:
    """Return a case-insensitive canonical ID from a supported image filename.

    ``mask_suffix`` is removed only when it occurs at the end of the stem. This
    function does not infer patient or lesion identity from filenames.
    """

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise PairingError(
            f"Unsupported image extension '{path.suffix}' for '{path.name}'. "
            f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    stem = path.stem
    if mask_suffix:
        suffix_folded = mask_suffix.casefold()
        if stem.casefold().endswith(suffix_folded):
            stem = stem[: -len(mask_suffix)]
    if not stem:
        raise PairingError(f"Empty sample ID derived from '{path.name}'.")
    return stem.casefold()


def _supported_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise PairingError(f"Dataset directory does not exist: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _index_by_id(paths: Iterable[Path], *, kind: str, mask_suffix: str | None = None) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        sample_id = canonical_sample_id(path, mask_suffix=mask_suffix)
        if sample_id in indexed:
            raise PairingError(
                f"duplicate {kind} ID '{sample_id}' from '{indexed[sample_id].name}' and '{path.name}'"
            )
        indexed[sample_id] = path
    return indexed


def build_pair_manifest(
    image_dir: Path | str,
    mask_dir: Path | str,
    mask_suffix: str = "_segmentation",
) -> list[SamplePair]:
    """Discover and validate one-to-one image-mask pairs.

    The two ID sets must match exactly. No use of ``zip`` or truncation is
    permitted, so missing, extra, or duplicate files fail loudly.
    """

    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    images = _index_by_id(_supported_files(image_root), kind="image")
    masks = _index_by_id(_supported_files(mask_root), kind="mask", mask_suffix=mask_suffix)

    image_ids = set(images)
    mask_ids = set(masks)
    missing_masks = sorted(image_ids - mask_ids)
    extra_masks = sorted(mask_ids - image_ids)
    if missing_masks or extra_masks:
        details: list[str] = []
        if missing_masks:
            details.append("missing masks for IDs: " + ", ".join(missing_masks))
        if extra_masks:
            details.append("extra masks for IDs: " + ", ".join(extra_masks))
        raise PairingError("Pairing failed: " + "; ".join(details))
    if not images:
        raise PairingError(f"No supported images found in {image_root}")

    return [SamplePair(sample_id, images[sample_id], masks[sample_id]) for sample_id in sorted(image_ids)]
