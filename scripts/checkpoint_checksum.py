#!/usr/bin/env python3
"""Print or verify a SHA-256 checksum for an external checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from skin_lesion_segmentation.artifacts import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected")
    args = parser.parse_args()
    actual = sha256_file(args.checkpoint)
    print(actual)
    if args.expected is not None and actual.casefold() != args.expected.casefold():
        raise SystemExit("Checkpoint checksum mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
