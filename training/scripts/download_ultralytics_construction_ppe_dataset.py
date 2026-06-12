#!/usr/bin/env python3
"""Download the Ultralytics Construction-PPE dataset for boot/no-boot labels."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml


DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Ultralytics Construction-PPE dataset")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--output", default=Path("training/datasets/boots_raw"), type=Path)
    parser.add_argument(
        "--archive",
        default=Path("training/datasets/downloads/construction-ppe.zip"),
        type=Path,
    )
    parser.add_argument("--keep-existing-archive", action="store_true")
    args = parser.parse_args()

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists() or not args.keep_existing_archive:
        print(f"[DOWNLOAD] {args.url}")
        urllib.request.urlretrieve(args.url, args.archive)
    else:
        print(f"[SKIP] Reusing existing archive: {args.archive}")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    print(f"[EXTRACT] {args.archive} -> {args.output}")
    with zipfile.ZipFile(args.archive) as archive:
        archive.extractall(args.output)

    data_yaml = args.output / "data.yaml"
    if not data_yaml.exists():
        print(f"[ERROR] Expected data.yaml not found at {data_yaml}")
        return 1

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = "."
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[OK] Dataset ready: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
