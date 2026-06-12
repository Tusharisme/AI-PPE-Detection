#!/usr/bin/env python3
"""Merge YOLO datasets that already share the same class order."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "valid", "test")
TARGET_SPLIT = {"valid": "val"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(int(key) for key in raw_names)]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    raise ValueError("data.yaml names must be a list or dict")


def resolve_root(data_yaml: Path, config: dict) -> Path:
    root = Path(str(config.get("path", data_yaml.parent)))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def split_image_dir(root: Path, config: dict, split: str) -> Path | None:
    value = config.get(split)
    if not value:
        return None
    if isinstance(value, list):
        value = value[0]
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / image_path.with_suffix(".txt").name


def link_or_copy(source: Path, target: Path, copy_files: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_files:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def merge_dataset(data_yaml: Path, output: Path, prefix: str, copy_files: bool) -> dict[str, int]:
    config = load_yaml(data_yaml)
    root = resolve_root(data_yaml, config)
    counts = {"images": 0, "labels": 0}

    for source_split in SPLITS:
        target_split = TARGET_SPLIT.get(source_split, source_split)
        image_dir = split_image_dir(root, config, source_split)
        if image_dir is None or not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_for_image(image_path)
            if not label_path.exists():
                continue
            stem = f"{prefix}_{image_path.name}"
            target_image = output / "images" / target_split / stem
            target_label = output / "labels" / target_split / Path(stem).with_suffix(".txt").name
            link_or_copy(image_path, target_image, copy_files)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, target_label)
            counts["images"] += 1
            counts["labels"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge same-class-order YOLO datasets")
    parser.add_argument("--source", action="append", nargs=2, metavar=("PREFIX", "DATA_YAML"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    sources = [(prefix, Path(path)) for prefix, path in args.source]
    for _, data_yaml in sources:
        if not data_yaml.exists():
            print(f"[ERROR] Missing source yaml: {data_yaml}")
            return 1

    names = parse_names(load_yaml(sources[0][1])["names"])
    for _, data_yaml in sources[1:]:
        other_names = parse_names(load_yaml(data_yaml)["names"])
        if other_names != names:
            print(f"[ERROR] Class names/order differ in {data_yaml}")
            print(f"[BASE] {names}")
            print(f"[OTHER] {other_names}")
            return 1

    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}")
            return 1
        shutil.rmtree(args.output)

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = {"images": 0, "labels": 0}
    for prefix, data_yaml in sources:
        counts = merge_dataset(data_yaml.resolve(), args.output, prefix, args.copy)
        total["images"] += counts["images"]
        total["labels"] += counts["labels"]
        print(f"[COUNT] {prefix}: images={counts['images']} labels={counts['labels']}")

    data = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    with (args.output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f"[OK] Wrote {args.output.resolve()} images={total['images']} labels={total['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
