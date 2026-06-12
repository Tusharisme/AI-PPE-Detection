#!/usr/bin/env python3
"""Create train/val/test splits from a YOLO dataset that only has train data."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_images(source_root: Path) -> list[Path]:
    image_dir = source_root / "train" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(source_root: Path, image_path: Path) -> Path:
    return source_root / "train" / "labels" / image_path.with_suffix(".txt").name


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def write_split(
    source_root: Path,
    output_root: Path,
    split: str,
    images: list[Path],
    copy_files: bool,
) -> int:
    written = 0
    for image_path in images:
        label_path = label_path_for_image(source_root, image_path)
        if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
            continue
        link_or_copy(image_path, output_root / split / "images" / image_path.name, copy_files)
        link_or_copy(label_path, output_root / split / "labels" / label_path.name, True)
        written += 1
    return written


def split_images(images: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def write_data_yaml(source_yaml: Path, output_root: Path) -> None:
    data = load_yaml(source_yaml)
    output_data = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": data["names"],
    }
    if "nc" in data:
        output_data["nc"] = data["nc"]
    (output_root / "data.yaml").write_text(yaml.safe_dump(output_data, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a train-only YOLO dataset")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    source_yaml = args.source / "data.yaml"
    if not source_yaml.exists():
        print(f"[ERROR] Missing source data.yaml: {source_yaml}")
        return 1
    if args.output.exists():
        shutil.rmtree(args.output)

    images = find_images(args.source)
    splits = split_images(images, args.train_ratio, args.val_ratio, args.seed)

    counts = {}
    for split, split_images_list in splits.items():
        counts[split] = write_split(args.source, args.output, split, split_images_list, args.copy)
    write_data_yaml(source_yaml, args.output)

    print(f"[OK] Wrote split dataset: {args.output.resolve()}")
    for split, count in counts.items():
        print(f"[COUNT] {split}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
