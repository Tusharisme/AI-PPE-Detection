#!/usr/bin/env python3
"""Build a camera-focused YOLO fine-tuning dataset.

The output keeps the broad PPE/boots dataset once, repeats only the onsite
camera train split, and keeps the onsite validation split once as a real
camera holdout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


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


def split_dir(data_yaml: Path, split: str) -> Path | None:
    config = load_yaml(data_yaml)
    value = config.get(split)
    if not value:
        return None
    if isinstance(value, list):
        value = value[0]
    path = Path(str(value))
    if not path.is_absolute():
        path = resolve_root(data_yaml, config) / path
    return path.resolve()


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
        return
    try:
        os.link(source, target)
    except OSError:
        target.symlink_to(source.resolve())


def image_paths(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    ]


def add_split(
    data_yaml: Path,
    source_split: str,
    target_split: str,
    output: Path,
    prefix: str,
    repeat: int,
    copy_files: bool,
) -> tuple[int, int]:
    images = image_paths(split_dir(data_yaml, source_split))
    count_images = 0
    count_labels = 0
    for repeat_index in range(1, repeat + 1):
        for image_path in images:
            label_path = label_for_image(image_path)
            if not label_path.exists():
                continue
            stem = f"{prefix}_r{repeat_index}_{image_path.name}"
            output_image = output / "images" / target_split / stem
            output_label = output / "labels" / target_split / Path(stem).with_suffix(".txt").name
            link_or_copy(image_path, output_image, copy_files)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, output_label)
            count_images += 1
            count_labels += 1
    return count_images, count_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Build broad PPE + repeated onsite camera YOLO dataset")
    parser.add_argument("--base-data", required=True, type=Path)
    parser.add_argument("--camera-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-train-repeats", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    if not args.base_data.exists():
        print(f"[ERROR] Missing base dataset YAML: {args.base_data}")
        return 1
    if not args.camera_data.exists():
        print(f"[ERROR] Missing camera dataset YAML: {args.camera_data}")
        return 1

    base_names = parse_names(load_yaml(args.base_data)["names"])
    camera_names = parse_names(load_yaml(args.camera_data)["names"])
    if base_names != camera_names:
        print("[ERROR] Base and camera class names differ")
        print(f"[BASE] {base_names}")
        print(f"[CAMERA] {camera_names}")
        return 1

    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}. Use --overwrite.")
            return 1
        shutil.rmtree(args.output)

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        counts[f"base_{split}"] = add_split(
            args.base_data.resolve(), split, split, args.output, f"base_{split}", 1, args.copy
        )

    counts["camera_train"] = add_split(
        args.camera_data.resolve(),
        "train",
        "train",
        args.output,
        "camera_train",
        args.camera_train_repeats,
        args.copy,
    )
    counts["camera_val"] = add_split(
        args.camera_data.resolve(), "val", "val", args.output, "camera_val", 1, args.copy
    )

    data = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(base_names)},
    }
    with (args.output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    total_images = 0
    total_labels = 0
    for key, (image_count, label_count) in counts.items():
        total_images += image_count
        total_labels += label_count
        print(f"[COUNT] {key}: images={image_count} labels={label_count}")
    print(f"[OK] Wrote {args.output.resolve()} images={total_images} labels={total_labels}")
    print(f"[OK] data.yaml: {args.output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
