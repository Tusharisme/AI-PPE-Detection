#!/usr/bin/env python3
"""Convert camera-87 Roboflow YOLO export to the production 12-class schema."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml


PRODUCTION_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
    10: "Safety Boots",
    11: "NO-Safety Boots",
}

NAME_TO_PRODUCTION_ID = {
    "hardhat": 0,
    "no hardhat": 2,
    "no-hardhat": 2,
    "no_hardhat": 2,
    "no safety vest": 4,
    "no-safety-vest": 4,
    "no_safety_vest": 4,
    "safety vest": 7,
    "safety-vest": 7,
    "safety_vest": 7,
    "safety boots": 10,
    "safety-boots": 10,
    "safety_boots": 10,
    "no safety boots": 11,
    "no-safety-boots": 11,
    "no_safety_boots": 11,
    "no safety boot": 11,
    "no-safety-boot": 11,
    "person": 5,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize camera-87 Roboflow YOLO export")
    parser.add_argument("--raw-dir", default=Path("training/datasets/camera87_raw"), type=Path)
    parser.add_argument("--output-dir", default=Path("training/datasets/camera87_3x_12class"), type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=87, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", " ").replace("-", " ")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def load_class_map(raw_dir: Path) -> dict[int, int]:
    data_yaml = raw_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing Roboflow data.yaml: {data_yaml}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        names_by_id = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, list):
        names_by_id = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        raise ValueError("data.yaml names must be a list or dict")

    class_map: dict[int, int] = {}
    for raw_id, raw_name in names_by_id.items():
        normalized = normalize_name(raw_name)
        production_id = NAME_TO_PRODUCTION_ID.get(normalized)
        if production_id is None:
            raise ValueError(f"No production mapping for Roboflow class {raw_id}: {raw_name!r}")
        class_map[raw_id] = production_id
    return class_map


def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = raw_dir / "train" / "images"
    label_dir = raw_dir / "train" / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Expected Roboflow train/images and train/labels folders")

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image: {image_path.name}")
        pairs.append((image_path, label_path))
    return pairs


def convert_label(label_path: Path, class_map: dict[int, int]) -> str:
    converted_lines = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label in {label_path}:{line_number}: {line!r}")
        raw_class_id = int(float(parts[0]))
        production_id = class_map.get(raw_class_id)
        if production_id is None:
            raise ValueError(f"Unmapped class id {raw_class_id} in {label_path}:{line_number}")
        converted_lines.append(" ".join([str(production_id), *parts[1:]]))
    return "\n".join(converted_lines) + ("\n" if converted_lines else "")


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in PRODUCTION_NAMES.items())
    content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        f"names:\n{names}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir
    output_dir = args.output_dir

    if output_dir.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {output_dir}. Use --overwrite to replace it.")
            return 1
        shutil.rmtree(output_dir)

    class_map = load_class_map(raw_dir)
    pairs = find_pairs(raw_dir)
    if len(pairs) < 2:
        print(f"[ERROR] Need at least 2 image/label pairs, found {len(pairs)}")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_ratio))
    val_pairs = set(pairs[:val_count])

    counts = {split: {"images": 0, "labels": 0, "classes": {class_id: 0 for class_id in PRODUCTION_NAMES}} for split in ("train", "val")}
    for image_path, label_path in pairs:
        split = "val" if (image_path, label_path) in val_pairs else "train"
        output_image = output_dir / "images" / split / image_path.name
        output_label = output_dir / "labels" / split / label_path.name

        link_or_copy(image_path, output_image)
        converted = convert_label(label_path, class_map)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text(converted, encoding="utf-8")

        counts[split]["images"] += 1
        counts[split]["labels"] += 1
        for line in converted.splitlines():
            class_id = int(line.split()[0])
            counts[split]["classes"][class_id] += 1

    write_data_yaml(output_dir)

    print(f"[OK] Wrote {output_dir.resolve()}")
    print(f"[OK] Class map: {class_map}")
    for split in ("train", "val"):
        print(f"[COUNT] {split}: images={counts[split]['images']} labels={counts[split]['labels']}")
        for class_id, class_name in PRODUCTION_NAMES.items():
            count = counts[split]["classes"][class_id]
            if count:
                print(f"[COUNT] {split}: {class_id} {class_name}: {count}")
    print(f"[OK] data.yaml: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
