#!/usr/bin/env python3
"""Build a single-detector PPE dataset from the v3 PPE and person ZIP exports.

Output classes:
0 Person
1 NO-Hardhat
2 NO-Safety Vest
3 NO-Safety Boots

The output contains both YOLOv8 labels and RF-DETR-style COCO annotations so
YOLOv8s and RF-DETR can train on the same normalized examples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
SPLITS = ("train", "valid", "test")
YOLO_SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}
COCO_SPLIT_DIR = {"train": "train", "valid": "valid", "test": "test"}
TARGET_NAMES = ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"]
TARGET_ID_BY_NAME = {name: index for index, name in enumerate(TARGET_NAMES)}
COCO_CATEGORIES = [
    {"id": index + 1, "name": name, "supercategory": "ppe"}
    for index, name in enumerate(TARGET_NAMES)
]
ALIASES = {
    "person": "Person",
    "no hardhat": "NO-Hardhat",
    "no-hardhat": "NO-Hardhat",
    "no_hardhat": "NO-Hardhat",
    "no helmet": "NO-Hardhat",
    "no-helmet": "NO-Hardhat",
    "no safety vest": "NO-Safety Vest",
    "no-safety vest": "NO-Safety Vest",
    "no-safety-vest": "NO-Safety Vest",
    "no_safety_vest": "NO-Safety Vest",
    "no vest": "NO-Safety Vest",
    "no-vest": "NO-Safety Vest",
    "no safety boots": "NO-Safety Boots",
    "no-safety boots": "NO-Safety Boots",
    "no-safety-boots": "NO-Safety Boots",
    "no_safety_boots": "NO-Safety Boots",
    "no safety boot": "NO-Safety Boots",
    "no-safety-boot": "NO-Safety Boots",
    "no safety shoe": "NO-Safety Boots",
    "not safety shoe": "NO-Safety Boots",
}


@dataclass(frozen=True)
class Dataset:
    root: Path
    names: dict[int, str]
    images_by_split: dict[str, list[Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO and COCO single-detector PPE dataset")
    parser.add_argument("--ppe-zip", default=Path("ppe 2.v3-no-augmentation.yolov8.zip"), type=Path)
    parser.add_argument(
        "--person-zip",
        default=Path("Person Detection.v2-no-augmentation.yolov8.zip"),
        type=Path,
    )
    parser.add_argument(
        "--raw-root",
        default=Path("training/datasets/raw"),
        type=Path,
        help="Where ZIP contents are extracted.",
    )
    parser.add_argument(
        "--output",
        default=Path("training/datasets/ppe_v3_person_single_detector"),
        type=Path,
    )
    parser.add_argument("--camera-repeat", default=4, type=int)
    parser.add_argument("--person-train-limit", default=2000, type=int)
    parser.add_argument(
        "--person-valid-limit",
        default=1550,
        type=int,
        help="Add this many web person-only validation images so person validation is not tiny.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlink/symlink.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing ZIP: {zip_path}")
    marker = target_dir / ".extracted_from"
    if marker.exists() and marker.read_text(encoding="utf-8") == str(zip_path.resolve()):
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    marker.write_text(str(zip_path.resolve()), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ValueError("data.yaml names must be a list or dict")


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def canonical_name(name: str) -> str | None:
    if name in TARGET_ID_BY_NAME:
        return name
    normalized = normalize_name(name)
    return ALIASES.get(normalized) or ALIASES.get(normalized.replace(" ", "-"))


def candidate_split_paths(root: Path, value: Any) -> list[Path]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    candidates: list[Path] = []
    for item in values:
        raw = Path(str(item))
        if raw.is_absolute():
            candidates.append(raw)
            continue
        candidates.append(root / raw)
        text = str(item)
        if text.startswith("../"):
            candidates.append(root / text[3:])
    return candidates


def resolve_split(root: Path, value: Any) -> list[Path]:
    for candidate in candidate_split_paths(root, value):
        if candidate.is_dir():
            return sorted(
                path for path in candidate.iterdir()
                if path.is_file() and path.suffix in IMAGE_SUFFIXES
            )
    return []


def load_dataset(data_yaml: Path) -> Dataset:
    config = load_yaml(data_yaml)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()
    return Dataset(
        root=root,
        names=parse_names(config["names"]),
        images_by_split={
            "train": resolve_split(root, config.get("train")),
            "valid": resolve_split(root, config.get("val") or config.get("valid")),
            "test": resolve_split(root, config.get("test")),
        },
    )


def build_class_map(names: dict[int, str]) -> dict[int, int]:
    class_map: dict[int, int] = {}
    for source_id, source_name in names.items():
        target = canonical_name(source_name)
        if target is not None:
            class_map[source_id] = TARGET_ID_BY_NAME[target]
    return class_map


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def remap_label(label_path: Path, class_map: dict[int, int]) -> list[str]:
    if not label_path.exists():
        return []
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            source_id = int(float(parts[0]))
        except ValueError:
            continue
        target_id = class_map.get(source_id)
        if target_id is None:
            continue
        lines.append(" ".join([str(target_id), *parts[1:5]]))
    return lines


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


def image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def yolo_line_to_coco_bbox(line: str, width: int, height: int) -> tuple[int, list[float], float]:
    parts = line.split()
    class_id = int(parts[0])
    xc, yc, bw, bh = [float(value) for value in parts[1:5]]
    box_w = bw * width
    box_h = bh * height
    x = (xc * width) - (box_w / 2.0)
    y = (yc * height) - (box_h / 2.0)
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    box_w = max(0.0, min(float(width) - x, box_w))
    box_h = max(0.0, min(float(height) - y, box_h))
    return class_id, [x, y, box_w, box_h], box_w * box_h


def write_data_yaml(output: Path) -> None:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_NAMES)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_coco_annotations(output: Path, split: str, image_paths: list[Path]) -> None:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, image_path in enumerate(sorted(image_paths), start=1):
        width, height = image_size(image_path)
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })
        label_path = output / split / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, bbox, area = yolo_line_to_coco_bbox(line, width, height)
            if area <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
    payload = {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}
    (output / split / "_annotations.coco.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_example(
    image_path: Path,
    output: Path,
    yolo_split: str,
    coco_split: str,
    prefix: str,
    class_map: dict[int, int],
    copy_files: bool,
    counts: dict[str, int],
) -> Path:
    lines = remap_label(label_for_image(image_path), class_map)
    image_name = f"{prefix}_{image_path.name}"
    label_name = Path(image_name).with_suffix(".txt").name

    yolo_image = output / "images" / yolo_split / image_name
    yolo_label = output / "labels" / yolo_split / label_name
    coco_image = output / coco_split / image_name
    coco_label = output / coco_split / label_name

    link_or_copy(image_path, yolo_image, copy_files)
    link_or_copy(image_path, coco_image, copy_files)
    for label_target in (yolo_label, coco_label):
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    for line in lines:
        class_id = int(line.split()[0])
        counts[TARGET_NAMES[class_id]] += 1
    return coco_image


def limited_sample(images: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or len(images) <= limit:
        return sorted(images)
    rng = random.Random(seed)
    selected = list(images)
    rng.shuffle(selected)
    return sorted(selected[:limit])


def main() -> int:
    args = parse_args()
    ppe_root = args.raw_root / "ppe_v3_no_augmentation_yolov8"
    person_root = args.raw_root / "person_detection_v2_no_augmentation_yolov8"

    extract_zip(args.ppe_zip, ppe_root)
    extract_zip(args.person_zip, person_root)

    ppe = load_dataset(ppe_root / "data.yaml")
    person = load_dataset(person_root / "data.yaml")
    ppe_map = build_class_map(ppe.names)
    person_map = build_class_map(person.names)

    missing = [name for name in TARGET_NAMES if TARGET_ID_BY_NAME[name] not in set(ppe_map.values())]
    if missing:
        print(f"[ERROR] PPE dataset is missing required classes: {', '.join(missing)}")
        print(f"[INFO] PPE names: {ppe.names}")
        return 1
    if set(person_map.values()) != {TARGET_ID_BY_NAME["Person"]}:
        print(f"[ERROR] Person dataset must map only to Person. names={person.names} map={person_map}")
        return 1

    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}. Use --overwrite.")
            return 1
        shutil.rmtree(args.output)

    for split in ("train", "val", "test"):
        for root in ("images", "labels"):
            (args.output / root / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (args.output / split).mkdir(parents=True, exist_ok=True)

    counts = {split: {name: 0 for name in TARGET_NAMES} for split in ("train", "val", "test")}
    image_counts = {split: 0 for split in ("train", "val", "test")}
    coco_images = {split: [] for split in ("train", "valid", "test")}

    for repeat_index in range(1, max(1, args.camera_repeat) + 1):
        for image_path in ppe.images_by_split["train"]:
            coco_images["train"].append(write_example(
                image_path,
                args.output,
                "train",
                "train",
                f"welspun_r{repeat_index}",
                ppe_map,
                args.copy,
                counts["train"],
            ))
            image_counts["train"] += 1

    person_train = limited_sample(person.images_by_split["train"], args.person_train_limit, args.seed)
    for image_path in person_train:
        coco_images["train"].append(write_example(
            image_path,
            args.output,
            "train",
            "train",
            "web_person",
            person_map,
            args.copy,
            counts["train"],
        ))
        image_counts["train"] += 1

    person_valid = limited_sample(person.images_by_split["valid"], args.person_valid_limit, args.seed + 1)
    for image_path in person_valid:
        coco_images["valid"].append(write_example(
            image_path,
            args.output,
            "val",
            "valid",
            "web_person_valid",
            person_map,
            args.copy,
            counts["val"],
        ))
        image_counts["val"] += 1

    for source_split in ("valid", "test"):
        yolo_split = YOLO_SPLIT_DIR[source_split]
        coco_split = COCO_SPLIT_DIR[source_split]
        for image_path in ppe.images_by_split[source_split]:
            coco_images[coco_split].append(write_example(
                image_path,
                args.output,
                yolo_split,
                coco_split,
                f"welspun_{source_split}",
                ppe_map,
                args.copy,
                counts[yolo_split],
            ))
            image_counts[yolo_split] += 1

    write_data_yaml(args.output)
    for split in ("train", "valid", "test"):
        write_coco_annotations(args.output, split, coco_images[split])

    print(f"[OK] Wrote dataset: {args.output.resolve()}")
    print(f"[OK] YOLO data: {args.output / 'data.yaml'}")
    print("[OK] RF-DETR COCO annotations: train/valid/test _annotations.coco.json")
    print(f"[INFO] PPE class map: {ppe_map}")
    print(f"[INFO] Person class map: {person_map}")
    for split in ("train", "val", "test"):
        print(f"[COUNT] {split}: images={image_counts[split]} labels={sum(counts[split].values())}")
        for name in TARGET_NAMES:
            print(f"  {name}: {counts[split][name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
