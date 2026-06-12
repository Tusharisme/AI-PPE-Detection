#!/usr/bin/env python3
"""Create a YOLO dataset of person crops from full-frame person annotations."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
TARGET_SPLIT = {"valid": "val"}
PPE_CLASS_NAMES = {
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Safety Vest",
    "Safety Boots",
    "NO-Safety Boots",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ValueError("data.yaml names must be list or dict")


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


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            labels.append((int(float(parts[0])), *(float(value) for value in parts[1:5])))
        except ValueError:
            continue
    return labels


def yolo_to_xyxy(label: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, float, float, float, float]:
    class_id, cx, cy, box_w, box_h = label
    abs_cx = cx * width
    abs_cy = cy * height
    abs_w = box_w * width
    abs_h = box_h * height
    return class_id, abs_cx - abs_w / 2, abs_cy - abs_h / 2, abs_cx + abs_w / 2, abs_cy + abs_h / 2


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def expand_person_box(box: tuple[int, float, float, float, float], width: int, height: int, margin: float) -> tuple[int, int, int, int] | None:
    _, x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    crop = (
        int(clamp(x1 - box_w * margin, 0, width - 1)),
        int(clamp(y1 - box_h * margin, 0, height - 1)),
        int(clamp(x2 + box_w * margin, 0, width - 1)),
        int(clamp(y2 + box_h * margin, 0, height - 1)),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    return crop


def remap_label_to_crop(
    box: tuple[int, float, float, float, float],
    crop: tuple[int, int, int, int],
    min_area_ratio: float,
) -> str | None:
    class_id, x1, y1, x2, y2 = box
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    inter_x1 = clamp(x1, crop_x1, crop_x2)
    inter_y1 = clamp(y1, crop_y1, crop_y2)
    inter_x2 = clamp(x2, crop_x1, crop_x2)
    inter_y2 = clamp(y2, crop_y1, crop_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 1 or inter_h <= 1:
        return None
    original_area = max(1.0, (x2 - x1) * (y2 - y1))
    if (inter_w * inter_h) / original_area < min_area_ratio:
        return None
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    cx = ((inter_x1 + inter_x2) / 2 - crop_x1) / crop_w
    cy = ((inter_y1 + inter_y2) / 2 - crop_y1) / crop_h
    width = inter_w / crop_w
    height = inter_h / crop_h
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create person-crop YOLO dataset")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--person-class", default="Person")
    parser.add_argument("--margin", default=0.15, type=float)
    parser.add_argument("--min-person-height", default=40, type=int)
    parser.add_argument("--min-area-ratio", default=0.35, type=float)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Missing data.yaml: {args.data}")
        return 1
    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}")
            return 1
        shutil.rmtree(args.output)

    config = load_yaml(args.data)
    root = resolve_root(args.data, config)
    names = parse_names(config["names"])
    person_ids = [class_id for class_id, name in names.items() if name == args.person_class]
    if not person_ids:
        print(f"[ERROR] No person class named {args.person_class}")
        return 1
    person_id = person_ids[0]
    ppe_ids = {class_id for class_id, name in names.items() if name in PPE_CLASS_NAMES}

    counts = {split: {"crops": 0, "labels": 0} for split in ("train", "val", "test")}
    for source_split in SPLITS:
        target_split = TARGET_SPLIT.get(source_split, source_split)
        image_dir = split_image_dir(root, config, source_split)
        if image_dir is None or not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            labels = read_labels(label_for_image(image_path))
            abs_boxes = [yolo_to_xyxy(label, width, height) for label in labels]
            person_boxes = [
                box for box in abs_boxes if box[0] == person_id and (box[4] - box[2]) >= args.min_person_height
            ]
            ppe_boxes = [box for box in abs_boxes if box[0] in ppe_ids]
            for person_index, person_box in enumerate(person_boxes, start=1):
                crop = expand_person_box(person_box, width, height, args.margin)
                if crop is None:
                    continue
                crop_lines = [
                    line
                    for box in ppe_boxes
                    if (line := remap_label_to_crop(box, crop, args.min_area_ratio)) is not None
                ]
                if not crop_lines:
                    continue
                x1, y1, x2, y2 = crop
                crop_image = image[y1:y2, x1:x2]
                output_stem = f"{image_path.stem}_person{person_index:02d}"
                output_image = args.output / "images" / target_split / f"{output_stem}.jpg"
                output_label = args.output / "labels" / target_split / f"{output_stem}.txt"
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_image), crop_image)
                output_label.write_text("\n".join(crop_lines) + "\n", encoding="utf-8")
                counts[target_split]["crops"] += 1
                counts[target_split]["labels"] += len(crop_lines)

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {class_id: names[class_id] for class_id in sorted(names)},
    }
    with (args.output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    print(f"[OK] Wrote {args.output.resolve()}")
    for split, split_counts in counts.items():
        print(f"[COUNT] {split}: crops={split_counts['crops']} labels={split_counts['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
