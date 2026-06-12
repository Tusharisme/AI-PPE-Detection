#!/usr/bin/env python3
"""Create a fuller PPE + boots dataset using teacher pseudo-labels.

The output keeps the original best.pt class order and appends boot classes:
10 Safety Boots, 11 NO-Safety Boots.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from prepare_ppe_boots_dataset import (
    SPLITS,
    TARGET_ID_BY_NAME,
    TARGET_NAMES,
    build_class_map,
    filter_class_map,
    label_path_for_image,
    load_dataset_config,
    make_link_or_copy,
    map_label_file,
    sample_boot_images,
    write_data_yaml,
)


OLD_PPE_IDS = set(range(10))
BOOT_IDS = {TARGET_ID_BY_NAME["Safety Boots"], TARGET_ID_BY_NAME["NO-Safety Boots"]}


def yolo_to_xyxy(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(value) for value in parts[1:5])
    except ValueError:
        return None
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2
    return class_id, (x1, y1, x2, y2)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def add_without_duplicates(existing_lines: list[str], new_lines: list[str], duplicate_iou: float) -> list[str]:
    parsed_existing = [parsed for line in existing_lines if (parsed := yolo_to_xyxy(line)) is not None]
    merged = list(existing_lines)
    for new_line in new_lines:
        parsed_new = yolo_to_xyxy(new_line)
        if parsed_new is None:
            continue
        new_class_id, new_box = parsed_new
        duplicate = False
        for existing_class_id, existing_box in parsed_existing:
            if existing_class_id == new_class_id and iou(existing_box, new_box) >= duplicate_iou:
                duplicate = True
                break
        if not duplicate:
            merged.append(new_line)
            parsed_existing.append(parsed_new)
    return merged


def prediction_lines(model, image_path: Path, allowed_class_ids: set[int], conf: float, imgsz: int) -> list[str]:
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        device=0,
    )
    lines: list[str] = []
    if not results:
        return lines
    boxes = results[0].boxes
    if boxes is None:
        return lines
    xywhn = boxes.xywhn.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    for class_value, box in zip(classes, xywhn):
        class_id = int(class_value)
        if class_id not in allowed_class_ids:
            continue
        cx, cy, width, height = box[:4]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines


def write_labeled_example(
    image_path: Path,
    output_root: Path,
    split: str,
    prefix: str,
    base_lines: list[str],
    pseudo_lines: list[str],
    copy_files: bool,
    duplicate_iou: float,
) -> bool:
    lines = add_without_duplicates(base_lines, pseudo_lines, duplicate_iou)
    if not lines:
        return False

    output_image = output_root / "images" / split / f"{prefix}_{image_path.name}"
    output_label = output_root / "labels" / split / f"{prefix}_{image_path.with_suffix('.txt').name}"
    make_link_or_copy(image_path, output_image, copy_files)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a pseudo-fully-labeled PPE + boots YOLO dataset")
    parser.add_argument("--ppe-data", required=True, type=Path)
    parser.add_argument("--boots-data", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ppe-teacher", default=Path("best.pt"), type=Path, help="Teacher for original PPE classes")
    parser.add_argument("--boot-teacher", type=Path, help="Optional teacher for Safety Boots / NO-Safety Boots on PPE images")
    parser.add_argument("--ppe-teacher-conf", default=0.35, type=float)
    parser.add_argument("--boot-teacher-conf", default=0.35, type=float)
    parser.add_argument("--ppe-teacher-imgsz", default=1920, type=int)
    parser.add_argument("--boot-teacher-imgsz", default=1920, type=int)
    parser.add_argument("--boots-train-limit", default=0, type=int)
    parser.add_argument("--boots-val-limit", default=0, type=int)
    parser.add_argument("--boots-test-limit", default=0, type=int)
    parser.add_argument("--boots-repeat", default=1, type=int)
    parser.add_argument("--duplicate-iou", default=0.80, type=float)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERROR] Missing ultralytics: {exc}")
        return 1

    if not args.ppe_teacher.exists():
        print(f"[ERROR] PPE teacher not found: {args.ppe_teacher}")
        return 1
    if args.boot_teacher and not args.boot_teacher.exists():
        print(f"[ERROR] Boot teacher not found: {args.boot_teacher}")
        return 1

    ppe = load_dataset_config(args.ppe_data)
    boots_configs = [load_dataset_config(path) for path in args.boots_data]
    ppe_class_map = build_class_map(ppe.names)
    boots_class_maps = [
        filter_class_map(build_class_map(boots_config.names), {"Safety Boots", "NO-Safety Boots"})
        for boots_config in boots_configs
    ]

    mapped_boot_targets = {
        TARGET_NAMES[target_id]
        for class_map in boots_class_maps
        for target_id in class_map.values()
    }
    missing_boot_targets = {"Safety Boots", "NO-Safety Boots"} - mapped_boot_targets
    if missing_boot_targets:
        print(f"[ERROR] Boots datasets did not map required classes: {sorted(missing_boot_targets)}")
        for index, boots_config in enumerate(boots_configs, start=1):
            print(f"[INFO] Boots source {index} classes: {boots_config.names}")
        return 1

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in SPLITS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    ppe_teacher = YOLO(str(args.ppe_teacher.resolve()))
    boot_teacher = YOLO(str(args.boot_teacher.resolve())) if args.boot_teacher else None

    counts = {
        split: {
            "ppe_images": 0,
            "boot_images": 0,
            "ppe_pseudo_on_boots": 0,
            "boot_pseudo_on_ppe": 0,
        }
        for split in SPLITS
    }

    for split in SPLITS:
        for image_path in ppe.split_images[split]:
            base_lines = map_label_file(label_path_for_image(image_path), ppe_class_map)
            pseudo_lines: list[str] = []
            if boot_teacher is not None:
                pseudo_lines = prediction_lines(
                    boot_teacher,
                    image_path,
                    BOOT_IDS,
                    args.boot_teacher_conf,
                    args.boot_teacher_imgsz,
                )
            if write_labeled_example(
                image_path,
                args.output,
                split,
                "ppe",
                base_lines,
                pseudo_lines,
                args.copy,
                args.duplicate_iou,
            ):
                counts[split]["ppe_images"] += 1
                counts[split]["boot_pseudo_on_ppe"] += len(pseudo_lines)

        limit = {
            "train": args.boots_train_limit,
            "val": args.boots_val_limit,
            "test": args.boots_test_limit,
        }[split]
        repeat_count = max(1, args.boots_repeat if split == "train" else 1)
        for dataset_index, (boots_config, boots_class_map) in enumerate(zip(boots_configs, boots_class_maps), start=1):
            selected_boots = sample_boot_images(list(boots_config.split_images[split]), limit, boots_class_map, args.seed)
            for repeat_index in range(repeat_count):
                for image_path in selected_boots:
                    base_lines = map_label_file(label_path_for_image(image_path), boots_class_map)
                    pseudo_lines = prediction_lines(
                        ppe_teacher,
                        image_path,
                        OLD_PPE_IDS,
                        args.ppe_teacher_conf,
                        args.ppe_teacher_imgsz,
                    )
                    prefix_parts = ["boots", f"d{dataset_index}"]
                    if repeat_count > 1:
                        prefix_parts.append(f"r{repeat_index + 1}")
                    if write_labeled_example(
                        image_path,
                        args.output,
                        split,
                        "_".join(prefix_parts),
                        base_lines,
                        pseudo_lines,
                        args.copy,
                        args.duplicate_iou,
                    ):
                        counts[split]["boot_images"] += 1
                        counts[split]["ppe_pseudo_on_boots"] += len(pseudo_lines)

    write_data_yaml(args.output)
    print(f"[OK] Wrote pseudo-fully-labeled dataset: {args.output.resolve()}")
    for split in SPLITS:
        print(
            "[COUNT] "
            f"{split}: ppe_images={counts[split]['ppe_images']} "
            f"boot_images={counts[split]['boot_images']} "
            f"ppe_pseudo_on_boots={counts[split]['ppe_pseudo_on_boots']} "
            f"boot_pseudo_on_ppe={counts[split]['boot_pseudo_on_ppe']}"
        )
    print(f"[OK] Class map: {TARGET_NAMES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
