#!/usr/bin/env python3
"""Build a YOLO person-only dataset from the merged 4-class PPE dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_data_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    required = ["path", "train", "val"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing required keys in {path}: {', '.join(missing)}")
    return values


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def person_only_lines(label_path: Path, source_person_class_id: int) -> list[str]:
    if not label_path.exists():
        return []
    kept: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id == source_person_class_id:
            kept.append("0 " + " ".join(parts[1:]))
    return kept


def link_or_copy_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    source_person_class_id: int,
    include_empty: bool,
    link_mode: str,
) -> dict[str, int]:
    source_images_dir = source_root / "images" / split
    source_labels_dir = source_root / "labels" / split
    output_images_dir = output_root / "images" / split
    output_labels_dir = output_root / "labels" / split
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_label_files": 0,
        "source_images": 0,
        "written_images": 0,
        "written_labels": 0,
        "person_labels": 0,
        "skipped_empty": 0,
        "missing_images": 0,
    }

    label_paths = sorted(source_labels_dir.glob("*.txt"))
    stats["source_label_files"] = len(label_paths)
    stats["source_images"] = len([path for path in source_images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])

    for label_path in label_paths:
        image_path = find_image_for_label(source_images_dir, label_path.stem)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        lines = person_only_lines(label_path, source_person_class_id)
        if not lines and not include_empty:
            stats["skipped_empty"] += 1
            continue

        output_image = output_images_dir / image_path.name
        output_label = output_labels_dir / f"{image_path.stem}.txt"
        link_or_copy_image(image_path, output_image, link_mode)
        output_label.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        stats["written_images"] += 1
        stats["written_labels"] += 1
        stats["person_labels"] += len(lines)

    return stats


def write_data_yaml(output_root: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: person",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a person-only YOLO dataset from the 4-class PPE dataset")
    parser.add_argument(
        "--source-data",
        default=Path("training/datasets/ppe2_archive_4class_from_best/data.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("training/datasets/ppe2_archive_person_only"),
        type=Path,
    )
    parser.add_argument("--source-person-class-id", default=0, type=int)
    parser.add_argument("--include-empty", action="store_true", help="Keep images with no person labels as negatives")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source_data.exists():
        print(f"[ERROR] Source data YAML not found: {args.source_data.resolve()}")
        return 1
    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output already exists. Pass --overwrite to replace: {args.output.resolve()}")
            return 1
        shutil.rmtree(args.output)

    source_config = parse_data_yaml(args.source_data)
    source_root = Path(source_config["path"]).expanduser()
    if not source_root.is_absolute():
        source_root = (args.source_data.parent / source_root).resolve()
    if not source_root.exists():
        print(f"[ERROR] Source dataset root not found: {source_root}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_data": str(args.source_data),
        "source_root": str(source_root),
        "output": str(args.output),
        "source_person_class_id": args.source_person_class_id,
        "include_empty": args.include_empty,
        "link_mode": args.link_mode,
        "splits": {},
    }

    for split in SPLITS:
        summary["splits"][split] = build_split(
            source_root,
            args.output,
            split,
            args.source_person_class_id,
            args.include_empty,
            args.link_mode,
        )

    write_data_yaml(args.output)
    summary_path = args.output / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Wrote dataset: {args.output / 'data.yaml'}")
    print(f"[OK] Wrote summary: {summary_path}")
    for split, stats in summary["splits"].items():
        print(
            f"[{split}] images={stats['written_images']} person_labels={stats['person_labels']} "
            f"skipped_empty={stats['skipped_empty']} missing_images={stats['missing_images']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
