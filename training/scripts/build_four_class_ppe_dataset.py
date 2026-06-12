#!/usr/bin/env python3
"""Build a focused 4-class PPE violation dataset.

The output class order is:
0 Person
1 NO-Hardhat
2 NO-Safety Vest
3 NO-Safety Boots

This is meant for production violation detection, not general PPE detection.
It keeps archive/broad data for person + no-hardhat/no-vest context and
weights real CCTV data by repeating only its training split.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
TARGET_NAMES = ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"]
TARGET_ID_BY_NAME = {name: index for index, name in enumerate(TARGET_NAMES)}

ALIASES = {
    "person": "Person",
    "no hardhat": "NO-Hardhat",
    "no-hardhat": "NO-Hardhat",
    "no_hardhat": "NO-Hardhat",
    "no helmet": "NO-Hardhat",
    "no-helmet": "NO-Hardhat",
    "no_helmet": "NO-Hardhat",
    "no safety vest": "NO-Safety Vest",
    "no-safety vest": "NO-Safety Vest",
    "no-safety-vest": "NO-Safety Vest",
    "no_safety_vest": "NO-Safety Vest",
    "no vest": "NO-Safety Vest",
    "no-vest": "NO-Safety Vest",
    "no_vest": "NO-Safety Vest",
    "no safety boots": "NO-Safety Boots",
    "no-safety boots": "NO-Safety Boots",
    "no-safety-boots": "NO-Safety Boots",
    "no_safety_boots": "NO-Safety Boots",
    "no safety boot": "NO-Safety Boots",
    "no-safety-boot": "NO-Safety Boots",
    "no_safety_boot": "NO-Safety Boots",
    "no safety shoe": "NO-Safety Boots",
    "no-safety-shoe": "NO-Safety Boots",
    "not safety shoe": "NO-Safety Boots",
    "not-safety-shoe": "NO-Safety Boots",
}


@dataclass(frozen=True)
class Dataset:
    root: Path
    names: dict[int, str]
    images_by_split: dict[str, list[Path]]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_names(raw_names) -> dict[int, str]:
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


def candidate_split_paths(root: Path, value) -> list[Path]:
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
        # Roboflow exports sometimes write ../train/images even when data.yaml
        # lives beside train/. Prefer the in-folder path if the literal one is absent.
        text = str(item)
        if text.startswith("../"):
            candidates.append(root / text[3:])
    return candidates


def resolve_split(root: Path, value) -> list[Path]:
    for candidate in candidate_split_paths(root, value):
        if candidate.is_dir():
            return sorted(
                file for file in candidate.iterdir() if file.is_file() and file.suffix in IMAGE_SUFFIXES
            )
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return [candidate]
    return []


def load_dataset(data_yaml: Path) -> Dataset:
    config = load_yaml(data_yaml)
    if "path" in config:
        root = Path(config["path"])
        if not root.is_absolute():
            root = data_yaml.parent / root
    else:
        root = data_yaml.parent
    root = root.resolve()
    names = parse_names(config["names"])
    return Dataset(
        root=root,
        names=names,
        images_by_split={
            "train": resolve_split(root, config.get("train")),
            "val": resolve_split(root, config.get("val") or config.get("valid")),
            "test": resolve_split(root, config.get("test")),
        },
    )


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def build_class_map(names: dict[int, str]) -> dict[int, int]:
    class_map: dict[int, int] = {}
    for source_id, source_name in names.items():
        target_name = canonical_name(source_name)
        if target_name is not None:
            class_map[source_id] = TARGET_ID_BY_NAME[target_name]
    return class_map


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
        lines.append(" ".join([str(target_id), *parts[1:]]))
    return lines


def base_key(image_path: Path) -> str:
    stem = image_path.name
    for marker in ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", "_JPG.rf.", "_PNG.rf."):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return image_path.stem


def stable_bucket(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def split_camera_train(images: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    if val_fraction <= 0:
        return images, []
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(base_key(image), []).append(image)
    threshold = int(val_fraction * 10_000)
    train: list[Path] = []
    val: list[Path] = []
    for key, group in groups.items():
        if stable_bucket(key) < threshold:
            val.extend(group)
        else:
            train.extend(group)
    return sorted(train), sorted(val)


def limit_by_group(images: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(images) <= limit:
        return sorted(images)
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(base_key(image), []).append(image)
    selected: list[Path] = []
    for key in sorted(groups, key=stable_bucket):
        group = sorted(groups[key])
        if len(selected) + len(group) > limit and selected:
            continue
        selected.extend(group)
        if len(selected) >= limit:
            break
    return sorted(selected[:limit])


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


def write_example(
    image_path: Path,
    output: Path,
    split: str,
    prefix: str,
    class_map: dict[int, int],
    copy_files: bool,
    counts: dict[str, int],
) -> bool:
    lines = remap_label(label_for_image(image_path), class_map)
    image_name = f"{prefix}_{image_path.name}"
    label_name = Path(image_name).with_suffix(".txt").name
    link_or_copy(image_path, output / "images" / split / image_name, copy_files)
    label_target = output / "labels" / split / label_name
    label_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    for line in lines:
        class_id = int(line.split()[0])
        counts[TARGET_NAMES[class_id]] += 1
    return True


def write_data_yaml(output: Path) -> None:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_NAMES)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_source_only_dataset(args: argparse.Namespace) -> int:
    source = load_dataset(args.source_data)
    class_map = build_class_map(source.names)
    mapped_names = {TARGET_NAMES[target] for target in class_map.values()}
    missing = [name for name in TARGET_NAMES if name not in mapped_names]
    if missing:
        print(f"[ERROR] Source dataset is missing mapped classes: {', '.join(missing)}")
        print(f"[INFO] Source names: {source.names}")
        print(f"[INFO] Source class map: {class_map}")
        return 1

    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}. Use --overwrite.")
            return 1
        shutil.rmtree(args.output)

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: {name: 0 for name in TARGET_NAMES} for split in ("train", "val", "test")}
    image_counts = {split: 0 for split in ("train", "val", "test")}
    source_val = source.images_by_split["val"]
    source_test = source.images_by_split["test"]

    if not source_val and args.source_val_fraction > 0:
        source_train, source_val = split_camera_train(
            source.images_by_split["train"], args.source_val_fraction
        )
    else:
        source_train = source.images_by_split["train"]

    for repeat_index in range(1, max(1, args.source_repeat) + 1):
        for image_path in source_train:
            image_counts["train"] += write_example(
                image_path,
                args.output,
                "train",
                f"source_r{repeat_index}",
                class_map,
                args.copy,
                counts["train"],
            )

    for image_path in source_val:
        image_counts["val"] += write_example(
            image_path, args.output, "val", "source_val", class_map, args.copy, counts["val"]
        )

    for image_path in source_test:
        image_counts["test"] += write_example(
            image_path, args.output, "test", "source_test", class_map, args.copy, counts["test"]
        )

    write_data_yaml(args.output)
    print(f"[OK] Wrote {args.output.resolve()}")
    print(f"[OK] data.yaml: {args.output / 'data.yaml'}")
    for split in ("train", "val", "test"):
        print(f"[COUNT] {split}: images={image_counts[split]} labels={sum(counts[split].values())}")
        for name in TARGET_NAMES:
            print(f"  {name}: {counts[split][name]}")
    print(f"[INFO] Source map: {class_map}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 4-class archive + CCTV PPE dataset")
    parser.add_argument("--archive-data", type=Path)
    parser.add_argument("--camera-data", type=Path)
    parser.add_argument(
        "--source-data",
        type=Path,
        help="Build a 4-class dataset from one YOLO dataset instead of merging archive + camera data.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-repeat", default=4, type=int)
    parser.add_argument("--camera-val-fraction", default=0.15, type=float)
    parser.add_argument("--source-repeat", default=1, type=int)
    parser.add_argument(
        "--source-val-fraction",
        default=0.0,
        type=float,
        help="Only used with --source-data when the source has no validation split.",
    )
    parser.add_argument(
        "--archive-train-limit",
        default=4000,
        type=int,
        help="Limit archive images so CCTV remains dominant. Use 0 for no limit.",
    )
    parser.add_argument("--include-archive-valid-test-in-train", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source_data:
        return build_source_only_dataset(args)

    if not args.archive_data or not args.camera_data:
        print("[ERROR] Provide either --source-data or both --archive-data and --camera-data.")
        return 1

    archive = load_dataset(args.archive_data)
    camera = load_dataset(args.camera_data)
    archive_map = build_class_map(archive.names)
    camera_map = build_class_map(camera.names)

    if "NO-Safety Boots" not in {TARGET_NAMES[target] for target in camera_map.values()}:
        print("[ERROR] Camera dataset must contain NO-Safety Boots")
        return 1

    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {args.output}. Use --overwrite.")
            return 1
        shutil.rmtree(args.output)

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: {name: 0 for name in TARGET_NAMES} for split in ("train", "val", "test")}
    image_counts = {split: 0 for split in ("train", "val", "test")}

    archive_train = list(archive.images_by_split["train"])
    if args.include_archive_valid_test_in_train:
        archive_train.extend(archive.images_by_split["val"])
        archive_train.extend(archive.images_by_split["test"])
    archive_train = limit_by_group(archive_train, args.archive_train_limit)
    for image_path in archive_train:
        image_counts["train"] += write_example(
            image_path, args.output, "train", "archive", archive_map, args.copy, counts["train"]
        )

    camera_train, camera_val_from_train = split_camera_train(
        camera.images_by_split["train"], args.camera_val_fraction
    )
    camera_val = sorted([*camera.images_by_split["val"], *camera_val_from_train])

    for repeat_index in range(1, max(1, args.camera_repeat) + 1):
        for image_path in camera_train:
            image_counts["train"] += write_example(
                image_path,
                args.output,
                "train",
                f"camera_r{repeat_index}",
                camera_map,
                args.copy,
                counts["train"],
            )

    for image_path in camera_val:
        image_counts["val"] += write_example(
            image_path, args.output, "val", "camera_val", camera_map, args.copy, counts["val"]
        )

    for image_path in camera.images_by_split["test"]:
        image_counts["test"] += write_example(
            image_path, args.output, "test", "camera_test", camera_map, args.copy, counts["test"]
        )

    write_data_yaml(args.output)
    print(f"[OK] Wrote {args.output.resolve()}")
    print(f"[OK] data.yaml: {args.output / 'data.yaml'}")
    for split in ("train", "val", "test"):
        print(f"[COUNT] {split}: images={image_counts[split]} labels={sum(counts[split].values())}")
        for name in TARGET_NAMES:
            print(f"  {name}: {counts[split][name]}")
    print(f"[INFO] Archive map: {archive_map}")
    print(f"[INFO] Camera map: {camera_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
