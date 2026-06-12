#!/usr/bin/env python3
"""
Merge the original PPE YOLO dataset with a small boots/no-boots YOLO dataset.

The output class order intentionally preserves the current best.pt class IDs and
appends the two boot classes. Images are symlinked by default to avoid consuming
extra disk space.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


TARGET_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
    "Safety Boots",
    "NO-Safety Boots",
]
TARGET_ID_BY_NAME = {name: index for index, name in enumerate(TARGET_NAMES)}

CLASS_ALIASES = {
    "hardhat": "Hardhat",
    "helmet": "Hardhat",
    "mask": "Mask",
    "no-hardhat": "NO-Hardhat",
    "no hardhat": "NO-Hardhat",
    "no_helmet": "NO-Hardhat",
    "no-helmet": "NO-Hardhat",
    "no helmet": "NO-Hardhat",
    "no-mask": "NO-Mask",
    "no_mask": "NO-Mask",
    "no mask": "NO-Mask",
    "no-safety vest": "NO-Safety Vest",
    "no_safety_vest": "NO-Safety Vest",
    "no-vest": "NO-Safety Vest",
    "no_vest": "NO-Safety Vest",
    "no vest": "NO-Safety Vest",
    "person": "Person",
    "safety cone": "Safety Cone",
    "safety_cone": "Safety Cone",
    "cone": "Safety Cone",
    "safety vest": "Safety Vest",
    "safety_vest": "Safety Vest",
    "vest": "Safety Vest",
    "machinery": "machinery",
    "machine": "machinery",
    "vehicle": "vehicle",
    "car": "vehicle",
    "truck": "vehicle",
    "boot": "Safety Boots",
    "boots": "Safety Boots",
    "safety boot": "Safety Boots",
    "safety boots": "Safety Boots",
    "safety_boot": "Safety Boots",
    "safety_boots": "Safety Boots",
    "safety-shoe": "Safety Boots",
    "safety shoe": "Safety Boots",
    "safety_shoe": "Safety Boots",
    "safety-shoes": "Safety Boots",
    "safety shoes": "Safety Boots",
    "safety_shoes": "Safety Boots",
    "no boot": "NO-Safety Boots",
    "no boots": "NO-Safety Boots",
    "no_boot": "NO-Safety Boots",
    "no_boots": "NO-Safety Boots",
    "no-boot": "NO-Safety Boots",
    "no-boots": "NO-Safety Boots",
    "no safety boot": "NO-Safety Boots",
    "no safety boots": "NO-Safety Boots",
    "no_safety_boot": "NO-Safety Boots",
    "no_safety_boots": "NO-Safety Boots",
    "no-safety-boot": "NO-Safety Boots",
    "no-safety-boots": "NO-Safety Boots",
    "no safety shoe": "NO-Safety Boots",
    "no safety shoes": "NO-Safety Boots",
    "no_safety_shoe": "NO-Safety Boots",
    "no_safety_shoes": "NO-Safety Boots",
    "no-safety-shoe": "NO-Safety Boots",
    "no-safety-shoes": "NO-Safety Boots",
    "not safety shoe": "NO-Safety Boots",
    "not safety shoes": "NO-Safety Boots",
    "not_safety_shoe": "NO-Safety Boots",
    "not_safety_shoes": "NO-Safety Boots",
    "not-safety-shoe": "NO-Safety Boots",
    "not-safety-shoes": "NO-Safety Boots",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    names: dict[int, str]
    split_images: dict[str, list[Path]]


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def canonical_name(name: str) -> str | None:
    if name in TARGET_ID_BY_NAME:
        return name
    normalized = normalize_name(name)
    return CLASS_ALIASES.get(normalized) or CLASS_ALIASES.get(normalized.replace(" ", "-"))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ValueError("data.yaml must contain names as a list or dict")


def resolve_split_paths(dataset_root: Path, value) -> list[Path]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for item in values:
        path = Path(str(item))
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_dir():
            paths.extend(sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = dataset_root / candidate
                if candidate.suffix.lower() in IMAGE_SUFFIXES:
                    paths.append(candidate)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return sorted(set(paths))


def load_dataset_config(data_yaml: Path) -> DatasetConfig:
    config = load_yaml(data_yaml)
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    names = parse_names(config["names"])
    split_images = {
        split: resolve_split_paths(dataset_root, config.get(split))
        for split in SPLITS
    }
    return DatasetConfig(root=dataset_root, names=names, split_images=split_images)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def map_label_file(label_path: Path, class_map: dict[int, int]) -> list[str]:
    if not label_path.exists():
        return []
    mapped_lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            source_class_id = int(float(parts[0]))
        except ValueError:
            continue
        target_class_id = class_map.get(source_class_id)
        if target_class_id is None:
            continue
        mapped_lines.append(" ".join([str(target_class_id), *parts[1:]]))
    return mapped_lines


def make_link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def build_class_map(names: dict[int, str]) -> dict[int, int]:
    class_map = {}
    for source_id, source_name in names.items():
        target_name = canonical_name(source_name)
        if target_name is not None:
            class_map[source_id] = TARGET_ID_BY_NAME[target_name]
    return class_map


def filter_class_map(class_map: dict[int, int], target_names: set[str]) -> dict[int, int]:
    target_ids = {TARGET_ID_BY_NAME[name] for name in target_names}
    return {
        source_id: target_id
        for source_id, target_id in class_map.items()
        if target_id in target_ids
    }


def write_example(
    image_path: Path,
    output_root: Path,
    split: str,
    prefix: str,
    class_map: dict[int, int],
    copy_files: bool,
) -> bool:
    mapped_lines = map_label_file(label_path_for_image(image_path), class_map)
    if not mapped_lines:
        return False

    output_image = output_root / "images" / split / f"{prefix}_{image_path.name}"
    output_label = output_root / "labels" / split / f"{prefix}_{image_path.with_suffix('.txt').name}"
    make_link_or_copy(image_path, output_image, copy_files)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(mapped_lines) + "\n", encoding="utf-8")
    return True


def source_ids_for_target(class_map: dict[int, int], target_name: str) -> set[int]:
    target_id = TARGET_ID_BY_NAME[target_name]
    return {source_id for source_id, mapped_id in class_map.items() if mapped_id == target_id}


def image_has_source_class(image_path: Path, source_class_ids: set[int]) -> bool:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            source_class_id = int(float(parts[0]))
        except ValueError:
            continue
        if source_class_id in source_class_ids:
            return True
    return False


def sample_images(images: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or len(images) <= limit:
        return images
    random.Random(seed).shuffle(images)
    return sorted(images[:limit])


def sample_boot_images(images: list[Path], limit: int, class_map: dict[int, int], seed: int) -> list[Path]:
    if limit <= 0 or len(images) <= limit:
        return images

    no_boot_source_ids = source_ids_for_target(class_map, "NO-Safety Boots")
    boot_source_ids = source_ids_for_target(class_map, "Safety Boots")
    no_boot_images = [
        image_path for image_path in images if image_has_source_class(image_path, no_boot_source_ids)
    ]
    boot_images = [
        image_path for image_path in images
        if image_path not in no_boot_images and image_has_source_class(image_path, boot_source_ids)
    ]
    other_images = [
        image_path for image_path in images
        if image_path not in no_boot_images and image_path not in boot_images
    ]

    rng = random.Random(seed)
    rng.shuffle(no_boot_images)
    rng.shuffle(boot_images)
    rng.shuffle(other_images)

    selected = [*no_boot_images[:limit]]
    if len(selected) < limit:
        selected.extend(boot_images[: limit - len(selected)])
    if len(selected) < limit:
        selected.extend(other_images[: limit - len(selected)])
    return sorted(selected)


def write_data_yaml(output_root: Path) -> None:
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_NAMES)},
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge PPE and boots YOLO datasets")
    parser.add_argument("--ppe-data", required=True, type=Path, help="Original PPE data.yaml")
    parser.add_argument("--boots-data", required=True, nargs="+", type=Path, help="Boots dataset data.yaml file(s)")
    parser.add_argument("--output", default=Path("training/datasets/merged_small"), type=Path)
    parser.add_argument("--boots-train-limit", type=int, default=240)
    parser.add_argument("--boots-val-limit", type=int, default=40)
    parser.add_argument("--boots-test-limit", type=int, default=20)
    parser.add_argument("--boots-repeat", type=int, default=1, help="Repeat boots-source images in the training split")
    parser.add_argument(
        "--boots-only",
        action="store_true",
        help="Use only Safety Boots / NO-Safety Boots labels from the boots dataset",
    )
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ppe = load_dataset_config(args.ppe_data)
    boots_configs = [load_dataset_config(path) for path in args.boots_data]
    ppe_class_map = build_class_map(ppe.names)
    boots_class_maps = []
    for boots_config in boots_configs:
        class_map = build_class_map(boots_config.names)
        if args.boots_only:
            class_map = filter_class_map(class_map, {"Safety Boots", "NO-Safety Boots"})
        boots_class_maps.append(class_map)

    mapped_boot_targets = {
        TARGET_NAMES[target_id]
        for class_map in boots_class_maps
        for target_id in class_map.values()
    }
    missing_boot_targets = {"Safety Boots", "NO-Safety Boots"} - mapped_boot_targets
    if missing_boot_targets:
        print(f"[ERROR] Boots dataset did not map required classes: {sorted(missing_boot_targets)}")
        for index, boots_config in enumerate(boots_configs, start=1):
            print(f"[INFO] Boots source {index} classes: {boots_config.names}")
        return 1

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in SPLITS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: {"ppe": 0, "boots": 0} for split in SPLITS}
    for split in SPLITS:
        for image_path in ppe.split_images[split]:
            if write_example(image_path, args.output, split, "ppe", ppe_class_map, args.copy):
                counts[split]["ppe"] += 1

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
                    prefix_parts = ["boots", f"d{dataset_index}"]
                    if repeat_count > 1:
                        prefix_parts.append(f"r{repeat_index + 1}")
                    prefix = "_".join(prefix_parts)
                    if write_example(image_path, args.output, split, prefix, boots_class_map, args.copy):
                        counts[split]["boots"] += 1

    write_data_yaml(args.output)
    print(f"[OK] Wrote merged dataset: {args.output.resolve()}")
    for split in SPLITS:
        print(f"[COUNT] {split}: ppe={counts[split]['ppe']} boots={counts[split]['boots']}")
    print(f"[OK] Class map: {TARGET_NAMES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
