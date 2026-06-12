#!/usr/bin/env python3
"""Create a deterministic 3x YOLO train split with box-safe augmentations."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 3x augmented YOLO dataset")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", default=8791, type=int)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_label(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(stripped.split())
    return rows


def write_label(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(" ".join(row) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def flip_labels_horizontally(rows: list[list[str]]) -> list[list[str]]:
    flipped = []
    for row in rows:
        updated = row.copy()
        updated[1] = f"{1.0 - float(row[1]):.6f}"
        flipped.append(updated)
    return flipped


def adjust_color(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_variant(image: np.ndarray, labels: list[list[str]], variant: int, rng: np.random.Generator):
    if variant == 1:
        augmented = cv2.flip(image, 1)
        augmented = adjust_color(augmented, alpha=1.08, beta=8)
        return add_noise(augmented, rng, sigma=3.0), flip_labels_horizontally(labels)

    augmented = adjust_color(image, alpha=0.92, beta=-6)
    augmented = cv2.GaussianBlur(augmented, (3, 3), 0)
    return add_noise(augmented, rng, sigma=4.0), labels


def image_paths(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_split(input_dir: Path, output_dir: Path, split: str) -> tuple[int, int]:
    count_images = 0
    count_labels = 0
    source_images = input_dir / "images" / split
    source_labels = input_dir / "labels" / split
    if not source_images.exists():
        return 0, 0

    for image_path in image_paths(source_images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}")
        link_or_copy(image_path, output_dir / "images" / split / image_path.name)
        output_label = output_dir / "labels" / split / label_path.name
        output_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, output_label)
        count_images += 1
        count_labels += 1
    return count_images, count_labels


def augment_train(input_dir: Path, output_dir: Path, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    source_images = input_dir / "images" / "train"
    source_labels = input_dir / "labels" / "train"
    count_images = 0
    count_labels = 0

    for image_path in image_paths(source_images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}")

        labels = read_label(label_path)
        link_or_copy(image_path, output_dir / "images" / "train" / image_path.name)
        write_label(output_dir / "labels" / "train" / label_path.name, labels)
        count_images += 1
        count_labels += 1

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read {image_path}")

        for variant in (1, 2):
            augmented_image, augmented_labels = augment_variant(image, labels, variant, rng)
            suffix = f"_aug{variant}"
            output_image = output_dir / "images" / "train" / f"{image_path.stem}{suffix}{image_path.suffix}"
            output_label = output_dir / "labels" / "train" / f"{label_path.stem}{suffix}.txt"
            cv2.imwrite(str(output_image), augmented_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_label(output_label, augmented_labels)
            count_images += 1
            count_labels += 1

    return count_images, count_labels


def write_data_yaml(input_dir: Path, output_dir: Path) -> None:
    source_yaml = input_dir / "data.yaml"
    content = source_yaml.read_text(encoding="utf-8")
    lines = []
    for line in content.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {output_dir.resolve()}")
        else:
            lines.append(line)
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not (input_dir / "data.yaml").exists():
        print(f"[ERROR] Missing data.yaml in {input_dir}")
        return 1
    if output_dir.exists():
        if not args.overwrite:
            print(f"[ERROR] Output exists: {output_dir}. Use --overwrite.")
            return 1
        shutil.rmtree(output_dir)

    train_counts = augment_train(input_dir, output_dir, args.seed)
    val_counts = copy_split(input_dir, output_dir, "val")
    test_counts = copy_split(input_dir, output_dir, "test")
    write_data_yaml(input_dir, output_dir)

    print(f"[OK] Wrote {output_dir.resolve()}")
    print(f"[COUNT] train: images={train_counts[0]} labels={train_counts[1]}")
    print(f"[COUNT] val: images={val_counts[0]} labels={val_counts[1]}")
    if test_counts != (0, 0):
        print(f"[COUNT] test: images={test_counts[0]} labels={test_counts[1]}")
    print(f"[OK] data.yaml: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
