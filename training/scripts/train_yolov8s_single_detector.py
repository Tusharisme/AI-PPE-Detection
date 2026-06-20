#!/usr/bin/env python3
"""Train a YOLOv8s single-detector PPE model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8s on the single-detector PPE dataset")
    parser.add_argument(
        "--data",
        default=Path("training/datasets/ppe_v3_person_single_detector/data.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Starting YOLOv8s weights. Use yolov8s.yaml only for true random scratch.",
    )
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--imgsz", default=1920, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=0.001, type=float)
    parser.add_argument("--lrf", default=0.05, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--warmup-epochs", default=3.0, type=float)
    parser.add_argument("--patience", default=25, type=int)
    parser.add_argument("--project", default=Path("training/runs"), type=Path)
    parser.add_argument("--name", default="ppe_v3_person_yolov8s_single_v1")
    parser.add_argument("--cache", default="False")
    parser.add_argument("--mosaic", default=0.5, type=float)
    parser.add_argument("--close-mosaic", default=15, type=int)
    parser.add_argument("--scale", default=0.3, type=float)
    parser.add_argument("--degrees", default=3.0, type=float)
    parser.add_argument("--translate", default=0.1, type=float)
    parser.add_argument("--fliplr", default=0.5, type=float)
    parser.add_argument("--hsv-h", default=0.015, type=float)
    parser.add_argument("--hsv-s", default=0.5, type=float)
    parser.add_argument("--hsv-v", default=0.35, type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    project_path = args.project.resolve()
    if not data_path.exists():
        print(f"[ERROR] Dataset YAML not found: {data_path}")
        print("[INFO] Run training/scripts/build_ppe_v3_person_single_detector_dataset.py first.")
        return 1

    cache_root = Path("training/.cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((cache_root / "matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_root / "ultralytics").resolve()))

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERROR] Missing training dependency: {exc}")
        return 1

    if str(args.device).lower() not in {"cpu", "mps"} and not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Use --device cpu for a smoke test or install CUDA PyTorch.")
        return 1

    pretrained = str(args.model).endswith(".pt")
    print(f"[MODEL] {args.model} pretrained={pretrained}")
    print(f"[DATA] {data_path}")
    print("[CLASSES] Person, NO-Hardhat, NO-Safety Vest, NO-Safety Boots")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        project=str(project_path),
        name=args.name,
        cache=args.cache,
        pretrained=pretrained,
        amp=True,
        deterministic=True,
        seed=0,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        scale=args.scale,
        degrees=args.degrees,
        translate=args.translate,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )

    save_dir = Path(getattr(results, "save_dir", project_path / args.name))
    print(f"[OK] Training finished: {save_dir}")
    print(f"[OK] Best weights: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
