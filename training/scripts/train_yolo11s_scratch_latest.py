#!/usr/bin/env python3
"""Train YOLO11s from scratch on the latest PPE dataset only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11s from scratch on ppe_latest_dataset")
    parser.add_argument("--model", default="yolo11s.yaml", help="YOLO architecture YAML. Use .yaml for scratch.")
    parser.add_argument(
        "--data",
        default=Path("training/datasets/ppe_latest_yolo11s_scratch/data.yaml"),
        type=Path,
        help="Dataset YAML",
    )
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--imgsz", default=1920, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--patience", default=12, type=int, help="Early stopping patience")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=Path("training/runs"), type=Path)
    parser.add_argument("--name", default="ppe_latest_yolo11s_scratch_v1")
    parser.add_argument("--cache", default="False")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=0.001, type=float)
    parser.add_argument("--lrf", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--warmup-epochs", default=3.0, type=float)
    parser.add_argument("--close-mosaic", default=10, type=int)
    parser.add_argument("--mosaic", default=0.5, type=float)
    parser.add_argument("--scale", default=0.3, type=float)
    parser.add_argument("--degrees", default=3.0, type=float)
    parser.add_argument("--translate", default=0.1, type=float)
    parser.add_argument("--fliplr", default=0.5, type=float)
    args = parser.parse_args()

    data_path = args.data.resolve()
    project_path = args.project.resolve()
    if not data_path.exists():
        print(f"[ERROR] Dataset YAML not found: {data_path}")
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
        print("[INFO] Install PyTorch and Ultralytics in .train-venv before training.")
        return 1

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Fix GPU/PyTorch before training.")
        return 1

    print(f"[GPU] count={torch.cuda.device_count()} active={torch.cuda.get_device_name(0)}")
    print(f"[MODEL] Training from scratch architecture: {args.model}")
    print(f"[DATA] {data_path}")
    print(f"[EARLY_STOP] patience={args.patience}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cache=args.cache,
        workers=args.workers,
        device=args.device,
        project=str(project_path),
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        scale=args.scale,
        degrees=args.degrees,
        translate=args.translate,
        fliplr=args.fliplr,
        pretrained=False,
    )

    save_dir = Path(getattr(results, "save_dir", project_path / args.name))
    print(f"[OK] Training finished: {save_dir}")
    print(f"[OK] Best weights: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
