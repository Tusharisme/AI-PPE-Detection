#!/usr/bin/env python3
"""Launch the small PPE + boots YOLO fine-tune."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune best.pt with Safety Boots / NO-Safety Boots")
    parser.add_argument("--model", default="best.pt", type=Path)
    parser.add_argument("--data", default=Path("training/datasets/merged_safety_shoe_balanced/data.yaml"), type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=Path("training/runs"), type=Path)
    parser.add_argument("--name", default="ppe_safety_shoe_v1")
    parser.add_argument("--cache", default="False")
    parser.add_argument("--lr0", default=0.002, type=float, help="Initial learning rate")
    parser.add_argument("--lrf", default=0.01, type=float, help="Final learning-rate fraction")
    parser.add_argument("--optimizer", default="AdamW", help="YOLO optimizer name; use an explicit value so lr0 is honored")
    parser.add_argument("--freeze", default=None, type=int, help="Number of early model layers to freeze")
    parser.add_argument("--close-mosaic", default=10, type=int)
    parser.add_argument("--mosaic", default=1.0, type=float)
    parser.add_argument("--scale", default=0.5, type=float)
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    project_path = args.project.resolve()

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1
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
        print("[INFO] Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("[INFO] Then install Ultralytics with: pip install ultralytics")
        return 1

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Fix GPU/PyTorch before training.")
        return 1

    print(f"[GPU] count={torch.cuda.device_count()} active={torch.cuda.get_device_name(0)}")
    model = YOLO(str(model_path))
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
        freeze=args.freeze,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        scale=args.scale,
    )
    save_dir = Path(getattr(results, "save_dir", project_path / args.name))
    print(f"[OK] Training finished: {save_dir}")
    print(f"[OK] Best weights: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
