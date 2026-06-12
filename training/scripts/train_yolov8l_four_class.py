#!/usr/bin/env python3
"""Train a focused 4-class YOLO detector for production PPE violations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO on the 4-class PPE violation dataset")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="yolov8l.pt",
        help=(
            "Starting weights/model. Use yolov8l.pt for YOLOv8l, or pass an existing "
            "best.pt to fine-tune that exact architecture."
        ),
    )
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--imgsz", default=1920, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=5e-5, type=float)
    parser.add_argument("--lrf", default=0.05, type=float)
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="ppe_4class_yolov8l_cctv_repeat4_v1")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Dataset YAML not found: {args.data.resolve()}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERROR] ultralytics is not installed in this environment: {exc}")
        return 1

    print(f"[TRAIN] model={args.model}")
    print(f"[TRAIN] data={args.data.resolve()}")
    print("[TRAIN] classes: Person, NO-Hardhat, NO-Safety Vest, NO-Safety Boots")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        amp=True,
        deterministic=True,
        seed=0,
        close_mosaic=5,
        mosaic=0.25,
        scale=0.2,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = Path(args.project) / args.name
    print(f"[OK] Training finished: {save_dir}")
    print(f"[OK] Best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
