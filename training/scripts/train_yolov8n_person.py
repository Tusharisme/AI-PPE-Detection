#!/usr/bin/env python3
"""Fine-tune YOLOv8n as a person-only detector for the PPE gate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a person-only YOLOv8n detector for PPE gating")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=0.0005, type=float)
    parser.add_argument("--lrf", default=0.05, type=float)
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="ppe_person_yolov8n_finetune_v1")
    parser.add_argument("--freeze", default=8, type=int, help="Freeze first N layers for a light fine-tune")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Dataset YAML not found: {args.data.resolve()}")
        return 1

    cache_root = Path("training/.cache").resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "ultralytics").mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERROR] ultralytics is not installed in this environment: {exc}")
        return 1

    print(f"[TRAIN] model={args.model}")
    print(f"[TRAIN] data={args.data.resolve()}")
    print("[TRAIN] classes: person")
    print("[TRAIN] purpose: replace the COCO person gate in the PPE pipeline")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        project=args.project,
        name=args.name,
        pretrained=True,
        amp=True,
        deterministic=True,
        seed=0,
        freeze=args.freeze,
        close_mosaic=5,
        mosaic=0.15,
        scale=0.20,
        translate=0.08,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = Path(args.project) / args.name
    print(f"[OK] Training finished: {save_dir}")
    print(f"[OK] Best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
