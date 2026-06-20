#!/usr/bin/env python3
"""Train an RF-DETR Medium single-detector PPE model."""

from __future__ import annotations

import argparse
import csv
import json
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR Medium on the single-detector PPE dataset")
    parser.add_argument(
        "--dataset-dir",
        default=Path("training/datasets/ppe_v3_person_single_detector"),
        type=Path,
        help="Dataset directory containing train/valid/test COCO folders.",
    )
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--grad-accum-steps", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-encoder", default=1.5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--resolution", default=1024, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=Path("training/runs/ppe_v3_person_rfdetr_medium_v1"), type=Path)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping-patience", default=25, type=int)
    parser.add_argument("--early-stopping-min-delta", default=0.001, type=float)
    parser.add_argument("--checkpoint-interval", default=5, type=int)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results-csv", default=None, type=Path)
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra RF-DETR train arg as key=value.")
    return parser.parse_args()


def parse_extra_args(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-arg {item!r}; expected key=value")
        key, raw_value = item.split("=", 1)
        value: Any = raw_value
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        parsed[key.replace("-", "_")] = value
    return parsed


def load_medium_model():
    try:
        from rfdetr import RFDETRMedium
        return RFDETRMedium()
    except ImportError as exc:
        raise ImportError(
            "RF-DETR could not be imported. Install compatible training dependencies in .train-venv, "
            'for example: pip install "rfdetr[train]" tensorboard "numpy<2". '
            f"Original import error: {exc}"
        ) from exc
    except Exception:
        from rfdetr import RFDETRBase
        print("[WARN] RFDETRMedium was unavailable; falling back to RFDETRBase.")
        return RFDETRBase()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def export_tensorboard_scalars_to_csv(output_dir: Path, results_csv: Path) -> bool:
    """Mirror RF-DETR TensorBoard scalar logs to a flat CSV file."""
    event_files = sorted(output_dir.rglob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        print(f"[WARN] No TensorBoard event files found under {output_dir}; results.csv was not written.")
        return False

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print(
            "[WARN] TensorBoard is not installed, so scalar logs could not be converted to CSV. "
            'Install with: pip install "rfdetr[train]" tensorboard'
        )
        return False

    rows_by_step: dict[int, dict[str, Any]] = {}
    wall_time_by_step: dict[int, float] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            print(f"[WARN] Could not read TensorBoard file {event_file}: {exc}")
            continue
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                row = rows_by_step.setdefault(event.step, {"epoch": event.step})
                row[tag] = event.value
                wall_time_by_step[event.step] = max(wall_time_by_step.get(event.step, 0.0), event.wall_time)

    if not rows_by_step:
        print(f"[WARN] TensorBoard files had no scalar events; results.csv was not written.")
        return False

    for step, wall_time in wall_time_by_step.items():
        rows_by_step[step]["wall_time"] = wall_time

    fieldnames = ["epoch", "wall_time"]
    metric_names = sorted({
        key
        for row in rows_by_step.values()
        for key in row
        if key not in {"epoch", "wall_time"}
    })
    fieldnames.extend(metric_names)

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(rows_by_step):
            writer.writerow(rows_by_step[step])

    print(f"[OK] Wrote {results_csv}")
    return True


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    results_csv = (args.results_csv or output_dir / "results.csv").resolve()
    for split in ("train", "valid", "test"):
        ann = dataset_dir / split / "_annotations.coco.json"
        if not ann.exists():
            print(f"[ERROR] Missing COCO annotations: {ann}")
            print("[INFO] Run training/scripts/build_ppe_v3_person_single_detector_dataset.py first.")
            return 1

    os.environ.setdefault("RFDETR_CACHE_DIR", str((Path("training/.cache") / "rfdetr").resolve()))
    Path(os.environ["RFDETR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        print(f"[ERROR] Missing PyTorch: {exc}")
        return 1

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Use --device cpu for a smoke test or install CUDA PyTorch.")
        return 1

    try:
        extra_args = parse_extra_args(args.extra_arg)
        model = load_medium_model()
    except (ImportError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] RF-DETR Medium")
    print(f"[DATA] {dataset_dir}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[RESULTS_CSV] {results_csv}")
    print("[CLASSES] Person, NO-Hardhat, NO-Safety Vest, NO-Safety Boots")

    write_json(output_dir / "run_args.json", vars(args))

    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "resolution": args.resolution,
        "output_dir": str(output_dir),
        "num_workers": args.num_workers,
        "device": args.device,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "checkpoint_interval": args.checkpoint_interval,
        "tensorboard": args.tensorboard,
        "run_test": args.run_test,
        "class_names": ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"],
        "notes": {
            "dataset": str(dataset_dir),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "classes": ["Person", "NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"],
            "source": "ppe_v3_person_single_detector",
        },
        **extra_args,
    }

    signature = inspect.signature(model.train)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs:
        supported = set(signature.parameters)
        dropped = sorted(set(train_kwargs) - supported)
        if dropped:
            print(f"[WARN] Installed RF-DETR train API does not expose: {', '.join(dropped)}")
        train_kwargs = {key: value for key, value in train_kwargs.items() if key in supported}

    write_json(output_dir / "train_kwargs.json", train_kwargs)

    try:
        model.train(**train_kwargs)
    finally:
        if args.tensorboard:
            export_tensorboard_scalars_to_csv(output_dir, results_csv)

    print(f"[OK] Training finished: {output_dir}")
    return 0 if results_csv.exists() else 2


if __name__ == "__main__":
    sys.exit(main())
