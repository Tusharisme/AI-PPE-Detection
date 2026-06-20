#!/usr/bin/env python3
"""Run ppe_worker_5 RF-DETR detection on local Images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


DEFAULT_MODEL = Path(
    "training/runs/ppe_v3_person_rfdetr_medium_1024_batch6_lr5e5_v2/checkpoint_best_ema.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ppe_worker_5 RF-DETR full-frame plus crop-recovery logic on local images."
    )
    parser.add_argument("--input", default=Path("Images"), type=Path)
    parser.add_argument("--save-dir", default=Path("local_ppe_worker_5_rfdetr_images"), type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--rfdetr-size", default="medium", choices=["base", "small", "medium", "large"])
    parser.add_argument("--detection-confidence", default=0.05, type=float)
    parser.add_argument("--person-confidence", default=0.25, type=float)
    parser.add_argument("--person-image-size", default=1024, type=int)
    parser.add_argument("--crop-image-size", default=1024, type=int)
    parser.add_argument("--image-size", default=1024, type=int)
    parser.add_argument(
        "--class-confidences",
        default="NO-Hardhat=0.10,NO-Safety Vest=0.20,NO-Safety Boots=0.10",
    )
    parser.add_argument("--person-min-box-width", default=8.0, type=float)
    parser.add_argument("--person-min-box-height", default=20.0, type=float)
    parser.add_argument("--person-min-aspect-ratio", default=0.50, type=float)
    parser.add_argument("--person-max-aspect-ratio", default=10.0, type=float)
    parser.add_argument("--crop-margin", default=0.15, type=float)
    parser.add_argument("--owner-min-overlap", default=0.20, type=float)
    parser.add_argument("--crop-duplicate-iou", default=0.35, type=float)
    parser.add_argument("--close-person-conf-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-person-height-ratio", default=0.60, type=float)
    parser.add_argument("--close-person-area-ratio", default=0.22, type=float)
    parser.add_argument("--close-person-conf-boost", default=0.05, type=float)
    parser.add_argument("--close-person-conf-max", default=0.35, type=float)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    cache_root = Path("training/.cache")
    defaults = {
        "MPLCONFIGDIR": str((cache_root / "matplotlib").resolve()),
        "YOLO_CONFIG_DIR": str((cache_root / "ultralytics").resolve()),
        "DETECTOR_BACKEND": "rfdetr",
        "RFDETR_MODEL_SIZE": args.rfdetr_size,
        "RFDETR_OPTIMIZE_INFERENCE": "TRUE",
        "PPE_MODEL_PATH": str(args.model),
        "PERSON_MODEL_PATH": str(args.model),
        "SINGLE_MODEL_MODE": "TRUE",
        "DETECTION_CONFIDENCE": str(args.detection_confidence),
        "PERSON_CONFIDENCE": str(args.person_confidence),
        "PERSON_IMAGE_SIZE": str(args.person_image_size),
        "PPE_CROP_INFERENCE_SIZE": str(args.crop_image_size),
        "IMAGE_SIZE": str(args.image_size),
        "CLASS_CONFIDENCES": args.class_confidences,
        "PERSON_MIN_BOX_WIDTH": str(args.person_min_box_width),
        "PERSON_MIN_BOX_HEIGHT": str(args.person_min_box_height),
        "PERSON_MIN_ASPECT_RATIO": str(args.person_min_aspect_ratio),
        "PERSON_MAX_ASPECT_RATIO": str(args.person_max_aspect_ratio),
        "PPE_CROP_MARGIN": str(args.crop_margin),
        "PPE_OWNER_MIN_OVERLAP": str(args.owner_min_overlap),
        "PPE_CROP_DUPLICATE_IOU": str(args.crop_duplicate_iou),
        "CLOSE_PERSON_CONF_ENABLED": str(args.close_person_conf_enabled).upper(),
        "CLOSE_PERSON_HEIGHT_RATIO": str(args.close_person_height_ratio),
        "CLOSE_PERSON_AREA_RATIO": str(args.close_person_area_ratio),
        "CLOSE_PERSON_CONF_BOOST": str(args.close_person_conf_boost),
        "CLOSE_PERSON_CONF_MAX": str(args.close_person_conf_max),
        "PERSON_FALLBACK_ENABLED": "FALSE",
        "ENABLE_TILED_PERSON_DETECTION": "FALSE",
        "ENABLE_CROWD_RECOVERY": "FALSE",
        "NO_VIOLATION_RECHECK_ENABLED": "FALSE",
        "ENABLE_BOOT_COLOR_CHECK": "FALSE",
        "DEBUG": "FALSE",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def crop_box(frame: Any, box: tuple[float, float, float, float]) -> Any | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def draw_label(cv2: Any, image: Any, label: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    y_top = max(0, y1 - text_h - baseline - 6)
    x2 = min(image.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(image, (x1, y_top), (x2, y_top + text_h + baseline + 6), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, y_top + text_h + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def save_crop(cv2: Any, frame: Any, box: tuple[float, float, float, float], path: Path) -> str | None:
    crop = crop_box(frame, box)
    if crop is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), crop):
        return None
    return str(path)


def main() -> int:
    args = parse_args()
    configure_env(args)

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model.resolve()}", flush=True)
        return 1
    paths = image_paths(args.input)
    if not paths:
        print(f"[ERROR] No images found under: {args.input.resolve()}", flush=True)
        return 1

    try:
        import cv2
        import torch
    except ImportError as exc:
        print(f"[ERROR] Missing runtime dependency: {exc}", flush=True)
        return 1

    if args.require_gpu and not torch.cuda.is_available():
        print("[ERROR] GPU is not visible to PyTorch. Check nvidia-smi/CUDA before running.", flush=True)
        return 1

    import ppe_worker_5 as worker

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[LOCAL] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    print(f"[LOCAL] Loading model: {args.model.resolve()}", flush=True)
    model = worker.RFDetrDetectorAdapter(str(args.model), worker.RFDETR_MODEL_SIZE, device=device)

    class_names = {int(k): str(v) for k, v in model.names.items()}
    person_class_ids = [
        cid for cid, name in class_names.items() if name.strip().lower() == "person"
    ]
    violation_class_ids = worker.build_violation_class_ids(class_names)
    worker.model_config = {
        "person_model": model,
        "person_class_names": class_names,
        "person_class_ids": person_class_ids,
        "ppe_model": model,
        "class_names": class_names,
        "violation_class_ids": violation_class_ids,
        "ppe_person_class_ids": worker.build_ppe_person_class_ids(class_names),
        "boot_class_ids": [cid for cid, name in class_names.items() if worker.is_boot_class(name)],
        "detector_backend": "rfdetr",
        "device": device,
    }
    print(
        f"[LOCAL] classes={class_names} person_ids={person_class_ids} "
        f"violation_ids={violation_class_ids}",
        flush=True,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    crop_root = args.save_dir / "crops"
    palette = {
        "Person": (255, 180, 0),
        "NO-Hardhat": (0, 0, 255),
        "NO-Safety Vest": (0, 80, 255),
        "NO-Safety Boots": (0, 0, 255),
    }

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for index, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[SKIP] OpenCV could not read: {path}", flush=True)
            continue

        frame_h, frame_w = frame.shape[:2]
        person_boxes = worker.detect_persons(frame, device)

        ppe_imgsz = worker.optimal_imgsz(frame, worker.IMAGE_SIZE_MAX)
        ppe_res = worker.model_config["ppe_model"].predict(
            source=frame,
            imgsz=ppe_imgsz,
            conf=worker.MODEL_INFERENCE_CONFIDENCE,
            iou=worker.DETECTION_IOU,
            classes=worker.model_config["violation_class_ids"],
            device=device,
            verbose=False,
        )
        raw_dets = []
        if ppe_res and len(ppe_res[0].boxes) > 0:
            for box, cls_id, conf in zip(
                ppe_res[0].boxes.xyxy.cpu().numpy(),
                ppe_res[0].boxes.cls.cpu().numpy(),
                ppe_res[0].boxes.conf.cpu().numpy(),
            ):
                cname = worker.model_config["class_names"].get(int(cls_id), str(int(cls_id)))
                cval = float(conf)
                worker.log_raw_confidence(cname, cval)
                if not worker.is_allowed_violation_class(cname):
                    continue
                if cval < worker.get_class_confidence_threshold(cname):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                raw_dets.append({
                    "class_name": cname,
                    "confidence": cval,
                    "box": (x1, y1, x2, y2),
                    "source": "full_frame",
                })

        if person_boxes:
            for crop_det in worker.detect_ppe_on_crops(frame, person_boxes, device):
                if not any(
                    crop_det["class_name"] == det["class_name"]
                    and worker.box_iou(crop_det["box"], det["box"]) > 0.30
                    for det in raw_dets
                ):
                    raw_dets.append(crop_det)

        raw_dets = worker.assign_detections_to_persons(raw_dets, person_boxes, frame_w, frame_h)
        raw_dets = worker.filter_detections_by_adaptive_confidence(
            raw_dets,
            person_boxes,
            frame_w,
            frame_h,
        )
        raw_dets = worker.filter_no_hardhat_by_geometry(raw_dets, person_boxes, frame_w, frame_h)
        raw_dets = worker.suppress_boot_violations_by_color(frame, raw_dets, person_boxes)
        raw_dets = worker.dedupe_detections(raw_dets)

        annotated = frame.copy()
        for pidx, person_box in enumerate(person_boxes):
            x1, y1, x2, y2 = clamp_box(person_box, frame_w, frame_h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), palette["Person"], 2)
            cv2.putText(
                annotated,
                f"Person p={pidx}",
                (x1, max(22, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                palette["Person"],
                2,
                cv2.LINE_AA,
            )

        for det in raw_dets:
            x1, y1, x2, y2 = clamp_box(det["box"], frame_w, frame_h)
            color = palette.get(det["class_name"], (0, 0, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            draw_label(
                cv2,
                annotated,
                f'{det["class_name"]} {det["confidence"]:.2f} p={det.get("person_index", -1)}',
                x1,
                y1,
                color,
            )

        out_image = args.save_dir / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(out_image), annotated)

        saved_person_crops: list[str] = []
        saved_detection_crops: list[str] = []
        if args.save_crops:
            image_crop_root = crop_root / safe_name(path.stem)
            for pidx, person_box in enumerate(person_boxes):
                saved = save_crop(
                    cv2,
                    frame,
                    person_box,
                    image_crop_root / "persons" / f"person_{pidx:03d}.jpg",
                )
                if saved:
                    saved_person_crops.append(saved)
            for didx, det in enumerate(raw_dets):
                saved = save_crop(
                    cv2,
                    frame,
                    det["box"],
                    image_crop_root
                    / "detections"
                    / f'det_{didx:03d}_{safe_name(det["class_name"])}_pidx_{det.get("person_index", -1)}.jpg',
                )
                if saved:
                    saved_detection_crops.append(saved)

        det_records = []
        for det in raw_dets:
            counts[det["class_name"]] += 1
            det_records.append({
                "class_name": det["class_name"],
                "confidence": round(float(det["confidence"]), 4),
                "box": [round(float(v), 1) for v in det["box"]],
                "person_index": det.get("person_index"),
                "crop_owner_index": det.get("crop_owner_index"),
                "ownership_score": round(float(det.get("ownership_score", 0.0)), 4),
                "source": det.get("source", "person_crop"),
            })

        records.append({
            "image": str(path),
            "annotated_image": str(out_image),
            "image_size": [frame_w, frame_h],
            "person_boxes": [[round(float(v), 1) for v in box] for box in person_boxes],
            "detections": det_records,
            "crops": {
                "persons": saved_person_crops,
                "detections": saved_detection_crops,
            },
        })
        summary_rows.append({
            "image": path.name,
            "width": frame_w,
            "height": frame_h,
            "persons": len(person_boxes),
            "detections": len(raw_dets),
            "no_hardhat": sum(1 for det in raw_dets if det["class_name"] == "NO-Hardhat"),
            "no_safety_vest": sum(1 for det in raw_dets if det["class_name"] == "NO-Safety Vest"),
            "no_safety_boots": sum(1 for det in raw_dets if det["class_name"] == "NO-Safety Boots"),
            "annotated_image": str(out_image),
        })

        brief = ", ".join(
            f'{det["class_name"]}({det["confidence"]:.2f},p={det.get("person_index", -1)})'
            for det in raw_dets
        ) or "none"
        print(
            f"[IMAGE {index:02d}/{len(paths)}] {path.name}: "
            f"persons={len(person_boxes)} detections={brief}",
            flush=True,
        )

    results_path = args.save_dir / "results.json"
    summary_path = args.save_dir / "summary.csv"
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image",
            "width",
            "height",
            "persons",
            "detections",
            "no_hardhat",
            "no_safety_vest",
            "no_safety_boots",
            "annotated_image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Wrote {results_path}", flush=True)
    print(f"[OK] Wrote {summary_path}", flush=True)
    worker.print_confidence_histogram()
    print(
        f"[SUMMARY] images={len(records)} "
        f"persons={sum(row['persons'] for row in summary_rows)} "
        f"detections={sum(row['detections'] for row in summary_rows)} "
        f"counts={dict(counts)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
