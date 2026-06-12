#!/usr/bin/env python3
"""Run ppe_worker_4.py detection/re-ID logic on local images only.

This script does not write to MySQL, S3, or Welspun. It enables DRY_RUN and
DEBUG before importing ppe_worker_4 so the worker uses local-safe settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PPE_MODEL = (
    "runs/detect/training/runs/"
    "ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt"
)
DEFAULT_PERSON_MODEL = (
    "runs/detect/training/runs/"
    "ppe_person_yolov8n_finetune_v1/weights/best.pt"
)
DEFAULT_FACE_MODEL = "test_face_detection/yolov8n-face.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ppe_worker_4 local image inference without DB/S3/Welspun writes"
    )
    parser.add_argument("--input", default="Images", type=Path, help="Image file or folder")
    parser.add_argument("--save-dir", default="local_ppe_worker_4_images", type=Path)
    parser.add_argument("--ppe-model", default=DEFAULT_PPE_MODEL, type=Path)
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL, type=Path)
    parser.add_argument("--face-model", default=DEFAULT_FACE_MODEL, type=Path)
    parser.add_argument("--image-size", default="1920")
    parser.add_argument("--person-image-size", default="1280")
    parser.add_argument("--person-confidence", default="0.33")
    parser.add_argument("--person-iou", default="0.60")
    parser.add_argument("--enable-tiled-person-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-crowd-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-crop-ppe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--person-min-box-width", default="12")
    parser.add_argument("--person-min-box-height", default="45")
    parser.add_argument("--person-min-aspect-ratio", default="1.10")
    parser.add_argument("--person-max-aspect-ratio", default="8.00")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    os.environ["DRY_RUN"] = "TRUE"
    os.environ["DEBUG"] = "TRUE" if args.debug else "FALSE"
    os.environ["WELSPUN_WEBHOOK_ENABLED"] = "false"
    os.environ["PPE_MODEL_PATH"] = str(args.ppe_model)
    os.environ["PERSON_MODEL_PATH"] = str(args.person_model)
    os.environ["FACE_MODEL_PATH"] = str(args.face_model)
    os.environ["IMAGE_SIZE"] = str(args.image_size)
    os.environ["PERSON_IMAGE_SIZE"] = str(args.person_image_size)
    os.environ["PERSON_CONFIDENCE"] = str(args.person_confidence)
    os.environ["PERSON_IOU"] = str(args.person_iou)
    os.environ["ENABLE_TILED_PERSON_DETECTION"] = "TRUE" if args.enable_tiled_person_detection else "FALSE"
    os.environ["ENABLE_CROWD_RECOVERY"] = "TRUE" if args.enable_crowd_recovery else "FALSE"
    os.environ["ENABLE_CROP_PPE"] = "TRUE" if args.enable_crop_ppe else "FALSE"
    os.environ["PERSON_MIN_BOX_WIDTH"] = str(args.person_min_box_width)
    os.environ["PERSON_MIN_BOX_HEIGHT"] = str(args.person_min_box_height)
    os.environ["PERSON_MIN_ASPECT_RATIO"] = str(args.person_min_aspect_ratio)
    os.environ["PERSON_MAX_ASPECT_RATIO"] = str(args.person_max_aspect_ratio)


def image_paths(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    if path.is_file():
        return [path] if path.suffix.lower() in suffixes else []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def draw_results(cv2, image, detections: list[dict[str, Any]], persons: dict[int, dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["box"]]
        is_clear = det.get("class_name") == "NO-Violation"
        color = (0, 180, 0) if is_clear else (0, 0, 255)
        pidx = det.get("person_index", -1)
        person_id = persons.get(pidx, {}).get("person_id")
        label = f"{det['class_name']} {float(det['confidence']):.2f}"
        if person_id is not None:
            label += f" pid={person_id}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def crop_box(image, box: tuple[float, float, float, float]):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def write_crop(cv2, crop, path: Path) -> str | None:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), crop):
        return None
    return str(path)


def save_crops(cv2, save_dir: Path, image_stem: str, frame, result: dict[str, Any]) -> dict[str, list[str]]:
    crop_root = save_dir / "crops" / safe_name(image_stem)
    saved: dict[str, list[str]] = {"persons": [], "detections": [], "reid": []}

    for pidx, box in enumerate(result.get("person_boxes", [])):
        crop = crop_box(frame, box)
        out = write_crop(cv2, crop, crop_root / "persons" / f"person_{pidx:03d}.jpg")
        if out:
            saved["persons"].append(out)

    for didx, det in enumerate(result.get("detections", [])):
        pidx = det.get("person_index", -1)
        cname = safe_name(det.get("class_name", "det"))
        crop = crop_box(frame, det["box"])
        out = write_crop(
            cv2,
            crop,
            crop_root / "detections" / f"det_{didx:03d}_{cname}_pidx_{pidx}.jpg",
        )
        if out:
            saved["detections"].append(out)

    for pidx, person in result.get("persons", {}).items():
        body_crop = person.get("body_crop")
        face_crop = person.get("face_crop")
        person_id = person.get("person_id")
        suffix = f"pidx_{pidx}_pid_{person_id if person_id is not None else 'none'}"

        out = write_crop(cv2, body_crop, crop_root / "reid" / f"body_{suffix}.jpg")
        if out:
            saved["reid"].append(out)

        out = write_crop(cv2, face_crop, crop_root / "reid" / f"face_{suffix}.jpg")
        if out:
            saved["reid"].append(out)

    return saved


def main() -> int:
    args = parse_args()
    configure_env(args)

    import cv2
    import ppe_worker_4 as worker

    for label, path in [
        ("PPE model", args.ppe_model),
        ("Person model", args.person_model),
        ("Face model", args.face_model),
        ("Input", args.input),
    ]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path.resolve()}")
            return 1

    paths = image_paths(args.input)
    if not paths:
        print(f"[ERROR] No images found under: {args.input.resolve()}")
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)

    worker.aws_creds = {}
    worker.s3_client = None
    worker.kvs_client = None
    worker.db_connection = None
    worker.id_store = worker.IdentityStore()
    worker.init_models()

    all_results: list[dict[str, Any]] = []

    print(f"[INFO] Images: {len(paths)}")
    print(f"[INFO] Save dir: {args.save_dir.resolve()}")

    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[SKIP] OpenCV could not read {path}")
            continue

        meta = [{
            "camera_id": "local",
            "frame": frame,
            "timestamp": worker.get_timestamp(),
            "local_path": str(path),
            "s3_url": None,
            "detections": [],
            "persons": {},
        }]

        result = worker.detect_and_reid(meta)[0]
        detections = result.get("detections", [])
        persons = result.get("persons", {})
        saved_crops = save_crops(cv2, args.save_dir, path.stem, frame, result)

        printable = []
        for det in detections:
            pidx = det.get("person_index", -1)
            person = persons.get(pidx, {})
            reid_person_id = person.get("person_id")
            printable.append({
                "class_name": det["class_name"],
                "confidence": round(float(det["confidence"]), 3),
                "box": [round(float(v), 1) for v in det["box"]],
                "person_index": pidx,
                "person_id": reid_person_id if reid_person_id is not None else pidx,
                "reid_person_id": reid_person_id,
            })

        annotated = frame.copy()
        draw_results(cv2, annotated, detections, persons)
        out_image = args.save_dir / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(out_image), annotated)

        record = {
            "image": str(path),
            "annotated_image": str(out_image),
            "crops": saved_crops,
            "detections": printable,
        }
        all_results.append(record)

        print(f"\n[IMAGE] {path.name}")
        print(json.dumps(record, indent=2))

    results_path = args.save_dir / "results.json"
    results_path.write_text(json.dumps(json_safe(all_results), indent=2))
    print(f"\n[OK] Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
