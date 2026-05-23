#!/usr/bin/env python3
"""
Test a YOLO PPE model on local image files.

This does not write to S3 or the database. It prints detections as JSON and can
save annotated images for visual inspection.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def is_violation_class(class_name: str) -> bool:
    return "NO" in class_name.upper()


def draw_detections(image, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = (0, 0, 255) if detection["is_violation"] else (0, 160, 0)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a YOLO PPE model on local images")
    parser.add_argument("--image", required=True, help="Image path to test")
    parser.add_argument("--model", default="best.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--save-dir", default="local_image_test", help="Directory for annotated image and results.json")
    parser.add_argument("--violations-only", action="store_true", help="Print/save only classes containing NO")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path.resolve()}")
        return 1
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path.resolve()}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] OpenCV could not read image: {image_path.resolve()}")
        return 1

    print(f"[MODEL] Loading {model_path}")
    model = YOLO(str(model_path))
    class_names = {int(k): str(v) for k, v in model.names.items()}
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Classes: {class_names}")
    print(f"[MODEL] Device: {device}")

    result = model.predict(
        source=image,
        imgsz=args.image_size,
        conf=args.confidence,
        iou=args.iou,
        device=device,
        verbose=False,
    )[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        raw_class_name = class_names.get(class_id, str(class_id))
        detection = {
            "class_id": class_id,
            "class_name": raw_class_name,
            "raw_class_name": raw_class_name,
            "is_violation": is_violation_class(raw_class_name),
            "confidence": round(float(box.conf[0]), 6),
            "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
        }
        detections.append(detection)

    if args.violations_only:
        detections = [item for item in detections if item["is_violation"]]

    record = {
        "image": str(image_path),
        "model": str(model_path),
        "confidence": args.confidence,
        "image_size": args.image_size,
        "detections": detections,
    }
    print(json.dumps(record, indent=2))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            annotated = image.copy()
            draw_detections(annotated, detections)
            output_path = save_dir / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(output_path), annotated)
            print(f"[DONE] Wrote {output_path}")
            print(f"[DONE] Wrote {save_dir / 'results.json'}")
        except PermissionError as exc:
            print(f"[WARN] Could not write output files: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
