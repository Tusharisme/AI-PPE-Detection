#!/usr/bin/env python3
"""
Local KVS PPE + Face Detection Test.

This script extends the KVS PPE test to include face detection for violators.
It does not write to S3 or the production database. It reads one KVS stream,
runs person detection, PPE violation detection, and face detection for violators.
Face crops are saved with metadata mapping violations to crop images.

Pipeline:
1. Capture frame from KVS
2. Detect persons using person model
3. Detect PPE violations using PPE model
4. For persons with violations, detect and crop faces
5. Save face crops and metadata CSV
"""

import argparse
import configparser
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
from ultralytics import YOLO


def load_aws_config(creds_file: str) -> dict[str, str]:
    path = Path(creds_file)
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)
    if "AWS" not in config:
        raise ValueError("Missing [AWS] section in credentials file")

    aws = config["AWS"]
    values = {
        "access_key": aws.get("aws_access_key_id"),
        "secret_key": aws.get("aws_secret_access_key"),
        "region_name": aws.get("region_name"),
    }

    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing required AWS values: {', '.join(missing)}")

    return values


def get_hls_url(creds: dict[str, str], stream_name: str) -> str:
    import boto3

    kvs = boto3.client(
        "kinesisvideo",
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region_name"],
    )
    kvs.describe_stream(StreamName=stream_name)
    endpoint = kvs.get_data_endpoint(
        StreamName=stream_name,
        APIName="GET_HLS_STREAMING_SESSION_URL",
    )["DataEndpoint"]

    archived = boto3.client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region_name"],
    )
    return archived.get_hls_streaming_session_url(
        StreamName=stream_name,
        PlaybackMode="LIVE",
        HLSFragmentSelector={"FragmentSelectorType": "SERVER_TIMESTAMP"},
        Expires=300,
    )["HLSStreamingSessionURL"]


def is_ppe_violation_class(class_name: str) -> bool:
    return "NO" in class_name.upper()


def detection_to_dict(box, class_names: dict[int, str], model_name: str) -> dict:
    class_id = int(box.cls[0])
    class_name = class_names.get(class_id, str(class_id))
    return {
        "model": model_name,
        "class_id": class_id,
        "class_name": class_name,
        "is_violation": is_ppe_violation_class(class_name),
        "confidence": round(float(box.conf[0]), 6),
        "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
    }


def box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def box_center_inside(inner_box: Tuple[float, float, float, float], outer_box: Tuple[float, float, float, float]) -> bool:
    """Check if the center of inner_box is inside outer_box."""
    x1, y1, x2, y2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2


def find_person_for_violation(violation_box: Tuple[float, float, float, float], person_boxes: List[Tuple[float, float, float, float]]) -> int:
    """Find the best matching person for a PPE violation."""
    if not person_boxes:
        return -1

    # First, try to find persons that contain the violation center
    containing_persons = []
    for i, person_box in enumerate(person_boxes):
        if box_center_inside(violation_box, person_box):
            containing_persons.append((i, box_iou(violation_box, person_box)))

    if containing_persons:
        return max(containing_persons, key=lambda x: x[1])[0]

    # If no containing person, find the one with highest IoU
    best_idx = -1
    best_iou = 0.0
    for i, person_box in enumerate(person_boxes):
        iou = box_iou(violation_box, person_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_iou > 0.1 else -1


def clamp_box(
    box: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(frame_w - 1, int(round(x1)))),
        max(0, min(frame_h - 1, int(round(y1)))),
        max(0, min(frame_w, int(round(x2)))),
        max(0, min(frame_h, int(round(y2)))),
    )


def format_box(box: Tuple[float, float, float, float] | List[float]) -> str:
    return f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"


def crop_face_from_person(
    frame,
    person_box: Tuple[float, float, float, float],
    face_boxes: List[Tuple[float, float, float, float]],
) -> Tuple[Any, Tuple[float, float, float, float]]:
    """Find and crop the best face whose center lies inside a person box."""
    if not face_boxes:
        return None, None

    px1, py1, px2, py2 = person_box
    person_h = max(1.0, py2 - py1)
    frame_h, frame_w = frame.shape[:2]
    best_face = None
    best_score = -1.0

    for face_box in face_boxes:
        fx1, fy1, fx2, fy2 = face_box
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        if not (px1 <= face_cx <= px2 and py1 <= face_cy <= py2):
            continue

        face_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        upper_body_score = max(0.0, 1.0 - ((face_cy - py1) / person_h))
        score = face_area + (upper_body_score * 1000.0)

        if score > best_score:
            best_score = score
            best_face = face_box

    if best_face is None:
        return None, None

    x1, y1, x2, y2 = clamp_box(best_face, frame_w, frame_h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, None

    return face_crop, best_face


def person_face_search_region(
    person_box: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    height_ratio: float,
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    region = (
        px1 - (pw * margin_ratio),
        py1 - (ph * margin_ratio * 0.5),
        px2 + (pw * margin_ratio),
        py1 + (ph * height_ratio),
    )
    return clamp_box(region, frame_w, frame_h)


def detect_face_from_person_crop(
    frame,
    person_box: Tuple[float, float, float, float],
    face_model,
    device,
    image_size: int,
    confidence: float,
    iou: float,
    height_ratio: float,
    margin_ratio: float,
) -> Tuple[Any, Tuple[float, float, float, float]]:
    """Fallback face detection on a violator person's upper-body crop."""
    frame_h, frame_w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = person_face_search_region(
        person_box,
        frame_w,
        frame_h,
        height_ratio,
        margin_ratio,
    )
    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    region = frame[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return None, None

    results = face_model.predict(
        source=region,
        imgsz=image_size,
        conf=confidence,
        iou=iou,
        device=device,
        verbose=False,
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None, None

    best_box = None
    best_conf = -1.0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            lx1, ly1, lx2, ly2 = [float(x) for x in box.xyxy[0].tolist()]
            best_box = (lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1)
            best_conf = conf

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = clamp_box(best_box, frame_w, frame_h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    return face_crop, best_box


def draw_detections(frame, detections: list[dict], person_boxes: List = None, face_boxes: List = None) -> None:
    """Draw detection boxes on frame."""
    # Draw PPE violations
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = (0, 0, 255) if detection["is_violation"] else (0, 160, 0)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw person boxes
    if person_boxes:
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Person {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw face boxes
    if face_boxes:
        for box in face_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def main() -> int:
    # Get script directory and parent directory for proper path resolution
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Test PPE + Face detection against a live KVS stream")
    parser.add_argument("--creds", default=str(parent_dir / "ppe_creds.txt"), help="Path to PPE credentials INI file")
    parser.add_argument("--stream-name", required=True, help="KVS stream name")
    parser.add_argument("--person-model", default=str(parent_dir / "runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt"), help="Person detection YOLO model path (using PPE model which includes Person class)")
    parser.add_argument("--ppe-model", default=str(parent_dir / "runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt"), help="PPE detection YOLO model path")
    parser.add_argument("--face-model", default=str(script_dir / "yolov8n-face.pt"), help="Face detection YOLO model path")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--person-confidence", type=float, default=0.33, help="Person detection confidence")
    parser.add_argument("--face-confidence", type=float, default=0.25, help="Full-frame face detection confidence")
    parser.add_argument("--face-crop-confidence", type=float, default=0.15, help="Fallback face detection confidence on violator person crops")
    parser.add_argument("--face-image-size", type=int, default=1280, help="Full-frame face model inference image size")
    parser.add_argument("--face-crop-image-size", type=int, default=960, help="Person-crop face model inference image size")
    parser.add_argument("--face-region-height-ratio", type=float, default=0.65, help="Fraction of person height used for fallback face search")
    parser.add_argument("--face-region-margin", type=float, default=0.20, help="Side margin around person box for fallback face search")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=640, help="Person/PPE YOLO inference image size")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for readable frames")
    parser.add_argument("--save-dir", default=str(script_dir / "violations"), help="Directory for results and face crops")
    parser.add_argument("--violations-only", action="store_true", help="Process only frames with violations")
    args = parser.parse_args()

    # Validate model files
    model_paths = [
        ("person", Path(args.person_model)),
        ("ppe", Path(args.ppe_model)),
        ("face", Path(args.face_model))
    ]

    for model_name, model_path in model_paths:
        if not model_path.exists():
            print(f"[ERROR] {model_name} model not found: {model_path.resolve()}")
            return 1

    # Setup save directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = save_dir / "violators_face_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = save_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Setup CSV for metadata
    metadata_file = save_dir / "metadata.csv"
    csv_file = open(metadata_file, "w", newline="", encoding="utf-8")
    csv_fields = [
        "timestamp",
        "frame_id",
        "source_frame_id",
        "person_index",
        "frame_path",
        "source_frame_path",
        "person_box",
        "face_crop_path",
        "face_box",
        "face_detection_source",
        "violations",
        "violation_boxes",
        "violation_confidences",
        "person_count",
        "face_count",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    try:
        # Load AWS credentials and connect to stream
        creds = load_aws_config(args.creds)
        print(f"[KVS] Connecting to stream {args.stream_name}")
        hls_url = get_hls_url(creds, args.stream_name)

        # Load models
        print("[MODEL] Loading models...")
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[MODEL] Device: {device}")

        person_model = YOLO(str(args.person_model))
        ppe_model = YOLO(str(args.ppe_model))
        face_model = YOLO(str(args.face_model))

        person_classes = {int(k): str(v) for k, v in person_model.names.items()}
        ppe_classes = {int(k): str(v) for k, v in ppe_model.names.items()}
        face_classes = {int(k): str(v) for k, v in face_model.names.items()}

        print(f"[MODEL] Person classes: {person_classes}")
        print(f"[MODEL] PPE classes: {ppe_classes}")
        print(f"[MODEL] Face classes: {face_classes}")

        # Open video capture
        cap = cv2.VideoCapture(hls_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[ERROR] OpenCV could not open the KVS HLS stream")
            return 1

        processed = 0
        failures = 0
        total_violations = 0
        total_face_crops = 0
        start = time.time()

        while processed < args.frames and time.time() - start < args.timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                failures += 1
                time.sleep(0.1)
                continue

            frame_start = time.time()

            # Step 1: Detect persons
            person_results = person_model.predict(
                source=frame,
                imgsz=args.image_size,
                conf=args.person_confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            person_boxes = []
            person_detections = []
            if person_results.boxes is not None:
                for box in person_results.boxes:
                    class_id = int(box.cls[0])
                    class_name = person_classes.get(class_id, str(class_id))
                    # Look for "Person" class (case-insensitive)
                    if class_name.lower() == "person":
                        coords = [float(x) for x in box.xyxy[0].tolist()]
                        person_boxes.append(tuple(coords))
                        person_detections.append({
                            "model": "person",
                            "class_name": class_name,
                            "confidence": float(box.conf[0]),
                            "box": coords
                        })

            # Step 2: Detect PPE violations
            ppe_results = ppe_model.predict(
                source=frame,
                imgsz=args.image_size,
                conf=args.confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            ppe_detections = []
            if ppe_results.boxes is not None:
                for box in ppe_results.boxes:
                    detection = detection_to_dict(box, ppe_classes, "ppe")
                    if detection["is_violation"]:
                        ppe_detections.append(detection)

            # Filter to only frames with violations if requested
            if args.violations_only and not ppe_detections:
                continue

            # Step 3: Detect faces
            face_results = face_model.predict(
                source=frame,
                imgsz=args.face_image_size,
                conf=args.face_confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            face_boxes = []
            face_detections = []
            if face_results.boxes is not None:
                for box in face_results.boxes:
                    coords = [float(x) for x in box.xyxy[0].tolist()]
                    face_boxes.append(tuple(coords))
                    face_detections.append({
                        "model": "face",
                        "class_name": "face",
                        "confidence": float(box.conf[0]),
                        "box": coords
                    })

            total_violations += len(ppe_detections)
            inference_ms = (time.time() - frame_start) * 1000

            frame_id = f"frame_{processed + 1:04d}"
            timestamp = datetime.now().isoformat()

            annotated_frame_name = f"{frame_id}_annotated.jpg"
            full_frame_path = frames_dir / annotated_frame_name
            annotated_frame_path = str(full_frame_path)

            person_violations = {}
            for violation in ppe_detections:
                violation_box = tuple(violation["box"])
                person_idx = find_person_for_violation(violation_box, person_boxes)
                if person_idx >= 0:
                    if person_idx not in person_violations:
                        person_violations[person_idx] = []
                    person_violations[person_idx].append(violation)

            crops_this_frame = 0
            for person_idx, violations in person_violations.items():
                person_box = person_boxes[person_idx]
                face_crop, face_box = crop_face_from_person(frame, person_box, face_boxes)

                face_source = "full_frame"
                if face_crop is None:
                    face_crop, face_box = detect_face_from_person_crop(
                        frame,
                        person_box,
                        face_model,
                        device,
                        args.face_crop_image_size,
                        args.face_crop_confidence,
                        args.iou,
                        args.face_region_height_ratio,
                        args.face_region_margin,
                    )
                    if face_crop is not None:
                        face_source = "person_crop_fallback"
                        face_boxes.append(tuple(face_box))

                face_crop_path = ""
                face_box_str = ""
                if face_crop is not None:
                    person_id = f"person_{person_idx + 1:02d}"
                    crop_filename = f"{frame_id}_{person_id}_{face_source}_face.jpg"
                    crop_path = crops_dir / crop_filename
                    if cv2.imwrite(str(crop_path), face_crop):
                        face_crop_path = str(crop_path)
                        face_box_str = format_box(face_box)
                        total_face_crops += 1
                        crops_this_frame += 1
                    else:
                        print(f"[WARN] Failed to save face crop: {crop_path}")
                else:
                    face_source = "not_detected"
                    print(f"[WARN] No face crop matched for {frame_id} person_{person_idx + 1:02d}")

                csv_writer.writerow({
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "source_frame_id": frame_id,
                    "person_index": person_idx + 1,
                    "frame_path": annotated_frame_path,
                    "source_frame_path": annotated_frame_path,
                    "person_box": format_box(person_box),
                    "face_crop_path": face_crop_path,
                    "face_box": face_box_str,
                    "face_detection_source": face_source,
                    "violations": "; ".join(f"{v['class_name']}({v['confidence']:.2f})" for v in violations),
                    "violation_boxes": "; ".join(f"{v['class_name']}:{format_box(v['box'])}" for v in violations),
                    "violation_confidences": "; ".join(f"{v['class_name']}:{v['confidence']:.3f}" for v in violations),
                    "person_count": len(person_boxes),
                    "face_count": len(face_boxes),
                })
                csv_file.flush()

            annotated = frame.copy()
            draw_detections(annotated, ppe_detections, person_boxes, face_boxes)
            if not cv2.imwrite(str(full_frame_path), annotated):
                print(f"[WARN] Failed to save annotated frame: {full_frame_path}")

            print(f"[FRAME {processed + 1:04d}] Persons: {len(person_boxes)}, Violations: {len(ppe_detections)}, Faces: {len(face_boxes)}, Crops: {crops_this_frame}, Inference: {inference_ms:.0f}ms")
            if ppe_detections:
                violation_summary = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" for d in ppe_detections])
                print(f"  Violations: {violation_summary}")

            processed += 1

        cap.release()
        elapsed = time.time() - start
        print(f"\n[DONE] Processed {processed} frames in {elapsed:.2f}s")
        print(f"[STATS] Read failures: {failures}")
        print(f"[STATS] Total violations: {total_violations}")
        print(f"[STATS] Face crops saved: {total_face_crops}")
        print(f"[STATS] Results saved to: {save_dir}")
        print(f"[STATS] Metadata CSV: {metadata_file}")

        if processed == 0:
            return 1

    except Exception as exc:
        print(f"[ERROR] Face detection test failed: {exc}")
        return 1
    finally:
        csv_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())