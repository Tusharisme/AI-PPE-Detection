#!/usr/bin/env python3
"""
Local KVS PPE model smoke test.

This script does not write to S3 or the production database. It reads one KVS
stream, runs one or two YOLO models locally, prints detections, and can save
annotated frames.
"""

import argparse
import configparser
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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


def draw_detections(frame, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        color = (0, 0, 255) if detection["is_violation"] else (0, 160, 0)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PPE models against a live KVS stream without DB/S3 writes")
    parser.add_argument("--creds", default="ppe_creds.txt", help="Path to PPE credentials INI file")
    parser.add_argument("--stream-name", required=True, help="KVS stream name; for PPE this is usually cam_id")
    parser.add_argument("--model", default="best.pt", help="YOLO model path")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for readable frames")
    parser.add_argument("--save-dir", default="", help="Optional directory for annotated frames and results.jsonl")
    parser.add_argument("--violations-only", action="store_true", help="Print/save only violation classes")
    args = parser.parse_args()

    model_paths = [("ppe", Path(args.model))]

    for _, model_path in model_paths:
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path.resolve()}")
            return 1

    save_dir = Path(args.save_dir) if args.save_dir else None
    results_file = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        results_file = open(save_dir / "results.jsonl", "w", encoding="utf-8")

    try:
        creds = load_aws_config(args.creds)
        print(f"[KVS] Connecting to stream {args.stream_name}")
        hls_url = get_hls_url(creds, args.stream_name)

        model_configs = []
        for model_name, model_path in model_paths:
            print(f"[MODEL] Loading {model_name}: {model_path}")
            model = YOLO(str(model_path))
            class_names = {int(k): str(v) for k, v in model.names.items()}
            model_configs.append(
                {
                    "name": model_name,
                    "model": model,
                    "class_names": class_names,
                }
            )
            print(f"[MODEL] {model_name} classes: {class_names}")

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[MODEL] Device: {device}")

        cap = cv2.VideoCapture(hls_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[ERROR] OpenCV could not open the KVS HLS stream")
            return 1

        processed = 0
        failures = 0
        start = time.time()
        while processed < args.frames and time.time() - start < args.timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                failures += 1
                time.sleep(0.1)
                continue

            infer_start = time.time()
            detections = []
            for model_config in model_configs:
                result = model_config["model"].predict(
                    source=frame,
                    imgsz=args.image_size,
                    conf=args.confidence,
                    iou=args.iou,
                    device=device,
                    verbose=False,
                )[0]
                detections.extend(
                    detection_to_dict(
                        box,
                        model_config["class_names"],
                        model_config["name"],
                    )
                    for box in result.boxes
                )
            inference_ms = (time.time() - infer_start) * 1000
            if args.violations_only:
                detections = [item for item in detections if item["is_violation"]]

            record = {
                "stream_name": args.stream_name,
                "frame_index": processed + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "inference_ms": round(inference_ms, 2),
                "detections": detections,
            }
            print(json.dumps(record, indent=2))
            if results_file:
                results_file.write(json.dumps(record) + "\n")

            if save_dir:
                annotated = frame.copy()
                draw_detections(annotated, detections)
                output_path = save_dir / f"frame_{processed + 1:04d}.jpg"
                cv2.imwrite(str(output_path), annotated)

            processed += 1

        cap.release()
        elapsed = time.time() - start
        print(f"[DONE] Processed {processed} frames in {elapsed:.2f}s; read failures={failures}")
        if processed == 0:
            return 1
    except Exception as exc:
        print(f"[ERROR] Local KVS test failed: {exc}")
        return 1
    finally:
        if results_file:
            results_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
