#!/usr/bin/env python3
"""
OfficeLens PPE Detection Worker.

Worker-only runtime:
- Reads selected cameras from OfficeLens_ppe_cameras
- Uses cam_id as the KVS stream name
- Captures one snapshot per camera per cycle
- Runs best.pt PPE detection
- Uploads only violation frames to S3
- Writes normalized frame and detection rows to MySQL
"""

import configparser
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import torch
from ultralytics import YOLO


DEBUG = os.environ.get("DEBUG", "FALSE").lower() == "true"
DRY_RUN = os.environ.get("DRY_RUN", "FALSE").lower() == "true"

PPE_CREDS_FILE = os.environ.get("PPE_CREDS_FILE", "ppe_creds.txt")
MODEL_PATH = os.environ.get("MODEL_PATH", "./best.pt")
FRAMES_DIR = os.environ.get("FRAMES_DIR", "frames")

PPE_SNAPSHOT_INTERVAL = int(os.environ.get("PPE_SNAPSHOT_INTERVAL", "10"))
CONFIG_REFRESH_INTERVAL = int(os.environ.get("CONFIG_REFRESH_INTERVAL", "60"))
CONNECTION_TIMEOUT = int(os.environ.get("CONNECTION_TIMEOUT", "10"))
CAPTURE_WORKERS = int(os.environ.get("CAPTURE_WORKERS", "10"))
UPLOAD_WORKERS = int(os.environ.get("UPLOAD_WORKERS", "10"))
DETECTION_BATCH_SIZE = int(os.environ.get("DETECTION_BATCH_SIZE", "8"))

DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.25"))
DETECTION_IOU = float(os.environ.get("DETECTION_IOU", "0.45"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "640"))

KVS_URL_CACHE_DURATION = 240
KVS_PLAYBACK_MODE = "LIVE"
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Asia/Kolkata"))

os.makedirs(FRAMES_DIR, exist_ok=True)

models: list[dict[str, Any]] = []
aws_creds: dict[str, Any] | None = None
s3_client = None
kvs_client = None
db_connection = None
camera_sources: dict[str, str] = {}
_kvs_url_cache: dict[str, tuple[str, float]] = {}
last_camera_refresh = 0.0
running = True


def debug(message: str) -> None:
    if DEBUG:
        print(message, flush=True)


def get_timestamp() -> datetime:
    return datetime.now(TIMEZONE)


def load_credentials() -> dict[str, Any]:
    if not os.path.exists(PPE_CREDS_FILE):
        raise FileNotFoundError(f"Credentials file not found: {PPE_CREDS_FILE}")

    config = configparser.ConfigParser()
    config.read(PPE_CREDS_FILE)

    if "AWS" not in config:
        raise ValueError("Missing [AWS] section in credentials file")
    if "DB" not in config:
        raise ValueError("Missing [DB] section in credentials file")

    creds = {
        "access_key": config["AWS"].get("aws_access_key_id"),
        "secret_key": config["AWS"].get("aws_secret_access_key"),
        "region_name": config["AWS"].get("region_name"),
        "bucket": config["AWS"].get("s3_bucket", "officelens-ppe"),
        "db_type": config["DB"].get("db_type", "mysql"),
        "db_host": config["DB"].get("db_host"),
        "db_port": config["DB"].getint("db_port", 3306),
        "db_user": config["DB"].get("db_user"),
        "db_password": config["DB"].get("db_password"),
        "db_name": config["DB"].get("db_name"),
        "cameras_table": config["DB"].get("cameras_table", "OfficeLens_cameras"),
        "ppe_cameras_table": config["DB"].get("ppe_cameras_table", "OfficeLens_ppe_cameras"),
        "ppe_frames_table": config["DB"].get("ppe_frames_table", "OfficeLens_ppe_frames"),
        "ppe_detections_table": config["DB"].get("ppe_detections_table", "OfficeLens_ppe_detections"),
    }

    required = [
        "access_key",
        "secret_key",
        "region_name",
        "bucket",
        "db_host",
        "db_user",
        "db_password",
        "db_name",
        "cameras_table",
        "ppe_cameras_table",
        "ppe_frames_table",
        "ppe_detections_table",
    ]
    missing = [key for key in required if not creds.get(key)]
    if missing:
        raise ValueError(f"Missing required credential values: {', '.join(missing)}")

    return creds


def init_s3_client(creds: dict[str, Any]):
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region_name"],
    )


def init_kvs_client(creds: dict[str, Any]):
    import boto3

    return boto3.client(
        "kinesisvideo",
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region_name"],
    )


def init_db_connection(creds: dict[str, Any]):
    import pymysql

    return pymysql.connect(
        host=creds["db_host"],
        port=creds.get("db_port", 3306),
        user=creds["db_user"],
        password=creds["db_password"],
        database=creds["db_name"],
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def reconnect_db() -> None:
    global db_connection

    if db_connection:
        try:
            db_connection.ping(reconnect=True)
            return
        except Exception:
            try:
                db_connection.close()
            except Exception:
                pass

    db_connection = init_db_connection(aws_creds)


def is_ppe_violation_class(class_name: str) -> bool:
    return "NO" in class_name.upper()


def init_models() -> None:
    global models

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file.resolve()}")

    print(f"[MODEL] Loading PPE model: {model_file}", flush=True)
    ppe_model = YOLO(str(model_file))
    ppe_class_names = {int(k): str(v) for k, v in ppe_model.names.items()}
    models = [
        {
            "name": model_file.stem,
            "kind": "ppe",
            "model": ppe_model,
            "class_names": ppe_class_names,
            "confidence": DETECTION_CONFIDENCE,
            "iou": DETECTION_IOU,
            "image_size": IMAGE_SIZE,
            "violation_filter": is_ppe_violation_class,
        }
    ]
    print(f"[MODEL] Loaded PPE model with {len(ppe_class_names)} classes: {ppe_class_names}", flush=True)


def check_kvs_stream_exists(stream_name: str) -> bool:
    try:
        from botocore.exceptions import ClientError

        kvs_client.describe_stream(StreamName=stream_name)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        print(f"[KVS] Error checking stream {stream_name}: {exc}", flush=True)
        return False
    except Exception as exc:
        print(f"[KVS] Unexpected error checking stream {stream_name}: {exc}", flush=True)
        return False


def get_hls_streaming_url(stream_name: str) -> str | None:
    cached = _kvs_url_cache.get(stream_name)
    if cached:
        url, timestamp = cached
        age = time.time() - timestamp
        if age < KVS_URL_CACHE_DURATION:
            debug(f"[KVS] Using cached HLS URL for stream {stream_name} age={age:.0f}s")
            return url

    try:
        import boto3

        endpoint_response = kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName="GET_HLS_STREAMING_SESSION_URL",
        )
        endpoint = endpoint_response["DataEndpoint"]

        archived_client = boto3.client(
            "kinesis-video-archived-media",
            endpoint_url=endpoint,
            aws_access_key_id=aws_creds["access_key"],
            aws_secret_access_key=aws_creds["secret_key"],
            region_name=aws_creds["region_name"],
        )
        hls_response = archived_client.get_hls_streaming_session_url(
            StreamName=stream_name,
            PlaybackMode=KVS_PLAYBACK_MODE,
            HLSFragmentSelector={"FragmentSelectorType": "SERVER_TIMESTAMP"},
            Expires=300,
        )
        hls_url = hls_response["HLSStreamingSessionURL"]
        _kvs_url_cache[stream_name] = (hls_url, time.time())
        return hls_url
    except Exception as exc:
        print(f"[KVS ERROR] Failed to get HLS URL for stream {stream_name}: {exc}", flush=True)
        return None


def load_configured_cameras() -> bool:
    global camera_sources, last_camera_refresh

    reconnect_db()
    cameras_table = aws_creds["cameras_table"]
    ppe_cameras_table = aws_creds["ppe_cameras_table"]

    try:
        with db_connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT c.id
                FROM {ppe_cameras_table} pc
                JOIN {cameras_table} c ON c.id = pc.cam_id
                ORDER BY c.id
                """
            )
            rows = cursor.fetchall()
    except Exception as exc:
        print(f"[DB ERROR] Failed to load PPE cameras: {exc}", flush=True)
        return False

    discovered = {str(row["id"]): str(row["id"]) for row in rows}
    if not discovered:
        print("[CAMERA] No cameras configured in PPE camera table", flush=True)
        camera_sources = {}
        return False

    print(f"[KVS] Validating {len(discovered)} configured streams...", flush=True)
    valid_sources = {}
    for cam_id, stream_name in discovered.items():
        if check_kvs_stream_exists(stream_name):
            valid_sources[cam_id] = stream_name
            debug(f"[KVS] Stream {stream_name} exists for camera {cam_id}")
        else:
            print(f"[KVS SKIP] Camera {cam_id}: stream '{stream_name}' does not exist", flush=True)

    camera_sources = valid_sources
    last_camera_refresh = time.time()
    print(f"[CAMERA] Loaded {len(camera_sources)} active PPE cameras", flush=True)
    return bool(camera_sources)


def refresh_configured_cameras_if_needed(force: bool = False) -> bool:
    if force or not camera_sources or time.time() - last_camera_refresh >= CONFIG_REFRESH_INTERVAL:
        return load_configured_cameras()
    return bool(camera_sources)


def capture_single_camera(cam_id: str, stream_name: str, timeout: int = CONNECTION_TIMEOUT):
    try:
        hls_url = get_hls_streaming_url(stream_name)
        if not hls_url:
            return cam_id, None, None, None

        cap = cv2.VideoCapture(hls_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        start = time.time()
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                timestamp = get_timestamp()
                filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{cam_id}_{uuid.uuid4().hex[:8]}.jpg"
                local_path = os.path.join(FRAMES_DIR, filename)
                cv2.imwrite(local_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cap.release()
                return cam_id, frame, timestamp, local_path
            time.sleep(0.1)

        cap.release()
        print(f"[WARN] Camera {cam_id}: timed out waiting for KVS frame", flush=True)
        return cam_id, None, None, None
    except Exception as exc:
        print(f"[ERROR] Camera {cam_id}: capture failed: {exc}", flush=True)
        return cam_id, None, None, None


def capture_all_cameras_parallel():
    print(f"[CAPTURE] Capturing {len(camera_sources)} cameras from KVS...", flush=True)
    start = time.time()
    captured = []

    with ThreadPoolExecutor(max_workers=CAPTURE_WORKERS) as executor:
        futures = {
            executor.submit(capture_single_camera, cam_id, stream_name): cam_id
            for cam_id, stream_name in camera_sources.items()
        }
        for future in as_completed(futures):
            cam_id, frame, timestamp, local_path = future.result()
            if frame is not None:
                captured.append(
                    {
                        "camera_id": cam_id,
                        "frame": frame,
                        "timestamp": timestamp,
                        "local_path": local_path,
                        "s3_url": None,
                        "detections": [],
                    }
                )

    elapsed = time.time() - start
    print(f"[CAPTURE] Captured {len(captured)}/{len(camera_sources)} frames in {elapsed:.2f}s", flush=True)
    return captured


def detect_ppe_violations(frame_metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not frame_metadata_list:
        return frame_metadata_list

    print(f"[DETECT] Running {len(models)} model(s) for {len(frame_metadata_list)} frames...", flush=True)
    start = time.time()
    total_violations = 0
    model_counts: dict[str, int] = {}

    device = 0 if torch.cuda.is_available() else "cpu"
    for batch_start in range(0, len(frame_metadata_list), DETECTION_BATCH_SIZE):
        batch = frame_metadata_list[batch_start : batch_start + DETECTION_BATCH_SIZE]
        for meta in batch:
            detections = []
            for model_config in models:
                results = model_config["model"].predict(
                    source=meta["frame"],
                    imgsz=model_config["image_size"],
                    conf=model_config["confidence"],
                    iou=model_config["iou"],
                    device=device,
                    verbose=False,
                )
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for box, cls_id, confidence in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.cls.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    ):
                        class_id = int(cls_id)
                        class_name = model_config["class_names"].get(class_id, str(class_id))
                        if not model_config["violation_filter"](class_name):
                            continue
                        x1, y1, x2, y2 = [float(value) for value in box]
                        detections.append(
                            {
                                "class_name": class_name,
                                "raw_class_name": class_name,
                                "confidence": float(confidence),
                                "box": (x1, y1, x2, y2),
                            }
                        )
                        model_counts[model_config["name"]] = model_counts.get(model_config["name"], 0) + 1

            meta["detections"] = detections
            if detections and (DRY_RUN or DEBUG):
                details = ", ".join(
                    f"{item['class_name']}({item['confidence']:.2f})"
                    for item in detections
                )
                print(f"[DETECT] Camera {meta['camera_id']} violations: {details}", flush=True)
            total_violations += len(detections)

    elapsed = time.time() - start
    print(f"[DETECT] Found {total_violations} PPE violations in {elapsed:.2f}s", flush=True)
    if model_counts:
        details = ", ".join(f"{name}={count}" for name, count in sorted(model_counts.items()))
        print(f"[DETECT] Violation breakdown: {details}", flush=True)
    return frame_metadata_list


def upload_frame_to_s3(local_path: str, camera_id: str, timestamp: datetime) -> str | None:
    s3_key = f"ppe/{camera_id}/{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    try:
        s3_client.upload_file(local_path, aws_creds["bucket"], s3_key)
        return f"https://{aws_creds['bucket']}.s3.{aws_creds['region_name']}.amazonaws.com/{s3_key}"
    except Exception as exc:
        print(f"[S3 ERROR] Failed to upload {local_path}: {exc}", flush=True)
        return None


def upload_violation_frames_parallel(frame_metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violation_frames = [meta for meta in frame_metadata_list if meta["detections"]]
    if not violation_frames:
        return frame_metadata_list

    if DRY_RUN:
        print(f"[DRY_RUN] Skipping S3 upload for {len(violation_frames)} violation frames", flush=True)
        return frame_metadata_list

    print(f"[UPLOAD] Uploading {len(violation_frames)} violation frames...", flush=True)
    start = time.time()

    def upload_one(meta: dict[str, Any]) -> dict[str, Any]:
        meta["s3_url"] = upload_frame_to_s3(meta["local_path"], meta["camera_id"], meta["timestamp"])
        return meta

    success = 0
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
        futures = {executor.submit(upload_one, meta): meta for meta in violation_frames}
        for future in as_completed(futures):
            updated = future.result()
            if updated["s3_url"]:
                success += 1

    elapsed = time.time() - start
    print(f"[UPLOAD] Uploaded {success}/{len(violation_frames)} frames in {elapsed:.2f}s", flush=True)
    return frame_metadata_list


def save_violations_to_db(frame_metadata_list: list[dict[str, Any]]) -> None:
    if DRY_RUN:
        violation_count = sum(len(meta["detections"]) for meta in frame_metadata_list if meta["detections"])
        print(f"[DRY_RUN] Skipping DB save for {violation_count} detections", flush=True)
        return

    frames_to_save = [meta for meta in frame_metadata_list if meta["detections"] and meta["s3_url"]]
    if not frames_to_save:
        print("[DB] No PPE violations to save", flush=True)
        return

    reconnect_db()
    frames_table = aws_creds["ppe_frames_table"]
    detections_table = aws_creds["ppe_detections_table"]

    try:
        with db_connection.cursor() as cursor:
            frame_rows = [
                (
                    int(meta["camera_id"]),
                    meta["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    meta["s3_url"],
                )
                for meta in frames_to_save
            ]
            cursor.executemany(
                f"""
                INSERT INTO {frames_table} (cam_id, timestamp, frame_url)
                VALUES (%s, %s, %s)
                """,
                frame_rows,
            )
            first_frame_id = cursor.lastrowid

            detection_rows = []
            for index, meta in enumerate(frames_to_save):
                frame_id = first_frame_id + index
                for detection in meta["detections"]:
                    x1, y1, x2, y2 = detection["box"]
                    detection_rows.append(
                        (
                            frame_id,
                            detection["class_name"],
                            detection["confidence"],
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )

            if detection_rows:
                cursor.executemany(
                    f"""
                    INSERT INTO {detections_table}
                    (frame_id, class_name, confidence, x1, y1, x2, y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    detection_rows,
                )

        db_connection.commit()
        print(f"[DB] Saved {len(frame_rows)} frames and {len(detection_rows)} detections", flush=True)
    except Exception as exc:
        db_connection.rollback()
        print(f"[DB ERROR] Failed to save PPE violations: {exc}", flush=True)
        import traceback

        traceback.print_exc()


def cleanup_local_frames(frame_metadata_list: list[dict[str, Any]]) -> None:
    for meta in frame_metadata_list:
        local_path = meta.get("local_path")
        if not local_path:
            continue
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[WARN] Failed to delete {local_path}: {exc}", flush=True)


def process_snapshot_set() -> None:
    set_start = time.time()
    print("\n" + "=" * 60, flush=True)
    print(f"[SET] Starting PPE snapshot at {get_timestamp().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    captured_frames = capture_all_cameras_parallel()
    if not captured_frames:
        print("[SET] No frames captured", flush=True)
        return

    frame_metadata = detect_ppe_violations(captured_frames)
    frame_metadata = upload_violation_frames_parallel(frame_metadata)
    save_violations_to_db(frame_metadata)
    cleanup_local_frames(frame_metadata)

    elapsed = time.time() - set_start
    print("=" * 60, flush=True)
    print(f"[SET] Complete in {elapsed:.2f}s", flush=True)
    print("=" * 60, flush=True)


def initialize_system() -> bool:
    global aws_creds, s3_client, kvs_client, db_connection

    print("=" * 60, flush=True)
    print("OfficeLens PPE Detection Worker", flush=True)
    print("=" * 60, flush=True)

    try:
        aws_creds = load_credentials()
        s3_client = init_s3_client(aws_creds)
        kvs_client = init_kvs_client(aws_creds)
        db_connection = init_db_connection(aws_creds)
        init_models()
    except Exception as exc:
        print(f"[ERROR] Initialization failed: {exc}", flush=True)
        return False

    if not load_configured_cameras():
        return False

    print("[INIT] System initialized successfully", flush=True)
    print(f"[INIT] Snapshot interval: {PPE_SNAPSHOT_INTERVAL}s", flush=True)
    print(f"[INIT] Capture workers: {CAPTURE_WORKERS}", flush=True)
    print(f"[INIT] Upload workers: {UPLOAD_WORKERS}", flush=True)
    print(f"[INIT] Detection confidence: {DETECTION_CONFIDENCE}", flush=True)
    print(f"[INIT] Detection IoU: {DETECTION_IOU}", flush=True)
    print(f"[INIT] Dry run: {DRY_RUN}", flush=True)
    return True


def main() -> int:
    global running

    if not initialize_system():
        return 1

    print("\n[RUNNING] PPE worker is active", flush=True)
    try:
        while running:
            cycle_start = time.time()
            try:
                refresh_configured_cameras_if_needed()
                if camera_sources:
                    process_snapshot_set()
                else:
                    print("[WAIT] No active PPE cameras configured", flush=True)
            except Exception as exc:
                print(f"[ERROR] Snapshot processing failed: {exc}", flush=True)
                import traceback

                traceback.print_exc()

            elapsed = time.time() - cycle_start
            wait_time = max(0, PPE_SNAPSHOT_INTERVAL - elapsed)
            if wait_time > 0:
                print(f"\n[WAIT] Next PPE snapshot in {wait_time:.0f}s\n", flush=True)
                time.sleep(wait_time)
            else:
                print(
                    f"\n[WARN] PPE snapshot took {elapsed:.0f}s, longer than interval {PPE_SNAPSHOT_INTERVAL}s\n",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...", flush=True)
        running = False
    finally:
        if db_connection:
            db_connection.close()

    print("[EXIT] PPE worker stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
