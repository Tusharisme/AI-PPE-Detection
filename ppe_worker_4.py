#!/usr/bin/env python3
"""
OfficeLens PPE Detection + FaceNet Re-Identification Worker
Version: v4 (snapshot method, all fixes applied)

Fixes applied over v3:
  F1 — frame_quality_score()          : combined sharpness+contrast, rejects dark/flat frames
  F2 — detect_persons_tiled()         : SAHI-style tiled inference for small/distant workers
  F3 — detect_ppe_on_crops()          : PPE model runs on per-person expanded crops at 640px
  F4 — box_overlap_fraction() gate    : replaces center-inside with area-overlap fraction
  F5 — DETECTION_DEDUPE_IOU lowered   : 0.55 → 0.35, crop+full-frame dets no longer eat each other
  F6 — fixed boot crop imgsz          : constant 640 instead of native crop size
  F7 — perceptual frame dedup         : skips inference when camera hasn't changed visually

Architecture overview
─────────────────────
1.  Continuous loop — reads camera list from MySQL (ppe_cameras_table),
    captures best-quality frames from KVS HLS, runs inference.

2.  Detection pipeline:
      • Person detection  — finetuned YOLOv8 + TTA + higher-res fallback +
                            crowd-recovery re-detection + SAHI tiling (NEW)
      • PPE violation     — violation-class-only YOLOv8 on full frame +
                            per-person crop inference at 640px (NEW)
      • Spatial gating    — overlap-fraction gate replaces center-inside (NEW)
      • Boot crop         — optional lower-body crop sub-detection
      • Boot color check  — HSV-based black/brown suppression

3.  Re-identification pipeline (per violating person):
      a. Full-frame YOLO-face → pick face whose centre is inside person box
      b. Fallback: YOLO-face on upper-body crop
      c. MTCNN alignment (pose-corrected 160×160)
      d. Head-pose estimation from MTCNN landmarks (yaw proxy)
      e. FaceNet InceptionResnetV1 → 512-d L2-normalised embedding
      f. Pose-aware prototype bank (up to MAX_PROTOTYPES per person)
      g. Matching: cosine sim against rolling mean + ALL prototypes
      h. Quality gate before accepting as "better" embedding

4.  Persistence:
      S3:   ppe/frames/{cam_id}/{ts}_{uuid}.jpg
            ppe/persons/embeddings/person_{id:06d}.npy
            ppe/persons/crops/person_{id:06d}.jpg
      MySQL: persons, frames, detections tables

Environment variables (all optional — defaults shown):
  PPE_CREDS_FILE              ppe_creds.txt
  PERSON_MODEL_PATH           ./person_model.pt
  PPE_MODEL_PATH              ./best.pt
  FACE_MODEL_PATH             ./yolov8n-face.pt
  PPE_SNAPSHOT_INTERVAL       40
  CONFIG_REFRESH_INTERVAL     60
  CONNECTION_TIMEOUT          10
  CAPTURE_WORKERS             10
  UPLOAD_WORKERS              10
  DETECTION_BATCH_SIZE        8
  CAPTURE_FRAMES_PER_CAM      3
  CAPTURE_MIN_BLUR_SCORE      50.0
  DETECTION_CONFIDENCE        0.10
  PERSON_CONFIDENCE           0.33
  PERSON_IOU                  0.70
  PERSON_IMAGE_SIZE           1280
  PERSON_AUGMENT_MAX_SCALE    1920
  PERSON_FALLBACK_ENABLED     TRUE
  PERSON_FALLBACK_THRESHOLD   3
  PERSON_FALLBACK_CONF_DELTA  0.05
  ENABLE_CROWD_RECOVERY       TRUE
  CROWD_RECOVERY_IOU_CAP      0.85
  CROWD_RECOVERY_CONF         0.20
  CROWD_RECOVERY_NEW_BOX_IOU  0.30
  ENABLE_TILED_PERSON_DETECTION TRUE
  TILING_SIZE                 640
  TILING_OVERLAP              0.20
  ENABLE_CROP_PPE             TRUE
  PPE_CROP_INFERENCE_SIZE     640
  PPE_CROP_MARGIN             0.15
  PPE_OVERLAP_GATE_THRESHOLD  0.40
  DETECTION_IOU               0.45
  IMAGE_SIZE                  1920
  CLASS_CONFIDENCES           NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15
  PERSON_MATCH_REQUIRED       TRUE
  PERSON_MATCH_CONF_BYPASS    0.40
  PERSON_BOX_EXPAND_TOP       0.20
  PERSON_BOX_EXPAND_BOTTOM    0.20
  PERSON_BOX_EXPAND_SIDES     0.10
  ENABLE_BOOT_CROPS           FALSE
  BOOT_CONFIDENCE             0.10
  BOOT_IOU                    0.45
  BOOT_IMAGE_SIZE             1920
  BOOT_CROP_FOOT_RATIO        0.45
  BOOT_CROP_MARGIN            0.20
  BOOT_MIN_PERSON_HEIGHT      45
  ENABLE_BOOT_COLOR_CHECK     FALSE
  BOOT_COLOR_MIN_RATIO        0.08
  DETECTION_DEDUPE_IOU        0.35
  REID_THRESHOLD              0.68
  FRONTAL_YAW_THRESHOLD       55.0
  MAX_PROTOTYPES              12
  MIN_FACE_SIZE               48
  FACE_CONFIDENCE             0.25
  FACE_CROP_CONFIDENCE        0.15
  FACE_IMAGE_SIZE             1280
  FACE_CROP_IMAGE_SIZE        960
  FACE_REGION_HEIGHT_RATIO    0.65
  FACE_REGION_MARGIN          0.20
  KVS_URL_CACHE_DURATION      240
  TIMEZONE                    Asia/Kolkata
  DEBUG                       FALSE
  DRY_RUN                     FALSE
"""

from __future__ import annotations

import configparser
import io
import json
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import boto3
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import urllib3
from PIL import Image
from ultralytics import YOLO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
except ImportError:
    print("[ERROR] facenet-pytorch is not installed.  pip install facenet-pytorch")
    sys.exit(1)


# ───────────────────────────────────────────────────────────────────────────
# Helpers — config parsing
# ───────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_class_confidences(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        sep = "=" if "=" in item else (":" if ":" in item else None)
        if sep is None:
            continue
        cname, val = item.split(sep, 1)
        try:
            conf = float(val.strip())
        except ValueError:
            continue
        if 0.0 <= conf <= 1.0:
            out[cname.strip()] = conf
    return out


# ───────────────────────────────────────────────────────────────────────────
# Constants from environment
# ───────────────────────────────────────────────────────────────────────────

DEBUG   = _env_bool("DEBUG",   False)
DRY_RUN = _env_bool("DRY_RUN", False)

PPE_CREDS_FILE    = _env("PPE_CREDS_FILE",    "ppe_creds.txt")
PERSON_MODEL_PATH = _env("PERSON_MODEL_PATH", "../runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt")
PPE_MODEL_PATH    = _env("PPE_MODEL_PATH",    "../runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt")
FACE_MODEL_PATH   = _env("FACE_MODEL_PATH",   "./yolov8n-face.pt")

PPE_SNAPSHOT_INTERVAL   = _env_int("PPE_SNAPSHOT_INTERVAL",   40)
CONFIG_REFRESH_INTERVAL = _env_int("CONFIG_REFRESH_INTERVAL", 60)
CONNECTION_TIMEOUT      = _env_int("CONNECTION_TIMEOUT",      10)
CAPTURE_WORKERS         = _env_int("CAPTURE_WORKERS",         10)
UPLOAD_WORKERS          = _env_int("UPLOAD_WORKERS",          10)
DETECTION_BATCH_SIZE    = _env_int("DETECTION_BATCH_SIZE",    8)
CAPTURE_FRAMES_PER_CAM  = _env_int("CAPTURE_FRAMES_PER_CAM", 3)
CAPTURE_MIN_BLUR_SCORE  = _env_float("CAPTURE_MIN_BLUR_SCORE", 50.0)

WELSPUN_WEBHOOK_BASE_URL = _env("WELSPUN_WEBHOOK_BASE_URL", "https://welappsuat.welspun.com")
WELSPUN_WEBHOOK_AUTH_KEY = _env("WELSPUN_WEBHOOK_AUTH_KEY", "U2blpNYCc8cIdS2ZpNd7")
WELSPUN_WEBHOOK_ENABLED  = _env_bool("WELSPUN_WEBHOOK_ENABLED", False)
WELSPUN_PRESIGN_EXPIRY   = _env_int("WELSPUN_PRESIGN_EXPIRY", 604800)

DETECTION_CONFIDENCE = _env_float("DETECTION_CONFIDENCE", 0.10)
PERSON_CONFIDENCE    = _env_float("PERSON_CONFIDENCE",    0.33)
PERSON_IOU           = _env_float("PERSON_IOU",           0.70)
PERSON_IMAGE_SIZE_MAX        = _env_int("PERSON_IMAGE_SIZE",        1280)
PERSON_AUGMENT_MAX_SCALE     = _env_int("PERSON_AUGMENT_MAX_SCALE", 1920)
PERSON_FALLBACK_ENABLED      = _env_bool("PERSON_FALLBACK_ENABLED",      True)
PERSON_FALLBACK_THRESHOLD    = _env_int( "PERSON_FALLBACK_THRESHOLD",    3)
PERSON_FALLBACK_CONF_DELTA   = _env_float("PERSON_FALLBACK_CONF_DELTA",  0.05)
ENABLE_CROWD_RECOVERY        = _env_bool("ENABLE_CROWD_RECOVERY",        True)
CROWD_RECOVERY_IOU_CAP       = _env_float("CROWD_RECOVERY_IOU_CAP",      0.85)
CROWD_RECOVERY_CONF          = _env_float("CROWD_RECOVERY_CONF",         0.20)
CROWD_RECOVERY_NEW_BOX_IOU   = _env_float("CROWD_RECOVERY_NEW_BOX_IOU",  0.30)

# F2 — tiled person detection
ENABLE_TILED_PERSON_DETECTION = _env_bool("ENABLE_TILED_PERSON_DETECTION", True)
TILING_SIZE                   = _env_int("TILING_SIZE",    640)
TILING_OVERLAP                = _env_float("TILING_OVERLAP", 0.20)

DETECTION_IOU  = _env_float("DETECTION_IOU", 0.45)
IMAGE_SIZE_MAX = _env_int("IMAGE_SIZE", 1920)
CLASS_CONFIDENCE_THRESHOLDS = parse_class_confidences(
    _env("CLASS_CONFIDENCES",
         "NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15")
)
MODEL_INFERENCE_CONFIDENCE = min(
    [DETECTION_CONFIDENCE, *CLASS_CONFIDENCE_THRESHOLDS.values()]
    or [DETECTION_CONFIDENCE]
)

PERSON_MATCH_REQUIRED    = _env_bool("PERSON_MATCH_REQUIRED",    True)
PERSON_MATCH_CONF_BYPASS = _env_float("PERSON_MATCH_CONF_BYPASS", 0.40)
PERSON_BOX_EXPAND_TOP    = _env_float("PERSON_BOX_EXPAND_TOP",    0.20)
PERSON_BOX_EXPAND_BOTTOM = _env_float("PERSON_BOX_EXPAND_BOTTOM", 0.20)
PERSON_BOX_EXPAND_SIDES  = _env_float("PERSON_BOX_EXPAND_SIDES",  0.10)
PERSON_BOX_DEDUPE_IOU    = _env_float("PERSON_BOX_DEDUPE_IOU",    0.45)
PERSON_BOX_DEDUPE_COVERAGE = _env_float("PERSON_BOX_DEDUPE_COVERAGE", 0.80)
PERSON_MIN_BOX_WIDTH     = _env_float("PERSON_MIN_BOX_WIDTH",     12.0)
PERSON_MIN_BOX_HEIGHT    = _env_float("PERSON_MIN_BOX_HEIGHT",    45.0)
PERSON_MIN_ASPECT_RATIO  = _env_float("PERSON_MIN_ASPECT_RATIO",  1.10)
PERSON_MAX_ASPECT_RATIO  = _env_float("PERSON_MAX_ASPECT_RATIO",  8.00)
NO_VIOLATION_MIN_PERSON_HEIGHT = _env_int("NO_VIOLATION_MIN_PERSON_HEIGHT", 45)

# F3 — per-person crop PPE inference
ENABLE_CROP_PPE         = _env_bool("ENABLE_CROP_PPE",         True)
PPE_CROP_INFERENCE_SIZE = _env_int("PPE_CROP_INFERENCE_SIZE",   640)
PPE_CROP_MARGIN         = _env_float("PPE_CROP_MARGIN",         0.15)

# F4 — overlap-fraction spatial gate
PPE_OVERLAP_GATE_THRESHOLD = _env_float("PPE_OVERLAP_GATE_THRESHOLD", 0.40)

ENABLE_BOOT_CROPS      = _env_bool("ENABLE_BOOT_CROPS",      False)
BOOT_CONFIDENCE        = _env_float("BOOT_CONFIDENCE",        0.10)
BOOT_IOU               = _env_float("BOOT_IOU",               0.45)
BOOT_IMAGE_SIZE_MAX    = _env_int("BOOT_IMAGE_SIZE",           1920)
BOOT_CROP_FOOT_RATIO   = _env_float("BOOT_CROP_FOOT_RATIO",   0.45)
BOOT_CROP_MARGIN       = _env_float("BOOT_CROP_MARGIN",        0.20)
BOOT_MIN_PERSON_HEIGHT = _env_int("BOOT_MIN_PERSON_HEIGHT",    45)
ENABLE_BOOT_COLOR_CHECK = _env_bool("ENABLE_BOOT_COLOR_CHECK", False)
BOOT_COLOR_MIN_RATIO    = _env_float("BOOT_COLOR_MIN_RATIO",   0.08)

# F5 — lowered dedup IoU so crop+full-frame dets don't erase each other
DETECTION_DEDUPE_IOU = _env_float("DETECTION_DEDUPE_IOU", 0.35)

# Re-ID
REID_THRESHOLD         = _env_float("REID_THRESHOLD",         0.68)
FRONTAL_YAW_THRESHOLD  = _env_float("FRONTAL_YAW_THRESHOLD",  55.0)
MAX_PROTOTYPES         = _env_int("MAX_PROTOTYPES",            12)
MIN_FACE_SIZE          = _env_int("MIN_FACE_SIZE",             48)
FACE_CONFIDENCE        = _env_float("FACE_CONFIDENCE",         0.25)
FACE_CROP_CONFIDENCE   = _env_float("FACE_CROP_CONFIDENCE",    0.15)
FACE_IMAGE_SIZE        = _env_int("FACE_IMAGE_SIZE",           1280)
FACE_CROP_IMAGE_SIZE   = _env_int("FACE_CROP_IMAGE_SIZE",      960)
FACE_REGION_HEIGHT_RATIO = _env_float("FACE_REGION_HEIGHT_RATIO", 0.65)
FACE_REGION_MARGIN       = _env_float("FACE_REGION_MARGIN",       0.20)

KVS_URL_CACHE_DURATION = _env_int("KVS_URL_CACHE_DURATION", 240)
KVS_PLAYBACK_MODE      = "LIVE"
TIMEZONE = ZoneInfo(_env("TIMEZONE", "Asia/Kolkata"))

BOOT_CLASS_NAMES              = {"Safety Boots", "NO-Safety Boots"}
ALLOWED_VIOLATION_CLASS_NAMES = {"NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"}

_FACENET_INPUT_SIZE = 160

# S3 key prefixes
S3_FRAMES_PREFIX     = "ppe/frames"
S3_EMBEDDINGS_PREFIX = "ppe/persons/embeddings"
S3_CROPS_PREFIX      = "ppe/persons/crops"

_WELSPUN_PPE_URL = f"{WELSPUN_WEBHOOK_BASE_URL}/webhooks/report_ppe_violation"
_WELSPUN_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "x-webhook-auth": WELSPUN_WEBHOOK_AUTH_KEY,
}


# ───────────────────────────────────────────────────────────────────────────
# Global state
# ───────────────────────────────────────────────────────────────────────────

model_config:        Optional[Dict[str, Any]] = None
aws_creds:           Optional[Dict[str, Any]] = None
s3_client                                     = None
kvs_client                                    = None
db_connection                                 = None
face_model                                    = None
mtcnn                                         = None
resnet                                        = None
facenet_device:      Optional[torch.device]   = None
id_store:            Optional["IdentityStore"] = None

camera_sources:      Dict[str, str]               = {}
_kvs_url_cache:      Dict[str, Tuple[str, float]] = {}
_last_frame_hashes:  Dict[str, int]               = {}   # F7 — perceptual dedup
last_camera_refresh: float                        = 0.0
running:             bool                         = True


# ───────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────

def debug(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)


def get_timestamp() -> datetime:
    return datetime.now(TIMEZONE)


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def optimal_imgsz(frame: np.ndarray, max_size: int) -> int:
    """Round the longer edge up to nearest 32-px YOLOv8 stride, then cap."""
    h, w   = frame.shape[:2]
    longer = max(h, w)
    rounded = int(((longer + 31) // 32) * 32)
    return min(rounded, max_size)


# F1 — combined quality score replaces bare Laplacian blur
def frame_quality_score(frame: np.ndarray) -> float:
    """
    Combined sharpness + Michelson contrast score.
    Rejects frames that are sharp but uniformly dark/white (overexposed,
    night-vision flat field) which pass the bare Laplacian gate but carry
    no useful texture for detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mn, mx = float(gray.min()), float(gray.max())
    contrast = (mx - mn) / (mx + mn + 1e-6)
    return blur * (0.5 + contrast)


# kept for any callers that still reference the old name
def frame_blur_score(frame: np.ndarray) -> float:
    return frame_quality_score(frame)


# F7 — perceptual hash for inter-snapshot frame deduplication
def frame_is_duplicate(cam_id: str, frame: np.ndarray) -> bool:
    """
    Compute a 256-bit perceptual hash and compare with the last stored hash
    for this camera.  Returns True when the frame is visually near-identical
    to the previous snapshot (static camera, empty scene) so we can skip
    inference entirely.

    Hamming distance < 10 out of 256 bits ≈ <4% difference — same scene.
    """
    small = cv2.resize(frame, (16, 16))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean  = float(gray.mean())
    bits  = (gray > mean).flatten()
    # Pack 256 bits into 32 bytes → convert to Python int for XOR
    packed = int.from_bytes(np.packbits(bits).tobytes(), "big")
    prev   = _last_frame_hashes.get(cam_id)
    _last_frame_hashes[cam_id] = packed
    if prev is None:
        return False
    hamming = bin(packed ^ prev).count("1")
    return hamming < 10


def get_class_confidence_threshold(class_name: str) -> float:
    return CLASS_CONFIDENCE_THRESHOLDS.get(class_name, DETECTION_CONFIDENCE)


def is_allowed_violation_class(name: str) -> bool:
    return name in ALLOWED_VIOLATION_CLASS_NAMES


def is_boot_class(name: str) -> bool:
    return name in BOOT_CLASS_NAMES


# ───────────────────────────────────────────────────────────────────────────
# Credentials & AWS clients
# ───────────────────────────────────────────────────────────────────────────

def load_credentials() -> Dict[str, Any]:
    if not os.path.exists(PPE_CREDS_FILE):
        raise FileNotFoundError(f"Credentials file not found: {PPE_CREDS_FILE}")
    config = configparser.ConfigParser()
    config.read(PPE_CREDS_FILE)
    for section in ("AWS", "DB"):
        if section not in config:
            raise ValueError(f"Missing [{section}] section in {PPE_CREDS_FILE}")
    aws = config["AWS"]
    db  = config["DB"]
    creds: Dict[str, Any] = {
        "access_key":           aws.get("aws_access_key_id"),
        "secret_key":           aws.get("aws_secret_access_key"),
        "region_name":          aws.get("region_name"),
        "bucket":               aws.get("s3_bucket", "officelens-ppe"),
        "db_type":              db.get("db_type",    "mysql"),
        "db_host":              db.get("db_host"),
        "db_port":              db.getint("db_port", 3306),
        "db_user":              db.get("db_user"),
        "db_password":          db.get("db_password"),
        "db_name":              db.get("db_name"),
        "cameras_table":        db.get("cameras_table",        "OfficeLens_cameras"),
        "ppe_cameras_table":    db.get("ppe_cameras_table",    "OfficeLens_ppe_cameras"),
        "ppe_frames_table":     db.get("ppe_frames_table",     "OfficeLens_ppe_frames"),
        "ppe_detections_table": db.get("ppe_detections_table", "OfficeLens_ppe_detections"),
        "persons_table":        db.get("persons_table",        "OfficeLens_persons"),
    }
    required = [
        "access_key", "secret_key", "region_name", "bucket",
        "db_host", "db_user", "db_password", "db_name",
    ]
    missing = [k for k in required if not creds.get(k)]
    if missing:
        raise ValueError(f"Missing credential values: {', '.join(missing)}")
    return creds


def _boto3_session():
    return boto3.Session(
        aws_access_key_id=aws_creds["access_key"],
        aws_secret_access_key=aws_creds["secret_key"],
        region_name=aws_creds["region_name"],
    )


def init_s3_client():
    return _boto3_session().client("s3")


def init_kvs_client():
    return _boto3_session().client("kinesisvideo")


def init_db_connection():
    import pymysql
    return pymysql.connect(
        host=aws_creds["db_host"],
        port=aws_creds.get("db_port", 3306),
        user=aws_creds["db_user"],
        password=aws_creds["db_password"],
        database=aws_creds["db_name"],
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
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
    db_connection = init_db_connection()


# ───────────────────────────────────────────────────────────────────────────
# S3 helpers
# ───────────────────────────────────────────────────────────────────────────

def s3_public_url(key: str) -> str:
    return (
        f"https://{aws_creds['bucket']}.s3."
        f"{aws_creds['region_name']}.amazonaws.com/{key}"
    )


def s3_upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    s3_client.put_object(
        Bucket=aws_creds["bucket"],
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return s3_public_url(key)


def s3_upload_image(bgr_frame: np.ndarray, key: str, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode frame for S3 key: {key}")
    return s3_upload_bytes(buf.tobytes(), key, "image/jpeg")


def s3_upload_numpy(arr: np.ndarray, key: str) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return s3_upload_bytes(buf.getvalue(), key, "application/octet-stream")


def s3_download_numpy(key: str) -> Optional[np.ndarray]:
    try:
        obj = s3_client.get_object(Bucket=aws_creds["bucket"], Key=key)
        buf = io.BytesIO(obj["Body"].read())
        return np.load(buf)
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        print(f"[S3 WARN] Could not download {key}: {exc}", flush=True)
        return None


def s3_list_keys(prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=aws_creds["bucket"], Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_upload_frame(local_path: str, camera_id: str, timestamp: datetime) -> Optional[str]:
    key = (
        f"{S3_FRAMES_PREFIX}/{camera_id}/"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    )
    try:
        s3_client.upload_file(local_path, aws_creds["bucket"], key)
        return s3_public_url(key)
    except Exception as exc:
        print(f"[S3 ERROR] frame upload failed: {exc}", flush=True)
        return None


# ───────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────────────────────

def box_iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def box_area(box: Tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection_area(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_coverage_fraction(inner: Tuple, outer: Tuple) -> float:
    area = max(1.0, box_area(inner))
    return box_intersection_area(inner, outer) / area


def boxes_refer_to_same_person(a: Tuple, b: Tuple) -> bool:
    return (
        box_iou(a, b) >= PERSON_BOX_DEDUPE_IOU
        or box_coverage_fraction(a, b) >= PERSON_BOX_DEDUPE_COVERAGE
        or box_coverage_fraction(b, a) >= PERSON_BOX_DEDUPE_COVERAGE
    )


# F4 — overlap fraction: what fraction of `inner` overlaps `outer`
def box_overlap_fraction(inner: Tuple, outer: Tuple) -> float:
    """
    Returns the fraction of `inner`'s area that is covered by `outer`.
    More robust than center-inside for PPE boxes that are wider than the
    person box (e.g. a vest detection that bleeds past the person boundary).
    """
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / inner_area


def box_center_inside(inner: Tuple, outer: Tuple) -> bool:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def clamp_box(box: Tuple, fw: int, fh: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(fw - 1, int(round(x1)))),
        max(0, min(fh - 1, int(round(y1)))),
        max(0, min(fw,     int(round(x2)))),
        max(0, min(fh,     int(round(y2)))),
    )


def expand_person_boxes(
    boxes: List[Tuple], fw: int, fh: int
) -> List[Tuple]:
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        out.append((
            max(0.0,        x1 - w * PERSON_BOX_EXPAND_SIDES),
            max(0.0,        y1 - h * PERSON_BOX_EXPAND_TOP),
            min(float(fw),  x2 + w * PERSON_BOX_EXPAND_SIDES),
            min(float(fh),  y2 + h * PERSON_BOX_EXPAND_BOTTOM),
        ))
    return out


# F4 — replacement for the old center-inside-only logic
def find_person_index_for_detection(
    det_box: Tuple,
    person_boxes: List[Tuple],
) -> Optional[int]:
    """
    Three-tier matching (most→least strict):
      1. Overlap fraction ≥ PPE_OVERLAP_GATE_THRESHOLD  (primary)
      2. Detection centre is inside person box           (secondary)
      3. Best IoU > 0.10 minimum                        (tertiary)

    Old code used center-inside as primary with IoU > 0 fallback.
    The new primary catches PPE boxes that are wider than the person
    box (vest, boots) or partially out of frame.
    The IoU tertiary minimum of 0.10 prevents stray far-away matches
    that the old >0 threshold allowed.
    """
    if not person_boxes:
        return None

    # Tier 1: overlap fraction
    best_overlap_idx, best_overlap = None, 0.0
    for i, pb in enumerate(person_boxes):
        ov = box_overlap_fraction(det_box, pb)
        if ov >= PPE_OVERLAP_GATE_THRESHOLD and ov > best_overlap:
            best_overlap_idx, best_overlap = i, ov
    if best_overlap_idx is not None:
        return best_overlap_idx

    # Tier 2: centre-inside
    for i, pb in enumerate(person_boxes):
        if box_center_inside(det_box, pb):
            return i

    # Tier 3: best IoU with a meaningful minimum
    best_idx, best_iou = None, 0.10
    for i, pb in enumerate(person_boxes):
        iou = box_iou(det_box, pb)
        if iou > best_iou:
            best_idx, best_iou = i, iou
    return best_idx


def person_face_search_region(
    person_box: Tuple, fw: int, fh: int,
) -> Tuple[int, int, int, int]:
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    return clamp_box((
        px1 - pw * FACE_REGION_MARGIN,
        py1 - ph * FACE_REGION_MARGIN * 0.5,
        px2 + pw * FACE_REGION_MARGIN,
        py1 + ph * FACE_REGION_HEIGHT_RATIO,
    ), fw, fh)


# ───────────────────────────────────────────────────────────────────────────
# Detection post-processing
# ───────────────────────────────────────────────────────────────────────────

def dedupe_detections(detections: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if not any(
            det["class_name"] == k["class_name"]
            and box_iou(det["box"], k["box"]) >= DETECTION_DEDUPE_IOU
            for k in kept
        ):
            kept.append(det)
    return kept


def dedupe_person_boxes(person_boxes: List[Tuple]) -> List[Tuple]:
    kept: List[Tuple] = []
    for box in sorted(person_boxes, key=box_area, reverse=True):
        if box_area(box) <= 0:
            continue
        if not any(boxes_refer_to_same_person(box, existing) for existing in kept):
            kept.append(box)
    if len(kept) != len(person_boxes):
        debug(f"[PERSON] Deduped boxes: {len(person_boxes)} -> {len(kept)}")
    return kept


def is_valid_person_box(box: Tuple) -> bool:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < PERSON_MIN_BOX_WIDTH or h < PERSON_MIN_BOX_HEIGHT:
        return False
    aspect = h / max(w, 1.0)
    return PERSON_MIN_ASPECT_RATIO <= aspect <= PERSON_MAX_ASPECT_RATIO


def filter_person_boxes(person_boxes: List[Tuple]) -> List[Tuple]:
    kept = [box for box in person_boxes if is_valid_person_box(box)]
    if len(kept) != len(person_boxes):
        debug(f"[PERSON] Quality filter: {len(person_boxes)} -> {len(kept)}")
    return kept


def assign_detections_to_persons(
    detections: List[Dict],
    person_boxes: List[Tuple],
    fw: int,
    fh: int,
) -> List[Dict]:
    if not person_boxes:
        if not detections:
            return []
        kept, suppressed = [], 0
        for det in detections:
            if not PERSON_MATCH_REQUIRED or det["confidence"] >= PERSON_MATCH_CONF_BYPASS:
                det["person_index"] = -1
                kept.append(det)
            else:
                suppressed += 1
        if suppressed:
            print(
                f"[GATE] No persons — kept {len(kept)} high-conf, "
                f"suppressed {suppressed} low-conf",
                flush=True,
            )
        return kept

    expanded   = expand_person_boxes(person_boxes, fw, fh)
    suppressed = 0
    bypassed   = 0
    by_person_class: Dict[Tuple[int, str], List[Dict]] = {}

    for det in detections:
        # Detections already tagged with person_index (from crop inference) skip matching
        pidx = det.get("person_index")
        if pidx is None:
            pidx = find_person_index_for_detection(det["box"], expanded)

        if pidx is None:
            if not PERSON_MATCH_REQUIRED or det["confidence"] >= PERSON_MATCH_CONF_BYPASS:
                by_person_class.setdefault((-1, det["class_name"]), []).append(det)
                bypassed += 1
            else:
                suppressed += 1
            continue

        det["person_index"] = pidx
        by_person_class.setdefault((pidx, det["class_name"]), []).append(det)

    if suppressed or bypassed:
        print(
            f"[GATE] Unmatched PPE: {bypassed} kept (bypass), "
            f"{suppressed} suppressed",
            flush=True,
        )

    kept: List[Dict] = []
    for (_pidx, cname), group in by_person_class.items():
        limit = 2 if cname == "NO-Safety Boots" else 1
        kept.extend(sorted(group, key=lambda d: d["confidence"], reverse=True)[:limit])
    return kept


# ───────────────────────────────────────────────────────────────────────────
# F3 — PPE inference on per-person expanded crops
# ───────────────────────────────────────────────────────────────────────────

def detect_ppe_on_crops(
    frame: np.ndarray,
    person_boxes: List[Tuple],
    device,
) -> List[Dict]:
    """
    Run PPE violation detection on each person's expanded crop at a fixed
    PPE_CROP_INFERENCE_SIZE resolution.

    This normalises small/distant workers to the same apparent input size
    as close workers, solving the root cause of missed violations on workers
    that are far from the camera (where the full-frame PPE pass sees them
    as tiny blobs even at imgsz=1920).

    Detections are returned with person_index already set (no spatial gate
    needed) and coordinates remapped to full-frame space.
    """
    if not model_config or not person_boxes:
        return []

    fh, fw     = frame.shape[:2]
    crop_dets: List[Dict] = []

    for pidx, pb in enumerate(person_boxes):
        px1, py1, px2, py2 = pb
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)

        # Expand: extra top margin to capture hard hat above head
        cx1 = max(0,  int(px1 - pw * PPE_CROP_MARGIN))
        cy1 = max(0,  int(py1 - ph * (PPE_CROP_MARGIN + 0.10)))
        cx2 = min(fw, int(px2 + pw * PPE_CROP_MARGIN))
        cy2 = min(fh, int(py2 + ph * PPE_CROP_MARGIN))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            continue

        results = model_config["ppe_model"].predict(
            source=crop,
            imgsz=PPE_CROP_INFERENCE_SIZE,
            conf=MODEL_INFERENCE_CONFIDENCE,
            iou=DETECTION_IOU,
            classes=model_config["violation_class_ids"],
            device=device,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue

        for box, cls_id, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            cname = model_config["class_names"].get(int(cls_id), str(int(cls_id)))
            cval  = float(conf)
            if not is_allowed_violation_class(cname):
                continue
            if cval < get_class_confidence_threshold(cname):
                continue
            lx1, ly1, lx2, ly2 = [float(v) for v in box]
            crop_dets.append({
                "class_name":   cname,
                "confidence":   cval,
                "box":          (lx1 + cx1, ly1 + cy1, lx2 + cx1, ly2 + cy1),
                "person_index": pidx,
            })

    return crop_dets


# ───────────────────────────────────────────────────────────────────────────
# Boot crop helpers
# ───────────────────────────────────────────────────────────────────────────

def make_foot_crop_box(
    frame: np.ndarray,
    person_box: Tuple,
) -> Optional[Tuple[int, int, int, int]]:
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    pw, ph = x2 - x1, y2 - y1
    if ph < BOOT_MIN_PERSON_HEIGHT or pw <= 0:
        return None
    cy1 = y2 - ph * BOOT_CROP_FOOT_RATIO
    cx1 = x1 - pw * BOOT_CROP_MARGIN
    cx2 = x2 + pw * BOOT_CROP_MARGIN
    cy2 = y2 + ph * BOOT_CROP_MARGIN * 0.5
    box = (
        clamp(int(cx1), 0, fw - 1),
        clamp(int(cy1), 0, fh - 1),
        clamp(int(cx2), 0, fw - 1),
        clamp(int(cy2), 0, fh - 1),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def black_or_brown_pixel_ratio(crop: np.ndarray) -> float:
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, (0, 0, 0),    (180, 100, 70))
    brown = cv2.inRange(hsv, (5, 40, 35),  (30, 255, 170))
    mask  = cv2.bitwise_or(black, brown)
    return float(cv2.countNonZero(mask)) / float(crop.shape[0] * crop.shape[1])


def person_has_boot_colored_lower_body(frame: np.ndarray, person_box: Tuple) -> bool:
    box = make_foot_crop_box(frame, person_box)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    return bool(crop.size) and black_or_brown_pixel_ratio(crop) >= BOOT_COLOR_MIN_RATIO


def suppress_boot_violations_by_color(
    frame: np.ndarray,
    detections: List[Dict],
    person_boxes: List[Tuple],
) -> List[Dict]:
    if not ENABLE_BOOT_COLOR_CHECK or not person_boxes:
        return detections
    cache: Dict[int, bool] = {}
    result = []
    for det in detections:
        if det["class_name"] != "NO-Safety Boots":
            result.append(det)
            continue
        pidx = det.get("person_index")
        if pidx is None:
            result.append(det)
            continue
        if pidx not in cache:
            cache[pidx] = person_has_boot_colored_lower_body(frame, person_boxes[pidx])
        if not cache[pidx]:
            result.append(det)
        else:
            debug(f"[BOOT COLOR] Suppressed NO-Safety Boots person {pidx+1}")
    return result


def detect_boot_violations_for_people(
    frame: np.ndarray,
    person_boxes: List[Tuple],
    device,
) -> List[Dict]:
    if not model_config or not person_boxes:
        return []
    boot_dets: List[Dict] = []
    for pidx, pb in enumerate(person_boxes):
        crop_box = make_foot_crop_box(frame, pb)
        if crop_box is None:
            continue
        cx1, cy1, cx2, cy2 = crop_box
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        # F6 — fixed 640 instead of native crop size
        results = model_config["ppe_model"].predict(
            source=crop, imgsz=640, conf=BOOT_CONFIDENCE, iou=BOOT_IOU,
            classes=model_config["boot_class_ids"], device=device, verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue
        for box, cls_id, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            cname = model_config["class_names"].get(int(cls_id), str(int(cls_id)))
            if cname != "NO-Safety Boots":
                continue
            lx1, ly1, lx2, ly2 = [float(v) for v in box]
            boot_dets.append({
                "class_name":   "NO-Safety Boots",
                "confidence":   float(conf),
                "box":          (lx1 + cx1, ly1 + cy1, lx2 + cx1, ly2 + cy1),
                "person_index": pidx,
            })
    return boot_dets


# ───────────────────────────────────────────────────────────────────────────
# Face detection helpers
# ───────────────────────────────────────────────────────────────────────────

def get_best_face_in_person(
    frame: np.ndarray,
    person_box: Tuple,
    face_boxes: List[Tuple],
) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
    if not face_boxes:
        return None, None
    px1, py1, px2, py2 = person_box
    ph = max(1.0, py2 - py1)
    fh, fw = frame.shape[:2]
    best_face, best_score = None, -1.0
    for fb in face_boxes:
        fx1, fy1, fx2, fy2 = fb
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
        area  = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        score = area + max(0.0, 1.0 - (cy - py1) / ph) * 1000.0
        if score > best_score:
            best_score = score
            best_face  = fb
    if best_face is None:
        return None, None
    x1, y1, x2, y2 = clamp_box(best_face, fw, fh)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2]
    return (None, None) if crop.size == 0 else (crop, best_face)


def get_face_from_person_crop(
    frame: np.ndarray,
    person_box: Tuple,
    device,
) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
    fh, fw = frame.shape[:2]
    rx1, ry1, rx2, ry2 = person_face_search_region(person_box, fw, fh)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, None
    region = frame[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return None, None
    results = face_model.predict(
        source=region,
        imgsz=optimal_imgsz(region, FACE_CROP_IMAGE_SIZE),
        conf=FACE_CROP_CONFIDENCE,
        iou=0.45,
        device=device,
        verbose=False,
    )[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, None
    best_box, best_conf = None, -1.0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            lx1, ly1, lx2, ly2 = [float(x) for x in box.xyxy[0].tolist()]
            best_box = (lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1)
            best_conf = conf
    if best_box is None:
        return None, None
    x1, y1, x2, y2 = clamp_box(best_box, fw, fh)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2]
    return (None, None) if crop.size == 0 else (crop, best_box)


# ───────────────────────────────────────────────────────────────────────────
# Head-pose (yaw) estimation from MTCNN landmarks
# ───────────────────────────────────────────────────────────────────────────

def estimate_yaw_from_landmarks(landmarks: np.ndarray) -> float:
    left_eye   = landmarks[0]
    right_eye  = landmarks[1]
    nose       = landmarks[2]
    eye_centre_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_width    = abs(right_eye[0] - left_eye[0])
    if eye_width < 1e-3:
        return 0.0
    deviation = (nose[0] - eye_centre_x) / eye_width
    yaw_deg = float(np.degrees(np.arctan(deviation * 2.0)))
    return yaw_deg


# ───────────────────────────────────────────────────────────────────────────
# FaceNet embedding extraction
# ───────────────────────────────────────────────────────────────────────────

def extract_embedding(
    face_crop_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, bool]:
    h, w = face_crop_bgr.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return None, 0.0, False

    pil_img = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

    aligned, prob = mtcnn(pil_img, return_prob=True)
    mtcnn_ok      = aligned is not None and (prob is None or prob > 0.85)

    yaw_deg = 0.0
    if mtcnn_ok:
        try:
            with torch.no_grad():
                boxes_t, _, landmarks_t = mtcnn.detect(pil_img, landmarks=True)
            if landmarks_t is not None and len(landmarks_t) > 0:
                yaw_deg = abs(estimate_yaw_from_landmarks(landmarks_t[0]))
        except Exception:
            pass
    else:
        resized = pil_img.resize((_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE), Image.BILINEAR)
        arr     = np.array(resized, dtype=np.float32) / 127.5 - 1.0
        aligned = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)

    if aligned.dim() == 3:
        aligned = aligned.unsqueeze(0)

    aligned = aligned.to(facenet_device)
    with torch.no_grad():
        emb = resnet(aligned)
        emb = F.normalize(emb, p=2, dim=1)

    return emb.squeeze(0).cpu().numpy(), yaw_deg, mtcnn_ok


# ───────────────────────────────────────────────────────────────────────────
# In-memory identity store
# ───────────────────────────────────────────────────────────────────────────

class IdentityStore:
    def __init__(self) -> None:
        self._mean_emb:    Dict[int, np.ndarray]       = {}
        self._prototypes:  Dict[int, List[np.ndarray]] = {}
        self._meta:        Dict[int, Dict[str, Any]]   = {}
        self._mean_matrix: Optional[np.ndarray]        = None
        self._mean_ids:    List[int]                   = []

    def load(self) -> None:
        reconnect_db()
        pt = aws_creds["persons_table"]
        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    f"SELECT person_id, first_seen, total_violations, "
                    f"embedding_s3_url, crop_s3_url FROM {pt}"
                )
                rows = cur.fetchall()
        except Exception as exc:
            print(f"[REID] WARNING: could not load persons from DB: {exc}", flush=True)
            rows = []

        if not rows:
            print("[REID] No existing persons in DB — starting fresh.", flush=True)
            return

        print(f"[REID] Loading {len(rows)} person embeddings from S3...", flush=True)
        loaded = 0
        for row in rows:
            pid     = int(row["person_id"])
            emb_url = row.get("embedding_s3_url") or ""
            emb_key = self._url_to_key(emb_url)
            if not emb_key:
                continue
            emb = s3_download_numpy(emb_key)
            if emb is None:
                print(f"[REID] WARNING: embedding missing in S3 for person {pid}", flush=True)
                continue
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self._mean_emb[pid]   = emb
            self._prototypes[pid] = [emb.copy()]
            self._meta[pid] = {
                "first_seen":       row.get("first_seen"),
                "total_violations": int(row.get("total_violations") or 0),
                "embedding_s3_url": emb_url,
                "crop_s3_url":      row.get("crop_s3_url") or "",
            }
            loaded += 1

        self._rebuild_matrix()
        print(f"[REID] Loaded {loaded} person embeddings into memory.", flush=True)

    @staticmethod
    def _url_to_key(url: str) -> str:
        if not url:
            return ""
        try:
            after_host = url.split(".amazonaws.com/", 1)
            if len(after_host) == 2:
                return after_host[1]
        except Exception:
            pass
        return ""

    def _rebuild_matrix(self) -> None:
        if not self._mean_emb:
            self._mean_matrix = None
            self._mean_ids    = []
            return
        ids = sorted(self._mean_emb.keys())
        mat = np.stack([self._mean_emb[i] for i in ids], axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._mean_matrix = mat / (norms + 1e-8)
        self._mean_ids    = ids

    def match(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        if self._mean_matrix is None or len(self._mean_ids) == 0:
            return None, 0.0

        q    = embedding / (np.linalg.norm(embedding) + 1e-8)
        sims = self._mean_matrix @ q

        top_k       = min(5, len(self._mean_ids))
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        best_pid, best_sim = None, -1.0

        for idx in top_indices:
            pid  = self._mean_ids[idx]
            sim  = float(sims[idx])
            protos = self._prototypes.get(pid, [])
            if protos:
                proto_mat  = np.stack(protos, axis=0)
                proto_sims = proto_mat @ q
                sim = max(sim, float(proto_sims.max()))
            if sim > best_sim:
                best_sim, best_pid = sim, pid

        return best_pid, best_sim

    def is_match(self, similarity: float) -> bool:
        return similarity >= REID_THRESHOLD

    def insert_new_person(
        self,
        embedding: np.ndarray,
        body_crop_bgr: np.ndarray,
        timestamp: datetime,
    ) -> int:
        if DRY_RUN:
            fake_id = -(len(self._mean_emb) + 1)
            self._mean_emb[fake_id]   = embedding.copy()
            self._prototypes[fake_id] = [embedding.copy()]
            self._meta[fake_id] = {
                "first_seen":       timestamp,
                "total_violations": 0,
                "embedding_s3_url": "",
                "crop_s3_url":      "",
            }
            self._rebuild_matrix()
            print(f"[DRY_RUN] Would insert new person (fake_id={fake_id})", flush=True)
            return fake_id

        reconnect_db()
        pt     = aws_creds["persons_table"]
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {pt} (first_seen, total_violations, "
                    f"embedding_s3_url, crop_s3_url) VALUES (%s, %s, %s, %s)",
                    (ts_str, 0, "", ""),
                )
                person_id = cur.lastrowid
            db_connection.commit()
        except Exception as exc:
            db_connection.rollback()
            raise RuntimeError(f"Failed to insert person row: {exc}") from exc

        emb_key  = f"{S3_EMBEDDINGS_PREFIX}/person_{person_id:06d}.npy"
        crop_key = f"{S3_CROPS_PREFIX}/person_{person_id:06d}.jpg"
        emb_url  = s3_upload_numpy(embedding, emb_key)
        crop_url = s3_upload_image(body_crop_bgr, crop_key)

        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    f"UPDATE {pt} SET embedding_s3_url=%s, crop_s3_url=%s "
                    f"WHERE person_id=%s",
                    (emb_url, crop_url, person_id),
                )
            db_connection.commit()
        except Exception as exc:
            db_connection.rollback()
            print(f"[WARN] Could not update S3 URLs for person {person_id}: {exc}", flush=True)

        self._mean_emb[person_id]   = embedding.copy()
        self._prototypes[person_id] = [embedding.copy()]
        self._meta[person_id] = {
            "first_seen":       timestamp,
            "total_violations": 0,
            "embedding_s3_url": emb_url,
            "crop_s3_url":      crop_url,
        }
        self._rebuild_matrix()
        print(f"[REID] New person registered: person_id={person_id}", flush=True)
        return person_id

    def maybe_update_person(
        self,
        person_id: int,
        new_embedding: np.ndarray,
        new_body_crop_bgr: np.ndarray,
        mtcnn_succeeded: bool,
        yaw_deg: float,
        similarity: float,
    ) -> None:
        if person_id not in self._mean_emb:
            return

        current_mean = self._mean_emb[person_id]
        is_frontal   = mtcnn_succeeded and yaw_deg < FRONTAL_YAW_THRESHOLD

        protos     = self._prototypes.setdefault(person_id, [current_mean.copy()])
        is_diverse = all(float(np.dot(new_embedding, p)) < 0.90 for p in protos)
        if is_diverse and len(protos) < MAX_PROTOTYPES:
            protos.append(new_embedding.copy())
            debug(
                f"[REID] person {person_id}: added prototype "
                f"(yaw={yaw_deg:.1f}°, bank size={len(protos)})"
            )

        if not is_frontal:
            debug(
                f"[REID] person {person_id}: skipping mean update "
                f"(mtcnn={'ok' if mtcnn_succeeded else 'fallback'}, "
                f"yaw={yaw_deg:.1f}°)"
            )
            return

        if similarity < 0.90:
            debug(
                f"[REID] person {person_id}: skipping mean update "
                f"(sim={similarity:.4f} < 0.90 quality gate)"
            )
            return

        n       = self._meta[person_id].get("embedding_count", 1)
        new_mean = (current_mean * n + new_embedding) / (n + 1)
        norm    = np.linalg.norm(new_mean)
        if norm > 1e-8:
            new_mean = new_mean / norm

        self._mean_emb[person_id] = new_mean
        self._meta[person_id]["embedding_count"] = n + 1
        self._rebuild_matrix()

        if not DRY_RUN:
            emb_key  = f"{S3_EMBEDDINGS_PREFIX}/person_{person_id:06d}.npy"
            crop_key = f"{S3_CROPS_PREFIX}/person_{person_id:06d}.jpg"
            try:
                emb_url  = s3_upload_numpy(new_mean, emb_key)
                crop_url = s3_upload_image(new_body_crop_bgr, crop_key)
                self._meta[person_id]["embedding_s3_url"] = emb_url
                self._meta[person_id]["crop_s3_url"]      = crop_url
                reconnect_db()
                pt = aws_creds["persons_table"]
                with db_connection.cursor() as cur:
                    cur.execute(
                        f"UPDATE {pt} SET embedding_s3_url=%s, crop_s3_url=%s "
                        f"WHERE person_id=%s",
                        (emb_url, crop_url, person_id),
                    )
                db_connection.commit()
                debug(f"[REID] person {person_id}: mean updated (n={n+1}, yaw={yaw_deg:.1f}°)")
            except Exception as exc:
                db_connection.rollback()
                print(f"[WARN] S3/DB update failed for person {person_id}: {exc}", flush=True)

    def increment_violations(self, person_id: int, count: int) -> int:
        if person_id not in self._meta:
            return 0
        self._meta[person_id]["total_violations"] = (
            self._meta[person_id].get("total_violations", 0) + count
        )
        new_total = self._meta[person_id]["total_violations"]
        if not DRY_RUN:
            try:
                reconnect_db()
                pt = aws_creds["persons_table"]
                with db_connection.cursor() as cur:
                    cur.execute(
                        f"UPDATE {pt} SET total_violations=%s WHERE person_id=%s",
                        (new_total, person_id),
                    )
                db_connection.commit()
            except Exception as exc:
                db_connection.rollback()
                print(f"[WARN] violation count update failed: {exc}", flush=True)
        return new_total

    def known_count(self) -> int:
        return len(self._mean_emb)

    def total_violations_for(self, person_id: int) -> int:
        return self._meta.get(person_id, {}).get("total_violations", 0)

    def metadata_for(self, person_id: int) -> Dict[str, Any]:
        meta = self._meta.get(person_id, {})
        return {
            "person_id": person_id,
            "first_seen": meta.get("first_seen"),
            "total_violations": meta.get("total_violations"),
            "crop_s3_url": meta.get("crop_s3_url"),
        }


# ───────────────────────────────────────────────────────────────────────────
# Model initialisation
# ───────────────────────────────────────────────────────────────────────────

def build_violation_class_ids(ppe_names: Dict[int, str]) -> List[int]:
    def norm(s: str) -> str:
        return s.lower().replace("-", " ").replace("_", " ").strip()
    allowed_norm = {norm(n): n for n in ALLOWED_VIOLATION_CLASS_NAMES}
    matched, unmatched = [], set(ALLOWED_VIOLATION_CLASS_NAMES)
    for cid, cname in ppe_names.items():
        if norm(cname) in allowed_norm:
            matched.append(cid)
            unmatched.discard(allowed_norm[norm(cname)])
            print(f"[MODEL] Violation class matched: '{cname}' (id={cid})", flush=True)
    if unmatched:
        print(f"[WARN] Violation classes NOT found in model: {unmatched}", flush=True)
    return matched


def init_models() -> None:
    global model_config, face_model, mtcnn, resnet, facenet_device

    for label, path in [
        ("Person", PERSON_MODEL_PATH),
        ("PPE",    PPE_MODEL_PATH),
        ("Face",   FACE_MODEL_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} model not found: {path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Inference device: {device}", flush=True)

    print(f"[MODEL] Loading person model: {PERSON_MODEL_PATH}", flush=True)
    person_model       = YOLO(str(PERSON_MODEL_PATH))
    person_class_names = {int(k): str(v) for k, v in person_model.names.items()}
    person_class_ids   = [
        cid for cid, cn in person_class_names.items()
        if cn.strip().lower() == "person"
    ]
    if not person_class_ids:
        raise ValueError(f"Person model has no 'person' class: {person_class_names}")

    print(f"[MODEL] Loading PPE model: {PPE_MODEL_PATH}", flush=True)
    ppe_model       = YOLO(str(PPE_MODEL_PATH))
    ppe_class_names = {int(k): str(v) for k, v in ppe_model.names.items()}

    violation_class_ids = build_violation_class_ids(ppe_class_names)
    if not violation_class_ids:
        raise ValueError(f"No violation classes matched: {ppe_class_names}")

    boot_class_ids = [
        cid for cid, cn in ppe_class_names.items() if is_boot_class(cn)
    ]

    print(f"[MODEL] Loading face model: {FACE_MODEL_PATH}", flush=True)
    face_model = YOLO(str(FACE_MODEL_PATH))

    print("[MODEL] Loading FaceNet (MTCNN + InceptionResnetV1)...", flush=True)
    facenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
        image_size=_FACENET_INPUT_SIZE,
        margin=14,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False,
        select_largest=True,
        device=facenet_device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(facenet_device)
    print(f"[MODEL] FaceNet device: {facenet_device}", flush=True)

    model_config = {
        "person_model":        person_model,
        "person_class_names":  person_class_names,
        "person_class_ids":    person_class_ids,
        "ppe_model":           ppe_model,
        "class_names":         ppe_class_names,
        "violation_class_ids": violation_class_ids,
        "boot_class_ids":      boot_class_ids,
        "device":              device,
    }
    print(
        f"[MODEL] Ready — person classes: {person_class_ids}, "
        f"violation class ids: {violation_class_ids}",
        flush=True,
    )


# ───────────────────────────────────────────────────────────────────────────
# KVS / camera helpers
# ───────────────────────────────────────────────────────────────────────────

def check_kvs_stream_exists(stream_name: str) -> bool:
    try:
        kvs_client.describe_stream(StreamName=stream_name)
        return True
    except Exception as exc:
        code = ""
        try:
            code = exc.response["Error"]["Code"]
        except Exception:
            pass
        if code != "ResourceNotFoundException":
            print(f"[KVS] Error checking stream {stream_name}: {exc}", flush=True)
        return False


def get_hls_streaming_url(stream_name: str) -> Optional[str]:
    cached = _kvs_url_cache.get(stream_name)
    if cached:
        url, ts = cached
        if time.time() - ts < KVS_URL_CACHE_DURATION:
            return url
    try:
        ep = kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName="GET_HLS_STREAMING_SESSION_URL",
        )["DataEndpoint"]
        archived = _boto3_session().client(
            "kinesis-video-archived-media", endpoint_url=ep
        )
        url = archived.get_hls_streaming_session_url(
            StreamName=stream_name,
            PlaybackMode=KVS_PLAYBACK_MODE,
            HLSFragmentSelector={"FragmentSelectorType": "SERVER_TIMESTAMP"},
            Expires=300,
        )["HLSStreamingSessionURL"]
        _kvs_url_cache[stream_name] = (url, time.time())
        return url
    except Exception as exc:
        print(f"[KVS ERROR] HLS URL for {stream_name}: {exc}", flush=True)
        return None


def load_configured_cameras() -> bool:
    global camera_sources, last_camera_refresh
    reconnect_db()
    ct  = aws_creds["cameras_table"]
    pct = aws_creds["ppe_cameras_table"]
    try:
        with db_connection.cursor() as cur:
            cur.execute(
                f"SELECT c.id FROM {pct} pc "
                f"JOIN {ct} c ON c.id=pc.cam_id ORDER BY c.id"
            )
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[DB ERROR] Failed to load PPE cameras: {exc}", flush=True)
        return False

    discovered = {str(r["id"]): str(r["id"]) for r in rows}
    if not discovered:
        print("[CAMERA] No cameras in PPE camera table", flush=True)
        camera_sources = {}
        return False

    valid: Dict[str, str] = {}
    for cam_id, sname in discovered.items():
        if check_kvs_stream_exists(sname):
            valid[cam_id] = sname
        else:
            print(f"[KVS SKIP] Camera {cam_id}: stream '{sname}' not found", flush=True)

    camera_sources      = valid
    last_camera_refresh = time.time()
    print(f"[CAMERA] {len(camera_sources)} active PPE cameras", flush=True)
    return bool(camera_sources)


def refresh_cameras_if_needed(force: bool = False) -> bool:
    if force or not camera_sources or time.time() - last_camera_refresh >= CONFIG_REFRESH_INTERVAL:
        return load_configured_cameras()
    return bool(camera_sources)


# ───────────────────────────────────────────────────────────────────────────
# Frame capture
# ───────────────────────────────────────────────────────────────────────────

def capture_single_camera(
    cam_id: str,
    stream_name: str,
) -> Tuple[str, Optional[np.ndarray], Optional[datetime], Optional[str]]:
    """
    Capture CAPTURE_FRAMES_PER_CAM frames from KVS HLS, return the
    highest-quality one.

    Inter-sample sleep is at least 0.5s so successive reads span different
    GOPs — purely temporal diversity, not just sharpness of the same keyframe.

    F1: uses frame_quality_score() (sharpness × contrast) instead of bare
        Laplacian so dark/overexposed flat frames are rejected.
    F7: perceptual hash check — if the best frame is visually identical to
        the previous snapshot for this camera, skip inference entirely.
    """
    INTER_SAMPLE_SLEEP = max(0.5, 2.0 / max(1, CAPTURE_FRAMES_PER_CAM))

    try:
        hls_url = get_hls_streaming_url(stream_name)
        if not hls_url:
            return cam_id, None, None, None

        cap = cv2.VideoCapture(hls_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Phase 1: drain — wait for stream to deliver any frame
        first_frame = None
        drain_start = time.time()
        while time.time() - drain_start < CONNECTION_TIMEOUT:
            ret, frame = cap.read()
            if ret and frame is not None:
                first_frame = frame
                break
            time.sleep(0.05)

        if first_frame is None:
            cap.release()
            print(
                f"[WARN] Camera {cam_id}: stream did not deliver a frame within "
                f"{CONNECTION_TIMEOUT}s",
                flush=True,
            )
            return cam_id, None, None, None

        # Phase 2: collect N samples spaced in time
        best_frame = first_frame
        best_score = frame_quality_score(first_frame)
        sampled    = 1

        while sampled < CAPTURE_FRAMES_PER_CAM:
            time.sleep(INTER_SAMPLE_SLEEP)
            ret, frame = cap.read()
            if not ret or frame is None:
                ret, frame = cap.read()
            if ret and frame is not None:
                score = frame_quality_score(frame)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
            sampled += 1

        cap.release()

        if best_frame is None:
            print(f"[WARN] Camera {cam_id}: no frames captured", flush=True)
            return cam_id, None, None, None

        if CAPTURE_MIN_BLUR_SCORE > 0 and best_score < CAPTURE_MIN_BLUR_SCORE:
            print(
                f"[WARN] Camera {cam_id}: quality={best_score:.1f} below threshold "
                f"{CAPTURE_MIN_BLUR_SCORE} — proceeding anyway",
                flush=True,
            )

        # F7 — skip inference if scene hasn't changed since last snapshot
        if frame_is_duplicate(cam_id, best_frame):
            debug(f"[DEDUP] Camera {cam_id}: frame identical to previous snapshot, skipping")
            cap_release_safe = True
            return cam_id, None, None, None

        ts   = get_timestamp()
        name = f"{ts.strftime('%Y%m%d_%H%M%S')}_{cam_id}_{uuid.uuid4().hex[:8]}.jpg"
        tmp_path = os.path.join(tempfile.gettempdir(), name)
        cv2.imwrite(tmp_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return cam_id, best_frame, ts, tmp_path

    except Exception as exc:
        print(f"[ERROR] Camera {cam_id}: capture failed: {exc}", flush=True)
        return cam_id, None, None, None


def capture_all_cameras_parallel() -> List[Dict[str, Any]]:
    print(f"[CAPTURE] Capturing {len(camera_sources)} cameras...", flush=True)
    t0       = time.time()
    captured = []
    with ThreadPoolExecutor(max_workers=CAPTURE_WORKERS) as ex:
        futures = {
            ex.submit(capture_single_camera, cid, sname): cid
            for cid, sname in camera_sources.items()
        }
        for fut in as_completed(futures):
            cid, frame, ts, path = fut.result()
            if frame is not None:
                captured.append({
                    "camera_id":  cid,
                    "frame":      frame,
                    "timestamp":  ts,
                    "local_path": path,
                    "s3_url":     None,
                    "detections": [],
                    "persons":    [],
                })
    print(
        f"[CAPTURE] {len(captured)}/{len(camera_sources)} frames in {time.time()-t0:.2f}s",
        flush=True,
    )
    return captured


# ───────────────────────────────────────────────────────────────────────────
# Person detection — primary + fallback + crowd recovery + tiling (F2)
# ───────────────────────────────────────────────────────────────────────────

def recover_close_persons(
    frame: np.ndarray,
    person_boxes: List[Tuple],
    device,
) -> List[Tuple]:
    if not model_config or not person_boxes:
        return person_boxes
    fh, fw    = frame.shape[:2]
    recovered = list(person_boxes)
    new_found = 0

    for px1, py1, px2, py2 in person_boxes:
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            continue
        expand = 0.50
        cx1 = max(0,  int(px1 - pw * expand))
        cy1 = max(0,  int(py1 - ph * expand))
        cx2 = min(fw, int(px2 + pw * expand))
        cy2 = min(fh, int(py2 + ph * expand))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        crop_imgsz = optimal_imgsz(crop, min(PERSON_IMAGE_SIZE_MAX * 2, 1280))
        results = model_config["person_model"].predict(
            source=crop, imgsz=crop_imgsz,
            conf=CROWD_RECOVERY_CONF, iou=CROWD_RECOVERY_IOU_CAP,
            classes=model_config["person_class_ids"],
            device=device, agnostic_nms=True, verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            continue
        for box in results[0].boxes.xyxy.cpu().numpy():
            lx1, ly1, lx2, ly2 = [float(v) for v in box]
            full = (lx1 + cx1, ly1 + cy1, lx2 + cx1, ly2 + cy1)
            if not any(box_iou(full, ex) > CROWD_RECOVERY_NEW_BOX_IOU for ex in recovered):
                recovered.append(full)
                new_found += 1

    if new_found:
        print(
            f"[CROWD] Recovered {new_found} additional person(s) "
            f"(total={len(recovered)})",
            flush=True,
        )
    return recovered


# F2 — SAHI-style tiled inference for small/distant workers
def detect_persons_tiled(frame: np.ndarray, device) -> List[Tuple]:
    """
    Slice the frame into overlapping TILING_SIZE×TILING_SIZE tiles and run
    person detection on each at a fixed 640px resolution.

    Workers far from the camera appear as small blobs in the full-frame
    pass (e.g. 35px tall in a 4K feed).  In a 640×640 tile centred on
    that region they appear at near-normal scale and are reliably detected.

    Results are merged with NMS using PERSON_IOU as the overlap threshold.
    Only fires when the frame area exceeds TILING_AREA_THRESHOLD or when
    the primary pass finds fewer than PERSON_FALLBACK_THRESHOLD persons,
    to avoid adding overhead on close-range / low-person-count cameras.
    """
    fh, fw    = frame.shape[:2]
    tile_sz   = TILING_SIZE
    stride    = int(tile_sz * (1.0 - TILING_OVERLAP))
    tile_conf = max(PERSON_CONFIDENCE - 0.05, 0.15)
    all_boxes: List[Tuple] = []

    for y0 in range(0, fh, stride):
        for x0 in range(0, fw, stride):
            x1c = min(x0 + tile_sz, fw)
            y1c = min(y0 + tile_sz, fh)
            tile = frame[y0:y1c, x0:x1c]
            if tile.shape[0] < 64 or tile.shape[1] < 64:
                continue
            res = model_config["person_model"].predict(
                source=tile,
                imgsz=tile_sz,
                conf=tile_conf,
                iou=PERSON_IOU,
                classes=model_config["person_class_ids"],
                device=device,
                augment=False,
                agnostic_nms=True,
                verbose=False,
            )
            if res and len(res[0].boxes) > 0:
                for b in res[0].boxes.xyxy.cpu().numpy():
                    all_boxes.append((
                        float(b[0]) + x0,
                        float(b[1]) + y0,
                        float(b[2]) + x0,
                        float(b[3]) + y0,
                    ))

    if not all_boxes:
        return []

    # NMS across all tiles — use torchvision if available, fallback to greedy
    try:
        import torchvision.ops as tv_ops
        boxes_t  = torch.tensor(all_boxes, dtype=torch.float32)
        scores_t = torch.ones(len(all_boxes), dtype=torch.float32)
        keep     = tv_ops.nms(boxes_t, scores_t, PERSON_IOU)
        return [all_boxes[i] for i in keep.tolist()]
    except ImportError:
        # Greedy NMS fallback (no torchvision)
        kept: List[Tuple] = []
        for box in sorted(all_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True):
            if not any(box_iou(box, k) > PERSON_IOU for k in kept):
                kept.append(box)
        return kept


def detect_persons(frame: np.ndarray, device) -> List[Tuple]:
    if not model_config:
        return []

    def _run(imgsz: int, conf: float, augment: bool) -> List[Tuple]:
        use_aug = augment and imgsz <= PERSON_AUGMENT_MAX_SCALE
        res = model_config["person_model"].predict(
            source=frame, imgsz=imgsz, conf=conf, iou=PERSON_IOU,
            classes=model_config["person_class_ids"],
            device=device, augment=use_aug, agnostic_nms=True, verbose=False,
        )
        out: List[Tuple] = []
        if res and len(res[0].boxes) > 0:
            for b in res[0].boxes.xyxy.cpu().numpy():
                out.append(tuple(float(v) for v in b))
        return out

    imgsz = optimal_imgsz(frame, PERSON_IMAGE_SIZE_MAX)
    boxes = _run(imgsz, PERSON_CONFIDENCE, augment=True)
    debug(f"[PERSON] Primary: {len(boxes)} person(s) at imgsz={imgsz}")

    # Standard higher-res fallback (when few persons found)
    if PERSON_FALLBACK_ENABLED and len(boxes) < PERSON_FALLBACK_THRESHOLD:
        fb_imgsz = min(imgsz * 2, 3840)
        fb_conf  = max(PERSON_CONFIDENCE - PERSON_FALLBACK_CONF_DELTA, 0.05)
        added = 0
        for fb in _run(fb_imgsz, fb_conf, augment=False):
            if not any(box_iou(fb, pb) > 0.50 for pb in boxes):
                boxes.append(fb)
                added += 1
        if added:
            print(f"[PERSON] Fallback found {added} more (total={len(boxes)})", flush=True)

    # F2 — tiled inference (fires when frame is large OR few persons found)
    if ENABLE_TILED_PERSON_DETECTION:
        fh, fw = frame.shape[:2]
        TILING_AREA_THRESHOLD = 1280 * 720
        use_tiling = (fw * fh > TILING_AREA_THRESHOLD) or (len(boxes) < PERSON_FALLBACK_THRESHOLD)
        if use_tiling:
            tiled = detect_persons_tiled(frame, device)
            added = 0
            for tb in tiled:
                if not any(box_iou(tb, pb) > 0.50 for pb in boxes):
                    boxes.append(tb)
                    added += 1
            if added:
                print(f"[PERSON] Tiling found {added} more (total={len(boxes)})", flush=True)

    # Crowd recovery around already-detected persons
    if ENABLE_CROWD_RECOVERY and boxes:
        boxes = recover_close_persons(frame, boxes, device)

    return filter_person_boxes(dedupe_person_boxes(boxes))


# ───────────────────────────────────────────────────────────────────────────
# Re-ID pipeline (per violating person)
# ───────────────────────────────────────────────────────────────────────────

def reid_for_person(
    frame: np.ndarray,
    person_box: Tuple,
    person_idx: int,
    face_boxes_full_frame: List[Tuple],
    device,
    timestamp: datetime,
    person_box_override: Optional[Tuple] = None,
) -> Dict[str, Any]:
    fh, fw = frame.shape[:2]

    effective_box = person_box_override if person_box_override is not None else person_box

    face_crop, face_box = get_best_face_in_person(frame, effective_box, face_boxes_full_frame)
    face_source = "full_frame"

    if face_crop is None:
        face_crop, face_box = get_face_from_person_crop(frame, effective_box, device)
        if face_crop is not None:
            face_source = "person_crop_fallback"
            face_boxes_full_frame.append(tuple(face_box))

    px1, py1, px2, py2 = clamp_box(effective_box, fw, fh)
    body_crop = frame[py1:py2, px1:px2].copy() if py2 > py1 and px2 > px1 else None

    if face_crop is None or body_crop is None:
        debug(
            f"[REID] person_{person_idx+1:02d}: face/body crop unavailable — skipping re-ID"
        )
        return {
            "person_id": None, "is_new": False, "reid_similarity": 0.0,
            "face_source": "not_detected", "face_box": None,
            "embedding": None, "face_crop": None, "body_crop": body_crop,
        }

    embedding, yaw_deg, mtcnn_ok = extract_embedding(face_crop)

    if embedding is None:
        print(
            f"[REID] person_{person_idx+1:02d}: FaceNet returned no embedding "
            f"(crop {face_crop.shape[1]}×{face_crop.shape[0]}px, "
            f"min={MIN_FACE_SIZE}px required) — skipping re-ID",
            flush=True,
        )
        return {
            "person_id": None, "is_new": False, "reid_similarity": 0.0,
            "face_source": face_source, "face_box": face_box,
            "embedding": None, "face_crop": face_crop, "body_crop": body_crop,
        }

    best_pid, best_sim = id_store.match(embedding)

    if best_pid is not None and id_store.is_match(best_sim):
        id_store.maybe_update_person(
            best_pid, embedding, body_crop, mtcnn_ok, yaw_deg, best_sim
        )
        print(
            f"  [REID] MATCH → person_id={best_pid}  "
            f"sim={best_sim:.4f}  yaw={yaw_deg:.1f}°  "
            f"mtcnn={'ok' if mtcnn_ok else 'fallback'}  "
            f"known={id_store.known_count()}",
            flush=True,
        )
        return {
            "person_id": best_pid, "is_new": False, "reid_similarity": best_sim,
            "face_source": face_source, "face_box": face_box,
            "embedding": embedding, "face_crop": face_crop, "body_crop": body_crop,
        }
    else:
        if body_crop is not None and body_crop.size > 0:
            try:
                new_pid = id_store.insert_new_person(embedding, body_crop, timestamp)
            except Exception as exc:
                print(f"[ERROR] insert_new_person failed: {exc}", flush=True)
                new_pid = None
        else:
            new_pid = None

        if new_pid is not None:
            print(
                f"  [REID] NEW → person_id={new_pid}  "
                f"yaw={yaw_deg:.1f}°  known={id_store.known_count()}",
                flush=True,
            )
        return {
            "person_id": new_pid, "is_new": True, "reid_similarity": 1.0,
            "face_source": face_source, "face_box": face_box,
            "embedding": embedding, "face_crop": face_crop, "body_crop": body_crop,
        }


# ───────────────────────────────────────────────────────────────────────────
# Main detection + re-ID pipeline
# ───────────────────────────────────────────────────────────────────────────

def detect_and_reid(
    frame_metadata_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not frame_metadata_list or not model_config:
        return frame_metadata_list

    device = model_config["device"]
    t0     = time.time()
    print(f"[DETECT] Running on {len(frame_metadata_list)} frames...", flush=True)

    for meta in frame_metadata_list:
        frame     = meta["frame"]
        fh, fw    = frame.shape[:2]
        cam_id    = meta["camera_id"]
        timestamp = meta["timestamp"]

        # ── Step 1: Person detection (primary + fallback + tiling + crowd) ─
        person_boxes = detect_persons(frame, device)
        if not person_boxes:
            action = "suppressed" if PERSON_MATCH_REQUIRED else "kept"
            print(
                f"[WARN] Camera {cam_id}: no persons detected — "
                f"PPE detections will be {action}",
                flush=True,
            )

        # ── Step 2: Full-frame PPE detection ─────────────────────────────
        ppe_imgsz = optimal_imgsz(frame, IMAGE_SIZE_MAX)
        ppe_res   = model_config["ppe_model"].predict(
            source=frame, imgsz=ppe_imgsz,
            conf=MODEL_INFERENCE_CONFIDENCE, iou=DETECTION_IOU,
            classes=model_config["violation_class_ids"],
            device=device, verbose=False,
        )
        raw_dets: List[Dict] = []
        if ppe_res and len(ppe_res[0].boxes) > 0:
            for box, cls_id, conf in zip(
                ppe_res[0].boxes.xyxy.cpu().numpy(),
                ppe_res[0].boxes.cls.cpu().numpy(),
                ppe_res[0].boxes.conf.cpu().numpy(),
            ):
                cname = model_config["class_names"].get(int(cls_id), str(int(cls_id)))
                cval  = float(conf)
                if not is_allowed_violation_class(cname):
                    continue
                if cval < get_class_confidence_threshold(cname):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                raw_dets.append({
                    "class_name": cname,
                    "confidence": cval,
                    "box":        (x1, y1, x2, y2),
                })

        # ── Step 3: F3 — per-person crop PPE inference ────────────────────
        if ENABLE_CROP_PPE and person_boxes:
            crop_dets = detect_ppe_on_crops(frame, person_boxes, device)
            added_from_crops = 0
            for cd in crop_dets:
                # Only add if not already covered by full-frame detection
                # Use a lower IoU threshold (0.30) so crop-remapped boxes
                # (slightly different scale) don't erase each other
                if not any(
                    cd["class_name"] == d["class_name"]
                    and box_iou(cd["box"], d["box"]) > 0.30
                    for d in raw_dets
                ):
                    raw_dets.append(cd)
                    added_from_crops += 1
            if added_from_crops:
                print(
                    f"[PPE] Camera {cam_id}: crop inference added "
                    f"{added_from_crops} detection(s)",
                    flush=True,
                )

        # ── Step 4: Optional boot crop detection ─────────────────────────
        if ENABLE_BOOT_CROPS:
            raw_dets.extend(detect_boot_violations_for_people(frame, person_boxes, device))

        # ── Step 5: Spatial gating (F4 — overlap-fraction primary) ───────
        raw_dets = assign_detections_to_persons(raw_dets, person_boxes, fw, fh)

        # ── Step 6: Boot color suppression ───────────────────────────────
        raw_dets = suppress_boot_violations_by_color(frame, raw_dets, person_boxes)

        # ── Step 7: Deduplication (F5 — DEDUPE_IOU=0.35) ─────────────────
        raw_dets = dedupe_detections(raw_dets)

        if not raw_dets and not person_boxes:
            meta["detections"]   = []
            meta["persons"]      = {}
            meta["person_boxes"] = []
            meta["face_boxes"]   = []
            continue

        # ── Step 8: Full-frame face detection ────────────────────────────
        face_res = face_model.predict(
            source=frame,
            imgsz=optimal_imgsz(frame, FACE_IMAGE_SIZE),
            conf=FACE_CONFIDENCE,
            iou=0.45,
            device=device,
            verbose=False,
        )[0]
        face_boxes_ff: List[Tuple] = []
        if face_res.boxes is not None:
            for fb in face_res.boxes:
                face_boxes_ff.append(
                    tuple(float(x) for x in fb.xyxy[0].tolist())
                )

        # ── Step 9: Group violations by person index ──────────────────────
        person_to_violations: Dict[int, List[Dict]] = {}
        for det in raw_dets:
            pidx = det.get("person_index", -1)
            person_to_violations.setdefault(pidx, []).append(det)

        # ── Step 10: Re-ID every detected person, plus unmatched violation groups.
        person_reid_results: Dict[int, Dict] = {}
        reid_targets = set(range(len(person_boxes))) | set(person_to_violations.keys())
        for pidx in sorted(reid_targets):
            violations = person_to_violations.get(pidx, [])

            if pidx >= 0 and pidx < len(person_boxes):
                p_box    = person_boxes[pidx]
                override = None
            elif pidx == -1:
                xs1 = min(v["box"][0] for v in violations)
                ys1 = min(v["box"][1] for v in violations)
                xs2 = max(v["box"][2] for v in violations)
                ys2 = max(v["box"][3] for v in violations)
                box_h = max(1.0, ys2 - ys1)
                box_w = max(1.0, xs2 - xs1)
                p_box = (
                    max(0.0, xs1 - box_w * 0.20),
                    max(0.0, ys1 - box_h * 1.20),
                    min(float(fw), xs2 + box_w * 0.20),
                    min(float(fh), ys2 + box_h * 0.30),
                )
                override = p_box
                debug(
                    f"[REID] Camera {cam_id}: bypass violation — "
                    f"synthesised person region {[round(v) for v in p_box]}"
                )
            else:
                continue

            reid = reid_for_person(
                frame, p_box, pidx,
                face_boxes_ff, device, timestamp,
                person_box_override=override,
            )
            person_reid_results[pidx] = reid
            if violations and reid["person_id"] is not None:
                id_store.increment_violations(reid["person_id"], len(violations))

        compliant_dets: List[Dict] = []
        violating_person_boxes = [
            person_boxes[pidx]
            for pidx in person_to_violations
            if pidx >= 0 and pidx < len(person_boxes)
        ]
        for pidx, p_box in enumerate(person_boxes):
            if (p_box[3] - p_box[1]) < NO_VIOLATION_MIN_PERSON_HEIGHT:
                continue
            if person_to_violations.get(pidx):
                continue
            if any(boxes_refer_to_same_person(p_box, v_box) for v_box in violating_person_boxes):
                continue
            compliant_dets.append({
                "class_name": "NO-Violation",
                "confidence": 1.0,
                "box": tuple(float(v) for v in p_box),
                "person_index": pidx,
            })

        meta["detections"]   = raw_dets + compliant_dets
        meta["persons"]      = person_reid_results
        meta["person_boxes"] = person_boxes
        meta["face_boxes"]   = face_boxes_ff

        if meta["detections"] and (DEBUG or DRY_RUN):
            summary = ", ".join(
                f"{d['class_name']}({d['confidence']:.2f})" for d in meta["detections"]
            )
            print(f"[DETECT] Camera {cam_id}: {summary}", flush=True)

    elapsed = time.time() - t0
    total_v = sum(
        1
        for m in frame_metadata_list
        for d in m["detections"]
        if d.get("class_name") != "NO-Violation"
    )
    total_ok = sum(
        1
        for m in frame_metadata_list
        for d in m["detections"]
        if d.get("class_name") == "NO-Violation"
    )
    print(
        f"[DETECT] {total_v} violation(s), {total_ok} no-violation person(s) "
        f"in {elapsed:.2f}s",
        flush=True,
    )
    return frame_metadata_list


# ───────────────────────────────────────────────────────────────────────────
# S3 upload (frames with PPE detections or NO-Violation rows)
# ───────────────────────────────────────────────────────────────────────────

def upload_frames_parallel(
    frame_metadata_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    vframes = [m for m in frame_metadata_list if m["detections"]]
    if not vframes:
        return frame_metadata_list
    if DRY_RUN:
        print(f"[DRY_RUN] Skip S3 frame upload ({len(vframes)} frames)", flush=True)
        return frame_metadata_list

    print(f"[UPLOAD] Uploading {len(vframes)} PPE result frames...", flush=True)
    t0, ok = time.time(), 0

    def _upload(meta: Dict) -> Dict:
        meta["s3_url"] = s3_upload_frame(
            meta["local_path"], meta["camera_id"], meta["timestamp"]
        )
        return meta

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as ex:
        for fut in as_completed({ex.submit(_upload, m): m for m in vframes}):
            if fut.result()["s3_url"]:
                ok += 1

    print(f"[UPLOAD] {ok}/{len(vframes)} in {time.time()-t0:.2f}s", flush=True)
    return frame_metadata_list


# ───────────────────────────────────────────────────────────────────────────
# Welspun DataHub webhook
# ───────────────────────────────────────────────────────────────────────────

def _presign(s3_client_obj, url: Optional[str]) -> Optional[str]:
    """Convert a raw S3 URL to a presigned URL. Falls back to original on error."""
    if not url:
        return url
    try:
        bucket = url.split(".s3.")[0].replace("https://", "")
        key = url.split(".amazonaws.com/", 1)[1]
        return s3_client_obj.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=WELSPUN_PRESIGN_EXPIRY,
        )
    except Exception as exc:
        print(f"[WELSPUN WARN] Presign failed for {url}: {exc}", flush=True)
        return url


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def push_ppe_to_welspun(
    s3_client_obj,
    frame_id: int,
    cam_id: int,
    timestamp: datetime,
    frame_url: str,
    person: Optional[Dict[str, Any]],
    detections: List[Dict[str, Any]],
) -> None:
    """Push one PPE frame/person group to Welspun DataHub. Never raises."""
    if not WELSPUN_WEBHOOK_ENABLED or not detections:
        return
    if not person or person.get("person_id") is None:
        print(
            f"[WELSPUN WARN] Skipping frame_id={frame_id}: missing person_id",
            flush=True,
        )
        return

    payload = {
        "frame": {
            "frame_id": frame_id,
            "cam_id": cam_id,
            "timestamp": _iso_or_none(timestamp),
            "frame_url": _presign(s3_client_obj, frame_url),
        },
        "person": {
            "person_id": person["person_id"],
            "first_seen": _iso_or_none(person.get("first_seen")),
            "total_violations": person.get("total_violations"),
            "crop_s3_url": _presign(s3_client_obj, person.get("crop_s3_url")),
        },
        "detections": [
            {
                "id": d["id"],
                "frame_id": frame_id,
                "person_id": d.get("person_id"),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
            }
            for d in detections
        ],
    }

    try:
        resp = requests.post(
            _WELSPUN_PPE_URL,
            json=payload,
            headers=_WELSPUN_HEADERS,
            timeout=10,
            verify=False,
        )
        if resp.status_code < 300 or resp.status_code == 409:
            print(
                f"[WELSPUN] Pushed frame_id={frame_id} "
                f"person_id={person['person_id']} status={resp.status_code}",
                flush=True,
            )
        else:
            print(
                "[WELSPUN WARN] Rejected "
                f"frame_id={frame_id} person_id={person['person_id']} "
                f"status={resp.status_code} body={resp.text[:300]}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[WELSPUN ERROR] Push failed frame_id={frame_id} "
            f"person_id={person['person_id']}: {exc}",
            flush=True,
        )


# ───────────────────────────────────────────────────────────────────────────
# Database persistence
# ───────────────────────────────────────────────────────────────────────────

def save_to_db(frame_metadata_list: List[Dict[str, Any]]) -> None:
    if DRY_RUN:
        n = sum(len(m["detections"]) for m in frame_metadata_list if m["detections"])
        print(f"[DRY_RUN] Skip DB save ({n} detections)", flush=True)
        return

    to_save = [m for m in frame_metadata_list if m["detections"] and m.get("s3_url")]
    if not to_save:
        print("[DB] No violations to save", flush=True)
        return

    reconnect_db()
    ft = aws_creds["ppe_frames_table"]
    dt = aws_creds["ppe_detections_table"]

    saved_frames = 0
    saved_detections = 0

    for meta in to_save:
        frame_id = None
        webhook_by_person: Dict[int, List[Dict[str, Any]]] = {}
        webhook_people: Dict[int, Dict[str, Any]] = {}
        try:
            cam_id = int(meta["camera_id"])
            timestamp = meta["timestamp"]
            frame_url = meta["s3_url"]
            person_reid = meta.get("persons", {})

            with db_connection.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {ft} (cam_id, timestamp, frame_url) "
                    f"VALUES (%s, %s, %s)",
                    (
                        cam_id,
                        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        frame_url,
                    ),
                )
                frame_id = cur.lastrowid

                for det in meta["detections"]:
                    pidx = det.get("person_index", -1)
                    reid = person_reid.get(pidx, {})
                    person_id = reid.get("person_id")
                    webhook_person_id = person_id
                    if webhook_person_id is None:
                        webhook_person_id = pidx if isinstance(pidx, int) and pidx >= 0 else -1
                    x1, y1, x2, y2 = det["box"]
                    cur.execute(
                        f"INSERT INTO {dt} "
                        f"(frame_id, class_name, confidence, x1, y1, x2, y2, person_id) "
                        f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            frame_id,
                            det["class_name"],
                            det["confidence"],
                            x1, y1, x2, y2,
                            person_id,
                        ),
                    )
                    webhook_det = {
                        "id": cur.lastrowid,
                        "frame_id": frame_id,
                        "person_id": webhook_person_id,
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                    webhook_key = int(webhook_person_id)
                    webhook_by_person.setdefault(webhook_key, []).append(webhook_det)
                    if webhook_key not in webhook_people:
                        if person_id is not None and id_store:
                            webhook_people[webhook_key] = id_store.metadata_for(person_id)
                        else:
                            webhook_people[webhook_key] = {
                                "person_id": webhook_key,
                                "first_seen": timestamp,
                                "total_violations": None,
                                "crop_s3_url": None,
                            }

            db_connection.commit()
            saved_frames += 1
            saved_detections += len(meta["detections"])

            for person_id, detections in webhook_by_person.items():
                person = webhook_people.get(person_id, {"person_id": person_id})
                push_ppe_to_welspun(
                    s3_client_obj=s3_client,
                    frame_id=frame_id,
                    cam_id=cam_id,
                    timestamp=timestamp,
                    frame_url=frame_url,
                    person=person,
                    detections=detections,
                )
        except Exception as exc:
            db_connection.rollback()
            print(f"[DB ERROR] frame_id={frame_id}: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    print(
        f"[DB] Saved {saved_frames} frame(s), {saved_detections} detection(s)",
        flush=True,
    )


# ───────────────────────────────────────────────────────────────────────────
# Local frame cleanup
# ───────────────────────────────────────────────────────────────────────────

def cleanup_local_frames(frame_metadata_list: List[Dict[str, Any]]) -> None:
    for meta in frame_metadata_list:
        path = meta.get("local_path")
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            debug(f"[CLEANUP] Could not delete {path}: {exc}")


# ───────────────────────────────────────────────────────────────────────────
# Snapshot processing loop
# ───────────────────────────────────────────────────────────────────────────

def process_snapshot_set() -> None:
    t0 = time.time()
    print("\n" + "=" * 60, flush=True)
    print(
        f"[SET] Snapshot at {get_timestamp().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    print("=" * 60, flush=True)

    captured = capture_all_cameras_parallel()
    if not captured:
        print("[SET] No frames captured", flush=True)
        return

    captured = detect_and_reid(captured)
    captured = upload_frames_parallel(captured)
    save_to_db(captured)
    cleanup_local_frames(captured)

    print("=" * 60, flush=True)
    print(f"[SET] Done in {time.time()-t0:.2f}s", flush=True)
    print("=" * 60, flush=True)


# ───────────────────────────────────────────────────────────────────────────
# Initialisation
# ───────────────────────────────────────────────────────────────────────────

def initialize() -> bool:
    global aws_creds, s3_client, kvs_client, db_connection, id_store

    print("=" * 60, flush=True)
    print("OfficeLens PPE + FaceNet Re-ID Worker  [v4]", flush=True)
    print("=" * 60, flush=True)

    try:
        aws_creds     = load_credentials()
        s3_client     = init_s3_client()
        kvs_client    = init_kvs_client()
        db_connection = init_db_connection()
        init_models()
    except Exception as exc:
        print(f"[ERROR] Init failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return False

    id_store = IdentityStore()
    try:
        id_store.load()
    except Exception as exc:
        print(f"[WARN] Identity store load failed (starting empty): {exc}", flush=True)

    if not load_configured_cameras():
        print("[ERROR] No active cameras found", flush=True)
        return False

    W = 44
    def p(label, val): print(f"[INIT] {label:<{W}}: {val}", flush=True)

    p("Snapshot interval",              f"{PPE_SNAPSHOT_INTERVAL}s")
    p("Cameras active",                 len(camera_sources))
    p("Known persons (loaded)",         id_store.known_count())
    p("--- Detection fixes ---",        "")
    p("Frame quality score (F1)",       "enabled (sharpness × contrast)")
    p("Tiled person detection (F2)",    ENABLE_TILED_PERSON_DETECTION)
    p("  Tile size / overlap",          f"{TILING_SIZE}px / {TILING_OVERLAP:.0%}")
    p("Crop PPE inference (F3)",        ENABLE_CROP_PPE)
    p("  Crop inference size",          f"{PPE_CROP_INFERENCE_SIZE}px")
    p("  Crop margin",                  f"{PPE_CROP_MARGIN:.0%}")
    p("Overlap gate threshold (F4)",    f"{PPE_OVERLAP_GATE_THRESHOLD:.0%}")
    p("Dedup IoU (F5, lowered)",        DETECTION_DEDUPE_IOU)
    p("Boot crop imgsz fixed 640 (F6)", "enabled")
    p("Perceptual frame dedup (F7)",    "enabled")
    p("--- Re-ID ---",                  "")
    p("REID threshold",                 REID_THRESHOLD)
    p("Max prototypes / person",        MAX_PROTOTYPES)
    p("Frontal yaw threshold",          f"{FRONTAL_YAW_THRESHOLD}°")
    p("Min face size",                  f"{MIN_FACE_SIZE}px")
    p("--- Gating ---",                 "")
    p("Person match required",          PERSON_MATCH_REQUIRED)
    p("Person box expand (T/B/S)",      f"{PERSON_BOX_EXPAND_TOP}/{PERSON_BOX_EXPAND_BOTTOM}/{PERSON_BOX_EXPAND_SIDES}")
    p("Boot crop detection",            ENABLE_BOOT_CROPS)
    p("Boot color check",               ENABLE_BOOT_COLOR_CHECK)
    p("Crowd recovery",                 ENABLE_CROWD_RECOVERY)
    p("--- Misc ---",                   "")
    p("Dry run",                        DRY_RUN)
    p("S3 bucket",                      aws_creds["bucket"])
    p("S3 frames prefix",               S3_FRAMES_PREFIX)
    p("S3 embeddings prefix",           S3_EMBEDDINGS_PREFIX)
    p("S3 crops prefix",                S3_CROPS_PREFIX)

    return True


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def main() -> int:
    global running

    if not initialize():
        return 1

    print("\n[RUNNING] PPE Re-ID worker is active\n", flush=True)

    try:
        while running:
            t0 = time.time()
            try:
                refresh_cameras_if_needed()
                if camera_sources:
                    process_snapshot_set()
                else:
                    print("[WAIT] No active PPE cameras configured", flush=True)
            except Exception as exc:
                print(f"[ERROR] Snapshot cycle failed: {exc}", flush=True)
                import traceback
                traceback.print_exc()

            elapsed   = time.time() - t0
            wait_time = max(0.0, PPE_SNAPSHOT_INTERVAL - elapsed)
            if wait_time > 0:
                print(f"[WAIT] Next snapshot in {wait_time:.0f}s\n", flush=True)
                time.sleep(wait_time)
            else:
                print(
                    f"[WARN] Cycle took {elapsed:.0f}s > "
                    f"interval {PPE_SNAPSHOT_INTERVAL}s\n",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...", flush=True)
        running = False
    finally:
        if db_connection:
            try:
                db_connection.close()
            except Exception:
                pass

    print("[EXIT] Worker stopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
