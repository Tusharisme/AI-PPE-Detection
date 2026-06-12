# #!/usr/bin/env python3
# """
# Local KVS / Video PPE + Face Detection + FaceNet Re-Identification.

# Pipeline:
# 1. Capture frame from KVS stream or local video file (--video)
# 2. Detect persons using person YOLO model
# 3. Detect PPE violations using PPE YOLO model
# 4. For persons with violations:
#    a. Detect faces using YOLOv8-face (bounding box)
#    b. Align face crop using MTCNN (for FaceNet accuracy)
#    c. Generate 512-d embedding using FaceNet (InceptionResnetV1)
#    d. Match embedding against stored person embeddings (cosine similarity)
#    e. Assign existing person_id if match found, else create new person_id
#    f. Update rolling mean embedding for that person
# 5. Save annotated frames, face crops, embeddings, and metadata CSV

# Storage layout under --save-dir (default: violations/):
#   annotated_frames/          - frames with drawn bounding boxes
#   violator_face_crops/       - raw face crop images per violation instance
#   embeddings/
#     person_XXXX.npy          - rolling mean 512-d FaceNet embedding
#     person_XXXX_meta.json    - identity metadata (total violations, history)
#   metadata.csv               - one row per violation instance, indexed on person_id

# Re-identification:
#   - Cosine similarity between new embedding and all stored mean embeddings
#   - Match threshold: --reid-threshold (default 0.75)
#   - Stored embedding updated as rolling mean across all matched crops
# """

# import argparse
# import configparser
# import csv
# import json
# import os
# import sys
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from ultralytics import YOLO

# # ---------------------------------------------------------------------------
# # FaceNet imports (facenet-pytorch)
# # ---------------------------------------------------------------------------
# try:
#     from facenet_pytorch import InceptionResnetV1, MTCNN
# except ImportError:
#     print("[ERROR] facenet-pytorch is not installed.")
#     print("        Install with:  pip install facenet-pytorch")
#     sys.exit(1)


# # ===========================================================================
# # AWS / KVS helpers
# # ===========================================================================

# def load_aws_config(creds_file: str) -> dict:
#     path = Path(creds_file)
#     if not path.exists():
#         raise FileNotFoundError(f"Credentials file not found: {path}")
#     config = configparser.ConfigParser()
#     config.read(path)
#     if "AWS" not in config:
#         raise ValueError("Missing [AWS] section in credentials file")
#     aws = config["AWS"]
#     values = {
#         "access_key": aws.get("aws_access_key_id"),
#         "secret_key": aws.get("aws_secret_access_key"),
#         "region_name": aws.get("region_name"),
#     }
#     missing = [k for k, v in values.items() if not v]
#     if missing:
#         raise ValueError(f"Missing required AWS values: {', '.join(missing)}")
#     return values


# def get_hls_url(creds: dict, stream_name: str) -> str:
#     import boto3
#     kvs = boto3.client(
#         "kinesisvideo",
#         aws_access_key_id=creds["access_key"],
#         aws_secret_access_key=creds["secret_key"],
#         region_name=creds["region_name"],
#     )
#     kvs.describe_stream(StreamName=stream_name)
#     endpoint = kvs.get_data_endpoint(
#         StreamName=stream_name,
#         APIName="GET_HLS_STREAMING_SESSION_URL",
#     )["DataEndpoint"]
#     archived = boto3.client(
#         "kinesis-video-archived-media",
#         endpoint_url=endpoint,
#         aws_access_key_id=creds["access_key"],
#         aws_secret_access_key=creds["secret_key"],
#         region_name=creds["region_name"],
#     )
#     return archived.get_hls_streaming_session_url(
#         StreamName=stream_name,
#         PlaybackMode="LIVE",
#         HLSFragmentSelector={"FragmentSelectorType": "SERVER_TIMESTAMP"},
#         Expires=300,
#     )["HLSStreamingSessionURL"]


# # ===========================================================================
# # Geometry helpers
# # ===========================================================================

# def box_iou(
#     box_a: Tuple[float, float, float, float],
#     box_b: Tuple[float, float, float, float],
# ) -> float:
#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#     ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
#     inter = iw * ih
#     if inter <= 0:
#         return 0.0
#     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
#     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
#     union = area_a + area_b - inter
#     return inter / union if union > 0 else 0.0


# def box_center_inside(
#     inner: Tuple[float, float, float, float],
#     outer: Tuple[float, float, float, float],
# ) -> bool:
#     x1, y1, x2, y2 = inner
#     ox1, oy1, ox2, oy2 = outer
#     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#     return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


# def clamp_box(
#     box: Tuple[float, float, float, float],
#     frame_w: int,
#     frame_h: int,
# ) -> Tuple[int, int, int, int]:
#     x1, y1, x2, y2 = box
#     return (
#         max(0, min(frame_w - 1, int(round(x1)))),
#         max(0, min(frame_h - 1, int(round(y1)))),
#         max(0, min(frame_w, int(round(x2)))),
#         max(0, min(frame_h, int(round(y2)))),
#     )


# def format_box(box) -> str:
#     return f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"


# def person_face_search_region(
#     person_box: Tuple[float, float, float, float],
#     frame_w: int,
#     frame_h: int,
#     height_ratio: float,
#     margin_ratio: float,
# ) -> Tuple[int, int, int, int]:
#     px1, py1, px2, py2 = person_box
#     pw = max(1.0, px2 - px1)
#     ph = max(1.0, py2 - py1)
#     region = (
#         px1 - pw * margin_ratio,
#         py1 - ph * margin_ratio * 0.5,
#         px2 + pw * margin_ratio,
#         py1 + ph * height_ratio,
#     )
#     return clamp_box(region, frame_w, frame_h)


# # ===========================================================================
# # PPE helpers
# # ===========================================================================

# def is_ppe_violation_class(class_name: str) -> bool:
#     return "NO" in class_name.upper()


# def detection_to_dict(box, class_names: dict, model_name: str) -> dict:
#     class_id = int(box.cls[0])
#     class_name = class_names.get(class_id, str(class_id))
#     return {
#         "model": model_name,
#         "class_id": class_id,
#         "class_name": class_name,
#         "is_violation": is_ppe_violation_class(class_name),
#         "confidence": round(float(box.conf[0]), 6),
#         "box": [round(float(v), 2) for v in box.xyxy[0].tolist()],
#     }


# def find_person_for_violation(
#     violation_box: Tuple[float, float, float, float],
#     person_boxes: List[Tuple[float, float, float, float]],
# ) -> int:
#     if not person_boxes:
#         return -1
#     containing = []
#     for i, pb in enumerate(person_boxes):
#         if box_center_inside(violation_box, pb):
#             containing.append((i, box_iou(violation_box, pb)))
#     if containing:
#         return max(containing, key=lambda x: x[1])[0]
#     best_idx, best_iou = -1, 0.0
#     for i, pb in enumerate(person_boxes):
#         iou = box_iou(violation_box, pb)
#         if iou > best_iou:
#             best_iou, best_idx = iou, i
#     return best_idx if best_iou > 0.1 else -1


# # ===========================================================================
# # YOLO face detection helpers (unchanged from original)
# # ===========================================================================

# def crop_face_from_person(
#     frame,
#     person_box: Tuple[float, float, float, float],
#     face_boxes: List[Tuple[float, float, float, float]],
# ) -> Tuple[Any, Optional[Tuple[float, float, float, float]]]:
#     """Pick the best YOLO-detected face whose centre is inside the person box."""
#     if not face_boxes:
#         return None, None
#     px1, py1, px2, py2 = person_box
#     person_h = max(1.0, py2 - py1)
#     frame_h, frame_w = frame.shape[:2]
#     best_face, best_score = None, -1.0
#     for fb in face_boxes:
#         fx1, fy1, fx2, fy2 = fb
#         cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
#         if not (px1 <= cx <= px2 and py1 <= cy <= py2):
#             continue
#         area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
#         score = area + max(0.0, 1.0 - (cy - py1) / person_h) * 1000.0
#         if score > best_score:
#             best_score, best_face = score, fb
#     if best_face is None:
#         return None, None
#     x1, y1, x2, y2 = clamp_box(best_face, frame_w, frame_h)
#     if x2 <= x1 or y2 <= y1:
#         return None, None
#     crop = frame[y1:y2, x1:x2]
#     return (None, None) if crop.size == 0 else (crop, best_face)


# def detect_face_from_person_crop(
#     frame,
#     person_box: Tuple[float, float, float, float],
#     face_model,
#     device,
#     image_size: int,
#     confidence: float,
#     iou: float,
#     height_ratio: float,
#     margin_ratio: float,
# ) -> Tuple[Any, Optional[Tuple[float, float, float, float]]]:
#     """Fallback: run YOLO-face on violator's upper-body crop."""
#     frame_h, frame_w = frame.shape[:2]
#     rx1, ry1, rx2, ry2 = person_face_search_region(
#         person_box, frame_w, frame_h, height_ratio, margin_ratio
#     )
#     if rx2 <= rx1 or ry2 <= ry1:
#         return None, None
#     region = frame[ry1:ry2, rx1:rx2]
#     if region.size == 0:
#         return None, None
#     results = face_model.predict(
#         source=region,
#         imgsz=image_size,
#         conf=confidence,
#         iou=iou,
#         device=device,
#         verbose=False,
#     )[0]
#     if results.boxes is None or len(results.boxes) == 0:
#         return None, None
#     best_box, best_conf = None, -1.0
#     for box in results.boxes:
#         conf = float(box.conf[0])
#         if conf > best_conf:
#             lx1, ly1, lx2, ly2 = [float(x) for x in box.xyxy[0].tolist()]
#             best_box = (lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1)
#             best_conf = conf
#     if best_box is None:
#         return None, None
#     x1, y1, x2, y2 = clamp_box(best_box, frame_w, frame_h)
#     if x2 <= x1 or y2 <= y1:
#         return None, None
#     crop = frame[y1:y2, x1:x2]
#     return (None, None) if crop.size == 0 else (crop, best_box)


# # ===========================================================================
# # FaceNet embedding helpers
# # ===========================================================================

# # Minimum pixel dimensions of a face crop for FaceNet to be reliable
# _FACENET_MIN_SIZE = 40  # pixels (width or height)
# # FaceNet expects 160x160 aligned input
# _FACENET_INPUT_SIZE = 160


# def build_facenet(device) -> Tuple[Any, Any]:
#     """
#     Returns (mtcnn, resnet).

#     MTCNN: used to align & re-crop from the coarse YOLO face crop.
#            keep_all=False → returns the single most prominent face.
#     InceptionResnetV1: pretrained on VGGFace2, outputs 512-d unit-norm embeddings.
#     """
#     mtcnn = MTCNN(
#         image_size=_FACENET_INPUT_SIZE,
#         margin=14,            # pixels of context around the face
#         min_face_size=20,
#         thresholds=[0.6, 0.7, 0.7],   # P-Net, R-Net, O-Net detection thresholds
#         factor=0.709,
#         post_process=True,    # normalise to [-1, 1] for InceptionResnetV1
#         keep_all=False,
#         select_largest=True,  # when multiple detections, take the largest
#         device=device,
#     )
#     resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
#     return mtcnn, resnet


# def extract_embedding(
#     face_crop_bgr: np.ndarray,
#     mtcnn,
#     resnet,
#     device,
# ) -> Optional[np.ndarray]:
#     """
#     Given a BGR face crop (from YOLO), return a 512-d unit-norm embedding.

#     Flow:
#       1. Reject crops that are too small (unreliable for FaceNet).
#       2. Convert BGR → RGB PIL Image.
#       3. Run MTCNN to align: this handles rotation, scale, and landmark
#          alignment that makes FaceNet embeddings much more stable.
#       4. If MTCNN finds no face (happens on very blurry/partial crops),
#          fall back to a direct resize + normalise so we still produce
#          *some* embedding rather than silently dropping the detection.
#       5. Run InceptionResnetV1 to get the 512-d embedding.
#       6. L2-normalise (resnet already does this if pretrained, but
#          we enforce it explicitly for safety).
#     """
#     h, w = face_crop_bgr.shape[:2]
#     if h < _FACENET_MIN_SIZE or w < _FACENET_MIN_SIZE:
#         return None

#     # BGR → RGB PIL
#     pil_img = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

#     # --- MTCNN alignment pass ---
#     aligned = mtcnn(pil_img)  # returns Tensor[3,160,160] or None

#     if aligned is None:
#         # Fallback: resize + manual normalise to [-1,1]
#         resized = pil_img.resize((_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE), Image.BILINEAR)
#         arr = np.array(resized, dtype=np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
#         aligned = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)  # CHW

#     # Ensure shape is [3, H, W] and add batch dim
#     if aligned.dim() == 3:
#         aligned = aligned.unsqueeze(0)  # [1, 3, H, W]

#     aligned = aligned.to(device)

#     with torch.no_grad():
#         embedding = resnet(aligned)          # [1, 512]
#         embedding = F.normalize(embedding, p=2, dim=1)

#     return embedding.squeeze(0).cpu().numpy()  # (512,)


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """Cosine similarity between two L2-normalised vectors."""
#     # Both should already be unit-norm, but normalise defensively
#     a = a / (np.linalg.norm(a) + 1e-8)
#     b = b / (np.linalg.norm(b) + 1e-8)
#     return float(np.dot(a, b))


# # ===========================================================================
# # Identity store  (in-memory + disk)
# # ===========================================================================

# class IdentityStore:
#     """
#     Manages person identities: embeddings, violation counts, metadata.

#     On-disk layout inside `embeddings_dir`:
#       person_0001.npy            – rolling mean 512-d embedding
#       person_0001_meta.json      – {person_id, first_seen, total_violations,
#                                     embedding_count, instances:[...]}
#     """

#     def __init__(self, embeddings_dir: Path, reid_threshold: float = 0.75):
#         self.embeddings_dir = embeddings_dir
#         self.embeddings_dir.mkdir(parents=True, exist_ok=True)
#         self.threshold = reid_threshold

#         # In-memory cache: person_id (str) → mean embedding (np.ndarray)
#         self._embeddings: Dict[str, np.ndarray] = {}
#         # person_id → metadata dict
#         self._meta: Dict[str, dict] = {}

#         self._load_from_disk()

#     # ------------------------------------------------------------------
#     # Disk I/O
#     # ------------------------------------------------------------------

#     def _person_npy_path(self, person_id: str) -> Path:
#         return self.embeddings_dir / f"{person_id}.npy"

#     def _person_meta_path(self, person_id: str) -> Path:
#         return self.embeddings_dir / f"{person_id}_meta.json"

#     def _load_from_disk(self) -> None:
#         """Load all existing embeddings and metadata from disk into memory."""
#         for npy_file in sorted(self.embeddings_dir.glob("person_*.npy")):
#             person_id = npy_file.stem  # e.g. "person_0001"
#             emb = np.load(str(npy_file))
#             self._embeddings[person_id] = emb

#             meta_path = self._person_meta_path(person_id)
#             if meta_path.exists():
#                 with open(meta_path, "r", encoding="utf-8") as f:
#                     self._meta[person_id] = json.load(f)
#             else:
#                 # Reconstruct minimal metadata if missing
#                 self._meta[person_id] = {
#                     "person_id": person_id,
#                     "first_seen": "unknown",
#                     "total_violations": 0,
#                     "embedding_count": 1,
#                     "instances": [],
#                 }
#         print(f"[REID] Loaded {len(self._embeddings)} known identities from disk.")

#     def _save_embedding(self, person_id: str) -> None:
#         np.save(str(self._person_npy_path(person_id)), self._embeddings[person_id])

#     def _save_meta(self, person_id: str) -> None:
#         with open(self._person_meta_path(person_id), "w", encoding="utf-8") as f:
#             json.dump(self._meta[person_id], f, indent=2, default=str)

#     # ------------------------------------------------------------------
#     # Identity management
#     # ------------------------------------------------------------------

#     def _next_person_id(self) -> str:
#         """Generate the next sequential person_id string."""
#         existing = [
#             int(pid.split("_")[1])
#             for pid in self._embeddings.keys()
#             if pid.startswith("person_") and pid.split("_")[1].isdigit()
#         ]
#         next_num = (max(existing) + 1) if existing else 1
#         return f"person_{next_num:04d}"

#     def match_or_create(
#         self,
#         embedding: np.ndarray,
#         timestamp: str,
#     ) -> Tuple[str, bool, float]:
#         """
#         Match `embedding` against all stored identities.

#         Returns:
#             (person_id, is_new, best_similarity)
#             is_new=True  → first time this person has been seen
#             is_new=False → matched an existing identity
#         """
#         best_pid, best_sim = None, -1.0

#         for pid, stored_emb in self._embeddings.items():
#             sim = cosine_similarity(embedding, stored_emb)
#             if sim > best_sim:
#                 best_sim, best_pid = sim, pid

#         if best_pid is not None and best_sim >= self.threshold:
#             return best_pid, False, best_sim

#         # --- New identity ---
#         new_pid = self._next_person_id()
#         self._embeddings[new_pid] = embedding.copy()
#         self._meta[new_pid] = {
#             "person_id": new_pid,
#             "first_seen": timestamp,
#             "total_violations": 0,
#             "embedding_count": 1,
#             "instances": [],
#         }
#         self._save_embedding(new_pid)
#         self._save_meta(new_pid)
#         return new_pid, True, 1.0  # perfect self-similarity

#     def update_embedding(self, person_id: str, new_embedding: np.ndarray) -> None:
#         """
#         Update stored embedding using rolling mean.
#         new_mean = (old_mean * n + new_emb) / (n + 1), then L2-normalise.
#         """
#         meta = self._meta[person_id]
#         n = meta["embedding_count"]
#         old_emb = self._embeddings[person_id]
#         mean_emb = (old_emb * n + new_embedding) / (n + 1)
#         # L2-normalise so cosine similarity stays meaningful
#         norm = np.linalg.norm(mean_emb)
#         if norm > 1e-8:
#             mean_emb = mean_emb / norm
#         self._embeddings[person_id] = mean_emb
#         meta["embedding_count"] = n + 1
#         self._save_embedding(person_id)

#     def record_violation(
#         self,
#         person_id: str,
#         instance: dict,
#     ) -> int:
#         """
#         Append a violation instance to the person's metadata.
#         Returns updated total_violations count.
#         """
#         meta = self._meta[person_id]
#         meta["total_violations"] += len(instance.get("violations_list", []))
#         meta["instances"].append(instance)
#         self._save_meta(person_id)
#         return meta["total_violations"]

#     def total_violations(self, person_id: str) -> int:
#         return self._meta.get(person_id, {}).get("total_violations", 0)

#     def known_count(self) -> int:
#         return len(self._embeddings)


# # ===========================================================================
# # Drawing
# # ===========================================================================

# def draw_detections(
#     frame,
#     ppe_detections: list,
#     person_boxes: List = None,
#     face_boxes: List = None,
#     person_labels: Dict[int, str] = None,
# ) -> None:
#     for det in ppe_detections:
#         x1, y1, x2, y2 = [int(v) for v in det["box"]]
#         color = (0, 0, 255) if det["is_violation"] else (0, 160, 0)
#         label = f"{det['class_name']} {det['confidence']:.2f}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, max(20, y1 - 8)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     if person_boxes:
#         for i, box in enumerate(person_boxes):
#             x1, y1, x2, y2 = [int(c) for c in box]
#             lbl = (person_labels or {}).get(i, f"Person {i+1}")
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
#             cv2.putText(frame, lbl, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#     if face_boxes:
#         for box in face_boxes:
#             x1, y1, x2, y2 = [int(c) for c in box]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             cv2.putText(frame, "Face", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# # ===========================================================================
# # Main
# # ===========================================================================

# def main() -> int:
#     script_dir = Path(__file__).parent
#     parent_dir = script_dir.parent

#     parser = argparse.ArgumentParser(
#         description="PPE + Face Detection + FaceNet Re-Identification against KVS or local video"
#     )
#     # Source — mutually exclusive: KVS stream OR local video
#     source_group = parser.add_mutually_exclusive_group(required=True)
#     source_group.add_argument("--stream-name", help="KVS stream name")
#     source_group.add_argument("--video", help="Path to a local MP4 video file")

#     # AWS credentials (only needed for KVS)
#     parser.add_argument(
#         "--creds",
#         default=str(parent_dir / "ppe_creds.txt"),
#         help="Path to PPE credentials INI file (required for --stream-name)",
#     )

#     # Model paths
#     parser.add_argument(
#         "--person-model",
#         default=str(parent_dir / "runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt"),
#         help="Person detection YOLO model path",
#     )
#     parser.add_argument(
#         "--ppe-model",
#         default=str(parent_dir / "runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt"),
#         help="PPE detection YOLO model path",
#     )
#     parser.add_argument(
#         "--face-model",
#         default=str(script_dir / "yolov8n-face.pt"),
#         help="Face detection YOLO model path (bounding box stage)",
#     )

#     # Frame / processing
#     parser.add_argument("--frames", type=int, default=10, help="Number of frames to process")
#     parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for readable frames (KVS only)")
#     parser.add_argument("--violations-only", action="store_true", help="Process only frames containing violations")

#     # YOLO inference
#     parser.add_argument("--confidence", type=float, default=0.25, help="PPE YOLO confidence threshold")
#     parser.add_argument("--person-confidence", type=float, default=0.33, help="Person detection confidence")
#     parser.add_argument("--face-confidence", type=float, default=0.25, help="Full-frame face detection confidence")
#     parser.add_argument("--face-crop-confidence", type=float, default=0.15, help="Fallback face detection confidence on violator person crops")
#     parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
#     parser.add_argument("--image-size", type=int, default=640, help="Person/PPE YOLO inference image size")
#     parser.add_argument("--face-image-size", type=int, default=1280, help="Full-frame face model inference image size")
#     parser.add_argument("--face-crop-image-size", type=int, default=960, help="Person-crop face model inference image size")
#     parser.add_argument("--face-region-height-ratio", type=float, default=0.65, help="Fraction of person height for fallback face search")
#     parser.add_argument("--face-region-margin", type=float, default=0.20, help="Side margin around person box for fallback face search")

#     # Re-identification
#     parser.add_argument(
#         "--reid-threshold",
#         type=float,
#         default=0.75,
#         help="Cosine similarity threshold for FaceNet re-identification (0–1, higher = more lenient)",
#     )

#     # Output
#     parser.add_argument("--save-dir", default=str(script_dir / "violations"), help="Root directory for all results")

#     args = parser.parse_args()

#     # ------------------------------------------------------------------
#     # Validate model files
#     # ------------------------------------------------------------------
#     model_paths = [
#         ("person", Path(args.person_model)),
#         ("ppe", Path(args.ppe_model)),
#         ("face", Path(args.face_model)),
#     ]
#     for model_name, model_path in model_paths:
#         if not model_path.exists():
#             print(f"[ERROR] {model_name} model not found: {model_path.resolve()}")
#             return 1

#     # ------------------------------------------------------------------
#     # Validate video file if provided
#     # ------------------------------------------------------------------
#     if args.video:
#         video_path = Path(args.video)
#         if not video_path.exists():
#             print(f"[ERROR] Video file not found: {video_path.resolve()}")
#             return 1

#     # ------------------------------------------------------------------
#     # Setup output directories
#     # ------------------------------------------------------------------
#     save_dir = Path(args.save_dir)
#     crops_dir = save_dir / "violator_face_crops"
#     frames_dir = save_dir / "annotated_frames"
#     embeddings_dir = save_dir / "embeddings"

#     for d in (save_dir, crops_dir, frames_dir, embeddings_dir):
#         d.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------------------------
#     # Setup metadata CSV
#     # ------------------------------------------------------------------
#     metadata_file = save_dir / "metadata.csv"
#     csv_fields = [
#         "person_id",
#         "is_new_person",
#         "reid_similarity",
#         "timestamp",
#         "frame_id",
#         "frame_path",
#         "face_crop_path",
#         "face_box",
#         "face_detection_source",
#         "person_box",
#         "person_detection_confidence",
#         "violations",
#         "violation_boxes",
#         "violation_confidences",
#         "violation_count_this_instance",
#         "total_violations_cumulative",
#         "person_count_in_frame",
#         "face_count_in_frame",
#         "embedding_path",
#     ]
#     # Append mode so re-runs accumulate history
#     csv_file = open(metadata_file, "a", newline="", encoding="utf-8")
#     write_header = csv_file.tell() == 0
#     csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
#     if write_header:
#         csv_writer.writeheader()

#     try:
#         # ------------------------------------------------------------------
#         # Load models
#         # ------------------------------------------------------------------
#         print("[MODEL] Loading YOLO models...")
#         device = 0 if torch.cuda.is_available() else "cpu"
#         print(f"[MODEL] Inference device: {device}")

#         person_model = YOLO(str(args.person_model))
#         ppe_model = YOLO(str(args.ppe_model))
#         face_model = YOLO(str(args.face_model))

#         person_classes = {int(k): str(v) for k, v in person_model.names.items()}
#         ppe_classes = {int(k): str(v) for k, v in ppe_model.names.items()}

#         print(f"[MODEL] Person classes: {person_classes}")
#         print(f"[MODEL] PPE classes: {ppe_classes}")

#         print("[MODEL] Loading FaceNet (MTCNN + InceptionResnetV1)...")
#         # FaceNet runs on CPU if CUDA not available; it's lightweight enough
#         facenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         mtcnn, resnet = build_facenet(facenet_device)
#         print(f"[MODEL] FaceNet device: {facenet_device}")

#         # ------------------------------------------------------------------
#         # Load identity store
#         # ------------------------------------------------------------------
#         id_store = IdentityStore(embeddings_dir, reid_threshold=args.reid_threshold)
#         print(f"[REID] Re-ID threshold: {args.reid_threshold}")

#         # ------------------------------------------------------------------
#         # Open video capture
#         # ------------------------------------------------------------------
#         if args.video:
#             print(f"[VIDEO] Opening local file: {args.video}")
#             cap = cv2.VideoCapture(str(args.video))
#             video_fps = cap.get(cv2.CAP_PROP_FPS)
#             print(f"[VIDEO] Detected FPS: {video_fps:.2f}")
#             frame_skip = max(1, int(video_fps // 2))  # process at ~2 FPS for local videos
#         else:
#             creds = load_aws_config(args.creds)
#             print(f"[KVS] Connecting to stream: {args.stream_name}")
#             hls_url = get_hls_url(creds, args.stream_name)
#             cap = cv2.VideoCapture(hls_url)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         if not cap.isOpened():
#             print("[ERROR] Could not open video source.")
#             return 1

#         # ------------------------------------------------------------------
#         # Main processing loop
#         # ------------------------------------------------------------------
#         processed = 0
#         failures = 0
#         total_violations_all = 0
#         total_face_crops = 0
#         start = time.time()

#         while processed < args.frames:
#             # Timeout guard (KVS only — local video can stall too if corrupt)
#             if time.time() - start > args.timeout and not args.video:
#                 print(f"[WARN] Timeout reached after {args.timeout}s.")
#                 break

#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 failures += 1
#                 if args.video:
#                     # End of file
#                     print("[VIDEO] End of file reached.")
#                     break
#                 time.sleep(0.1)
#                 continue

#             frame_start = time.time()

#             # ---- Step 1: Person detection ----
#             person_results = person_model.predict(
#                 source=frame,
#                 imgsz=args.image_size,
#                 conf=args.person_confidence,
#                 iou=args.iou,
#                 device=device,
#                 verbose=False,
#             )[0]

#             person_boxes: List[Tuple[float, float, float, float]] = []
#             person_confidences: List[float] = []

#             if person_results.boxes is not None:
#                 for box in person_results.boxes:
#                     class_id = int(box.cls[0])
#                     class_name = person_classes.get(class_id, str(class_id))
#                     if class_name.lower() == "person":
#                         coords = tuple(float(x) for x in box.xyxy[0].tolist())
#                         person_boxes.append(coords)
#                         person_confidences.append(float(box.conf[0]))

#             # ---- Step 2: PPE violation detection ----
#             ppe_results = ppe_model.predict(
#                 source=frame,
#                 imgsz=args.image_size,
#                 conf=args.confidence,
#                 iou=args.iou,
#                 device=device,
#                 verbose=False,
#             )[0]

#             ppe_detections = []
#             if ppe_results.boxes is not None:
#                 for box in ppe_results.boxes:
#                     det = detection_to_dict(box, ppe_classes, "ppe")
#                     if det["is_violation"]:
#                         ppe_detections.append(det)

#             if args.violations_only and not ppe_detections:
#                 continue

#             # ---- Step 3: Full-frame YOLO face detection ----
#             face_results = face_model.predict(
#                 source=frame,
#                 imgsz=args.face_image_size,
#                 conf=args.face_confidence,
#                 iou=args.iou,
#                 device=device,
#                 verbose=False,
#             )[0]

#             face_boxes: List[Tuple[float, float, float, float]] = []
#             if face_results.boxes is not None:
#                 for box in face_results.boxes:
#                     coords = tuple(float(x) for x in box.xyxy[0].tolist())
#                     face_boxes.append(coords)

#             total_violations_all += len(ppe_detections)
#             inference_ms = (time.time() - frame_start) * 1000

#             frame_id = f"frame_{processed + 1:04d}"
#             timestamp = datetime.now().isoformat()

#             # Annotated frame path (always saved)
#             annotated_frame_path = str(frames_dir / f"{frame_id}_annotated.jpg")

#             # ---- Step 4: Assign violations → persons ----
#             person_violations: Dict[int, List[dict]] = {}
#             for violation in ppe_detections:
#                 pidx = find_person_for_violation(tuple(violation["box"]), person_boxes)
#                 if pidx >= 0:
#                     person_violations.setdefault(pidx, []).append(violation)

#             # ---- Step 5: For each violating person — face crop + embedding ----
#             person_id_labels: Dict[int, str] = {}  # for drawing on frame
#             crops_this_frame = 0

#             for person_idx, violations in person_violations.items():
#                 person_box = person_boxes[person_idx]
#                 person_conf = person_confidences[person_idx]

#                 # --- YOLO face detection (two-stage) ---
#                 face_crop_bgr, face_box = crop_face_from_person(frame, person_box, face_boxes)
#                 face_source = "full_frame"

#                 if face_crop_bgr is None:
#                     face_crop_bgr, face_box = detect_face_from_person_crop(
#                         frame,
#                         person_box,
#                         face_model,
#                         device,
#                         args.face_crop_image_size,
#                         args.face_crop_confidence,
#                         args.iou,
#                         args.face_region_height_ratio,
#                         args.face_region_margin,
#                     )
#                     if face_crop_bgr is not None:
#                         face_source = "person_crop_fallback"
#                         face_boxes.append(tuple(face_box))

#                 # --- FaceNet embedding ---
#                 embedding = None
#                 if face_crop_bgr is not None:
#                     embedding = extract_embedding(face_crop_bgr, mtcnn, resnet, facenet_device)
#                     if embedding is None:
#                         print(f"[WARN] FaceNet returned no embedding for {frame_id} person_{person_idx+1:02d} (crop too small or undetectable)")

#                 # --- Re-identification ---
#                 person_id = None
#                 is_new = False
#                 reid_sim = 0.0

#                 if embedding is not None:
#                     person_id, is_new, reid_sim = id_store.match_or_create(embedding, timestamp)
#                     if not is_new:
#                         id_store.update_embedding(person_id, embedding)
#                     print(
#                         f"  [REID] {'NEW' if is_new else 'MATCH'} → {person_id}  "
#                         f"sim={reid_sim:.4f}  total_known={id_store.known_count()}"
#                     )
#                 else:
#                     # No embedding: generate a temporary anonymous label for the frame
#                     person_id = f"unknown_{frame_id}_p{person_idx+1:02d}"
#                     face_source = "not_detected"

#                 person_id_labels[person_idx] = person_id

#                 # --- Save face crop image ---
#                 face_crop_path = ""
#                 face_box_str = ""

#                 if face_crop_bgr is not None:
#                     crop_filename = f"{frame_id}_{person_id}_{face_source}_face.jpg"
#                     crop_path = crops_dir / crop_filename
#                     if cv2.imwrite(str(crop_path), face_crop_bgr):
#                         face_crop_path = str(crop_path)
#                         face_box_str = format_box(face_box)
#                         total_face_crops += 1
#                         crops_this_frame += 1
#                     else:
#                         print(f"[WARN] Failed to save face crop: {crop_path}")
#                 else:
#                     print(f"[WARN] No face crop for {frame_id} person_{person_idx+1:02d}")

#                 # --- Record violation instance in identity store ---
#                 violation_count_instance = len(violations)
#                 instance_record = {
#                     "timestamp": timestamp,
#                     "frame_id": frame_id,
#                     "frame_path": annotated_frame_path,
#                     "face_crop_path": face_crop_path,
#                     "violations_list": [v["class_name"] for v in violations],
#                     "violation_count": violation_count_instance,
#                     "reid_similarity": round(reid_sim, 6),
#                 }

#                 cumulative_violations = 0
#                 emb_path = ""
#                 if person_id and not person_id.startswith("unknown_"):
#                     cumulative_violations = id_store.record_violation(person_id, instance_record)
#                     emb_path = str(id_store._person_npy_path(person_id))

#                 # --- Write metadata CSV row ---
#                 csv_writer.writerow({
#                     "person_id": person_id,
#                     "is_new_person": is_new,
#                     "reid_similarity": round(reid_sim, 6),
#                     "timestamp": timestamp,
#                     "frame_id": frame_id,
#                     "frame_path": annotated_frame_path,
#                     "face_crop_path": face_crop_path,
#                     "face_box": face_box_str,
#                     "face_detection_source": face_source,
#                     "person_box": format_box(person_box),
#                     "person_detection_confidence": round(person_conf, 6),
#                     "violations": "; ".join(
#                         f"{v['class_name']}({v['confidence']:.2f})" for v in violations
#                     ),
#                     "violation_boxes": "; ".join(
#                         f"{v['class_name']}:{format_box(v['box'])}" for v in violations
#                     ),
#                     "violation_confidences": "; ".join(
#                         f"{v['class_name']}:{v['confidence']:.3f}" for v in violations
#                     ),
#                     "violation_count_this_instance": violation_count_instance,
#                     "total_violations_cumulative": cumulative_violations,
#                     "person_count_in_frame": len(person_boxes),
#                     "face_count_in_frame": len(face_boxes),
#                     "embedding_path": emb_path,
#                 })
#                 csv_file.flush()

#             # ---- Draw and save annotated frame ----
#             annotated = frame.copy()
#             draw_detections(annotated, ppe_detections, person_boxes, face_boxes, person_id_labels)
#             if not cv2.imwrite(annotated_frame_path, annotated):
#                 print(f"[WARN] Failed to save annotated frame: {annotated_frame_path}")

#             print(
#                 f"[FRAME {processed + 1:04d}] "
#                 f"Persons: {len(person_boxes)}, "
#                 f"Violations: {len(ppe_detections)}, "
#                 f"Faces: {len(face_boxes)}, "
#                 f"Crops: {crops_this_frame}, "
#                 f"Inference: {inference_ms:.0f}ms"
#             )
#             if ppe_detections:
#                 summary = ", ".join(
#                     f"{d['class_name']}({d['confidence']:.2f})" for d in ppe_detections
#                 )
#                 print(f"  Violations: {summary}")

#             processed += 1

#         # ------------------------------------------------------------------
#         # Summary
#         # ------------------------------------------------------------------
#         cap.release()
#         elapsed = time.time() - start

#         print(f"\n[DONE] Processed {processed} frames in {elapsed:.2f}s")
#         print(f"[STATS] Read failures           : {failures}")
#         print(f"[STATS] Total violations         : {total_violations_all}")
#         print(f"[STATS] Face crops saved         : {total_face_crops}")
#         print(f"[STATS] Known identities (total) : {id_store.known_count()}")
#         print(f"[STATS] Results saved to         : {save_dir}")
#         print(f"[STATS] Metadata CSV             : {metadata_file}")

#         if processed == 0:
#             return 1

#     except Exception as exc:
#         import traceback
#         print(f"[ERROR] Pipeline failed: {exc}")
#         traceback.print_exc()
#         return 1
#     finally:
#         csv_file.close()

#     return 0


# if __name__ == "__main__":
#     sys.exit(main())



#!/usr/bin/env python3
"""
Local KVS / Video PPE + Face Detection + FaceNet Re-Identification.

Pipeline:
1. Capture frame from KVS stream or local video file (--video)
2. Detect persons using person YOLO model
3. Detect PPE violations using PPE YOLO model
4. For persons with violations:
   a. Detect faces using YOLOv8-face (bounding box)
   b. Align face crop using MTCNN (for FaceNet accuracy)
   c. Generate 512-d embedding using FaceNet (InceptionResnetV1)
   d. Match embedding against stored person embeddings (cosine similarity)
   e. Assign existing person_id if match found, else create new person_id
   f. Update rolling mean embedding for that person
5. Save annotated frames, face crops, embeddings, and metadata CSV

Storage layout under --save-dir (default: violations/):
  annotated_frames/          - frames with drawn bounding boxes
  violator_face_crops/       - raw face crop images per violation instance
  embeddings/
    person_XXXX.npy          - rolling mean 512-d FaceNet embedding
    person_XXXX_meta.json    - identity metadata (total violations, history)
  metadata.csv               - one row per violation instance, indexed on person_id

Re-identification:
  - Cosine similarity between new embedding and stored mean + all angle prototypes
  - Up to 8 diverse per-angle prototype embeddings stored per person
  - Match threshold: --reid-threshold (default 0.68)
  - Stored mean embedding updated as rolling mean across all matched crops
  - Video mode: 2 frames per second sampled (frame-skip based on source FPS)
"""

import argparse
import configparser
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# FaceNet imports (facenet-pytorch)
# ---------------------------------------------------------------------------
try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
except ImportError:
    print("[ERROR] facenet-pytorch is not installed.")
    print("        Install with:  pip install facenet-pytorch")
    sys.exit(1)


# ===========================================================================
# AWS / KVS helpers
# ===========================================================================

def load_aws_config(creds_file: str) -> dict:
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
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise ValueError(f"Missing required AWS values: {', '.join(missing)}")
    return values


def get_hls_url(creds: dict, stream_name: str) -> str:
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


# ===========================================================================
# Geometry helpers
# ===========================================================================

def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center_inside(
    inner: Tuple[float, float, float, float],
    outer: Tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


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


def format_box(box) -> str:
    return f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"


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
        px1 - pw * margin_ratio,
        py1 - ph * margin_ratio * 0.5,
        px2 + pw * margin_ratio,
        py1 + ph * height_ratio,
    )
    return clamp_box(region, frame_w, frame_h)


# ===========================================================================
# PPE helpers
# ===========================================================================

def is_ppe_violation_class(class_name: str) -> bool:
    return "NO" in class_name.upper()


def detection_to_dict(box, class_names: dict, model_name: str) -> dict:
    class_id = int(box.cls[0])
    class_name = class_names.get(class_id, str(class_id))
    return {
        "model": model_name,
        "class_id": class_id,
        "class_name": class_name,
        "is_violation": is_ppe_violation_class(class_name),
        "confidence": round(float(box.conf[0]), 6),
        "box": [round(float(v), 2) for v in box.xyxy[0].tolist()],
    }


def find_person_for_violation(
    violation_box: Tuple[float, float, float, float],
    person_boxes: List[Tuple[float, float, float, float]],
) -> int:
    if not person_boxes:
        return -1
    containing = []
    for i, pb in enumerate(person_boxes):
        if box_center_inside(violation_box, pb):
            containing.append((i, box_iou(violation_box, pb)))
    if containing:
        return max(containing, key=lambda x: x[1])[0]
    best_idx, best_iou = -1, 0.0
    for i, pb in enumerate(person_boxes):
        iou = box_iou(violation_box, pb)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx if best_iou > 0.1 else -1


# ===========================================================================
# YOLO face detection helpers (unchanged from original)
# ===========================================================================

def crop_face_from_person(
    frame,
    person_box: Tuple[float, float, float, float],
    face_boxes: List[Tuple[float, float, float, float]],
) -> Tuple[Any, Optional[Tuple[float, float, float, float]]]:
    """Pick the best YOLO-detected face whose centre is inside the person box."""
    if not face_boxes:
        return None, None
    px1, py1, px2, py2 = person_box
    person_h = max(1.0, py2 - py1)
    frame_h, frame_w = frame.shape[:2]
    best_face, best_score = None, -1.0
    for fb in face_boxes:
        fx1, fy1, fx2, fy2 = fb
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue
        area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        score = area + max(0.0, 1.0 - (cy - py1) / person_h) * 1000.0
        if score > best_score:
            best_score, best_face = score, fb
    if best_face is None:
        return None, None
    x1, y1, x2, y2 = clamp_box(best_face, frame_w, frame_h)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2]
    return (None, None) if crop.size == 0 else (crop, best_face)


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
) -> Tuple[Any, Optional[Tuple[float, float, float, float]]]:
    """Fallback: run YOLO-face on violator's upper-body crop."""
    frame_h, frame_w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = person_face_search_region(
        person_box, frame_w, frame_h, height_ratio, margin_ratio
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
    best_box, best_conf = None, -1.0
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
    crop = frame[y1:y2, x1:x2]
    return (None, None) if crop.size == 0 else (crop, best_box)


# ===========================================================================
# FaceNet embedding helpers
# ===========================================================================

# Minimum pixel dimensions of a face crop for FaceNet to be reliable
_FACENET_MIN_SIZE = 40  # pixels (width or height)
# FaceNet expects 160x160 aligned input
_FACENET_INPUT_SIZE = 160


def build_facenet(device) -> Tuple[Any, Any]:
    """
    Returns (mtcnn, resnet).

    MTCNN: used to align & re-crop from the coarse YOLO face crop.
           keep_all=False → returns the single most prominent face.
    InceptionResnetV1: pretrained on VGGFace2, outputs 512-d unit-norm embeddings.
    """
    mtcnn = MTCNN(
        image_size=_FACENET_INPUT_SIZE,
        margin=14,            # pixels of context around the face
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],   # P-Net, R-Net, O-Net detection thresholds
        factor=0.709,
        post_process=True,    # normalise to [-1, 1] for InceptionResnetV1
        keep_all=False,
        select_largest=True,  # when multiple detections, take the largest
        device=device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


def extract_embedding(
    face_crop_bgr: np.ndarray,
    mtcnn,
    resnet,
    device,
) -> Optional[np.ndarray]:
    """
    Given a BGR face crop (from YOLO), return a 512-d unit-norm embedding.

    Flow:
      1. Reject crops that are too small (unreliable for FaceNet).
      2. Convert BGR → RGB PIL Image.
      3. Run MTCNN to align: this handles rotation, scale, and landmark
         alignment that makes FaceNet embeddings much more stable.
      4. If MTCNN finds no face (happens on very blurry/partial crops),
         fall back to a direct resize + normalise so we still produce
         *some* embedding rather than silently dropping the detection.
      5. Run InceptionResnetV1 to get the 512-d embedding.
      6. L2-normalise (resnet already does this if pretrained, but
         we enforce it explicitly for safety).
    """
    h, w = face_crop_bgr.shape[:2]
    if h < _FACENET_MIN_SIZE or w < _FACENET_MIN_SIZE:
        return None

    # BGR → RGB PIL
    pil_img = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

    # --- MTCNN alignment pass ---
    aligned = mtcnn(pil_img)  # returns Tensor[3,160,160] or None

    if aligned is None:
        # Fallback: resize + manual normalise to [-1,1]
        resized = pil_img.resize((_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
        aligned = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)  # CHW

    # Ensure shape is [3, H, W] and add batch dim
    if aligned.dim() == 3:
        aligned = aligned.unsqueeze(0)  # [1, 3, H, W]

    aligned = aligned.to(device)

    with torch.no_grad():
        embedding = resnet(aligned)          # [1, 512]
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy()  # (512,)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    # Both should already be unit-norm, but normalise defensively
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ===========================================================================
# Identity store  (in-memory + disk)
# ===========================================================================

class IdentityStore:
    """
    Manages person identities: embeddings, violation counts, metadata.

    On-disk layout inside `embeddings_dir`:
      person_0001.npy            – rolling mean 512-d embedding
      person_0001_meta.json      – {person_id, first_seen, total_violations,
                                    embedding_count, instances:[...]}
    """

    def __init__(self, embeddings_dir: Path, reid_threshold: float = 0.68):
        self.embeddings_dir = embeddings_dir
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = reid_threshold
        self.max_prototypes = 8  # max diverse per-angle embeddings per person

        # In-memory cache: person_id (str) → mean embedding (np.ndarray)
        self._embeddings: Dict[str, np.ndarray] = {}
        # person_id → list of diverse raw prototype embeddings (angle robustness)
        self._all_embeddings: Dict[str, List[np.ndarray]] = {}
        # person_id → metadata dict
        self._meta: Dict[str, dict] = {}

        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _person_npy_path(self, person_id: str) -> Path:
        return self.embeddings_dir / f"{person_id}.npy"

    def _person_meta_path(self, person_id: str) -> Path:
        return self.embeddings_dir / f"{person_id}_meta.json"

    def _load_from_disk(self) -> None:
        """Load all existing embeddings and metadata from disk into memory."""
        for npy_file in sorted(self.embeddings_dir.glob("person_*.npy")):
            person_id = npy_file.stem  # e.g. "person_0001"
            emb = np.load(str(npy_file))
            self._embeddings[person_id] = emb
            # Seed prototype list with the stored mean as the single known prototype
            self._all_embeddings[person_id] = [emb.copy()]

            meta_path = self._person_meta_path(person_id)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._meta[person_id] = json.load(f)
            else:
                # Reconstruct minimal metadata if missing
                self._meta[person_id] = {
                    "person_id": person_id,
                    "first_seen": "unknown",
                    "total_violations": 0,
                    "embedding_count": 1,
                    "instances": [],
                }
        print(f"[REID] Loaded {len(self._embeddings)} known identities from disk.")

    def _save_embedding(self, person_id: str) -> None:
        np.save(str(self._person_npy_path(person_id)), self._embeddings[person_id])

    def _save_meta(self, person_id: str) -> None:
        with open(self._person_meta_path(person_id), "w", encoding="utf-8") as f:
            json.dump(self._meta[person_id], f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Identity management
    # ------------------------------------------------------------------

    def _next_person_id(self) -> str:
        """Generate the next sequential person_id string."""
        existing = [
            int(pid.split("_")[1])
            for pid in self._embeddings.keys()
            if pid.startswith("person_") and pid.split("_")[1].isdigit()
        ]
        next_num = (max(existing) + 1) if existing else 1
        return f"person_{next_num:04d}"

    def match_or_create(
        self,
        embedding: np.ndarray,
        timestamp: str,
    ) -> Tuple[str, bool, float]:
        """
        Match `embedding` against all stored identities.

        Checks both the rolling mean embedding (fast path) and all per-angle
        prototype embeddings (angle robustness), taking the highest similarity.

        Returns:
            (person_id, is_new, best_similarity)
            is_new=True  → first time this person has been seen
            is_new=False → matched an existing identity
        """
        best_pid, best_sim = None, -1.0

        for pid, stored_emb in self._embeddings.items():
            # Fast path: check mean embedding
            sim = cosine_similarity(embedding, stored_emb)
            # Angle robustness: also check all stored prototypes
            for proto in self._all_embeddings.get(pid, []):
                sim = max(sim, cosine_similarity(embedding, proto))
            if sim > best_sim:
                best_sim, best_pid = sim, pid

        if best_pid is not None and best_sim >= self.threshold:
            return best_pid, False, best_sim

        # --- New identity ---
        new_pid = self._next_person_id()
        self._embeddings[new_pid] = embedding.copy()
        self._all_embeddings[new_pid] = [embedding.copy()]
        self._meta[new_pid] = {
            "person_id": new_pid,
            "first_seen": timestamp,
            "total_violations": 0,
            "embedding_count": 1,
            "instances": [],
        }
        self._save_embedding(new_pid)
        self._save_meta(new_pid)
        return new_pid, True, 1.0  # perfect self-similarity

    def update_embedding(self, person_id: str, new_embedding: np.ndarray) -> None:
        """
        Update stored mean embedding (rolling mean) and add new_embedding as a
        prototype if it is sufficiently diverse from all existing prototypes
        (cosine similarity < 0.92), up to max_prototypes per person.
        """
        meta = self._meta[person_id]
        n = meta["embedding_count"]
        old_emb = self._embeddings[person_id]
        mean_emb = (old_emb * n + new_embedding) / (n + 1)
        # L2-normalise so cosine similarity stays meaningful
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        self._embeddings[person_id] = mean_emb
        meta["embedding_count"] = n + 1

        # Add as a new prototype only if it represents a diverse angle
        protos = self._all_embeddings.setdefault(person_id, [])
        is_diverse = all(cosine_similarity(new_embedding, p) < 0.92 for p in protos)
        if is_diverse and len(protos) < self.max_prototypes:
            protos.append(new_embedding.copy())

        self._save_embedding(person_id)

    def record_violation(
        self,
        person_id: str,
        instance: dict,
    ) -> int:
        """
        Append a violation instance to the person's metadata.
        Returns updated total_violations count.
        """
        meta = self._meta[person_id]
        meta["total_violations"] += len(instance.get("violations_list", []))
        meta["instances"].append(instance)
        self._save_meta(person_id)
        return meta["total_violations"]

    def total_violations(self, person_id: str) -> int:
        return self._meta.get(person_id, {}).get("total_violations", 0)

    def known_count(self) -> int:
        return len(self._embeddings)


# ===========================================================================
# Drawing
# ===========================================================================

def draw_detections(
    frame,
    ppe_detections: list,
    person_boxes: List = None,
    face_boxes: List = None,
    person_labels: Dict[int, str] = None,
) -> None:
    for det in ppe_detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = (0, 0, 255) if det["is_violation"] else (0, 160, 0)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if person_boxes:
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = [int(c) for c in box]
            lbl = (person_labels or {}).get(i, f"Person {i+1}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if face_boxes:
        for box in face_boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="PPE + Face Detection + FaceNet Re-Identification against KVS or local video"
    )
    # Source — mutually exclusive: KVS stream OR local video
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--stream-name", help="KVS stream name")
    source_group.add_argument("--video", help="Path to a local MP4 video file")

    # AWS credentials (only needed for KVS)
    parser.add_argument(
        "--creds",
        default=str(parent_dir / "ppe_creds.txt"),
        help="Path to PPE credentials INI file (required for --stream-name)",
    )

    # Model paths
    parser.add_argument(
        "--person-model",
        default=str(parent_dir / "runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt"),
        help="Person detection YOLO model path",
    )
    parser.add_argument(
        "--ppe-model",
        default=str(parent_dir / "runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt"),
        help="PPE detection YOLO model path",
    )
    parser.add_argument(
        "--face-model",
        default=str(script_dir / "yolov8n-face.pt"),
        help="Face detection YOLO model path (bounding box stage)",
    )

    # Frame / processing
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for readable frames (KVS only)")
    parser.add_argument("--violations-only", action="store_true", help="Process only frames containing violations")

    # YOLO inference
    parser.add_argument("--confidence", type=float, default=0.25, help="PPE YOLO confidence threshold")
    parser.add_argument("--person-confidence", type=float, default=0.33, help="Person detection confidence")
    parser.add_argument("--face-confidence", type=float, default=0.25, help="Full-frame face detection confidence")
    parser.add_argument("--face-crop-confidence", type=float, default=0.15, help="Fallback face detection confidence on violator person crops")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--image-size", type=int, default=640, help="Person/PPE YOLO inference image size")
    parser.add_argument("--face-image-size", type=int, default=1280, help="Full-frame face model inference image size")
    parser.add_argument("--face-crop-image-size", type=int, default=960, help="Person-crop face model inference image size")
    parser.add_argument("--face-region-height-ratio", type=float, default=0.65, help="Fraction of person height for fallback face search")
    parser.add_argument("--face-region-margin", type=float, default=0.20, help="Side margin around person box for fallback face search")

    # Re-identification
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.68,
        help="Cosine similarity threshold for FaceNet re-identification (0–1, higher = more lenient)",
    )

    # Output
    parser.add_argument("--save-dir", default=str(script_dir / "violations"), help="Root directory for all results")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate model files
    # ------------------------------------------------------------------
    model_paths = [
        ("person", Path(args.person_model)),
        ("ppe", Path(args.ppe_model)),
        ("face", Path(args.face_model)),
    ]
    for model_name, model_path in model_paths:
        if not model_path.exists():
            print(f"[ERROR] {model_name} model not found: {model_path.resolve()}")
            return 1

    # ------------------------------------------------------------------
    # Validate video file if provided
    # ------------------------------------------------------------------
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {video_path.resolve()}")
            return 1

    # ------------------------------------------------------------------
    # Setup output directories
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    crops_dir = save_dir / "violator_face_crops"
    frames_dir = save_dir / "annotated_frames"
    embeddings_dir = save_dir / "embeddings"

    for d in (save_dir, crops_dir, frames_dir, embeddings_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup metadata CSV
    # ------------------------------------------------------------------
    metadata_file = save_dir / "metadata.csv"
    csv_fields = [
        "person_id",
        "is_new_person",
        "reid_similarity",
        "timestamp",
        "frame_id",
        "frame_path",
        "face_crop_path",
        "face_box",
        "face_detection_source",
        "person_box",
        "person_detection_confidence",
        "violations",
        "violation_boxes",
        "violation_confidences",
        "violation_count_this_instance",
        "total_violations_cumulative",
        "person_count_in_frame",
        "face_count_in_frame",
        "embedding_path",
    ]
    # Append mode so re-runs accumulate history
    csv_file = open(metadata_file, "a", newline="", encoding="utf-8")
    write_header = csv_file.tell() == 0
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        csv_writer.writeheader()

    try:
        # ------------------------------------------------------------------
        # Load models
        # ------------------------------------------------------------------
        print("[MODEL] Loading YOLO models...")
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[MODEL] Inference device: {device}")

        person_model = YOLO(str(args.person_model))
        ppe_model = YOLO(str(args.ppe_model))
        face_model = YOLO(str(args.face_model))

        person_classes = {int(k): str(v) for k, v in person_model.names.items()}
        ppe_classes = {int(k): str(v) for k, v in ppe_model.names.items()}

        print(f"[MODEL] Person classes: {person_classes}")
        print(f"[MODEL] PPE classes: {ppe_classes}")

        print("[MODEL] Loading FaceNet (MTCNN + InceptionResnetV1)...")
        # FaceNet runs on CPU if CUDA not available; it's lightweight enough
        facenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn, resnet = build_facenet(facenet_device)
        print(f"[MODEL] FaceNet device: {facenet_device}")

        # ------------------------------------------------------------------
        # Load identity store
        # ------------------------------------------------------------------
        id_store = IdentityStore(embeddings_dir, reid_threshold=args.reid_threshold)
        print(f"[REID] Re-ID threshold: {args.reid_threshold}")

        # ------------------------------------------------------------------
        # Open video capture
        # ------------------------------------------------------------------
        if args.video:
            print(f"[VIDEO] Opening local file: {args.video}")
            cap = cv2.VideoCapture(str(args.video))
        else:
            creds = load_aws_config(args.creds)
            print(f"[KVS] Connecting to stream: {args.stream_name}")
            hls_url = get_hls_url(creds, args.stream_name)
            cap = cv2.VideoCapture(hls_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("[ERROR] Could not open video source.")
            return 1

        # For video mode: compute frame skip to sample at 2fps
        if args.video:
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 0:
                frame_skip = max(1, round(src_fps / 2))
                print(f"[VIDEO] Source FPS: {src_fps:.2f} → sampling every {frame_skip} frames (2fps)")
            else:
                frame_skip = 1
                print("[VIDEO] Could not read FPS, processing every frame")
        else:
            frame_skip = 1

        # ------------------------------------------------------------------
        # Main processing loop
        # ------------------------------------------------------------------
        processed = 0
        failures = 0
        total_violations_all = 0
        total_face_crops = 0
        start = time.time()

        while processed < args.frames:
            # Timeout guard (KVS only — local video can stall too if corrupt)
            if time.time() - start > args.timeout and not args.video:
                print(f"[WARN] Timeout reached after {args.timeout}s.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                failures += 1
                if args.video:
                    # End of file
                    print("[VIDEO] End of file reached.")
                    break
                time.sleep(0.1)
                continue

            # Video mode: discard intermediate frames to achieve 2fps sampling
            if frame_skip > 1:
                for _ in range(frame_skip - 1):
                    skip_ret, _ = cap.read()
                    if not skip_ret:
                        break

            frame_start = time.time()

            # ---- Step 1: Person detection ----
            person_results = person_model.predict(
                source=frame,
                imgsz=args.image_size,
                conf=args.person_confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            person_boxes: List[Tuple[float, float, float, float]] = []
            person_confidences: List[float] = []

            if person_results.boxes is not None:
                for box in person_results.boxes:
                    class_id = int(box.cls[0])
                    class_name = person_classes.get(class_id, str(class_id))
                    if class_name.lower() == "person":
                        coords = tuple(float(x) for x in box.xyxy[0].tolist())
                        person_boxes.append(coords)
                        person_confidences.append(float(box.conf[0]))

            # ---- Step 2: PPE violation detection ----
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
                    det = detection_to_dict(box, ppe_classes, "ppe")
                    if det["is_violation"]:
                        ppe_detections.append(det)

            if args.violations_only and not ppe_detections:
                continue

            # ---- Step 3: Full-frame YOLO face detection ----
            face_results = face_model.predict(
                source=frame,
                imgsz=args.face_image_size,
                conf=args.face_confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )[0]

            face_boxes: List[Tuple[float, float, float, float]] = []
            if face_results.boxes is not None:
                for box in face_results.boxes:
                    coords = tuple(float(x) for x in box.xyxy[0].tolist())
                    face_boxes.append(coords)

            total_violations_all += len(ppe_detections)
            inference_ms = (time.time() - frame_start) * 1000

            frame_id = f"frame_{processed + 1:04d}"
            timestamp = datetime.now().isoformat()

            # Annotated frame path (always saved)
            annotated_frame_path = str(frames_dir / f"{frame_id}_annotated.jpg")

            # ---- Step 4: Assign violations → persons ----
            person_violations: Dict[int, List[dict]] = {}
            for violation in ppe_detections:
                pidx = find_person_for_violation(tuple(violation["box"]), person_boxes)
                if pidx >= 0:
                    person_violations.setdefault(pidx, []).append(violation)

            # ---- Step 5: For each violating person — face crop + embedding ----
            person_id_labels: Dict[int, str] = {}  # for drawing on frame
            crops_this_frame = 0

            for person_idx, violations in person_violations.items():
                person_box = person_boxes[person_idx]
                person_conf = person_confidences[person_idx]

                # --- YOLO face detection (two-stage) ---
                face_crop_bgr, face_box = crop_face_from_person(frame, person_box, face_boxes)
                face_source = "full_frame"

                if face_crop_bgr is None:
                    face_crop_bgr, face_box = detect_face_from_person_crop(
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
                    if face_crop_bgr is not None:
                        face_source = "person_crop_fallback"
                        face_boxes.append(tuple(face_box))

                # --- FaceNet embedding ---
                embedding = None
                if face_crop_bgr is not None:
                    embedding = extract_embedding(face_crop_bgr, mtcnn, resnet, facenet_device)
                    if embedding is None:
                        print(f"[WARN] FaceNet returned no embedding for {frame_id} person_{person_idx+1:02d} (crop too small or undetectable)")

                # --- Re-identification ---
                person_id = None
                is_new = False
                reid_sim = 0.0

                if embedding is not None:
                    person_id, is_new, reid_sim = id_store.match_or_create(embedding, timestamp)
                    if not is_new:
                        id_store.update_embedding(person_id, embedding)
                    print(
                        f"  [REID] {'NEW' if is_new else 'MATCH'} → {person_id}  "
                        f"sim={reid_sim:.4f}  total_known={id_store.known_count()}"
                    )
                else:
                    # No embedding: generate a temporary anonymous label for the frame
                    person_id = f"unknown_{frame_id}_p{person_idx+1:02d}"
                    face_source = "not_detected"

                person_id_labels[person_idx] = person_id

                # --- Save face crop image ---
                face_crop_path = ""
                face_box_str = ""

                if face_crop_bgr is not None:
                    crop_filename = f"{frame_id}_{person_id}_{face_source}_face.jpg"
                    crop_path = crops_dir / crop_filename
                    if cv2.imwrite(str(crop_path), face_crop_bgr):
                        face_crop_path = str(crop_path)
                        face_box_str = format_box(face_box)
                        total_face_crops += 1
                        crops_this_frame += 1
                    else:
                        print(f"[WARN] Failed to save face crop: {crop_path}")
                else:
                    print(f"[WARN] No face crop for {frame_id} person_{person_idx+1:02d}")

                # --- Record violation instance in identity store ---
                violation_count_instance = len(violations)
                instance_record = {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "frame_path": annotated_frame_path,
                    "face_crop_path": face_crop_path,
                    "violations_list": [v["class_name"] for v in violations],
                    "violation_count": violation_count_instance,
                    "reid_similarity": round(reid_sim, 6),
                }

                cumulative_violations = 0
                emb_path = ""
                if person_id and not person_id.startswith("unknown_"):
                    cumulative_violations = id_store.record_violation(person_id, instance_record)
                    emb_path = str(id_store._person_npy_path(person_id))

                # --- Write metadata CSV row ---
                csv_writer.writerow({
                    "person_id": person_id,
                    "is_new_person": is_new,
                    "reid_similarity": round(reid_sim, 6),
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "frame_path": annotated_frame_path,
                    "face_crop_path": face_crop_path,
                    "face_box": face_box_str,
                    "face_detection_source": face_source,
                    "person_box": format_box(person_box),
                    "person_detection_confidence": round(person_conf, 6),
                    "violations": "; ".join(
                        f"{v['class_name']}({v['confidence']:.2f})" for v in violations
                    ),
                    "violation_boxes": "; ".join(
                        f"{v['class_name']}:{format_box(v['box'])}" for v in violations
                    ),
                    "violation_confidences": "; ".join(
                        f"{v['class_name']}:{v['confidence']:.3f}" for v in violations
                    ),
                    "violation_count_this_instance": violation_count_instance,
                    "total_violations_cumulative": cumulative_violations,
                    "person_count_in_frame": len(person_boxes),
                    "face_count_in_frame": len(face_boxes),
                    "embedding_path": emb_path,
                })
                csv_file.flush()

            # ---- Draw and save annotated frame ----
            annotated = frame.copy()
            draw_detections(annotated, ppe_detections, person_boxes, face_boxes, person_id_labels)
            if not cv2.imwrite(annotated_frame_path, annotated):
                print(f"[WARN] Failed to save annotated frame: {annotated_frame_path}")

            print(
                f"[FRAME {processed + 1:04d}] "
                f"Persons: {len(person_boxes)}, "
                f"Violations: {len(ppe_detections)}, "
                f"Faces: {len(face_boxes)}, "
                f"Crops: {crops_this_frame}, "
                f"Inference: {inference_ms:.0f}ms"
            )
            if ppe_detections:
                summary = ", ".join(
                    f"{d['class_name']}({d['confidence']:.2f})" for d in ppe_detections
                )
                print(f"  Violations: {summary}")

            processed += 1

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        cap.release()
        elapsed = time.time() - start

        print(f"\n[DONE] Processed {processed} frames in {elapsed:.2f}s")
        print(f"[STATS] Read failures           : {failures}")
        print(f"[STATS] Total violations         : {total_violations_all}")
        print(f"[STATS] Face crops saved         : {total_face_crops}")
        print(f"[STATS] Known identities (total) : {id_store.known_count()}")
        print(f"[STATS] Results saved to         : {save_dir}")
        print(f"[STATS] Metadata CSV             : {metadata_file}")

        if processed == 0:
            return 1

    except Exception as exc:
        import traceback
        print(f"[ERROR] Pipeline failed: {exc}")
        traceback.print_exc()
        return 1
    finally:
        csv_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())