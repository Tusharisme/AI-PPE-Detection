#!/usr/bin/env python3
"""Run a single PPE detector on local images with ppe_worker_4-style gating."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


CLASS_NAMES = {0: "Person", 1: "NO-Hardhat", 2: "NO-Safety Vest", 3: "NO-Safety Boots"}
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
VIOLATION_CLASS_NAMES = {"NO-Hardhat", "NO-Safety Vest", "NO-Safety Boots"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    person_index: int | None = None
    source: str = "full_frame"


class Detector(Protocol):
    def predict(self, image: Any, imgsz: int, conf: float, iou: float) -> list[Detection]:
        ...


def parse_class_confidences(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        sep = "=" if "=" in item else (":" if ":" in item else None)
        if sep is None:
            continue
        key, value = item.split(sep, 1)
        try:
            conf = float(value)
        except ValueError:
            continue
        if 0.0 <= conf <= 1.0:
            out[key.strip()] = conf
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-model PPE detection on local images")
    parser.add_argument("--backend", choices=["yolo", "rfdetr"], default="yolo")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", default=Path("Images"), type=Path)
    parser.add_argument("--save-dir", default=Path("local_single_model_images"), type=Path)
    parser.add_argument("--image-size", default=1920, type=int)
    parser.add_argument("--crop-image-size", default=960, type=int)
    parser.add_argument("--confidence", default=0.10, type=float)
    parser.add_argument("--person-confidence", default=0.33, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument(
        "--class-confidences",
        default="NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15",
    )
    parser.add_argument("--person-match-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--person-match-conf-bypass", default=0.40, type=float)
    parser.add_argument("--person-box-expand-top", default=0.20, type=float)
    parser.add_argument("--person-box-expand-bottom", default=0.20, type=float)
    parser.add_argument("--person-box-expand-sides", default=0.10, type=float)
    parser.add_argument("--person-min-box-width", default=12.0, type=float)
    parser.add_argument("--person-min-box-height", default=45.0, type=float)
    parser.add_argument("--person-min-aspect-ratio", default=1.10, type=float)
    parser.add_argument("--person-max-aspect-ratio", default=8.00, type=float)
    parser.add_argument("--no-violation-min-person-height", default=45.0, type=float)
    parser.add_argument("--crop-margin", default=0.15, type=float)
    parser.add_argument("--overlap-gate-threshold", default=0.40, type=float)
    parser.add_argument("--detection-dedupe-iou", default=0.35, type=float)
    parser.add_argument("--person-dedupe-iou", default=0.45, type=float)
    parser.add_argument("--person-dedupe-coverage", default=0.80, type=float)
    parser.add_argument("--enable-crop-ppe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--rfdetr-size", default="medium", choices=["base", "small", "medium", "large"])
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def optimal_imgsz(shape: tuple[int, int], max_size: int) -> int:
    height, width = shape
    longer = max(height, width)
    rounded = int(math.ceil(longer / 32.0) * 32)
    return min(rounded, max_size)


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_overlap_fraction(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = box_area((ix1, iy1, ix2, iy2))
    area = box_area(inner)
    return inter / area if area > 0 else 0.0


def box_center_inside(box: tuple[float, float, float, float], container: tuple[float, float, float, float]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def box_coverage(smaller: tuple[float, float, float, float], larger: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(smaller[0], larger[0]), max(smaller[1], larger[1])
    ix2, iy2 = min(smaller[2], larger[2]), min(smaller[3], larger[3])
    inter = box_area((ix1, iy1, ix2, iy2))
    area = box_area(smaller)
    return inter / area if area > 0 else 0.0


def boxes_refer_to_same_person(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    iou_threshold: float,
    coverage_threshold: float,
) -> bool:
    if box_iou(a, b) >= iou_threshold:
        return True
    smaller, larger = (a, b) if box_area(a) <= box_area(b) else (b, a)
    return box_coverage(smaller, larger) >= coverage_threshold


def is_valid_person_box(box: tuple[float, float, float, float], args: argparse.Namespace) -> bool:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    if width < args.person_min_box_width or height < args.person_min_box_height:
        return False
    aspect = height / max(width, 1.0)
    return args.person_min_aspect_ratio <= aspect <= args.person_max_aspect_ratio


def dedupe_person_boxes(
    person_boxes: list[tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> list[tuple[float, float, float, float]]:
    kept: list[tuple[float, float, float, float]] = []
    for box in sorted(person_boxes, key=box_area, reverse=True):
        if box_area(box) <= 0 or not is_valid_person_box(box, args):
            continue
        if not any(
            boxes_refer_to_same_person(
                box,
                existing,
                args.person_dedupe_iou,
                args.person_dedupe_coverage,
            )
            for existing in kept
        ):
            kept.append(box)
    return kept


def dedupe_detections(detections: list[Detection], args: argparse.Namespace) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if not any(
            det.class_name == existing.class_name
            and box_iou(det.box, existing.box) >= args.detection_dedupe_iou
            for existing in kept
        ):
            kept.append(det)
    return kept


def expand_person_boxes(
    person_boxes: list[tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
    args: argparse.Namespace,
) -> list[tuple[float, float, float, float]]:
    expanded: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in person_boxes:
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        expanded.append((
            max(0.0, x1 - width * args.person_box_expand_sides),
            max(0.0, y1 - height * args.person_box_expand_top),
            min(float(frame_width), x2 + width * args.person_box_expand_sides),
            min(float(frame_height), y2 + height * args.person_box_expand_bottom),
        ))
    return expanded


def find_person_index_for_detection(
    det_box: tuple[float, float, float, float],
    person_boxes: list[tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> int | None:
    best_idx, best_overlap = None, 0.0
    for index, person_box in enumerate(person_boxes):
        overlap = box_overlap_fraction(det_box, person_box)
        if overlap >= args.overlap_gate_threshold and overlap > best_overlap:
            best_idx, best_overlap = index, overlap
    if best_idx is not None:
        return best_idx

    for index, person_box in enumerate(person_boxes):
        if box_center_inside(det_box, person_box):
            return index

    best_idx, best_iou = None, 0.10
    for index, person_box in enumerate(person_boxes):
        iou = box_iou(det_box, person_box)
        if iou > best_iou:
            best_idx, best_iou = index, iou
    return best_idx


def assign_detections_to_persons(
    detections: list[Detection],
    person_boxes: list[tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
    args: argparse.Namespace,
) -> list[Detection]:
    if not person_boxes:
        kept: list[Detection] = []
        for det in detections:
            if not args.person_match_required or det.confidence >= args.person_match_conf_bypass:
                det.person_index = -1
                kept.append(det)
        return kept

    expanded = expand_person_boxes(person_boxes, frame_width, frame_height, args)
    by_person_class: dict[tuple[int, str], list[Detection]] = {}
    for det in detections:
        pidx = det.person_index
        if pidx is None:
            pidx = find_person_index_for_detection(det.box, expanded, args)
        if pidx is None:
            if not args.person_match_required or det.confidence >= args.person_match_conf_bypass:
                pidx = -1
            else:
                continue
        det.person_index = pidx
        by_person_class.setdefault((pidx, det.class_name), []).append(det)

    kept: list[Detection] = []
    for (_pidx, class_name), group in by_person_class.items():
        limit = 2 if class_name == "NO-Safety Boots" else 1
        kept.extend(sorted(group, key=lambda item: item.confidence, reverse=True)[:limit])
    return kept


class YoloDetector:
    def __init__(self, model_path: Path, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def predict(self, image: Any, imgsz: int, conf: float, iou: float) -> list[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return []
        out: list[Detection] = []
        for box, class_id, score in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            cid = int(class_id)
            out.append(Detection(
                class_name=self.names.get(cid, CLASS_NAMES.get(cid, str(cid))),
                confidence=float(score),
                box=tuple(float(value) for value in box),
            ))
        return out


class RFDetrDetector:
    def __init__(self, model_path: Path, size: str):
        try:
            import rfdetr
        except ImportError as exc:
            raise ImportError(
                "RF-DETR is not installed. Install: pip install rfdetr supervision pycocotools"
            ) from exc

        class_name = {
            "base": "RFDETRBase",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "large": "RFDETRLarge",
        }[size]
        base_class = getattr(rfdetr, "RFDETRBase", None)
        model_class = getattr(rfdetr, class_name, base_class)
        if model_class is None:
            raise RuntimeError("Installed rfdetr package does not expose an RF-DETR model class.")
        checkpoint = self._resolve_checkpoint(model_path)
        if hasattr(model_class, "from_checkpoint"):
            self.model = model_class.from_checkpoint(str(checkpoint))
        else:
            try:
                self.model = model_class(pretrain_weights=str(checkpoint))
            except TypeError:
                self.model = model_class()
                if hasattr(self.model, "load"):
                    self.model.load(str(checkpoint))
                elif hasattr(self.model, "load_state_dict"):
                    raise RuntimeError(
                        "This RF-DETR version requires manual checkpoint loading; "
                        "use a version that supports pretrain_weights or model.load()."
                    )
        if hasattr(self.model, "optimize_for_inference"):
            try:
                self.model.optimize_for_inference()
            except Exception:
                pass

    @staticmethod
    def _resolve_checkpoint(path: Path) -> Path:
        if path.is_file():
            return path
        candidates: list[Path] = []
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            candidates.extend(path.rglob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No RF-DETR checkpoint found under: {path}")
        return sorted(candidates, key=lambda item: ("best" not in item.name.lower(), item.name))[0]

    def predict(self, image: Any, imgsz: int, conf: float, iou: float) -> list[Detection]:
        del imgsz, iou
        if getattr(image, "ndim", None) == 3 and getattr(image, "shape", [None, None, None])[2] == 3:
            image = image[..., ::-1].copy()
        try:
            detections = self.model.predict(image, threshold=conf)
        except TypeError:
            detections = self.model.predict(image)
        return self._normalize_detections(detections, conf)

    @staticmethod
    def _normalize_class_id(raw_id: int) -> int:
        if raw_id in CLASS_NAMES:
            return raw_id
        if raw_id - 1 in CLASS_NAMES:
            return raw_id - 1
        return raw_id

    def _normalize_detections(self, detections: Any, conf: float) -> list[Detection]:
        out: list[Detection] = []
        xyxy = getattr(detections, "xyxy", None)
        class_ids = getattr(detections, "class_id", None)
        confidences = getattr(detections, "confidence", None)
        if xyxy is None and isinstance(detections, dict):
            xyxy = first_present(detections, "xyxy", "boxes")
            class_ids = first_present(detections, "class_id", "class_ids", "labels")
            confidences = first_present(detections, "confidence", "confidences", "scores")
        if xyxy is None and isinstance(detections, list):
            for item in detections:
                if not isinstance(item, dict):
                    continue
                box = first_present(item, "xyxy", "box", "bbox")
                raw_class_id = first_present(item, "class_id", "label", "category_id")
                score = first_present(item, "confidence", "score")
                if box is None or raw_class_id is None:
                    continue
                cval = float(score if score is not None else 1.0)
                if cval < conf:
                    continue
                class_id = self._normalize_class_id(int(raw_class_id))
                out.append(Detection(
                    class_name=CLASS_NAMES.get(class_id, str(class_id)),
                    confidence=cval,
                    box=tuple(float(value) for value in box[:4]),
                ))
            return out
        if xyxy is None or class_ids is None:
            raise RuntimeError("Unsupported RF-DETR prediction output shape.")
        if confidences is None:
            confidences = [1.0] * len(xyxy)
        for box, raw_class_id, score in zip(xyxy, class_ids, confidences):
            cval = float(score)
            if cval < conf:
                continue
            class_id = self._normalize_class_id(int(raw_class_id))
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            out.append(Detection(
                class_name=class_name,
                confidence=cval,
                box=tuple(float(value) for value in box[:4]),
            ))
        return out


def load_detector(args: argparse.Namespace) -> Detector:
    if args.backend == "yolo":
        return YoloDetector(args.model, args.device)
    return RFDetrDetector(args.model, args.rfdetr_size)


def split_full_frame_detections(
    detections: list[Detection],
    class_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[tuple[float, float, float, float]], list[Detection]]:
    persons: list[tuple[float, float, float, float]] = []
    violations: list[Detection] = []
    for det in detections:
        if det.class_name == "Person":
            if det.confidence >= args.person_confidence:
                persons.append(det.box)
            continue
        if det.class_name not in VIOLATION_CLASS_NAMES:
            continue
        if det.confidence < class_thresholds.get(det.class_name, args.confidence):
            continue
        violations.append(det)
    return dedupe_person_boxes(persons, args), violations


def detect_on_person_crops(
    detector: Detector,
    frame: Any,
    person_boxes: list[tuple[float, float, float, float]],
    class_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> list[Detection]:
    frame_height, frame_width = frame.shape[:2]
    crop_dets: list[Detection] = []
    for pidx, person_box in enumerate(person_boxes):
        x1, y1, x2, y2 = person_box
        person_width = max(1.0, x2 - x1)
        person_height = max(1.0, y2 - y1)
        cx1 = max(0, int(x1 - person_width * args.crop_margin))
        cy1 = max(0, int(y1 - person_height * (args.crop_margin + 0.10)))
        cx2 = min(frame_width, int(x2 + person_width * args.crop_margin))
        cy2 = min(frame_height, int(y2 + person_height * args.crop_margin))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        detections = detector.predict(crop, args.crop_image_size, args.confidence, args.iou)
        for det in detections:
            if det.class_name not in VIOLATION_CLASS_NAMES:
                continue
            if det.confidence < class_thresholds.get(det.class_name, args.confidence):
                continue
            lx1, ly1, lx2, ly2 = det.box
            crop_dets.append(Detection(
                class_name=det.class_name,
                confidence=det.confidence,
                box=(lx1 + cx1, ly1 + cy1, lx2 + cx1, ly2 + cy1),
                person_index=pidx,
                source="person_crop",
            ))
    return crop_dets


def add_compliant_persons(
    detections: list[Detection],
    person_boxes: list[tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> list[Detection]:
    person_to_violations: dict[int, list[Detection]] = {}
    for det in detections:
        if det.person_index is not None:
            person_to_violations.setdefault(det.person_index, []).append(det)

    violating_person_boxes = [
        person_boxes[pidx]
        for pidx in person_to_violations
        if 0 <= pidx < len(person_boxes)
    ]
    compliant: list[Detection] = []
    for pidx, person_box in enumerate(person_boxes):
        if (person_box[3] - person_box[1]) < args.no_violation_min_person_height:
            continue
        if person_to_violations.get(pidx):
            continue
        if any(
            boxes_refer_to_same_person(
                person_box,
                violation_box,
                args.person_dedupe_iou,
                args.person_dedupe_coverage,
            )
            for violation_box in violating_person_boxes
        ):
            continue
        compliant.append(Detection(
            class_name="NO-Violation",
            confidence=1.0,
            box=person_box,
            person_index=pidx,
            source="person",
        ))
    return detections + compliant


def crop_box(frame: Any, box: tuple[float, float, float, float]) -> Any | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_crops(cv2: Any, save_dir: Path, image_stem: str, frame: Any, detections: list[Detection], persons: list[tuple[float, float, float, float]]) -> dict[str, list[str]]:
    crop_root = save_dir / "crops" / safe_name(image_stem)
    saved = {"persons": [], "detections": []}
    for pidx, person_box in enumerate(persons):
        crop = crop_box(frame, person_box)
        if crop is None:
            continue
        out = crop_root / "persons" / f"person_{pidx:03d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out), crop):
            saved["persons"].append(str(out))
    for didx, det in enumerate(detections):
        crop = crop_box(frame, det.box)
        if crop is None:
            continue
        out = crop_root / "detections" / f"det_{didx:03d}_{safe_name(det.class_name)}_pidx_{det.person_index}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out), crop):
            saved["detections"].append(str(out))
    return saved


def draw_results(cv2: Any, image: Any, detections: list[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.box]
        color = (0, 180, 0) if det.class_name == "NO-Violation" else (0, 0, 255)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.person_index is not None:
            label += f" p={det.person_index}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def detection_to_json(det: Detection) -> dict[str, Any]:
    return {
        "class_name": det.class_name,
        "confidence": round(det.confidence, 4),
        "box": [round(float(value), 1) for value in det.box],
        "person_index": det.person_index,
        "source": det.source,
    }


def main() -> int:
    args = parse_args()
    cache_root = Path("training/.cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((cache_root / "matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_root / "ultralytics").resolve()))

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model.resolve()}")
        return 1
    paths = image_paths(args.input)
    if not paths:
        print(f"[ERROR] No images found under: {args.input.resolve()}")
        return 1

    try:
        import cv2
    except ImportError as exc:
        print(f"[ERROR] Missing OpenCV: {exc}")
        return 1

    class_thresholds = parse_class_confidences(args.class_confidences)
    model_conf = min([args.confidence, args.person_confidence, *class_thresholds.values()])

    try:
        detector = load_detector(args)
    except Exception as exc:
        print(f"[ERROR] Failed to load detector: {exc}")
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    print(f"[INFO] backend={args.backend} images={len(paths)} save_dir={args.save_dir.resolve()}")

    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[SKIP] OpenCV could not read {path}")
            continue
        frame_height, frame_width = frame.shape[:2]
        imgsz = optimal_imgsz((frame_height, frame_width), args.image_size)

        full_dets = detector.predict(frame, imgsz, model_conf, args.iou)
        person_boxes, raw_violations = split_full_frame_detections(full_dets, class_thresholds, args)

        if args.enable_crop_ppe and person_boxes:
            for crop_det in detect_on_person_crops(detector, frame, person_boxes, class_thresholds, args):
                if not any(
                    crop_det.class_name == det.class_name and box_iou(crop_det.box, det.box) > 0.30
                    for det in raw_violations
                ):
                    raw_violations.append(crop_det)

        assigned = assign_detections_to_persons(raw_violations, person_boxes, frame_width, frame_height, args)
        assigned = dedupe_detections(assigned, args)
        detections = add_compliant_persons(assigned, person_boxes, args)

        annotated = frame.copy()
        draw_results(cv2, annotated, detections)
        out_image = args.save_dir / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(out_image), annotated)
        crops = save_crops(cv2, args.save_dir, path.stem, frame, detections, person_boxes)

        record = {
            "image": str(path),
            "annotated_image": str(out_image),
            "image_size": [frame_width, frame_height],
            "person_boxes": [[round(float(value), 1) for value in box] for box in person_boxes],
            "detections": [detection_to_json(det) for det in detections],
            "crops": crops,
        }
        all_results.append(record)
        summary = ", ".join(f"{det.class_name}({det.confidence:.2f})" for det in detections) or "none"
        print(f"[IMAGE] {path.name}: persons={len(person_boxes)} detections={summary}")

    results_path = args.save_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
