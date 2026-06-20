FROM python:3.11-slim

ARG MODEL_SOURCE=training/runs/ppe_v3_person_rfdetr_medium_1024_batch6_lr5e5_v2/checkpoint_best_ema.pth
ARG MODEL_TARGET=checkpoint_best_ema.pth
ARG FACE_MODEL_SOURCE=test_face_detection/yolov8n-face.pt
ARG WORKER_SOURCE=ppe_worker_5.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/${MODEL_TARGET} \
    PPE_MODEL_PATH=/app/${MODEL_TARGET} \
    PERSON_MODEL_PATH=/app/${MODEL_TARGET} \
    SINGLE_MODEL_MODE=TRUE \
    DETECTOR_BACKEND=rfdetr \
    RFDETR_MODEL_SIZE=medium \
    RFDETR_OPTIMIZE_INFERENCE=TRUE \
    FACE_MODEL_PATH=/app/yolov8n-face.pt \
    DETECTION_CONFIDENCE=0.05 \
    PERSON_CONFIDENCE=0.25 \
    PERSON_IMAGE_SIZE=1024 \
    ENABLE_TILED_PERSON_DETECTION=FALSE \
    ENABLE_CROWD_RECOVERY=FALSE \
    PERSON_FALLBACK_ENABLED=FALSE \
    PERSON_MIN_BOX_WIDTH=8 \
    PERSON_MIN_BOX_HEIGHT=20 \
    PERSON_MIN_ASPECT_RATIO=0.50 \
    PERSON_MAX_ASPECT_RATIO=10.00 \
    PPE_CROP_INFERENCE_SIZE=1024 \
    IMAGE_SIZE=1024 \
    CLASS_CONFIDENCES=NO-Hardhat=0.10,NO-Safety\ Vest=0.20,NO-Safety\ Boots=0.10 \
    CLOSE_PERSON_CONF_ENABLED=TRUE \
    CLOSE_PERSON_HEIGHT_RATIO=0.60 \
    CLOSE_PERSON_AREA_RATIO=0.22 \
    CLOSE_PERSON_CONF_BOOST=0.05 \
    CLOSE_PERSON_CONF_MAX=0.35 \
    PPE_OWNER_MIN_OVERLAP=0.20 \
    PPE_CROP_DUPLICATE_IOU=0.35 \
    NO_VIOLATION_RECHECK_ENABLED=TRUE \
    NO_VIOLATION_RECHECK_IMAGE_SIZE=960 \
    NO_VIOLATION_RECHECK_CONF_MULTIPLIER=1.25 \
    NO_VIOLATION_RECHECK_PERSON_CONF=0.65 \
    PPE_CREDS_FILE=/app/ppe_creds.txt \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps facenet-pytorch==2.6.0

COPY ${WORKER_SOURCE} /app/ppe_worker.py
COPY ppe_creds.txt ./
COPY ${MODEL_SOURCE} /app/${MODEL_TARGET}
COPY ${FACE_MODEL_SOURCE} /app/yolov8n-face.pt

RUN mkdir -p /tmp/ultralytics \
    && chown -R app:app /app /tmp/ultralytics /home/app
USER app

CMD ["python", "ppe_worker.py"]
