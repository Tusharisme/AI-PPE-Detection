FROM python:3.11-slim

ARG MODEL_SOURCE=runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt
ARG PERSON_MODEL_SOURCE=runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt
ARG FACE_MODEL_SOURCE=test_face_detection/yolov8n-face.pt
ARG WORKER_SOURCE=ppe_worker_4.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/best.pt \
    PPE_MODEL_PATH=/app/best.pt \
    PERSON_MODEL_PATH=/app/person_model.pt \
    FACE_MODEL_PATH=/app/yolov8n-face.pt \
    PERSON_CONFIDENCE=0.45 \
    ENABLE_TILED_PERSON_DETECTION=FALSE \
    ENABLE_CROWD_RECOVERY=FALSE \
    PERSON_MIN_BOX_WIDTH=12 \
    PERSON_MIN_BOX_HEIGHT=45 \
    PERSON_MIN_ASPECT_RATIO=1.10 \
    PERSON_MAX_ASPECT_RATIO=8.00 \
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
COPY ${MODEL_SOURCE} /app/best.pt
COPY ${PERSON_MODEL_SOURCE} /app/person_model.pt
COPY ${FACE_MODEL_SOURCE} /app/yolov8n-face.pt

RUN mkdir -p /tmp/ultralytics \
    && chown -R app:app /app /tmp/ultralytics /home/app
USER app

CMD ["python", "ppe_worker.py"]
