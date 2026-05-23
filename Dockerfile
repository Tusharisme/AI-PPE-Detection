FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/best.pt \
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
    && python -m pip install -r requirements.txt

COPY ppe_worker.py best.pt ppe_creds.txt ./

RUN mkdir -p /tmp/ultralytics \
    && chown -R app:app /app /tmp/ultralytics /home/app
USER app

CMD ["python", "ppe_worker.py"]
