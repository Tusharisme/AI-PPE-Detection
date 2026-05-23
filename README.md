# AI PPE Detection Worker

Production Docker worker for OfficeLens PPE violation detection. The service
uses `best.pt`, reads selected cameras from MySQL, consumes AWS Kinesis Video
Streams, uploads violation frames to S3, and stores normalized detection records.

## Runtime Flow

1. Load bundled `ppe_creds.txt`.
2. Load YOLO model from `best.pt`.
3. Read selected cameras from `OfficeLens_ppe_cameras`.
4. Use each `cam_id` as the KVS stream name.
5. Capture one snapshot per camera per cycle.
6. Store only detections whose class name contains `NO`.
7. Upload violation frames to `s3://officelens-ppe/ppe/{cam_id}/...jpg`.
8. Insert frame and detection rows into MySQL.

## Production DB Setup

Create the PPE tables before starting the worker:

```bash
python setup_ppe_tables.py --creds ppe_creds.txt --dry-run
python setup_ppe_tables.py --creds ppe_creds.txt
```

The setup script is intentionally excluded from the Docker image.

Tables created:

- `OfficeLens_ppe_cameras`: selected PPE cameras; only stores `cam_id`.
- `OfficeLens_ppe_frames`: one row per uploaded violation frame.
- `OfficeLens_ppe_detections`: one row per violation bounding box.

## Credentials

`ppe_creds.txt` uses the same INI format as the attendance service:

```ini
[AWS]
aws_access_key_id = ...
aws_secret_access_key = ...
region_name = ap-south-1
s3_bucket = officelens-ppe

[DB]
db_type = mysql
db_host = ...
db_port = 3306
db_user = ...
db_password = ...
db_name = ...

cameras_table = OfficeLens_cameras
ppe_cameras_table = OfficeLens_ppe_cameras
ppe_frames_table = OfficeLens_ppe_frames
ppe_detections_table = OfficeLens_ppe_detections
```

The Docker image copies `ppe_creds.txt` into `/app/ppe_creds.txt`, as requested.
Restrict ECR access because the image contains production credentials.

## Docker

Build:

```bash
docker build -t ai-ppe-detection:latest .
```

Run:

```bash
docker run -d \
  --name ppe-worker \
  --restart unless-stopped \
  ai-ppe-detection:latest
```

Docker Compose:

```bash
docker compose up --build
```

Follow logs for a background container:

```bash
docker logs -f ppe-worker
```

For a production-path local test without S3/DB writes:

```bash
DRY_RUN=TRUE docker compose up --build
```

When the logs look correct and you want real S3 uploads plus DB inserts:

```bash
DRY_RUN=FALSE docker compose up --build
```

## Local KVS Model Test

Use this when you want to check model performance without S3 uploads or DB
writes:

```bash
python3 test_kvs_ppe.py \
  --creds ppe_creds.txt \
  --stream-name <cam_id> \
  --frames 10 \
  --save-dir local_ppe_test
```

For PPE, `<cam_id>` should be the KVS stream name because production uses
`OfficeLens_cameras.id` as the stream name. The script prints detections as JSON
and saves annotated frames plus `results.jsonl` under `local_ppe_test/`.

To focus only on violations:

```bash
python3 test_kvs_ppe.py \
  --creds ppe_creds.txt \
  --stream-name <cam_id> \
  --frames 10 \
  --violations-only \
  --save-dir local_ppe_test
```

## ECR Push

Use the Python helper when the AWS CLI is unavailable:

```bash
python3 -m pip install boto3
python3 push_to_ecr.py --creds ppe_creds.txt --repository ai-ppe-detection --tag latest
```

The helper creates the ECR repository if missing, logs Docker into ECR, builds
the image, tags it, and pushes it.

## Runtime Configuration

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `PPE_CREDS_FILE` | `ppe_creds.txt` | Credentials file path |
| `MODEL_PATH` | `./best.pt` | YOLO model path |
| `PPE_SNAPSHOT_INTERVAL` | `10` | Seconds between snapshot cycles |
| `CONNECTION_TIMEOUT` | `10` | KVS frame capture timeout |
| `CAPTURE_WORKERS` | `10` | Parallel camera capture workers |
| `UPLOAD_WORKERS` | `10` | Parallel S3 upload workers |
| `DETECTION_BATCH_SIZE` | `8` | Detection loop batch chunk size |
| `DETECTION_CONFIDENCE` | `0.25` | YOLO confidence threshold |
| `DETECTION_IOU` | `0.45` | YOLO IoU threshold |
| `IMAGE_SIZE` | `1280` | YOLO inference image size |
| `DEBUG` | `FALSE` | Verbose logs |
| `DRY_RUN` | `FALSE` | Skip S3 uploads and DB inserts while keeping KVS/model flow |

## Files

```text
.
├── ppe_worker.py          # Worker entrypoint used by Docker
├── setup_ppe_tables.py    # One-time production DB setup, excluded from image
├── ppe_creds.txt          # Bundled runtime credentials
├── best.pt                # PPE detection model
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
