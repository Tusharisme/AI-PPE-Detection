# PPE 4-Class Model And Dataset Documentation

This document describes the dataset, preprocessing, training run, model output, and runtime configuration for the current PPE violation model.

## Objective

The goal was to train a YOLOv8 model for production PPE violation detection on CCTV/site camera frames.

The final model predicts only these 4 classes:

| ID | Class |
| ---: | --- |
| 0 | Person |
| 1 | NO-Hardhat |
| 2 | NO-Safety Vest |
| 3 | NO-Safety Boots |

The production logic uses detected people to gate violation detections, so only violations assigned to a person are kept.

## Source Datasets

Two datasets were used to create the final training dataset.

### 1. CCTV PPE Dataset

Path:

```text
ppe 2.v2i.yolov8/data.yaml
```

Original classes:

| Original ID | Original Class | Final Class Used |
| ---: | --- | --- |
| 0 | No hardhat | NO-Hardhat |
| 1 | No safety Boots | NO-Safety Boots |
| 2 | No safety vest | NO-Safety Vest |
| 3 | Safety Boots | Not used |
| 4 | Safety Vest | Not used |
| 5 | hardhat | Not used |
| 6 | person | Person |

This dataset is the only source that contains `NO-Safety Boots`.

### 2. Archive PPE Dataset

Path:

```text
archive/data.yaml
```

Original classes:

```text
Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask, NO-Gloves,
NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
Safety Cone, Safety Vest
```

Classes used from this dataset:

| Original ID | Original Class | Final Class Used |
| ---: | --- | --- |
| 8 | NO-Hardhat | NO-Hardhat |
| 10 | NO-Safety Vest | NO-Safety Vest |
| 11 | Person | Person |

The archive dataset was used to add general PPE context for person, no-hardhat, and no-vest detections. It did not contain boot violation labels.

## Dataset Build Process

The merged 4-class dataset was built with:

```bash
MPLCONFIGDIR=training/.cache/matplotlib \
YOLO_CONFIG_DIR=training/.cache/ultralytics \
.train-venv/bin/python training/scripts/build_four_class_ppe_dataset.py \
  --archive-data archive/data.yaml \
  --camera-data "ppe 2.v2i.yolov8/data.yaml" \
  --output training/datasets/ppe2_archive_4class_from_best \
  --archive-train-limit 4000 \
  --camera-repeat 4 \
  --overwrite
```

Important choices:

- The output was normalized to 4 classes only.
- Archive train images were capped at 4000.
- CCTV training data was repeated 4 times to give more weight to real site/camera data.
- CCTV validation and test splits were not repeated.
- Images were linked/copied into YOLO format under `training/datasets/ppe2_archive_4class_from_best`.

Final dataset path:

```text
training/datasets/ppe2_archive_4class_from_best/data.yaml
```

## CCTV Dataset Counts

These counts are from the raw CCTV source dataset `ppe 2.v2i.yolov8`, mapped to the final 4 classes.

`labels` means bounding boxes. `images` means images containing at least one object of that class.

| Class | Labels | Images |
| --- | ---: | ---: |
| Person | 7,254 | 1,323 |
| NO-Hardhat | 5,698 | 1,225 |
| NO-Safety Vest | 5,820 | 1,255 |
| NO-Safety Boots | 2,922 | 849 |

Total CCTV images across train/valid/test:

```text
1,363
```

Raw CCTV split counts:

| Split | Images | Person Labels | NO-Hardhat Labels | NO-Safety Vest Labels | NO-Safety Boots Labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,200 | 6,768 | 5,292 | 5,405 | 2,749 |
| valid | 152 | 458 | 382 | 391 | 161 |
| test | 11 | 28 | 24 | 24 | 12 |

## Final Merged Dataset Counts

Path:

```text
training/datasets/ppe2_archive_4class_from_best
```

Overall class counts:

| Class | Labels | Images |
| --- | ---: | ---: |
| Person | 24,426 | 4,354 |
| NO-Hardhat | 20,224 | 4,431 |
| NO-Safety Vest | 19,619 | 4,149 |
| NO-Safety Boots | 9,651 | 2,799 |

Split-level counts:

| Split | Images | Person Labels | NO-Hardhat Labels | NO-Safety Vest Labels | NO-Safety Boots Labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,996 | 22,853 | 18,958 | 18,340 | 8,972 |
| val | 353 | 1,545 | 1,242 | 1,255 | 667 |
| test | 11 | 28 | 24 | 24 | 12 |

Training split source breakdown:

| Source | Images | Person Labels | NO-Hardhat Labels | NO-Safety Vest Labels | NO-Safety Boots Labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| archive | 4,000 | 129 | 1,230 | 176 | 0 |
| camera/CCTV repeated | 3,996 | 22,724 | 17,728 | 18,164 | 8,972 |

Validation and test data came from the CCTV dataset only.

## Training Setup

Pretrained base model:

```text
best.pt
```

This was the older PPE model with 10 classes:

```text
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
Safety Cone, Safety Vest, machinery, vehicle
```

For the new training run, YOLO changed the model head from 10 classes to 4 classes and transferred compatible weights.

Training command:

```bash
MPLCONFIGDIR=training/.cache/matplotlib \
YOLO_CONFIG_DIR=training/.cache/ultralytics \
.train-venv/bin/python training/scripts/train_yolov8n_four_class.py \
  --data training/datasets/ppe2_archive_4class_from_best/data.yaml \
  --model best.pt \
  --epochs 150 \
  --patience 20 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.05 \
  --name ppe2_archive_4class_from_best_150ep_pat20_v1
```

Run directory:

```text
runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1
```

Best weights:

```text
runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt
```

## Final Training Result

Training used early stopping:

```text
max epochs: 150
patience: 20
stopped at: epoch 91
best epoch: epoch 71
```

Best epoch metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.66968 |
| Recall | 0.51770 |
| mAP50 | 0.51741 |
| mAP50-95 | 0.22826 |

Final validation report from the best model:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.670 | 0.518 | 0.518 | 0.228 |
| Person | 0.823 | 0.739 | 0.766 | 0.408 |
| NO-Hardhat | 0.563 | 0.424 | 0.370 | 0.122 |
| NO-Safety Vest | 0.791 | 0.661 | 0.689 | 0.314 |
| NO-Safety Boots | 0.502 | 0.249 | 0.245 | 0.0695 |

## Local Image Validation

The best model was tested with the person-gated inference script:

```bash
.train-venv/bin/python test_person_gated_violations.py \
  --input Images \
  --model runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt \
  --save-dir local_person_gated_ppe2_archive_best_150ep
```

Output path:

```text
local_person_gated_ppe2_archive_best_150ep
```

This generated 13 annotated images, `summary.csv`, and `results.json`.

## Inference Logic

For local image testing and the live worker, the production pipeline now uses two models:

1. A COCO person detector (`yolov8n.pt`) detects `person` boxes above `PERSON_CONFIDENCE`.
2. The trained PPE model detects only `NO-Hardhat`, `NO-Safety Vest`, and `NO-Safety Boots` on the full frame.
3. PPE detections that cannot be assigned to a detected person are dropped.
4. The worker keeps the best detection per person/class, with up to two `NO-Safety Boots` boxes per person.
5. Optional boot crop/color checks can run after person assignment when enabled.
6. The live worker uploads only frames with violations to S3 and writes frame/detection rows to MySQL.

## Current Production Runtime Settings

Current Docker Compose command pattern:

```bash
MODEL_SOURCE="runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt" \
PERSON_MODEL_SOURCE="yolov8n.pt" \
PERSON_CONFIDENCE=0.33 \
PERSON_IMAGE_SIZE=1280 \
CLASS_CONFIDENCES="NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15" \
DRY_RUN=FALSE \
docker compose up --build
```

Effective runtime thresholds:

| Class | Confidence |
| --- | ---: |
| Person model `person` | 0.33 |
| PPE model `NO-Hardhat` | 0.20 |
| PPE model `NO-Safety Vest` | 0.20 |
| PPE model `NO-Safety Boots` | 0.15 |

Other important runtime settings:

| Setting | Value |
| --- | --- |
| `IMAGE_SIZE` | 1920 |
| `PERSON_IMAGE_SIZE` | 1280 |
| `PERSON_IOU` | 0.45 |
| `DETECTION_IOU` | 0.45 |
| `PPE_SNAPSHOT_INTERVAL` | 40 seconds |
| `DRY_RUN` | FALSE |
| `ENABLE_BOOT_CROPS` | FALSE |

## Docker/ECR Deployment

Build and push the production two-model image to ECR:

```bash
python3 push_to_ecr.py \
  --creds ppe_creds.txt \
  --repository ai-ppe-detection \
  --tag ppe2-two-model-person033 \
  --model-path runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt \
  --person-model-path yolov8n.pt
```

Run on EC2:

```bash
ECR_IMAGE_URI="577006449946.dkr.ecr.ap-south-1.amazonaws.com/ai-ppe-detection:ppe2-two-model-person033"
AWS_REGION="ap-south-1"

aws ecr get-login-password --region "$AWS_REGION" | \
  sudo docker login --username AWS --password-stdin 577006449946.dkr.ecr.ap-south-1.amazonaws.com

sudo docker pull "$ECR_IMAGE_URI"

sudo docker stop ppe-worker || true
sudo docker rm ppe-worker || true

sudo docker run -d \
  --name ppe-worker \
  --restart unless-stopped \
  -e PPE_SNAPSHOT_INTERVAL=40 \
  -e DETECTION_CONFIDENCE=0.10 \
  -e PERSON_CONFIDENCE=0.33 \
  -e PERSON_IMAGE_SIZE=1280 \
  -e PERSON_IOU=0.45 \
  -e IMAGE_SIZE=1920 \
  -e CLASS_CONFIDENCES="NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15" \
  -e DRY_RUN=FALSE \
  "$ECR_IMAGE_URI"

sudo docker logs -f ppe-worker
```

## S3 Backfill Inference

Use `backfill_s3_ppe_inference.py` when frames are already present in S3 and
only DB rows need to be created. The script lists S3 images, infers `cam_id`
from the key folder, runs the same two-model person-gated pipeline, and inserts
the existing S3 URL into `OfficeLens_ppe_frames`.

Dry run on a small sample:

```bash
.train-venv/bin/python backfill_s3_ppe_inference.py \
  --creds ppe_creds.txt \
  --prefix "ppe/" \
  --limit 20 \
  --dry-run \
  --summary-json local_s3_ppe_backfill_dry_run.json
```

Write detected violation frames to DB:

```bash
.train-venv/bin/python backfill_s3_ppe_inference.py \
  --creds ppe_creds.txt \
  --prefix "ppe/" \
  --summary-json local_s3_ppe_backfill.json
```

If you want to process one camera folder only:

```bash
.train-venv/bin/python backfill_s3_ppe_inference.py \
  --creds ppe_creds.txt \
  --prefix "ppe/<cam_id>/"
```

By default the script uses S3 `LastModified` for the DB timestamp and skips
`frame_url` values that already exist in the PPE frames table.

## Important Notes

- `NO-Safety Boots` is the weakest class in validation metrics because it has fewer examples and comes only from the CCTV dataset.
- The production image includes both `/app/person_model.pt` and `/app/best.pt`; do not mount over those paths unless intentionally changing models.
- The model performs best when violations are person-gated, because full-frame violation detections can otherwise attach to background objects.
- Increasing `PERSON_CONFIDENCE` reduces false person boxes but can also remove small/far workers, which then removes their assigned violations.
- The runtime class thresholds should be tuned from real CCTV results, not only validation metrics.
