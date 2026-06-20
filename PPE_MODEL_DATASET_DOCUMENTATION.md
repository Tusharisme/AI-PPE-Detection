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

Historical pretrained base model:

```text
best.pt
```

This was the older PPE model with 10 classes:

```text
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
Safety Cone, Safety Vest, machinery, vehicle
```

For the new training run, YOLO changed the model head from 10 classes to 4 classes and transferred compatible weights. The old top-level `best.pt` file is no longer kept in the cleaned runtime repo; the active production PPE model is the trained 4-class `best.pt` under the run directory shown below.

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

The current local validation entrypoint is `run_ppe_worker_4_local_images.py`. It runs the same worker logic without writing to MySQL, S3, or Welspun.

```bash
.train-venv/bin/python run_ppe_worker_4_local_images.py \
  --input Images \
  --save-dir local_ppe_worker_4_strict_images \
  --person-confidence 0.45 \
  --no-enable-tiled-person-detection \
  --no-enable-crowd-recovery
```

To match the current stricter EC2 settings and recover the far-person hardhat case tested on `Images/2.png`, pass these environment overrides:

```bash
PERSON_FALLBACK_ENABLED=FALSE PPE_CROP_INFERENCE_SIZE=960 \
.train-venv/bin/python run_ppe_worker_4_local_images.py \
  --input Images/2.png \
  --save-dir local_ppe_worker_4_image2_no_fallback_crop960 \
  --person-confidence 0.45 \
  --no-enable-tiled-person-detection \
  --no-enable-crowd-recovery
```

The local runner writes annotated images, crop images, and `results.json` under the requested `--save-dir`. Generated local inference folders are not part of the cleaned runtime repository.

## Inference Logic

For local image testing and the live worker, the production pipeline uses these models:

| Purpose | Local Path | Image Path |
| --- | --- | --- |
| PPE violation model | `runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt` | `/app/best.pt` |
| Person model | `runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt` | `/app/person_model.pt` |
| Face model | `test_face_detection/yolov8n-face.pt` | `/app/yolov8n-face.pt` |

Current worker flow:

1. `ppe_worker_4.py` detects people using the custom person model.
2. The PPE model detects `NO-Hardhat`, `NO-Safety Vest`, and `NO-Safety Boots` on the full frame.
3. The PPE model also runs on expanded per-person crops when `ENABLE_CROP_PPE=TRUE`.
4. PPE detections that cannot be assigned to a detected person are dropped unless the configured confidence bypass allows them.
5. People with no assigned PPE violation are saved as `NO-Violation` rows so the frontend can draw green boxes.
6. Re-ID uses the face model and FaceNet where a usable face crop exists.
7. The live worker uploads frames/crops to S3, writes frame/detection rows to MySQL, and can fire-and-forget POST the same frame data to Welspun DataHub.

## Current Production Runtime Settings

Current Docker image build inputs:

```bash
docker build \
  --build-arg WORKER_SOURCE=ppe_worker_4.py \
  --build-arg MODEL_SOURCE=runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt \
  --build-arg PERSON_MODEL_SOURCE=runs/detect/training/runs/ppe_person_yolov8n_finetune_v1/weights/best.pt \
  --build-arg FACE_MODEL_SOURCE=test_face_detection/yolov8n-face.pt \
  -t ai-ppe-detection:latest .
```

Recommended strict runtime settings:

| Setting | Value | Reason |
| --- | --- | --- |
| `PERSON_CONFIDENCE` | `0.45` | Reduces false person boxes. |
| `PERSON_IMAGE_SIZE` | `1280` | Current person inference size. |
| `PERSON_IOU` | `0.70` | Current person NMS IoU. |
| `PERSON_FALLBACK_ENABLED` | `FALSE` | Avoids high-resolution fallback false positives seen in local testing. |
| `ENABLE_TILED_PERSON_DETECTION` | `FALSE` | Avoids extra false positives in wide frames. |
| `ENABLE_CROWD_RECOVERY` | `FALSE` | Local testing produced more false detections than the strict profile. |
| `PPE_CROP_INFERENCE_SIZE` | `960` | Helps recover far-person hardhat misses from crop inference. |
| `ENABLE_CROP_PPE` | `TRUE` | Keeps per-person crop PPE recovery enabled. |
| `IMAGE_SIZE` | `1920` | Full-frame PPE inference size. |
| `CLASS_CONFIDENCES` | `NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15` | Current class thresholds. |

Other active runtime settings:

| Setting | Value |
| --- | --- |
| `DETECTION_IOU` | 0.45 |
| `PPE_SNAPSHOT_INTERVAL` | 40 seconds |
| `DRY_RUN` | FALSE |
| `ENABLE_BOOT_CROPS` | FALSE |
| `ENABLE_BOOT_COLOR_CHECK` | FALSE |
| `WELSPUN_WEBHOOK_ENABLED` | TRUE in Welspun test/prod runs, FALSE for local tests |

## Docker/ECR Deployment

Build and push the current worker image to ECR:

```bash
python3 push_to_ecr.py \
  --creds ppe_creds.txt \
  --repository ai-ppe-detection \
  --tag ppe2-two-model-person033 \
  --model-path training/runs/ppe_v3_person_yolov8s_single_gpu_batch6_foreground_v1/weights/best.pt \
  --worker-source ppe_worker_4.py
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
  -e DRY_RUN=FALSE \
  -e DEBUG=FALSE \
  -e SINGLE_MODEL_MODE=TRUE \
  -e PPE_MODEL_PATH=/app/best.pt \
  -e PERSON_MODEL_PATH=/app/best.pt \
  -e PERSON_CONFIDENCE=0.33 \
  -e PERSON_IMAGE_SIZE=1920 \
  -e PERSON_IOU=0.70 \
  -e PERSON_FALLBACK_ENABLED=FALSE \
  -e ENABLE_TILED_PERSON_DETECTION=FALSE \
  -e ENABLE_CROWD_RECOVERY=FALSE \
  -e PPE_CROP_INFERENCE_SIZE=960 \
  -e IMAGE_SIZE=1920 \
  -e CLASS_CONFIDENCES="NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15" \
  -e WELSPUN_WEBHOOK_ENABLED=true \
  -e WELSPUN_WEBHOOK_BASE_URL=https://welappsuat.welspun.com \
  -e WELSPUN_WEBHOOK_AUTH_KEY='U2blpNYCc8cIdS2ZpNd7' \
  -e WELSPUN_PRESIGN_EXPIRY=604800 \
  "$ECR_IMAGE_URI"

sudo docker logs -f ppe-worker
```

## Welspun DataHub Webhook

`ppe_worker_4.py` can push one payload per saved PPE frame to Welspun DataHub after the local DB commit. This is fire-and-forget: failures are logged and do not stop the detection loop.

Endpoint:

```text
POST https://welappsuat.welspun.com/webhooks/report_ppe_violation
```

Runtime variables:

| Variable | Value |
| --- | --- |
| `WELSPUN_WEBHOOK_ENABLED` | `true` to send, `false` to disable |
| `WELSPUN_WEBHOOK_BASE_URL` | `https://welappsuat.welspun.com` |
| `WELSPUN_WEBHOOK_AUTH_KEY` | Provided webhook key |
| `WELSPUN_PRESIGN_EXPIRY` | `604800` seconds |

Image URLs are presigned before sending. HTTP `409` is treated as success because it means the frame was already received.

Welspun project ID mapping is applied only in the webhook payload:

| Local Project ID | Welspun Project ID |
| ---: | ---: |
| 1 | 6428 |
| 2 | 6427 |

The local MySQL writes keep the local project IDs unchanged.

## Cleaned Runtime Repository

The cleaned runtime repo intentionally keeps only the current worker surface and active models.

Kept:

- `ppe_worker_4.py`
- `run_ppe_worker_4_local_images.py`
- `push_to_ecr.py`
- `Dockerfile`
- `docker-compose.yml`
- `Images/`
- `test_face_detection/`
- the two active runtime model weights under `runs/detect/training/runs/.../weights/best.pt`

Removed/generated paths are not required for current production runtime:

- old workers `ppe_worker.py`, `ppe_worker_2.py`, `ppe_worker_3.py`
- old local test scripts
- generated `local_ppe_worker_4_*` output folders
- unused top-level weights such as `best.pt`, `yolov8n.pt`, `yolov8s.pt`, `yolo26n.pt`
- unused training-run `last.pt` files

## Important Notes

- `NO-Safety Boots` is the weakest class in validation metrics because it has fewer examples and comes only from the CCTV dataset.
- The production image includes `/app/best.pt`, `/app/person_model.pt`, and `/app/yolov8n-face.pt`; do not mount over those paths unless intentionally changing models.
- The model performs best when violations are person-gated, because full-frame violation detections can otherwise attach to background objects.
- Increasing `PERSON_CONFIDENCE` reduces false person boxes but can also remove small/far workers, which then removes their assigned violations.
- `PERSON_FALLBACK_ENABLED=FALSE` is recommended for the current strict profile because high-resolution fallback produced false person boxes in local testing.
- `PPE_CROP_INFERENCE_SIZE=960` recovered a hardhat miss on a far person in `Images/2.png`; the same frame still missed the vest because the PPE model did not emit a `NO-Safety Vest` detection for that person.
- `NO-Violation` rows are intentionally saved for detected people with no assigned PPE violation so the frontend can render green boxes.
- The runtime class thresholds should be tuned from real CCTV results, not only validation metrics.

## RF-DETR Training

The active RF-DETR dataset is the same four-class single-detector dataset used by the YOLOv8s run:

```text
training/datasets/ppe_v3_person_single_detector
```

It contains RF-DETR/COCO folders:

```text
train/_annotations.coco.json
valid/_annotations.coco.json
test/_annotations.coco.json
```

Install RF-DETR training dependencies:

```bash
.train-venv/bin/python -m pip install --upgrade --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1

.train-venv/bin/python -m pip install --upgrade "rfdetr[train]" tensorboard "numpy<2"
```

Run RF-DETR Medium training:

```bash
.train-venv/bin/python training/scripts/train_rfdetr_medium_single_detector.py \
  --dataset-dir training/datasets/ppe_v3_person_single_detector \
  --output-dir training/runs/ppe_v3_person_rfdetr_medium_1024_batch2_accum8_v1 \
  --results-csv training/runs/ppe_v3_person_rfdetr_medium_1024_batch2_accum8_v1/results.csv \
  --device cuda \
  --epochs 150 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --resolution 1024 \
  --lr 0.0001 \
  --lr-encoder 0.00015 \
  --num-workers 4 \
  --early-stopping \
  --early-stopping-patience 25 \
  --tensorboard \
  --run-test
```

The script writes `run_args.json`, `train_kwargs.json`, RF-DETR checkpoints, TensorBoard event logs, and `results.csv` under the output directory. If `1024` resolution runs out of GPU memory, rerun with `--batch-size 1 --grad-accum-steps 16` to keep the effective batch size at 16.
