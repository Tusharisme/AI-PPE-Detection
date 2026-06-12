# PPE Experiment Tracker

This file tracks model and pipeline choices that helped, hurt, or need more evidence.

## Current Production-Favouring Choices

- Direct `NO-*` detections are better for production evidence.
  - Use `NO-Hardhat`, `NO-Safety Vest`, and `NO-Safety Boots` boxes from the model.
  - Do not draw synthetic "missing PPE" boxes when the screenshot/DB evidence needs tight geometry.

- Ignore mask classes for the current PPE workflow.
  - `Mask` and `NO-Mask` are not part of the production violation set.

- Use class-specific confidence thresholds.
  - Current local/prod command uses:
    - `NO-Hardhat=0.15`
    - `NO-Safety Vest=0.20`
    - `NO-Safety Boots=0.10`

- Keep worker-like per-person duplicate suppression.
  - One `NO-Hardhat` per person.
  - One `NO-Safety Vest` per person.
  - Up to two `NO-Safety Boots` per person.

- `IMAGE_SIZE=1920` helps small/far PPE objects.
  - Lower image sizes missed important small detections.

- `PPE_SNAPSHOT_INTERVAL=40` is more realistic than `10`.
  - KVS capture dominates runtime and can take 30+ seconds for 5 cameras.
  - Model inference itself is usually only a few seconds.

## What Did Not Work Well

- Synthetic missing-PPE boxes from positive-class absence.
  - Accuracy for compliance can look good, but boxes are not true object detections.
  - This is not ideal for production evidence images.

- Boot crop-only training/flooding.
  - It improved some boot cases but hurt broader PPE behaviour.
  - Full-frame onsite images should remain dominant.

- Adding boots data without enough onsite full-frame balance.
  - It often reduced the original `best.pt` model's strong `NO-Hardhat` / `NO-Safety Vest` behaviour.

## Current Important Models

- Current broad-camera production candidate:
  - `training/runs/ppe_broad_camera_repeat4_from_best_12ep_lr5e5_v1/weights/best.pt`
  - Stronger practical production baseline so far.

- YOLO11s scratch model:
  - `training/runs/ppe_latest_yolo11s_scratch_v1/weights/best.pt`
  - Trained from scratch on `ppe_latest_dataset`.
  - Early stopped at epoch 50.
  - Final best validation:
    - Overall `mAP50`: `0.574`
    - Overall `mAP50-95`: `0.292`
  - Weak classes on validation:
    - `NO-Hardhat`
    - `NO-Safety Boots`

## Next Evaluation Rule

Do not decide from validation metrics alone. Compare models on the same onsite image folder with production-style direct `NO-*` filtering:

```bash
.train-venv/bin/python compare_ppe_models.py \
  --images Images \
  --model-a training/runs/ppe_broad_camera_repeat4_from_best_12ep_lr5e5_v1/weights/best.pt \
  --model-a-name broad_camera \
  --model-b training/runs/ppe_latest_yolo11s_scratch_v1/weights/best.pt \
  --model-b-name yolo11_scratch \
  --confidence 0.10 \
  --image-size 1920 \
  --save-dir local_compare_broad_vs_yolo11_scratch
```

Inspect annotated images under:

- `local_compare_broad_vs_yolo11_scratch/broad_camera`
- `local_compare_broad_vs_yolo11_scratch/yolo11_scratch`

## Production A/B Deployment

Build and push the current broad-camera model:

```bash
sudo env PATH="$PATH" .venv/bin/python push_to_ecr.py \
  --creds ppe_creds.txt \
  --repository ai-ppe-detection \
  --tag broad-camera-v1 \
  --model-path training/runs/ppe_broad_camera_repeat4_from_best_12ep_lr5e5_v1/weights/best.pt
```

Build and push the YOLO11 scratch model:

```bash
sudo env PATH="$PATH" .venv/bin/python push_to_ecr.py \
  --creds ppe_creds.txt \
  --repository ai-ppe-detection \
  --tag yolo11-scratch-v1 \
  --model-path training/runs/ppe_latest_yolo11s_scratch_v1/weights/best.pt
```

Use the same runtime environment variables for both image tags, then compare DB/S3 output on the same camera set and time window.





Based on the deployment doc/config, the ECR image we set up is:

  577006449946.dkr.ecr.ap-south-1.amazonaws.com/ai-ppe-detection:ppe2-two-model-person033

  It contains two models:

   Purpose                   Model inside Docker     Source model
  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Person detector           /app/person_model.pt    yolov8n.pt
  ────────────────────────  ──────────────────────  ────────────────────────────────────────────────────────────────────────────────────────
   PPE violation detector    /app/best.pt            runs/detect/training/runs/ppe2_archive_4class_from_best_150ep_pat20_v1/weights/best.pt

  The deployed runtime is not a Lambda function. It is a Docker worker process.

  Docker runs:

  python ppe_worker.py

  The main deployed function flow is in ppe_worker.py:

  main()

  Inside that, the worker:

  1. Loads both models using init_models()
  2. Reads active PPE cameras from DB
  3. Captures CCTV/Kinesis snapshots
  4. Runs person detection first using yolov8n.pt
  5. Runs PPE violation detection using the trained PPE model
  6. Uploads violation frames to S3
  7. Updates MySQL tables with frame and detection records

  The deployed PPE classes are:

  NO-Hardhat
  NO-Safety Vest
  NO-Safety Boots

  The current documented production thresholds are:

  PERSON_CONFIDENCE=0.33
  CLASS_CONFIDENCES="NO-Hardhat=0.20,NO-Safety Vest=0.20,NO-Safety Boots=0.15"

  So in short: ECR contains the two-model CCTV PPE worker, with yolov8n.pt for person detection and our trained ppe2_archive_4class_from_best_150ep_pat20_v1
  model for PPE violations.