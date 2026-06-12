# PPE + Boots Fine-Tuning

This workspace fine-tunes the current `best.pt` with two added classes:

```text
Safety Boots
NO-Safety Boots
```

The production worker already saves any class containing `NO`, so no DB change
is needed for `NO-Safety Boots`.

## 1. Training Environment

```bash
python3 -m venv .train-venv
source .train-venv/bin/activate
export MPLCONFIGDIR="$PWD/training/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/training/.cache/ultralytics"
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
```

Verify GPU:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## 2. Expected Dataset Inputs

The original PPE dataset is placed at:

```text
training/datasets/ppe_original/
```

Download the boots-capable source dataset into:

```text
training/datasets/boots_raw/
```

Each source should be YOLO format and include a `data.yaml`, for example:

```text
dataset/
  data.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Use the Kaggle PPE dataset from the original notebook for `ppe_original`.
For boot/no-boot labels, use the Ultralytics Construction-PPE dataset:

```bash
source .train-venv/bin/activate
python training/scripts/download_ultralytics_construction_ppe_dataset.py
```

This dataset includes `boots` and `no_boots`; the merge script maps them to
`Safety Boots` and `NO-Safety Boots`.

If using a Roboflow export that only contains a `train` split, split it first:

```bash
source .train-venv/bin/activate
python training/scripts/split_single_split_yolo_dataset.py \
  --source "training/datasets/safety shoe.yolov8" \
  --output training/datasets/safety_shoe_split
```

For the current safety-shoe dataset, use `training/datasets/safety_shoe_split/data.yaml`
as the boots dataset input.

## 3. Merge Small Dataset

The merger preserves the existing 10 PPE class IDs and appends:

```text
10: Safety Boots
11: NO-Safety Boots
```

Run:

```bash
source .train-venv/bin/activate
export MPLCONFIGDIR="$PWD/training/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/training/.cache/ultralytics"
python training/scripts/prepare_ppe_boots_dataset.py \
  --ppe-data training/datasets/ppe_original/data.yaml \
  --boots-data training/datasets/safety_shoe_split/data.yaml \
  --output training/datasets/merged_safety_shoe_balanced \
  --boots-train-limit 0 \
  --boots-val-limit 0 \
  --boots-test-limit 0 \
  --boots-only \
  --boots-repeat 5
```

By default, images are symlinked to save disk. Add `--copy` only if the source
folders need to be moved later. `--boots-only` avoids importing unrelated
classes from the shoe dataset, and `--boots-repeat` gives the new boot classes
more training weight without duplicating image bytes.

## 4. Small Fine-Tune

```bash
source .train-venv/bin/activate
export MPLCONFIGDIR="$PWD/training/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/training/.cache/ultralytics"
python training/scripts/train_ppe_boots_small.py \
  --model best.pt \
  --data training/datasets/merged_safety_shoe_balanced/data.yaml \
  --epochs 50 \
  --imgsz 1536 \
  --batch 2 \
  --device 0 \
  --lr0 0.002 \
  --name ppe_safety_shoe_balanced_1536_v1
```

If VRAM is comfortable, try `--batch 4`. If training is too slow or unstable,
try `--imgsz 1280`.

## 5. Validate On Local Images

```bash
python test_image_ppe.py \
  --image Images/2.png \
  --model training/runs/ppe_safety_shoe_balanced_1536_v1/weights/best.pt \
  --confidence 0.10 \
  --image-size 1536 \
  --violations-only \
  --save-dir local_finetune_test
```

Acceptance:

- Existing `NO-Hardhat`, `NO-Mask`, and `NO-Safety Vest` still work.
- `NO-Safety Boots` appears on clear no-boot examples.
- False positives are acceptable at `DETECTION_CONFIDENCE=0.25`.

## 6. Promote To Docker

After validation:

```bash
cp training/runs/ppe_boots_small_v1/weights/best.pt best.pt
sudo env PATH="$PATH" .venv/bin/python push_to_ecr.py \
  --creds ppe_creds.txt \
  --repository ai-ppe-detection \
  --tag boots-small-v1
```

## Current PPE 2 YOLOv8n 4-Class Run

The `ppe 2.v2i.yolov8` Roboflow export has 7 raw classes. Production only
needs 4 worker-compatible classes:

```text
0 Person
1 NO-Hardhat
2 NO-Safety Vest
3 NO-Safety Boots
```

Build the normalized dataset:

```bash
source .train-venv/bin/activate
python training/scripts/build_four_class_ppe_dataset.py \
  --source-data "ppe 2.v2i.yolov8/data.yaml" \
  --output training/datasets/ppe2_4class_yolov8n \
  --overwrite
```

Train YOLOv8n:

```bash
source .train-venv/bin/activate
python training/scripts/train_yolov8n_four_class.py \
  --data training/datasets/ppe2_4class_yolov8n/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --name ppe2_4class_yolov8n_v1
```

Use `--device cpu` only for a quick smoke run; full training should run on a
CUDA GPU. The output weights will be:

```text
training/runs/ppe2_4class_yolov8n_v1/weights/best.pt
```
