# Face Detection Test for PPE Violations

This folder contains a test script that extends the KVS PPE testing to include face detection for violators.

## Overview

The `test_kvs_face_detection.py` script performs the following pipeline:

1. **Connect to KVS**: Reads live video stream from AWS Kinesis Video Streams
2. **Person Detection**: Uses YOLO person model to detect people in frames
3. **PPE Violation Detection**: Uses PPE model to detect violations (NO-Hardhat, NO-Safety Vest, NO-Safety Boots)
4. **Face Detection**: Uses YOLO face model to detect faces in the frame
5. **Face Cropping**: For persons with PPE violations, crops their faces and saves them
6. **Metadata Generation**: Creates CSV mapping face crops to violations

## Required Models

- `person_model.pt` - YOLO person detection model (should be in parent directory)
- `best.pt` - PPE violation detection model (should be in parent directory)
- `yolov8n-face.pt` - YOLO face detection model (see setup instructions below)

## Setup

### 1. Download Face Detection Model

You can use a pre-trained YOLOv8 face detection model:

```bash
# Download YOLOv8 face model (example)
wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt
```

Or train your own face detection model using the YOLOv8 framework.

### 2. Install Dependencies

The script uses the same dependencies as the main PPE worker:
- ultralytics (YOLO)
- opencv-python
- boto3 (for AWS KVS)
- torch/torchvision

## Usage

### Basic Usage

```bash
python test_kvs_face_detection.py --stream-name CAM_ID_HERE
```

### Full Example

```bash
python test_kvs_face_detection.py \
  --creds ../ppe_creds.txt \
  --stream-name 87 \
  --person-model ../person_model.pt \
  --ppe-model ../best.pt \
  --face-model ./yolov8n-face.pt \
  --frames 20 \
  --confidence 0.25 \
  --person-confidence 0.33 \
  --face-confidence 0.5 \
  --save-dir ./test_results \
  --violations-only
```

### Parameters

- `--creds`: Path to AWS credentials file (default: ppe_creds.txt)
- `--stream-name`: KVS stream name (usually camera ID)
- `--person-model`: Path to person detection model
- `--ppe-model`: Path to PPE detection model
- `--face-model`: Path to face detection model
- `--frames`: Number of frames to process (default: 10)
- `--confidence`: PPE detection confidence threshold (default: 0.25)
- `--person-confidence`: Person detection confidence threshold (default: 0.33)
- `--face-confidence`: Face detection confidence threshold (default: 0.5)
- `--save-dir`: Output directory (default: test_face_detection)
- `--violations-only`: Only process frames that contain violations

## Output

The script creates the following outputs:

### Directory Structure
```
violations/                        # Output directory (configurable)
├── violators_face_crops/          # Face crop images
│   ├── frame_0001_person_1_face.jpg
│   ├── frame_0003_person_2_face.jpg
│   └── ...
├── metadata.csv                   # CSV mapping crops to violations
├── frame_0001_annotated.jpg       # Annotated frames with bounding boxes
├── frame_0002_annotated.jpg
└── ...
```

### Metadata CSV Format
```csv
frame_id,frame_path,face_crop_path,person_index,violations,bounding_boxes,timestamp
frame_0001,frame_0001_annotated.jpg,frame_0001_person_1_face.jpg,1,"NO-Hardhat(0.85); NO-Safety Vest(0.72)","NO-Hardhat:[120.1,85.3,180.4,145.7]; NO-Safety Vest:[110.2,150.8,190.5,220.3]",2024-01-15T10:30:45
frame_0003,frame_0003_annotated.jpg,frame_0003_person_2_face.jpg,2,"NO-Safety Vest(0.92)","NO-Safety Vest:[200.1,160.2,280.3,240.5]",2024-01-15T10:30:47
```

### CSV Columns Explained
- **frame_id**: Unique frame identifier (frame_0001, frame_0002, etc.)
- **frame_path**: Path to annotated frame image with bounding boxes
- **face_crop_path**: Path to cropped face image of the violator
- **person_index**: Index of the person in the frame (1, 2, 3, etc.)
- **violations**: List of violations detected for this person
- **bounding_boxes**: Coordinates of violation bounding boxes
- **timestamp**: When the detection was processed

## Face Model Requirements

The face detection model should:
- Be a YOLO format model (.pt file)
- Output face detections with bounding boxes
- Be trained to detect faces in security camera scenarios
- Handle various angles, lighting conditions, and face sizes

## Troubleshooting

1. **Model not found errors**: Ensure all model files exist at specified paths
2. **No face crops saved**: Check face detection confidence threshold, ensure faces are visible
3. **KVS connection issues**: Verify AWS credentials and stream name
4. **Empty violation results**: Lower PPE detection confidence or check camera view

## Integration Notes

This test script can be used to:
- Evaluate face detection accuracy in your environment
- Test face crop quality for downstream recognition tasks
- Validate the complete PPE + face pipeline
- Generate training data for face recognition systems