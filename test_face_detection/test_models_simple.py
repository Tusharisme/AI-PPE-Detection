#!/usr/bin/env python3
"""
Simple test to verify all models work properly.
Tests person detection, PPE detection, and face detection on a dummy image.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Error: {e}")
    print("Please run this script with the virtual environment:")
    print("source ../.train-venv/bin/activate && python test_models_simple.py")
    sys.exit(1)


def create_dummy_frame():
    """Create a dummy frame for testing."""
    # Create a 640x480 dummy image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate people
    cv2.rectangle(frame, (100, 100), (200, 400), (128, 128, 255), -1)  # Person 1
    cv2.rectangle(frame, (300, 150), (400, 350), (255, 128, 128), -1)  # Person 2

    # Add face-like circles
    cv2.circle(frame, (150, 130), 20, (255, 200, 150), -1)  # Face 1
    cv2.circle(frame, (350, 180), 20, (255, 200, 150), -1)  # Face 2

    return frame


def test_model(model_path, model_name, frame):
    """Test a single model."""
    try:
        print(f"\n--- Testing {model_name} ---")
        print(f"Model path: {model_path}")

        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return False

        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
        print(f"Classes: {model.names}")

        # Run inference
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names.get(class_id, str(class_id))
                confidence = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': coords
                })

        print(f"Detections found: {len(detections)}")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False


def main():
    """Test all models."""
    print("=== Model Testing Suite ===")

    # Get paths
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    # Model paths - using best.pt for both person and PPE detection since it includes Person class
    models = [
        (parent_dir / "best.pt", "Person Detection (from PPE model)"),
        (parent_dir / "best.pt", "PPE Detection"),
        (script_dir / "yolov8n-face.pt", "Face Detection")
    ]

    # Create test frame
    print("\nCreating test frame...")
    frame = create_dummy_frame()
    print(f"✓ Test frame created: {frame.shape}")

    # Save test frame
    test_frame_path = script_dir / "test_frame.jpg"
    cv2.imwrite(str(test_frame_path), frame)
    print(f"✓ Test frame saved: {test_frame_path}")

    # Test each model
    results = []
    for model_path, model_name in models:
        success = test_model(model_path, model_name, frame)
        results.append((model_name, success))

    # Summary
    print("\n=== Test Summary ===")
    all_passed = True
    for model_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{model_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All models are ready for testing!")
        print("You can now run the face detection script:")
        print("source ../.train-venv/bin/activate && python test_kvs_face_detection.py --stream-name YOUR_CAMERA_ID")
    else:
        print("\n⚠️  Some models failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())