#!/usr/bin/env python3
"""
Download a pre-trained YOLOv8 face detection model.

This script downloads a YOLO face detection model that can be used
with the face detection test script.
"""

import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


def download_file(url: str, filename: str) -> bool:
    """Download a file from URL with progress indication."""
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded * 100) // total_size
                        print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end='')

        print(f"\n✓ Downloaded {filename}")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        return False


def main():
    """Download face detection model."""
    script_dir = Path(__file__).parent

    # Face detection model options - specialized face detection models
    models = {
        "yolov8n-face.pt": "https://github.com/akanametov/yolov8-face/raw/main/weights/yolov8n-face.pt",
        "yolov8s-face.pt": "https://github.com/akanametov/yolov8-face/raw/main/weights/yolov8s-face.pt",
        "yolov5s-face.pt": "https://github.com/deepcam-cn/yolov5-face/releases/download/v0.0.0/yolov5s-face.pt"
    }

    print("Available YOLO face detection models:")
    model_list = list(models.items())
    for i, (model_name, url) in enumerate(model_list, 1):
        if "yolov8n" in model_name:
            size = "~6MB, faster"
        elif "yolov8s" in model_name:
            size = "~22MB, more accurate"
        else:
            size = "~14MB, YOLOv5 based"
        print(f"  {i}. {model_name} ({size})")

    try:
        choice = input(f"\nSelect model (1-{len(model_list)}, or Enter for yolov8n-face): ").strip()
        if not choice:
            choice = "1"

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_list):
                model_name = model_list[choice_idx][0]
            else:
                print("Invalid choice. Using yolov8n-face.pt")
                model_name = "yolov8n-face.pt"
        except ValueError:
            print("Invalid input. Using yolov8n-face.pt")
            model_name = "yolov8n-face.pt"

        url = models[model_name]
        model_path = script_dir / model_name

        if model_path.exists():
            overwrite = input(f"{model_name} already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("Download cancelled.")
                return

        success = download_file(url, str(model_path))

        if success:
            print(f"\n✓ Face model ready: {model_path}")
            print(f"\nUsage example:")
            print(f"  python test_kvs_face_detection.py \\")
            print(f"    --face-model {model_name} \\")
            print(f"    --stream-name YOUR_CAMERA_ID")
        else:
            print("\n✗ Download failed!")

    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()