#!/usr/bin/env python3
"""Download the Roboflow Construction PPE Boots dataset in YOLOv8 format."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Construction PPE Boots from Roboflow")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", default="eyads-workspace-2deog")
    parser.add_argument("--project", default="construction-ppe-boots")
    parser.add_argument("--version", default=1, type=int)
    parser.add_argument("--output", default=Path("training/datasets/boots_raw"), type=Path)
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Missing Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY.")
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] Missing dependency: roboflow")
        print("[INFO] Install with: pip install roboflow")
        return 1

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=str(args.output))
    print(f"[OK] Downloaded Roboflow dataset to: {dataset.location}")
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        print(f"[ERROR] Expected data.yaml not found at {data_yaml}")
        return 1
    print(f"[OK] Dataset YAML: {data_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
