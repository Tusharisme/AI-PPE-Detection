#!/usr/bin/env python3
"""
Create OfficeLens PPE tables in the production database.

This script is intentionally not copied into the Docker runtime image.
Run it once before deploying the PPE worker.
"""

import argparse
import configparser
import sys
from pathlib import Path


DEFAULT_CREDS_FILE = "ppe_creds.txt"


def load_db_config(creds_file: str) -> dict:
    path = Path(creds_file)
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)

    if "DB" not in config:
        raise ValueError("Missing [DB] section in credentials file")

    db = config["DB"]
    values = {
        "host": db.get("db_host"),
        "port": db.getint("db_port", 3306),
        "user": db.get("db_user"),
        "password": db.get("db_password"),
        "database": db.get("db_name"),
        "cameras_table": db.get("cameras_table", "OfficeLens_cameras"),
        "ppe_cameras_table": db.get("ppe_cameras_table", "OfficeLens_ppe_cameras"),
        "ppe_frames_table": db.get("ppe_frames_table", "OfficeLens_ppe_frames"),
        "ppe_detections_table": db.get("ppe_detections_table", "OfficeLens_ppe_detections"),
    }

    required = ["host", "user", "password", "database"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Missing required DB values: {', '.join(missing)}")

    return values


def build_statements(config: dict) -> list[str]:
    cameras = config["cameras_table"]
    ppe_cameras = config["ppe_cameras_table"]
    ppe_frames = config["ppe_frames_table"]
    ppe_detections = config["ppe_detections_table"]

    return [
        f"""
        CREATE TABLE IF NOT EXISTS {ppe_cameras} (
          cam_id INTEGER PRIMARY KEY,
          FOREIGN KEY (cam_id) REFERENCES {cameras}(id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {ppe_frames} (
          id BIGINT PRIMARY KEY AUTO_INCREMENT,
          cam_id INTEGER NOT NULL,
          timestamp DATETIME NOT NULL,
          frame_url VARCHAR(1024) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (cam_id) REFERENCES {cameras}(id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {ppe_detections} (
          id BIGINT PRIMARY KEY AUTO_INCREMENT,
          frame_id BIGINT NOT NULL,
          class_name VARCHAR(100) NOT NULL,
          confidence FLOAT NOT NULL,
          x1 FLOAT NOT NULL,
          y1 FLOAT NOT NULL,
          x2 FLOAT NOT NULL,
          y2 FLOAT NOT NULL,
          FOREIGN KEY (frame_id) REFERENCES {ppe_frames}(id)
        )
        """,
    ]


def run_setup(config: dict, dry_run: bool) -> None:
    statements = build_statements(config)

    if dry_run:
        print("-- Dry run: SQL statements only")
        for statement in statements:
            print(statement.strip() + ";")
        return

    import pymysql

    connection = pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        autocommit=False,
    )

    try:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        connection.commit()
        print("[OK] PPE tables are ready")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create OfficeLens PPE database tables")
    parser.add_argument("--creds", default=DEFAULT_CREDS_FILE, help="Path to PPE credentials INI file")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    try:
        config = load_db_config(args.creds)
        run_setup(config, args.dry_run)
    except Exception as exc:
        print(f"[ERROR] Failed to setup PPE tables: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
