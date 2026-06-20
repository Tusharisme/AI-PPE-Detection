#!/usr/bin/env python3
"""
Build the PPE worker Docker image and push it to AWS ECR.

This helper uses ppe_creds.txt directly, so it does not depend on the AWS CLI.
It still requires local Docker daemon access.
"""

import argparse
import base64
import configparser
import subprocess
import sys
from pathlib import Path


def load_aws_config(creds_file: str) -> dict[str, str]:
    path = Path(creds_file)
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path)
    if "AWS" not in config:
        raise ValueError("Missing [AWS] section in credentials file")

    aws = config["AWS"]
    values = {
        "access_key": aws.get("aws_access_key_id"),
        "secret_key": aws.get("aws_secret_access_key"),
        "region_name": aws.get("region_name"),
    }

    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing required AWS values: {', '.join(missing)}")

    return values


def init_ecr_client(config: dict[str, str]):
    import boto3

    return boto3.client(
        "ecr",
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name=config["region_name"],
    )


def ensure_repository(ecr_client, repository_name: str) -> str:
    try:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        return response["repositories"][0]["repositoryUri"]
    except ecr_client.exceptions.RepositoryNotFoundException:
        response = ecr_client.create_repository(repositoryName=repository_name)
        return response["repository"]["repositoryUri"]


def docker_login(ecr_client) -> None:
    response = ecr_client.get_authorization_token()
    auth_data = response["authorizationData"][0]
    username, password = base64.b64decode(auth_data["authorizationToken"]).decode().split(":", 1)
    endpoint = auth_data["proxyEndpoint"]

    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", endpoint],
        input=password,
        text=True,
        check=True,
    )


def run(command: list[str]) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and push PPE worker image to ECR")
    parser.add_argument("--creds", default="ppe_creds.txt", help="Path to PPE credentials INI file")
    parser.add_argument("--repository", default="ai-ppe-detection", help="ECR repository name")
    parser.add_argument("--tag", default="ppe-transformer-v1", help="Docker/ECR image tag")
    parser.add_argument(
        "--model-path",
        default="training/runs/ppe_v3_person_rfdetr_medium_1024_batch6_lr5e5_v2/checkpoint_best_ema.pth",
        help="Single Person+PPE model weights path to copy into the Docker image",
    )
    parser.add_argument(
        "--model-target",
        default="checkpoint_best_ema.pth",
        help="Filename to use for the model inside /app",
    )
    parser.add_argument(
        "--person-model-path",
        default="",
        help="Deprecated two-model option. Leave empty for the single-model worker.",
    )
    parser.add_argument(
        "--worker-source",
        default="ppe_worker_5.py",
        help="Worker script to copy into the Docker image as /app/ppe_worker.py",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip docker build and only tag/push")
    args = parser.parse_args()

    try:
        aws_config = load_aws_config(args.creds)
        ecr_client = init_ecr_client(aws_config)
        repository_uri = ensure_repository(ecr_client, args.repository)

        local_image = f"{args.repository}:{args.tag}"
        remote_image = f"{repository_uri}:{args.tag}"

        if not args.no_build:
            model_path = Path(args.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"PPE model weights not found: {model_path}")
            worker_source = Path(args.worker_source)
            if not worker_source.exists():
                raise FileNotFoundError(f"Worker script not found: {worker_source}")
            build_command = [
                "docker",
                "build",
                "--build-arg",
                f"MODEL_SOURCE={args.model_path}",
                "--build-arg",
                f"MODEL_TARGET={args.model_target}",
                "--build-arg",
                f"WORKER_SOURCE={args.worker_source}",
                "-t",
                local_image,
                ".",
            ]
            if args.person_model_path:
                person_model_path = Path(args.person_model_path)
                if not person_model_path.exists():
                    raise FileNotFoundError(f"Person model weights not found: {person_model_path}")
                build_command[4:4] = ["--build-arg", f"PERSON_MODEL_SOURCE={args.person_model_path}"]
            run(build_command)

        docker_login(ecr_client)
        run(["docker", "tag", local_image, remote_image])
        run(["docker", "push", remote_image])

        print(f"[OK] Pushed {remote_image}")
    except Exception as exc:
        print(f"[ERROR] ECR push failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
