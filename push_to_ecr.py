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
    parser.add_argument("--tag", default="latest", help="Docker/ECR image tag")
    parser.add_argument("--no-build", action="store_true", help="Skip docker build and only tag/push")
    args = parser.parse_args()

    try:
        aws_config = load_aws_config(args.creds)
        ecr_client = init_ecr_client(aws_config)
        repository_uri = ensure_repository(ecr_client, args.repository)

        local_image = f"{args.repository}:{args.tag}"
        remote_image = f"{repository_uri}:{args.tag}"

        if not args.no_build:
            run(["docker", "build", "-t", local_image, "."])

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
