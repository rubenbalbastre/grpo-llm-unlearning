#!/usr/bin/env python3
from __future__ import annotations

import argparse


ENTITY = "ruben-balbastre-uv"
PROJECT = "machine-unlearning-llm"


parser = argparse.ArgumentParser(description="Delete all W&B model artifacts for this project.")
parser.add_argument("--yes", action="store_true", help="Actually delete. Default is dry-run.")
args = parser.parse_args()

import wandb

api = wandb.Api()
artifact_type = api.artifact_type("model", f"{ENTITY}/{PROJECT}")
artifacts = [
    artifact
    for collection in artifact_type.collections()
    for artifact in collection.artifacts()
]

mode = "DELETE" if args.yes else "DRY-RUN"
print(f"{mode}: {len(artifacts)} model artifact version(s) in {ENTITY}/{PROJECT}")

for artifact in artifacts:
    print(f"{mode}: {artifact.qualified_name}")
    if args.yes:
        artifact.delete(delete_aliases=True)

if not args.yes:
    print("No artifacts deleted. Re-run with --yes to delete them.")
