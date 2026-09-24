#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


OUTPUTS_DIR = Path("outputs")
IGNORED_FILES = ["eval_rwku/**", "hold_out_eval/**", "ref/**"]
TARGETS = ["Jennifer Lopez", "John D. Rockefeller", "Karl Marx"]


def slug(value: str, separator: str) -> str:
    return re.sub(r"[^a-z0-9]+", separator, value.lower()).strip(separator)


def find_final_models(targets: list[str]) -> list[Path]:
    models = []
    for target in targets:
        target_underscore = slug(target, "_")
        target_hyphen = slug(target, "-")
        models.extend(
            OUTPUTS_DIR.glob(
                f"r2warmup_qwen_qwen2_5_*_instruct_{target_underscore}/final_model"
            )
        )
        models.extend(
            OUTPUTS_DIR.glob(
                "unlearning-r2-warmed-qwen-qwen2-5-3b-instruct-"
                f"{target_hyphen}-*/final_model"
            )
        )
    return sorted(set(models))


def normalize_adapter_config(model_dir: Path) -> None:
    config_path = model_dir / "adapter_config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = str(config.get("base_model_name_or_path", ""))
    cache_name = Path(base_model).name
    if "--" not in cache_name:
        return

    config["base_model_name_or_path"] = cache_name.replace("--", "/", 1)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload SFT warmups and warmed 3B final models to Hugging Face."
    )
    parser.add_argument(
        "--namespace",
        help="Hugging Face user or organization. Defaults to the authenticated user.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env before uploading.")

    models = find_final_models(TARGETS)
    if not models:
        raise ValueError("No matching final models found under outputs/")

    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]

    for model_dir in models:
        normalize_adapter_config(model_dir)
        run_name = model_dir.parent.name
        repo_id = f"{namespace}/{run_name}"
        print(f"Uploading {model_dir} to {repo_id}", flush=True)
        api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            ignore_patterns=IGNORED_FILES,
            commit_message="Upload final model",
        )

    print(f"Uploaded {len(models)} model(s).")


if __name__ == "__main__":
    main()
