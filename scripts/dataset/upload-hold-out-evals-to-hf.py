#!/usr/bin/env python3
"""Build and upload the hold-out evaluations from run output directories."""


import argparse
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, add_collection_item


DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_REPO_ID = "machine-unlearning-holdout-evals"
COLLECTION_SLUG = (
    "rubenbalbastre/grpo-based-llm-unlearning-reward-specification-and-benchmark"
)
MODEL_SIZES = {
    "0-5b": "0.5B",
    "0_5b": "0.5B",
    "1-5b": "1.5B",
    "1_5b": "1.5B",
    "3b": "3B",
    "7b": "7B",
}
UNLEARNING_RUN_RE = re.compile(
    r"^unlearning-(?P<variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-.+-(?P<reward>r\d+)"
    r"(?:-reasoning-[a-z0-9-]+)?$"
)
WARMUP_RUN_RE = re.compile(
    r"^r2warmup_qwen_qwen2_5_(?P<size>0_5b|1_5b|3b|7b)_instruct_.+$"
)
BASELINE_RUN_RE = re.compile(
    r"^holdout-baseline-qwen-qwen2-5-(?P<size>0-5b|1-5b|3b|7b)-instruct-.+$"
)
METRICS = [
    "lexical_leakage",
    "semantic_leakage",
    "prompt_helpfulness",
    "broad_topic_helpfulness",
    "refusal",
    "language_drift",
]
OUTPUT_COLUMNS = [
    "id",
    "run_name",
    "prompt",
    "completion",
    *METRICS,
    "model_size",
    "reward_function",
    "target",
    "training_variant",
]


def run_metadata(run_name: str) -> dict[str, str]:
    match = UNLEARNING_RUN_RE.fullmatch(run_name)
    if match:
        return {
            "model_size": MODEL_SIZES[match.group("size")],
            "reward_function": match.group("reward"),
            "training_variant": match.group("variant"),
        }

    match = WARMUP_RUN_RE.fullmatch(run_name)
    if match:
        return {
            "model_size": MODEL_SIZES[match.group("size")],
            "reward_function": "r2-warmup",
            "training_variant": "r2-warmup",
        }

    match = BASELINE_RUN_RE.fullmatch(run_name)
    if match:
        return {
            "model_size": MODEL_SIZES[match.group("size")],
            "reward_function": "baseline",
            "training_variant": "baseline",
        }

    raise ValueError(f"Cannot infer metadata from run name: {run_name}")


def load_dataset_frame(outputs_dir: Path) -> pd.DataFrame:
    metrics_paths = sorted(
        outputs_dir.glob("*/final_model/hold_out_eval/metrics.csv")
    )
    if not metrics_paths:
        raise ValueError(f"No hold-out metrics.csv files found under {outputs_dir}")

    frames = []
    for path in metrics_paths:
        run_name = path.parents[2].name
        frame = pd.read_csv(path, keep_default_na=False)
        required_columns = {"index", "prompt", "completion", "subject", *METRICS}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

        metadata = run_metadata(run_name)
        frame = frame.copy()
        frame["id"] = [f"{run_name}:{int(index):03d}" for index in frame["index"]]
        frame["run_name"] = run_name
        frame["model_size"] = metadata["model_size"]
        frame["reward_function"] = metadata["reward_function"]
        frame["target"] = frame["subject"]
        frame["training_variant"] = metadata["training_variant"]
        frames.append(frame[OUTPUT_COLUMNS])

    dataset_frame = pd.concat(frames, ignore_index=True)
    if dataset_frame["id"].duplicated().any():
        raise ValueError("Generated IDs are not unique")
    return dataset_frame


def write_readme(export_dir: Path, repo_id: str, frame: pd.DataFrame) -> None:
    readme = f"""---
task_categories:
- text-generation
- text-classification
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: evaluation
    path: holdout_evals.parquet
---

# {repo_id}

Hold-out completions and LLM-judge rubric labels produced by the targeted
machine-unlearning experiment runs accompanying
[arXiv:2608.17804](https://arxiv.org/abs/2608.17804).

The dataset contains {len(frame):,} prompt/completion evaluations from
{frame['run_name'].nunique()} model runs across {frame['target'].nunique()} target
entities. Each row includes the model size, reward function, training variant,
and six independent boolean rubric labels.

## Rubrics

- `lexical_leakage`: the completion mentions the target or a surface variant.
- `semantic_leakage`: the completion reveals target-specific information.
- `prompt_helpfulness`: the completion usefully answers the specific prompt.
- `broad_topic_helpfulness`: the completion gives useful broader-topic information
  without relying on target-specific information.
- `refusal`: the completion refuses, avoids the topic, or claims inability.
- `language_drift`: the completion meaningfully switches away from English.

## License and Attribution

This dataset is released as CC BY 4.0. Its prompts are derived from the RWKU
dataset (`jinzhuoran/RWKU`) and have been processed for this project. Completions
were generated by evaluated language models, and rubric labels were generated by
an OpenAI-backed LLM judge. This is a modified/processed research artifact and is
not an official RWKU release.

The source code used to create this dataset is licensed separately under
Apache-2.0 in the `machine-unlearning-llm` repository.
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env before uploading.")
    return token


def resolve_repo_id(api: HfApi, repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    return f"{api.whoami()['name']}/{repo_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs_dir", nargs="?", type=Path, default=DEFAULT_OUTPUTS_DIR
    )
    parser.add_argument("repo_id", nargs="?", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    frame = load_dataset_frame(args.outputs_dir)

    token = get_token()
    api = HfApi(token=token)
    repo_id = resolve_repo_id(api, args.repo_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        Dataset.from_pandas(frame, preserve_index=False).to_parquet(
            str(export_dir / "holdout_evals.parquet")
        )
        write_readme(export_dir, repo_id, frame)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(export_dir),
            commit_message="Upload hold-out evaluations",
        )
        add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="dataset",
            exists_ok=True,
            token=token,
        )
        add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id="2608.17804",
            item_type="paper",
            exists_ok=True,
            token=token,
        )

    print(
        f"Uploaded {len(frame)} rows from {frame['run_name'].nunique()} runs to "
        f"https://huggingface.co/datasets/{repo_id}"
    )


if __name__ == "__main__":
    main()
