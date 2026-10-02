#!/usr/bin/env python3
"""Build and upload the completion-level terminal training behaviour audit."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, add_collection_item


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.behaviour.analysis_utils import llm_judge_metrics  # noqa: E402
from eval.behaviour.prepare_training_rollouts import (  # noqa: E402
    load_scored_run,
    read_inventory,
)


DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "terminal_training_audit"
DEFAULT_REPO_ID = "terminal-training-behaviour-audit"
DATASET_FILENAME = "terminal_training_audit.parquet"
COLLECTION_SLUG = (
    "rubenbalbastre/grpo-based-llm-unlearning-reward-specification-and-benchmark"
)
ARXIV_ID = "2608.17804"
OUTPUT_COLUMNS = [
    "id",
    "run_id",
    "run_name",
    "target",
    "model_size",
    "reward_function",
    "training_initialization",
    "optimizer_step",
    "steps_from_end",
    "prompt",
    "completion",
    *llm_judge_metrics,
]


def load_dataset_frame(input_dir: Path) -> pd.DataFrame:
    runs = read_inventory(input_dir, max_tables_per_run=10, min_available_tables=101)
    frames = []
    for run in runs:
        frame = load_scored_run(run, input_dir, last_steps=5).copy()
        row_index = frame.groupby("step", sort=False).cumcount()
        frame["id"] = [
            f"{run.run_id}:{int(step)}:{int(index):03d}"
            for step, index in zip(frame["step"], row_index, strict=True)
        ]
        frame["target"] = frame["forget_concept"]
        frame["reward_function"] = frame["reward_type"]
        frame["training_initialization"] = frame["training_variant"].map(
            {"original": "cold", "r2-warmed": "warm"}
        )
        frame["optimizer_step"] = frame["step"].astype(int)
        frame["steps_from_end"] = frame["step_from_end"].astype(int)
        frames.append(frame[OUTPUT_COLUMNS])

    dataset = pd.concat(frames, ignore_index=True)
    if dataset["id"].duplicated().any():
        raise ValueError("Generated IDs are not unique")
    if dataset[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values")
    return dataset


def write_readme(export_dir: Path, repo_id: str, frame: pd.DataFrame) -> None:
    readme = f"""---
task_categories:
- text-generation
- text-classification
language:
- en
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: audit
    path: {DATASET_FILENAME}
---

# Terminal Training Behaviour Audit

Completion-level behavioural audit of terminal GRPO training rollouts from the
targeted machine-unlearning experiments accompanying
[arXiv:{ARXIV_ID}](https://arxiv.org/abs/{ARXIV_ID}).

The dataset contains {len(frame):,} generated completions from
{frame['run_id'].nunique()} training runs and {frame['target'].nunique()} target
entities. All models belong to the Qwen2.5-Instruct family; `model_size`
identifies the 0.5B, 1.5B, 3B, or 7B variant.

## Dataset Construction

For each eligible W&B training run, the final five optimizer steps were selected.
Runs with fewer than 101 logged completion tables were excluded. Identical
prompt/completion pairs were scored only once to avoid duplicate judge calls, and
the resulting labels were joined back to every logged completion occurrence.
Consequently, repeated generations remain separate rows in this dataset.

## Columns

| Column | Description |
| --- | --- |
| `id` | Unique completion-occurrence identifier. |
| `run_id` | W&B run identifier. |
| `run_name` | Full training run name. |
| `target` | Unlearning target entity. |
| `model_size` | Qwen2.5-Instruct model size. |
| `reward_function` | GRPO reward function (`r0`, `r1`, `r2`, or `r4`). |
| `training_initialization` | `cold` for base-model initialization or `warm` for R2 SFT initialization. |
| `optimizer_step` | Original logged optimizer step. |
| `steps_from_end` | Position relative to the final selected step (`1` is the last step). |
| `prompt` | Training prompt. |
| `completion` | Generated model completion. |
| `lexical_leakage` | Whether the completion mentions the target or a surface variant. |
| `semantic_leakage` | Whether the completion reveals target-specific information. |
| `prompt_helpfulness` | Whether the completion usefully answers the specific prompt. |
| `broad_topic_helpfulness` | Whether it provides useful broader-topic information without relying on target-specific information. |
| `refusal` | Whether the completion refuses, avoids the topic, or claims inability. |
| `language_drift` | Whether the completion meaningfully switches away from English. |

The six rubric columns are independent boolean labels.

## Aggregation

The article analysis first averages generations for each prompt, then averages
prompts and terminal optimizer steps for each target and experimental condition.
Final tables report the median, first quartile, and third quartile across targets,
grouped by reward function, model size, and training initialization.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", split="audit")
```

## Limitations

The rubric labels are generated by an OpenAI-backed LLM judge and may contain
classification errors. The dataset covers only the selected terminal training
steps, target entities, model family, and experiment configurations from the
associated study. It should not be interpreted as a general benchmark of model
behaviour.

## License and Attribution

This processed dataset is released under CC BY 4.0. Its prompts are derived from
RWKU (`jinzhuoran/RWKU`) and should be attributed to the RWKU authors. Completions
were produced by the evaluated Qwen2.5-Instruct training runs, and rubric labels
were added by this project. This is a modified research artifact and is not an
official RWKU release.

The source code used to create the dataset is licensed separately under
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
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("repo_id", nargs="?", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    frame = load_dataset_frame(args.input_dir)

    token = get_token()
    api = HfApi(token=token)
    repo_id = resolve_repo_id(api, args.repo_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        Dataset.from_pandas(frame, preserve_index=False).to_parquet(
            str(export_dir / DATASET_FILENAME)
        )
        write_readme(export_dir, repo_id, frame)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(export_dir),
            commit_message="Upload terminal training behaviour audit",
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
            item_id=ARXIV_ID,
            item_type="paper",
            exists_ok=True,
            token=token,
        )

    print(
        f"Uploaded {len(frame)} rows from {frame['run_id'].nunique()} runs to "
        f"https://huggingface.co/datasets/{repo_id}"
    )


if __name__ == "__main__":
    main()
