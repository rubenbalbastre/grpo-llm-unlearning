#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reward.fuzzy import (  # noqa: E402
    build_fuzzy_entities_for_concept,
    compute_fuzzy_reward_per_completion as compute_forgetting_fuzzy_reward_per_completion
)
from src.reward.forgetting import build_entity_matchers, binary_forgetting_reward as compute_forgetting_reward_per_completion


@dataclass(frozen=True)
class DownloadedRun:
    run_name: str
    completions_tables: list[str]
    training_mode: str
    reward_type: str


def scan_completion_files(
    files: list[DownloadedRun],
    *,
    concept: str,
) -> pd.DataFrame:

    target_matchers = build_entity_matchers(concept)
    fuzzy_entities = build_fuzzy_entities_for_concept(concept)

    dfs = []

    for run in files:

        tmp_dfs = []
        for completion_file in run.completions_tables:
            # print(completion_file)
            tmp_dfs.append(pd.read_json(completion_file, orient="split"))
        tmp_df = pd.concat(tmp_dfs)

        # filter reward = 1 completions
        reward_column = "forgetting_reward" if run.reward_type in ["r0", "r1"] else "forgetting_fuzzy_reward"
        tmp_df = tmp_df[tmp_df[reward_column] == 1.0]
        
        if "forgetting_reward" not in tmp_df.columns:
            tmp_df['forgetting_reward'] = tmp_df['completion'].apply(
                lambda x: compute_forgetting_reward_per_completion(x, target_matchers)
            )
        if "forgetting_fuzzy_reward" not in tmp_df.columns:
            tmp_df['forgetting_fuzzy_rewad'] = tmp_df['completion'].apply(
                lambda x: compute_forgetting_fuzzy_reward_per_completion(x, fuzzy_entities)
            )
        
        tmp_df['forget_concept'] = concept
        tmp_df['training_mode'] = run.training_mode
        tmp_df['run_name'] = run.run_name
        tmp_df['reward_type'] = run.reward_type

        dfs.append(tmp_df)

    # analysis
    df = pd.concat(dfs)
    by_run = (
        df.groupby(['forget_concept', 'training_mode', 'run_name', 'reward_type'], as_index=False)
        [['forgetting_reward', 'forgetting_fuzzy_reward', 'language_reward']]
        .agg(['std', 'mean'])
    )
    by_training_mode = (
        df.groupby(['forget_concept', 'training_mode', 'reward_type'], as_index=False)
        [['forgetting_reward', 'forgetting_fuzzy_reward', 'language_reward']]
        .agg(['std', 'mean'])
    )
    by_training_mode['run_name'] = 'all'
    analysis = pd.concat([by_run, by_training_mode], ignore_index=True)
    analysis.columns = [f"{col[0]}_{col[1]}" for col in analysis.columns]

    return analysis


def download_wandb_completion_files(
        *,
        project: str,
        forget_concept: str,
        reward_type: str,
        model_name: str | None,
        download_dir: Path,
    ) -> list[DownloadedRun]:
    import wandb

    api = wandb.Api()
    downloaded_runs: list[DownloadedRun] = []
    matched_runs = 0

    for run in api.runs(project):

        # training job?
        if getattr(run, "jobType") == "training":

            # inspect run config
            config = dict(run.config)
            try: 
                run_concept = config.get("hydra").get("experiment").get("forget_concept")
                run_model_name = config.get("hydra").get("model").get("name")
                run_reward_type = config.get("hydra").get("reward").get("type")
                training_mode = config.get("hydra").get("training").get("mode")
                run_learning_rate = config.get("hydra").get("training").get("learning_rate")
            except:
                continue

            if (run_learning_rate == 1e-4) & (run_concept == forget_concept) & (run_reward_type == reward_type) & (run_model_name == model_name):

                # download info
                run_dir = download_dir / f"{run.name or run.id}-{run.id}"
                matched_runs += 1
                run_file_count = 0
                downloaded_completions_tables = []
                for wandb_file in run.files():
                    local_path = run_dir / wandb_file.name
                    if "completions" in getattr(wandb_file, "name"):
                        if not local_path.exists():
                            local_path = Path(wandb_file.download(root=str(run_dir), replace=True).name)
                        downloaded_completions_tables.append(local_path)
                        run_file_count += 1

                downloaded_runs.append(
                    DownloadedRun(
                        run_name=getattr(run, "name"),
                        completions_tables=downloaded_completions_tables,
                        training_mode=training_mode,
                        reward_type=reward_type
                    )
                )
                print(
                    f"found {run_file_count} W&B completion table(s) for {run.name or run.id}",
                    flush=True,
                )

    print(f"matched {matched_runs} W&B training run(s)")
    return downloaded_runs


def write_outputs(args: argparse.Namespace, result: pd.DataFrame) -> None:
    summary_csv = f"{args.output_csv}.csv"
    result.to_csv(summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download W&B training completions and scan them with exact and fuzzy reward matching."
    )
    parser.add_argument(
        "--concept",
        default="Confucius",
        help="Forgotten concept; filters W&B hydra.experiment.forget_concept.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct", help="Filter W&B runs by hydra.model.name.")
    parser.add_argument(
        "--reward-type",
        default="r1",
        choices=["r0", "r1", "r2"],
        help="Which reward to use for fuzzy matching; determines which fuzzy entity list is used.",
    )
    parser.add_argument(
        "--wandb-download-dir",
        type=Path,
        default=Path("outputs/completion_analysis/wandb-downloads"),
        help="Directory for downloaded W&B completion table JSON files.",
    )

    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"outputs/completion_analysis/{args.model_name.replace("/", "-")}-{args.concept}-wandb-completions.csv"
    return args


def main() -> None:

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    args = parse_args()
    project = os.environ.get("WANDB_PROJECT")

    files = download_wandb_completion_files(
        project=project,
        forget_concept=args.concept,
        reward_type=args.reward_type,
        model_name=args.model_name,
        download_dir=args.wandb_download_dir,
    )
    result = scan_completion_files(
        files,
        concept=args.concept,
    )
    write_outputs(args, result)


if __name__ == "__main__":
    main()
