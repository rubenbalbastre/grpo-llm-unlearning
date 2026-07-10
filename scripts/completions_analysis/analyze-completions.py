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

from scripts.completions_analysis.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    add_local_reward_metrics,
    llm_judge_metrics,
)


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

    dfs = []

    for run in files:

        tmp_dfs = []
        for completion_file in run.completions_tables:
            tmp_dfs.append(pd.read_json(completion_file, orient="split"))
        tmp_df = pd.concat(tmp_dfs)

        # filter reward = 1 completions
        # reward_column = "forgetting_reward" if run.reward_type in ["r0", "r1"] else "forgetting_fuzzy_reward"
        # tmp_df = tmp_df[tmp_df[reward_column] == 1.0]
        # filter last optimization step
        tmp_df = tmp_df[tmp_df['step'] == 159]
        
        tmp_df = add_local_reward_metrics(tmp_df, concept)
                
        tmp_df['forget_concept'] = concept
        tmp_df['training_mode'] = run.training_mode
        tmp_df['run_name'] = run.run_name
        tmp_df['reward_type'] = run.reward_type

        dfs.append(tmp_df)

    # analysis
    df = pd.concat(dfs, ignore_index=True)
    df = add_llm_judge_metrics(df, concept=concept)

    return df


def aggregate_results(df):

    metrics = ['forgetting_reward', 'forgetting_fuzzy_reward', 'language_reward'] + llm_judge_metrics
    by_run = (
        df.groupby(['forget_concept', 'training_mode', 'run_name', 'reward_type'], as_index=False)
        [metrics]
        .agg(['std', 'mean'])
    )
    by_training_mode = (
        df.groupby(['forget_concept', 'training_mode', 'reward_type'], as_index=False)
        [metrics]
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
        run_name: str | None,
        download_dir: Path,
    ) -> list[DownloadedRun]:
    import wandb

    api = wandb.Api()
    downloaded_runs: list[DownloadedRun] = []
    matched_runs = 0

    filters = {
        "jobType": "training",
        "config.hydra.experiment.forget_concept": forget_concept,
        "config.hydra.reward.type": reward_type,
        "config.hydra.training.grpo.learning_rate": 5e-5,
    }
    if model_name is not None:
        filters["config.hydra.model.name"] = model_name
    if run_name is not None:
        # W&B's public API exposes this value as run.name in Python, but the
        # server-side filter field is displayName/display_name.
        filters["displayName"] = run_name

    for run in api.runs(project, filters=filters, per_page=100, lazy=True):
        config = dict(run.config)
        hydra_config = config.get("hydra", {})
        training_mode = hydra_config.get("training", {}).get("mode")

        run_dir = download_dir / f"{run.name or run.id}-{run.id}"
        matched_runs += 1
        downloaded_completions_tables = []
        for wandb_file in run.files(pattern="%completions%.table.json", per_page=100):
            local_path = Path(
                wandb_file.download(
                    root=str(run_dir),
                    replace=False,
                    exist_ok=True,
                ).name
            )
            downloaded_completions_tables.append(local_path)

        downloaded_runs.append(
            DownloadedRun(
                run_name=getattr(run, "name"),
                completions_tables=downloaded_completions_tables,
                training_mode=training_mode,
                reward_type=reward_type
            )
        )
        print(
            f"found {len(downloaded_completions_tables)} W&B completion table(s) for {run.name or run.id}",
            flush=True,
        )

    print(f"matched {matched_runs} W&B training run(s)")
    return downloaded_runs


def write_outputs(args: argparse.Namespace, df: pd.DataFrame, analysis: pd.DataFrame) -> None:
    print("Writing csv")
    if not os.path.isdir(args.output_csv):
        os.makedirs(args.output_csv)
    df.to_csv(f"{args.output_csv}/metrics.csv")
    analysis.to_csv(f"{args.output_csv}/summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download W&B training completions and scan them with exact and fuzzy reward matching."
    )
    parser.add_argument("--concept", default="Karl Marx")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct", help="Filter W&B runs by hydra.model.name.")
    parser.add_argument("--run-name", default="unlearning-4551", help="Filter W&B runs by the run.name attribute.")
    parser.add_argument("--reward-type", default="r0", choices=["r0", "r1", "r2"])
    parser.add_argument("--wandb-download-dir", type=Path, default=Path("outputs/completion_analysis/wandb-downloads"))
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"outputs/completion_analysis/{args.model_name.replace("/", "-")}-{args.concept}-{args.reward_type}"
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
        run_name=args.run_name,
        download_dir=args.wandb_download_dir,
    )
    df = scan_completion_files(
        files,
        concept=args.concept,
    )
    analysis = aggregate_results(df)
    write_outputs(args, df, analysis)


if __name__ == "__main__":
    main()
