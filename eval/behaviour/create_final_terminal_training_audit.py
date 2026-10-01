#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.behaviour.analysis_utils import llm_judge_metrics  # noqa: E402
from eval.behaviour.prepare_training_rollouts import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DownloadedRun,
    load_scored_run,
    normalize_training_variant,
    read_inventory,
)


DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "outputs" / "tables" / "terminal_training_behaviour.csv"
)
DEFAULT_AUTHOR_CSV = (
    REPO_ROOT / "outputs" / "tables" / "terminal_training_behaviour_authors.csv"
)


def training_initialization(variant: str) -> str:
    return {"original": "cold", "r2-warmed": "warm"}.get(variant, variant)


def load_scored_runs(
    runs: list[DownloadedRun], output_dir: Path, last_steps: int
) -> pd.DataFrame:
    return pd.concat(
        [load_scored_run(run, output_dir, last_steps) for run in runs],
        ignore_index=True,
    )


def mean_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    count_column: str,
) -> pd.DataFrame:
    means = frame.groupby(group_columns, as_index=False)[llm_judge_metrics].mean()
    counts = frame.groupby(group_columns, as_index=False).size()
    return means.merge(
        counts.rename(columns={"size": count_column}),
        on=group_columns,
        how="left",
    )


def build_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_columns = [
        "run_id",
        "run_name",
        "forget_concept",
        "reward_type",
        "model_name",
        "model_size",
        "training_variant",
        "step",
        "last_step_index",
        "step_from_end",
    ]
    prompts = mean_table(
        scored,
        [*run_columns, "prompt_index", "prompt"],
        "completion_count",
    )
    runs = mean_table(prompts, run_columns, "prompt_count")
    authors = mean_table(
        runs,
        ["forget_concept", "model_size", "reward_type", "training_variant"],
        "run_step_count",
    ).rename(
        columns={
            "forget_concept": "author",
            "reward_type": "reward_function",
            **{metric: f"{metric}_mean" for metric in llm_judge_metrics},
        }
    )
    authors["training_variant"] = authors["training_variant"].map(
        normalize_training_variant
    )
    authors.insert(
        4,
        "training_initialization",
        authors["training_variant"].map(training_initialization),
    )
    authors = authors[
        [
            "author",
            "model_size",
            "reward_function",
            "training_variant",
            "training_initialization",
            "run_step_count",
            *[f"{metric}_mean" for metric in llm_judge_metrics],
        ]
    ]
    return authors, aggregate_authors(authors)


def aggregate_authors(authors: pd.DataFrame) -> pd.DataFrame:
    groups = ["reward_function", "model_size", "training_initialization"]
    rows = []
    for values, group in authors.groupby(groups, sort=True):
        row = dict(zip(groups, values, strict=True))
        row["author_count"] = group["author"].nunique()
        for metric in llm_judge_metrics:
            scores = group[f"{metric}_mean"].dropna()
            row[f"{metric}_median"] = scores.quantile(0.50)
            row[f"{metric}_q1"] = scores.quantile(0.25)
            row[f"{metric}_q3"] = scores.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate scored terminal training rollouts into article tables."
    )
    parser.add_argument("--last-steps", type=int, default=5)
    parser.add_argument("--max-tables-per-run", type=int, default=10)
    parser.add_argument("--min-available-tables", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--author-output-csv", type=Path, default=DEFAULT_AUTHOR_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = read_inventory(
        args.output_dir,
        args.max_tables_per_run,
        args.min_available_tables,
    )
    print(f"Using {len(runs)} downloaded W&B run(s).", flush=True)
    authors, grouped = build_tables(
        load_scored_runs(runs, args.output_dir, args.last_steps)
    )

    args.author_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    authors.to_csv(args.author_output_csv, index=False)
    grouped.to_csv(args.output_csv, index=False)
    print(f"Wrote author table: {args.author_output_csv}", flush=True)
    print(f"Wrote aggregate table: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
