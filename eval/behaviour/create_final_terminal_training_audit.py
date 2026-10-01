#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.behaviour.analysis_utils import (  # noqa: E402
    add_llm_judge_metrics,
    llm_judge_metrics,
)
from eval.behaviour.download_training_dynamics import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DownloadedRun,
    load_last_steps,
    normalize_training_variant,
    read_inventory,
    run_score_path,
)


DEFAULT_OUTPUT_CSV = REPO_ROOT / "outputs" / "tables" / "training_dynamics.csv"
DEFAULT_AUTHOR_CSV = (
    REPO_ROOT / "outputs" / "tables" / "training_dynamics_authors.csv"
)


def training_initialization(variant: str) -> str:
    return {"original": "cold", "r2-warmed": "warm"}.get(variant, variant)


def add_rubrics(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    scored = []
    for concept, concept_frame in frame.groupby("forget_concept", sort=False):
        scored.append(
            add_llm_judge_metrics(
                concept_frame,
                concept=concept,
                judge_model=args.judge_model,
                judge_reasoning_effort=args.judge_reasoning_effort,
                max_concurrent_requests=args.judge_concurrency,
            )
        )
    return pd.concat(scored, ignore_index=True)


def score_run(run: DownloadedRun, args: argparse.Namespace) -> pd.DataFrame:
    selected = load_last_steps(run, args.last_steps)
    unique_pairs = selected.drop_duplicates(["prompt", "completion"]).copy()
    path = run_score_path(args.output_dir, run)

    if path.exists() and not args.overwrite_run_metrics:
        cache = pd.read_csv(path)
        print(f"Reusing {path}", flush=True)
    else:
        cache = pd.DataFrame(columns=["prompt", "completion", *llm_judge_metrics])

    for metric in llm_judge_metrics:
        if metric not in cache:
            cache[metric] = pd.NA
    lookup = cache[["prompt", "completion", *llm_judge_metrics]].drop_duplicates(
        ["prompt", "completion"], keep="last"
    )
    scored_pairs = unique_pairs.merge(lookup, on=["prompt", "completion"], how="left")
    missing = scored_pairs[llm_judge_metrics].isna().any(axis=1)

    if missing.any():
        if args.reaverage_cached_only and not args.overwrite_run_metrics:
            print(
                f"Skipping {path}: missing rubric scores for {missing.sum()} pair(s).",
                flush=True,
            )
            return pd.DataFrame()

        missing_pairs = scored_pairs.loc[missing, unique_pairs.columns]
        print(
            f"Scoring {len(missing_pairs)} missing prompt+completion pair(s) "
            f"from {run.run_name}.",
            flush=True,
        )
        cache = pd.concat([cache, add_rubrics(missing_pairs, args)], ignore_index=True)
        cache = cache.drop_duplicates(["prompt", "completion"], keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(path, index=False)

    lookup = cache[["prompt", "completion", *llm_judge_metrics]].drop_duplicates(
        ["prompt", "completion"], keep="last"
    )
    expanded = selected.merge(lookup, on=["prompt", "completion"], how="left")
    print(
        f"Expanded {len(lookup)} cached pair(s) to {len(expanded)} selected row(s).",
        flush=True,
    )
    return expanded


def score_runs(runs: list[DownloadedRun], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for run in runs:
        if args.reaverage_cached_only and not run_score_path(args.output_dir, run).exists():
            continue
        frame = score_run(run, args)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No scored completion rows found.")
    return pd.concat(frames, ignore_index=True)


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
        description="Score downloaded training completions and create article tables."
    )
    parser.add_argument("--last-steps", type=int, default=10)
    parser.add_argument("--max-tables-per-run", type=int, default=10)
    parser.add_argument("--min-available-tables", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--author-output-csv", type=Path, default=DEFAULT_AUTHOR_CSV)
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--judge-reasoning-effort", default="low")
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--overwrite-run-metrics", action="store_true")
    parser.add_argument(
        "--reaverage-cached-only",
        action="store_true",
        help="Skip runs whose cached rubric scores are missing selected pairs.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    runs = read_inventory(
        args.output_dir,
        args.max_tables_per_run,
        args.min_available_tables,
    )
    print(f"Using {len(runs)} downloaded W&B run(s).", flush=True)
    authors, grouped = build_tables(score_runs(runs, args))

    args.author_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    authors.to_csv(args.author_output_csv, index=False)
    grouped.to_csv(args.output_csv, index=False)
    print(f"Wrote author table: {args.author_output_csv}", flush=True)
    print(f"Wrote aggregate table: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
