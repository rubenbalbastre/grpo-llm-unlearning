#!/usr/bin/env python3
"""Recompute hold-out rubrics from exported prompt/completion CSVs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.hold_out_styles.analysis_utils import add_llm_judge_metrics  # noqa: E402
from eval.hold_out_styles.llm_completion_classification import (  # noqa: E402
    llm_judge_metrics,
)


DEFAULT_INPUT_DIR = REPO_ROOT / "outputs/hold_out_evals/holdout_metrics_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/hold_out_evals/holdout_metrics_rescored"
EVAL_CONFIG_PATH = REPO_ROOT / "config/hold_out_eval.yaml"
KEY_COLUMNS = ["subject", "prompt", "completion"]
SOURCE_COLUMNS = ["prompt", "completion", "subject", "index"]
RUN_RE = re.compile(
    r"^unlearning-(?P<training_variant>original|r2-warmed)-"
    r"qwen-qwen2-5-(?P<model_size>0-5b|1-5b|3b|7b)-instruct-"
    r"(?P<author>.+)-(?P<reward_function>r\d+)"
    r"(?:-reasoning-[a-z0-9-]+)?_metrics\.csv$",
    re.IGNORECASE,
)
BASELINE_RE = re.compile(
    r"^holdout-baseline-qwen-qwen2-5-"
    r"(?P<model_size>0-5b|1-5b|3b|7b)-instruct-"
    r"(?P<author>.+)_metrics\.csv$",
    re.IGNORECASE,
)
WARMUP_RE = re.compile(
    r"^r2warmup_qwen_qwen2_5_"
    r"(?P<model_size>0_5b|1_5b|3b|7b)_instruct_"
    r"(?P<author>.+)_metrics\.csv$",
    re.IGNORECASE,
)
MODEL_SIZES = {
    "0-5b": "0.5B",
    "0_5b": "0.5B",
    "1-5b": "1.5B",
    "1_5b": "1.5B",
    "3b": "3B",
    "7b": "7B",
}


def run_metadata(path: Path) -> dict[str, str]:
    match = RUN_RE.fullmatch(path.name)
    if match is not None:
        reward_function = match.group("reward_function").lower()
        training_variant = match.group("training_variant").lower()
    elif match := BASELINE_RE.fullmatch(path.name):
        reward_function = "baseline"
        training_variant = "baseline"
    elif match := WARMUP_RE.fullmatch(path.name):
        reward_function = "r2-warmup"
        training_variant = "r2-warmup"
    else:
        raise ValueError(f"Cannot extract run metadata from {path.name}")

    model_size = MODEL_SIZES[match.group("model_size").lower()]
    author = re.sub(r"[-_]+", " ", match.group("author")).title()
    return {
        "run_name": path.stem.removesuffix("_metrics"),
        "author": author,
        "model_name": f"Qwen/Qwen2.5-{model_size}-Instruct",
        "model_size": model_size,
        "reward_function": reward_function,
        "training_variant": training_variant,
    }


def read_source(path: Path, max_items: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df[["subject", "prompt"]].eq("").any().any():
        raise ValueError(f"{path} has missing subject or prompt values")
    return df[SOURCE_COLUMNS].head(max_items).copy()


def load_score_cache(output_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(output_dir.glob("*_metrics.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        if set([*KEY_COLUMNS, *llm_judge_metrics]).issubset(df.columns):
            complete = df.dropna(subset=llm_judge_metrics)
            frames.append(complete[[*KEY_COLUMNS, *llm_judge_metrics]])
    if not frames:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *llm_judge_metrics])
    return pd.concat(frames, ignore_index=True).drop_duplicates(KEY_COLUMNS)


def score_file(
    source_path: Path,
    output_path: Path,
    cache: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    source = read_source(source_path, args.max_items_per_run)
    if output_path.exists() and not args.overwrite:
        existing = pd.read_csv(output_path, keep_default_na=False)
        if (
            set(llm_judge_metrics).issubset(existing.columns)
            and len(existing) == len(source)
            and not existing[llm_judge_metrics].isna().any().any()
        ):
            print(f"Reusing {output_path}", flush=True)
            return cache

    unique_pairs = source[KEY_COLUMNS].drop_duplicates()
    missing = unique_pairs.merge(cache[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", KEY_COLUMNS]

    scored_parts = []
    for concept, concept_df in missing.groupby("subject", sort=False):
        scored_parts.append(
            add_llm_judge_metrics(
                concept_df,
                concept=concept,
                judge_model=args.judge_model,
                judge_reasoning_effort=args.judge_reasoning_effort,
                max_concurrent_requests=args.judge_concurrency,
            )[[*KEY_COLUMNS, *llm_judge_metrics]]
        )
    if scored_parts:
        cache = pd.concat([cache, *scored_parts], ignore_index=True).drop_duplicates(KEY_COLUMNS)

    rescored = source.merge(cache, on=KEY_COLUMNS, how="left", validate="many_to_one")
    if rescored[llm_judge_metrics].isna().any().any():
        raise ValueError(f"Failed to score every row in {source_path}")
    rescored.to_csv(output_path, index=False)
    print(f"Wrote {len(rescored)} rows to {output_path}", flush=True)
    return cache


def aggregate(output_dir: Path) -> None:
    run_rows = []
    for path in sorted(output_dir.glob("*_metrics.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        if not set(llm_judge_metrics).issubset(df.columns):
            continue
        run_rows.append(
            {
                **run_metadata(path),
                "item_count": len(df),
                **{metric: df[metric].mean() for metric in llm_judge_metrics},
            }
        )
    if not run_rows:
        raise ValueError(f"No rescored run CSVs found under {output_dir}")

    per_run = pd.DataFrame(run_rows)
    per_run.to_csv(output_dir / "per_run_summary.csv", index=False)

    group_columns = ["model_name", "model_size", "reward_function", "training_variant"]
    summary_rows = []
    final_rows = []
    for values, group in per_run.groupby(group_columns, sort=True):
        metadata = dict(zip(group_columns, values, strict=True))
        summary_row = {
            **metadata,
            "run_count": len(group),
            "author_count": group["author"].nunique(),
        }
        final_row = summary_row.copy()
        for metric in llm_judge_metrics:
            q1, median, q3 = group[metric].quantile([0.25, 0.50, 0.75])
            summary_row[f"{metric}_q1"] = q1
            summary_row[f"{metric}_median"] = median
            summary_row[f"{metric}_q3"] = q3
            final_row[metric] = f"{median:.3f} [{q1:.3f}, {q3:.3f}]"
        summary_rows.append(summary_row)
        final_rows.append(final_row)

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_median_iqr.csv", index=False)
    pd.DataFrame(final_rows).to_csv(output_dir / "final_table.csv", index=False)
    print(f"Wrote summaries for {len(summary_rows)} experiment groups to {output_dir}")


def parse_args() -> argparse.Namespace:
    eval_config = OmegaConf.load(EVAL_CONFIG_PATH)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-items-per-run", type=int, default=None)
    parser.add_argument("--judge-model", default=str(eval_config.judge_model))
    parser.add_argument(
        "--judge-reasoning-effort",
        default=str(eval_config.judge_reasoning_effort),
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=int(eval_config.judge_concurrency),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild summaries from existing rescored run CSVs without API calls.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        source_paths = sorted(args.input_dir.glob("*_metrics.csv"))[: args.max_runs]
        if not source_paths:
            raise ValueError(f"No run CSVs found under {args.input_dir}")
        for path in source_paths:
            run_metadata(path)

        cache = (
            pd.DataFrame(columns=[*KEY_COLUMNS, *llm_judge_metrics])
            if args.overwrite
            else load_score_cache(args.output_dir)
        )
        print(
            f"Found {len(source_paths)} runs; reusing {len(cache)} cached unique scores.",
            flush=True,
        )
        for source_path in source_paths:
            cache = score_file(
                source_path,
                args.output_dir / source_path.name,
                cache,
                args,
            )

    aggregate(args.output_dir)


if __name__ == "__main__":
    main()
