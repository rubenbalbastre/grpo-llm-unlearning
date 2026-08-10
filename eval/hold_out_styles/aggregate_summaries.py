#!/usr/bin/env python3
"""Combine hold-out-style summaries and average run means across authors."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import stdev


METRICS = [
    "lexical_leakage",
    "semantic_leakage",
    "helpful_relevant_answer",
    "refusal",
    "unhelpful_or_degenerate",
    "language_drift",
]
DETAIL_COLUMNS = [
    "forget_concept",
    "model_name_or_path",
    "reward_function",
    "stat",
    *METRICS,
]
SUMMARY_COLUMNS = [
    "model_name_or_path",
    "reward_function",
    *METRICS,
    *(f"{metric}_std_across_authors" for metric in METRICS),
]

TRAINED_MODEL_RE = re.compile(
    r"(?:^|/)unlearning-(?P<variant>original|r2-warmed)-"
    r"qwen-qwen2-5-(?P<size>0-5b|1-5b|3b)-instruct-.*-"
    r"(?P<reward>r\d+)(?:/final_model)?$",
    re.IGNORECASE,
)
SIZE_NAMES = {"0-5b": "0.5B", "1-5b": "1.5B", "3b": "3B"}


def model_and_reward(model_name_or_path: str) -> tuple[str, str]:
    """Return an author-independent model name and experiment reward label."""
    match = TRAINED_MODEL_RE.search(model_name_or_path)
    if match is None:
        return model_name_or_path, "baseline"

    model = f"Qwen/Qwen2.5-{SIZE_NAMES[match.group('size').lower()]}-Instruct"
    reward = f"{match.group('variant').lower()}-{match.group('reward').lower()}"
    return model, reward


def read_rows(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Deliberately scan immediate run directories only; ``old/`` is archival.
    for path in sorted(input_dir.glob("*/summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(["forget_concept", "model_name_or_path", "stat", *METRICS]) - set(
                reader.fieldnames or []
            )
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for source_row in reader:
                model, reward = model_and_reward(source_row["model_name_or_path"])
                rows.append(
                    {
                        "forget_concept": source_row["forget_concept"],
                        "model_name_or_path": model,
                        "reward_function": reward,
                        "stat": source_row["stat"],
                        **{metric: source_row[metric] for metric in METRICS},
                    }
                )
    return rows


def author_mean_summary(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    for row in rows:
        if row["stat"] != "mean":
            continue
        key = (row["model_name_or_path"], row["reward_function"])
        for metric in METRICS:
            values[key][metric].append(float(row[metric]))

    summary = []
    for (model, reward), metric_values in sorted(values.items()):
        summary.append(
            {
                "model_name_or_path": model,
                "reward_function": reward,
                **{
                    metric: sum(samples) / len(samples)
                    for metric, samples in metric_values.items()
                },
                **{
                    f"{metric}_std_across_authors": stdev(samples)
                    if len(samples) > 1
                    else ""
                    for metric, samples in metric_values.items()
                },
            }
        )
    return summary


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/hold_out_styles"))
    parser.add_argument("--all-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_output = args.all_output or args.input_dir / "all_summary.csv"
    summary_output = args.summary_output or args.input_dir / "authors_mean_summary.csv"
    rows = read_rows(args.input_dir)
    if not rows:
        raise ValueError(f"No immediate */summary.csv files found under {args.input_dir}")
    write_csv(all_output, DETAIL_COLUMNS, rows)
    summary = author_mean_summary(rows)
    write_csv(summary_output, SUMMARY_COLUMNS, summary)
    print(f"Wrote {len(rows)} rows to {all_output}")
    print(f"Wrote {len(summary)} rows to {summary_output}")


if __name__ == "__main__":
    main()
