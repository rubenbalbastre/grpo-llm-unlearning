#!/usr/bin/env python3
"""Create aggregate hold-out-style result tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_TABLE_ROOT = DEFAULT_INPUT_ROOT / "hold_out_styles"
DEFAULT_OUTPUT_CSV = DEFAULT_TABLE_ROOT / "final_table.csv"
DEFAULT_EXPANDED_OUTPUT_CSV = DEFAULT_TABLE_ROOT / "final_table_by_author.csv"

SUMMARY_METADATA_COLUMNS = {"forget_concept", "model_name_or_path", "stat"}
ROW_METADATA_COLUMNS = {
    "author",
    "model_name_or_path",
    "model_size",
    "reward_type",
    "training_variant",
    "rlvr_mode",
    "run_name",
    "experiment_seed",
}
FINAL_GROUP_COLUMNS = ["model_name_or_path", "reward_type", "rlvr_mode"]
EXPANDED_GROUP_COLUMNS = [*FINAL_GROUP_COLUMNS, "author"]
COUNT_COLUMNS = ["run_count", "author_count"]
STAT_NAMES = ["mean", "std", "Q1", "Q3", "median"]

STOPPED_RUN_RE = re.compile(
    r"^unlearning-(?P<training_variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-(?P<concept>.+)-"
    r"(?P<reward_type>r\d+)$",
    re.IGNORECASE,
)
WARMUP_RUN_RE = re.compile(
    r"^r2warmup_qwen_qwen2_5_"
    r"(?P<size>0_5b|1_5b|3b|7b)_instruct_.+$",
    re.IGNORECASE,
)
HOLDOUT_BASELINE_RUN_RE = re.compile(
    r"^holdout-baseline-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-.+$",
    re.IGNORECASE,
)
MODEL_SIZES = {
    "0.5b": "0.5B",
    "0-5b": "0.5B",
    "0_5b": "0.5B",
    "1.5b": "1.5B",
    "1-5b": "1.5B",
    "1_5b": "1.5B",
    "3b": "3B",
    "7b": "7B",
}


def parse_simple_hydra_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            key, _, value = stripped.partition(":")
            section = key if not value.strip() else None
            continue

        if section is None or indent != 2:
            continue

        key, separator, value = stripped.partition(":")
        if separator and value.strip():
            values[f"{section}.{key}"] = value.strip().strip("'\"")

    return values


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nested_run_metadata(
    summary_path: Path, rows: list[dict[str, str]]
) -> dict[str, str]:
    if (
        summary_path.parent.name != "hold_out_eval"
        or summary_path.parent.parent.name != "final_model"
    ):
        raise ValueError(
            "Hold-out summaries must use "
            f"<run>/final_model/hold_out_eval/summary.csv: {summary_path}"
        )

    run_name = summary_path.parents[2].name
    match = STOPPED_RUN_RE.fullmatch(run_name)
    if match:
        size = MODEL_SIZES[match.group("size").lower()]
        training_variant = match.group("training_variant").lower()
        reward_type = match.group("reward_type").lower()
    else:
        match = WARMUP_RUN_RE.fullmatch(run_name)
        if match:
            size = MODEL_SIZES[match.group("size").lower()]
            training_variant = "r2-warmup"
            reward_type = "r2-warmup"
        else:
            match = HOLDOUT_BASELINE_RUN_RE.fullmatch(run_name)
            if not match:
                raise ValueError(
                    f"Cannot extract metadata from nested hold-out run: {run_name}"
                )
            size = MODEL_SIZES[match.group("size").lower()]
            training_variant = "baseline"
            reward_type = "baseline"

    first_row = rows[0] if rows else {}
    author = first_row.get("forget_concept", "").strip()
    if not author:
        raise ValueError(f"Missing forget_concept in hold-out summary: {summary_path}")

    config = parse_simple_hydra_scalars(summary_path.parents[2] / "hydra_config.yaml")
    return {
        "author": author,
        "model_name_or_path": f"Qwen/Qwen2.5-{size}-Instruct",
        "model_size": size,
        "reward_type": reward_type,
        "training_variant": training_variant,
        "rlvr_mode": {
            "original": "zero-RLVR",
            "r2-warmed": "warm-up",
            "r2-warmup": "warm-up",
            "baseline": "baseline",
        }[training_variant],
        "run_name": run_name,
        "experiment_seed": config.get("experiment.seed", ""),
    }


def metric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in SUMMARY_METADATA_COLUMNS and column not in columns:
                columns.append(column)
    return columns


def flatten_mean_metrics(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return one metric-value mapping per run from a summary.csv table.

    Preferred input is one row per statistic with stat=mean. Older flattened
    summaries that already contain *_mean columns are also accepted.
    """
    output: dict[str, str] = {}
    for row in rows:
        stat = row.get("stat", "").strip()
        if stat == "mean":
            for column in metric_columns(rows):
                output[f"{column}_mean"] = row.get(column, "")
        elif not stat:
            for column in metric_columns(rows):
                if column.endswith("_mean"):
                    output[column] = row.get(column, "")
    return output


def numeric_value(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "Q1": percentile(values, 0.25),
        "Q3": percentile(values, 0.75),
        "median": statistics.median(values) if values else None,
    }


def metric_value_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column, value in row.items():
            if (
                column not in ROW_METADATA_COLUMNS
                and column.endswith("_mean")
                and numeric_value(value) is not None
                and column not in columns
            ):
                columns.append(column)
    return columns


def build_run_rows(summary_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in summary_paths:
        summary_rows = read_summary(summary_path)
        if not summary_rows:
            continue
        rows.append(
            {
                **nested_run_metadata(summary_path, summary_rows),
                **flatten_mean_metrics(summary_rows),
            }
        )
    return rows


def aggregate_rows(
    rows: list[dict[str, str]],
    group_columns: list[str],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column, "") for column in group_columns)].append(row)

    value_columns = metric_value_columns(rows)
    aggregate = []
    for key, group in sorted(groups.items()):
        output = {column: value for column, value in zip(group_columns, key, strict=True)}
        output["run_count"] = str(len(group))
        output["author_count"] = str(len({row.get("author", "") for row in group}))
        for column in value_columns:
            metric = column.removesuffix("_mean")
            values = [
                value
                for value in (numeric_value(row.get(column)) for row in group)
                if value is not None
            ]
            for stat_name, stat_value in summary_stats(values).items():
                output[f"{metric}_{stat_name}"] = format_number(stat_value)
        aggregate.append(output)
    return aggregate


def ordered_fieldnames(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    fieldnames = preferred.copy()
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, str]], preferred_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows, preferred_columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "<br>").replace("\n", " ")


def write_markdown(path: Path, rows: list[dict[str, str]], preferred_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ordered_fieldnames(rows, preferred_columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(markdown_escape(row.get(column, "")) for column in headers) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--expanded-output-csv",
        type=Path,
        default=DEFAULT_EXPANDED_OUTPUT_CSV,
        help="Output CSV grouped by model, reward type, RLVR mode, and author.",
    )
    parser.add_argument("--expanded-output-md", type=Path, default=None)
    parser.add_argument(
        "--summary-glob",
        default="*/final_model/hold_out_eval/summary.csv",
        help="Glob below --input-root selecting completion summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = sorted(args.input_root.glob(args.summary_glob))
    if not summary_paths:
        raise SystemExit(f"No summary CSV files found with {args.input_root / args.summary_glob}")

    run_rows = build_run_rows(summary_paths)
    if not run_rows:
        raise SystemExit("No non-empty summary CSV files found.")

    final_rows = aggregate_rows(run_rows, FINAL_GROUP_COLUMNS)
    expanded_rows = aggregate_rows(run_rows, EXPANDED_GROUP_COLUMNS)
    output_md = args.output_md or args.output_csv.with_suffix(".md")
    expanded_output_md = (
        args.expanded_output_md or args.expanded_output_csv.with_suffix(".md")
    )

    final_columns = [*FINAL_GROUP_COLUMNS, *COUNT_COLUMNS]
    expanded_columns = [*EXPANDED_GROUP_COLUMNS, *COUNT_COLUMNS]
    write_csv(args.output_csv, final_rows, preferred_columns=final_columns)
    write_markdown(output_md, final_rows, preferred_columns=final_columns)
    write_csv(args.expanded_output_csv, expanded_rows, preferred_columns=expanded_columns)
    write_markdown(
        expanded_output_md,
        expanded_rows,
        preferred_columns=expanded_columns,
    )

    print(f"read {len(summary_paths)} summary CSV file(s)")
    print(f"aggregated {len(run_rows)} run row(s)")
    print(f"wrote final table: {args.output_csv}")
    print(f"wrote final markdown: {output_md}")
    print(f"wrote expanded table: {args.expanded_output_csv}")
    print(f"wrote expanded markdown: {expanded_output_md}")


if __name__ == "__main__":
    main()
