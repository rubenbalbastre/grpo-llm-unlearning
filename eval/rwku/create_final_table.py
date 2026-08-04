#!/usr/bin/env python3
"""Aggregate RWKU results across authors using matched model baselines."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUTS_ROOT / "rwku_final_table.csv"
DEFAULT_EXPANDED_OUTPUT_CSV = DEFAULT_OUTPUTS_ROOT / "rwku_final_table_by_author.csv"

TRAINED_RUN_RE = re.compile(
    r"^unlearning-(?P<variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-(?P<author>.+)-"
    r"(?P<reward_type>r\d+)$",
    re.IGNORECASE,
)
WARMUP_RUN_RE = re.compile(
    r"^r2warmup_qwen_qwen2_5_"
    r"(?P<size>0_5b|1_5b|3b|7b)_instruct_(?P<author>.+)$",
    re.IGNORECASE,
)
BASELINE_RUN_RE = re.compile(
    r"^rwku-baseline-qwen-qwen2-5-(?P<size>0-5b|1-5b|3b|7b)-"
    r"instruct-(?P<author>.+)$",
    re.IGNORECASE,
)
SIZE_NAMES = {
    "0-5b": "0.5B",
    "0_5b": "0.5B",
    "1-5b": "1.5B",
    "1_5b": "1.5B",
    "3b": "3B",
    "7b": "7B",
}

CSV_METRICS = {
    "Forget Set FB down": "forget_fb",
    "Forget Set QA down": "forget_qa",
    "Forget Set AA down": "forget_aa",
    "Neighbor Set FB up": "neighbor_fb",
    "Neighbor Set QA up": "neighbor_qa",
    "MIA Set FM up": "mia_fm",
    "MIA Set RM down": "mia_rm",
    "Utility Set GA up": "utility_ga",
    "Utility Set RA up": "utility_ra",
    "Utility Set TRU up": "utility_tru",
    "Utility Set FAC up": "utility_fac",
    "Utility Set FLU up": "utility_flu",
}
DELTA_METRICS = [
    "forget_fb",
    "forget_qa",
    "forget_aa",
    "neighbor_fb",
    "neighbor_qa",
    "mia_fm",
    "mia_rm",
]
SPLIT_AGGREGATES = {
    "forget": ["forget_fb", "forget_qa", "forget_aa"],
    "neighbor": ["neighbor_fb", "neighbor_qa"],
}
UTILITY_METRICS = [
    "utility_ga",
    "utility_ra",
    "utility_tru",
    "utility_fac",
    "utility_flu",
]
ROW_METADATA_COLUMNS = {
    "author",
    "model_name_or_path",
    "model_size",
    "reward_type",
    "rlvr_mode",
    "run_name",
    "baseline_run_name",
}
FINAL_GROUP_COLUMNS = ["model_name_or_path", "reward_type", "rlvr_mode"]
EXPANDED_GROUP_COLUMNS = [*FINAL_GROUP_COLUMNS, "author"]
COUNT_COLUMNS = ["run_count", "author_count"]


def author_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_number(value: str | None) -> float | None:
    if value is None or value.strip().upper() in {"", "NA", "NAN"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def read_rwku_summary(path: Path) -> dict[str, float | None]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty RWKU summary: {path}")
    missing = set(CSV_METRICS) - set(row)
    if missing:
        raise ValueError(f"{path} is missing metrics: {sorted(missing)}")
    return {name: parse_number(row[column]) for column, name in CSV_METRICS.items()}


def find_baselines(
    outputs_root: Path,
) -> dict[tuple[str, str], tuple[str, dict[str, float | None]]]:
    baselines = {}
    pattern = "rwku-baseline-*/eval_rwku/rwku_summary_table.csv"
    for path in sorted(outputs_root.glob(pattern)):
        match = BASELINE_RUN_RE.fullmatch(path.parents[1].name)
        if match is None:
            continue
        size = SIZE_NAMES[match.group("size").lower()]
        key = (size, author_key(match.group("author")))
        if key in baselines:
            raise ValueError(f"Duplicate RWKU baseline for model/author {key}: {path}")
        baselines[key] = (path.parents[1].name, read_rwku_summary(path))
    return baselines


def trained_summary_paths(outputs_root: Path) -> list[Path]:
    return sorted(
        [
            *outputs_root.glob(
                "unlearning-*/final_model/eval_rwku/rwku_summary_table.csv"
            ),
            *outputs_root.glob(
                "r2warmup_*/final_model/eval_rwku/rwku_summary_table.csv"
            ),
        ]
    )


def run_metadata(path: Path) -> dict[str, str] | None:
    run_name = path.parents[2].name
    match = TRAINED_RUN_RE.fullmatch(run_name)
    warmup_match = WARMUP_RUN_RE.fullmatch(run_name)
    if match is None and warmup_match is None:
        return None

    if match is not None:
        size = SIZE_NAMES[match.group("size").lower()]
        author = match.group("author")
        variant = match.group("variant").lower()
        reward_type = match.group("reward_type").lower()
        rlvr_mode = "zero-RLVR" if variant == "original" else "warm-up"
    else:
        assert warmup_match is not None
        size = SIZE_NAMES[warmup_match.group("size").lower()]
        author = warmup_match.group("author")
        reward_type = "r2-warmup"
        rlvr_mode = "warm-up"

    return {
        "author": author,
        "model_name_or_path": f"Qwen/Qwen2.5-{size}-Instruct",
        "model_size": size,
        "reward_type": reward_type,
        "rlvr_mode": rlvr_mode,
        "run_name": run_name,
    }


def build_run_rows(outputs_root: Path) -> tuple[list[dict[str, object]], list[Path]]:
    baselines = find_baselines(outputs_root)
    rows: list[dict[str, object]] = []
    unmatched: list[Path] = []

    for path in trained_summary_paths(outputs_root):
        metadata = run_metadata(path)
        if metadata is None:
            continue

        baseline_entry = baselines.get(
            (metadata["model_size"], author_key(metadata["author"]))
        )
        if baseline_entry is None:
            unmatched.append(path)
            continue

        baseline_name, baseline = baseline_entry
        trained = read_rwku_summary(path)
        row: dict[str, object] = {
            **metadata,
            "baseline_run_name": baseline_name,
        }
        for metric in DELTA_METRICS:
            value = trained[metric]
            baseline_value = baseline[metric]
            row[f"{metric}_delta"] = (
                value - baseline_value
                if value is not None and baseline_value is not None
                else None
            )
        for aggregate, components in SPLIT_AGGREGATES.items():
            component_values = [row.get(f"{metric}_delta") for metric in components]
            row[f"{aggregate}_delta"] = (
                statistics.mean(float(value) for value in component_values)
                if all(value is not None for value in component_values)
                else None
            )
        for metric in UTILITY_METRICS:
            value = trained[metric]
            baseline_value = baseline[metric]
            row[metric] = value
            row[f"{metric}_delta"] = (
                value - baseline_value
                if value is not None and baseline_value is not None
                else None
            )
        rows.append(row)
    return rows, unmatched


def numeric_value(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
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


def metric_value_columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column, value in row.items():
            if (
                column not in ROW_METADATA_COLUMNS
                and numeric_value(value) is not None
                and column not in columns
            ):
                columns.append(column)
    return columns


def aggregate_rows(
    rows: list[dict[str, object]],
    group_columns: list[str],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(column, "")) for column in group_columns)].append(row)

    value_columns = metric_value_columns(rows)
    aggregate = []
    for key, group in sorted(groups.items()):
        output = {column: value for column, value in zip(group_columns, key, strict=True)}
        output["run_count"] = str(len(group))
        output["author_count"] = str(len({str(row.get("author", "")) for row in group}))
        for column in value_columns:
            values = [
                value
                for value in (numeric_value(row.get(column)) for row in group)
                if value is not None
            ]
            for stat_name, stat_value in summary_stats(values).items():
                output[f"{column}_{stat_name}"] = format_number(stat_value)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
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
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--expanded-output-csv",
        "--author-output-csv",
        type=Path,
        default=DEFAULT_EXPANDED_OUTPUT_CSV,
        help="Output CSV grouped by model, reward type, RLVR mode, and author.",
    )
    parser.add_argument("--expanded-output-md", type=Path, default=None)
    parser.add_argument(
        "--require-complete-baselines",
        action="store_true",
        help="Fail if any trained RWKU summary lacks its matching model/author baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rows, unmatched = build_run_rows(args.outputs_root)
    if unmatched and args.require_complete_baselines:
        examples = "\n".join(f"  {path}" for path in unmatched[:10])
        raise SystemExit(
            f"{len(unmatched)} trained RWKU result(s) lack a matching baseline:\n{examples}"
        )

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

    print(f"matched {len(run_rows)} trained result(s) to baselines")
    print(f"unmatched trained results: {len(unmatched)}")
    print(f"wrote final table: {args.output_csv}")
    print(f"wrote final markdown: {output_md}")
    print(f"wrote expanded table: {args.expanded_output_csv}")
    print(f"wrote expanded markdown: {expanded_output_md}")


if __name__ == "__main__":
    main()
