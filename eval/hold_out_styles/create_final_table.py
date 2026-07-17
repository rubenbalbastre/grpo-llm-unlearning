#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs" / "completion_analysis"
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_ROOT / "final_table.csv"
SUMMARY_METADATA_COLUMNS = {"forget_concept", "model_name_or_path", "stat"}
RUN_RE = re.compile(r"(unlearning-\d+)")
GOOD_HIGH_METRICS = {"forgetting_reward", "forgetting_fuzzy_reward", "refusal"}
BAD_HIGH_METRICS = {
    "lexical_leakage",
    "semantic_leakage",
    "helpful_relevant_answer",
    "unhelpful_or_degenerate",
    "language_drift",
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


def find_run_name(summary_path: Path, rows: list[dict[str, str]]) -> str:
    for row in rows:
        match = RUN_RE.search(row.get("model_name_or_path", ""))
        if match:
            return match.group(1)

    match = RUN_RE.search(str(summary_path.parent))
    if match:
        return match.group(1)

    return summary_path.parent.name


def run_config_path(run_name: str) -> Path:
    return REPO_ROOT / "outputs" / run_name / "hydra_config.yaml"


def summary_metric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in SUMMARY_METADATA_COLUMNS and column not in columns:
                columns.append(column)
    return columns


def flatten_summary_metrics(rows: list[dict[str, str]]) -> dict[str, str]:
    output: dict[str, str] = {}
    metric_columns = summary_metric_columns(rows)
    for row in rows:
        stat = row.get("stat", "").strip()
        if not stat:
            continue
        for metric in metric_columns:
            output[f"{metric}_{stat}"] = row.get(metric, "")
    return output


def metadata_for_summary(summary_path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    first_row = rows[0] if rows else {}
    run_name = find_run_name(summary_path, rows)
    hydra = parse_simple_hydra_scalars(run_config_path(run_name))
    return {
        "forget_concept": hydra.get("experiment.forget_concept", first_row.get("forget_concept", "")),
        "reward_type": hydra.get("reward.type", ""),
        "run_name": hydra.get("wandb.run_name", run_name),
        "experiment_seed": hydra.get("experiment.seed", ""),
    }


def build_table(summary_paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for summary_path in summary_paths:
        summary_rows = read_summary(summary_path)
        if not summary_rows:
            continue
        rows.append(
            {
                **metadata_for_summary(summary_path, summary_rows),
                **flatten_summary_metrics(summary_rows),
            }
        )
    return rows


def numeric_value(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column, value in row.items():
            if column in {"forget_concept", "reward_type", "run_name", "experiment_seed"}:
                continue
            if numeric_value(value) is not None and column not in columns:
                columns.append(column)
    return columns


def aggregate_by_reward_type(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("reward_type", "")].append(row)

    metric_columns = numeric_columns(rows)
    aggregate_rows = []
    for reward_type, group_rows in sorted(groups.items()):
        output = {
            "reward_type": reward_type,
            "run_count": str(len(group_rows)),
            "forget_concepts": "|".join(sorted({row.get("forget_concept", "") for row in group_rows})),
            "run_names": "|".join(sorted({row.get("run_name", "") for row in group_rows})),
            "experiment_seeds": "|".join(sorted({row.get("experiment_seed", "") for row in group_rows})),
        }
        for column in metric_columns:
            values = [
                value
                for value in (numeric_value(row.get(column, "")) for row in group_rows)
                if value is not None
            ]
            output[column] = format_number(statistics.mean(values) if values else None)
            output[f"{column}_across_runs_std"] = format_number(
                statistics.stdev(values) if len(values) > 1 else None
            )
        aggregate_rows.append(output)
    return aggregate_rows


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


def metric_mean_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column.endswith("_mean") and column not in columns:
                columns.append(column)
    return columns


def color_for_metric(metric: str, value: float | None) -> str | None:
    if value is None:
        return None
    if metric in GOOD_HIGH_METRICS:
        if value >= 0.9:
            return "#167c3b"
        if value >= 0.7:
            return "#946200"
        return "#b42318"
    if metric in BAD_HIGH_METRICS:
        if value <= 0.1:
            return "#167c3b"
        if value <= 0.3:
            return "#946200"
        return "#b42318"
    return None


def format_metric_cell(metric: str, mean: str, std: str) -> str:
    mean_value = numeric_value(mean)
    std_value = numeric_value(std)
    if mean_value is None:
        return ""

    text = f"{mean_value:.3f}"
    if std_value is not None:
        text = f"{text} +/- {std_value:.3f}"

    color = color_for_metric(metric, mean_value)
    if color is None:
        return text
    return f'<span style="color:{color}">{text}</span>'


def markdown_escape(value: str) -> str:
    return value.replace("|", "<br>").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def write_run_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    mean_columns = metric_mean_columns(rows)
    headers = ["forget_concept", "reward_type", "run_name", "experiment_seed"]
    headers.extend(column.removesuffix("_mean") for column in mean_columns)

    table_rows = []
    for row in rows:
        table_row = [
            row.get("forget_concept", ""),
            row.get("reward_type", ""),
            row.get("run_name", ""),
            row.get("experiment_seed", ""),
        ]
        for mean_column in mean_columns:
            metric = mean_column.removesuffix("_mean")
            table_row.append(format_metric_cell(metric, row.get(mean_column, ""), row.get(f"{metric}_std", "")))
        table_rows.append(table_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(headers, table_rows) + "\n", encoding="utf-8")


def write_reward_type_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    mean_columns = metric_mean_columns(rows)
    headers = ["reward_type", "run_count", "forget_concepts", "experiment_seeds"]
    headers.extend(column.removesuffix("_mean") for column in mean_columns)

    table_rows = []
    for row in rows:
        table_row = [
            row.get("reward_type", ""),
            row.get("run_count", ""),
            row.get("forget_concepts", ""),
            row.get("experiment_seeds", ""),
        ]
        for mean_column in mean_columns:
            metric = mean_column.removesuffix("_mean")
            table_row.append(
                format_metric_cell(
                    metric,
                    row.get(mean_column, ""),
                    row.get(f"{mean_column}_across_runs_std", ""),
                )
            )
        table_rows.append(table_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(headers, table_rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one final table from generated completions summary.csv files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown output for the per-run final table.",
    )
    parser.add_argument(
        "--by-reward-type-csv",
        type=Path,
        default=None,
        help="Output CSV for rows aggregated by reward_type.",
    )
    parser.add_argument(
        "--by-reward-type-md",
        type=Path,
        default=None,
        help="Markdown output for rows aggregated by reward_type.",
    )
    parser.add_argument(
        "--summary-glob",
        default="generated-completions-*/summary.csv",
        help="Glob below --input-root selecting completion summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = sorted(args.input_root.glob(args.summary_glob))
    if not summary_paths:
        raise SystemExit(f"No summary CSV files found with {args.input_root / args.summary_glob}")

    rows = build_table(summary_paths)
    if not rows:
        raise SystemExit("No non-empty summary CSV files found.")

    by_reward_type_csv = args.by_reward_type_csv
    if by_reward_type_csv is None:
        by_reward_type_csv = args.output_csv.with_name(f"{args.output_csv.stem}_by_reward_type.csv")
    output_md = args.output_md or args.output_csv.with_suffix(".md")
    by_reward_type_md = args.by_reward_type_md or by_reward_type_csv.with_suffix(".md")

    reward_type_rows = aggregate_by_reward_type(rows)
    write_csv(
        args.output_csv,
        rows,
        preferred_columns=["forget_concept", "reward_type", "run_name", "experiment_seed"],
    )
    write_csv(
        by_reward_type_csv,
        reward_type_rows,
        preferred_columns=["reward_type", "run_count", "forget_concepts", "run_names", "experiment_seeds"],
    )
    write_run_markdown(output_md, rows)
    write_reward_type_markdown(by_reward_type_md, reward_type_rows)
    print(f"read {len(summary_paths)} summary CSV file(s)")
    print(f"wrote final table: {args.output_csv}")
    print(f"wrote reward-type aggregate table: {by_reward_type_csv}")
    print(f"wrote final markdown table: {output_md}")
    print(f"wrote reward-type aggregate markdown table: {by_reward_type_md}")


if __name__ == "__main__":
    main()
