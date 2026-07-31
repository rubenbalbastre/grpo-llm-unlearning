#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_TABLE_ROOT = DEFAULT_INPUT_ROOT / "hold_out_styles"
DEFAULT_OUTPUT_CSV = DEFAULT_TABLE_ROOT / "final_table.csv"
DEFAULT_SFT_WARMUP_CSV = DEFAULT_TABLE_ROOT / "final_table_sft_warmup.csv"
DEFAULT_CONDITIONAL_CSV = DEFAULT_TABLE_ROOT / "final_table_conditional.csv"
DEFAULT_FULL_CSV = DEFAULT_TABLE_ROOT / "final_table_full.csv"
SUMMARY_METADATA_COLUMNS = {"forget_concept", "model_name_or_path", "stat"}
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
GOOD_HIGH_METRICS = {"refusal"}
BAD_HIGH_METRICS = {
    "lexical_leakage",
    "semantic_leakage",
    "helpful_relevant_answer",
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
    forget_concept = first_row.get("forget_concept", "").strip()
    if not forget_concept:
        raise ValueError(f"Missing forget_concept in hold-out summary: {summary_path}")

    config = parse_simple_hydra_scalars(summary_path.parents[2] / "hydra_config.yaml")
    return {
        "forget_concept": forget_concept,
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
    return nested_run_metadata(summary_path, rows)


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


def concept_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def add_observed_provenance(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **row,
            "result_source": "observed",
            "run_status": "observed",
            "stop_reason": "",
        }
        for row in rows
    ]


def add_confirmed_no_learning_baselines(
    rows: list[dict[str, str]], outputs_root: Path
) -> list[dict[str, str]]:
    """Add baseline-valued rows only for runs carrying a low-reward stop marker."""
    baselines = {
        (concept_key(row["forget_concept"]), row["model_size"]): row
        for row in rows
        if row.get("training_variant") == "baseline"
    }
    observed_keys = {
        (
            concept_key(row["forget_concept"]),
            row["training_variant"],
            row["model_size"],
            row["reward_type"],
        )
        for row in rows
    }
    imputed_rows: list[dict[str, str]] = []
    for marker_path in sorted(outputs_root.glob("*/low_reward_stop.json")):
        match = STOPPED_RUN_RE.fullmatch(marker_path.parent.name)
        if not match:
            continue
        size = MODEL_SIZES[match.group("size").lower()]
        training_variant = match.group("training_variant").lower()
        reward_type = match.group("reward_type").lower()
        key = (concept_key(match.group("concept")), training_variant, size, reward_type)
        if key in observed_keys:
            continue
        baseline = baselines.get((key[0], size))
        if baseline is None:
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        imputed_rows.append(
            {
                **baseline,
                "reward_type": reward_type,
                "training_variant": training_variant,
                "rlvr_mode": "zero-RLVR" if training_variant == "original" else "warm-up",
                "run_name": marker_path.parent.name,
                "result_source": "baseline-imputed",
                "run_status": "no-learning",
                "stop_reason": str(marker.get("reason", "")),
            }
        )
    return [*rows, *imputed_rows]


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
            if column in {
                "forget_concept",
                "model_name_or_path",
                "model_size",
                "reward_type",
                "training_variant",
                "rlvr_mode",
                "run_name",
                "experiment_seed",
            }:
                continue
            if numeric_value(value) is not None and column not in columns:
                columns.append(column)
    return columns


def aggregate_by_reward_type(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("reward_type", ""),
            row.get("model_size", ""),
            row.get("rlvr_mode", ""),
        )
        groups[key].append(row)

    metric_columns = numeric_columns(rows)
    aggregate_rows = []
    for (reward_type, model_size, rlvr_mode), group_rows in sorted(groups.items()):
        output = {
            "reward_type": reward_type,
            "model_size": model_size,
            "model_name_or_path": group_rows[0].get("model_name_or_path", ""),
            "rlvr_mode": rlvr_mode,
            "training_variants": "|".join(
                sorted({row.get("training_variant", "") for row in group_rows})
            ),
            "run_count": str(len(group_rows)),
            "observed_run_count": str(
                sum(row.get("result_source", "observed") == "observed" for row in group_rows)
            ),
            "no_learning_count": str(
                sum(row.get("run_status") == "no-learning" for row in group_rows)
            ),
            "baseline_imputed_count": str(
                sum(row.get("result_source") == "baseline-imputed" for row in group_rows)
            ),
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


def aggregate_sft_warmup_by_model(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("training_variant") == "r2-warmup":
            groups[row.get("model_size", "")].append(row)

    mean_columns = metric_mean_columns(rows)
    aggregate_rows = []
    for model_size, group_rows in sorted(groups.items()):
        output = {
            "model_size": model_size,
            "model_name_or_path": group_rows[0].get("model_name_or_path", ""),
            "author_count": str(len(group_rows)),
        }
        for column in mean_columns:
            values = [
                value
                for value in (numeric_value(row.get(column, "")) for row in group_rows)
                if value is not None
            ]
            output[column] = format_number(statistics.mean(values) if values else None)
            output[f"{column}_across_authors_std"] = format_number(
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
    headers = [
        "forget_concept",
        "model_name_or_path",
        "model_size",
        "reward_type",
        "training_variant",
        "rlvr_mode",
        "run_name",
        "experiment_seed",
    ]
    headers.extend(column.removesuffix("_mean") for column in mean_columns)

    table_rows = []
    for row in rows:
        table_row = [
            row.get("forget_concept", ""),
            row.get("model_name_or_path", ""),
            row.get("model_size", ""),
            row.get("reward_type", ""),
            row.get("training_variant", ""),
            row.get("rlvr_mode", ""),
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
    headers = [
        "reward_type",
        "model_size",
        "rlvr_mode",
        "training_variants",
        "run_count",
        "observed_run_count",
        "no_learning_count",
        "baseline_imputed_count",
        "experiment_seeds",
    ]
    headers.extend(column.removesuffix("_mean") for column in mean_columns)

    table_rows = []
    for row in rows:
        table_row = [
            row.get("reward_type", ""),
            row.get("model_size", ""),
            row.get("rlvr_mode", ""),
            row.get("training_variants", ""),
            row.get("run_count", ""),
            row.get("observed_run_count", ""),
            row.get("no_learning_count", ""),
            row.get("baseline_imputed_count", ""),
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


def write_sft_warmup_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    mean_columns = metric_mean_columns(rows)
    headers = ["model_size", "author_count"]
    headers.extend(column.removesuffix("_mean") for column in mean_columns)

    table_rows = []
    for row in rows:
        table_row = [row.get("model_size", ""), row.get("author_count", "")]
        for mean_column in mean_columns:
            metric = mean_column.removesuffix("_mean")
            table_row.append(
                format_metric_cell(
                    metric,
                    row.get(mean_column, ""),
                    row.get(f"{mean_column}_across_authors_std", ""),
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
        "--sft-warmup-csv",
        type=Path,
        default=DEFAULT_SFT_WARMUP_CSV,
        help="Output CSV averaging SFT warm-up model results across authors.",
    )
    parser.add_argument(
        "--sft-warmup-md",
        type=Path,
        default=None,
        help="Markdown output averaging SFT warm-up model results across authors.",
    )
    parser.add_argument(
        "--conditional-csv",
        type=Path,
        default=DEFAULT_CONDITIONAL_CSV,
        help="Aggregate output using observed/successful runs only.",
    )
    parser.add_argument(
        "--full-csv",
        type=Path,
        default=DEFAULT_FULL_CSV,
        help="Aggregate output with confirmed no-learning runs replaced by baselines.",
    )
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

    rows = add_observed_provenance(build_table(summary_paths))
    if not rows:
        raise SystemExit("No non-empty summary CSV files found.")

    by_reward_type_csv = args.by_reward_type_csv
    if by_reward_type_csv is None:
        by_reward_type_csv = args.output_csv.with_name(f"{args.output_csv.stem}_by_reward_type.csv")
    output_md = args.output_md or args.output_csv.with_suffix(".md")
    by_reward_type_md = args.by_reward_type_md or by_reward_type_csv.with_suffix(".md")
    sft_warmup_md = args.sft_warmup_md or args.sft_warmup_csv.with_suffix(".md")
    conditional_md = args.conditional_csv.with_suffix(".md")
    full_md = args.full_csv.with_suffix(".md")

    reward_type_rows = aggregate_by_reward_type(rows)
    conditional_rows = reward_type_rows
    full_run_rows = add_confirmed_no_learning_baselines(rows, REPO_ROOT / "outputs")
    full_rows = aggregate_by_reward_type(full_run_rows)
    sft_warmup_rows = aggregate_sft_warmup_by_model(rows)
    write_csv(
        args.output_csv,
        rows,
        preferred_columns=[
            "forget_concept",
            "model_name_or_path",
            "model_size",
            "reward_type",
            "training_variant",
            "rlvr_mode",
            "run_name",
            "experiment_seed",
        ],
    )
    write_csv(
        by_reward_type_csv,
        reward_type_rows,
        preferred_columns=[
            "reward_type",
            "model_size",
            "model_name_or_path",
            "rlvr_mode",
            "training_variants",
            "run_count",
            "forget_concepts",
            "run_names",
            "experiment_seeds",
        ],
    )
    write_csv(
        args.sft_warmup_csv,
        sft_warmup_rows,
        preferred_columns=["model_size", "model_name_or_path", "author_count"],
    )
    aggregate_columns = [
        "reward_type",
        "model_size",
        "model_name_or_path",
        "rlvr_mode",
        "training_variants",
        "run_count",
        "observed_run_count",
        "no_learning_count",
        "baseline_imputed_count",
        "forget_concepts",
        "run_names",
        "experiment_seeds",
    ]
    write_csv(args.conditional_csv, conditional_rows, preferred_columns=aggregate_columns)
    write_csv(args.full_csv, full_rows, preferred_columns=aggregate_columns)
    write_run_markdown(output_md, rows)
    write_reward_type_markdown(by_reward_type_md, reward_type_rows)
    write_sft_warmup_markdown(sft_warmup_md, sft_warmup_rows)
    write_reward_type_markdown(conditional_md, conditional_rows)
    write_reward_type_markdown(full_md, full_rows)
    print(f"read {len(summary_paths)} summary CSV file(s)")
    print(f"wrote final table: {args.output_csv}")
    print(f"wrote reward-type aggregate table: {by_reward_type_csv}")
    print(f"wrote final markdown table: {output_md}")
    print(f"wrote reward-type aggregate markdown table: {by_reward_type_md}")
    print(f"wrote SFT warm-up author aggregate table: {args.sft_warmup_csv}")
    print(f"wrote SFT warm-up author aggregate markdown table: {sft_warmup_md}")
    print(f"wrote conditional aggregate table: {args.conditional_csv}")
    print(f"wrote conditional aggregate markdown table: {conditional_md}")
    print(f"wrote full baseline-substituted table: {args.full_csv}")
    print(f"wrote full baseline-substituted markdown table: {full_md}")


if __name__ == "__main__":
    main()
