#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "tables"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "behaviour.csv"
DEFAULT_AUTHOR_CSV = DEFAULT_OUTPUT_ROOT / "behaviour_authors.csv"
SUMMARY_METADATA_COLUMNS = {"forget_concept", "model_name_or_path", "stat"}
BASELINE_FOLDER_RE = re.compile(
    r"^(?P<concept>.+)-Qwen-Qwen2\.5-(?P<size>0\.5B|1\.5B|3B|7B)-Instruct$",
    re.IGNORECASE,
)
UNLEARNING_FOLDER_RE = re.compile(
    r"^(?P<concept>.+)-\.-outputs-unlearning-"
    r"(?P<training_variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-.+-(?P<reward_type>r\d+)"
    r"(?:-reasoning-[a-z0-9-]+)?-final_model$",
    re.IGNORECASE,
)
WARMUP_FOLDER_RE = re.compile(
    r"^(?P<concept>.+)-\.-outputs-r2warmup_qwen_qwen2_5_"
    r"(?P<size>0_5b|1_5b|3b|7b)_instruct_.+-final_model$",
    re.IGNORECASE,
)
WARMUP_RUN_RE = re.compile(
    r"^r2warmup_qwen_qwen2_5_(?P<size>0_5b|1_5b|3b|7b)_instruct_.+$",
    re.IGNORECASE,
)
MODEL_PATH_RE = re.compile(
    r"Qwen2\.5-(?P<size>0\.5B|1\.5B|3B|7B)-Instruct$",
    re.IGNORECASE,
)
STOPPED_RUN_RE = re.compile(
    r"^unlearning-(?P<training_variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-(?P<concept>.+)-"
    r"(?P<reward_type>r\d+)(?:-reasoning-[a-z0-9-]+)?$",
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
def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def folder_metadata(summary_path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    """Extract run metadata from a behaviour run directory name."""
    folder_name = summary_path.parent.name
    first_row = rows[0] if rows else {}

    if folder_name == "hold_out_eval":
        model_dir = summary_path.parents[1]
        run_dir = model_dir.parent if model_dir.name == "final_model" else model_dir
        run_name = run_dir.name
        model_match = MODEL_PATH_RE.search(first_row.get("model_name_or_path", ""))
        run_match = STOPPED_RUN_RE.fullmatch(run_name)
        warmup_match = WARMUP_RUN_RE.fullmatch(run_name)

        if run_match:
            size = MODEL_SIZES[run_match.group("size").lower()]
            training_variant = run_match.group("training_variant").lower()
            reward_type = run_match.group("reward_type").lower()
        elif warmup_match:
            size = MODEL_SIZES[warmup_match.group("size").lower()]
            training_variant = "r2-warmup"
            reward_type = "r2-warmup"
        elif run_name.startswith("holdout-baseline-") and model_match:
            size = MODEL_SIZES[model_match.group("size").lower()]
            training_variant = "baseline"
            reward_type = "baseline"
        else:
            raise ValueError(f"Cannot extract metadata from hold-out run: {run_dir}")

        return {
            "author": first_row.get("forget_concept", "").strip(),
            "model_name_or_path": f"Qwen/Qwen2.5-{size}-Instruct",
            "model_size": size,
            "reward_function": reward_type,
            "training_variant": training_variant,
            "training_initialization": {
                "original": "cold",
                "r2-warmed": "warm",
                "r2-warmup": "warm",
                "baseline": "baseline",
            }[training_variant],
            "run_name": run_name,
            "experiment_seed": "",
        }

    match = BASELINE_FOLDER_RE.fullmatch(folder_name)
    if match:
        training_variant = "baseline"
        reward_type = "baseline"
    else:
        match = UNLEARNING_FOLDER_RE.fullmatch(folder_name)
        if match:
            training_variant = match.group("training_variant").lower()
            reward_type = match.group("reward_type").lower()
        else:
            match = WARMUP_FOLDER_RE.fullmatch(folder_name)
            if not match:
                raise ValueError(
                    f"Cannot extract metadata from hold-out-style folder: {summary_path.parent}"
                )
            training_variant = "r2-warmup"
            reward_type = "r2-warmup"

    size = MODEL_SIZES[match.group("size").lower()]
    folder_concept = match.group("concept")
    csv_concept = first_row.get("forget_concept", "").strip()
    csv_concept_slug = re.sub(r"[\s_]+", "-", csv_concept.lower()).strip("-")
    forget_concept = csv_concept if csv_concept_slug == folder_concept.lower() else folder_concept

    return {
        "author": forget_concept,
        "model_name_or_path": f"Qwen/Qwen2.5-{size}-Instruct",
        "model_size": size,
        "reward_function": reward_type,
        "training_variant": training_variant,
        "training_initialization": {
            "original": "cold",
            "r2-warmed": "warm",
            "r2-warmup": "warm",
            "baseline": "baseline",
        }[training_variant],
        "run_name": folder_name,
        "experiment_seed": "",
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
    return folder_metadata(summary_path, rows)


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
        (concept_key(row["author"]), row["model_size"]): row
        for row in rows
        if row.get("training_variant") == "baseline"
    }
    observed_keys = {
        (
            concept_key(row["author"]),
            row["training_variant"],
            row["model_size"],
            row["reward_function"],
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
                "reward_function": reward_type,
                "training_variant": training_variant,
                "training_initialization": "cold" if training_variant == "original" else "warm",
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


def metric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column.endswith("_mean"):
                metric = column.removesuffix("_mean")
                if metric not in columns:
                    columns.append(metric)
    return columns


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("reward_function", ""),
            row.get("model_size", ""),
            row.get("training_initialization", ""),
        )
        groups[key].append(row)

    metrics = metric_columns(rows)
    aggregate_rows = []
    for (reward_function, model_size, training_initialization), group_rows in sorted(groups.items()):
        output = {
            "reward_function": reward_function,
            "model_size": model_size,
            "training_initialization": training_initialization,
        }
        for metric in metrics:
            values = [
                value
                for value in (
                    numeric_value(row.get(f"{metric}_mean", "")) for row in group_rows
                )
                if value is not None
            ]
            output[f"{metric}_median"] = format_number(quantile(values, 0.5))
            output[f"{metric}_q1"] = format_number(quantile(values, 0.25))
            output[f"{metric}_q3"] = format_number(quantile(values, 0.75))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create author-level and aggregate behavioural tables."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--author-output-csv",
        type=Path,
        default=DEFAULT_AUTHOR_CSV,
        help="CSV output containing one row per author and run.",
    )
    parser.add_argument(
        "--summary-glob",
        default="**/hold_out_eval/summary.csv",
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

    author_rows = add_confirmed_no_learning_baselines(rows, REPO_ROOT / "outputs")
    grouped_rows = aggregate_rows(author_rows)
    write_csv(
        args.author_output_csv,
        author_rows,
        preferred_columns=[
            "author",
            "model_name_or_path",
            "model_size",
            "reward_function",
            "training_variant",
            "training_initialization",
            "run_name",
            "experiment_seed",
        ],
    )
    write_csv(
        args.output_csv,
        grouped_rows,
        preferred_columns=[
            "reward_function",
            "model_size",
            "training_initialization",
        ],
    )
    print(f"read {len(summary_paths)} summary CSV file(s)")
    print(f"wrote author table: {args.author_output_csv}")
    print(f"wrote aggregate table: {args.output_csv}")


if __name__ == "__main__":
    main()
