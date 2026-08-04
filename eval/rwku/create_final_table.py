#!/usr/bin/env python3
"""Aggregate RWKU results across authors using matched original-model baselines."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUTS_ROOT / "rwku_final_table.csv"
DEFAULT_AUTHOR_CSV = DEFAULT_OUTPUTS_ROOT / "rwku_author_deltas.csv"

TRAINED_RUN_RE = re.compile(
    r"^unlearning-(?P<variant>original|r2-warmed)-qwen-qwen2-5-"
    r"(?P<size>0-5b|1-5b|3b|7b)-instruct-(?P<author>.+)-"
    r"(?P<reward_type>r\d+)(?:-reasoning-low)?$",
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
METRIC_LABELS = {
    "forget": "Δ Forget ↓",
    "neighbor": "Δ Neighbor ↑",
    "forget_fb": "Δ Forget FB ↓",
    "forget_qa": "Δ Forget QA ↓",
    "forget_aa": "Δ Forget AA ↓",
    "neighbor_fb": "Δ Neighbor FB ↑",
    "neighbor_qa": "Δ Neighbor QA ↑",
    "mia_fm": "Δ MIA FM ↑",
    "mia_rm": "Δ MIA RM ↓",
    "utility_ga": "Δ Utility GA ↑",
    "utility_ra": "Δ Utility RA ↑",
    "utility_tru": "Δ Utility TRU ↑",
    "utility_fac": "Δ Utility FAC ↑",
    "utility_flu": "Δ Utility FLU ↑",
}

AUTHOR_COLUMNS = [
    "author",
    "model_name_or_path",
    "model_size",
    "reward_function",
    "rlvr_mode",
    "run_name",
    "baseline_run_name",
    *[f"{metric}_delta" for metric in SPLIT_AGGREGATES],
    *[f"{metric}_delta" for metric in DELTA_METRICS],
    *[f"{metric}_delta" for metric in UTILITY_METRICS],
    *UTILITY_METRICS,
]
FINAL_COLUMNS = [
    "model_name_or_path",
    "model_size",
    "reward_function",
    "rlvr_mode",
    "author_count",
]
for _metric in SPLIT_AGGREGATES:
    FINAL_COLUMNS.extend(
        [
            f"{_metric}_delta_mean",
            f"{_metric}_delta_std_across_authors",
            f"{_metric}_author_count",
        ]
    )
for _metric in DELTA_METRICS:
    FINAL_COLUMNS.extend(
        [
            f"{_metric}_delta_mean",
            f"{_metric}_delta_std_across_authors",
            f"{_metric}_author_count",
        ]
    )
for _metric in UTILITY_METRICS:
    FINAL_COLUMNS.extend(
        [
            f"{_metric}_delta_mean",
            f"{_metric}_delta_std_across_authors",
            f"{_metric}_delta_author_count",
            f"{_metric}_mean",
            f"{_metric}_std_across_authors",
            f"{_metric}_author_count",
        ]
    )


def author_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_number(value: str | None) -> float | None:
    if value is None or value.strip().upper() in {"", "NA", "NAN"}:
        return None
    return float(value)


def read_rwku_summary(path: Path) -> dict[str, float | None]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Empty RWKU summary: {path}")
    missing = set(CSV_METRICS) - set(row)
    if missing:
        raise ValueError(f"{path} is missing metrics: {sorted(missing)}")
    return {name: parse_number(row[column]) for column, name in CSV_METRICS.items()}


def find_baselines(outputs_root: Path) -> dict[tuple[str, str], tuple[str, dict[str, float | None]]]:
    baselines = {}
    for path in sorted(outputs_root.glob("rwku-baseline-*/eval_rwku/rwku_summary_table.csv")):
        match = BASELINE_RUN_RE.fullmatch(path.parents[1].name)
        if match is None:
            continue
        size = SIZE_NAMES[match.group("size").lower()]
        key = (size, author_key(match.group("author")))
        if key in baselines:
            raise ValueError(f"Duplicate RWKU baseline for model/author {key}: {path}")
        baselines[key] = (path.parents[1].name, read_rwku_summary(path))
    return baselines


def build_author_rows(
    outputs_root: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    baselines = find_baselines(outputs_root)
    rows: list[dict[str, object]] = []
    unmatched: list[Path] = []
    paths = sorted(
        [
            *outputs_root.glob(
                "unlearning-*/final_model/eval_rwku/rwku_summary_table.csv"
            ),
            *outputs_root.glob(
                "r2warmup_*/final_model/eval_rwku/rwku_summary_table.csv"
            ),
        ]
    )
    for path in paths:
        run_name = path.parents[2].name
        match = TRAINED_RUN_RE.fullmatch(run_name)
        warmup_match = WARMUP_RUN_RE.fullmatch(run_name)
        if match is None and warmup_match is None:
            continue

        if match is not None:
            size = SIZE_NAMES[match.group("size").lower()]
            author = match.group("author")
            variant = match.group("variant").lower()
            reward_function = match.group("reward_type").lower()
            rlvr_mode = "zero-RLVR" if variant == "original" else "warm-up"
        else:
            assert warmup_match is not None
            size = SIZE_NAMES[warmup_match.group("size").lower()]
            author = warmup_match.group("author")
            reward_function = "r2-warmup"
            rlvr_mode = "warm-up"

        baseline_entry = baselines.get((size, author_key(author)))
        if baseline_entry is None:
            unmatched.append(path)
            continue

        baseline_name, baseline = baseline_entry
        trained = read_rwku_summary(path)
        row: dict[str, object] = {
            "author": author,
            "model_name_or_path": f"Qwen/Qwen2.5-{size}-Instruct",
            "model_size": size,
            "reward_function": reward_function,
            "rlvr_mode": rlvr_mode,
            "run_name": run_name,
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


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate_author_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model_name_or_path"]),
            str(row["model_size"]),
            str(row["reward_function"]),
            str(row["rlvr_mode"]),
        )
        groups[key].append(row)

    output_rows = []
    for (model, size, reward, mode), group in sorted(groups.items()):
        output: dict[str, object] = {
            "model_name_or_path": model,
            "model_size": size,
            "reward_function": reward,
            "rlvr_mode": mode,
            "author_count": len({str(row["author"]) for row in group}),
        }
        for metric in SPLIT_AGGREGATES:
            values = [
                float(row[f"{metric}_delta"])
                for row in group
                if row.get(f"{metric}_delta") is not None
            ]
            output[f"{metric}_delta_mean"] = mean(values)
            output[f"{metric}_delta_std_across_authors"] = sample_std(values)
            output[f"{metric}_author_count"] = len(values)
        for metric in DELTA_METRICS:
            values = [
                float(row[f"{metric}_delta"])
                for row in group
                if row.get(f"{metric}_delta") is not None
            ]
            output[f"{metric}_delta_mean"] = mean(values)
            output[f"{metric}_delta_std_across_authors"] = sample_std(values)
            output[f"{metric}_author_count"] = len(values)
        for metric in UTILITY_METRICS:
            delta_values = [
                float(row[f"{metric}_delta"])
                for row in group
                if row.get(f"{metric}_delta") is not None
            ]
            output[f"{metric}_delta_mean"] = mean(delta_values)
            output[f"{metric}_delta_std_across_authors"] = sample_std(delta_values)
            output[f"{metric}_delta_author_count"] = len(delta_values)
            values = [
                float(row[metric]) for row in group if row.get(metric) is not None
            ]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_std_across_authors"] = sample_std(values)
            output[f"{metric}_author_count"] = len(values)
        output_rows.append(output)
    return output_rows


def csv_value(value: object) -> object:
    return "" if value is None else value


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: csv_value(row.get(column)) for column in columns} for row in rows
        )


def format_mean_std(mean_value: object, std_value: object) -> str:
    if mean_value is None:
        return "NA"
    text = f"{float(mean_value):.3f}"
    if std_value is not None:
        text += f" ± {float(std_value):.3f}"
    return text


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "model_size",
        "reward_function",
        "rlvr_mode",
        "author_count",
        METRIC_LABELS["forget"],
        METRIC_LABELS["neighbor"],
        *[METRIC_LABELS[metric] for metric in ["mia_fm", "mia_rm", *UTILITY_METRICS]],
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [
            str(row["model_size"]),
            str(row["reward_function"]),
            str(row["rlvr_mode"]),
            str(row["author_count"]),
        ]
        for metric in SPLIT_AGGREGATES:
            cells.append(
                format_mean_std(
                    row.get(f"{metric}_delta_mean"),
                    row.get(f"{metric}_delta_std_across_authors"),
                )
            )
        for metric in ["mia_fm", "mia_rm"]:
            cells.append(
                format_mean_std(
                    row.get(f"{metric}_delta_mean"),
                    row.get(f"{metric}_delta_std_across_authors"),
                )
            )
        for metric in UTILITY_METRICS:
            cells.append(
                format_mean_std(
                    row.get(f"{metric}_delta_mean"),
                    row.get(f"{metric}_delta_std_across_authors"),
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--author-output-csv", type=Path, default=DEFAULT_AUTHOR_CSV)
    parser.add_argument(
        "--require-complete-baselines",
        action="store_true",
        help="Fail if any trained RWKU summary lacks its matching model/author baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.output_csv.with_suffix(".md")
    author_rows, unmatched = build_author_rows(args.outputs_root)
    if unmatched and args.require_complete_baselines:
        examples = "\n".join(f"  {path}" for path in unmatched[:10])
        raise SystemExit(
            f"{len(unmatched)} trained RWKU result(s) lack a matching baseline:\n{examples}"
        )

    final_rows = aggregate_author_rows(author_rows)
    write_csv(args.author_output_csv, AUTHOR_COLUMNS, author_rows)
    write_csv(args.output_csv, FINAL_COLUMNS, final_rows)
    write_markdown(output_md, final_rows)
    print(f"matched {len(author_rows)} trained result(s) to baselines")
    print(f"unmatched trained results: {len(unmatched)}")
    print(f"wrote author-level deltas: {args.author_output_csv}")
    print(f"wrote final table: {args.output_csv}")
    print(f"wrote final markdown: {output_md}")


if __name__ == "__main__":
    main()
