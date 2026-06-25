#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LEFT = Path("outputs/unlearning-4261/checkpoint-159/eval_rwku/rwku_generations.jsonl")
DEFAULT_RIGHT = Path("outputs/unlearning-4260/checkpoint-159/eval_rwku/rwku_generations.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/rwku_generation_diffs/unlearning-4261_vs_unlearning-4260_checkpoint-159")
KEY_FIELDS = ("split", "subject", "level", "type", "query", "answer")
RESULT_FIELDS = ("prediction", "rouge_l_recall")
SUMMARY_FIELDS = [
    "group",
    "rows",
    "changed_rows",
    "same_rows",
    "only_left_rows",
    "only_right_rows",
    "task_key_mismatch_rows",
    "prediction_changed_rows",
    "rouge_l_recall_changed_rows",
    "left_mean_rouge_l_recall",
    "right_mean_rouge_l_recall",
    "mean_rouge_l_recall_delta",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def task_key(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {field: row.get(field) for field in KEY_FIELDS}


def changed_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    fields = sorted(set(left) | set(right))
    return [field for field in fields if left.get(field) != right.get(field)]


def row_status(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    if left is None:
        return "only_right"
    if right is None:
        return "only_left"
    if left == right:
        return "same"
    return "changed"


def compare_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    max_rows = max(len(left_rows), len(right_rows))
    for index in range(max_rows):
        left = left_rows[index] if index < len(left_rows) else None
        right = right_rows[index] if index < len(right_rows) else None
        status = row_status(left, right)
        fields = changed_fields(left, right) if left is not None and right is not None else []
        comparisons.append(
            {
                "row_index": index + 1,
                "status": status,
                "task_key_match": left is not None and right is not None and task_key(left) == task_key(right),
                "changed_fields": fields,
                "left_key": task_key(left),
                "right_key": task_key(right),
                "left_prediction": None if left is None else left.get("prediction"),
                "right_prediction": None if right is None else right.get("prediction"),
                "left_rouge_l_recall": None if left is None else left.get("rouge_l_recall"),
                "right_rouge_l_recall": None if right is None else right.get("rouge_l_recall"),
                "left": left,
                "right": right,
            }
        )
    return comparisons


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def summary_row(group: str, comparisons: list[dict[str, Any]]) -> dict[str, str | int]:
    left_rouge_values = [
        value
        for value in (as_float(row["left_rouge_l_recall"]) for row in comparisons)
        if value is not None
    ]
    right_rouge_values = [
        value
        for value in (as_float(row["right_rouge_l_recall"]) for row in comparisons)
        if value is not None
    ]
    left_mean = mean(left_rouge_values)
    right_mean = mean(right_rouge_values)
    return {
        "group": group,
        "rows": len(comparisons),
        "changed_rows": sum(row["status"] == "changed" for row in comparisons),
        "same_rows": sum(row["status"] == "same" for row in comparisons),
        "only_left_rows": sum(row["status"] == "only_left" for row in comparisons),
        "only_right_rows": sum(row["status"] == "only_right" for row in comparisons),
        "task_key_mismatch_rows": sum(
            row["status"] == "changed" and not row["task_key_match"] for row in comparisons
        ),
        "prediction_changed_rows": sum("prediction" in row["changed_fields"] for row in comparisons),
        "rouge_l_recall_changed_rows": sum("rouge_l_recall" in row["changed_fields"] for row in comparisons),
        "left_mean_rouge_l_recall": format_float(left_mean),
        "right_mean_rouge_l_recall": format_float(right_mean),
        "mean_rouge_l_recall_delta": format_float(
            None if left_mean is None or right_mean is None else left_mean - right_mean
        ),
    }


def grouped_summaries(comparisons: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = comparisons
    for row in comparisons:
        key = row["left_key"] or row["right_key"]
        groups[f"split={key.get('split', 'unknown')}"].append(row)
        groups[f"type={key.get('type', 'unknown')}"].append(row)
        groups[f"subject={key.get('subject', 'unknown')}"].append(row)
    return [summary_row(group, rows) for group, rows in sorted(groups.items())]


def write_diff_jsonl(path: Path, comparisons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in comparisons:
            if row["status"] == "same":
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_report(comparisons: list[dict[str, Any]], diff_jsonl: Path, summary_csv: Path) -> None:
    status_counts = Counter(row["status"] for row in comparisons)
    changed_field_counts = Counter(
        field
        for row in comparisons
        for field in row["changed_fields"]
        if row["status"] == "changed"
    )
    all_summary = summary_row("all", comparisons)

    print("RWKU generation comparison")
    print(f"rows: {all_summary['rows']}")
    print(f"same_rows: {status_counts['same']}")
    print(f"changed_rows: {status_counts['changed']}")
    print(f"only_left_rows: {status_counts['only_left']}")
    print(f"only_right_rows: {status_counts['only_right']}")
    print(f"task_key_mismatch_rows: {all_summary['task_key_mismatch_rows']}")
    print(f"left_mean_rouge_l_recall: {all_summary['left_mean_rouge_l_recall']}")
    print(f"right_mean_rouge_l_recall: {all_summary['right_mean_rouge_l_recall']}")
    print(f"mean_rouge_l_recall_delta: {all_summary['mean_rouge_l_recall_delta']}")
    if changed_field_counts:
        print("changed fields:")
        for field, count in changed_field_counts.most_common():
            print(f"  {field}: {count}")
    print(f"wrote diff JSONL: {diff_jsonl}")
    print(f"wrote summary CSV: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two RWKU generation JSONL files row by row.")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--diff-jsonl", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_rows = load_jsonl(args.left)
    right_rows = load_jsonl(args.right)
    comparisons = compare_rows(left_rows, right_rows)

    diff_jsonl = args.diff_jsonl or args.output_dir / "rwku_generation_differences.jsonl"
    summary_csv = args.summary_csv or args.output_dir / "rwku_generation_difference_summary.csv"
    write_diff_jsonl(diff_jsonl, comparisons)
    write_summary_csv(summary_csv, grouped_summaries(comparisons))
    print_report(comparisons, diff_jsonl, summary_csv)


if __name__ == "__main__":
    main()
