#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_POINTS = Path(
    "results/figures/wandb-training-diagnostics/training_diagnostic_points.csv"
)
DEFAULT_OUTPUT = Path(
    "results/figures/wandb-training-diagnostics/training_diagnostic_runtimes.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sum W&B execution time for runs used by the training diagnostics figures."
        )
    )
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip runs without W&B summary['_runtime'] instead of failing.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch W&B runtimes even if --output-csv already exists.",
    )
    return parser.parse_args()


def read_figure_runs(points_csv: Path) -> list[dict[str, str]]:
    runs = {}
    with points_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"run_id", "run_name", "model_size", "initialization", "reward_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{points_csv} is missing columns: {sorted(missing)}")

        for row in reader:
            run_id = row["run_id"]
            runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "run_name": row["run_name"],
                    "model_size": row["model_size"],
                    "initialization": row["initialization"],
                    "reward_type": row["reward_type"],
                },
            )
    return sorted(runs.values(), key=lambda row: row["run_name"])


def run_runtime_seconds(run) -> float | None:
    runtime = run.summary.get("_runtime")
    return float(runtime) if runtime is not None else None


def read_runtime_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["runtime_seconds"] = float(row["runtime_seconds"])
            rows.append(row)
    return rows


def write_runtime_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "run_name",
                "model_size",
                "initialization",
                "reward_type",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def fetch_runtime_rows(args: argparse.Namespace, runs: list[dict[str, str]]) -> list[dict]:
    env = dotenv_values(".env")
    project = args.project or env.get("WANDB_PROJECT")
    entity = args.entity if args.entity is not None else env.get("WANDB_ENTITY")
    api_key = env.get("WANDB_API_KEY")
    if not project:
        raise SystemExit("WANDB_PROJECT is required in .env or --project.")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is required in .env.")

    import wandb

    overrides = {"project": project}
    if entity:
        overrides["entity"] = entity
    api = wandb.Api(api_key=api_key, overrides=overrides)

    rows = []
    missing = []
    for row in runs:
        run = api.run(row["run_id"])
        runtime = run_runtime_seconds(run)
        if runtime is None:
            missing.append(row["run_id"])
            if args.allow_missing:
                continue
            continue
        rows.append({**row, "runtime_seconds": runtime})

    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing summary['_runtime'] for runs: "
            f"{', '.join(missing[:10])}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    return rows


def print_summary(rows: list[dict], output_csv: Path) -> None:
    totals = defaultdict(float)
    for row in rows:
        totals["all"] += row["runtime_seconds"]
        totals[("model_size", row["model_size"])] += row["runtime_seconds"]
        totals[("initialization", row["initialization"])] += row["runtime_seconds"]
        totals[("reward_type", row["reward_type"])] += row["runtime_seconds"]

    print(f"Runs counted: {len(rows)}")
    print(f"Total seconds: {totals['all']:.0f}")
    print(f"Total time: {format_duration(totals['all'])}")
    print(f"Total days: {totals['all'] / 86400:.2f}")
    print(f"Per-run runtimes: {output_csv}")

    for key_type in ("model_size", "initialization", "reward_type"):
        print(f"\nBy {key_type}:")
        grouped_items = []
        for total_key, seconds in totals.items():
            if not isinstance(total_key, tuple):
                continue
            group_type, key = total_key
            if group_type == key_type:
                grouped_items.append((key, seconds))
        for key, seconds in sorted(grouped_items):
            print(f"  {key}: {format_duration(seconds)} ({seconds / 3600:.2f} h)")


def main() -> None:
    args = parse_args()
    figure_runs = read_figure_runs(args.points_csv)

    if args.output_csv.exists() and not args.refresh:
        rows = read_runtime_rows(args.output_csv)
        expected_run_ids = {row["run_id"] for row in figure_runs}
        cached_run_ids = {row["run_id"] for row in rows}
        if cached_run_ids == expected_run_ids:
            print(f"Reusing cached W&B runtimes from {args.output_csv}")
            print_summary(rows, args.output_csv)
            return

    rows = fetch_runtime_rows(args, figure_runs)
    write_runtime_rows(args.output_csv, rows)
    print_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
