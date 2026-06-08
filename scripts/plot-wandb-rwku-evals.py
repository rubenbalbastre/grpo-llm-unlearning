#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a matrix of line plots for W&B rwku/* metrics from evaluation runs, "
            "grouped by Hydra training mode and checkpoint token milestone."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Path to repository .env file.")
    parser.add_argument("--entity", default=None, help="W&B entity. Defaults to WANDB_ENTITY or API default.")
    parser.add_argument("--project", default=None, help="W&B project. Defaults to WANDB_PROJECT from .env.")
    parser.add_argument("--output-dir", default="outputs/wandb-rwku-plots", help="Directory for PNG/CSV outputs.")
    parser.add_argument("--forget-concept", default="Rihanna", help="Filter hydra.experiment.forget_concept.")
    parser.add_argument("--metric-prefix", default="rwku/", help="Metric prefix to plot.")
    parser.add_argument("--job-type", default="evaluation", help="Filter W&B run job type. Use '' to disable.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional maximum number of W&B runs to scan.")
    parser.add_argument("--ncols", type=int, default=3, help="Number of columns in the output plot matrix.")
    return parser.parse_args()


def nested_get(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_present(data: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = nested_get(data, path)
        if value is not None:
            return value
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "metric"


def collect_eval_rows(
    *,
    entity: str | None,
    project: str,
    metric_prefix: str,
    forget_concept: str,
    job_type: str,
    max_runs: int | None,
) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    rows: list[dict[str, Any]] = []
    scanned = 0

    for run in api.runs(path):
        if max_runs is not None and scanned >= max_runs:
            break
        scanned += 1

        run_job_type = (
            getattr(run, "job_type", None)
            or getattr(run, "jobType", None)
            or getattr(run, "_attrs", {}).get("jobType")
        )
        if job_type and run_job_type and run_job_type != job_type:
            continue

        config = dict(run.config or {})
        summary = dict(run.summary or {})
        hydra = config.get("hydra") or {}
        run_forget_concept = first_present(
            config,
            [
                "hydra.experiment.forget_concept",
                "experiment.forget_concept",
            ],
        )
        if run_forget_concept is None:
            run_forget_concept = nested_get(hydra, "experiment.forget_concept")
        if run_forget_concept != forget_concept:
            continue

        milestone = first_present(
            config,
            [
                "checkpoint.milestone_num_tokens",
                "evaluation.checkpoint.milestone_num_tokens",
            ],
        )
        if milestone is None:
            milestone = nested_get(config.get("checkpoint") or {}, "milestone_num_tokens")
        if milestone is None:
            continue

        try:
            milestone = int(milestone)
        except (TypeError, ValueError):
            continue

        training_mode = first_present(
            config,
            [
                "hydra.training.mode",
                "training.mode",
            ],
        )
        if training_mode is None:
            training_mode = nested_get(hydra, "training.mode")
        if training_mode is None:
            training_mode = "unknown"

        for metric_name, metric_value in summary.items():
            if not metric_name.startswith(metric_prefix) or not is_number(metric_value):
                continue
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "training_mode": str(training_mode),
                    "forget_concept": str(run_forget_concept),
                    "checkpoint_milestone_num_tokens": milestone,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    return rows


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows)
    return (
        df.groupby(["metric", "training_mode", "checkpoint_milestone_num_tokens"], as_index=False)
        .agg(mean=("value", "mean"), min=("value", "min"), max=("value", "max"), n=("value", "count"))
        .sort_values(["metric", "training_mode", "checkpoint_milestone_num_tokens"])
    )


def plot_metric_matrix(grouped: Any, output_dir: Path, ncols: int) -> Path:
    import matplotlib.pyplot as plt

    metrics = list(grouped["metric"].drop_duplicates())
    if not metrics:
        raise ValueError("No metrics available to plot.")
    ncols = max(1, ncols)
    nrows = math.ceil(len(metrics) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    modes = list(grouped["training_mode"].drop_duplicates())
    mode_colors = {
        mode: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, mode in enumerate(modes)
    }

    legend_handles = []
    legend_labels = []
    for ax, metric in zip(axes.ravel(), metrics, strict=False):
        metric_df = grouped[grouped["metric"] == metric]
        for training_mode, mode_df in metric_df.groupby("training_mode"):
            xs = mode_df["checkpoint_milestone_num_tokens"]
            color = mode_colors[training_mode]
            (line,) = ax.plot(
                xs,
                mode_df["mean"],
                marker="o",
                color=color,
                label=f"{training_mode} mean",
            )
            ax.scatter(xs, mode_df["min"], marker="v", s=35, color=color, alpha=0.65)
            ax.scatter(xs, mode_df["max"], marker="^", s=35, color=color, alpha=0.65)
            label = f"{training_mode} mean"
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(metric.replace("rwku/", ""), fontsize=10)
        ax.set_xlabel("checkpoint.milestone_num_tokens")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), max(1, ncols)),
            frameon=False,
        )
    fig.suptitle("RWKU Evaluation Metrics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    output_path = output_dir / "rwku_metrics_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    env = dotenv_values(args.env_file)
    project = args.project or env.get("WANDB_PROJECT")
    entity = args.entity or env.get("WANDB_ENTITY")
    api_key = env.get("WANDB_API_KEY")
    if not project:
        raise SystemExit("WANDB_PROJECT is required in .env or via --project.")

    if api_key:
        import wandb

        wandb.login(key=api_key, relogin=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_eval_rows(
        entity=entity,
        project=str(project),
        metric_prefix=args.metric_prefix,
        forget_concept=args.forget_concept,
        job_type=args.job_type,
        max_runs=args.max_runs,
    )
    if not rows:
        raise SystemExit(
            "No matching W&B eval rows found. Check project, entity, "
            f"forget_concept={args.forget_concept!r}, and checkpoint metadata."
        )

    import pandas as pd

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(output_dir / "rwku_eval_points.csv", index=False)
    grouped = aggregate_metric_rows(all_rows.to_dict("records"))
    grouped.to_csv(output_dir / "rwku_eval_metric_summary.csv", index=False)
    plot_path = plot_metric_matrix(grouped, output_dir, args.ncols)

    print(f"Wrote metric matrix plot to {plot_path}")
    print(f"Wrote raw point table to {output_dir / 'rwku_eval_points.csv'}")
    print(f"Wrote summary table to {output_dir / 'rwku_eval_metric_summary.csv'}")


if __name__ == "__main__":
    main()
