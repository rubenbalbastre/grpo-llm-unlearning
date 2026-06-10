#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from dotenv import dotenv_values


NULL_MILESTONE_NUM_TOKENS = int(6e6)
ENV_FILE = ".env"
JOB_TYPE = "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a matrix of line plots for W&B rwku/* metrics from evaluation runs, "
            "grouped by Hydra training mode and checkpoint token milestone."
        )
    )
    parser.add_argument("--project", default=None, help="W&B project. Defaults to WANDB_PROJECT from .env.")
    parser.add_argument("--output-dir", default="outputs/wandb-rwku-plots", help="Directory for PNG/CSV outputs.")
    parser.add_argument("--forget-concept", default="Rihanna", help="Filter hydra.experiment.forget_concept.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Filter hydra.model.name. Use '' to disable.",
    )
    parser.add_argument("--metric-prefix", default="rwku/", help="Metric prefix to plot.")
    parser.add_argument("--ncols", type=int, default=3, help="Number of columns in the output plot matrix.")
    return parser.parse_args()


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def collect_eval_rows(
    *,
    entity: str | None,
    project: str,
    metric_prefix: str,
    forget_concept: str,
    model_name: str,
) -> list[dict[str, object]]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    rows: list[dict[str, object]] = []

    for run in api.runs(path):
        run_job_type = (
            getattr(run, "job_type", None)
            or getattr(run, "jobType", None)
            or getattr(run, "_attrs", {}).get("jobType")
        )
        try:
            run_forget_concept = run.config.get("hydra").get("experiment").get("forget_concept")
            run_model_name = run.config.get("hydra").get("model").get("name")
            training_mode = run.config.get("hydra").get("training").get("mode")
            milestone = run.config.get("checkpoint").get("milestone_num_tokens")
        except AttributeError:
            run_forget_concept = None
            run_model_name = None
            training_mode = None
            milestone = None

        if run_forget_concept and run_model_name and training_mode:
            if run_job_type and run_job_type != JOB_TYPE:
                continue

            summary = dict(run.summary or {})
            if run_forget_concept != forget_concept:
                continue

            if model_name and run_model_name != model_name:
                continue

            if milestone is None:
                milestone = NULL_MILESTONE_NUM_TOKENS

            try:
                milestone = int(milestone)
            except (TypeError, ValueError):
                continue

            if training_mode is None:
                training_mode = "unknown"

            generator_mode = None if training_mode == "standard" else run.config.get("hydra").get("data_generator").get("data_proposer").get("mode")
            for metric_name, metric_value in summary.items():
                if not metric_name.startswith(metric_prefix) or not is_number(metric_value):
                    continue
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "training_mode": f"{training_mode}-{generator_mode}" if generator_mode else training_mode,
                        "forget_concept": str(run_forget_concept),
                        "model_name": str(run_model_name),
                        "checkpoint_milestone_num_tokens": milestone,
                        "metric": metric_name,
                        "value": float(metric_value),
                    }
                )

    return rows


def aggregate_metric_rows(rows: list[dict[str, object]]) -> object:
    import pandas as pd

    df = pd.DataFrame(rows)
    return (
        df.groupby(["metric", "training_mode", "checkpoint_milestone_num_tokens"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .sort_values(["metric", "training_mode", "checkpoint_milestone_num_tokens"])
    )


def plot_metric_matrix(grouped: object, output_dir: Path, ncols: int, model_name: str, forget_concept: str) -> Path:
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
            mode_df = mode_df.sort_values("checkpoint_milestone_num_tokens")
            xs = mode_df["checkpoint_milestone_num_tokens"].tolist()
            means = mode_df["mean"]
            color = mode_colors[training_mode]
            (line,) = ax.plot(
                xs,
                means.tolist(),
                marker="o",
                color=color,
                label=f"{training_mode} mean",
            )
            std = mode_df["std"].fillna(0.0)
            ax.fill_between(
                xs,
                (means - std).tolist(),
                (means + std).tolist(),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
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
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(legend_labels), max(1, ncols)),
            frameon=True,
        )
    fig.suptitle(f"Model: {model_name}, Forget Concept: {forget_concept}, RWKU Evaluation Metrics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = output_dir / "rwku_metrics_matrix.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    env = dotenv_values(ENV_FILE)
    project = args.project or env.get("WANDB_PROJECT")
    entity = env.get("WANDB_ENTITY")
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
        model_name=args.model_name,
    )
    if not rows:
        raise SystemExit(
            "No matching W&B eval rows found. Check project, entity, "
            f"forget_concept={args.forget_concept!r}, model_name={args.model_name!r}, "
            "and checkpoint metadata."
        )

    import pandas as pd

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(output_dir / "rwku_eval_points.csv", index=False)
    grouped = aggregate_metric_rows(all_rows.to_dict("records"))
    grouped.to_csv(output_dir / "rwku_eval_metric_summary.csv", index=False)
    plot_path = plot_metric_matrix(grouped, output_dir, args.ncols, args.model_name, args.forget_concept)

    print(f"Wrote metric matrix plot to {plot_path}")
    print(f"Wrote raw point table to {output_dir / 'rwku_eval_points.csv'}")
    print(f"Wrote summary table to {output_dir / 'rwku_eval_metric_summary.csv'}")


if __name__ == "__main__":
    main()
