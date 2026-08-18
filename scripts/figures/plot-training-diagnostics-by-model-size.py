#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values


MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B"]
MODEL_NAMES = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}
KL_SMOOTHING_WINDOW = 7
METRICS = {
    "train/reward": ("Reward", "Reward"),
    "active_group": ("Active group", "Active-group fraction"),
    "train/kl": ("KL", "KL divergence"),
    "train/completions/mean_length": (
        "Completion length",
        "Mean completion length (tokens)",
    ),
}
SIZE_FROM_MODEL_RE = re.compile(r"Qwen2\.5-(0\.5B|1\.5B|3B|7B)-Instruct", re.I)
SIZE_FROM_RUN_RE = re.compile(r"qwen-qwen2-5-(0-5b|1-5b|3b|7b)-instruct", re.I)
REWARD_FROM_RUN_RE = re.compile(r"-(r\d+)$", re.I)
RUN_SIZE_NAMES = {"0-5b": "0.5B", "1-5b": "1.5B", "3b": "3B", "7b": "7B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/wandb-training-diagnostics"))
    parser.add_argument(
        "--experiment-notes",
        default=None,
        help="Defaults to experiment.notes from config/train.yaml. Use '' to disable.",
    )
    parser.add_argument("--job-type", default="training")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def train_config_notes() -> str:
    path = Path("config/train.yaml")
    if not path.exists():
        return ""

    in_experiment = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            key, _, value = stripped.partition(":")
            in_experiment = key == "experiment" and not value.strip()
        elif in_experiment and indent == 2:
            key, _, value = stripped.partition(":")
            if key == "notes":
                return value.strip().strip("'\"")
    return ""


def nested(config: dict, *keys: str) -> object:
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def model_size(run) -> str | None:
    config = run.config or {}
    hydra = config.get("hydra", {}) if isinstance(config.get("hydra"), dict) else {}
    model_name = nested(hydra, "model", "name") or nested(config, "model", "name") or ""

    match = SIZE_FROM_MODEL_RE.search(str(model_name))
    if match:
        return match.group(1).replace("b", "B")

    match = SIZE_FROM_RUN_RE.search(run.name or "")
    if match:
        return RUN_SIZE_NAMES[match.group(1).lower()]
    return None


def configured_model_name(run) -> str:
    config = run.config or {}
    hydra = config.get("hydra", {}) if isinstance(config.get("hydra"), dict) else {}
    return str(nested(hydra, "model", "name") or nested(config, "model", "name") or "")


def reward_type(run) -> str | None:
    config = run.config or {}
    hydra = config.get("hydra", {}) if isinstance(config.get("hydra"), dict) else {}
    reward = nested(hydra, "reward", "type") or nested(config, "reward", "type")
    if reward:
        return str(reward).lower()

    match = REWARD_FROM_RUN_RE.search(run.name or "")
    return match.group(1).lower() if match else None


def experiment_notes(run) -> str:
    config = run.config or {}
    hydra = config.get("hydra", {}) if isinstance(config.get("hydra"), dict) else {}
    return str(
        nested(hydra, "experiment", "notes")
        or nested(config, "experiment", "notes")
        or ""
    )


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def history_keys() -> list[str]:
    return [
        "train/global_step",
        "train/reward",
        "train/frac_reward_zero_std",
        "train/kl",
        "train/completions/mean_length",
    ]


def collect_run_rows(run, notes: str, samples: int) -> list[dict]:
    size = model_size(run)
    reward = reward_type(run)
    if size not in MODEL_SIZES or reward is None:
        return []
    if notes and experiment_notes(run) != notes:
        return []
    run_model_name = configured_model_name(run)
    if run_model_name != MODEL_NAMES[size]:
        return []

    rows = []
    for point in run.history(keys=history_keys(), samples=samples, pandas=False):
        x = point.get("train/global_step")
        if not is_number(x):
            continue

        values = {
            "train/reward": point.get("train/reward"),
            "active_group": (
                1.0 - point["train/frac_reward_zero_std"]
                if is_number(point.get("train/frac_reward_zero_std"))
                else None
            ),
            "train/kl": point.get("train/kl"),
            "train/completions/mean_length": point.get(
                "train/completions/mean_length"
            ),
        }
        for metric, value in values.items():
            if is_number(value):
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "model_size": size,
                        "model_name": run_model_name,
                        "reward_type": reward,
                        "metric": metric,
                        "train_global_step": float(x),
                        "value": float(value),
                    }
                )
    return rows


def collect_rows(
    project: str,
    entity: str | None,
    notes: str,
    job_type: str,
    samples: int,
    workers: int,
) -> list[dict]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {}
    if job_type:
        filters["jobType"] = job_type
    if notes:
        filters["config.hydra.experiment.notes"] = notes
    filters["config.hydra.model.name"] = {"$in": list(MODEL_NAMES.values())}

    runs = list(api.runs(path, filters=filters))
    rows = []
    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(collect_run_rows, run, notes, samples): run
            for run in runs
        }
        for future in as_completed(futures):
            run = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                print(f"Skipping {run.name or run.id}: {exc}")
    return rows


def set_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figures(grouped, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = {
        reward: color
        for reward, color in zip(
            sorted(grouped["reward_type"].unique()),
            ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"],
            strict=False,
        )
    }
    paths = []

    for size in MODEL_SIZES:
        size_df = grouped[grouped["model_size"] == size]
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1), sharex=True)
        handles = {}

        for ax, (metric, (_, ylabel)) in zip(axes.ravel(), METRICS.items(), strict=True):
            metric_df = size_df[size_df["metric"] == metric]
            for reward, reward_df in metric_df.groupby("reward_type"):
                reward_df = reward_df.sort_values("train_global_step")
                x = reward_df["train_global_step"].to_numpy()
                mean_series = reward_df["mean"]
                std_series = reward_df["std"].fillna(0.0)
                if metric == "train/kl":
                    mean_series = mean_series.rolling(
                        window=KL_SMOOTHING_WINDOW,
                        center=True,
                        min_periods=1,
                    ).median()
                    std_series = std_series.rolling(
                        window=KL_SMOOTHING_WINDOW,
                        center=True,
                        min_periods=1,
                    ).median()
                mean = mean_series.to_numpy()
                std = std_series.to_numpy()
                (line,) = ax.plot(x, mean, color=colors[reward], label=reward)
                ax.fill_between(x, mean - std, mean + std, color=colors[reward], alpha=0.11)
                handles.setdefault(reward, line)

            ax.set_ylabel(ylabel)
            ax.grid(True, color="#b0b0b0")
            if metric in {"train/reward", "active_group"}:
                ax.set_ylim(-0.03, 1.03)

        for ax in axes[1, :]:
            ax.set_xlabel("Optimization step")

        if handles:
            fig.legend(
                list(handles.values()),
                list(handles.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                ncol=min(len(handles), 5),
                handlelength=2.2,
                columnspacing=1.1,
            )
        fig.suptitle(f"Qwen2.5-{size}-Instruct", y=0.985, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.89))

        stem = f"training_diagnostics_qwen2_5_{size.lower().replace('.', '_')}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        paths.extend([png_path, pdf_path])

    return paths


def main() -> None:
    args = parse_args()
    env = dotenv_values(".env")
    project = args.project or env.get("WANDB_PROJECT")
    entity = args.entity if args.entity is not None else env.get("WANDB_ENTITY")
    api_key = env.get("WANDB_API_KEY")
    notes = train_config_notes() if args.experiment_notes is None else args.experiment_notes
    if not project:
        raise SystemExit("WANDB_PROJECT is required in .env or --project.")

    if api_key:
        import wandb

        wandb.login(key=api_key, relogin=True)

    import pandas as pd

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "training_diagnostic_points.csv"
    required_columns = {"train_global_step", "model_name"}
    if points_path.exists():
        points = pd.read_csv(points_path)
        if required_columns.issubset(points.columns):
            print(f"Reusing cached W&B points from {points_path}")
        else:
            points = None
    else:
        points = None

    if points is None:
        rows = collect_rows(
            str(project), entity, notes, args.job_type, args.samples, args.workers
        )
        if not rows:
            raise SystemExit(
                f"No matching W&B rows found for experiment.notes={notes!r} "
                f"and model names {list(MODEL_NAMES.values())!r}"
            )
        points = pd.DataFrame(rows)
        points.to_csv(points_path, index=False)
        print(f"Saved W&B points to {points_path}")

    grouped = (
        points.groupby(["model_size", "reward_type", "metric", "train_global_step"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("run_id", "nunique"))
        .sort_values(["model_size", "reward_type", "metric", "train_global_step"])
    )

    grouped.to_csv(output_dir / "training_diagnostic_summary.csv", index=False)

    set_paper_style()
    for path in plot_figures(grouped, output_dir):
        print(f"Wrote {path}")
    print(f"Prefiltered W&B runs by jobType={args.job_type!r}")
    print(f"Filtered by experiment.notes={notes!r}")
    print(f"Filtered by model.name in {list(MODEL_NAMES.values())!r}")


if __name__ == "__main__":
    main()
