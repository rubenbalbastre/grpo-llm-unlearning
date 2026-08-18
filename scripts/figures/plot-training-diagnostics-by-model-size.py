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
FIXED_Y_LIMITS = {
    "train/reward": (0.0, 1.0),
    "active_group": (0.0, 1.0),
    "train/completions/mean_length": (0.0, 256.0),
}
SIZE_FROM_MODEL_RE = re.compile(r"Qwen2\.5-(0\.5B|1\.5B|3B|7B)-Instruct", re.I)
SIZE_FROM_RUN_RE = re.compile(r"qwen-qwen2-5-(0-5b?|1-5b?|3b?|7b?)(?:-|_|$)", re.I)
REWARD_FROM_RUN_RE = re.compile(r"-(r\d+)$", re.I)
RUN_SIZE_NAMES = {
    "0-5": "0.5B",
    "0-5b": "0.5B",
    "1-5": "1.5B",
    "1-5b": "1.5B",
    "3": "3B",
    "3b": "3B",
    "7": "7B",
    "7b": "7B",
}
INITIALIZATION_LABELS = {
    "original": "cold-start",
    "original initialization": "cold-start",
    "r2 warmed": "warm-start",
    "r2 warmup": "warm-start",
    "warmup sft initialization": "warm-start",
    "warm up sft initialization": "warm-start",
    "sft warmup initialization": "warm-start",
    "sft warm up initialization": "warm-start",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/wandb-training-diagnostics"))
    parser.add_argument(
        "--experiment-notes",
        default="lluis-vives-runs-1-1M",
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

    return model_size_from_run_name(run.name or "")


def model_size_from_run_name(run_name: object) -> str | None:
    match = SIZE_FROM_RUN_RE.search(str(run_name))
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


def initialization_variant(run) -> str:
    config = run.config or {}
    hydra = config.get("hydra", {}) if isinstance(config.get("hydra"), dict) else {}
    variant = (
        nested(hydra, "experiment", "initialization")
        or nested(hydra, "experiment", "initialization_variant")
        or nested(config, "experiment", "initialization")
        or nested(config, "experiment", "initialization_variant")
    )
    if variant:
        return display_initialization_label(str(variant))

    return initialization_from_run_name(run.name or "")


def initialization_from_run_name(run_name: object) -> str:
    run_name = str(run_name).lower()
    if re.search(r"(?:^|-)original(?:-|$)", run_name):
        return "cold-start"
    if (
        re.search(r"(?:^|-)r\d+-warmed(?:-|$)", run_name)
        or re.search(r"(?:^|-)warmed(?:-|$)", run_name)
        or "warmup" in run_name
    ):
        return "warm-start"
    return ""


def display_initialization_label(value: object) -> str:
    label = str(value).strip()
    normalized = re.sub(r"[\s_]+", " ", label.lower().replace("-", " "))
    return INITIALIZATION_LABELS.get(normalized, label)


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
    model_name_match = SIZE_FROM_MODEL_RE.search(run_model_name)
    if model_name_match and run_model_name != MODEL_NAMES[size]:
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
                        "initialization": initialization_variant(run),
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


def metric_y_limits(size_df, metric: str) -> tuple[float, float] | None:
    if metric in FIXED_Y_LIMITS:
        return FIXED_Y_LIMITS[metric]

    metric_df = size_df[size_df["metric"] == metric]
    if metric_df.empty:
        return None

    if metric == "train/kl":
        group_columns = [
            column
            for column in ("initialization", "reward_type")
            if column in metric_df.columns
        ]
        grouped_metric = (
            metric_df.groupby(group_columns)
            if group_columns
            else [(None, metric_df)]
        )
        lower_values = []
        upper_values = []
        for _, reward_df in grouped_metric:
            reward_df = reward_df.sort_values("train_global_step")
            mean_series = reward_df["mean"].rolling(
                window=KL_SMOOTHING_WINDOW,
                center=True,
                min_periods=1,
            ).median()
            std_series = reward_df["std"].fillna(0.0).rolling(
                window=KL_SMOOTHING_WINDOW,
                center=True,
                min_periods=1,
            ).median()
            lower_values.append((mean_series - std_series).min())
            upper_values.append((mean_series + std_series).max())
        lower = min(lower_values)
        upper = max(upper_values)
    else:
        mean = metric_df["mean"]
        std = metric_df["std"].fillna(0.0)
        lower = (mean - std).min()
        upper = (mean + std).max()

    if not is_number(float(lower)) or not is_number(float(upper)):
        return None

    lower = max(0.0, float(lower))
    upper = float(upper)
    if upper <= lower:
        padding = 0.05 if upper == 0.0 else abs(upper) * 0.05
        return max(0.0, lower - padding), upper + padding

    padding = (upper - lower) * 0.05
    return max(0.0, lower - padding), upper + padding


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
    metric_items = list(METRICS.items())
    metric_grid_rows = 2
    metric_grid_cols = 2

    for size in MODEL_SIZES:
        size_df = grouped[grouped["model_size"] == size]
        y_limits = {
            metric: metric_y_limits(size_df, metric)
            for metric in METRICS
        }
        present_initializations = {
            value
            for value in size_df["initialization"].dropna().unique()
            if value
        }
        initialization_blocks = [
            initialization
            for initialization in ["cold-start", "warm-start"]
            if initialization in present_initializations
        ]
        initialization_blocks.extend(
            sorted(present_initializations - set(initialization_blocks))
        )
        if not initialization_blocks:
            initialization_blocks = [""]

        nrows = metric_grid_rows * len(initialization_blocks)
        fig_height = 2.55 * nrows
        fig, axes = plt.subplots(
            nrows,
            metric_grid_cols,
            figsize=(7.0, fig_height),
            sharex=True,
            squeeze=False,
        )
        handles = {}

        for block_index, initialization in enumerate(initialization_blocks):
            plot_df = (
                size_df[size_df["initialization"] == initialization]
                if initialization
                else size_df
            )
            row_offset = block_index * metric_grid_rows

            for metric_index, (metric, (_, ylabel)) in enumerate(metric_items):
                row = row_offset + metric_index // metric_grid_cols
                col = metric_index % metric_grid_cols
                ax = axes[row, col]
                metric_df = plot_df[plot_df["metric"] == metric]
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
                    ax.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=colors[reward],
                        alpha=0.11,
                    )
                    handles.setdefault(reward, line)

                ax.set_ylabel(ylabel)
                ax.grid(True, color="#b0b0b0")
                if y_limits[metric] is not None:
                    ax.set_ylim(*y_limits[metric])

            if initialization:
                axes[row_offset, 0].set_title(
                    initialization,
                    loc="left",
                    fontweight="bold",
                    pad=14,
                )

        for ax in axes[-1, :]:
            ax.set_xlabel("Optimization step")

        if handles:
            fig.legend(
                list(handles.values()),
                list(handles.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=min(len(handles), 5),
                handlelength=2.2,
                columnspacing=1.1,
            )
        fig.suptitle(f"Qwen2.5-{size}-Instruct", y=0.995, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

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
    if "initialization" not in points.columns:
        points["initialization"] = points["run_name"].map(initialization_from_run_name)
    else:
        inferred_initialization = points["run_name"].map(initialization_from_run_name)
        points["initialization"] = points["initialization"].fillna("")
        points.loc[points["initialization"] == "", "initialization"] = inferred_initialization
    points["initialization"] = points["initialization"].fillna("").map(display_initialization_label)
    if "model_size" in points.columns and "run_name" in points.columns:
        inferred_model_size = points["run_name"].map(model_size_from_run_name)
        points["model_size"] = points["model_size"].fillna(inferred_model_size)

    grouped = (
        points.groupby(
            ["model_size", "initialization", "reward_type", "metric", "train_global_step"],
            as_index=False,
        )
        .agg(mean=("value", "mean"), std=("value", "std"), n=("run_id", "nunique"))
        .sort_values(["model_size", "initialization", "reward_type", "metric", "train_global_step"])
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
