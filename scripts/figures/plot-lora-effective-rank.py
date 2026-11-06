#!/usr/bin/env python3
"""Plot two-seed LoRA 95%-energy rank by model size and reward function."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B"]
REWARDS = ["r0", "r1", "r2", "r4"]
COLORS = {
    "r0": "#0072B2",
    "r1": "#D55E00",
    "r2": "#009E73",
    "r4": "#CC79A7",
}
LABELS = {
    "r0": "R0-Lex",
    "r1": "R1-AntiRefusal",
    "r2": "R2-Rubric",
    "r4": "R4-Refusal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/lora-effective-rank-tables/lora_effective_rank_runs.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/lora-effective-rank-tables/"
            "lora_effective_rank_by_model_size.pdf"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[data["training_initialization"] == "warm"]
    target_values = (
        data.groupby(
            [
                "experiment_seed",
                "target",
                "model_size",
                "reward_function",
            ],
            as_index=False,
        )["effective_rank_95_mean"]
        .mean()
        .groupby(
            ["target", "model_size", "reward_function"], as_index=False
        )["effective_rank_95_mean"]
        .mean()
    )
    summary = (
        target_values.groupby(["model_size", "reward_function"])[
            "effective_rank_95_mean"
        ]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    x = np.arange(len(MODEL_SIZES))

    for reward in REWARDS:
        reward_data = summary[summary["reward_function"] == reward].set_index(
            "model_size"
        )
        medians = reward_data.reindex(MODEL_SIZES)["median"].to_numpy()
        q1 = reward_data.reindex(MODEL_SIZES)["q1"].to_numpy()
        q3 = reward_data.reindex(MODEL_SIZES)["q3"].to_numpy()
        ax.errorbar(
            x,
            medians,
            yerr=np.vstack((medians - q1, q3 - medians)),
            color=COLORS[reward],
            label=LABELS[reward],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2.5,
            elinewidth=0.9,
            capthick=0.9,
        )

    ax.set_xticks(x, MODEL_SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("95%-energy rank")
    ax.set_ylim(34, 50)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
