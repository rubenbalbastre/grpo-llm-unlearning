#!/usr/bin/env python3
"""Plot LoRA effective rank by model size and reward function."""

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
        default=Path("outputs/tables/lora_effective_rank_runs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/lora_effective_rank_by_model_size.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[data["training_initialization"] == "warm"]
    summary = (
        data.groupby(["model_size", "reward_function"])[
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
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(MODEL_SIZES))
    width = 0.19

    for index, reward in enumerate(REWARDS):
        reward_data = summary[summary["reward_function"] == reward].set_index(
            "model_size"
        )
        medians = reward_data.reindex(MODEL_SIZES)["median"].to_numpy()
        q1 = reward_data.reindex(MODEL_SIZES)["q1"].to_numpy()
        q3 = reward_data.reindex(MODEL_SIZES)["q3"].to_numpy()
        positions = x + (index - 1.5) * width
        ax.bar(
            positions,
            medians,
            width,
            color=COLORS[reward],
            label=LABELS[reward],
            yerr=np.vstack((medians - q1, q3 - medians)),
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

    ax.set_xticks(x, MODEL_SIZES)
    ax.set_xlabel("Model size")
    ax.set_ylabel("95% energy effective rank")
    ax.set_ylim(0, 64)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
