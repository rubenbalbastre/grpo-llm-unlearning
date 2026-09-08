#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("results/figures/wandb-training-diagnostics/training_diagnostic_summary.csv")
OUTPUT = Path("results/figures/wandb-training-diagnostics/training_diagnostics_support_qwen2_5_3b.pdf")
METRICS = {
    "train/reward": "Reward",
    "active_group": "Active-group fraction",
}
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


def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    data = pd.read_csv(INPUT)
    data = data[
        (data["model_size"] == "3B")
        & data["metric"].isin(METRICS)
        & data["initialization"].isin(["cold-start", "warm-start"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    handles = {}
    for row, initialization in enumerate(["cold-start", "warm-start"]):
        block = data[data["initialization"] == initialization]
        for col, (metric, ylabel) in enumerate(METRICS.items()):
            ax = axes[row, col]
            metric_data = block[block["metric"] == metric]
            for reward, reward_data in metric_data.groupby("reward_type"):
                reward_data = reward_data.sort_values("train_global_step")
                x = reward_data["train_global_step"].to_numpy()
                mean = reward_data["mean"].to_numpy()
                std = reward_data["std"].fillna(0.0).to_numpy()
                (line,) = ax.plot(x, mean, color=COLORS[reward], label=LABELS[reward])
                ax.fill_between(x, mean - std, mean + std, color=COLORS[reward], alpha=0.11)
                handles.setdefault(reward, line)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.65)
            if col == 0:
                ax.set_title(initialization, loc="left", fontweight="bold")
            if row == 1:
                ax.set_xlabel("Optimization step")

    fig.legend(
        [handles[key] for key in sorted(handles)],
        [LABELS[key] for key in sorted(handles)],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
