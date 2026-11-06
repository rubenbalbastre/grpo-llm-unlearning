#!/usr/bin/env python3
"""Plot two-seed LoRA 95%-energy rank by module and model size."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B"]
MODULE_GROUPS = {
    "Attention projections": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "MLP projections": ["gate_proj", "up_proj", "down_proj"],
}
COLORS = {
    "q_proj": "#0072B2",
    "k_proj": "#E69F00",
    "v_proj": "#009E73",
    "o_proj": "#CC79A7",
    "gate_proj": "#D55E00",
    "up_proj": "#56B4E9",
    "down_proj": "#000000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/lora-effective-rank-tables/lora_effective_rank_modules.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/lora-effective-rank-tables/lora_effective_rank_by_module.pdf"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[data["training_initialization"] == "warm"]

    # Average layers, seeds, and rewards before aggregating across targets.
    target_values = (
        data.groupby(
            [
                "experiment_seed",
                "target",
                "model_size",
                "reward_function",
                "module",
            ],
            as_index=False,
        )["effective_rank_95"]
        .mean()
        .groupby(
            ["target", "model_size", "reward_function", "module"],
            as_index=False,
        )["effective_rank_95"]
        .mean()
        .groupby(["target", "model_size", "module"], as_index=False)[
            "effective_rank_95"
        ]
        .mean()
    )
    summary = (
        target_values.groupby(["model_size", "module"])["effective_rank_95"]
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
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.25),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 3]},
    )

    for ax, (title, modules) in zip(axes, MODULE_GROUPS.items()):
        x = np.arange(len(MODEL_SIZES))
        for module in modules:
            module_data = summary[summary["module"] == module].set_index(
                "model_size"
            )
            medians = module_data.reindex(MODEL_SIZES)["median"].to_numpy()
            q1 = module_data.reindex(MODEL_SIZES)["q1"].to_numpy()
            q3 = module_data.reindex(MODEL_SIZES)["q3"].to_numpy()
            ax.errorbar(
                x,
                medians,
                yerr=np.vstack((medians - q1, q3 - medians)),
                color=COLORS[module],
                label=module.removesuffix("_proj"),
                marker="o",
                markersize=3.5,
                linewidth=1.3,
                capsize=2,
                elinewidth=0.8,
                capthick=0.8,
            )

        ax.set_xticks(x, MODEL_SIZES)
        ax.set_xlabel("Model size")
        ax.set_title(title)
        ax.set_ylim(20, 56)
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        ax.legend(ncol=2, frameon=False, loc="upper left")

    axes[0].set_ylabel("95%-energy rank")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
