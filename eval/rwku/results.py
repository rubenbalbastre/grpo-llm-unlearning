from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from constants import FORGET_SPLITS, NEIGHBOR_SPLITS, SUMMARY_SPLIT_MAP, SUMMARY_TABLE_COLUMNS

MIA_METRIC_DESCRIPTIONS = {
    "loss": "subject-summed mean token negative log-likelihood",
    "loss_per_example": "mean token negative log-likelihood averaged over examples",
    "total_loss": "subject-summed total sequence negative log-likelihood",
    "total_loss_per_example": "total sequence negative log-likelihood averaged over examples",
    "zlib": "loss normalized by zlib-compressed text length",
    "min_k": "Min-K% token likelihood score",
    "min_k_plus_plus": "Min-K%++ token likelihood score",
}
def aggregate_generation(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)

    result = {}

    if df.empty:
        return {
            "overall": {
                "n": 0,
                "rouge_l_recall": None,
            },
            "forget": {
                "n": 0,
                "rouge_l_recall": None,
                "interpretation": "lower is better after unlearning",
            },
            "neighbor_locality": {
                "n": 0,
                "rouge_l_recall": None,
                "interpretation": "higher is better; nearby retained knowledge should survive",
            },
            "by_split": [],
            "by_split_and_type": [],
            "by_subject": [],
        }

    result["overall"] = {
        "n": int(len(df)),
        "rouge_l_recall": float(df["rouge_l_recall"].mean()),
    }

    forget_df = df[df["split"].isin(FORGET_SPLITS)]
    neighbor_df = df[df["split"].isin(NEIGHBOR_SPLITS)]

    result["forget"] = {
        "n": int(len(forget_df)),
        "rouge_l_recall": float(forget_df["rouge_l_recall"].mean()) if not forget_df.empty else None,
        "interpretation": "lower is better after unlearning",
    }

    result["neighbor_locality"] = {
        "n": int(len(neighbor_df)),
        "rouge_l_recall": float(neighbor_df["rouge_l_recall"].mean()) if not neighbor_df.empty else None,
        "interpretation": "higher is better; nearby retained knowledge should survive",
    }

    result["by_split"] = (
        df.groupby("split")["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    result["by_split_and_type"] = (
        df.groupby(["split", "type"])["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    result["by_subject"] = (
        df.groupby(["split", "subject"])["rouge_l_recall"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "rouge_l_recall"})
        .to_dict(orient="records")
    )

    return result


def aggregate_mia(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "by_split": [],
            "metrics": [],
            "metric_descriptions": MIA_METRIC_DESCRIPTIONS,
            "summary_metric": "loss",
        }

    metrics = [
        metric
        for metric in ["loss", "total_loss", "zlib", "min_k", "min_k_plus_plus"]
        if metric in df.columns
    ]
    by_split = df.groupby("split").size().reset_index(name="n")
    if "subject" in df.columns:
        by_split = by_split.merge(
            df.groupby("split")["subject"]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={"subject": "subjects"}),
            on="split",
            how="left",
        )
    else:
        by_split["subjects"] = None
    metrics_set = set(metrics)
    for metric in metrics:
        if "subject" in df.columns and df["subject"].notna().any():
            subject_scores = (
                df.groupby(["split", "subject"], dropna=False)[metric]
                .sum()
                .reset_index()
            )
            metric_df = subject_scores.groupby("split")[metric].mean().reset_index()
        else:
            metric_df = df.groupby("split")[metric].sum().reset_index()
        by_split = by_split.merge(metric_df, on="split", how="left")
        per_example_df = (
            df.groupby("split")[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: f"{metric}_per_example"})
        )
        by_split = by_split.merge(per_example_df, on="split", how="left")

    return {
        "by_split": by_split.to_dict(orient="records"),
        "metrics": metrics,
        "metric_descriptions": {
            metric: description
            for metric, description in MIA_METRIC_DESCRIPTIONS.items()
            if metric in metrics_set or metric.removesuffix("_per_example") in metrics_set
        },
        "summary_metric": "loss",
        "aggregation": (
            "Per-example loss is mean token negative log-likelihood. "
            "Split-level MIA loss sums that value within each subject and averages the subject sums, "
            "matching RWKU table scale for single-target or all-target evaluation."
        ),
        "interpretation": {
            "mia_forget": "higher loss is better after unlearning",
            "mia_retain": "lower loss is better to preserve retained-member likelihood",
        },
    }


def aggregate_utility(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"by_split": []}

    return {
        "by_split": (
            df.groupby("split")["score"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "score"})
            .to_dict(orient="records")
        ),
        "metric": "official RWKU utility metrics: MMLU next-token accuracy; BBH exact match; TruthfulQA MC1; TriviaQA max-alias token F1; weighted bi-/tri-gram entropy",
        "interpretation": "higher is better",
    }


def format_table_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    formatted = f"{value:.3f}"
    if formatted.startswith("0."):
        return formatted[1:]
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted


def build_summary_table_row(
    metrics: Dict,
    mia_metrics: Dict,
    utility_metrics: Dict,
    model_label: str,
) -> Dict[str, str]:
    by_split = {
        row["split"]: row["rouge_l_recall"]
        for row in metrics.get("by_split", [])
    }
    mia_by_split = {row["split"]: row for row in mia_metrics.get("by_split", [])}
    utility_by_split = {
        row["split"]: row["score"]
        for row in utility_metrics.get("by_split", [])
    }
    table_row = {
        "model": model_label,
        "mia_fm": format_table_score(mia_by_split.get("mia_forget", {}).get("loss")),
        "mia_rm": format_table_score(mia_by_split.get("mia_retain", {}).get("loss")),
        "utility_ga": format_table_score(utility_by_split.get("utility_general")),
        "utility_ra": format_table_score(utility_by_split.get("utility_reason")),
        "utility_tru": format_table_score(utility_by_split.get("utility_truthfulness")),
        "utility_fac": format_table_score(utility_by_split.get("utility_factuality")),
        "utility_flu": format_table_score(utility_by_split.get("utility_fluency")),
    }

    for column, split_name in SUMMARY_SPLIT_MAP.items():
        table_row[column] = format_table_score(by_split.get(split_name))

    return table_row


def write_summary_tables(output_dir: Path, table_row: Dict[str, str]) -> None:
    headers = [label for _, label in SUMMARY_TABLE_COLUMNS]
    values = [table_row[key] for key, _ in SUMMARY_TABLE_COLUMNS]

    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
        "",
    ]
    (output_dir / "rwku_summary_table.md").write_text("\n".join(markdown), encoding="utf-8")

    csv_lines = [
        ",".join(headers),
        ",".join(values),
        "",
    ]
    (output_dir / "rwku_summary_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")
