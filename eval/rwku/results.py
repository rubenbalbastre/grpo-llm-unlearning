from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from constants import FORGET_SPLITS, NEIGHBOR_SPLITS, SUMMARY_SPLIT_MAP, SUMMARY_TABLE_COLUMNS


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
        return {"by_split": []}

    return {
        "by_split": (
            df.groupby("split")["total_nll"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "total_nll"})
            .to_dict(orient="records")
        ),
        "metric": "mean total negative log-likelihood",
        "interpretation": {
            "mia_forget": "higher is better after unlearning",
            "mia_retain": "lower is better to preserve retained-member likelihood",
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
        "metric": "accuracy for multiple choice; exact match for reasoning; token F1 for factuality; weighted bi-/tri-gram entropy for fluency",
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
    mia_by_split = {
        row["split"]: row["total_nll"]
        for row in mia_metrics.get("by_split", [])
    }
    utility_by_split = {
        row["split"]: row["score"]
        for row in utility_metrics.get("by_split", [])
    }
    table_row = {
        "model": model_label,
        "mia_fm": format_table_score(mia_by_split.get("mia_forget")),
        "mia_rm": format_table_score(mia_by_split.get("mia_retain")),
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
