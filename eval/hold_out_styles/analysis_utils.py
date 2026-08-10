from __future__ import annotations

import asyncio

import pandas as pd

from eval.hold_out_styles.llm_completion_classification import (
    AsyncCompletionClassificationTool,
    llm_judge_metrics,
)


def add_llm_judge_metrics(
    df: pd.DataFrame,
    *,
    concept: str,
    judge_model: str = "gpt-5.4-nano",
    judge_reasoning_effort: str = "low",
    max_concurrent_requests: int = 4,
    index_column: str = "index",
) -> pd.DataFrame:
    tool = AsyncCompletionClassificationTool(
        target_concept=concept,
        model_name=judge_model,
        reasoning_effort=judge_reasoning_effort,
        max_concurrent_request=max_concurrent_requests,
    )

    df = df.copy()
    df[index_column] = range(len(df))
    prompt_groups = (
        df.groupby("prompt", sort=False, as_index=False)
        .agg({"completion": list, index_column: list})
        .to_dict(orient="records")
    )
    llm_metrics = asyncio.run(tool.classify_prompt_groups_completitions(prompt_groups))

    expanded_metrics = []
    for group in llm_metrics:
        for row_idx, metrics in zip(group[index_column], group["metrics"], strict=True):
            expanded_metrics.append({index_column: row_idx} | metrics)

    metrics_df = pd.DataFrame(expanded_metrics)
    return df.merge(metrics_df, on=index_column, how="left")


def aggregate_metric_summary(
    df: pd.DataFrame,
    *,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    present_metrics = [metric for metric in llm_judge_metrics if metric in df.columns]
    summary = df[present_metrics].agg(["mean"]).reset_index(names="stat")
    for column, value in reversed(list((metadata or {}).items())):
        summary.insert(0, column, value)
    return summary
