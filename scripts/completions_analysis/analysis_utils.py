from __future__ import annotations

import asyncio

import pandas as pd

from scripts.completions_analysis.llm_completion_classification import (
    AsyncCompletionClassificationTool,
    llm_judge_metrics,
)
from src.reward.components.simple_match import (
    binary_forgetting_reward as compute_forgetting_reward_per_completion,
    build_entity_matchers,
)
from src.reward.components.fuzzy_match import (
    build_fuzzy_entities_for_concept,
    compute_fuzzy_reward_per_completion as compute_forgetting_fuzzy_reward_per_completion,
)


def add_local_reward_metrics(
    df: pd.DataFrame,
    concept: str,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    if not overwrite and {"forgetting_reward", "forgetting_fuzzy_reward"}.issubset(df.columns):
        return df

    target_matchers = build_entity_matchers(concept)
    fuzzy_entities = build_fuzzy_entities_for_concept(concept)

    if overwrite or "forgetting_reward" not in df.columns:
        exact_results = df["completion"].map(
            lambda completion: compute_forgetting_reward_per_completion(
                str(completion),
                target_matchers,
            )
        )
        df["forgetting_reward"] = exact_results.map(lambda result: result[0])
        df["matched_entities"] = exact_results.map(lambda result: result[1])

    if overwrite or "forgetting_fuzzy_reward" not in df.columns:
        df["forgetting_fuzzy_reward"] = df["completion"].map(
            lambda completion: compute_forgetting_fuzzy_reward_per_completion(
                str(completion),
                fuzzy_entities,
            )
        )

    return df


def add_llm_judge_metrics(
    df: pd.DataFrame,
    *,
    concept: str,
    judge_model: str = "gpt-5.4-nano",
    max_concurrent_requests: int = 4,
    index_column: str = "index",
) -> pd.DataFrame:
    tool = AsyncCompletionClassificationTool(
        target_concept=concept,
        model_name=judge_model,
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
    metrics = ["forgetting_reward", "forgetting_fuzzy_reward"] + llm_judge_metrics
    present_metrics = [metric for metric in metrics if metric in df.columns]
    summary = df[present_metrics].agg(["mean", "std"]).reset_index(names="stat")
    for column, value in reversed(list((metadata or {}).items())):
        summary.insert(0, column, value)
    return summary
