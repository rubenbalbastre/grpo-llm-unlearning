import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import dotenv_values


def run_evaluation(cfg: DictConfig) -> None:
    evaluation = cfg.evaluation
    if evaluation.metrics.generation != "rouge_l_recall":
        raise ValueError(
            "evaluation.metrics.generation must be 'rouge_l_recall'; "
            "no other RWKU generation metric is implemented."
        )
    if evaluation.batch_size <= 0:
        raise ValueError("evaluation.batch_size must be >= 1")
    if evaluation.mia_batch_size <= 0:
        raise ValueError("evaluation.mia_batch_size must be >= 1")
    if evaluation.utility_batch_size <= 0:
        raise ValueError("evaluation.utility_batch_size must be >= 1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from generation_eval import evaluate_forget_set, evaluate_neighbor_set
    from mia_eval import evaluate_mia
    from results import (
        aggregate_generation,
        aggregate_mia,
        aggregate_utility,
        build_summary_table_row,
        write_summary_tables,
    )
    from utility_eval import evaluate_utility_set

    output_dir = Path(evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = None
    if evaluation.subjects:
        subjects = [s.strip() for s in str(evaluation.subjects).split(",") if s.strip()]
        if any(s.lower() in {"all", "none"} for s in subjects):
            subjects = None

    env_values = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    hf_token = (env_values.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    from_pretrained_kwargs = {
        "token": hf_token or False,
        "trust_remote_code": evaluation.model.trust_remote_code,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.model_name_or_path,
        **from_pretrained_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": evaluation.model.torch_dtype,
        "device_map": "auto",
    }
    if evaluation.model.attn_implementation:
        model_kwargs["attn_implementation"] = evaluation.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        evaluation.model_name_or_path,
        **model_kwargs,
        **from_pretrained_kwargs,
    )
    model.eval()

    forget_rows = []
    if evaluation.sets.forget:
        forget_rows = evaluate_forget_set(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=evaluation.limits.max_examples,
            max_new_tokens=evaluation.generation.max_new_tokens,
            temperature=evaluation.generation.temperature,
            batch_size=evaluation.batch_size,
        )

    neighbor_rows = []
    if evaluation.sets.neighbor:
        neighbor_rows = evaluate_neighbor_set(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=evaluation.limits.max_examples,
            max_new_tokens=evaluation.generation.max_new_tokens,
            temperature=evaluation.generation.temperature,
            batch_size=evaluation.batch_size,
        )
    all_rows = forget_rows + neighbor_rows

    all_mia_rows = []
    if evaluation.sets.mia:
        all_mia_rows = evaluate_mia(
            model=model,
            tokenizer=tokenizer,
            subjects=subjects,
            max_examples=(
                evaluation.limits.max_mia_examples
                if evaluation.limits.max_mia_examples is not None
                else evaluation.limits.max_examples
            ),
            batch_size=evaluation.mia_batch_size,
            loss=evaluation.metrics.mia.loss,
            zlib=evaluation.metrics.mia.zlib,
            min_k=evaluation.metrics.mia.min_k,
            min_k_plus_plus=evaluation.metrics.mia.min_k_plus_plus,
        )

    all_utility_rows = []
    if evaluation.sets.utility:
        all_utility_rows = evaluate_utility_set(
            model=model,
            tokenizer=tokenizer,
            max_examples=(
                evaluation.limits.max_utility_examples
                if evaluation.limits.max_utility_examples is not None
                else evaluation.limits.max_examples
            ),
            max_new_tokens=evaluation.generation.max_new_tokens,
            temperature=evaluation.generation.temperature,
            batch_size=evaluation.utility_batch_size,
        )

    metrics = aggregate_generation(all_rows)
    mia_metrics = aggregate_mia(all_mia_rows)
    utility_metrics = aggregate_utility(all_utility_rows)
    model_label = evaluation.model.label or Path(evaluation.model_name_or_path).name or evaluation.model_name_or_path
    summary_table_row = build_summary_table_row(metrics, mia_metrics, utility_metrics, model_label)

    with open(output_dir / "rwku_generations.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_mia.jsonl", "w", encoding="utf-8") as f:
        for row in all_mia_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "rwku_utility.jsonl", "w", encoding="utf-8") as f:
        for row in all_utility_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics["mia"] = mia_metrics
    metrics["utility"] = utility_metrics

    with open(output_dir / "rwku_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_summary_tables(output_dir, summary_table_row)

    print(json.dumps({
        "forget": metrics["forget"],
        "neighbor_locality": metrics["neighbor_locality"],
        "mia": mia_metrics,
        "utility": utility_metrics,
        "by_split": metrics["by_split"],
        "summary_table_row": summary_table_row,
    }, indent=2))


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
