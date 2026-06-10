import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import dotenv_values


def log_wandb_results(
    evaluation: DictConfig,
    env_values: dict,
    output_dir: Path,
    metrics: dict,
    mia_metrics: dict,
    utility_metrics: dict,
) -> None:
    if not evaluation.wandb.enabled:
        return

    api_key = (env_values.get("WANDB_API_KEY") or "").strip()
    if not api_key:
        print("Skipping W&B evaluation logging: WANDB_API_KEY is not set in .env.")
        return
    project = (env_values.get("WANDB_PROJECT") or "").strip()
    if not project:
        raise ValueError(
            "evaluation.wandb.enabled=true requires WANDB_PROJECT in the repository .env file."
        )

    import wandb

    wandb.login(key=api_key, relogin=True)
    model_dir = Path(evaluation.model_name_or_path)
    training_run_info = None
    if evaluation.wandb.link_to_training_run:
        run_info_path = model_dir / "wandb_run.json"
        if run_info_path.exists():
            training_run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
            if training_run_info["project"] != project:
                raise ValueError(
                    "WANDB_PROJECT in .env must match the project recorded for the "
                    "linked training run: "
                    f"{training_run_info['project']}."
                )
        else:
            raise ValueError(
                "evaluation.wandb.link_to_training_run=true requires "
                "wandb_run.json in evaluation.model_name_or_path. Set it to false "
                "to create a separate evaluation run."
            )

    generation_by_split = {
        row["split"]: row["rouge_l_recall"]
        for row in metrics["by_split"]
    }
    mia_by_split = {
        row["split"]: row.get("loss")
        for row in mia_metrics["by_split"]
    }
    utility_by_split = {
        row["split"]: row["score"]
        for row in utility_metrics["by_split"]
    }
    scalar_metrics = {
        "rwku/forget/rouge_l_recall": metrics["forget"]["rouge_l_recall"],
        "rwku/neighbor/rouge_l_recall": metrics["neighbor_locality"]["rouge_l_recall"],
        **{f"rwku/{split}/rouge_l_recall": value for split, value in generation_by_split.items()},
        **{f"rwku/{split}/loss": value for split, value in mia_by_split.items()},
        **{f"rwku/{split}/score": value for split, value in utility_by_split.items()},
    }
    checkpoint_metadata = OmegaConf.to_container(
        evaluation.get("checkpoint", {}),
        resolve=True,
    )
    training_config_path = evaluation.get("training_config_path")
    hydra_config = None
    if training_config_path:
        training_config_file = Path(training_config_path)
        if training_config_file.exists():
            hydra_config = OmegaConf.to_container(
                OmegaConf.load(training_config_file),
                resolve=True,
            )
        else:
            print(f"Training Hydra config not found: {training_config_file}")

    artifact = wandb.Artifact(
        name=str(evaluation.wandb.artifact_name),
        type="model" if evaluation.wandb.log_model_artifact else "evaluation",
        metadata={
            "model_name_or_path": str(evaluation.model_name_or_path),
            "subjects": str(evaluation.subjects),
            "checkpoint": checkpoint_metadata,
        },
    )
    if evaluation.wandb.log_model_artifact:
        if not model_dir.is_dir():
            raise ValueError(
                "evaluation.wandb.log_model_artifact=true requires "
                "evaluation.model_name_or_path to be a local model directory."
            )
        artifact.add_dir(str(model_dir), name="model")
    artifact.add_dir(str(output_dir), name="rwku_evaluation")

    init_kwargs = {
        "project": project,
        "entity": evaluation.wandb.entity,
        "name": evaluation.wandb.run_name,
        "config": {
            "model_name_or_path": str(evaluation.model_name_or_path),
            "subjects": str(evaluation.subjects),
            "sets": OmegaConf.to_container(evaluation.sets, resolve=True),
            "metrics": OmegaConf.to_container(evaluation.metrics, resolve=True),
            "checkpoint": checkpoint_metadata,
            "training_config_path": str(training_config_path) if training_config_path else None,
            "hydra": hydra_config,
        },
    }
    if training_run_info:
        init_kwargs.update(
            {
                "id": training_run_info["id"],
                "project": training_run_info["project"],
                "entity": training_run_info["entity"],
                "name": training_run_info["name"],
                "resume": "must",
            }
        )
    else:
        init_kwargs["job_type"] = "evaluation"

    with wandb.init(**init_kwargs) as run:
        run.log({key: value for key, value in scalar_metrics.items() if value is not None})
        run.log_artifact(artifact)


def run_evaluation(cfg: DictConfig) -> None:
    evaluation = cfg.evaluation
    shard = evaluation.get("shard")
    shard_count = int(shard.count) if shard is not None else 1
    shard_index = int(shard.index) if shard is not None else 0
    if shard_count <= 0:
        raise ValueError("evaluation.shard.count must be >= 1.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "evaluation.shard.index must be between 0 and "
            f"evaluation.shard.count - 1, got {shard_index}/{shard_count}."
        )
    output_suffix = f".shard{shard_index}" if shard_count > 1 else ""

    if evaluation.metrics.generation != "rouge_l_recall":
        raise ValueError(
            "evaluation.metrics.generation must be 'rouge_l_recall'; "
            "no other RWKU generation metric is implemented."
        )
    if evaluation.batch_size <= 0:
        raise ValueError("evaluation.batch_size must be >= 1")
    if evaluation.mia_batch_size <= 0:
        raise ValueError("evaluation.mia_batch_size must be >= 1")
    if evaluation.choice_likelihood_batch_size <= 0:
        raise ValueError("evaluation.choice_likelihood_batch_size must be >= 1")

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
            max_examples=None,
            shard=shard,
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
            max_examples=None,
            shard=shard,
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
            max_examples=None,
            shard=shard,
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
                else None
            ),
            sample_strategy=evaluation.limits.utility_sample_strategy,
            sample_seed=evaluation.limits.utility_sample_seed,
            shard=shard,
            max_new_tokens=evaluation.generation.max_new_tokens,
            temperature=evaluation.generation.temperature,
            batch_sizes=evaluation.utility_batch_size,
            choice_likelihood_batch_size=evaluation.choice_likelihood_batch_size,
        )

    metrics = aggregate_generation(all_rows)
    mia_metrics = aggregate_mia(all_mia_rows)
    utility_metrics = aggregate_utility(all_utility_rows)
    model_label = evaluation.model.label or Path(evaluation.model_name_or_path).name or evaluation.model_name_or_path
    summary_table_row = build_summary_table_row(metrics, mia_metrics, utility_metrics, model_label)

    with open(output_dir / f"rwku_generations{output_suffix}.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / f"rwku_mia{output_suffix}.jsonl", "w", encoding="utf-8") as f:
        for row in all_mia_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / f"rwku_utility{output_suffix}.jsonl", "w", encoding="utf-8") as f:
        for row in all_utility_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics["mia"] = mia_metrics
    metrics["utility"] = utility_metrics

    with open(output_dir / f"rwku_results{output_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if shard_count == 1:
        write_summary_tables(output_dir, summary_table_row)
        log_wandb_results(evaluation, env_values, output_dir, metrics, mia_metrics, utility_metrics)

    print(json.dumps({
        "forget": metrics["forget"],
        "neighbor_locality": metrics["neighbor_locality"],
        "mia": mia_metrics,
        "utility": utility_metrics,
        "by_split": metrics["by_split"],
        "summary_table_row": summary_table_row,
    }, indent=2))


@hydra.main(version_base=None, config_path="../../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
