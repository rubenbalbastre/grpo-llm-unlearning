import json
from pathlib import Path

import hydra
from dotenv import dotenv_values
from omegaconf import DictConfig

from results import (
    aggregate_generation,
    aggregate_mia,
    aggregate_utility,
    build_summary_table_row,
    write_summary_tables,
)
from rwku import log_wandb_results


def read_jsonl(path: Path, *, required: bool = False) -> list[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required shard output: {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_sharded_rows(output_dir: Path, stem: str, shard_count: int) -> list[dict]:
    rows = []
    for shard_index in range(shard_count):
        rows.extend(
            read_jsonl(
                output_dir / f"{stem}.shard{shard_index}.jsonl",
                required=True,
            )
        )
    return rows


def merge_shards(cfg: DictConfig) -> None:
    evaluation = cfg.evaluation
    shard = evaluation.get("shard")
    shard_count = int(shard.count) if shard is not None else 1
    if shard_count <= 1:
        print("No sharded outputs to merge because evaluation.shard.count <= 1.")
        return

    output_dir = Path(evaluation.output_dir)
    generation_rows = read_sharded_rows(output_dir, "rwku_generations", shard_count)
    mia_rows = read_sharded_rows(output_dir, "rwku_mia", shard_count)
    utility_rows = read_sharded_rows(output_dir, "rwku_utility", shard_count)

    write_jsonl(output_dir / "rwku_generations.jsonl", generation_rows)
    write_jsonl(output_dir / "rwku_mia.jsonl", mia_rows)
    write_jsonl(output_dir / "rwku_utility.jsonl", utility_rows)

    metrics = aggregate_generation(generation_rows)
    mia_metrics = aggregate_mia(mia_rows)
    utility_metrics = aggregate_utility(utility_rows)
    metrics["mia"] = mia_metrics
    metrics["utility"] = utility_metrics

    model_label = (
        evaluation.model.label
        or Path(evaluation.model_name_or_path).name
        or evaluation.model_name_or_path
    )
    summary_table_row = build_summary_table_row(
        metrics,
        mia_metrics,
        utility_metrics,
        model_label,
    )

    with open(output_dir / "rwku_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_summary_tables(output_dir, summary_table_row)

    env_values = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    log_wandb_results(
        evaluation,
        env_values,
        output_dir,
        metrics,
        mia_metrics,
        utility_metrics,
    )

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
    merge_shards(cfg)


if __name__ == "__main__":
    main()
