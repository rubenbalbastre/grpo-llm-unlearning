#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path(".")
OUTPUT_ROOT = STORAGE_ROOT / "outputs"
DATA_ROOT = STORAGE_ROOT / "data"
STOP_MARKER = "low_reward_stop.json"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
TARGETS = [
    "Jennifer Lopez",
    "Tony Blair",
    "Marlon Brando",
    "Bruce Lee",
    "Serena Williams",
    "John D. Rockefeller",
    "Tom Clancy",
    "Vincent van Gogh",
    "Karl Marx",
    "Confucius",
]
REWARDS = ["r0", "r1", "r2", "r4"]
RUN_RWKU_EVAL = False
RUN_HOLD_OUT_EVAL = False


@dataclass
class TrainingRun:
    run_name: str
    checkpoint_root: Path
    model_path: str
    job_id: str | None = None
    blocked: bool = False


def slug(value: str, separator: str = "-") -> str:
    return re.sub(r"[^a-z0-9]+", separator, value.lower()).strip(separator)


def reasoning_effort() -> str:
    config = OmegaConf.load(REPO_ROOT / "config/train.yaml")
    return str(config.reward.functions["llm-judge"].reasoning_effort)


def submit(
    script: str,
    *args: str,
    dependencies: tuple[str | None, ...] = (),
    exports: dict[str, str] | None = None,
) -> str:
    dependency_ids = [job_id for job_id in dependencies if job_id]
    command = ["sbatch", "--parsable"]
    if dependency_ids:
        command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
    if exports:
        values = ",".join(f"{key}={value}" for key, value in exports.items())
        command.append(f"--export=ALL,{values}")
    command.extend([f"scripts/{script}", *args])
    print(f"$ {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "REPO_DIR": str(REPO_ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split(";", 1)[0]


def require_datasets() -> None:
    required = (
        "dataset_dict.json",
        "grpo/train",
        "grpo/test",
        "sft/train",
        "sft/test",
    )
    missing = [
        str(DATA_ROOT / target)
        for target in TARGETS
        if any(not (DATA_ROOT / target / relative).exists() for relative in required)
    ]
    if missing:
        raise SystemExit("Missing generated datasets:\n" + "\n".join(missing))


def submit_sft(model: str, target: str, previous_job: str | None) -> TrainingRun:
    run_name = f"r2warmup_{slug(model, '_')}_{slug(target, '_')}"
    checkpoint_root = OUTPUT_ROOT / run_name
    final_model = checkpoint_root / "final_model"
    run = TrainingRun(run_name, checkpoint_root, str(final_model))

    if final_model.is_dir():
        print(f"SFT {target}: already exists ({run_name})")
        return run
    if checkpoint_root.exists():
        print(f"SFT {target}: incomplete output ({run_name})")
        run.blocked = True
        return run

    run.job_id = submit(
        "run-sft.sh",
        f"wandb.run_name={run_name}",
        f"model.name={model}",
        f"experiment.forget_concept={target}",
        f"paths.storage_root={STORAGE_ROOT}",
        "training.sft.objective=broad",
        "reward.type=r2",
        "training.sft.save_final_model=true",
        dependencies=(previous_job,),
    )
    print(f"SFT {target}: {run.job_id} ({run_name})")
    return run


def submit_grpo(
    *,
    variant: str,
    base_model: str,
    model_label: str,
    target: str,
    reward: str,
    dependencies: tuple[str | None, ...],
    r2_reasoning_effort: str,
) -> TrainingRun:
    run_name = f"unlearning-{slug(variant)}-{slug(model_label)}-{slug(target)}-{reward}"
    if reward == "r2":
        run_name += f"-reasoning-{slug(r2_reasoning_effort)}"
    checkpoint_root = OUTPUT_ROOT / run_name
    final_model = checkpoint_root / "final_model"
    run = TrainingRun(run_name, checkpoint_root, str(final_model))

    if final_model.is_dir():
        print(f"GRPO {variant} | {target} | {reward}: already exists ({run_name})")
        return run
    if checkpoint_root.exists():
        print(
            f"GRPO {variant} | {target} | {reward}: incomplete output ({run_name})"
        )
        run.blocked = True
        return run

    run.job_id = submit(
        "run-grpo.sh",
        f"model.name={base_model}",
        f"experiment.forget_concept={target}",
        f"paths.storage_root={STORAGE_ROOT}",
        f"reward.type={reward}",
        "training.grpo.save_final_model=true",
        dependencies=dependencies,
        exports={"RUN_NAME": run_name},
    )
    print(f"GRPO {variant} | {target} | {reward}: {run.job_id} ({run_name})")
    return run


def holdout_output_dir(
    target: str,
    model_path: str,
    checkpoint_root: Path | None,
) -> Path:
    if checkpoint_root is not None:
        return Path(model_path) / "hold_out_eval"
    return OUTPUT_ROOT / f"hold-out-baseline-{slug(model_path)}-{slug(target)}" / "hold_out_eval"


def submit_rwku(
    *,
    target: str,
    model_path: str,
    dependency: str | None = None,
    checkpoint_root: Path | None = None,
    baseline_label: str | None = None,
) -> None:
    if not RUN_RWKU_EVAL:
        return

    if checkpoint_root is not None:
        output_dir = Path(model_path) / "eval_rwku"
        marker = checkpoint_root / STOP_MARKER
        if marker.is_file():
            print(f"RWKU {target}: skipped due to {marker}")
            return
        if (output_dir / "rwku_summary_table.csv").is_file():
            print(f"RWKU {target}: already exists")
            return
        job_id = submit(
            "run-eval-rwku.sh",
            dependencies=(dependency,),
            exports={
                "CHECKPOINT_ROOT": str(checkpoint_root),
                "SKIP_IF_MARKER": str(marker),
            },
        )
    else:
        assert baseline_label is not None
        output_dir = OUTPUT_ROOT / baseline_label / "eval_rwku"
        if (output_dir / "rwku_summary_table.csv").is_file():
            print(f"RWKU baseline {target}: already exists")
            return
        job_id = submit(
            "run-eval-rwku.sh",
            f"evaluation.model_name_or_path={model_path}",
            f"evaluation.output_dir={output_dir}",
            f"evaluation.subjects={target}",
            "evaluation.model.label=baseline",
            f"evaluation.wandb.run_name={baseline_label}",
            f"evaluation.wandb.artifact_name={baseline_label}",
        )
    print(f"RWKU {target}: {job_id}")


def submit_holdout(
    *,
    target: str,
    model_path: str,
    dependency: str | None = None,
    checkpoint_root: Path | None = None,
) -> None:
    if not RUN_HOLD_OUT_EVAL:
        return

    output_dir = holdout_output_dir(target, model_path, checkpoint_root)
    if (output_dir / "metrics.csv").is_file() and (output_dir / "summary.csv").is_file():
        print(f"Hold-out {target}: already exists")
        return
    if output_dir.exists():
        print(f"Hold-out {target}: incomplete output ({output_dir})")
        return

    exports = None
    if checkpoint_root is not None:
        marker = checkpoint_root / STOP_MARKER
        if marker.is_file():
            print(f"Hold-out {target}: skipped due to {marker}")
            return
        exports = {"SKIP_IF_MARKER": str(marker)}

    job_id = submit(
        "run-eval-behaviour.sh",
        f"concept={target}",
        f"model_name_or_path={model_path}",
        f"output_dir={output_dir}",
        f"paths.storage_root={STORAGE_ROOT}",
        dependencies=(dependency,),
        exports=exports,
    )
    print(f"Hold-out {target}: {job_id}")


def submit_evaluations(run: TrainingRun, target: str) -> None:
    if run.blocked:
        return
    submit_rwku(
        target=target,
        model_path=run.model_path,
        dependency=run.job_id,
        checkpoint_root=run.checkpoint_root,
    )
    submit_holdout(
        target=target,
        model_path=run.model_path,
        dependency=run.job_id,
        checkpoint_root=run.checkpoint_root,
    )


def main() -> None:
    os.chdir(REPO_ROOT)
    require_datasets()
    r2_effort = reasoning_effort()
    previous_sft_job: str | None = None
    previous_r2_job: str | None = None

    for original_model in MODELS:
        print(f"\nModel: {original_model}", flush=True)
        for target in TARGETS:
            print(f"\nTarget: {target}", flush=True)
            model_slug = slug(original_model)
            target_slug = slug(target)

            submit_rwku(
                target=target,
                model_path=original_model,
                baseline_label=f"rwku-baseline-{model_slug}-{target_slug}",
            )
            submit_holdout(target=target, model_path=original_model)

            sft_run = submit_sft(original_model, target, previous_sft_job)
            if sft_run.job_id:
                previous_sft_job = sft_run.job_id
            if not sft_run.blocked:
                submit_holdout(
                    target=target,
                    model_path=sft_run.model_path,
                    dependency=sft_run.job_id,
                    checkpoint_root=sft_run.checkpoint_root,
                )

            for reward in REWARDS:
                original_dependencies = (
                    (previous_r2_job,) if reward == "r2" else ()
                )
                original_run = submit_grpo(
                    variant="original",
                    base_model=original_model,
                    model_label=original_model,
                    target=target,
                    reward=reward,
                    dependencies=original_dependencies,
                    r2_reasoning_effort=r2_effort,
                )
                if reward == "r2" and original_run.job_id:
                    previous_r2_job = original_run.job_id
                submit_evaluations(original_run, target)

                if sft_run.blocked:
                    continue
                warmed_dependencies = (sft_run.job_id,)
                if reward == "r2":
                    warmed_dependencies += (previous_r2_job,)
                warmed_run = submit_grpo(
                    variant="r2-warmed",
                    base_model=sft_run.model_path,
                    model_label=original_model,
                    target=target,
                    reward=reward,
                    dependencies=warmed_dependencies,
                    r2_reasoning_effort=r2_effort,
                )
                if reward == "r2" and warmed_run.job_id:
                    previous_r2_job = warmed_run.job_id
                submit_evaluations(warmed_run, target)


if __name__ == "__main__":
    main()
