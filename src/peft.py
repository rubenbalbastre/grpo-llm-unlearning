from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig


def get_peft_config(cfg: DictConfig, num_hidden_layers: int) -> LoraConfig:
    lora_args = cfg.get("peft", None).get("lora", None)
    number_layers_to_transform = int(lora_args.number_layers_to_transform)
    if number_layers_to_transform == -1:
        layers_to_transform = list(range(num_hidden_layers))
    elif 1 <= number_layers_to_transform <= num_hidden_layers:
        layers_to_transform = list(
            range(num_hidden_layers - number_layers_to_transform, num_hidden_layers)
        )
    else:
        raise ValueError(
            "peft.lora.number_layers_to_transform must be -1 or between 1 "
            f"and {num_hidden_layers}, got {number_layers_to_transform}."
        )
    return LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.alpha,
        init_lora_weights=lora_args.init_lora_weights,
        target_modules=OmegaConf.to_container(
            lora_args.target_modules, resolve=True
        ),
        task_type="CAUSAL_LM",
        bias="none",
        layers_pattern="layers",
        layers_to_transform=layers_to_transform,
    )
