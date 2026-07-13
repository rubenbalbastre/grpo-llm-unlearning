from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig


def has_peft_adapter(model) -> bool:
    return bool(getattr(model, "peft_config", None))


def make_loaded_peft_adapter_trainable(model) -> None:
    if not has_peft_adapter(model):
        return

    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()

    for peft_config in getattr(model, "peft_config", {}).values():
        peft_config.inference_mode = False

    for name, parameter in model.named_parameters():
        train_adapter_parameter = "lora_" in name or "modules_to_save" in name
        parameter.requires_grad_(train_adapter_parameter)
        if train_adapter_parameter:
            parameter.data = parameter.data.float()


def get_peft_config(cfg: DictConfig, num_hidden_layers: int) -> LoraConfig | None:
    peft_args = cfg.get("peft")
    if peft_args is None or not bool(peft_args.get("enabled", True)):
        return None

    lora_args = peft_args.get("lora")
    if lora_args is None:
        raise ValueError("peft.enabled=true requires peft.lora to be configured.")

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
