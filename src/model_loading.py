from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import distributed_barrier, is_main_process


def is_peft_adapter_path(model_name: str) -> bool:
    return (Path(model_name) / "adapter_config.json").is_file()


def is_local_model_source(model_name: str) -> bool:
    return Path(model_name).exists()


def model_cache_dir(model_name: str, storage_root: str | Path = ".") -> Path:
    return Path(storage_root) / "outputs" / "model" / model_name.replace("/", "--")


def tokenizer_cache_dir(model_name: str, storage_root: str | Path = ".") -> Path:
    return Path(storage_root) / "outputs" / "tokenizer" / model_name.replace("/", "--")


def has_cached_model(path: Path) -> bool:
    return (path / "config.json").is_file()


def has_cached_tokenizer(path: Path) -> bool:
    return (
        (path / "tokenizer_config.json").is_file()
        or (path / "tokenizer.json").is_file()
    )


def load_cached_model(
    model_name: str,
    storage_root: str | Path = ".",
    *,
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    if is_local_model_source(model_name):
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    cache_dir = model_cache_dir(model_name, storage_root)
    if has_cached_model(cache_dir):
        print(f"Loading model from local cache: {cache_dir}", flush=True)
        return AutoModelForCausalLM.from_pretrained(cache_dir, **model_kwargs)

    print(f"Downloading model {model_name} and caching at {cache_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if is_main_process():
        cache_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cache_dir)
    distributed_barrier()
    return model


def load_cached_tokenizer(tokenizer_name: str, storage_root: str | Path = "."):
    if is_local_model_source(tokenizer_name):
        return AutoTokenizer.from_pretrained(tokenizer_name)

    cache_dir = tokenizer_cache_dir(tokenizer_name, storage_root)
    if has_cached_tokenizer(cache_dir):
        print(f"Loading tokenizer from local cache: {cache_dir}", flush=True)
        return AutoTokenizer.from_pretrained(cache_dir)

    print(f"Downloading tokenizer {tokenizer_name} and caching at {cache_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if is_main_process():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(cache_dir)
    distributed_barrier()
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    storage_root: str | Path = ".",
    *,
    torch_dtype: str | None = None,
    attn_implementation: str | None = None,
):
    resolved_dtype = getattr(torch, torch_dtype) if torch_dtype else None
    tokenizer_source = model_name
    if is_peft_adapter_path(model_name):
        peft_config = PeftConfig.from_pretrained(model_name)
        tokenizer_source = model_name
        base_model = load_cached_model(
            peft_config.base_model_name_or_path,
            storage_root,
            torch_dtype=resolved_dtype,
            attn_implementation=attn_implementation,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_name,
            is_trainable=True,
        )
    else:
        model = load_cached_model(
            model_name,
            storage_root,
            torch_dtype=resolved_dtype,
            attn_implementation=attn_implementation,
        )

    tokenizer = load_cached_tokenizer(tokenizer_source, storage_root)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer
