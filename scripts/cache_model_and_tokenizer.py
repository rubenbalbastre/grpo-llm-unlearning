import shutil
import sys
from pathlib import Path

from src.model_loading import (
    load_model_and_tokenizer,
    model_cache_dir,
    tokenizer_cache_dir,
)


model_name = sys.argv[1]
storage_root = Path(sys.argv[2])
model_dir = model_cache_dir(model_name, storage_root)
tokenizer_dir = tokenizer_cache_dir(model_name, storage_root)
completion_marker = model_dir / ".cache_complete"

shutil.rmtree(model_dir, ignore_errors=True)
shutil.rmtree(tokenizer_dir, ignore_errors=True)
load_model_and_tokenizer(
    model_name,
    storage_root=storage_root,
    torch_dtype="bfloat16",
)
completion_marker.write_text("complete\n", encoding="utf-8")
