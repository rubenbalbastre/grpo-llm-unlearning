import sys
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


model_name = sys.argv[1]
storage_root = Path(sys.argv[2])
cache_name = model_name.replace("/", "--")
model_dir = storage_root / "outputs" / "model" / cache_name
tokenizer_dir = storage_root / "outputs" / "tokenizer" / cache_name
completion_marker = model_dir / ".cache_complete"

shutil.rmtree(model_dir, ignore_errors=True)
shutil.rmtree(tokenizer_dir, ignore_errors=True)
snapshot_download(repo_id=model_name, local_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.save_pretrained(tokenizer_dir)
completion_marker.write_text("complete\n", encoding="utf-8")
