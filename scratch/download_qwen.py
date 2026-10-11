import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("[*] Starting fast snapshot download from hf-mirror.com...", flush=True)
t0 = time.time()
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    allow_patterns=["*.safetensors", "*.json", "*.txt"]
)
print(f"[+] Download complete in {time.time() - t0:.1f}s: {path}", flush=True)
