import os
import sys
import time
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
filename = "model.safetensors"

print(f"[*] Resuming download of {filename} from {repo_id}...")
for attempt in range(1, 10):
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True
        )
        print(f"[+] Download complete: {path}")
        break
    except Exception as e:
        print(f"[!] Attempt {attempt} failed: {e}. Retrying in 3s...")
        time.sleep(3)
