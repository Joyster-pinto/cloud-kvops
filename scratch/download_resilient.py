import os
import sys
import subprocess
import shutil

INCOMPLETE_BLOB = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee.incomplete")
FINAL_BLOB = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee")
SNAPSHOT_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, "model.safetensors")

URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
TARGET_SIZE = 3087467144

cmd = [
    "curl.exe",
    "-C", "-",
    "-L",
    "--retry", "50",
    "--retry-all-errors",
    "--retry-delay", "1",
    "--speed-limit", "1000",
    "--speed-time", "15",
    "-o", INCOMPLETE_BLOB,
    URL
]

print(f"[*] Running resilient curl download targeting {TARGET_SIZE} bytes...")
ret = subprocess.run(cmd)

if ret.returncode == 0:
    curr_size = os.path.getsize(INCOMPLETE_BLOB)
    print(f"[+] Curl exited with 0. Current size: {curr_size} bytes.")
    if curr_size == TARGET_SIZE:
        if os.path.exists(FINAL_BLOB):
            os.remove(FINAL_BLOB)
        os.rename(INCOMPLETE_BLOB, FINAL_BLOB)
        print(f"[+] Final blob verified and renamed: {FINAL_BLOB}")
        
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        if os.path.exists(SNAPSHOT_FILE):
            os.remove(SNAPSHOT_FILE)
        try:
            os.link(FINAL_BLOB, SNAPSHOT_FILE)
            print("[+] Hardlinked into snapshot directory.")
        except Exception:
            shutil.copyfile(FINAL_BLOB, SNAPSHOT_FILE)
            print("[+] Copied into snapshot directory.")
            
        print("[+] SUCCESS: Qwen2.5-1.5B-Instruct safetensors ready!")
    else:
        print(f"[!] Size mismatch: {curr_size} != {TARGET_SIZE}")
else:
    print(f"[!] Curl exited with error code {ret.returncode}")
