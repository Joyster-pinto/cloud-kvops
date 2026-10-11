import os
import sys
import time
import requests

TARGET_FILE = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee.incomplete")
FINAL_BLOB = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee")
SNAPSHOT_FILE = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors")
URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
TOTAL_SIZE = 3087467144
CHUNK_SIZE = 10 * 1024 * 1024 # 10MB chunks

while True:
    current_size = os.path.getsize(TARGET_FILE)
    if current_size >= TOTAL_SIZE:
        print(f"\n[+] File reached total size: {current_size} bytes!", flush=True)
        break
    
    start = current_size
    end = min(TOTAL_SIZE - 1, start + CHUNK_SIZE - 1)
    expected_len = end - start + 1
    
    success = False
    for attempt in range(1, 5):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=20)
            if r.status_code in (200, 206) and len(r.content) == expected_len:
                with open(TARGET_FILE, "ab") as f:
                    f.write(r.content)
                pct = (os.path.getsize(TARGET_FILE) / TOTAL_SIZE) * 100
                print(f"  Downloaded bytes {start}-{end} ({len(r.content)/(1024*1024):.1f}MB) -> {pct:.1f}%", flush=True)
                success = True
                break
            else:
                print(f"  Attempt {attempt}: Bad response ({r.status_code}, len={len(r.content)} vs {expected_len})", flush=True)
                time.sleep(1)
        except Exception as e:
            print(f"  Attempt {attempt} error: {e}", flush=True)
            time.sleep(1)
            
    if not success:
        print("[!] Failed after 5 attempts on chunk. Stopping.", flush=True)
        sys.exit(1)

# Verification & Installation
final_size = os.path.getsize(TARGET_FILE)
if final_size == TOTAL_SIZE:
    if os.path.exists(FINAL_BLOB):
        os.remove(FINAL_BLOB)
    os.rename(TARGET_FILE, FINAL_BLOB)
    print(f"[+] Blob saved: {FINAL_BLOB}", flush=True)
    
    import shutil
    shutil.copyfile(FINAL_BLOB, SNAPSHOT_FILE)
    print(f"[+] Successfully installed to: {SNAPSHOT_FILE}", flush=True)
else:
    print(f"[!] Size mismatch: {final_size} != {TOTAL_SIZE}", flush=True)
