import os
import sys
import requests
import concurrent.futures

TARGET_FILE = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee.incomplete")
FINAL_BLOB = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee")
SNAPSHOT_FILE = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors")

URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"
TOTAL_SIZE = 3087467144

current_size = os.path.getsize(TARGET_FILE)
print(f"[*] Resuming from byte {current_size} to {TOTAL_SIZE} (Remaining: {(TOTAL_SIZE - current_size)/(1024*1024):.1f} MB)")

# Split remaining bytes into 10 worker chunks
remaining = TOTAL_SIZE - current_size
NUM_WORKERS = 10
chunk_size = (remaining + NUM_WORKERS - 1) // NUM_WORKERS

ranges = []
for i in range(NUM_WORKERS):
    start = current_size + i * chunk_size
    end = min(TOTAL_SIZE - 1, start + chunk_size - 1)
    if start <= end:
        ranges.append((i, start, end))

print(f"[*] Starting {len(ranges)} parallel download workers...")

def download_range(item):
    worker_id, start, end = item
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            r = requests.get(URL, headers=headers, stream=True, timeout=30)
            if r.status_code in (200, 206):
                data = r.content
                if len(data) == (end - start + 1):
                    # Write into target file at offset
                    with open(TARGET_FILE, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    print(f"  [Worker {worker_id}] Saved bytes {start}-{end} ({len(data)/(1024*1024):.1f} MB)")
                    return True
                else:
                    print(f"  [Worker {worker_id}] Incomplete data ({len(data)} != {end - start + 1}), retrying...")
            else:
                print(f"  [Worker {worker_id}] HTTP {r.status_code}, retrying...")
        except Exception as e:
            print(f"  [Worker {worker_id}] Exception: {e}, retrying...")
    return False

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = list(executor.map(download_range, ranges))

if all(results):
    final_size = os.path.getsize(TARGET_FILE)
    print(f"[+] Download complete! Final size: {final_size} bytes (Target: {TOTAL_SIZE})")
    if os.path.exists(FINAL_BLOB):
        os.remove(FINAL_BLOB)
    os.rename(TARGET_FILE, FINAL_BLOB)
    print(f"[+] Renamed blob to {FINAL_BLOB}")
    
    # Copy or hardlink to snapshot
    import shutil
    shutil.copyfile(FINAL_BLOB, SNAPSHOT_FILE)
    print(f"[+] Installed model.safetensors to snapshot: {SNAPSHOT_FILE}")
else:
    print("[!] Some chunks failed. Re-run to complete remaining.")
