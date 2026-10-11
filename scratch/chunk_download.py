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
remaining = TOTAL_SIZE - current_size
print(f"[*] Resuming from byte {current_size} to {TOTAL_SIZE} (Remaining: {remaining/(1024*1024):.1f} MB)", flush=True)

CHUNK_DIR = os.path.join(os.path.dirname(__file__), "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

NUM_WORKERS = 8
chunk_size = (remaining + NUM_WORKERS - 1) // NUM_WORKERS

ranges = []
for i in range(NUM_WORKERS):
    start = current_size + i * chunk_size
    end = min(TOTAL_SIZE - 1, start + chunk_size - 1)
    if start <= end:
        chunk_path = os.path.join(CHUNK_DIR, f"part_{i}.bin")
        ranges.append((i, start, end, chunk_path))

print(f"[*] Dispatching {len(ranges)} parallel chunks to {CHUNK_DIR}...", flush=True)

def download_chunk(item):
    worker_id, start, end, chunk_path = item
    expected_len = end - start + 1
    
    # Check if already downloaded
    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) == expected_len:
        print(f"  [Chunk {worker_id}] Already complete ({expected_len/(1024*1024):.1f} MB)", flush=True)
        return True

    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(1, 6):
        try:
            print(f"  [Chunk {worker_id}] Downloading bytes {start}-{end} ({expected_len/(1024*1024):.1f} MB)...", flush=True)
            r = requests.get(URL, headers=headers, stream=True, timeout=60)
            if r.status_code in (200, 206):
                with open(chunk_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                if os.path.getsize(chunk_path) == expected_len:
                    print(f"  [Chunk {worker_id}] Done!", flush=True)
                    return True
                else:
                    print(f"  [Chunk {worker_id}] Incomplete size, retrying...", flush=True)
            else:
                print(f"  [Chunk {worker_id}] HTTP {r.status_code}, retrying...", flush=True)
        except Exception as e:
            print(f"  [Chunk {worker_id}] Error: {e}, retrying attempt {attempt}...", flush=True)
    return False

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = list(executor.map(download_chunk, ranges))

if all(results):
    print("[*] All chunks downloaded successfully! Merging into main file...", flush=True)
    with open(TARGET_FILE, "ab") as dest:
        for worker_id, start, end, chunk_path in ranges:
            with open(chunk_path, "rb") as src:
                dest.write(src.read())
            os.remove(chunk_path)
            print(f"  Merged chunk {worker_id}", flush=True)
    
    final_size = os.path.getsize(TARGET_FILE)
    print(f"[+] Merge complete! Final size: {final_size} bytes (Target: {TOTAL_SIZE})", flush=True)
    
    if final_size == TOTAL_SIZE:
        if os.path.exists(FINAL_BLOB):
            os.remove(FINAL_BLOB)
        os.rename(TARGET_FILE, FINAL_BLOB)
        print(f"[+] Renamed blob to {FINAL_BLOB}", flush=True)
        
        import shutil
        shutil.copyfile(FINAL_BLOB, SNAPSHOT_FILE)
        print(f"[+] Successfully installed model.safetensors to: {SNAPSHOT_FILE}", flush=True)
    else:
        print(f"[!] Size mismatch: {final_size} != {TOTAL_SIZE}", flush=True)
else:
    print("[!] Some chunks failed. Re-run to retry.", flush=True)
