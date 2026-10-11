import os
import sys
import time
import hashlib
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPECTED_SIZE = 3087467144
EXPECTED_SHA256 = "b6cf5fd6630a3eb466d3643e8e224d9a73f739047eeb4c9960800e530dfcbb3e"
URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors"
SNAPSHOT_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a")
DEST = os.path.join(SNAPSHOT_DIR, "model.safetensors")
PART = DEST + ".part"
DONE_FILE = PART + ".done"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Resolve CDN URL
print("[*] Resolving direct CDN URL from Hugging Face...", flush=True)
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    direct_url = resp.geturl()

print(f"[*] CDN URL: {direct_url[:90]}...", flush=True)

CHUNK_SIZE = 8 * 1024 * 1024 # 8 MB chunks
NUM_WORKERS = 6
TOTAL_CHUNKS = (EXPECTED_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE

# Ensure .part file exists with full size
if not os.path.exists(PART) or os.path.getsize(PART) < EXPECTED_SIZE:
    with open(PART, "a+b") as f:
        f.seek(EXPECTED_SIZE - 1)
        f.write(b"\0")

done_chunks = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                done_chunks.add(int(line))
else:
    # If starting from an existing prefix
    blob = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/blobs/c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8.incomplete")
    if os.path.exists(blob):
        existing_bytes = os.path.getsize(blob)
        full_chunks = existing_bytes // CHUNK_SIZE
        print(f"[*] Reusing {full_chunks} sequential chunks from blob", flush=True)
        for c in range(full_chunks):
            done_chunks.add(c)
        with open(DONE_FILE, "w") as f:
            for c in done_chunks:
                f.write(f"{c}\n")

missing_chunks = [c for c in range(TOTAL_CHUNKS) if c not in done_chunks]
print(f"[*] Total chunks: {TOTAL_CHUNKS}, Completed: {len(done_chunks)}, Remaining: {len(missing_chunks)}", flush=True)

file_lock = threading.Lock()

def download_chunk(chunk_idx):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE - 1, EXPECTED_SIZE - 1)
    chunk_len = end - start + 1
    
    for attempt in range(6):
        try:
            req = urllib.request.Request(direct_url, headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={start}-{end}"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) != chunk_len:
                    time.sleep(1)
                    continue
                with file_lock:
                    with open(PART, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    with open(DONE_FILE, "a") as f:
                        f.write(f"{chunk_idx}\n")
                return chunk_idx
        except Exception as e:
            time.sleep(1 + attempt)
    return None

completed_count = len(done_chunks)
t_start = time.time()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(download_chunk, c): c for c in missing_chunks}
    for future in as_completed(futures):
        c = future.result()
        if c is not None:
            completed_count += 1
            elapsed = time.time() - t_start
            downloaded = (completed_count - (TOTAL_CHUNKS - len(missing_chunks))) * CHUNK_SIZE
            speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
            pct = (completed_count / TOTAL_CHUNKS) * 100
            print(f"\r[{pct:5.1f}%] {completed_count}/{TOTAL_CHUNKS} chunks ({speed:4.1f} MB/s)", end="", flush=True)
        else:
            print(f"\n[!] Chunk failed: {futures[future]}", flush=True)

print("\n[*] All chunks finished! Verifying SHA256 checksum...", flush=True)
h = hashlib.sha256()
with open(PART, "rb") as f:
    while block := f.read(16 * 1024 * 1024):
        h.update(block)

calc_sha = h.hexdigest()
print(f"Calculated SHA256: {calc_sha}", flush=True)
print(f"Expected SHA256  : {EXPECTED_SHA256}", flush=True)

if calc_sha == EXPECTED_SHA256:
    if os.path.exists(DEST):
        os.remove(DEST)
    os.rename(PART, DEST)
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)
    print("[+] SUCCESS! model.safetensors verified and ready!", flush=True)
else:
    print("[!] CHECKSUM MISMATCH!", flush=True)
