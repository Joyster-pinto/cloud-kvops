import os
import sys
import time
import hashlib
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPECTED_SIZE = 988097824
EXPECTED_SHA256 = "6a565607bdd63c4ac10851c253ac34bd1056be59131c7fd431c0dc111e194600"
URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/model.safetensors"
SNAPSHOT_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25")
DEST = os.path.join(SNAPSHOT_DIR, "model.safetensors")
PART = DEST + ".part"
DONE_FILE = PART + ".done"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

print("[*] Resolving direct CDN URL from Hugging Face...", flush=True)
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    direct_url = resp.geturl()

print(f"[*] CDN URL resolved: {direct_url[:80]}...", flush=True)

CHUNK_SIZE = 8 * 1024 * 1024 # 8 MB chunks
NUM_WORKERS = 6
TOTAL_CHUNKS = (EXPECTED_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE

# Create clean sparse file
if not os.path.exists(PART) or os.path.getsize(PART) < EXPECTED_SIZE:
    with open(PART, "wb") as f:
        f.seek(EXPECTED_SIZE - 1)
        f.write(b"\0")

done_chunks = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE, "r") as f:
        for line in f:
            if line.strip():
                done_chunks.add(int(line.strip()))

missing_chunks = [c for c in range(TOTAL_CHUNKS) if c not in done_chunks]
print(f"[*] Total chunks: {TOTAL_CHUNKS}, Completed: {len(done_chunks)}, Remaining: {len(missing_chunks)}", flush=True)

file_lock = threading.Lock()

def download_chunk(chunk_idx):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE - 1, EXPECTED_SIZE - 1)
    chunk_len = end - start + 1
    
    for attempt in range(8):
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
    print("[+] SUCCESS! model.safetensors 100% verified and loaded!", flush=True)
else:
    print("[!] CHECKSUM MISMATCH!", flush=True)
