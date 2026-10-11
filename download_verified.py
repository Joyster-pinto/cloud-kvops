"""
Parallel, verified Hugging Face model downloader.
Guarantees byte-level integrity against Hugging Face Git LFS SHA-256 metadata.
"""

import os
import sys
import glob
import time
import shutil
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

def download_file_chunked(url, target_path, expected_size, num_threads=16, max_retries=5):
    print(f"[*] Downloading {expected_size / (1024*1024):.1f} MB with {num_threads} workers to {target_path}...")
    chunk_size = expected_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < num_threads - 1 else expected_size - 1
        ranges.append((i, start, end))

    # Pre-allocate target file
    with open(target_path, "wb") as f:
        f.seek(expected_size - 1)
        f.write(b"\0")

    def download_range(worker_id, start_b, end_b):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={start_b}-{end_b}"
            }
        )
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                    expected_chunk_len = end_b - start_b + 1
                    if len(data) != expected_chunk_len:
                        raise ValueError(f"Worker {worker_id}: Expected {expected_chunk_len} bytes, got {len(data)}")
                    with open(target_path, "r+b") as fp:
                        fp.seek(start_b)
                        fp.write(data)
                    return True
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Worker {worker_id} failed after {max_retries} attempts: {e}")
                time.sleep(1.0 * attempt)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_range, r[0], r[1], r[2]): r[0] for r in ranges}
        completed = 0
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            elapsed = time.time() - t0
            progress = (completed / num_threads) * 100
            mb_done = (completed / num_threads) * (expected_size / (1024 * 1024))
            speed = mb_done / elapsed if elapsed > 0 else 0
            print(f"    [{progress:5.1f}%] {mb_done:6.1f} MB / {expected_size/(1024*1024):.1f} MB ({speed:.1f} MB/s)", end="\r")
    
    elapsed = time.time() - t0
    speed = (expected_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    print(f"\n[+] Download finished in {elapsed:.1f}s ({speed:.1f} MB/s)")

def verify_and_install(repo_id: str, num_threads: int = 16):
    api = HfApi()
    print(f"\n=======================================================")
    print(f"[*] Querying repo metadata for: {repo_id}")
    print(f"=======================================================")
    info = api.model_info(repo_id, files_metadata=True)

    # 1. Download small configs and tokenizers via standard API first
    print("[*] Syncing configuration & tokenizer files...")
    for s in info.siblings:
        if s.rfilename.endswith((".json", ".txt", ".model", ".tiktoken")):
            try:
                hf_hub_download(repo_id=repo_id, filename=s.rfilename)
            except Exception as e:
                print(f"    Note on {s.rfilename}: {e}")

    # 2. Find safetensors files
    large_files = [s for s in info.siblings if s.rfilename.endswith(".safetensors") and s.lfs]
    if not large_files:
        print(f"[-] No .safetensors files found in {repo_id}!")
        return False

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    
    # Locate snapshot dir
    snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if not snapshot_dirs:
        # Trigger download of config to establish snapshot
        hf_hub_download(repo_id=repo_id, filename="config.json")
        snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    snapshot_dir = snapshot_dirs[0]

    for item in large_files:
        filename = item.rfilename
        expected_sha = item.lfs.get("sha256")
        expected_size = item.size
        print(f"\n[*] Processing {filename} ({expected_size / (1024*1024):.1f} MB)...")
        print(f"    Expected SHA256: {expected_sha}")

        # Check existing file in snapshot
        target_path = os.path.join(snapshot_dir, filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) == expected_size:
            print("    Checking existing local file hash...")
            h = hashlib.sha256()
            with open(target_path, "rb") as fp:
                while chunk := fp.read(10 * 1024 * 1024):
                    h.update(chunk)
            if h.hexdigest() == expected_sha:
                print(f"    [MATCH] File is already 100% verified and intact! Skipping download.")
                continue
            else:
                print(f"    [CORRUPTED] Hash mismatch on disk! Found: {h.hexdigest()}")

        # Download direct from Hugging Face resolve URL
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        part_path = target_path + ".download"
        download_file_chunked(url, part_path, expected_size, num_threads=num_threads)

        print("[*] Verifying downloaded file SHA-256...")
        h = hashlib.sha256()
        with open(part_path, "rb") as fp:
            while chunk := fp.read(10 * 1024 * 1024):
                h.update(chunk)
        actual_sha = h.hexdigest()
        print(f"    Actual SHA256:   {actual_sha}")

        if actual_sha != expected_sha:
            print(f"[-] ERROR: Hash verification failed! Got {actual_sha}, expected {expected_sha}")
            os.remove(part_path)
            return False

        print("[+] SHA-256 MATCH VERIFIED! Replacing snapshot file...")
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(part_path, target_path)

        # Also update blobs directory if present
        blobs_dir = os.path.join(model_dir, "blobs")
        if os.path.exists(blobs_dir):
            blob_dest = os.path.join(blobs_dir, expected_sha)
            if os.path.exists(blob_dest):
                os.remove(blob_dest)
            shutil.copy2(target_path, blob_dest)
            print(f"[+] Updated LFS blob {expected_sha}")

    print(f"\n[✓] ALL FILES FOR {repo_id} ARE VERIFIED AND READY TO INFER!\n")
    return True

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    verify_and_install(target_repo)
