import os
import sys
import time
import hashlib
import urllib.request

EXPECTED_SIZE = 3087467144
EXPECTED_SHA256 = "b6cf5fd6630a3eb466d3643e8e224d9a73f739047eeb4c9960800e530dfcbb3e"
URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model.safetensors"
SNAPSHOT_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a")
DEST = os.path.join(SNAPSHOT_DIR, "model.safetensors")
PART = DEST + ".part"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# First get redirect URL
print("[*] Resolving direct download link from Hugging Face...")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    direct_url = resp.geturl()

print(f"[*] Direct CDN URL resolved:\n    {direct_url[:100]}...")

CHUNK_SIZE = 8 * 1024 * 1024 # 8 MB chunks

def get_current_size():
    return os.path.getsize(PART) if os.path.exists(PART) else 0

while True:
    cur_size = get_current_size()
    if cur_size >= EXPECTED_SIZE:
        print(f"[+] Download complete: {cur_size} bytes. Verifying SHA256...")
        h = hashlib.sha256()
        with open(PART, "rb") as f:
            while chunk := f.read(16 * 1024 * 1024):
                h.update(chunk)
        calc_sha = h.hexdigest()
        print(f"Calculated: {calc_sha}")
        print(f"Expected  : {EXPECTED_SHA256}")
        if calc_sha == EXPECTED_SHA256:
            if os.path.exists(DEST):
                os.remove(DEST)
            os.rename(PART, DEST)
            print("[+] SUCCESS: model.safetensors verified and saved!")
            break
        else:
            print("[!] Checksum mismatch! Redownloading from scratch...")
            os.remove(PART)
            continue

    # Download next chunk with Range header
    end_byte = min(cur_size + CHUNK_SIZE - 1, EXPECTED_SIZE - 1)
    req = urllib.request.Request(direct_url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={cur_size}-{end_byte}"
    })
    
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) == 0:
                print("[!] Zero bytes read, retrying...")
                time.sleep(1)
                continue
            with open(PART, "ab") as f:
                f.write(data)
        elapsed = time.time() - t0
        cur_size = get_current_size()
        pct = (cur_size / EXPECTED_SIZE) * 100
        speed_mb = (len(data) / (1024 * 1024)) / max(elapsed, 0.001)
        print(f"\r[{pct:5.1f}%] {cur_size / (1024**2):.1f} / {EXPECTED_SIZE / (1024**2):.1f} MB ({speed_mb:.1f} MB/s)", end="", flush=True)
    except Exception as e:
        print(f"\n[!] Chunk failed ({e}), reconnecting in 2s...")
        time.sleep(2)
        # Refresh direct url in case token expired
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                direct_url = resp.geturl()
        except Exception:
            pass
