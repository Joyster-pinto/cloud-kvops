import requests
import json

url = "http://localhost:8000/api/stream_single"
params = {
    "prompt": "Tell me a fun and engaging short story about an offline robot on a mission.",
    "mode": "kvops",
    "max_tokens": 150,
    "budget_slots": 192,
    "temperature": 0.7,
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct"
}

print("[*] Connecting to stream_single (KVOps)...")
response = requests.get(url, params=params, stream=True)

final_text = ""
step_count = 0
for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith("data: "):
            payload = json.loads(decoded[6:])
            if payload.get("is_complete"):
                print("\n[+] Stream Complete!")
                break
            step = payload.get("step")
            token = payload.get("token", "")
            final_text += token
            step_count += 1
            if step_count % 25 == 0:
                print(f"  Step {step}: kv_mb={payload.get('kv_cache_mb')}, evicted={payload.get('tokens_evicted')}")

print("\n--- Final Generated Text (First 300 chars) ---")
print(final_text[:300])
print(f"Total steps generated: {step_count}")
