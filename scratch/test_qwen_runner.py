import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from src.engine.real_llm_runner import RealLLMRunner

print("[*] Initializing RealLLMRunner with Qwen2.5-1.5B-Instruct...")
runner = RealLLMRunner(model_name="Qwen/Qwen2.5-1.5B-Instruct")

prompt = """Task:
Infix to Postfix Translation using SDT

how do i start coding this?"""

print("\n[*] Testing baseline generation (10 tokens)...")
gen = runner.generate_streaming_baseline(prompt, max_new_tokens=10)
for step in gen:
    print(step["token_text"], end="", flush=True)

print("\n\n[*] Testing KVOps generation (10 tokens)...")
gen_kv = runner.generate_streaming_kvops(prompt, max_new_tokens=10, budget_capacity=64)
for step in gen_kv:
    print(step["token_text"], end="", flush=True)

print("\n\n[+] Both generators succeeded!")
