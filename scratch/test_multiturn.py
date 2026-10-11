import requests
import json

url = "http://localhost:8000/api/stream_single"

prompt_turn1 = """Task:
Infix to Postfix Translation using SDT

Scenario:
A compiler for a stack-based virtual machine requires expressions to be converted into postfix notation before execution. The developer uses Syntax Directed to convert infix expressions into postfix form during parsing.

Design an SDT scheme to translate infix expressions into postfix expressions.

how do i start coding this?"""

# Turn 1
history = [
    {"role": "user", "content": prompt_turn1}
]

print("[*] Testing Turn 1 via POST /api/stream_single...")
res1 = requests.post(url, json={
    "messages": history,
    "mode": "baseline",
    "max_tokens": 100,
    "temperature": 0.7,
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct"
}, stream=True)

bot_reply1 = ""
for line in res1.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith("data: "):
            d = json.loads(decoded[6:])
            if d.get("token"):
                bot_reply1 += d["token"]

print("\n--- Turn 1 Reply (Preview) ---")
print(bot_reply1[:200])

# Turn 2
history.append({"role": "assistant", "content": bot_reply1})
history.append({"role": "user", "content": "give complete c code for this"})

print("\n[*] Testing Turn 2 with FULL conversation history...")
res2 = requests.post(url, json={
    "messages": history,
    "mode": "baseline",
    "max_tokens": 150,
    "temperature": 0.7,
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct"
}, stream=True)

bot_reply2 = ""
for line in res2.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        if decoded.startswith("data: "):
            d = json.loads(decoded[6:])
            if d.get("token"):
                bot_reply2 += d["token"]

print("\n--- Turn 2 Reply (Preview) ---")
print(bot_reply2[:300])
