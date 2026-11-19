import json
import re

import httpx

raw = open(r"D:\项目\gpt-researcher-main\.env", encoding="utf-8-sig").read()
vals = {}
for line in raw.splitlines():
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line.strip())
    if m:
        vals[m.group(1)] = m.group(2).strip()
key = vals["OPENAI_API_KEY"]
base = vals["OPENAI_BASE_URL"].rstrip("/")
h = {"Authorization": f"Bearer {key}"}

with httpx.Client(trust_env=False, timeout=30) as c:
    try:
        r = c.get(base + "/models", headers=h)
        print("models:", r.status_code, r.text[:600])
    except Exception as e:
        print("models ERR", e)

    for model in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen3-max", "qwen2.5-7b-instruct"]:
        try:
            r = c.post(
                base + "/chat/completions",
                headers={**h, "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
            )
            body = r.text[:180]
            print(f"{model:22} -> {r.status_code} {body}")
        except Exception as e:
            print(f"{model:22} -> ERR {type(e).__name__} {str(e)[:120]}")
