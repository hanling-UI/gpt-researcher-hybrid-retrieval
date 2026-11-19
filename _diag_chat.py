import json
import re
import time

import httpx

raw = open(r"D:\项目\gpt-researcher-main\.env", encoding="utf-8-sig").read()
vals = {}
for line in raw.splitlines():
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$", line.strip())
    if m:
        vals[m.group(1)] = m.group(2).strip()

key = vals["OPENAI_API_KEY"]
PROXY = "http://127.0.0.1:12450"
bases = {
    "maas": vals["OPENAI_BASE_URL"],
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def post(base, proxy, model="qwen-turbo"):
    t0 = time.time()
    kw = {"trust_env": False, "timeout": 60.0}
    if proxy:
        kw["proxy"] = proxy
    try:
        with httpx.Client(**kw) as c:
            r = c.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "say ok"}], "max_tokens": 8},
            )
        print(f"  {base[:55]:55} proxy={bool(proxy)} -> {r.status_code} {round(time.time()-t0,1)}s {r.text[:150]}")
    except Exception as e:
        print(f"  {base[:55]:55} proxy={bool(proxy)} -> ERR {type(e).__name__} {str(e)[:100]} {round(time.time()-t0,1)}s")


for name, base in bases.items():
    print(name)
    post(base, None)
    post(base, PROXY)
