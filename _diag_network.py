"""Diagnose which route (direct / proxy) works for DashScope-compatible and Tavily."""
import json
import os
import re
import sys
import time

import httpx

env_path = r"D:\项目\gpt-researcher-main\.env"
raw = open(env_path, encoding="utf-8-sig").read()

vals = {}
for line in raw.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$", line)
    if m:
        vals[m.group(1)] = m.group(2).strip()

key = vals.get("OPENAI_API_KEY", "")
tav = vals.get("TAVILY_API_KEY", "")
bases = {
    "maas": vals.get("OPENAI_BASE_URL", ""),
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
PROXY = os.environ.get("GPTR_TEST_PROXY", "http://127.0.0.1:12450")


def mask(s):
    return (s[:10] + "..." + s[-4:]) if len(s) > 16 else s


def try_chat(base, proxy, label):
    t0 = time.time()
    kwargs = {"trust_env": False, "timeout": 20.0}
    if proxy:
        kwargs["proxy"] = proxy
    try:
        with httpx.Client(**kwargs) as c:
            r = c.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "qwen-turbo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            )
        print(f"[LLM {label}] {r.status_code} {round(time.time()-t0,1)}s :: {r.text[:200]}")
    except Exception as e:
        print(f"[LLM {label}] ERROR {type(e).__name__}: {str(e)[:200]}")


def try_tavily(proxy, label):
    t0 = time.time()
    kwargs = {"trust_env": False, "timeout": 25.0}
    if proxy:
        kwargs["proxy"] = proxy
    try:
        with httpx.Client(**kwargs) as c:
            r = c.post(
                "https://api.tavily.com/search",
                json={"api_key": tav, "query": "OpenAI", "max_results": 2, "search_depth": "basic"},
            )
        ok = "results" in r.text
        print(f"[TAVILY {label}] {r.status_code} {round(time.time()-t0,1)}s results={ok} :: {r.text[:160]}")
    except Exception as e:
        print(f"[TAVILY {label}] ERROR {type(e).__name__}: {str(e)[:200]}")


print("key:", mask(key), "| tavily:", mask(tav))
print("maas base:", bases["maas"])
for name, base in bases.items():
    if not base:
        continue
    try_chat(base, None, f"{name}/direct")
    try_chat(base, PROXY, f"{name}/proxy")
try_tavily(None, "direct")
try_tavily(PROXY, "proxy")
