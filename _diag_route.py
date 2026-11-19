import time
import httpx

PROXY = "http://127.0.0.1:12450"
targets = [
    "https://www.baidu.com/",
    "https://dashscope.aliyuncs.com/",
    "https://ws-d2rj43y0qchw060q.cn-beijing.maas.aliyuncs.com/",
    "https://www.aliyun.com/",
    "https://api.tavily.com/",
]


def go(url, proxy):
    t0 = time.time()
    kw = {"trust_env": False, "timeout": 15.0, "follow_redirects": True}
    if proxy:
        kw["proxy"] = proxy
    try:
        with httpx.Client(**kw) as c:
            r = c.get(url)
        return f"{r.status_code} ({round(time.time()-t0,1)}s)"
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:90]} ({round(time.time()-t0,1)}s)"


for url in targets:
    print(f"{url:60} proxy -> {go(url, PROXY)}")
print()
for url in targets:
    print(f"{url:60} direct -> {go(url, None)}")

print()
print("=== DoH via proxy (real IPs) ===")
try:
    with httpx.Client(trust_env=False, proxy=PROXY, timeout=15) as c:
        for host in ["dashscope.aliyuncs.com", "ws-d2rj43y0qchw060q.cn-beijing.maas.aliyuncs.com", "api.tavily.com"]:
            r = c.get("https://223.5.5.5/resolve", params={"name": host, "type": "A"})
            data = r.json()
            print(host, [a.get("data") for a in data.get("Answer", [])])
except Exception as e:
    print("DoH ERR", type(e).__name__, str(e)[:200])
