"""Smoke test: LLM goes direct, Tavily goes through WEB_PROXY, no env pollution."""
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Load .env exactly like main.py does.
from dotenv import load_dotenv

load_dotenv()

# Simulate the worst case: a stale proxy in the environment, plus the
# Windows registry proxy that is actually configured on this machine.
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

print("routes:", __import__("gpt_researcher.network", fromlist=["x"]).describe_routes())
print("env proxy vars left in os.environ:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})

provider = GenericLLMProvider.from_provider(
    "openai", model="qwen-flash", temperature=0.1, max_tokens=16
)
print("client base_url:", provider.llm.openai_api_base)
print("http_client:", type(provider.llm.http_client).__name__, "trust_env=", provider.llm.http_client._trust_env)
print("async_client:", type(provider.llm.http_async_client).__name__)


async def call_llm():
    try:
        out = await provider.get_chat_response(
            [{"role": "user", "content": "reply with the single word: pong"}], stream=False
        )
        print("LLM OK ->", repr(out)[:200])
    except Exception as exc:
        print("LLM FAILED ->", type(exc).__name__, str(exc)[:300])


async def call_tavily():
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch

    results = TavilySearch("gpt researcher open source").search(max_results=3)
    print(f"TAVILY OK -> {len(results)} results", [r['href'][:60] for r in results])


asyncio.run(call_llm())
asyncio.run(call_tavily())
