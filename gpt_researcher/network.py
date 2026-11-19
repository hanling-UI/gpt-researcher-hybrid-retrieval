"""Explicit outbound HTTP routing for this deployment.

Why this module exists
----------------------
This app needs two *opposite* network routes at the same time:

* the LLM endpoint (Aliyun DashScope, OpenAI-compatible) must go out
  **without** any proxy -- a proxied hop is reset / handshake-times-out;
* web search (Tavily) and page scraping need the local proxy, because
  the target sites are not reachable from the mainland network directly.

Proxy settings are process-global, so a single ``HTTP_PROXY`` cannot
express both requirements. Worse, on Windows they are picked up from more
than one place:

* ``requests`` / ``urllib3`` read ``HTTP(S)_PROXY`` **and** the registry
  proxy (``ProxyEnable`` / ``ProxyServer``) via
  ``urllib.request.getproxies()``;
* ``httpx`` reads the same two sources whenever ``trust_env=True``
  (its default), which is what ``langchain-openai`` and ``openai`` build;
* ``langchain-openai`` then logs ``detected HTTP_PROXY ...`` and, more
  importantly, routes the model request through that proxy.

Clearing the env vars is therefore not enough on Windows as long as the
system proxy switch is on -- the registry value is still read.

The fix implemented here is to stop relying on ambient configuration:

* :func:`direct_client` / :func:`direct_async_client` return ``httpx``
  clients built with ``trust_env=False``. That flag disables env-var
  proxies, registry proxies, ``.netrc`` and ``SSL_CERT_FILE`` lookups, so
  the LLM traffic is pinned to a direct connection no matter what the
  shell or the proxy client does.
* :func:`web_post_json` sends the search request to an *explicit* proxy
  read from ``WEB_PROXY`` (and falls back to a direct connection when the
  proxy is unreachable), again with ``trust_env=False`` so the choice is
  honoured exactly as configured.

Configuration
-------------
``WEB_PROXY`` (or ``GPTR_WEB_PROXY``): proxy URL for web traffic, e.g.
``http://127.0.0.1:12450``. Set it to ``direct``/``none``/``off`` to
force web traffic to bypass the proxy as well. When unset, web traffic
goes direct.
"""
from __future__ import annotations

import asyncio
import logging
import os
import weakref
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Long read timeout: a full report can stream for several minutes.
_LLM_TIMEOUT = httpx.Timeout(connect=20.0, read=300.0, write=120.0, pool=20.0)
_WEB_TIMEOUT = httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=20.0)

_DIRECT: set[str] = {"direct", "none", "off", "0", "false", "no"}  # no proxy

_direct_sync: httpx.Client | None = None
_direct_async: "weakref.WeakKeyDictionary[Any, httpx.AsyncClient]" = weakref.WeakKeyDictionary()
_web_clients: dict[str | None, httpx.Client] = {}


# ---------------------------------------------------------------------------
# Direct (no proxy) clients -- used for the LLM / embeddings / domestic APIs
# ---------------------------------------------------------------------------
def direct_client() -> httpx.Client:
    """A process-wide ``httpx`` client that never uses a proxy."""
    global _direct_sync
    if _direct_sync is None or _direct_sync.is_closed:
        _direct_sync = httpx.Client(trust_env=False, timeout=_LLM_TIMEOUT)
    return _direct_sync


def direct_async_client() -> httpx.AsyncClient:
    """Loop-bound, reused ``httpx.AsyncClient`` that never uses a proxy.

    ``httpx.AsyncClient`` is tied to the event loop that created it, so one
    instance per running loop is cached instead of one per LLM call (this
    module is hit on every chat completion). When no loop is running yet
    (``from_provider`` can be called outside async code) an uncached client
    is returned, so the caller still gets a proxy-free transport.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return httpx.AsyncClient(trust_env=False, timeout=_LLM_TIMEOUT)

    client = _direct_async.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(trust_env=False, timeout=_LLM_TIMEOUT)
        _direct_async[loop] = client
    return client


# ---------------------------------------------------------------------------
# Web clients -- search / scraping, optionally through an explicit proxy
# ---------------------------------------------------------------------------
def web_proxy_url() -> str | None:
    """Proxy URL for outbound web traffic, or ``None`` for a direct call."""
    for name in ("WEB_PROXY", "GPTR_WEB_PROXY"):
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        if raw.lower() in _DIRECT:
            return None
        return raw
    return None


def _web_client(proxy: str | None) -> httpx.Client:
    client = _web_clients.get(proxy)
    if client is None or client.is_closed:
        client = httpx.Client(trust_env=False, proxy=proxy, timeout=_WEB_TIMEOUT)
        _web_clients[proxy] = client
    return client


def web_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> httpx.Response:
    """POST JSON to a web endpoint through ``WEB_PROXY``, falling back to direct.

    Only *transport* failures trigger the fallback: a 4xx/5xx answer means the
    route worked and the response is returned untouched. This keeps the app
    usable both when the proxy client is running and when it is not.
    """
    proxy = web_proxy_url()
    candidates: list[str | None] = [proxy] + ([None] if proxy else [])
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return _web_client(candidate).post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout if timeout is not None else _WEB_TIMEOUT,
            )
        except httpx.HTTPError as exc:  # connect/timeout/proxy errors only
            last_error = exc
            label = candidate or "direct"
            logger.warning("Web request to %s via %s failed: %s", url, label, exc)

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def describe_routes() -> dict[str, str]:
    """Human-readable summary of the routing policy, for a startup log line."""
    return {
        "llm": "direct (trust_env=False)",
        "web": web_proxy_url() or "direct",
    }
