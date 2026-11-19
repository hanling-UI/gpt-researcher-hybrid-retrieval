"""Rerank 精排：给「query + 候选段落」逐对打分，再截断到 top_n。

为什么需要它
============
hybrid_retrieve 返回的是 RRF 融合后的候选。RRF 只看"名次"、不看"内容"，
所以它没法判断某一条到底能不能回答问题——融合结果的尾部必然掺着
"两路都提过但都不太相关"的段落（实测里就是天气和菜谱）。

精排模型把 query 和候选拼成一个输入送进模型，直接输出相关性分数。因为要
逐对计算，它只能用在少量候选上（10~50 条），这正是"先粗排 20、再精排 3"
这个两阶段结构的由来。

接口说明（2026-11 实测确认）
===========================
* 只有「原生接口」能用，OpenAI 兼容路径 /compatible-mode/v1/reranks 返回 404；
* 用 qwen3-rerank。gte-rerank-v2 官方公告已排期下线，且同样三条输入它只给
  0.338 / 0.120 / 0.006，分差被压平；qwen3-rerank 给 0.946 / 0.335 / 0.166，
  判别力强得多；
* 返回结构：{"output": {"results": [{"index": 0, "relevance_score": 0.95}, ...]}}
  其中 index 是**输入 documents 数组的下标**，不是 chunk_id，取回时要映射。
"""

from __future__ import annotations

import logging
import os

import httpx

from .hybrid import Chunk

logger = logging.getLogger(__name__)

_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
_DEFAULT_MODEL = "qwen3-rerank"
_TIMEOUT = 20.0

# 最终送给大模型的条数。3 条是"够回答一个问题"和"省 token"之间的折中，
# 可以通过环境变量调，不用改代码。
_DEFAULT_TOP_N = int(os.environ.get("RERANK_TOP_K", "3"))


class RerankError(RuntimeError):
    """精排失败。调用方收到它应当退回 RRF 顺序，而不是把整条链路带崩。"""


class DashScopeReranker:
    """阿里云百炼 qwen3-rerank 的精排实现。"""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = _TIMEOUT,
    ):
        self.model = model
        # 复用 OPENAI_API_KEY：同一个百炼账号，不需要再配一个新的 key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int = _DEFAULT_TOP_N,
    ) -> list[tuple[Chunk, float]]:
        """给 chunks 逐条打分，返回分数最高的 top_n 条（Chunk, 分数）。"""
        if not chunks:
            return []
        if not self.api_key:
            raise RerankError("缺少 OPENAI_API_KEY，无法调用重排接口")

        documents = [chunk.text for chunk in chunks]
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {
                # top_n 超过文档数会报错，先夹一下
                "top_n": min(top_n, len(documents)),
                # 不要它把原文再回传一遍：我们手里已经有 chunks，
                # 只要 index 和分数就够了，回传原文纯属浪费带宽和 token
                "return_documents": False,
            },
        }

        try:
            response = httpx.post(
                _ENDPOINT,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
                # 直连，理由和 network.py 里 LLM 走直连一样：
                # 阿里云是国内服务，经代理只会多一次握手
                trust_env=False,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RerankError(f"重排请求失败：{exc}") from exc

        results = (data.get("output") or {}).get("results") or []
        picked: list[tuple[Chunk, float]] = []
        for item in results:
            index = item.get("index")
            # 防一手：接口返回的下标理论上一定合法，但越界会让整个检索崩掉，
            # 这种"理论上不会发生"的地方正是最容易出事故的地方
            if not isinstance(index, int) or not (0 <= index < len(chunks)):
                continue
            picked.append((chunks[index], float(item.get("relevance_score") or 0.0)))

        if not picked:
            raise RerankError("重排接口返回了空结果")

        # 不依赖接口已经排好序：自己再排一次，代价为零
        picked.sort(key=lambda pair: -pair[1])
        return picked[:top_n]


def get_reranker(backend: str | None = None):
    """按 RERANK_BACKEND 选择后端；返回 None 表示"不做精排"。"""
    backend = (backend or os.environ.get("RERANK_BACKEND", "dashscope")).strip().lower()
    if backend in ("none", "off", "0", "false", "no"):
        return None
    if backend == "dashscope":
        return DashScopeReranker()
    raise RerankError(f"未知的 RERANK_BACKEND: {backend}")


def rerank_or_fallback(
    query: str,
    chunks: list[Chunk],
    top_n: int = _DEFAULT_TOP_N,
) -> list[tuple[Chunk, float]]:
    """精排并截断；失败就退回"按原顺序截断"，永不抛异常。

    把"重排可能失败"收敛在这一个函数里，select.py 那边就不需要写 try/except。
    这是整条链路"任何一环挂了都还能出报告"的最后一环。
    """
    if not chunks:
        return []

    try:
        reranker = get_reranker()
        if reranker is None:
            return [(chunk, 0.0) for chunk in chunks[:top_n]]
        return reranker.rerank(query, chunks, top_n)
    except RerankError as exc:
        logger.warning("重排不可用（%s），沿用 RRF 顺序截断", exc)
        return [(chunk, 0.0) for chunk in chunks[:top_n]]


__all__ = ["DashScopeReranker", "RerankError", "get_reranker", "rerank_or_fallback"]