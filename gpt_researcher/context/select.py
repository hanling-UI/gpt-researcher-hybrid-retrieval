"""Turns pages of text into LLM context, the same way everywhere.

Research sub-queries (scraped pages) and report chat (the finished report)
both call ``select_context``. ``CONTEXT_FILTER`` picks how chunks are chosen:

  auto        jev when TYPESAFE_API_KEY is set, otherwise keyword
  jev         TypeSafe's Jev scores each chunk's usefulness (needs a key)
  keyword     BM25 keyword ranking: local, no API, no model, no embeddings
  embeddings  embedding similarity (needs an embedding provider)
  none        no filtering: every page, in full

Anything that fails degrades to ``keyword``, so no mode can leave a run
without context. Benchmarks: ``evals/context_filter``.
"""

import asyncio
import logging
import os
from typing import Callable

from langchain_core.documents import Document

from ..prompts import PromptFamily
from .jev_filter import JevContextCompressor, JevError
from .lexical import LexicalContextCompressor
from .retriever import SearchAPIRetriever

logger = logging.getLogger(__name__)

CONTEXT_FILTERS = ("auto", "jev", "keyword", "embeddings", "hybrid",  "none")

#: Keyword chunks must score at least this fraction of the best chunk, and
#: up to this many are kept. Chosen by evals/context_filter: the threshold
#: is what beats embeddings; plain top-10 BM25 loses to them.
KEYWORD_RELATIVE_THRESHOLD = float(os.environ.get("KEYWORD_RELATIVE_THRESHOLD", "0.5"))
KEYWORD_MAX_RESULTS = int(os.environ.get("KEYWORD_MAX_RESULTS", "25"))
#: 精排后最终送给大模型的条数。
#: 3 条是"够回答一个问题"和"省 token"之间的折中，可用环境变量调。
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "3"))

def resolve_context_filter(setting: str | None) -> str:
    """The filter to use; ``auto`` means Jev when a TypeSafe key is set, else keyword."""
    mode = (setting or "auto").strip().lower()
    if mode not in CONTEXT_FILTERS:
        logger.warning(f"Unknown CONTEXT_FILTER {setting!r}; expected one of {CONTEXT_FILTERS}. Using 'auto'.")
        mode = "auto"
    if mode == "auto":
        return "jev" if os.environ.get("TYPESAFE_API_KEY") else "keyword"
    return mode


def _as_documents(pages: list[dict]) -> list[Document]:
    return SearchAPIRetriever(pages=pages).invoke("")


async def select_context(
    query: str,
    pages: list[dict],
    cfg,
    *,
    max_results: int = 10,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    cost_callback: Callable[[float], None] | None = None,
    embeddings: Callable[[], object] | None = None,
) -> str:
    """Context for ``query`` from ``pages`` (dicts with ``raw_content``, ``url``, ``title``).

    ``embeddings`` is a zero-argument factory, called only in embeddings mode,
    so nothing builds an embedding model unless that mode is chosen.
    """
    mode = resolve_context_filter(getattr(cfg, "context_filter", None))
    prompt_family = prompt_family or PromptFamily

    if mode == "none":
        return prompt_family.pretty_print_docs(_as_documents(pages))

    # Little content needs no filtering, whatever the mode.
    # 但 hybrid 是例外：它是用户显式选的模式，价值就在排序，不能被"内容少"截胡。
    # 真实抓取经常只有 2~3 页成功（反爬 403 很常见），总字数不到 8000、
    # 页数也少于 max_results，正好命中这条捷径——hybrid 会被静默跳过。
    total_chars = sum(len(str(p.get("raw_content") or "")) for p in pages)
    if (
            mode != "hybrid"
            and total_chars < int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))
            and len(pages) <= max_results
    ):
        return prompt_family.pretty_print_docs(_as_documents(pages)[:max_results], max_results)

    if mode == "jev":
        try:
            return await JevContextCompressor(documents=pages, prompt_family=prompt_family).async_get_context(
                query=query, max_results=max_results, cost_callback=cost_callback)
        except JevError as e:
            logger.warning(f"Jev context filter unavailable ({e}); using keyword ranking")

    if mode == "embeddings":
        try:
            from .compression import ContextCompressor

            return await ContextCompressor(
                documents=pages,
                embeddings=embeddings() if embeddings else None,
                similarity_threshold=getattr(cfg, "similarity_threshold", None),
                prompt_family=prompt_family,
            ).async_get_context(query=query, max_results=max_results, cost_callback=cost_callback)
        except Exception as e:
            logger.warning(f"Embedding context filter unavailable ({e}); using keyword ranking")
    if mode == "hybrid":
        # 混合检索：BM25 关键词召回 + 向量语义召回，用 RRF 融合。
        #
        # 位置很关键：必须放在上面"小内容快速返回"之后。因为 select_context
        # 有一条捷径——总字数小于 COMPRESSION_THRESHOLD 且页面数不超过
        # max_results 时，直接把原文返回、不做任何排序。如果你发现
        # CONTEXT_FILTER=hybrid 完全没生效，先看这里是不是被捷径截胡了。
        try:
            from .hybrid import hybrid_retrieve

            # embeddings 传进来的是"工厂函数"而不是实例：构造 embedding 客户端
            # 要读 key、建 HTTP 连接，不该在不用向量的模式下白付这个成本。
            embedding_model = embeddings() if embeddings else None

            # hybrid_retrieve 里既有 CPU 密集的 BM25 打分，也有同步的 HTTP 请求。
            # 直接 await 会卡住事件循环——多个 sub-query 并发时会退化成串行。
            # to_thread 把它丢进线程池，事件循环继续处理别的 sub-query。
            fused = await asyncio.to_thread(
                hybrid_retrieve, query, pages, embedding_model, top_k=20
            )

            if fused:
                # 精排：把 RRF 融合后的候选（最多 20 条）重新打分并截断到 3 条。
                # rerank_or_fallback 内部已经兜住所有异常（接口挂了 / 没配 key /
                # 返回空），失败时退回 RRF 顺序截断，所以这里不用再写 try/except。
                from .rerank import rerank_or_fallback

                candidates = [chunk for chunk, _score in fused]
                final = await asyncio.to_thread(
                    rerank_or_fallback, query, candidates, RERANK_TOP_K
                )

                # Chunk.as_document() 转成 LangChain 的 Document，
                # 复用原有的 pretty_print_docs，输出格式和其他模式完全一致。
                docs = [chunk.as_document() for chunk, _score in final]
                return prompt_family.pretty_print_docs(docs, max_results)

            # 两路都没命中（极少见）：明确地降级，而不是返回空字符串。
            logger.info("混合检索未命中任何候选，退回关键词排序")

        except Exception as e:  # noqa: BLE001 - 检索失败不能把整条链路带崩
            logger.warning(f"混合检索不可用（{e}），退回关键词排序")

    return await LexicalContextCompressor(
        documents=pages, relative_threshold=KEYWORD_RELATIVE_THRESHOLD, prompt_family=prompt_family,
    ).async_get_context(query, KEYWORD_MAX_RESULTS)
