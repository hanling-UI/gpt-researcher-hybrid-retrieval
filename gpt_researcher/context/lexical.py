"""Keyword (BM25) relevance filtering of scraped content.

The dependency-free context filter: no API, no model, no embeddings. Chunks
are ranked by BM25, the lexical scoring behind most search engines, computed
over the chunks of the pages scraped for one sub-query. Runs locally in
milliseconds.
"""

import math
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..prompts import PromptFamily
from .retriever import SearchAPIRetriever

# 切词统一委托给 context/tokenize.py：中文 jieba、英文小写+去停用词+复数还原。
#
# 这里原来是一句 ``re.compile(r"\w+", re.UNICODE)``：``\w`` 在 Unicode 下
# 包含汉字，``+`` 又是贪婪匹配，所以**整串中文被切成一个 token**，
# "混合检索对大模型应用有什么好处" 只会产出一个 token，与任何正文都匹配
# 不上，BM25 全篇 0 分（检索等价于按原始顺序取前 N 段）。
# 英文因为天然有空格，从来没暴露这个 bug。
#
# 用「重导出」而不是改调用点：下面 bm25_scores() 和模块末尾的
# rank_written_sections() 都调用 tokenize，一处修好，两处自动受益；
# 外部（select.py / context_manager.py）也一直是从本模块拿东西，导入路径不变。
from .tokenize import tokenize  # noqa: F401  (re-export, 保持对外接口不变)


def bm25_scores(query: str, passages: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of every passage for ``query``, with IDF taken from the passages themselves."""
    docs = [Counter(tokenize(p)) for p in passages]
    if not docs:
        return []
    avg_len = sum(sum(d.values()) for d in docs) / len(docs) or 1.0
    query_terms = set(tokenize(query))
    df = {t: sum(1 for d in docs if t in d) for t in query_terms}
    idf = {t: math.log((len(docs) - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
    scores = []
    for d in docs:
        length = sum(d.values())
        score = 0.0
        for t in query_terms:
            tf = d.get(t, 0)
            if tf:
                score += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores


class LexicalContextCompressor:
    """Keeps the chunks with the highest BM25 score for the query.

    ``relative_threshold`` drops chunks scoring below that fraction of the
    best chunk, so a page with one relevant paragraph doesn't pull in its
    filler just to fill the budget.
    """

    def __init__(
        self,
        documents,
        relative_threshold: float = 0.0,
        chunk_size: int = 1000,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.documents = documents
        self.relative_threshold = relative_threshold
        self.chunk_size = chunk_size
        self.prompt_family = prompt_family

    def _chunks(self) -> list[Document]:
        pages = SearchAPIRetriever(pages=self.documents).invoke("")
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size // 10)
        return [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    def select(self, query: str, max_results: int) -> list[Document]:
        chunks = self._chunks()
        scores = bm25_scores(query, [c.page_content for c in chunks])
        best = max(scores, default=0.0)
        if best <= 0:
            # No keyword overlap at all: fall back to the pages' opening chunks.
            return chunks[:max_results]
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        keep = [i for i in ranked if scores[i] >= self.relative_threshold * best][:max_results]
        return [chunks[i] for i in keep]

    async def async_get_context(self, query: str, max_results: int = 10, cost_callback=None) -> str:
        return self.prompt_family.pretty_print_docs(self.select(query, max_results), max_results)


def rank_written_sections(query: str, sections: list[dict], max_results: int = 10,
                          relative_threshold: float = 0.5) -> list[str]:
    """Previously written report sections most related to ``query``.

    The keyword counterpart of ``WrittenContentCompressor``, used when no
    embedding model is available. ``sections`` are
    ``{"section_title": ..., "written_content": ...}`` dicts.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [
        (section.get("section_title", ""), chunk)
        for section in sections
        for chunk in splitter.split_text(section.get("written_content") or "")
        if chunk.strip()
    ]
    scores = bm25_scores(query, [f"{title}\n{text}" for title, text in chunks])
    best = max(scores, default=0.0)
    if best <= 0:
        return []
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [
        f"Title: {chunks[i][0]}\nContent: {chunks[i][1]}\n"
        for i in ranked if scores[i] >= relative_threshold * best
    ][:max_results]
