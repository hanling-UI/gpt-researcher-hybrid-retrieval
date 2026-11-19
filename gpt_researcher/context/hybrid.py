"""混合检索（Hybrid Search）第一块：分块 + BM25 关键词召回。

这一版故意只做两件事：
    1. chunk_pages()  把抓回来的页面切成检索单元（Chunk）
    2. bm25_rank()    用 BM25 做关键词那一路召回

**故意不 import 任何 embedding 相关的东西**：这一版没有网络请求，
纯本地计算，跑起来快、失败原因单一。向量路和 RRF 下一张卡再加。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .tokenize import tokenize

# 中文友好分隔符。
# 默认的 ["\n\n", "\n", " ", ""] 在中文长文上会一路退到最后的 ""（按字符硬切），
# 把词和句子拦腰砍断；补上中文句末/句中标点后，切点会优先落在句子边界上。
_CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 分块为什么用 500 而不是仓库原来的 1000：见 chunk_pages 的注释。
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 80

# 短于这个长度的碎片没有检索价值（多半是"相关阅读""（完）"这类残留）。
_MIN_CHUNK_CHARS = 30


@dataclass
class Chunk:
    """一个检索单元：一段正文 + 它来自哪一页。

    用 dataclass 而不是 dict，是为了让后面的代码写 chunk.text 而不是
    chunk["text"]——拼错属性名会立刻报错，拼错字典键只会静默返回 None。
    """

    text: str
    url: str
    title: str
    chunk_id: int

    def as_document(self) -> Document:
        """转成 LangChain 的 Document，复用 prompt_family.pretty_print_docs()。

        pretty_print_docs 只认 metadata 里的 source 和 title 两个键，
        少给一个就会打印出 "Source: None"。
        """
        return Document(
            page_content=self.text,
            metadata={"source": self.url, "title": self.title},
        )


def chunk_pages(
    pages: list[dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """把抓取结果切成检索单元。

    chunk_size=500 字符的理由：后面要送 bge-reranker 精排，它的输入上限是
    512 个 token，而中文基本 1 个字 = 1 个 token。chunk 超过 500 字，
    rerank 会被**静默截断**——你打分的只是段落的前半截，后半截根本没参与
    排序，而且这个过程不会报错。这种坑只能靠预先算好尺寸来避免。

    overlap=80 是为了让"跨句子的答案"不被切断：相邻两段共享 80 个字，
    即使答案正好压在切点上，也至少有一块能拿到完整信息。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_CHUNK_SEPARATORS,
        # keep_separator="end"：把"。"留在前一段末尾，而不是让它跑到下一段
        # 开头变成"。。这是……"这种别扭的开头。
        keep_separator="end",
    )

    chunks: list[Chunk] = []
    for page in pages:
        text = (page.get("raw_content") or "").strip()
        if not text:
            continue
        # 抓取结果里的字段名不统一：爬虫给 url，向量库给 source，两个都认。
        url = page.get("url") or page.get("source") or ""
        title = page.get("title") or ""

        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < _MIN_CHUNK_CHARS:
                continue
            # chunk_id 用全局递增，保证跨页唯一，排序时也靠它做稳定排序。
            chunks.append(Chunk(text=piece, url=url, title=title, chunk_id=len(chunks)))

    return chunks


def bm25_rank(
    query: str,
    chunks: list[Chunk],
    top_k: int = 20,
) -> list[tuple[Chunk, float]]:
    """用 BM25 给 chunks 打分，返回分数最高的 top_k 个。

    BM25 的直觉：一个词在这一段里出现得越多（tf 高），同时在**这一批段落**里
    越稀有（idf 高），它就越能代表这段的主题。

    注意 idf 是在"本次这批 chunks"内部现算的，不是全互联网语料——所以每次
    查询都要重新建索引。好处是快（几百个 chunk 只要几毫秒），坏处是同一段
    文字在不同批次的分数没有可比性，**不能拿绝对值当阈值**用。
    """
    if not chunks:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        # query 全是停用词（"怎么办""有哪些"），没有可匹配的关键词。
        # 这一路就该空手而回，交给向量那一路兜底——这是双路召回存在的意义。
        return []

    corpus = [tokenize(chunk.text) for chunk in chunks]
    if not any(corpus):
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    # 只保留有正分的 chunk。
    # BM25 对"一个查询词都没命中"的文档给 0 分，如果不过滤，这些 0 分文档
    # 会按原始顺序被 top_k 凑数带进候选列表，把真正有信号的候选挤出去。
    # 过滤之后，关键词那一路没命中就老实交白卷，由向量那一路兜底。
    ranked = sorted(
        (i for i in range(len(chunks)) if float(scores[i]) > 0),
        key=lambda i: (-float(scores[i]), chunks[i].chunk_id),
    )
    return [(chunks[i], float(scores[i])) for i in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 第二路：向量召回
# ---------------------------------------------------------------------------
# 阿里云 embedding 单次最多 20 条。这个数字不是我猜的——报错原文就写着
# "batch size is invalid, it should not be larger than 20"。
# 之前保守写成 10，代价是请求数翻倍：85 个块要发 9 次，每次约 1.5 秒。
_EMBED_BATCH_SIZE = 20

# 同时发几批。串行发 5 批要 7.5 秒，并发只要 1.5 秒左右。
# 不要调太大，阿里云有 QPS 限流，超了会 429；实测 5 比较稳。
_EMBED_CONCURRENCY = 5


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：两个向量夹角的余弦，越接近 1 越相似。

    为什么不直接算欧氏距离：embedding 的"长度"没有语义意义，只有"方向"有。
    余弦只看向量方向，天然忽略长度差异，所以是 embedding 检索的标配。

    为什么不用 numpy：几十到几百个 chunk、1024 维，纯 Python 求和大约几十毫秒，
    还省掉一次类型转换。等语料规模到几万条再换 numpy 或向量库。
    """
    dot = sum(x * y for x, y in zip(a, b))       # 点积
    norm_a = sum(x * x for x in a) ** 0.5        # a 的模长
    norm_b = sum(y * y for y in b) ** 0.5        # b 的模长
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_documents_in_batches(embeddings, texts: list[str]) -> list[list[float]]:
    """分批并发调用 embed_documents。

    为什么要并发：一次 embedding 请求的往返约 1.5 秒，而这个耗时几乎是纯等待。
    85 个块按 20 条一批是 5 批：串行 7.5 秒，并发 1.5 秒。

    这是整条链路里最大的性能瓶颈——比 BM25、RRF、精排加起来都大，
    因为它跟算法复杂度无关，纯粹是在等网络。

    httpx.Client 本身线程安全，所以直接丢线程池就行，不需要把
    vector_rank 改成 async（调用方本来就在 asyncio.to_thread 里跑）。
    """
    batches = [
        texts[start : start + _EMBED_BATCH_SIZE]
        for start in range(0, len(texts), _EMBED_BATCH_SIZE)
    ]
    if not batches:
        return []
    if len(batches) == 1:
        return embeddings.embed_documents(batches[0])

    # pool.map 保序：返回顺序和 batches 一致，所以下面可以直接 flatten。
    with ThreadPoolExecutor(max_workers=min(_EMBED_CONCURRENCY, len(batches))) as pool:
        batch_vectors = list(pool.map(embeddings.embed_documents, batches))

    return [vector for batch in batch_vectors for vector in batch]


def vector_rank(
    query: str,
    chunks: list[Chunk],
    embeddings,
    top_k: int = 20,
) -> list[tuple[Chunk, float]]:
    """用向量相似度给 chunks 打分，返回最相似的 top_k 个。

    embeddings 传 None 时返回空列表——这不是异常，是**降级**：
    整个混合检索在 embedding 不可用时应当退化成纯 BM25，而不是崩掉。
    这一条和 select.py 里"任何模式失败都退回 keyword"是同一个设计原则。
    """
    if not chunks or embeddings is None:
        return []

    # 一次批量编码所有 chunk（内部已分批），比逐条 embed_query 快一个数量级。
    # 这是上一卡探测出"单条 2.15s、批量 3 条 0.67s"带来的直接结论。
    doc_vectors = _embed_documents_in_batches(embeddings, [c.text for c in chunks])
    query_vector = embeddings.embed_query(query)

    scores = [_cosine(query_vector, vector) for vector in doc_vectors]

    ranked = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], chunks[i].chunk_id),
    )
    return [(chunks[i], float(scores[i])) for i in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 融合：手写 RRF
# ---------------------------------------------------------------------------
def rrf_fuse(
    ranked_lists: list[list[tuple[Chunk, float]]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion：只用「名次」融合多路召回结果。

    公式：score(d) = Σ 每一条命中的路 1 / (k + rank(d))

    为什么用 RRF 而不是"把分数加权相加"：BM25 的分数可以到 20 多，余弦相似度
    只有 0~1，两者的尺度完全没有可比性。想相加就必须先校准（归一化、z-score、
    学习权重），而校准本身就需要标注数据。RRF 绕开了这个问题：它只问"这条路
    把你排第几"，不问"这条路给你打了多少分"。

    k 的作用：
        k 越大，1/(k+rank) 随 rank 衰减越平缓 → 更信任"两路都提过"的共识；
        k 越小，头部优势越明显 → 更信任"某一路的排名第一"。
        k=60 是原论文的经验值，实践中 40~80 都正常。
    """
    fused: dict[int, float] = {}          # chunk_id -> 融合分数
    by_id: dict[int, Chunk] = {}          # chunk_id -> Chunk 本体

    for ranked in ranked_lists:
        # enumerate 从 1 开始：公式里的 rank 是名次，第一名 rank=1。
        # 如果从 0 开始，第一名会变成 1/k，整体分数失真。
        for rank, (chunk, _score) in enumerate(ranked, start=1):
            key = chunk.chunk_id
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            by_id[key] = chunk

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(by_id[key], score) for key, score in ordered[:top_k]]


# ---------------------------------------------------------------------------
# 总入口：两路召回 -> RRF 融合
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    query: str,
    pages: list[dict],
    embeddings,
    *,
    bm25_top_k: int = 20,
    vector_top_k: int = 20,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[tuple[Chunk, float]]:
    """一条 query 走完"分块 -> 双路召回 -> RRF 融合"，返回融合后的候选。

    两路各取 20 条是刻意的：单路取太少，融合就失去意义（两路重合度高）；
    取太多，融合结果里会塞满单路才认可的边角内容。20 是常见起点。
    """
    chunks = chunk_pages(pages)
    if not chunks:
        return []

    bm25_hits = bm25_rank(query, chunks, top_k=bm25_top_k)
    vector_hits = vector_rank(query, chunks, embeddings, top_k=vector_top_k)

    return rrf_fuse([bm25_hits, vector_hits], k=rrf_k, top_k=top_k)


__all__ = [
    "Chunk",
    "chunk_pages",
    "bm25_rank",
    "vector_rank",
    "rrf_fuse",
    "hybrid_retrieve",
]