"""核心检索链路的离线单元测试：分词 / BM25 / RRF / 切块 / 清洗去重。

全部不联网、不调用大模型，秒级跑完，可在 CI 与本地稳定复现。

覆盖的两类风险：

1. **回归**——``tokenize`` 之前用正则 ``\\w+``，中文整句被当成一个 token，
   BM25 对中文恒为 0 分。这类 bug 不会报错，只会让"关键词检索"静默失效，
   所以必须有断言盯住。
2. **公式**——RRF 的 ``1/(k+rank)`` 如果 rank 从 0 开始、或误用原始分数，
   排序看起来"差不多正确"，但分数是错的。这里用解析解做精确断言。
"""

import math

import pytest

from gpt_researcher.context.hybrid import (
    Chunk,
    bm25_rank,
    chunk_pages,
    rrf_fuse,
)
from gpt_researcher.context.tokenize import tokenize
from gpt_researcher.scraper.cleaning import (
    clean_page,
    deduplicate_chunks,
    find_boilerplate,
    normalize_text,
)


def make_chunk(chunk_id: int, text: str | None = None) -> Chunk:
    return Chunk(
        text=text if text is not None else f"chunk-{chunk_id}",
        url=f"https://example.com/{chunk_id}",
        title=f"title-{chunk_id}",
        chunk_id=chunk_id,
    )


# ---------------------------------------------------------------------------
# 分词：中文必须被切成多个词条
# ---------------------------------------------------------------------------


def test_tokenize_splits_chinese_into_multiple_tokens():
    """中文整句不能被当成一个 token —— 这是 BM25 恒 0 分的根因。"""
    tokens = tokenize("混合检索对大模型应用有什么好处")

    assert len(tokens) >= 3, f"中文被切成了 {tokens}，说明又退回了整句一个 token"
    assert any("检索" in token for token in tokens), tokens


def test_tokenize_drops_chinese_stopwords():
    tokens = tokenize("这个报告的结论是什么")

    assert "的" not in tokens
    assert "什么" not in tokens


def test_tokenize_stems_english_plurals():
    """复数还原：batteries -> battery，costs -> cost。"""
    assert tokenize("batteries") == ["battery"]
    assert tokenize("costs") == ["cost"]


@pytest.mark.parametrize("value", ["", None, "   ", "\n\n"])
def test_tokenize_handles_empty_input(value):
    assert tokenize(value) == []


def test_tokenize_keeps_duplicates_for_term_frequency():
    """BM25 需要 tf，所以同一个词重复出现时必须保留，不能去重。"""
    tokens = tokenize("检索 检索 检索")

    assert tokens.count("检索") == 3


# ---------------------------------------------------------------------------
# RRF：分数必须等于 1/(k+rank) 的和，且与原始分数无关
# ---------------------------------------------------------------------------


def test_rrf_fuse_matches_formula():
    first = make_chunk(1)
    second = make_chunk(2)

    fused = rrf_fuse(
        [
            [(first, 9.9), (second, 0.1)],  # 关键词路
            [(second, 0.9)],                # 向量路
        ],
        k=60,
    )
    scores = {chunk.chunk_id: score for chunk, score in fused}

    # rank 从 1 开始：第一名 1/(k+1)
    assert math.isclose(scores[1], 1 / 61, rel_tol=1e-12)
    # 两路都命中的第二名：1/(k+2) + 1/(k+1)
    assert math.isclose(scores[2], 1 / 62 + 1 / 61, rel_tol=1e-12)
    # 两路共识的 chunk 应当排在只有单路支持的 chunk 前面
    assert [chunk.chunk_id for chunk, _ in fused] == [2, 1]


def test_rrf_ignores_raw_scores():
    """RRF 只看名次：把 BM25 的分数放大 1e6 倍不应改变排序。"""
    first = make_chunk(1)
    second = make_chunk(2)

    huge_first = rrf_fuse([[(first, 1e6), (second, 1e-6)]], k=60)
    huge_second = rrf_fuse([[(first, 1e-6), (second, 1e6)]], k=60)

    assert [chunk.chunk_id for chunk, _ in huge_first] == [1, 2]
    assert [chunk.chunk_id for chunk, _ in huge_second] == [1, 2]


def test_rrf_accumulates_same_chunk_across_lists():
    shared = make_chunk(5)

    fused = rrf_fuse([[(shared, 1.0)], [(shared, 2.0)]], k=60)

    assert len(fused) == 1
    assert math.isclose(fused[0][1], 2 / 61, rel_tol=1e-12)


def test_rrf_respects_top_k():
    ranked = [(make_chunk(i), 1.0) for i in range(5)]

    assert len(rrf_fuse([ranked], k=60, top_k=3)) == 3


# ---------------------------------------------------------------------------
# 切块：尺寸、来源信息、空页处理
# ---------------------------------------------------------------------------


def test_chunk_pages_respects_size_and_tracks_source():
    pages = [
        {
            "url": "https://example.com/a",
            "title": "示例页面",
            "raw_content": "混合检索" * 200,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)

    assert chunks
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert all(chunk.url == "https://example.com/a" for chunk in chunks)
    assert all(chunk.title == "示例页面" for chunk in chunks)
    # chunk_id 全局递增且唯一
    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_chunk_pages_skips_empty_content():
    assert chunk_pages([{"url": "u", "raw_content": "   "}]) == []


def test_chunk_pages_accepts_source_field_as_url():
    """抓取结果字段名不统一：url 缺失时回退到 source。"""
    pages = [{"source": "https://example.com/from-vdb", "raw_content": "混合检索" * 100}]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)

    assert chunks
    assert all(chunk.url == "https://example.com/from-vdb" for chunk in chunks)


# ---------------------------------------------------------------------------
# BM25：只保留真正命中的 chunk
# ---------------------------------------------------------------------------


def test_bm25_keeps_only_matching_chunks():
    # 语料不能只有 2 篇：BM25Okapi 在 N=2 时，只出现在 1 篇里的词
    # idf = log(1.5) - log(1.5) = 0，所有分数都是 0，看起来像"没命中"。
    chunks = [
        Chunk(text="混合检索把 BM25 与向量召回融合", url="u1", title="t", chunk_id=0),
        Chunk(text="今天天气不错适合出门散步", url="u2", title="t", chunk_id=1),
        Chunk(text="数据库选型要考虑读写比例与一致性", url="u3", title="t", chunk_id=2),
        Chunk(text="前端性能优化的重点是减少重排", url="u4", title="t", chunk_id=3),
        Chunk(text="分布式系统的幂等设计很关键", url="u5", title="t", chunk_id=4),
        Chunk(text="缓存穿透可以用布隆过滤器缓解", url="u6", title="t", chunk_id=5),
    ]

    ranked = bm25_rank("混合检索", chunks, top_k=5)

    assert ranked, "关键词命中却召回为空"
    assert ranked[0][0].chunk_id == 0
    assert ranked[0][1] > 0
    # 未命中的 chunk 不应以 0 分混进候选列表
    assert all(chunk.chunk_id != 1 for chunk, _ in ranked)


def test_bm25_returns_empty_for_stopword_only_query():
    chunks = [make_chunk(0, "混合检索")]

    assert bm25_rank("的 了 和 是", chunks) == []


# ---------------------------------------------------------------------------
# 清洗：归一化保留中文标点、模板行识别、chunk 去重
# ---------------------------------------------------------------------------


def test_normalize_keeps_chinese_punctuation():
    """不能用 NFKC：它会把全角中文标点转成半角，破坏中文排版。"""
    text = normalize_text("第一句，第二句；第三句（补充）。")

    for mark in "，；（）。":
        assert mark in text, f"中文标点 {mark} 被转换或丢弃了"


def test_normalize_collapses_blank_lines_and_spaces():
    assert normalize_text("A   B\n\n\n\nC  ") == "A B\n\nC"


def test_find_boilerplate_detects_shared_navigation():
    nav = "首页 产品 文档 关于我们"
    pages = [{"raw_content": f"{nav}\n第{i}篇文章的正文内容"} for i in range(3)]

    found = find_boilerplate(pages)

    assert nav in found


def test_clean_page_removes_boilerplate_lines():
    nav = "首页 产品 文档 关于我们"
    text = f"{nav}\n这是正文第一段，内容足够长以通过内容判定。\n这是正文第二段，继续补充信息。"

    cleaned = clean_page(text, frozenset({nav}))

    assert nav not in cleaned
    assert "正文第一段" in cleaned


def test_deduplicate_chunks_removes_near_duplicates():
    base = "混合检索通过 RRF 融合 BM25 与向量召回结果，再送入重排模型截断到 Top3。"
    near_duplicate = base + "（转载版本）"
    distinct = "数据清洗负责去掉导航、页脚与跨页重复内容，降低噪音。"

    result = deduplicate_chunks([base, near_duplicate, distinct])

    assert len(result) == 2, result
    assert result[0] == base
    assert distinct in result
