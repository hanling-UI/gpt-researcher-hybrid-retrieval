"""离线回放评测：固定语料，只换排序策略，比较 5 种上下文构建方式。

为什么必须回放
==============
每次跑都重新搜索抓取，语料就变了。字符数、命中率的差异里混着
"这次运气好抓到 5 页"的影响，没法归因到排序策略。
所以第一遍把 pages 存盘，之后所有模式都吃同一份 pages。

两组命令
========
    python eval_context.py --collect    # 第一遍：查询 -> 搜索 -> 抓取 -> 存盘
    python eval_context.py              # 之后反复跑：只比较排序，不再联网抓取

成本
====
collect 只调 Tavily 和抓网页，不调大模型；replay 只调 embedding 和 rerank。
整轮不到一毛钱。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.config.config import Config
from gpt_researcher.context.select import select_context
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.retrievers import TavilySearch
from gpt_researcher.scraper import Scraper
from gpt_researcher.utils.workers import WorkerPool

RUNS_DIR = Path("eval_runs")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0"

# 5 个查询 + 每个查询的"必须命中的要点"。
# must_have 是你自己定的人工标准：只要这些词出现在最终上下文里，
# 就说明这条链路没把关键信息丢掉。第一次跑完如果发现某个词在语料里
# 根本不出现，脚本会警告你——那是标注的问题，不是模型的问题。
QUERIES = [
    {
        "id": "rrf",
        "query": "混合检索 RRF 怎么融合两路召回",
        "must_have": ["1/(k+rank)", "倒数排名", "k=60"],
        "gold": "混合检索把 BM25 与向量两路召回结果用 RRF 按名次融合，"
                "公式是 1/(k+rank)，k 通常取 60，融合后再做精排。",
    },
    {
        "id": "bm25",
        "query": "BM25 和向量检索有什么区别",
        "must_have": ["词频", "逆文档频率", "余弦相似度"],
        "gold": "BM25 依靠词频和逆文档频率做词面匹配，擅长精确关键词；"
                "向量检索用余弦相似度比较语义，擅长同义改写，两者互补。",
    },
    {
        "id": "rerank",
        "query": "rerank 重排模型在 RAG 里怎么用",
        "must_have": ["精排", "候选", "交叉编码"],
        "gold": "重排模型对少量候选段落逐对打分，把真正相关的排在前面，"
                "通常放在召回之后、送大模型之前。",
    },
    {
        "id": "chunk",
        "query": "RAG 文本分块 chunk 大小怎么定",
        "must_have": ["分块", "重叠", "512"],
        "gold": "分块大小要兼顾语义完整和检索精度，常见几百字一块并设置重叠，"
                "避免在句子中间切断。",
    },
    {
        "id": "zh",
        "query": "中文分词对关键词检索的影响",
        "must_have": ["分词", "中文", "词典"],
        "gold": "中文没有空格，必须先分词才能做词频统计，"
                "否则整句会被当作一个词，关键词检索失效。",
    },
]

# 五组对照。env 里的值会在每次运行前写进 os.environ，
# 因为 Config() 是在构造时读环境变量的，所以每组都要重新 Config()。
MODES = [
    ("none（全量）",     {"CONTEXT_FILTER": "none",     "RERANK_BACKEND": "none"}),
    ("keyword（原版）",  {"CONTEXT_FILTER": "keyword",  "RERANK_BACKEND": "none"}),
    ("embeddings",       {"CONTEXT_FILTER": "embeddings", "RERANK_BACKEND": "none"}),
    ("hybrid（无精排）", {"CONTEXT_FILTER": "hybrid",   "RERANK_BACKEND": "none"}),
    ("hybrid+rerank",    {"CONTEXT_FILTER": "hybrid",   "RERANK_BACKEND": "dashscope"}),
]


def flat(text: str) -> str:
    """去掉所有空白再比较，避免'1/(k+rank)' 和 '1/(k + rank)' 匹配不上。"""
    return re.sub(r"\s+", "", text or "")
def split_blocks(text: str) -> list[str]:
    """把 pretty_print_docs 的输出按 'Source:' 切成一段段。

    每段就是"一个候选块"的完整呈现。评测语义覆盖时必须以块为单位，
    因为最终是大模型按块读的，不是按字符串读的。
    """
    return [part.strip() for part in text.split("Source:")[1:] if part.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。和 hybrid.py 里那个同源，这里独立写一份，
    避免评测脚本依赖业务代码的私有函数。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def blocks_within_budget(text: str, budget: int = 2000) -> list[str]:
    """按顺序取块，直到累计字符超过预算。

    这样比较才公平：keyword 一次给 25 块、hybrid 只给 3 块，
    不设预算就是比"谁块多"。预算统一之后，比的是"同样的篇幅里谁的内容更好"。
    """
    kept, used = [], 0
    for block in split_blocks(text):
        if used + len(block) > budget:
            # 第一块就超预算时不能什么都不留：否则"全量"模式会被判成 0 分
            # （实测 5 个查询里有 4 个的 none 模式拿到 0.000，就是这条分支的锅）。
            # 截断到剩余预算，保证每个模式至少有内容可评。
            if not kept:
                kept.append(block[: budget - used])
            break
        kept.append(block)
        used += len(block)
    return kept

async def collect(item: dict) -> None:
    """跑一次真实搜索 + 抓取 + 清洗，把 pages 存盘。"""
    path = RUNS_DIR / f"{item['id']}.json"
    if path.exists():
        print(f"[collect] {item['id']} 已存在，跳过（想重抓就删掉这个文件）")
        return

    results = TavilySearch(query=item["query"]).search(max_results=5)
    urls = [result["href"] for result in results][:5]
    if not urls:
        print(f"[collect] {item['id']}: 搜索没有结果，跳过")
        return

    # Scraper.run 内部已经做了清洗 + 跨页去重，所以存下来的是"清洗后"的页面。
    # 后续所有模式共用这份清洗结果——清洗效果不在这次评测的变量里。
    pool = WorkerPool(max_workers=4)
    pages = await Scraper(urls, USER_AGENT, "bs", pool).run()

    RUNS_DIR.mkdir(exist_ok=True)
    payload = {
        "query": item["query"],
        "must_have": item["must_have"],
        # 标准答案也存进去：以后 gold 改了，语料不用重抓也能复现当时的评测口径
        "gold": item["gold"],
        "pages": pages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(page.get("raw_content") or "") for page in pages)
    print(f"[collect] {item['id']}: 抓 {len(urls)} 个 URL -> 保留 {len(pages)} 页，共 {total} 字符")


async def replay(item: dict, embeddings) -> list[dict]:
    """对同一批 pages 跑所有模式，返回每行的指标。"""
    path = RUNS_DIR / f"{item['id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data["pages"]
    must_have = data["must_have"]
    # 语料可能是旧版 collect 存的（没有 gold 字段），所以回退到
    # QUERIES 里的定义。data.get 而不是 data[]，就是为了兼容旧文件。
    must_have_gold = data.get("gold") or item["gold"]

    # 先体检：如果某个要点在整批语料里都找不到，说明标注有问题
    corpus = flat("".join(page.get("raw_content") or "" for page in pages))
    missing = [term for term in must_have if flat(term) not in corpus]
    if missing:
        print(f"  ⚠ 要点 {missing} 在语料里根本没出现，请换掉这几个词")

    rows = []
    # gold 向量每个查询只算一次，所有模式共用
    gold_vector = embeddings.embed_query(must_have_gold)
    for label, env in MODES:
        os.environ.update(env)
        cfg = Config()          # 每轮重建，才能读到新的 CONTEXT_FILTER

        started = time.perf_counter()
        error = ""
        text = ""
        try:
            text = await select_context(
                item["query"],
                pages,
                cfg,
                max_results=10,     # 和真实链路保持一致
                embeddings=lambda: embeddings,
            )
        except Exception as exc:  # 某一组挂了不能让整轮评测中断
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        # 计时一：关键词命中（保留，但只作参考）
        BUDGET = 2000
        body = flat(text)[:BUDGET]
        hits = [term for term in must_have if flat(term) in body]
        sources = set(re.findall(r"Source:\s*(\S+)", text))

        # 计时二：语义覆盖——同样的 2000 字符里，最接近标准答案的那一块有多像。
        # 关键词命中会惩罚"正确但换了说法"的段落（实测 Q2 精排选了标准答案段，
        # 却因为没有逐字出现"词频"被判 0 分），语义相似度才看得见同义表达。
        blocks = blocks_within_budget(text, BUDGET)
        semantic = 0.0
        if blocks:
            try:
                block_vectors = embeddings.embed_documents(blocks)
                semantic = max(cosine(gold_vector, v) for v in block_vectors)
            except Exception as exc:
                error = error or f"语义评分失败：{exc}"

        rows.append({
            "mode": label,
            "chars": len(text),
            "semantic": semantic,
            "hits": len(hits),
            "total": len(must_have),
            "sources": len(sources),
            "seconds": elapsed,
            "error": error,
        })
    return rows


def print_table(query: str, rows: list[dict]) -> None:
    print("=" * 78)
    print(f"查询：{query}")
    print(f"  {'模式':<18}{'字符数':>8}{'要点命中':>10}{'语义覆盖':>10}{'来源数':>8}{'耗时':>9}")
    for row in rows:
        hit = f"{row['hits']}/{row['total']}"
        print(f"  {row['mode']:<18}{row['chars']:>8}{hit:>10}"
              f"{row['semantic']:>10.3f}{row['sources']:>8}{row['seconds']:>8.2f}s")
        if row["error"]:
            print(f"      ✗ {row['error']}")


async def main(do_collect: bool) -> None:
    RUNS_DIR.mkdir(exist_ok=True)

    # 关掉"小内容快速返回"的捷径。
    # 否则小语料会直接原样返回，五种模式的输出一模一样，评测就没意义了。
    # 这是评测专用设定，不影响你正常跑报告。
    os.environ["COMPRESSION_THRESHOLD"] = "0"

    if do_collect:
        for item in QUERIES:
            await collect(item)
        print("\n语料已存到 eval_runs/，以后直接跑 `python eval_context.py` 就行\n")

    existing = [item for item in QUERIES if (RUNS_DIR / f"{item['id']}.json").exists()]
    if not existing:
        print("没有语料。先跑一次：python eval_context.py --collect")
        return

    cfg = Config()
    embeddings = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

    all_rows: list[dict] = []
    for item in existing:
        rows = await replay(item, embeddings)
        print_table(item["query"], rows)
        all_rows.extend(rows)

    print("=" * 78)
    print("汇总（所有查询平均）")
    print(f"  {'模式':<18}{'平均字符':>10}{'要点命中率':>12}{'平均语义':>10}{'平均耗时':>10}")
    for label, _ in MODES:
        subset = [row for row in all_rows if row["mode"] == label]
        if not subset:
            continue
        chars = sum(row["chars"] for row in subset) / len(subset)
        hits = sum(row["hits"] for row in subset)
        total = sum(row["total"] for row in subset)
        seconds = sum(row["seconds"] for row in subset) / len(subset)
        semantic_avg = sum(row["semantic"] for row in subset) / len(subset)
        rate = hits / total if total else 0
        print(f"  {label:<18}{chars:>10.0f}{rate:>11.1%}{semantic_avg:>10.3f}{seconds:>9.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="第一遍：抓取并保存语料")
    asyncio.run(main(parser.parse_args().collect))