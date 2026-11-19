"""验证 select.py 接上了 hybrid：对比 keyword 和 hybrid 两种模式的输出。"""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.context.select import select_context

PAGES = [
    {
        "url": "https://a.example/rrf",
        "title": "RRF 原理",
        "raw_content": (
            "混合检索同时跑两路召回：BM25 做关键词匹配，向量做语义匹配。"
            "两路结果用 RRF 按名次融合，公式 1/(k+rank)，k 一般取 60。"
            "融合后再用 cross-encoder 精排，把候选压到三到五条。"
        ),
    },
    {
        "url": "https://b.example/bm25",
        "title": "BM25 原理",
        "raw_content": (
            "BM25 基于词频和逆文档频率打分，词频越高、词越稀有得分越高。"
            "它对专有名词和编号这类精确匹配很有效，但对同义改写无能为力。"
        ),
    },
    {
        "url": "https://c.example/vector",
        "title": "向量与精排",
        "raw_content": (
            "向量检索把文本编码成稠密向量，用余弦相似度衡量语义接近程度。"
            "它擅长处理同义表达，但对精确的型号编号不敏感，需要关键词路互补。"
        ),
    },
    {
        "url": "https://d.example/weather",
        "title": "天气",
        "raw_content": "今天白天多云转晴，最高气温 26 度，夜间最低 18 度，东南风三级。",
    },
    {
        "url": "https://e.example/recipe",
        "title": "菜谱",
        "raw_content": "番茄炒蛋的做法：鸡蛋打散，番茄切块，热油先炒蛋再下番茄，加盐出锅。",
    },
    {
        "url": "https://f.example/stock",
        "title": "股市",
        "raw_content": "上证指数今日收涨，成交额较前一交易日放大，券商板块领涨，市场情绪回暖。",
    },
]

QUERY = "混合检索 RRF 怎么融合两路召回"

cfg = Config()
memory = Memory(cfg.embedding_provider, cfg.embedding_model)


async def main():
    for mode in ("keyword", "hybrid"):
        # 直接改配置对象，不动 .env
        cfg.context_filter = mode
        text = await select_context(
            QUERY,
            PAGES,
            cfg,
            # max_results=5 而页面有 6 个：故意让"小内容快速返回"的捷径
            # 不成立（它要求 页面数 <= max_results），否则 hybrid 根本不会被执行。
            max_results=5,
            embeddings=lambda: memory.get_embeddings(),
        )
        print("=" * 60)
        print(f"[{mode}] 输出 {len(text)} 字符")
        print(text[:400])
        print("...")


asyncio.run(main())