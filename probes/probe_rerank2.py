"""端到端验证：分块 -> BM25 + 向量 -> RRF -> 精排 top3。"""
from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.context.hybrid import chunk_pages, hybrid_retrieve
from gpt_researcher.context.rerank import rerank_or_fallback

PAGES = [
    {"url": "https://a.example/rrf", "title": "RRF 原理", "raw_content": (
        "混合检索同时跑两路召回：BM25 做关键词匹配，向量做语义匹配。"
        "两路结果用 RRF 按名次融合，公式 1/(k+rank)，k 一般取 60。"
        "融合后再用 cross-encoder 精排，把候选压到三到五条。")},
    {"url": "https://b.example/bm25", "title": "BM25 原理", "raw_content": (
        "BM25 基于词频和逆文档频率打分，词频越高、词越稀有得分越高。"
        "它对专有名词和编号这类精确匹配很有效，但对同义改写无能为力。")},
    {"url": "https://c.example/vector", "title": "向量与精排", "raw_content": (
        "向量检索把文本编码成稠密向量，用余弦相似度衡量语义接近程度。"
        "它擅长处理同义表达，但对精确的型号编号不敏感，需要关键词路互补。")},
    {"url": "https://d.example/weather", "title": "天气", "raw_content":
        "今天白天多云转晴，最高气温 26 度，夜间最低 18 度，东南风三级。"},
    {"url": "https://e.example/recipe", "title": "菜谱", "raw_content":
        "番茄炒蛋的做法：鸡蛋打散，番茄切块，热油先炒蛋再下番茄，加盐出锅。"},
    {"url": "https://f.example/stock", "title": "股市", "raw_content":
        "上证指数今日收涨，成交额较前一交易日放大，券商板块领涨，市场情绪回暖。"},
]

QUERY = "混合检索 RRF 怎么融合两路召回"

cfg = Config()
embeddings = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

chunks = chunk_pages(PAGES)
fused = hybrid_retrieve(QUERY, PAGES, embeddings, top_k=20)

print("=" * 64)
print(f"RRF 融合候选（共 {len(fused)} 条）:")
for chunk, score in fused:
    print(f"  {score:.5f} | {chunk.title}")

final = rerank_or_fallback(QUERY, [chunk for chunk, _ in fused], top_n=3)

print("=" * 64)
print(f"精排后 top3（共 {sum(len(c.text) for c, _ in final)} 字符）:")
for chunk, score in final:
    print(f"  {score:.4f} | {chunk.title}")

titles = [chunk.title for chunk, _ in final]
print("\n垃圾页是否被踢出:")
print(f"  天气在结果里: {'天气' in titles}")
print(f"  菜谱在结果里: {'菜谱' in titles}")
print(f"  股市在结果里: {'股市' in titles}")