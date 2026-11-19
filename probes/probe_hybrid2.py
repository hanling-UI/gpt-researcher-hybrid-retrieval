"""验证 hybrid.py：A 分块是否生效，B 两路分歧时 RRF 是否真的在起作用。"""
from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.context.hybrid import (
    bm25_rank,
    chunk_pages,
    rrf_fuse,
    vector_rank,
)

# ============ 检查 A：分块器到底有没有在切 ============
# 用程序生成 1391 字的文本（40 个短段拼起来），确保长度可控。
LONG = "".join(
    f"第{i}段：混合检索的要点是两路召回加融合排序，这一段用于测试分块边界。"
    for i in range(1, 41)
)
long_chunks = chunk_pages([{"url": "long", "title": "长文", "raw_content": LONG}])
print("=" * 60)
print(f"A) 长文 {len(LONG)} 字 -> 切成 {len(long_chunks)} 块")
print(f"   各块字数: {[len(c.text) for c in long_chunks]}")
print(f"   各块结尾字符: {[c.text[-1] for c in long_chunks]}")

# ============ 检查 B：两路分歧 + RRF 融合 ============
# 每页只放一段，短于 500 字，所以每页正好切成 1 块 —— 块数和页数一一对应，
# 方便下面按 title 读结果。
PAGES = [
    {
        "url": "https://a.example/glossary",
        "title": "术语表",
        # 关键词陷阱：堆满查询词，但没有实质内容。BM25 会给高分。
        "raw_content": (
            "术语表：BM25、RRF、混合检索、召回率、精确率、精排、向量检索。"
            "以上术语按拼音排序，详细定义见附录 A。"
        ),
    },
    {
        "url": "https://b.example/synonym",
        "title": "同义改写",
        # 语义正是 RRF，但一个查询词都不出现。BM25 看不见，向量看得见。
        "raw_content": (
            "把两条候选列表按名次合并，不需要比较各自的打分尺度，"
            "也不需要任何标注数据。名次越靠前的候选获得的加分越多，"
            "两条列表都提到的候选自然排在前面。"
        ),
    },
    {
        "url": "https://c.example/rrf",
        "title": "RRF 定义",
        # 两路都该给高分的那段
        "raw_content": (
            "RRF 把 BM25 和向量两路召回的结果按名次融合，公式是 1/(k+rank)，"
            "k 通常取 60。它避免了两路分数尺度不可比的问题。"
        ),
    },
    {"url": "https://d.example/weather", "title": "天气",
     "raw_content": "今天白天多云转晴，最高气温 26 度，夜间最低 18 度，东南风三级。"},
    {"url": "https://e.example/recipe", "title": "菜谱",
     "raw_content": "番茄炒蛋：两个鸡蛋打散，番茄切块，热油先炒蛋再下番茄，加盐出锅。"},
    {"url": "https://f.example/stock", "title": "股市",
     "raw_content": "上证指数今日收涨 0.8%，成交额较前一交易日放大，券商板块领涨。"},
    {"url": "https://g.example/sports", "title": "体育",
     "raw_content": "昨晚的足球比赛主队二比一获胜，下半场连入两球完成逆转。"},
    {"url": "https://h.example/travel", "title": "旅游",
     "raw_content": "去云南建议避开暑期高峰，大理和丽江之间可以坐动车，两小时左右。"},
    {"url": "https://i.example/car", "title": "汽车",
     "raw_content": "新能源车的保养项目和燃油车差别不小，主要看电池健康度和电控系统。"},
    {"url": "https://i.example/home", "title": "装修",
     "raw_content": "小户型装修建议先定家具尺寸再定水电点位，否则很容易返工。"},
]

cfg = Config()
embeddings = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

chunks = chunk_pages(PAGES)
print("=" * 60)
print(f"B) {len(PAGES)} 页 -> {len(chunks)} 块")

QUERY = "混合检索 RRF 怎么融合两路召回"

# 两路各只取前 5（总共 10 块）。如果取 20，两路都会包含全部候选，
# 就看不到"某一路漏掉了什么"，融合的价值也就无法体现。
bm25_hits = bm25_rank(QUERY, chunks, top_k=5)
vector_hits = vector_rank(QUERY, chunks, embeddings, top_k=5)
fused = rrf_fuse([bm25_hits, vector_hits], k=60, top_k=5)

def labels(hits):
    return [chunk.title for chunk, _ in hits]

print(f"\n[BM25 路 top5]  {labels(bm25_hits)}")
print(f"[向量路 top5]   {labels(vector_hits)}")
print(f"[RRF 融合 top5] {labels(fused)}")

print("\n关键对照：'同义改写'那段（没有任何查询关键词）")
print(f"  BM25 有没有捞到: {'同义改写' in labels(bm25_hits)}")
print(f"  融合后有没有出现: {'同义改写' in labels(fused)}")

print("\n关键对照：'术语表'那段（堆满关键词）")
print(f"  BM25 排名第几: {labels(bm25_hits).index('术语表') + 1 if '术语表' in labels(bm25_hits) else '未进前五'}")
print(f"  融合后排名第几: {labels(fused).index('术语表') + 1 if '术语表' in labels(fused) else '未进前五'}")

print("\n融合分数的明细（应为 1/(60+rank) 的累加）:")
for chunk, score in fused:
    print(f"  {score:.5f} | {chunk.title}")