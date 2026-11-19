"""探测脚本：验证阿里云 embedding 能不能调通、维度多少、批量耗时多少。

这个文件不参与项目逻辑，只是体检用，跑完可以删掉。
"""
import os
import time

from dotenv import load_dotenv

# 必须在 import gpt_researcher 之前读 .env：
# Config 是在构造时读 os.environ 的，晚一步读进来它就看不到新配置。
load_dotenv()

from gpt_researcher.config.config import Config        # noqa: E402
from gpt_researcher.memory.embeddings import Memory    # noqa: E402


def cos(a, b):
    """余弦相似度：两个向量夹角的余弦，取值 -1~1，越接近 1 越相似。

    检索里"向量相似度"就是它。自己写一遍是为了不依赖 numpy 的版本差异。
    """
    dot = sum(x * y for x, y in zip(a, b))              # 点积
    norm_a = sum(x * x for x in a) ** 0.5               # a 的模长
    norm_b = sum(y * y for y in b) ** 0.5               # b 的模长
    return dot / (norm_a * norm_b)


def main():
    # 1) 先确认 Config 解析出了什么（这一步能看出 .env 有没有生效）
    cfg = Config()
    print(f"provider = {cfg.embedding_provider}")
    print(f"model    = {cfg.embedding_model}")
    print(f"base_url = {os.environ.get('OPENAI_BASE_URL')}")
    print(f"api_key  = {'已设置' if os.environ.get('OPENAI_API_KEY') else '缺失'}")

    # 2) 造 embedding 客户端（这一步不会发请求）
    embeddings = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

    # 3) 单条 query：检索时用户提问走这条路径
    started = time.perf_counter()
    vector = embeddings.embed_query("混合检索怎么融合 BM25 和向量召回")
    elapsed = time.perf_counter() - started
    print(f"\n单条 query: 维度={len(vector)} 耗时={elapsed:.2f}s")
    print("前 3 个分量:", [round(x, 5) for x in vector[:3]])

    # 4) 批量文档：建索引时 chunk 走这条。批量比逐条循环快很多，
    #    因为一次 HTTP 请求能带多条文本（阿里云单次最多 10 条左右）。
    docs = [
        "中文分词对 BM25 打分的影响很大",
        "RRF 只用排名信息融合两路召回",
        "rerank 只对少量候选做精排",
    ]
    started = time.perf_counter()
    vectors = embeddings.embed_documents(docs)
    elapsed = time.perf_counter() - started
    print(f"\n批量 {len(docs)} 条: {len(vectors)}x{len(vectors[0])} 耗时={elapsed:.2f}s")

    # 5) 相似度自检：前两句都跟检索相关，第三句也相关但角度不同。
    #    如果三条之间相似度全都在 0.99 以上，说明模型没在区分语义，要警惕。
    print("\n相似度自检:")
    print("  [0] vs [1] =", round(cos(vectors[0], vectors[1]), 4))
    print("  [0] vs [2] =", round(cos(vectors[0], vectors[2]), 4))
    print("  [1] vs [2] =", round(cos(vectors[1], vectors[2]), 4))


if __name__ == "__main__":
    main()