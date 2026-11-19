"""检索侧统一切词器：中文走 jieba，英文走「小写 + 去停用词 + 复数还原」。

为什么需要这个文件
==================
``context/lexical.py`` 原来用正则 ``r"\\w+"`` 切词。``\\w`` 在 Unicode 模式下
包含 CJK 汉字，``+`` 又是贪婪匹配，于是**一整串中文会被切成一个 token**：

    输入  混合检索对大模型应用有什么好处
    切出  ['混合检索对大模型应用有什么好处']      # 只有 1 个 token

查询词和正文因此永远匹配不上，BM25 在中文内容上恒为 0 分——所谓"关键词
检索"退化成"按原始顺序取前 N 段"，相关和不相关的段落得分一样。英文没有
这个问题（空格天然分词），所以这个 bug 在没有中文数据的仓库里一直没被发现。

这个模块用 jieba 修掉它，并给 BM25、chunk 去重、rerank 前粗排提供**同一个
词序列出口**，避免三处各切一套、口径不一致（那种情况下你没法解释为什么
去重阈值 0.85 在 A 处合适、在 B 处不合适）。

中文分词的四个坑（先读，少踩）
==============================
1. jieba 首次调用要加载约 5MB 词典（~0.5 秒）。所以放在函数里懒加载，
   而不是 import 时加载，避免拖慢整个服务启动。
2. ``search`` 模式会把长词再拆出短词条（"中华人民共和国" →
   中华 / 华人 / 人民 / 共和 / 共和国 / 中华人民共和国），召回更好，但
   会抬高词频；``precise`` 模式只输出最长词。检索侧默认 ``search``，
   因为 query 通常很短，宁可多召回。
3. jieba 对纯英文输入同样工作（按空白和标点切），中英混排不需要分支。
4. 没装 jieba 时**绝不能静默退回"整句一个 token"的老 bug**。这里退回
   汉字二元组（bigram）：中文照样能匹配，只是粒度粗、词表大——这是
   可解释的降级，不是无声的失效。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 停用词
# ---------------------------------------------------------------------------
# 中文虚词 + 英文虚词，两套都去掉。它们几乎不携带检索信息，却在每个 chunk
# 里高频出现：BM25 的 IDF 是 log((N-df+0.5)/(df+0.5)+1)，df 接近 N 时
# IDF 接近 0，等于白占文本长度、稀释真正关键词的分数。
_CN_STOPWORDS = frozenset("""
的 了 和 是 在 有 就 不 也 都 而 及 与 或 等 这 那 其 之 于 为 以 被 把 让 从 到 对 由 向 给
我 你 他 她 它 我们 你们 他们 她们 它们 自己 什么 怎么 怎样 如何 为什么 哪些 哪个 多少
一个 一些 一种 这个 那个 这些 那些 这样 那样 如果 因为 所以 但是 而且 然后 虽然 不过
可以 能够 需要 应该 已经 还是 只是 就是 不是 没有 很多 非常 比较 更加 以及 同时 因此
年 月 日 上 下 中 里 后 前 时 时候 目前 现在 今天 之前 之后
""".split())

_EN_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())

# 汉字区间：CJK 基本区 + 扩展 A。用来判断一个 token 是不是中文。
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# token 里至少得有一个"字"（字母/数字/汉字），否则是纯标点，丢掉。
_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# jieba 句柄。None 表示不可用（没装 / 加载失败），此时走二元组降级。
_JIEBA = None
_JIEBA_TRIED = False


def _ensure_jieba() -> None:
    """懒加载 jieba：第一次切词时才付加载词典的那 0.5 秒。

    失败不抛异常，只把 ``_JIEBA`` 置 None 交给降级路径——检索链路的可用性
    不应该被一个可选依赖绑死（与 CONTEXT_FILTER 其它模式失败后降级 keyword
    是同一个原则）。
    """
    global _JIEBA, _JIEBA_TRIED
    if _JIEBA_TRIED:
        return
    _JIEBA_TRIED = True
    try:
        import jieba

        # 默认 INFO 级别会把 "Building prefix dict..." 打到 stderr，误导用户
        # 以为程序报错；降到 WARNING。
        jieba.setLogLevel(logging.WARNING)
        jieba.initialize()
        _JIEBA = jieba
    except Exception as exc:  # ImportError / 词典损坏 / 权限问题
        logger.warning("jieba 不可用（%s），中文检索降级为二元组切词", exc)
        _JIEBA = None


def warmup() -> None:
    """可选：在服务启动时提前加载词典，避免第一条 query 慢 0.5 秒。"""
    _ensure_jieba()


def _fallback_cut(text: str) -> list[str]:
    """没有 jieba 时的降级切法：英文按词、中文按二元组（bigram）。

    二元组的好处是"混合检索"会被切成 混合 / 合检 / 检索，query 里的
    "检索"仍能命中；坏处是词表膨胀约 2 倍、同义不同字序匹配不上。
    """
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text):
        if _CJK.match(run):
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            out.append(run)
    return out


def _cut(text: str, mode: str) -> list[str]:
    """切成原始词条（还没去停用词、没做复数还原）。"""
    _ensure_jieba()
    if _JIEBA is None:
        return _fallback_cut(text)
    if mode == "precise":
        return list(_JIEBA.lcut(text))
    # search 模式：长词会额外拆出子词条，召回优先
    return list(_JIEBA.lcut_for_search(text))


def _stem_en(token: str) -> str:
    """英文轻量复数还原，让 batteries 命中 battery、costs 命中 cost。

    只处理最常见的复数形态，不做完整词形还原（那需要 nltk/spaCy 和词表，
    对 BM25 的收益远小于成本）。中文没有单复数，调用方会跳过它。
    """
    if len(token) <= 4 or token.endswith("ss"):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str, mode: str = "search") -> list[str]:
    """把文本切成检索用的词序列（已去停用词、已做英文复数还原）。

    BM25 打分、chunk 去重、rerank 前粗排都应当调用它，保证三处口径一致。

    Args:
        text: 原始文本，中英混排均可；None / 空串返回空列表。
        mode: ``"search"``（默认）长词再拆子词条，召回优先；
              ``"precise"`` 只输出最长词，词频更准，适合给长文档打分。

    Returns:
        小写词条列表，**保留重复**（BM25 需要词频 tf，不能去重）。
    """
    if not text:
        return []

    tokens: list[str] = []
    for raw in _cut(str(text).lower(), mode):
        token = raw.strip()
        # 丢掉纯空白、纯标点（"，"、"——"、"·" 之类）
        if not token or not _HAS_WORD_CHAR.search(token):
            continue
        if token in _CN_STOPWORDS or token in _EN_STOPWORDS:
            continue

        if _CJK.search(token):
            # 中文词不做复数还原。单字保留："猫""车""税""债"本身就是有效关键词。
            tokens.append(token)
            continue

        # 纯英文/数字词：单字母（a 已在停用词里）通常是变量名噪音，丢掉；
        # 纯数字（2024、3.5）保留，因为年份/版本号常常是关键检索条件。
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.append(_stem_en(token))

    return tokens
