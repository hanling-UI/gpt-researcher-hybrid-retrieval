"""抓取结果的清洗与去重流水线。

为什么单独一个文件
==================
原来 ``scraper.py`` 里只对**每页各自**做一次正则替换，能删掉"扫码关注"
这类固定文案，但有三件事做不到：

1. **跨页面模板识别**：同一批抓取里，十几家站点会共用同一段页脚、导航、
   版权声明、相关推荐。单页看不出"这是模板"，十几页一起看，命中次数一
   统计就露馅了——这是本项目最有效的一招，因为它不需要维护站点规则。
2. **重复页面**：同一篇新闻稿被多个域名转载，正文大段重复。单页去重看不见，
   页面级去重才看得见。重复页面会同时污染 BM25 的 IDF 和向量的"语义重心"。
3. **结构性噪音**：只有符号没有字的分隔线、零宽字符、全角空格、连续空行。
   这些不改变语义，却会把 BM25 的词频和平均长度算歪。

处理顺序（顺序不能随便换）
==========================
    归一化 → 页面级去重 → 噪音/导航行过滤 → 模板行过滤 → 行级去重
    → 头尾外壳裁剪 → 收尾归一化

* **归一化必须最先**：后面所有比较都基于"同一套字符"。否则全角空格和半角
  空格会让两段本该相同的文字看起来不一样，去重直接失效。
* **页面级去重要在行级之前**：整页都重复时，没必要逐行再算一遍。
* **头尾外壳裁剪放最后**：它按"连续短行"判断，必须在噪音行、模板行、
  重复行都被清掉之后再做，否则一条噪音行就会打断"连续"这个前提。

代价
====
全部是 O(n) 的字符串操作，加一处 O(n²) 的集合求交。页面级去重对每页只取
前 4000 字符做 3-gram 签名：15 个页面约 105 次比较、每次几千个集合元素，
实测毫秒级，不会拖慢抓取。**故意不用 difflib.SequenceMatcher**——它是
O(n²) 字符级动态规划，长文档会卡到秒级，而且对中文按字符比时"改动两个词"
和"完全无关"的比值都很接近，阈值很难调。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 字符级归一化
# ---------------------------------------------------------------------------
# 零宽字符、方向控制符、软连字符：肉眼看不见，却会让"看起来相同的两段文字"
# 在 == 比较时不等，是去重失效的头号原因（常见于从 Word/PDF 复制的内容）。
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")
# Unicode 行/段分隔符，NFKC 不会把它们变成 \n，需要手动换掉。
_LINE_SEP = re.compile("[\u2028\u2029\u0085]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u3000]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n)")

# "有意义的字符"：汉字、英文字母、数字。标点和符号都不算。
_MEANINGFUL = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9]")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

# 全角【字母和数字】→ 半角：Ａ→A、１→1、ａ→a。
# 故意只映射字母和数字，**不碰标点**——这一点很关键，见 normalize_text 的说明。
_FULLWIDTH_ALNUM = {
    code: code - 0xFEE0
    for code in (
        list(range(0xFF10, 0xFF1A))  # ０-９
        + list(range(0xFF21, 0xFF3B))  # Ａ-Ｚ
        + list(range(0xFF41, 0xFF5B))  # ａ-ｚ
    )
}

# 一行里"有意义字符"的最低占比。低于它的判定为噪音行。
# 例："————————" 是 0%，"| | | |" 是 0%，"【1】" 是 1/3≈0.33（保留）。
_MIN_MEANINGFUL_RATIO = 0.3

# 导航串：用 3 个以上竖线/箭头把短词连起来，是典型的多级菜单。
# 这类行不需要任何站点规则就能识别，而且几乎不会出现在正文里。
_NAV_SEPARATORS = "|｜»›"
_NAV_MIN_SEPARATORS = 3
_NAV_MAX_LEN = 160

# 全站通用的 UI 文案：整行完全相等才丢（不做局部匹配，避免误删正文里的词）。
# 这是本文件里唯一"写死"的清单，故意保持极短——写多了就变成维护站点规则。
_UI_LINES = frozenset({
    "skip to content",
    "skip to main content",
    "回到顶部",
    "返回顶部",
    "阅读原文",
    "展开阅读全文",
    "关注我们",
})


def _looks_like_nav(line: str) -> bool:
    """是不是"首页 | 产品 | 方案 | 关于"这种导航串。"""
    if len(line) > _NAV_MAX_LEN:
        return False
    return sum(line.count(ch) for ch in _NAV_SEPARATORS) >= _NAV_MIN_SEPARATORS


def normalize_text(text: str | None) -> str:
    """把文本归一化成"同一套字符 + 每行无首尾空格 + 最多一个空行"。

    为什么**不用** ``unicodedata.normalize("NFKC")``
    -------------------------------------------------
    NFKC 是最省事的归一化，但它会把中文全角标点一起转成半角："，"→","、
    "："→":"、"（）"→"()"。这些字符会原样进入提示词，转换后中文排版变成
    "半角标点混排"，可读性下降，也让模型看到的是被改坏的中文。检索链路里
    每一段正文最终都要送给大模型，所以这里只做"语义无关"的字符级修正：

    * 全角空格/不间断空格 → 普通空格（排版噪音）
    * 全角字母数字 → 半角（同一句话的两种写法要能去重）
    * 零宽字符/方向控制符 → 删除（肉眼不可见，却让 ``==`` 比较失效）
    * 中文全角标点 **原样保留**
    """
    if not text:
        return ""

    text = _LINE_SEP.sub("\n", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_FULLWIDTH_ALNUM)
    text = text.replace("\u3000", " ").replace("\u00a0", " ")
    text = _TRAILING_SPACE.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)

    # 先按行 strip，再拼回去：把"行首缩进"这种纯排版信息去掉，
    # 但保留行与行之间的边界（后面模板行识别依赖"一行一个元素"）。
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _lines(text: str) -> list[str]:
    """拆行并丢掉空行。"""
    return [line for line in text.split("\n") if line.strip()]


def _meaningful_ratio(line: str) -> float:
    """有意义的字符（汉字/字母/数字）占该行长度的比例。"""
    if not line:
        return 0.0
    return len(_MEANINGFUL.findall(line)) / len(line)


def _is_noise_line(line: str) -> bool:
    """判断一行是不是结构性噪音（分隔线、纯符号、导航串、UI 文案等）。"""
    if not line:
        return True
    # 单字符行：中文单字可能是标题（"序"），保留；英文/数字/符号单字符丢掉。
    if len(line) == 1 and not _CJK_CHAR.search(line):
        return True
    # 纯符号行：分隔线、"| | | |"、表格边框、装饰性 emoji 串
    if _meaningful_ratio(line) < _MIN_MEANINGFUL_RATIO:
        return True
    # 同一字符重复：===== / ------ / 。。。。。。
    if len(line) >= 6 and len(set(line)) <= 2:
        return True
    # 导航串："首页 | 产品中心 | 解决方案 | 关于我们"（3 个以上分隔符）
    if _looks_like_nav(line):
        return True
    # 全站通用的 UI 文案（整行匹配，不做局部删除）
    if line.lower() in _UI_LINES:
        return True
    return False


# ---------------------------------------------------------------------------
# 模板行：跨页面统计出来的"每页都有"的行
# ---------------------------------------------------------------------------
def find_boilerplate(
    pages: list[dict],
    *,
    min_len: int = 3,
    max_len: int = 120,
    min_pages: int = 3,
    ratio: float = 0.5,
) -> frozenset[str]:
    """统计出"在同一批页面里反复出现的短行"，视为模板（导航/页脚/版权）。

    原理：正常正文段落几乎不可能在多篇不同文章里一模一样地出现，而导航、
    页脚、版权声明一定会。所以"出现页数 ≥ 阈值"就是很强的模板信号，
    不需要为每个站点写规则。

    参数含义：
        min_len/max_len: 只统计长度在区间内的行。太短（"首页"）容易误伤，
            太长（整段正文）说明是转载而非模板。
        min_pages: 至少覆盖多少页才算模板（会被"页数-1"和 page_ratio 调整）。
        ratio: 覆盖多少比例的页面算模板。最终取两者较大值。

    Returns:
        模板行集合（未归一化的行是比不上的，调用方需先 normalize）。
    """
    if len(pages) < 2:
        return frozenset()

    counts: Counter[str] = Counter()
    for page in pages:
        text = normalize_text(page.get("raw_content") or "")
        # set 去重：同一页里同一行出现 3 次只算 1 页
        counts.update({
            line for line in _lines(text) if min_len <= len(line) <= max_len
        })

    # 阈值的两个约束：
    # 1) 比例约束：至少覆盖 ratio 比例的页面（默认 50%）；
    # 2) 绝对约束：至少 min_pages 页，但**不能要求"每一页都有"**——
    #    页面级去重之后可能只剩 3 页，而页脚常常有 1 页因为抓取不全而缺失，
    #    要求"3 页全有"会让模板彻底识别不出来（第一版就踩了这个坑）。
    #    所以绝对约束的上限是 页数-1，同时不低于 2。
    page_count = len(pages)
    absolute_floor = min(min_pages, max(2, page_count - 1))
    threshold = max(absolute_floor, math.ceil(page_count * ratio))
    threshold = min(threshold, page_count)
    if threshold < 2:
        return frozenset()

    return frozenset(line for line, count in counts.items() if count >= threshold)


# ---------------------------------------------------------------------------
# 页面级去重：同一篇文章的多个转载
# ---------------------------------------------------------------------------
def _trigrams(text: str, limit: int = 0) -> set[str]:
    """字符 3-gram 集合（先去掉所有空白，让排版差异不影响相似度）。

    为什么用 3-gram 而不是分词：去重要在分词器之外独立成立。用汉字 3-gram
    对中文天然有效（不需要词典），对英文也能抓住"同一段落"的特征。
    """
    compact = re.sub(r"\s+", "", text)
    if limit:
        compact = compact[:limit]
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度 = 交集 / 并集，取值 0~1。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _can_never_match(len_a: int, len_b: int, threshold: float) -> bool:
    """长度差太大时 Jaccard 不可能达到阈值，提前跳过，省掉集合求交。

    数学依据：|A∩B| ≤ min(|A|,|B|)，|A∪B| ≥ max(|A|,|B|)，
    所以 Jaccard ≤ min/max。若 min/max < 阈值，直接淘汰。
    """
    if not len_a or not len_b:
        return True
    return min(len_a, len_b) / max(len_a, len_b) < threshold


def find_duplicate_pages(
    pages: list[dict],
    *,
    threshold: float = 0.9,
    sample_chars: int = 4000,
    min_content_chars: int = 200,
) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """找出内容高度重复的页面，保留第一次出现的那份。

    只比对每页前 ``sample_chars`` 个字符：同一篇文章的转载，开头（标题、
    导语、第一段）几乎必然相同，而全文比对要多花 10 倍时间却换不来更好的
    判断。短于 ``min_content_chars`` 的页面不参与比对——样本太小，
    相似度不可信，容易误杀。

    Returns:
        (保留的页面列表, [(被丢弃的 url, 与之重复的 url, 相似度), ...])
    """
    kept: list[dict] = []
    signatures: list[set[str]] = []
    dropped: list[tuple[str, str, float]] = []

    for page in pages:
        text = page.get("raw_content") or ""
        if len(text) < min_content_chars:
            # 太短：不参与去重判断，原样保留（由上游的长度检查负责丢弃）
            kept.append(page)
            signatures.append(_trigrams(text, sample_chars))
            continue

        signature = _trigrams(text, sample_chars)
        duplicate_of: tuple[str, float] | None = None
        for index, other in enumerate(signatures):
            if _can_never_match(len(signature), len(other), threshold):
                continue
            score = _jaccard(signature, other)
            if score >= threshold:
                duplicate_of = (kept[index].get("url") or "", score)
                break

        if duplicate_of is None:
            kept.append(page)
            signatures.append(signature)
        else:
            dropped.append((page.get("url") or "", duplicate_of[0], duplicate_of[1]))

    return kept, dropped


# ---------------------------------------------------------------------------
# 单页清洗
# ---------------------------------------------------------------------------
# 固定文案的正则。这些是历史积累的规则，便宜、直接，但**不可扩展**：
# 每个站点换一套文案就得加一条。所以它只是兜底，主力是上面的跨页面模板
# 统计（find_boilerplate）——那个不需要维护规则。
# 匹配到就**整行丢弃**，而不是删掉匹配片段：只删片段会留下残句
# （"上一篇：xxx 下一篇：yyy" 删掉前半段后剩 "：yyy"，反而制造噪音）。
_TEMPLATE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"点击查看.{0,12}详情",
        r"扫码关注.{0,12}(公众号|二维码)",
        r"版权所有.{0,16}保留",
        r"登录.{0,8}注册",
        r"上一篇.{0,24}下一篇",
        r"责任编辑[：:].{0,24}",
        r"(分享|转发)到[：:]?.{0,12}(微信|微博|QQ)",
    )
)


def _is_template_line(line: str) -> bool:
    """这一行是否命中"固定文案"正则（命中即整行丢弃）。"""
    return any(pattern.search(line) for pattern in _TEMPLATE_PATTERNS)


# 站点外壳（标题重复、作者、日期、面包屑、系列导航、页脚链接）在抓取结果里
# 集中在页面**头尾**，而且有可用的结构特征：它们是"短行 + 没有句末标点"。
# 正文行通常要么足够长，要么以句末标点收尾。用这个特征裁剪，不需要任何
# 站点规则，也不会伤到正文中段。
_MAX_META_LINES = 12    # 最多往回看多少行
_META_MIN_RUN = 3       # 至少连续几条才动手（1~2 条可能是正文短句）


def _is_content_line(line: str) -> bool:
    """能不能当成"正文行"：够长，或者"够长 + 以句末标点结尾"。

    为什么要加 40 字的下限：中文标题也常以"？"结尾
    （"BM25、向量检索与混合检索怎么选？"）。只按标点判断，标题就会被当成
    正文，页头裁剪永远启动不了。
    """
    if len(line) >= 80:
        return True
    return len(line) >= 40 and line[-1] in "。！？.!?；;"


def _trim_meta_blocks(lines: list[str]) -> list[str]:
    """裁掉开头/结尾连续的"站点外壳"行。

    只裁**连续**的一段，一旦碰到正文行就停手，所以正文中间的东西不会被动。
    """
    if len(lines) < 10:
        # 页面本身太短，行结构不可信（可能是论坛短回复），宁可不裁。
        return lines

    # 头部：从头往后数，直到遇到第一条正文行为止。
    head = 0
    for index, line in enumerate(lines[:_MAX_META_LINES]):
        if _is_content_line(line):
            head = index
            break
    else:
        # 前 12 行里一条正文行都没有：判断不可靠，不裁。
        head = 0

    if head >= _META_MIN_RUN:
        lines = lines[head:]

    # 尾部：从结尾往回数，直到遇到最后一条正文行。
    tail = len(lines)
    start = max(len(lines) - _MAX_META_LINES, 0)
    for index in range(len(lines) - 1, start - 1, -1):
        if _is_content_line(lines[index]):
            tail = index + 1
            break

    if len(lines) - tail >= _META_MIN_RUN:
        lines = lines[:tail]

    return lines


def clean_page(text: str | None, boilerplate: frozenset[str] | set[str] = frozenset()) -> str:
    """清洗单页文本：归一化 → 噪音行 → 模板行 → 行级去重 → 收尾。

    Args:
        text: 抓取到的原始正文。
        boilerplate: ``find_boilerplate()`` 的结果；单页清洗时传空集合即可。

    Returns:
        清洗后的文本，可能为空串（表示这页没有可用内容）。
    """
    text = normalize_text(text)
    if not text:
        return ""

    kept: list[str] = []
    seen: set[str] = set()
    for line in _lines(text):
        if _is_noise_line(line):
            continue
        if line in boilerplate:
            continue
        if _is_template_line(line):
            continue
        # 行级去重：同一页里完全相同的行只留第一次。
        # 只做"完全相同"的判断，不做近似——近似判断留给 chunk 级，
        # 因为在行这一层误删一句话的代价远大于收益。
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)

    # 最后裁掉页头/页尾的站点外壳（导航、作者、日期、页脚链接）。
    kept = _trim_meta_blocks(kept)

    # 丢掉整行后可能留下空行，收尾再归一化一次。
    return normalize_text("\n".join(kept))


def clean_text_noise(text: str) -> str:
    """兼容旧名字：``scraper/utils.py`` 里原来的清洗函数。

    保留这个入口是因为已经写成 ``from .utils import clean_text_noise`` 的
    调用点（含你本地的调试脚本）不该因为内部重构而报错。
    """
    return clean_page(text)


# ---------------------------------------------------------------------------
# chunk 级近重复去重
# ---------------------------------------------------------------------------
def deduplicate_chunks(chunks: list[str], threshold: float = 0.85) -> list[str]:
    """去掉内容高度相似的 chunk，保留第一次出现的。

    和原来的实现（``difflib.SequenceMatcher`` 比前 200 个字符）相比：
    * 用 3-gram Jaccard，对中文有效（SequenceMatcher 在中文上比值集中在
      0.5 附近，阈值 0.85 几乎筛不掉东西）；
    * 加长度预筛，实际比较次数远少于 O(n²)；200 个 chunk 实测约 50ms。

    Args:
        chunks: 文本切片列表，顺序即优先级（靠前的先保留）。
        threshold: Jaccard 相似度阈值，0.85 表示"八成五的 3-gram 相同"。
    """
    unique: list[str] = []
    signatures: list[set[str]] = []

    for chunk in chunks:
        text = (chunk or "").strip()
        if not text:
            continue
        signature = _trigrams(text)

        duplicate = False
        for other in signatures:
            if _can_never_match(len(signature), len(other), threshold):
                continue
            if _jaccard(signature, other) >= threshold:
                duplicate = True
                break

        if not duplicate:
            unique.append(chunk)
            signatures.append(signature)

    return unique


# ---------------------------------------------------------------------------
# 批量入口：scraper 抓完之后调这一个
# ---------------------------------------------------------------------------
def clean_pages(
    pages: list[dict],
    *,
    verbose: bool = False,
    drop_duplicate_pages: bool = True,
    dedupe_threshold: float = 0.9,
) -> list[dict]:
    """对一批抓取结果做统一清洗，就地改写 ``raw_content`` 并返回保留的页面。

    这是 ``Scraper.run()`` 的唯一调用点。分三步：
    1. 页面级去重（转载、同一 URL 的多份抓取）
    2. 用剩余页面统计模板行
    3. 逐页清洗，丢掉清洗后为空的页面

    Args:
        pages: ``[{"url":..., "title":..., "raw_content":...}, ...]``
        verbose: 是否逐页打印清洗前后长度（保留原来的调试输出习惯）。
        drop_duplicate_pages: 关掉可只做单页清洗（对比实验用）。
        dedupe_threshold: 页面级重复判定阈值。

    Returns:
        清洗后仍然有内容的页面列表（新列表，元素是原 dict，已就地更新）。
    """
    if not pages:
        return []

    stats = {
        "pages_in": len(pages),
        "duplicate_pages": 0,
        "boilerplate_lines": 0,
        "before": 0,
        "after": 0,
    }

    kept = pages
    if drop_duplicate_pages and len(pages) > 1:
        kept, dropped = find_duplicate_pages(pages, threshold=dedupe_threshold)
        stats["duplicate_pages"] = len(dropped)
        for url, duplicate_of, score in dropped:
            logger.info(
                "页面内容重复，已丢弃 %s（与 %s 相似度 %.2f）", url, duplicate_of, score
            )

    boilerplate = find_boilerplate(kept)
    stats["boilerplate_lines"] = len(boilerplate)

    cleaned_pages: list[dict] = []
    for page in kept:
        original = page.get("raw_content") or ""
        cleaned = clean_page(original, boilerplate)

        page["raw_content"] = cleaned
        stats["before"] += len(original)
        stats["after"] += len(cleaned)

        if verbose:
            print(f"\n【清洗前】原始长度: {len(original)}")
            print(f"【清洗后】去噪长度: {len(cleaned)}")

        if not cleaned:
            logger.info("清洗后无剩余内容，丢弃页面：%s", page.get("url") or "(无 url)")
            continue
        cleaned_pages.append(page)

    if stats["before"]:
        saved = 1 - stats["after"] / stats["before"]
        logger.info(
            "清洗完成：%d 页 → %d 页（重复页 %d），模板行 %d 条，字符 %d → %d（-%.1f%%）",
            stats["pages_in"],
            len(cleaned_pages),
            stats["duplicate_pages"],
            stats["boilerplate_lines"],
            stats["before"],
            stats["after"],
            saved * 100,
        )
        if verbose:
            print(
                f"\n【汇总】{stats['pages_in']} 页 → 保留 {len(cleaned_pages)} 页，"
                f"丢弃重复页 {stats['duplicate_pages']} 页，"
                f"识别模板行 {stats['boilerplate_lines']} 条，"
                f"字符 {stats['before']} → {stats['after']}（-{saved:.1%}）"
            )

    return cleaned_pages


__all__ = [
    "clean_page",
    "clean_pages",
    "clean_text_noise",
    "deduplicate_chunks",
    "find_boilerplate",
    "find_duplicate_pages",
    "normalize_text",
]
