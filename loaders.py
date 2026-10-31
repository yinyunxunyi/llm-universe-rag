"""
loaders.py — 多格式文档加载器（「格式 → 解析函数」的路由表）

职责只有一件事：给一个文件路径，返回 list[Document]。
不做清洗，不做切分，不碰网络。

为什么要单独一个文件：
    加一种格式 = 在 LOADERS 表里加一行。流程代码（step4a）一个字都不用动。
    以前这些判断挤在 step4a 的 if/elif 里，加 4 种格式那里会变成一坨。

新增格式的做法：
    1) 写一个 load_xxx(path) -> list[Document]，必须给每个 Document 填
       metadata["source"] = path（下游靠它认文件）
    2) 在 LOADERS 里注册扩展名

依赖说明：
    全部用 Python 标准库（json / csv / re / hashlib），不装 unstructured。
    原因见 实验记录.md：装了 unstructured 会拖 spacy+numba，把 numpy 压回 1.26，
    而 numpy 1.26 在 Python 3.13 上没有预编译包，会走源码编译 C++ → 必失败。
"""

import csv
import hashlib
import json
import os
import re

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document


# ============================================================
# 1. 原有三种格式
# ============================================================

def load_pdf(path):
    """PyMuPDF 逐页解析。每页一个 Document，metadata 里自带 page（第几页）。"""
    return PyMuPDFLoader(path).load()


def load_text(path):
    """md / txt 当纯文本读。必须显式指定 utf-8，Windows 默认是 GBK，会把中文读成乱码。"""
    return TextLoader(path, encoding="utf-8").load()


# ============================================================
# 2. 时间轴类格式（字幕）的公共部分
# ============================================================

# 匹配 "00:00:01,000 --> 00:00:03,000"（srt，逗号）
#      "00:01.000 --> 00:03.000"（vtt，只有分:秒.毫秒）
TS_LINE = re.compile(
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2})(?::(\d{2}))?[.,](\d{3})"
)


def _ts_to_ms(h, m, s, ms):
    """把 时:分:秒.毫秒 换算成毫秒整数。srt 有小时位，vtt 没有，用 0 补。"""
    return ((int(h) * 60 + int(m)) * 60 + (int(s or 0))) * 1000 + int(ms)


def parse_cues(text):
    """
    把字幕全文切成 [(start_ms, end_ms, 台词), ...]。

    srt 和 vtt 的排版几乎一样，唯一区别是 vtt 开头有一行 "WEBVTT"，
    且时间码可能没有小时位。用同一个正则能吃掉两者，所以不引入 srt / webvtt 这两个包。
    """
    cues = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = TS_LINE.search(line)
        if not m:
            continue
        start = _ts_to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_ms(m.group(5), m.group(6), m.group(7), m.group(8))
        # 时间行的下一行就是台词；遇到空行或下一条时间行就停
        j = i + 1
        words = []
        while j < len(lines) and lines[j].strip() and not TS_LINE.search(lines[j]):
            words.append(lines[j].strip())
            j += 1
        cue_text = " ".join(words).strip()
        if cue_text:
            cues.append((start, end, cue_text))
    return cues


def merge_cues(cues, window_chars=400):
    """
    把逐条字幕合并成「一段」，减少碎片。

    为什么必须合并：字幕是一句话一条（这里有 372 条，平均 11 个字）。
    如果一条字幕算一个片段，检索出来的永远是残句，模型看不懂上下文。
    合并策略：按时间顺序累加，累计字数接近 window_chars 就切一段，
    每段记住「第一条的起点」和「最后一条的终点」当时间戳。

    window_chars 取 400 而不是 500：后面 step4a 还要按 500 字再切一次，
    这里留点余量，避免一段正好 500 字被切刀二次剁开。
    """
    blocks = []
    buf, start, end = [], None, None
    size = 0

    for s, e, t in cues:
        if start is None:
            start = s
        buf.append(t)
        size += len(t)
        end = e
        if size >= window_chars:
            blocks.append((start, end, "\n".join(buf)))
            buf, start, end, size = [], None, None, 0

    if buf:                                   # 收尾：最后不满一段的也要留下
        blocks.append((start, end, "\n".join(buf)))
    return blocks


def _from_blocks(path, blocks, fmt):
    """把 merge_cues 的结果转成 Document 列表，并带上时间轴 metadata。"""
    docs = []
    for start, end, text in blocks:
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path,
                "format": fmt,
                "start_ms": int(start),
                "end_ms": int(end),
            },
        ))
    return docs


def load_srt(path):
    with open(path, encoding="utf-8") as f:
        cues = parse_cues(f.read())
    return _from_blocks(path, merge_cues(cues), "srt")


def load_vtt(path):
    with open(path, encoding="utf-8") as f:
        cues = parse_cues(f.read())
    return _from_blocks(path, merge_cues(cues), "vtt")


# ============================================================
# 3. 表格 / 结构化格式
# ============================================================

def load_tsv(path):
    """
    读 start / end / text 三列的转写表。
    用 csv 模块而不是 pandas：这里只需要遍历三列，csv 是标准库、更轻。
    注意分隔符必须显式指定 \\t，否则 csv 会按逗号切，整行变成一列。
    """
    cues = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                start = int(float(row["start"]))
                end = int(float(row["end"]))
            except (KeyError, ValueError):
                continue                      # 表头不规范的行直接跳过，不要整批失败
            text = (row.get("text") or "").strip()
            if text:
                cues.append((start, end, text))
    return _from_blocks(path, merge_cues(cues), "tsv")


def load_json(path):
    """
    处理两种常见的 json 形态：
      A) {"text": "...", "segments": [{"start":1.0,"end":3.0,"text":"..."}, ...]}  ← 本项目
      B) [{"text": "..."}, ...] 或 {"content": "..."} 之类的扁平结构

    不用 LangChain 的 JSONLoader：它要求你用 jq 语法写提取路径（如 "$.segments[*]"），
    写错就是一句看不懂的报错。这里只有两种形状，用普通 dict 取值更好懂、更好改。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # A) 带时间轴的分段
    segments = data.get("segments") if isinstance(data, dict) else None
    if segments:
        cues = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            cues.append((
                int(float(seg.get("start", 0)) * 1000),   # 这个文件里 start 是「秒」
                int(float(seg.get("end", 0)) * 1000),
                text,
            ))
        return _from_blocks(path, merge_cues(cues), "json")

    # B) 只有一大段文字
    if isinstance(data, dict):
        for key in ("text", "content", "body"):
            if isinstance(data.get(key), str):
                return [Document(page_content=data[key],
                                 metadata={"source": path, "format": "json"})]

    # C) 兜底：整个 json 序列化成文本（宁可难看，也别静默丢掉文件）
    return [Document(page_content=json.dumps(data, ensure_ascii=False),
                     metadata={"source": path, "format": "json"})]


# ============================================================
# 4. 路由表
# ============================================================

LOADERS = {
    ".pdf":  load_pdf,
    ".md":   load_text,
    ".txt":  load_text,
    ".json": load_json,
    ".tsv":  load_tsv,
    ".srt":  load_srt,
    ".vtt":  load_vtt,
}

SUPPORTED = tuple(LOADERS)          # step4a 扫描文件时用这个，保证两边不会不一致


def load_any(path):
    """不认识的格式返回空列表并打印，而不是抛异常——异构目录里混着 mp4 是常态。"""
    ext = os.path.splitext(path)[1].lower()
    fn = LOADERS.get(ext)
    if fn is None:
        print(f"       跳过（未支持）：{os.path.basename(path)}")
        return []

    docs = fn(path)
    for d in docs:
        d.metadata["source"] = path     # 统一盖一遍，避免各 loader 写法不一致
        d.metadata["format"] = ext.lstrip(".")
    return docs


# ============================================================
# 5. 内容指纹（给去重用）
# ============================================================

def content_key(path):
    """
    算一个文件的「内容指纹」，用来判断两个文件是不是同一份内容。

    做法：把全文取出来 → 删掉所有空白和标点 → 算 sha1。
    为什么删空白：同一份转写稿，srt 带时间码、txt 不带，但正文字符是同一串，
    去掉空白后就完全一样，指纹也就一样。

    二进制格式（pdf/docx/mp4）没法廉价取出正文，退化成「文件字节的指纹」：
    仍能去掉完全重复的同一个文件，但发现不了「pdf 和它的 txt 版」这种跨格式重复。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".pdf", ".docx"):
        with open(path, "rb") as f:
            raw = f.read()
        return hashlib.sha1(raw).hexdigest()

    try:
        text = _plain_text(path)
    except Exception:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()

    norm = re.sub(r"[\s\W_]+", "", text)      # 去掉空白 + 标点 + 下划线，只留字母数字汉字
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _plain_text(path):
    """只取「正文」，用于算指纹。和 load_xxx 分开写，因为这里不需要 metadata。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        docs = load_json(path)
        return "\n".join(d.page_content for d in docs)
    if ext == ".tsv":
        with open(path, encoding="utf-8", newline="") as f:
            return "\n".join((row.get("text") or "") for row in csv.DictReader(f, delimiter="\t"))
    if ext in (".srt", ".vtt"):
        with open(path, encoding="utf-8") as f:
            return "".join(t for _, _, t in parse_cues(f.read()))
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()
