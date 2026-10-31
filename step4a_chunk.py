"""
step4a_chunk.py — 加载 + 去重 + 清洗 + 切分（对应文档 Step 4.2 ~ 4.4）

和新版的分工：
    loaders.py    负责「一个文件 → list[Document]」（认识 7 种格式）
    本文件        负责「一个目录 → 一批切片」（去重、清洗、切分、存档）

四步：
  1) 扫描 data_base/knowledge_db 下所有支持的格式
  2) 内容去重：同一份内容如果有多种格式（如 srt/txt/json/vtt/tsv 都在），只留一份
  3) 按格式分别清洗（PDF 去断行 / MD 剥语法 / 其他只压空白）
  4) 切块 chunk_size=500 overlap=50，存 chunks.pkl

为什么存 pkl：切一次要解析 196 页 PDF，约 20 秒。
建库要反复试错，有了 pkl 就不用每次重解析。

运行：  D:\\anconda\\envs\\llm-universe\\python.exe step4a_chunk.py
"""

import collections
import os
import pickle
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import SUPPORTED, content_key, load_any

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "data_base", "knowledge_db")
PKL = os.path.join(HERE, "chunks.pkl")
CHUNK_SIZE, OVERLAP_SIZE = 500, 50

# 同一份内容有多种格式时，保留谁。数字越小越优先。
# srt 排第一：它是视频的原始字幕，时间轴最准、最贴近「原文」，而且人眼可校对。
FORMAT_PRIORITY = {".srt": 0, ".vtt": 1, ".tsv": 2, ".json": 3, ".txt": 4, ".md": 5, ".pdf": 6}


def rel(p):
    return os.path.relpath(p, KB_DIR)


# ---------- 1. 收集文件 ----------
file_paths = []
for root, _, files in os.walk(KB_DIR):
    for f in files:
        if f.lower().endswith(SUPPORTED):
            file_paths.append(os.path.join(root, f))
file_paths.sort()

print(f"[1/4] 扫描到 {len(file_paths)} 个受支持的文件：")
for p in file_paths:
    print("       ", rel(p))

skipped = [f for root, _, files in os.walk(KB_DIR) for f in files
           if not f.lower().endswith(SUPPORTED)]
if skipped:
    print(f"       另有 {len(skipped)} 个文件格式不支持，将忽略：{', '.join(skipped)}")

# ---------- 2. 内容去重 ----------
print("\n[2/4] 内容去重 ...")
groups = collections.OrderedDict()
for p in file_paths:
    groups.setdefault(content_key(p), []).append(p)

keep, dropped = [], []
for key, paths in groups.items():
    if len(paths) == 1:
        keep.append(paths[0])
        continue
    # 同一份内容的多个格式：按优先级留一个
    ordered = sorted(paths, key=lambda x: FORMAT_PRIORITY.get(os.path.splitext(x)[1].lower(), 99))
    winner, losers = ordered[0], ordered[1:]
    keep.append(winner)
    dropped.extend(losers)
    print(f"       发现同源多格式（{len(paths)} 份，内容一致）：")
    print(f"         保留 {rel(winner)}")
    for l in losers:
        print(f"         跳过 {rel(l)}")

keep.sort()
dropped.sort()
if not dropped:
    print("       没有重复内容，全部保留。")
else:
    print(f"       去重后：{len(file_paths)} 个文件 → 实际加载 {len(keep)} 个（省掉 {len(dropped)} 个重复）")

# ---------- 3. 加载 + 按格式清洗 ----------
# PDF 是按排版断行的：一句话可能被切成三行，中间塞满 \n。
# 不洗直接切块，一句话会被切碎，检索时召回的片段是残句。
PDF_NEWLINE = re.compile(r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", re.DOTALL)


def clean_pdf(text: str) -> str:
    text = PDF_NEWLINE.sub(lambda m: m.group(0).replace("\n", ""), text)
    text = text.replace("•", "").replace(" ", "")
    return text


def clean_md(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)       # 代码块整段删掉
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)             # 图片语法
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)         # 链接只留文字
    text = re.sub(r"\n{2,}", "\n", text)                    # 空行压成单行
    text = re.sub(r"[#*>`|\-]{1,}", "", text)               # 标题/强调/表格符
    return text


def clean_plain(text: str) -> str:
    """
    字幕/表格/json 抽出来的正文：本来就是干净的口语文字，没有 markdown 语法。
    唯一要处理的是空白：全角空格、多余空行。
    千万别套 clean_md —— 它会把所有 "-" 删掉，口语里的连字符会消失。
    """
    text = text.replace("\r\n", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


CLEANERS = {
    ".pdf": clean_pdf,
    ".md": clean_md,
    ".txt": clean_plain,
    ".json": clean_plain,
    ".tsv": clean_plain,
    ".srt": clean_plain,
    ".vtt": clean_plain,
}

# ---------- 页眉 / 页脚 / 水印：按"出现频率"自动识别 ----------
# 思路：页眉页脚的特征不是"写了什么"，而是"几乎每页都在同一位置重复"。
# 所以不写死字符串（换了书就失效），而是统计：只看每页的头 3 行和尾 3 行，
# 某个短行如果出现在 30% 以上的页里，就判定为版面噪音，整本删掉。
EDGE_LINES = 3            # 每页检查头几行、尾几行
NOISE_MAX_LEN = 80        # 只把短行当候选；正文长句即使重复也不动
NOISE_MIN_RATIO = 0.3     # 出现页数占比阈值
NOISE_MIN_PAGES = 5       # 总页数太少（<5）时统计不可信，直接跳过


def _edge_key(line: str):
    """把一行压成用于比对的"指纹"：去掉所有空白。太长的行不算候选（返回 None）。"""
    key = re.sub(r"\s+", "", line)
    return key if 0 < len(key) <= NOISE_MAX_LEN else None


def drop_repeated_edges(docs, label=""):
    """
    删掉一整份 PDF 里"每页都在重复"的版面噪音（页眉、页脚、水印、购书广告）。

    为什么必须在 clean_pdf 之前做：
        clean_pdf 会把"两边都不是汉字"的换行删掉，页脚会和正文黏成一行，
        到那时就分不出哪部分是页脚了。所以在原始按行排版的文本上先切干净。
    """
    if len(docs) < NOISE_MIN_PAGES:
        return docs, []

    counter = collections.Counter()
    for d in docs:
        lines = [l for l in d.page_content.splitlines() if l.strip()]
        # 同一页里重复的同一行（如每页出现两次的装饰符）只记 1 次，
        # 否则一页里的多次出现会把占比虚高
        keys = {k for k in (_edge_key(l) for l in lines[:EDGE_LINES] + lines[-EDGE_LINES:]) if k}
        counter.update(keys)

    threshold = max(3, int(len(docs) * NOISE_MIN_RATIO))
    noise = {k for k, c in counter.items() if c >= threshold}
    if not noise:
        return docs, []

    removed = 0
    for d in docs:
        kept = []
        for line in d.page_content.splitlines():
            key = _edge_key(line)
            if key and key in noise:
                removed += 1
                continue
            kept.append(line)
        d.page_content = "\n".join(kept)

    print(f"       {label} 识别到 {len(noise)} 条版面噪音（出现 ≥{threshold} 页），共删 {removed} 行：")
    for k in sorted(noise, key=len, reverse=True):
        print(f"         - {k[:60]}")
    return docs, sorted(noise)


print("\n[3/4] 加载 + 清洗 ...")
docs = []
for p in keep:
    ext = os.path.splitext(p)[1].lower()
    loaded = load_any(p)

    # PDF 先去掉页眉页脚（必须在 clean_pdf 之前，理由见函数注释）
    if ext == ".pdf":
        loaded, _ = drop_repeated_edges(loaded, rel(p))

    cleaner = CLEANERS.get(ext, clean_plain)
    kept = 0
    for d in loaded:
        d.page_content = cleaner(d.page_content)
        if d.page_content.strip():          # 清洗后变空的记录直接丢掉
            docs.append(d)
            kept += 1
    print(f"       {rel(p):46s} {ext:6s} -> {kept:4d} 条")

print(f"       合计 {len(docs)} 条原始记录")

# ---------- 4. 切分 ----------
print(f"\n[4/4] 切分 (size={CHUNK_SIZE}, overlap={OVERLAP_SIZE}) ...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE,
)
split_docs = splitter.split_documents(docs)

lens = [len(d.page_content) for d in split_docs]
print(f"       切片总数：{len(split_docs)}")
print(f"       长度：最短 {min(lens)} / 平均 {sum(lens) // len(lens)} / 最长 {max(lens)}")

with open(PKL, "wb") as f:
    pickle.dump(split_docs, f)
print(f"       已保存 -> chunks.pkl ({os.path.getsize(PKL) / 1024 / 1024:.1f} MB)")

# ---------- 来源分布（核对用） ----------
print("\n=========== 各来源贡献的切片数 ===========")
by_file = collections.Counter(os.path.basename(d.metadata.get("source", "?")) for d in split_docs)
for name, n in by_file.most_common():
    print(f"       {n:5d}  {name}")

by_fmt = collections.Counter(d.metadata.get("format", "?") for d in split_docs)
print("\n=========== 各格式贡献的切片数 ===========")
for fmt, n in by_fmt.most_common():
    print(f"       {n:5d}  .{fmt}")

# ---------- 抽样人工检查 ----------
print("\n=========== 随手抽 3 片看看像不像人话 ===========")
step = max(1, len(split_docs) // 3)
for i in range(0, min(len(split_docs), step * 3), step):
    d = split_docs[i]
    md = d.metadata
    tag = f"{os.path.basename(md.get('source', '?'))}"
    if md.get("page") is not None:
        tag += f" p{md['page']}"
    if md.get("start_ms") is not None:
        tag += f" {md['start_ms'] // 1000}s-{md['end_ms'] // 1000}s"
    print(f"\n--- #{i}  {tag} ---")
    print(d.page_content[:220].replace("\n", " | "))
