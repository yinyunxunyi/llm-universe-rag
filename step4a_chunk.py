"""
step4a_chunk.py — 加载 + 清洗 + 切分（对应文档 Step 4.2 ~ 4.4）

做三件事，不碰网络、不花钱：
  1) 把 data_base/knowledge_db 下的 pdf/md/txt 全读成 Document
  2) 按文件类型分别清洗（PDF 去断行，MD 剥语法符号）
  3) 按 chunk_size=500 / overlap=50 切块，存到 chunks.pkl

为什么单独存 pkl：切一次要解析 196 页 PDF，约 20 秒。
下一步建库要反复试错，有了 pkl 就不用每次都重解析。

运行：  D:\\anconda\\envs\\llm-universe\\python.exe step4a_chunk.py
"""

import os
import pickle
import re

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "data_base", "knowledge_db")
PKL = os.path.join(HERE, "chunks.pkl")
CHUNK_SIZE, OVERLAP_SIZE = 500, 50
SUPPORTED = (".pdf", ".md", ".txt")


# ---------- 1. 收集文件 ----------
file_paths = []
for root, _, files in os.walk(KB_DIR):
    for f in files:
        if f.lower().endswith(SUPPORTED):
            file_paths.append(os.path.join(root, f))
file_paths.sort()

print(f"[1/4] 扫描到 {len(file_paths)} 个文件：")
for p in file_paths:
    print("       ", os.path.relpath(p, KB_DIR))

# ---------- 2. 加载 ----------
print("\n[2/4] 加载中 ...")
raw_docs = []
for p in file_paths:
    if p.lower().endswith(".pdf"):
        loaded = PyMuPDFLoader(p).load()
    else:
        loaded = TextLoader(p, encoding="utf-8").load()
    print(f"       {os.path.basename(p):38s} -> {len(loaded):4d} 条")
    raw_docs.extend(loaded)
print(f"       合计 {len(raw_docs)} 条原始记录")

# ---------- 3. 清洗 ----------
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


print("\n[3/4] 清洗 ...")
docs = []
for d in raw_docs:
    src = d.metadata.get("source", "")
    d.page_content = clean_pdf(d.page_content) if src.lower().endswith(".pdf") else clean_md(d.page_content)
    if d.page_content.strip():          # 清洗后变空的页直接丢掉
        docs.append(d)
print(f"       清洗后剩 {len(docs)} 条（丢掉 {len(raw_docs) - len(docs)} 条空记录）")

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

# ---------- 抽样人工检查 ----------
print("\n=========== 随手抽 3 片看看像不像人话 ===========")
step = max(1, len(split_docs) // 3)
for i in range(0, min(len(split_docs), step * 3), step):
    d = split_docs[i]
    print(f"\n--- #{i}  来源: {os.path.basename(d.metadata.get('source', '?'))} ---")
    print(d.page_content[:220].replace("\n", " | "))
