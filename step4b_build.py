"""
step4b_build.py — 建 Chroma 向量库（对应文档 Step 4.5）

读 step4a 产出的 chunks.pkl，逐批向量化并写入 data_base/vector_db/chroma。

三个关键点（文档里都踩过）：
  1) 不要写 vectordb.persist()  —— chromadb 1.x 已删除该方法，1.x 自动落盘
  2) 用 langchain_chroma.Chroma，不要用 langchain_community 里那个（已废弃，不认 1.x）
  3) 只跑一次 —— 重复跑会追加数据，同一段被检索到两次

运行：  D:\\anconda\\envs\\llm-universe\\python.exe step4b_build.py
重建：  先删 data_base\\vector_db\\chroma 目录，再跑
"""

import os
import pickle
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

PKL = os.path.join(HERE, "chunks.pkl")
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")
BATCH = 64

# ---------- 前置检查 ----------
if not os.path.exists(PKL):
    sys.exit("X 找不到 chunks.pkl，先跑 step4a_chunk.py")

sqlite = os.path.join(PERSIST, "chroma.sqlite3")
if os.path.exists(sqlite):
    sys.exit(
        "X 向量库已存在：" + PERSIST + "\n"
        "  重复写入会导致同一片段被检索两次。\n"
        "  要重建请先删掉这个目录，再重新运行本脚本。"
    )

with open(PKL, "rb") as f:
    docs = pickle.load(f)
print(f"读入 {len(docs)} 个片段")

# ---------- embedding：必须与查询时逐字一致 ----------
emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    check_embedding_ctx_length=False,   # 漏了这行就 400
    chunk_size=32,                      # 每次请求送 32 条
)

# 不传 collection_name，用默认的 "langchain"，
# 这样第 5 节按文档写法 Chroma(persist_directory=...) 能直接打开同一个集合
vectordb = Chroma(embedding_function=emb, persist_directory=PERSIST)

# ---------- 分批写入 ----------
print(f"开始写入（每批 {BATCH} 条）...\n")
t0 = time.time()
for i in range(0, len(docs), BATCH):
    batch = docs[i:i + BATCH]
    vectordb.add_documents(batch)
    done = min(i + BATCH, len(docs))
    pct = done / len(docs) * 100
    print(f"  [{done:4d}/{len(docs)}]  {pct:5.1f}%   已用 {time.time() - t0:5.1f}s", flush=True)

# ---------- 验收 ----------
count = vectordb._collection.count()
print("\n" + "=" * 50)
print(f"向量库记录数：{count}")
print(f"落盘位置：{PERSIST}")
print(f"磁盘文件：{os.listdir(PERSIST)}")
print(f"总耗时：{time.time() - t0:.1f}s")

if count != len(docs):
    print("\n! 记录数与片段数不一致，可能中途失败，建议删库重跑")
else:
    print("\nOK  建库完成，下一步跑 step4c_search.py 验证检索质量")
