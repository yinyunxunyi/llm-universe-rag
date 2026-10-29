"""
step4c_search.py — 检索验证（对应文档 Step 4.6）

建完库先别急着接大模型。先确认"检索本身是准的"：
召回片段跟问题无关的话，后面 prompt 写得再漂亮也救不回来。

用法：
    python step4c_search.py                      # 用默认问题
    python step4c_search.py "什么是查全率"        # 自定义问题
"""

import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")

query = sys.argv[1] if len(sys.argv) > 1 else "什么是大语言模型"

# 这里的配置必须和 step4b 建库时逐字一致：
# 模型不同 / 漏了 check_embedding_ctx_length，都会让查询向量和库里的向量不在同一个空间，
# 结果是不报错、但检索全乱——RAG 最隐蔽的翻车点。
emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    check_embedding_ctx_length=False,
)

vectordb = Chroma(persist_directory=PERSIST, embedding_function=emb)
print(f"库内记录数：{vectordb._collection.count()}")
print(f"查询：{query}\n")

print("=" * 60)
print("相似度检索 top-3（数字越小越相近）")
print("=" * 60)
for i, (doc, score) in enumerate(vectordb.similarity_search_with_score(query, k=3), 1):
    src = os.path.basename(doc.metadata.get("source", "?"))
    page = doc.metadata.get("page", "-")
    print(f"\n--- 结果 {i} | 距离 {score:.4f} | {src} p{page} ---")
    print(doc.page_content[:200].replace("\n", " "))

print("\n" + "=" * 60)
print("MMR 检索 top-3（牺牲一点相关性换多样性，避免三条说同一件事）")
print("=" * 60)
for i, doc in enumerate(vectordb.max_marginal_relevance_search(query, k=3), 1):
    src = os.path.basename(doc.metadata.get("source", "?"))
    print(f"\n--- 结果 {i} | {src} ---")
    print(doc.page_content[:200].replace("\n", " "))
