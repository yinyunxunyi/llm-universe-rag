"""
check_env.py — 开建库之前的冒烟测试

一次性验证三条链路：
  1) .env 里的密钥能不能被读到
  2) 向量模型（BGE-M3）能不能用，维度对不对
  3) 生成模型（DeepSeek）能不能用

三段互相独立：前一段失败不会挡住后一段，方便一次看全所有问题。
在项目根目录运行：  python check_env.py
"""

import os
import sys

from dotenv import load_dotenv, find_dotenv

# find_dotenv() 从当前目录向上找 .env，所以必须在项目根目录运行
load_dotenv(find_dotenv())

KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()
URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").strip()
EMB = os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3").strip()
MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3").strip()

print("-" * 56)
print("base_url  :", URL)
print("embedding :", EMB)
print("llm       :", MODEL)
print("api_key   :", f"{KEY[:7]}...{KEY[-4:]}" if len(KEY) > 12 else "(空)")
print("-" * 56)

if not KEY.startswith("sk-"):
    sys.exit("X 没读到 SILICONFLOW_API_KEY，检查 .env 是否在项目根目录、变量名是否写对")

# ---- 0. 先列出账号里可用的模型 ----
# 放在最前面：万一下面 model id 写错报 400，这份列表就是改错的依据
try:
    import requests

    r = requests.get(f"{URL}/models",
                     headers={"Authorization": f"Bearer {KEY}"},
                     timeout=15)
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    hits = sorted(i for i in ids if "deepseek" in i.lower() or "bge-m3" in i.lower())
    if hits:
        print("账号可用（deepseek / bge-m3）:")
        for i in hits:
            print("    ", i)
    else:
        print("(接口通了，但没筛到 deepseek / bge-m3，共", len(ids), "个模型)")
    print()
except Exception as e:
    print("(跳过模型列表检查:", type(e).__name__, str(e)[:60], ")\n")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---- 1. 向量模型 ----
# check_embedding_ctx_length=False 是关键：默认会先用 tiktoken 把文本转成 token id 再发，
# 那是 OpenAI 官方接口的规矩；硅基流动只收原始字符串，漏了这行会 400。
try:
    emb = OpenAIEmbeddings(
        model=EMB,
        api_key=KEY,
        base_url=URL,
        check_embedding_ctx_length=False,
        chunk_size=32,
    )
    vec = emb.embed_query("南瓜书是什么")
    print(f"[OK] embedding 维度 = {len(vec)}   (BGE-M3 应为 1024)")
except Exception as e:
    print("[FAIL] embedding:", type(e).__name__, str(e)[:200])

# ---- 2. 生成模型 ----
try:
    llm = ChatOpenAI(model=MODEL, api_key=KEY, base_url=URL, temperature=0)
    msg = llm.invoke("用一句话介绍你自己").content
    print("[OK] LLM 回复:", msg[:60].replace("\n", " "))
except Exception as e:
    print("[FAIL] LLM:", type(e).__name__, str(e)[:200])

print("\n--- 到此结束 ---")
