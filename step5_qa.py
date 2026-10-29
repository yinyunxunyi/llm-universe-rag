"""
step5_qa.py — 单轮问答链 + 「裸模型 vs 加知识库」对照实验（对应文档 Step 5.1 ~ 5.3）

这条链在干什么：
    问题 → 向量检索 top-k → 把片段拼成 context → 塞进 prompt → DeepSeek → 答案

输出是流式的（边生成边打印）。不加流式的话，终端会盲等 30 秒，看起来像卡死。

用法：
    python step5_qa.py                 # 跑内置的对照实验
    python step5_qa.py "什么是南瓜书"   # 问自己的问题
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")
TOP_K = 3

# ---------- 5.1 接模型 ----------
# temperature=0 不是可选项：问答要的是稳定复现，不是发挥。
# timeout：不设的话网络异常时会无限等待，看起来就像卡死。
llm = ChatOpenAI(
    model=os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0,
    timeout=120,
)

# ---------- 5.2 加载库 + 检索器 ----------
embedding = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    check_embedding_ctx_length=False,
)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# ---------- 5.3 拼链 ----------
def combine_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


template = """请只依据下面【资料】中的内容回答问题。
如果【资料】中没有相关信息，请回答"资料中未提及该内容"。
不要使用你自身的知识进行补充或推测。

【资料】
{context}

问题: {input}
"""
prompt = PromptTemplate(template=template)

qa_chain = (
    RunnableParallel(
        context=retriever | RunnableLambda(combine_docs),
        input=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)


def stream_out(run, payload, head: str) -> str:
    """流式打印：run 可以是 llm，也可以是整条链"""
    print(head)
    t0 = time.time()
    first = True
    parts = []
    for chunk in run.stream(payload):
        text = getattr(chunk, "content", chunk)   # llm 吐 AIMessageChunk，链吐 str
        if not text:
            continue
        if first:
            print(f"  (首字 {time.time() - t0:.1f}s)", flush=True)
            first = False
        print(text, end="", flush=True)
        parts.append(text)
    print(f"\n  (共 {time.time() - t0:.1f}s)")
    return "".join(parts)


def ask(question: str) -> None:
    print("=" * 64)
    print(f"问题：{question}")
    print("=" * 64)

    stream_out(llm, question, "\n【裸模型，没有任何知识库】")
    stream_out(qa_chain, question, "\n【加知识库（RAG）】")

    docs = retriever.invoke(question)
    print(f"\n【这次用到的 {len(docs)} 段原文】")
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "?"))
        page = d.metadata.get("page", "-")
        print(f"  {i}. {src} p{page} :: {d.page_content[:70].replace(chr(10), ' ')}...")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        ask("什么是南瓜书？")
