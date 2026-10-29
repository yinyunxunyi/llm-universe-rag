"""
step5_chat.py — 多轮对话 + 问题压缩（对应文档 Step 5.4）

单轮问答是：问题 → 检索 → 生成。
多轮多了个坎：用户第二句说"它是什么""有什么不同"，这个"它"拿去检索是查不到的。

所以要先用一次 LLM 把问题补全成"能检索的完整问句"：
    "南瓜书跟它有什么关系？"  →  "南瓜书和西瓜书是什么关系？"
这一步叫「问题压缩 / query rewriting」。多花的这一次调用是值得的。

用法：
    python step5_chat.py          # 跑内置的两轮演示
    python step5_chat.py -i       # 交互模式，随便聊（exit 退出）
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")
TOP_K = 3

llm = ChatOpenAI(
    model=os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    temperature=0,
    timeout=120,
)

embedding = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    check_embedding_ctx_length=False,
)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# ---------- 问题压缩 ----------
# 关键设计：历史不能以「对话消息」的形式塞进 prompt。
# 如果上一条是 assistant 的回复，模型会顺着那个风格"接着回答"，
# 而不是执行"改写问题"的指令——实测会把答案当成改写结果吐出来。
# 所以把历史降级成 system 里的一段普通文本，模型就不会把它当对话上下文。
def history_to_text(history) -> str:
    if not history:
        return "（无）"
    lines = []
    for role, content in history[-6:]:          # 只留最近 3 轮，防止 prompt 越来越长
        lines.append(("用户：" if role == "human" else "助手：") + content)
    return "\n".join(lines)


condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "下面是一段对话记录，以及用户最新的追问。\n"
               "你的任务是把它改写成一个可以独立检索的完整问句。\n"
               "规则：不要回答问题；只输出改写后的问句本身；不要任何解释、前缀或引号；"
               "如果原句本身已经完整，就原样输出。\n\n"
               "对话记录：\n{history_text}"),
    ("human", "{input}"),
])

condense_chain = (
    RunnableLambda(lambda x: {"input": x["input"],
                              "history_text": history_to_text(x.get("chat_history"))})
    | condense_prompt
    | llm
    | StrOutputParser()
)

# 有历史 → 先压缩再检索；没历史 → 直接拿原问题检索（省一次调用）
retrieve_docs = RunnableBranch(
    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
    condense_chain | retriever,
)

# ---------- 生成 ----------
qa_prompt = ChatPromptTemplate.from_messages([
    # 与 app.py 保持同一版 prompt（A/B 实验的 P2 变体）
    ("system", "请只依据下面【资料】中的内容回答问题。\n"
               "如果【资料】中没有相关信息，请回答\"资料中未提及该内容\"。\n"
               "不要使用你自身的知识进行补充或推测。\n\n"
               "【资料】\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "问题：{input}"),
])

qa_chain = (
    RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(d.page_content for d in x["context"])
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)


def stream_out(run, payload) -> str:
    t0 = time.time()
    first = True
    parts = []
    for chunk in run.stream(payload):
        text = getattr(chunk, "content", chunk)
        if not text:
            continue
        if first:
            print(f"AI ({time.time() - t0:.1f}s 起)：", end="", flush=True)
            first = False
        print(text, end="", flush=True)
        parts.append(text)
    print()
    return "".join(parts)


def turn(question: str, history: list, echo: bool = True) -> str:
    if echo:
        print(f"\n你：{question}")

    if history:
        rewritten = condense_chain.invoke(
            {"input": question, "chat_history": history}
        ).strip()
        if rewritten != question:
            print(f"  [压缩后的问题] {rewritten}")
    else:
        rewritten = question
        print("  [首轮无历史，不压缩]")

    docs = retriever.invoke(rewritten)
    answer = stream_out(qa_chain, {
        "input": question,
        "chat_history": history,
        "context": docs,
    })

    print("  [检索到] " + " | ".join(
        f"{os.path.basename(d.metadata.get('source', '?'))} p{d.metadata.get('page', '-')}"
        for d in docs
    ))

    history.append(("human", question))
    history.append(("ai", answer))
    return answer


if __name__ == "__main__":
    if "-i" in sys.argv:
        history = []
        print("交互模式，输入 exit 退出。")
        while True:
            try:
                q = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", "q", ""):
                break
            turn(q, history, echo=False)
    else:
        history = []
        print("=" * 64)
        print("第一轮：没有历史，直接检索")
        print("=" * 64)
        turn("西瓜书是什么？", history)
        print("\n" + "=" * 64)
        print("第二轮：问句里有「它」，必须靠历史补全才能检索")
        print("=" * 64)
        turn("南瓜书跟它有什么关系？", history)
