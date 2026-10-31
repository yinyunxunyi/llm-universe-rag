"""
app.py — 个人知识库助手网页版（对应文档 Step 6）

和第 5 节的脚本是同一套逻辑，只是把终端换成了浏览器：
    输入框 → 问题压缩（有历史时）→ 检索 → 拼 prompt → 流式输出 → 显示引用来源

启动（必须在项目根目录）：
    D:\\anconda\\envs\\llm-universe\\Scripts\\streamlit.exe run app.py

启动后终端会打印 http://localhost:8501 ，按住 Ctrl 点它就能打开。
停止：回到终端按 Ctrl+C。
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")

LLM_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
EMB_MODEL = os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3")

st.set_page_config(page_title="个人知识库助手", page_icon="📚", layout="wide")


# ---------------- 资源只初始化一次 ----------------
# Streamlit 每次交互都会重跑整个脚本。不加缓存的话，
# 每点一下都要重新加载向量库、重建模型客户端，慢到没法用。
@st.cache_resource(show_spinner="正在加载向量库…")
def get_resources(top_k: int):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        temperature=0,
        timeout=120,
    )
    emb = OpenAIEmbeddings(
        model=EMB_MODEL,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        check_embedding_ctx_length=False,
    )
    db = Chroma(persist_directory=PERSIST, embedding_function=emb)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # 问题压缩：历史以纯文本塞进 system，避免模型模仿上一轮答案的风格
    condense = (
        ChatPromptTemplate.from_messages([
            ("system", "下面是一段对话记录，以及用户最新的追问。\n"
                       "你的任务是把它改写成一个可以独立检索的完整问句。\n"
                       "规则：不要回答问题；只输出改写后的问句本身；不要任何解释、前缀或引号；"
                       "如果原句本身已经完整，就原样输出。\n\n"
                       "对话记录：\n{history_text}"),
            ("human", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    qa = (
        ChatPromptTemplate.from_messages([
            # 这一版是 A/B 实验选出来的（step7_optimize.py，变体 P2）。
            # 关键改动：判断标准从"模型的认知"改成"资料里有没有"。
            # 原来的写法是"如果你不知道答案就说不知道"，但模型对库外问题（如 BERT、
            # 王阳明）其实都"知道"，所以从不拒答，5/5 全部编造。
            ("system", "请只依据下面【资料】中的内容回答问题。\n"
                       "如果【资料】中没有相关信息，请回答\"资料中未提及该内容\"。\n"
                       "不要使用你自身的知识进行补充或推测。\n\n"
                       "【资料】\n{context}"),
            ("placeholder", "{chat_history}"),
            ("human", "问题：{input}"),
        ])
        | llm
        | StrOutputParser()
    )

    return {"llm": llm, "db": db, "retriever": retriever, "condense": condense, "qa": qa}


def history_text(history) -> str:
    if not history:
        return "（无）"
    lines = []
    for role, content in history[-6:]:
        lines.append(("用户：" if role == "human" else "助手：") + content)
    return "\n".join(lines)


def mmss(ms: int) -> str:
    """毫秒 → 分:秒。字幕类文档靠它告诉用户「这句话在视频的第几分钟」。"""
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def locate(md) -> str:
    """
    给一个切片算出「它在原文的哪个位置」。
    不同格式能提供的定位信息不一样，这里做统一：pdf 给页码，字幕给时间区间，
    纯文本类（txt/md）没有位置概念，就不显示。
    """
    if md.get("start_ms") is not None:
        return f"{mmss(md['start_ms'])}–{mmss(md['end_ms'])}"
    if md.get("page") is not None:
        return f"p{md['page']}"
    return ""


def cite(d) -> str:
    md = d.metadata
    name = os.path.basename(md.get("source", "?"))
    fmt = md.get("format", "")
    pos = locate(md)
    head = f"{name}"
    if fmt:
        head += f"  [{fmt}]"
    if pos:
        head += f"  {pos}"
    return f"{head}\n{d.page_content[:160]}"


def condense_question(res, question: str, history) -> str:
    if not history:
        return question
    out = res["condense"].invoke({
        "input": question,
        "history_text": history_text(history),
    }).strip()
    return out or question


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.subheader("设置")
    top_k = st.slider("检索片段数 top-k", 1, 8, 3,
                      help="每次从库里取几段原文。取多了上下文更全，但也更容易混入噪音。")
    show_sources = st.checkbox("显示引用片段", value=True)
    st.divider()
    st.caption(f"生成模型：`{LLM_MODEL}`")
    st.caption(f"向量模型：`{EMB_MODEL}`")
    st.caption(f"温度：`0`（固定，保证可复现）")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- 主体 ----------------
st.title("📚 个人知识库助手")
st.caption("基于 llm-universe 默认语料（南瓜书 + 提示工程教程 + 强化学习入门指南视频字幕）")

if not os.path.exists(os.path.join(PERSIST, "chroma.sqlite3")):
    st.error("找不到向量库。请先在项目根目录运行 step4b_build.py 建库。")
    st.stop()

res = get_resources(top_k)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 回放历史
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("rewritten"):
            st.caption(f"↳ 检索用的问句：{m['rewritten']}")
        if show_sources and m.get("sources"):
            with st.expander(f"引用片段（{len(m['sources'])}）"):
                for s in m["sources"]:
                    st.text(s)

# 新问题
question = st.chat_input("问点什么…（例如：什么是南瓜书）")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 历史转成 (human/ai, 内容)，最后一条是本轮问题，不算历史
    history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        rewritten = condense_question(res, question, history)
        if rewritten != question:
            st.caption(f"↳ 检索用的问句：{rewritten}")

        docs = res["retriever"].invoke(rewritten)

        placeholder = st.empty()
        buf = ""
        for chunk in res["qa"].stream({
            "input": question,
            "chat_history": history,
            "context": "\n\n".join(d.page_content for d in docs),
        }):
            buf += chunk
            placeholder.markdown(buf + "▌")     # 光标效果，让人知道还在写
        placeholder.markdown(buf)

        sources = [cite(d) for d in docs]
        if show_sources:
            with st.expander(f"引用片段（{len(sources)}）"):
                for s in sources:
                    st.text(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": buf,
        "sources": sources,
        "rewritten": rewritten if rewritten != question else "",
    })
