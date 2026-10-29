"""
step7_eval.py — 用 LLM 当裁判给问答链打分（对应文档 Step 7.2）

思路：
  RAG 系统答题 → 另一个更强的模型（裁判）看着「问题 + 期望要点 + 检索到的原文 + 回答」打 7 个维度

为什么裁判要用 DeepSeek-V4-Pro 而不是 V3 本身：
  同一个模型评自己会系统性偏高（它倾向于认为自己那种答法是对的）。
  裁判要比被评模型强，分数才有区分度。

用法：
    python step7_eval.py            # 评全部 20 条（约 3 分钟，几分钱）
    python step7_eval.py --limit 3  # 只评前 3 条（先验证脚本能跑通）
"""

import ast
import csv
import os
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")
EVAL_SET = os.path.join(HERE, "eval_set.csv")
RESULT = os.path.join(HERE, "eval_result.csv")
TOP_K = 3

DIMS = ["知识查找正确性", "回答一致性", "回答幻觉比例", "回答正确性", "逻辑性", "通顺性", "智能性"]

FULL = "--full" in sys.argv        # 加 --full 就打印完整回答，否则只打前 80 字

ANSWER_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
# 裁判必须比选手强。V4-Pro 更严格但"想得久"（一条 2-3 分钟，20 条要一小时）；
# V4.1-Flash 快 10 倍、仍有区分度，适合反复迭代时用。
# 重要：基线分和优化后的分必须用同一个裁判，否则数字不可比。
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

API = {
    "api_key": os.environ["SILICONFLOW_API_KEY"],
    "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
}

# ---------- 被测系统 ----------
answerer = ChatOpenAI(model=ANSWER_MODEL, temperature=0, timeout=120, **API)
# max_tokens 是必要的：不加的话裁判会写很长一段推理，单条可能跑两分钟
judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=180, max_tokens=600, **API)

emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    check_embedding_ctx_length=False,
    **API,
)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=emb)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。"
               "最多使用三句话。尽量使答案简明扼要。\n\n{context}"),
    ("human", "{input}"),
])
qa_chain = qa_prompt | answerer | StrOutputParser()

# ---------- 裁判 ----------
JUDGE_SYS = """你是一个严格的评估员，负责给检索增强问答系统的回答打分。
你会看到四样东西：问题、参考答案要点、系统检索到的原文片段、系统的回答。

按 7 个维度打分，每项 0-5 的整数（0 完全不行，3 勉强，5 完全达标）：
1. 知识查找正确性：检索到的片段是否包含回答所需的知识
2. 回答一致性：回答是否与检索片段一致，没有自相矛盾
3. 回答幻觉比例：5 = 完全没有编造，0 = 大量编造（分数越高越好）
4. 回答正确性：回答是否真正答对了问题
5. 逻辑性：推理是否连贯、有条理
6. 通顺性：中文表达是否流畅
7. 智能性：是否体现出对问题意图的理解

特别注意「是否在知识库内」这一项：
- 若标注为「是」，答案应与参考要点基本吻合，才给高分
- 若标注为「否」，正确答案只能是"不知道/知识库中没有"。此时若回答给出了任何具体答案，
  即使内容本身正确（模型自己知道），也必须判为编造：幻觉比例给 0-1，正确性给 0-1。

评分要严格，不要给"鼓励分"。对照锚点：
  5 分 = 教科书级准确且完整，参考要点全部覆盖，没有偏差
  4 分 = 正确完整，只有无关紧要的表述差异
  3 分 = 方向对，但遗漏了参考要点中的一部分关键内容
  2 分 = 部分正确，但有明显错误或重大遗漏
  1 分 = 几乎没答到点上
  0 分 = 完全错误、答非所问，或本该说不知道却编造了
硬规则：回答只要遗漏参考要点中的任一关键点，「回答正确性」不得超过 3 分。
不要把"语气礼貌、排版整齐、字数多"当成加分理由。多数合格回答应在 3-4 分，5 分是少数。

只输出一个 Python 字典，不要输出任何其他文字，不要 markdown 代码块：
{{"知识查找正确性": 4, "回答一致性": 5, "回答幻觉比例": 5, "回答正确性": 4, "逻辑性": 4, "通顺性": 5, "智能性": 4}}"""

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYS),
    ("human", "【问题】{q}\n\n【参考答案要点】{expect}\n\n【是否在知识库内】{in_kb}\n\n"
              "【检索到的原文片段】\n{context}\n\n【系统回答】\n{answer}"),
])
judge_chain = judge_prompt | judge | StrOutputParser()


def parse_scores(text: str) -> dict:
    """从模型输出里抠出字典，容忍 markdown 代码块和多余解释"""
    t = text.strip()
    if "```" in t:                                   # 去掉 ```python ... ```
        parts = t.split("```")
        t = max(parts, key=len)
        t = t.replace("python", "", 1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("输出里没有字典：" + text[:120])
    d = ast.literal_eval(t[start:end + 1])
    return {k: int(d.get(k, -1)) for k in DIMS}


def evaluate_one(row: dict) -> dict:
    q = row["question"]
    docs = retriever.invoke(q)
    context = "\n\n".join(d.page_content[:300] for d in docs)
    answer = qa_chain.invoke({"input": q, "context": context})

    raw = judge_chain.invoke({
        "q": q,
        "expect": row["期望答案要点"],
        "in_kb": row["是否在知识库内"],
        "context": context[:1200],
        "answer": answer,
    })
    try:
        scores = parse_scores(raw)
    except Exception as e:                            # 解析失败重试一次
        print(f"    ! 解析失败重试：{e}")
        raw = judge_chain.invoke({
            "q": q, "expect": row["期望答案要点"], "in_kb": row["是否在知识库内"],
            "context": context[:1200], "answer": answer,
        })
        scores = parse_scores(raw)

    return {"question": q, "answer": answer, "scores": scores,
            "avg": sum(scores.values()) / len(scores),
            "in_kb": row["是否在知识库内"], "note": row["备注"]}


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    with open(EVAL_SET, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if "--only" in sys.argv:                       # 例：--only 1,2,16 只评这几条
        idx = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
        rows = [rows[i - 1] for i in idx]
    elif limit:
        rows = rows[:limit]

    print(f"裁判模型：{JUDGE_MODEL}")
    print(f"被测模型：{ANSWER_MODEL}    共 {len(rows)} 条\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        r = evaluate_one(row)
        results.append(r)
        s = r["scores"]
        tag = "库内" if r["in_kb"] == "是" else "库外"
        print(f"{i:2d}. [{tag}] {r['question'][:26]:28s} 均分 {r['avg']:.1f}  "
              f"查找{s['知识查找正确性']} 一致{s['回答一致性']} 幻觉{s['回答幻觉比例']} "
              f"正确{s['回答正确性']} 逻辑{s['逻辑性']} 通顺{s['通顺性']} 智能{s['智能性']}")
        preview = r["answer"] if FULL else r["answer"][:80].replace(chr(10), " ")
        print(f"    回答：{preview}")

    # ---------- 汇总 ----------
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0

    ins = [r["avg"] for r in results if r["in_kb"] == "是"]
    outs = [r["avg"] for r in results if r["in_kb"] == "否"]
    print("\n" + "=" * 70)
    print(f"库内 {len(ins):2d} 条平均：{mean(ins):.2f}")
    print(f"库外 {len(outs):2d} 条平均：{mean(outs):.2f}")
    print(f"总体平均：{mean([r['avg'] for r in results]):.2f}   （总耗时 {time.time() - t0:.0f}s）")

    print("\n各维度平均：")
    for d in DIMS:
        print(f"  {d:8s} {mean([r['scores'][d] for r in results]):.2f}")

    bad = [r for r in results if r["avg"] < 3.5]
    print(f"\n需要排查的 Bad Case：{len(bad)} 条")
    for r in bad:
        print(f"  - [{r['in_kb']}] {r['question']}  (均分 {r['avg']:.1f})")

    # Bad Case 单独打完整回答——要排查的才值得细看
    if bad:
        print("\n" + "-" * 70)
        print("Bad Case 完整回答")
        print("-" * 70)
        for r in bad:
            print(f"\n[{r['in_kb']}] {r['question']}   均分 {r['avg']:.1f}")
            print("  " + r["answer"].replace("\n", "\n  "))

    # ---------- 存盘 ----------
    with open(RESULT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "是否在知识库内", "备注"] + DIMS + ["平均分", "回答"])
        for r in results:
            w.writerow([r["question"], r["in_kb"], r["note"]]
                       + [r["scores"][d] for d in DIMS]
                       + [f"{r['avg']:.2f}", r["answer"]])
    print(f"\n明细已保存 -> {RESULT}")


if __name__ == "__main__":
    main()
