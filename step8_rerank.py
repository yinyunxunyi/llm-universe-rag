"""
step8_rerank.py — 两段式检索：向量粗排 top-10 → rerank 精排 top-3（对应文档 Step 7.4）

为什么需要它：
  向量检索（bi-encoder）把问题和文档分开编码，编码时互相看不见，
  所以它只能做"大致相关"的判断，top-3 里经常混进封面页、无关公式。

  rerank（cross-encoder）把 [问题 + 文档] 拼起来一起过模型，
  能在词级别比对，判断准得多，但慢，所以只对少量候选做。

  架构：向量 top-10（求"别漏"）→ rerank top-3（求"别错"）
  召回和精度分开优化，这是 RAG 的标准做法。

用法：
    python step8_rerank.py              # 全部 20 条
    python step8_rerank.py --only 12    # 只跑某条，看排序变化
"""

import ast
import csv
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
PERSIST = os.path.join(HERE, "data_base", "vector_db", "chroma")
EVAL_SET = os.path.join(HERE, "eval_set.csv")
OUT = os.path.join(HERE, "rerank_ab.csv")

TOP_K = 3          # 最终给 LLM 几段
CANDIDATES = 10    # 向量先取几段交给 rerank

DIMS = ["知识查找正确性", "回答一致性", "回答幻觉比例", "回答正确性", "逻辑性", "通顺性", "智能性"]

API = {
    "api_key": os.environ["SILICONFLOW_API_KEY"],
    "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
}
ANSWER_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
RERANK_MODEL = os.environ.get("SF_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ---------- 与 step7 相同的 prompt（P2 变体）----------
QA_SYS = """请只依据下面【资料】中的内容回答问题。
如果【资料】中没有相关信息，请回答"资料中未提及该内容"。
不要使用你自身的知识进行补充或推测。

【资料】
{context}"""

answerer = ChatOpenAI(model=ANSWER_MODEL, temperature=0, timeout=120, **API)
judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=180, max_tokens=600, **API)

qa_chain = (
    ChatPromptTemplate.from_messages([("system", QA_SYS), ("human", "问题：{input}")])
    | answerer | StrOutputParser()
)

emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    check_embedding_ctx_length=False,
    **API,
)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=emb)
retriever3 = vectordb.as_retriever(search_kwargs={"k": TOP_K})
retriever10 = vectordb.as_retriever(search_kwargs={"k": CANDIDATES})

JUDGE_SYS = """你是一个严格的评估员，负责给检索增强问答系统的回答打分。
你会看到：问题、参考答案要点、系统检索到的原文片段、系统的回答。

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
  即使内容本身正确，也必须判为编造：幻觉比例给 0-1，正确性给 0-1。

评分要严格，不要给"鼓励分"。对照锚点：
  5 分 = 教科书级准确且完整，参考要点全部覆盖，没有偏差
  4 分 = 正确完整，只有无关紧要的表述差异
  3 分 = 方向对，但遗漏了参考要点中的一部分关键内容
  2 分 = 部分正确，但有明显错误或重大遗漏
  1 分 = 几乎没答到点上
  0 分 = 完全错误、答非所问，或本该说不知道却编造了
硬规则：回答只要遗漏参考要点中的任一关键点，「回答正确性」不得超过 3 分。

只输出一个 Python 字典，不要输出任何其他文字，不要 markdown 代码块：
{{"知识查找正确性": 4, "回答一致性": 5, "回答幻觉比例": 5, "回答正确性": 4, "逻辑性": 4, "通顺性": 5, "智能性": 4}}"""

judge_chain = (
    ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYS),
        ("human", "【问题】{q}\n\n【参考答案要点】{expect}\n\n【是否在知识库内】{in_kb}\n\n"
                  "【检索到的原文片段】\n{context}\n\n【系统回答】\n{answer}"),
    ]) | judge | StrOutputParser()
)


def api_rerank(query: str, docs: list, top_n: int = TOP_K):
    """调硅基流动 rerank 接口，返回 [(doc, score), ...] 按分数降序"""
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [d.page_content[:1500] for d in docs],
        "top_n": top_n,
        "return_documents": False,
    }
    r = requests.post(
        f"{API['base_url'].rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {API['api_key']}",
                 "Content-Type": "application/json"},
        json=payload, timeout=60,
    )
    r.raise_for_status()
    return [(docs[item["index"]], item["relevance_score"]) for item in r.json()["results"]]


def parse_scores(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        t = max(t.split("```"), key=len).replace("python", "", 1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("没有字典：" + text[:100])
    d = ast.literal_eval(t[s:e + 1])
    return {k: int(d.get(k, -1)) for k in DIMS}


def grade(row, answer, context) -> dict:
    for attempt in range(2):
        try:
            raw = judge_chain.invoke({
                "q": row["question"], "expect": row["期望答案要点"],
                "in_kb": row["是否在知识库内"], "context": context[:1200], "answer": answer})
            return parse_scores(raw)
        except Exception as e:
            if attempt:
                print("      ! 评分失败", type(e).__name__, str(e)[:60])
                return {k: -1 for k in DIMS}


def short(doc) -> str:
    name = os.path.basename(doc.metadata.get("source", "?"))
    return f"{name} p{doc.metadata.get('page', '-')}"


def load_baseline():
    """复用 prompt_ab.csv 里已跑过的 P2（纯向量 top-3）分数，避免重复花钱。
    只有同一份验证集、同一个裁判下，复用才有意义。"""
    path = os.path.join(HERE, "prompt_ab.csv")
    base = {}
    if not os.path.exists(path):
        return base
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["变体"] != "P2_材料边界":
                continue
            base[row["question"]] = {
                "avg": float(row["平均分"]),
                "find": int(row["知识查找正确性"]),
                "src": "",
                "answer": row["回答"],
            }
    return base


def main():
    with open(EVAL_SET, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if "--only" in sys.argv:
        idx = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
        rows = [rows[i - 1] for i in idx]

    baseline = load_baseline()
    only_rerank = bool(baseline) and "--both" not in sys.argv

    print(f"粗排：向量 top-{CANDIDATES} → 精排：{RERANK_MODEL} top-{TOP_K}")
    print(f"被测：{ANSWER_MODEL}   裁判：{JUDGE_MODEL}   题目 {len(rows)} 条")
    print("基线来源：" + ("prompt_ab.csv 里的 P2 分数（复用，不重跑）" if only_rerank
                       else "现场重跑") + "\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        print(f"[{i}/{len(rows)}] {q}  ({row['是否在知识库内']})")
        r = {"q": q, "in_kb": row["是否在知识库内"]}

        # ---- A. 基线 ----
        if only_rerank and q in baseline:
            r.update(A_avg=baseline[q]["avg"], A_find=baseline[q]["find"],
                     A_src="(复用 prompt_ab.csv)", A_answer=baseline[q]["answer"])
        else:
            vec3 = retriever3.invoke(q)
            ctx_a = "\n\n".join(d.page_content[:300] for d in vec3)
            ans_a = qa_chain.invoke({"input": q, "context": ctx_a})
            sc_a = grade(row, ans_a, ctx_a)
            r.update(A_avg=sum(sc_a.values()) / len(sc_a), A_find=sc_a["知识查找正确性"],
                     A_src="; ".join(short(d) for d in vec3), A_answer=ans_a)

        # ---- B. rerank ----
        cand = retriever10.invoke(q)
        ranked = api_rerank(q, cand, TOP_K)
        ctx_b = "\n\n".join(d.page_content[:300] for d, _ in ranked)
        ans_b = qa_chain.invoke({"input": q, "context": ctx_b})
        sc_b = grade(row, ans_b, ctx_b)
        r.update(B_avg=sum(sc_b.values()) / len(sc_b), B_find=sc_b["知识查找正确性"],
                 B_src="; ".join(f"{short(d)}({s:.3f})" for d, s in ranked), B_answer=ans_b)

        print(f"    纯向量 {r['A_avg']:4.1f} (查找{r['A_find']})   "
              f"rerank {r['B_avg']:4.1f} (查找{r['B_find']})")
        print(f"      rerank top3: {r['B_src']}")
        results.append(r)

    # ---------- 汇总 ----------
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0

    print("\n" + "=" * 66)
    print(f"{'方案':<14}{'库内':>8}{'库外':>8}{'总体':>8}{'查找正确性':>12}")
    print("-" * 66)
    for label, key, fkey in [("纯向量 top-3", "A_avg", "A_find"), ("rerank top-3", "B_avg", "B_find")]:
        ins = [r[key] for r in results if r["in_kb"] == "是"]
        outs = [r[key] for r in results if r["in_kb"] == "否"]
        print(f"{label:<14}{mean(ins):>8.2f}{mean(outs):>8.2f}"
              f"{mean([r[key] for r in results]):>8.2f}{mean([r[fkey] for r in results]):>12.2f}")
    print(f"\n总耗时 {time.time() - t0:.0f}s")

    better = [r for r in results if r["B_avg"] > r["A_avg"] + 0.05]
    worse = [r for r in results if r["B_avg"] < r["A_avg"] - 0.05]
    print(f"rerank 变好的：{len(better)} 条   变差的：{len(worse)} 条")
    for r in better:
        print(f"  + {r['q'][:24]:26s} {r['A_avg']:.1f} → {r['B_avg']:.1f}")
    for r in worse:
        print(f"  - {r['q'][:24]:26s} {r['A_avg']:.1f} → {r['B_avg']:.1f}")

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "是否在知识库内", "纯向量均分", "rerank均分",
                    "纯向量_查找正确性", "rerank_查找正确性",
                    "纯向量_来源", "rerank_来源", "纯向量_回答", "rerank_回答"])
        for r in results:
            w.writerow([r["q"], r["in_kb"], f"{r['A_avg']:.2f}", f"{r['B_avg']:.2f}",
                        r["A_find"], r["B_find"], r["A_src"], r["B_src"],
                        r["A_answer"], r["B_answer"]])
    print(f"\n明细已保存 -> {OUT}")


if __name__ == "__main__":
    main()
