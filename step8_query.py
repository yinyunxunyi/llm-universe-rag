"""
step8_query.py — 多查询召回 / query 改写（对应文档 Step 7.4）

问题：用户用的词和语料的词不是一套。
    你问 "precision 和 recall 有啥区别"，语料里写的是 "查准率、查全率"。
    向量模型对术语差异的容忍度有限，直接检索会偏离。

做法：
    1) 让 LLM 把问题改写成 3 个"语料友好的"中文查询
       （英文→中文术语、口语→书面、同义词/上位词）
    2) 原问题 + 3 个改写，各自检索
    3) 用 RRF（倒数排名融合）合并——只看名次，不看分数
       因为不同查询的相似度分数不在同一尺度上，不可直接相加

验证集用 hard_eval_set.csv（同一批答案，换成"措辞不匹配"的问法）。

用法：
    python step8_query.py            # 全部 5 条
    python step8_query.py --only 1,3 # 只跑某几条
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
HARD_SET = os.path.join(HERE, "hard_eval_set.csv")
OUT = os.path.join(HERE, "query_ab.csv")

TOP_K = 3
PER_QUERY_K = 5      # 每个查询先各取 5 段，再融合
RRF_K = 60           # RRF 平滑常数，原论文经验值

DIMS = ["知识查找正确性", "回答一致性", "回答幻觉比例", "回答正确性", "逻辑性", "通顺性", "智能性"]

API = {
    "api_key": os.environ["SILICONFLOW_API_KEY"],
    "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
}
ANSWER_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

QA_SYS = """请只依据下面【资料】中的内容回答问题。
如果【资料】中没有相关信息，请回答"资料中未提及该内容"。
不要使用你自身的知识进行补充或推测。

【资料】
{context}"""

llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0, timeout=120, **API)
judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=180, max_tokens=600, **API)

qa_chain = (ChatPromptTemplate.from_messages(
    [("system", QA_SYS), ("human", "问题：{input}")]) | llm | StrOutputParser())

# 改写链：关键是"翻译成语料的说法"，不是"换个问法"
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是检索查询改写器。把用户的问题改写成 3 个用于检索的中文查询。规则：\n"
                   "1. 含英文术语或缩写时，译成中文标准术语（precision→查准率，SVM→支持向量机，"
                   "recall→查全率，ROC→ROC曲线）\n"
                   "2. 口语表达改成教材里的书面表述\n"
                   "3. 三个查询角度略微不同（同义词、上位词、具体化）\n"
                   "只输出三行查询，每行一个，不要编号、不要解释、不要标点结尾。"),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()
)

emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    check_embedding_ctx_length=False, **API)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=emb)
retriever = vectordb.as_retriever(search_kwargs={"k": PER_QUERY_K})

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

若回答给出了任何资料外的具体内容，即使内容本身正确，也必须判为编造。
评分要严格，不要给"鼓励分"：5 分是教科书级完美，多数合格回答在 3-4 分。
硬规则：回答只要遗漏参考要点中的任一关键点，「回答正确性」不得超过 3 分。

只输出一个 Python 字典，不要输出任何其他文字，不要 markdown 代码块：
{{"知识查找正确性": 4, "回答一致性": 5, "回答幻觉比例": 5, "回答正确性": 4, "逻辑性": 4, "通顺性": 5, "智能性": 4}}"""

judge_chain = (ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYS),
    ("human", "【问题】{q}\n\n【参考答案要点】{expect}\n\n【是否在知识库内】{in_kb}\n\n"
              "【检索到的原文片段】\n{context}\n\n【系统回答】\n{answer}"),
]) | judge | StrOutputParser())


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


def rrf_merge(list_of_docs, k: int = RRF_K, top_n: int = TOP_K):
    """倒数排名融合：score = Σ 1/(k + rank)，只用名次不用分数"""
    scores, pool = {}, {}
    for docs in list_of_docs:
        for rank, d in enumerate(docs, 1):
            key = d.page_content[:80]          # 用文本前缀做去重键
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            pool[key] = d
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [pool[key] for key, _ in ranked], [(pool[key], score) for key, score in ranked]


def short(d) -> str:
    return f"{os.path.basename(d.metadata.get('source', '?'))} p{d.metadata.get('page', '-')}"


def main():
    with open(HARD_SET, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if "--only" in sys.argv:
        idx = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
        rows = [rows[i - 1] for i in idx]

    print(f"被测：{ANSWER_MODEL}   裁判：{JUDGE_MODEL}   难问法 {len(rows)} 条\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        print(f"[{i}/{len(rows)}] {q}")

        # 改写
        raw = rewrite_chain.invoke({"input": q})
        rewrites = [x.strip(" -·.。") for x in raw.strip().splitlines() if x.strip()][:3]
        print(f"    改写: {' | '.join(rewrites)}")

        # ---- A. 只用原问题 ----
        docs_a = retriever.invoke(q)[:TOP_K]
        ctx_a = "\n\n".join(d.page_content[:300] for d in docs_a)
        ans_a = qa_chain.invoke({"input": q, "context": ctx_a})
        sc_a = grade(row, ans_a, ctx_a)

        # ---- B. 原问题 + 3 个改写，RRF 融合 ----
        lists = [retriever.invoke(x) for x in [q] + rewrites]
        docs_b, detail = rrf_merge(lists)
        ctx_b = "\n\n".join(d.page_content[:300] for d in docs_b)
        ans_b = qa_chain.invoke({"input": q, "context": ctx_b})
        sc_b = grade(row, ans_b, ctx_b)

        avg_a, avg_b = sum(sc_a.values()) / len(sc_a), sum(sc_b.values()) / len(sc_b)
        print(f"    仅原问题 {avg_a:4.1f} (查找{sc_a['知识查找正确性']})  "
              f"改写+RRF {avg_b:4.1f} (查找{sc_b['知识查找正确性']})")
        print(f"      原问题 top3: {', '.join(short(d) for d in docs_a)}")
        print(f"      融合后 top3: {', '.join(f'{short(d)}({s:.4f})' for d, s in detail)}")

        results.append({"q": q, "A_avg": avg_a, "B_avg": avg_b,
                        "A_find": sc_a["知识查找正确性"], "B_find": sc_b["知识查找正确性"],
                        "rewrites": " | ".join(rewrites),
                        "A_src": "; ".join(short(d) for d in docs_a),
                        "B_src": "; ".join(short(d) for d, _ in detail),
                        "A_answer": ans_a, "B_answer": ans_b})

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0

    print("\n" + "=" * 60)
    print(f"{'方案':<18}{'均分':>8}{'查找正确性':>12}")
    print("-" * 60)
    for label, k, f in [("仅原问题", "A_avg", "A_find"), ("改写 + RRF 融合", "B_avg", "B_find")]:
        print(f"{label:<18}{mean([r[k] for r in results]):>8.2f}{mean([r[f] for r in results]):>12.2f}")
    print(f"\n总耗时 {time.time() - t0:.0f}s")

    for r in results:
        d = r["B_avg"] - r["A_avg"]
        flag = "+" if d > 0.05 else ("-" if d < -0.05 else "=")
        print(f"  {flag} {r['q'][:26]:28s} {r['A_avg']:.1f} → {r['B_avg']:.1f}")

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "改写查询", "仅原问题_均分", "改写RRF_均分",
                    "仅原问题_查找", "改写RRF_查找", "仅原问题_来源", "改写RRF_来源",
                    "仅原问题_回答", "改写RRF_回答"])
        for r in results:
            w.writerow([r["q"], r["rewrites"], f"{r['A_avg']:.2f}", f"{r['B_avg']:.2f}",
                        r["A_find"], r["B_find"], r["A_src"], r["B_src"],
                        r["A_answer"], r["B_answer"]])
    print(f"\n明细已保存 -> {OUT}")


if __name__ == "__main__":
    main()
