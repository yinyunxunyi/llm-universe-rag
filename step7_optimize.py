"""
step7_optimize.py — prompt A/B 实验（对应文档 Step 7.3 优化生成）

做一件事：同一份验证集、同一次检索结果、同一个裁判，只换 prompt，看分数怎么动。
检索结果只取一次并复用，是为了控制变量——否则分不清提升来自 prompt 还是检索波动。

用法：
    python step7_optimize.py --only 16,17,18,19,20        # 只跑库外 5 条（快速筛选）
    python step7_optimize.py                              # 全部 20 条（最终验证）
    python step7_optimize.py --only 1,16 --prompts P1,P2   # 只比这两个变体
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
OUT = os.path.join(HERE, "prompt_ab.csv")
TOP_K = 3

DIMS = ["知识查找正确性", "回答一致性", "回答幻觉比例", "回答正确性", "逻辑性", "通顺性", "智能性"]

API = {
    "api_key": os.environ["SILICONFLOW_API_KEY"],
    "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
}
ANSWER_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

# ============================================================
# 四个 prompt 变体。它们的主任务完全相同，只在"怎么约束"上不同。
# ============================================================
PROMPTS = {
    # P1：原始写法。问题在于"不知道"指的是模型的认知，不是材料边界。
    "P1_基线": """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
最多使用三句话。尽量使答案简明扼要。

{context}""",

    # P2：只改一处——判断标准从"模型的认知"搬到"资料里有没有"。
    # 模型没法可靠判断"我知道吗"，但能可靠判断"这段文字里出现了吗"。
    "P2_材料边界": """请只依据下面【资料】中的内容回答问题。
如果【资料】中没有相关信息，请回答"资料中未提及该内容"。
不要使用你自身的知识进行补充或推测。

【资料】
{context}""",

    # P3：在 P2 之上，把隐式判断拆成显式步骤（思维链）。
    # 一旦"判断"变成必须写出来的中间步骤，模型就很难跳过它直接开答。
    "P3_判断步骤": """你只能依据【资料】作答。请严格按以下步骤执行，不要跳步：

第1步：判断【资料】中是否包含回答该问题所需的关键信息。
第2步：如果不包含，直接输出"资料中未提及该内容"，不要补充任何资料以外的知识，即使你知道答案。
第3步：如果包含，用不超过三句话回答，并在末尾用「依据：」引用【资料】中的原句。

【资料】
{context}""",

    # P4：规则化。编号清单 + "禁止"（比"不要试图"硬）+ 指定唯一可用的拒答句。
    # 关键是第 2 条给了确定输出，空位不会被自由发挥填满。
    "P4_规则清单": """你是一个严格依据资料作答的助手。必须遵守以下规则：
1. 只能使用【资料】中的信息，禁止使用任何外部知识或常识补充。
2. 若【资料】中找不到答案，只输出这一句：资料中未提及该内容。
3. 若找到答案，回答不超过三句话，并在末尾用「依据：」引用原文。
4. 不要解释你为什么这样回答，不要附加建议。

【资料】
{context}""",
}

# ---------- 组件 ----------
answerer = ChatOpenAI(model=ANSWER_MODEL, temperature=0, timeout=120, **API)
judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, timeout=180, max_tokens=600, **API)

emb = OpenAIEmbeddings(
    model=os.environ.get("SF_EMBEDDING_MODEL", "BAAI/bge-m3"),
    check_embedding_ctx_length=False,
    **API,
)
vectordb = Chroma(persist_directory=PERSIST, embedding_function=emb)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})


def build_chain(system_tpl: str):
    return (
        ChatPromptTemplate.from_messages([
            ("system", system_tpl),
            ("human", "问题：{input}"),
        ])
        | answerer
        | StrOutputParser()
    )


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

judge_chain = (
    ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYS),
        ("human", "【问题】{q}\n\n【参考答案要点】{expect}\n\n【是否在知识库内】{in_kb}\n\n"
                  "【检索到的原文片段】\n{context}\n\n【系统回答】\n{answer}"),
    ])
    | judge
    | StrOutputParser()
)


def parse_scores(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        t = max(t.split("```"), key=len).replace("python", "", 1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("输出里没有字典：" + text[:120])
    d = ast.literal_eval(t[s:e + 1])
    return {k: int(d.get(k, -1)) for k in DIMS}


def grade(row, answer, context) -> dict:
    for attempt in range(2):
        try:
            raw = judge_chain.invoke({
                "q": row["question"], "expect": row["期望答案要点"],
                "in_kb": row["是否在知识库内"], "context": context[:1200], "answer": answer,
            })
            return parse_scores(raw)
        except Exception as e:
            if attempt == 1:
                print(f"      ! 评分失败：{type(e).__name__} {str(e)[:80]}")
                return {k: -1 for k in DIMS}


def main():
    # 选择题目
    with open(EVAL_SET, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if "--only" in sys.argv:
        idx = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")]
        rows = [rows[i - 1] for i in idx]

    # 选择变体
    names = list(PROMPTS)
    if "--prompts" in sys.argv:
        wanted = sys.argv[sys.argv.index("--prompts") + 1].split(",")
        names = [n for n in names if n.split("_")[0] in wanted]

    chains = {n: build_chain(PROMPTS[n]) for n in names}

    print(f"被测模型：{ANSWER_MODEL}   裁判：{JUDGE_MODEL}")
    print(f"题目 {len(rows)} 条   变体 {len(chains)} 个   共 {len(rows) * len(chains)} 次评估\n")

    results = []          # [{q, variant, answer, scores, avg, in_kb}]
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        # 关键：检索只做一次，四个变体共用同一份 context（控制变量）
        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content[:300] for d in docs)
        print(f"[{i}/{len(rows)}] {q}  ({row['是否在知识库内']})")

        for name in names:
            ans = chains[name].invoke({"input": q, "context": context})
            sc = grade(row, ans, context)
            avg = sum(sc.values()) / len(sc)
            results.append({"q": q, "variant": name, "answer": ans,
                            "scores": sc, "avg": avg, "in_kb": row["是否在知识库内"]})
            print(f"    {name:12s} {avg:4.1f}   {ans[:56].replace(chr(10), ' ')}")

    # ---------- 汇总 ----------
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0

    print("\n" + "=" * 72)
    print(f"{'变体':<14}{'库内':>8}{'库外':>8}{'总体':>8}")
    print("-" * 72)
    summary = {}
    for name in names:
        rs = [r for r in results if r["variant"] == name]
        ins = [r["avg"] for r in rs if r["in_kb"] == "是"]
        outs = [r["avg"] for r in rs if r["in_kb"] == "否"]
        summary[name] = (mean(ins), mean(outs), mean([r["avg"] for r in rs]))
        print(f"{name:<14}{mean(ins):>8.2f}{mean(outs):>8.2f}{mean([r['avg'] for r in rs]):>8.2f}")
    print("\n提示：同一裁判、同一份题，两次运行天然有 ±0.1 抖动，")
    print("      提升幅度小于 0.2 的不算数。总耗时 {:.0f}s".format(time.time() - t0))

    # ---------- 库外题的答案对比（最有说服力的部分） ----------
    # 去重：results 里每个问题有 len(names) 条记录
    out_qs = list(dict.fromkeys(r["q"] for r in results if r["in_kb"] == "否"))
    if out_qs:
        print("\n" + "=" * 72)
        print("库外题在四个变体下的回答（重点看有没有拒答）")
        print("=" * 72)
        for q in out_qs:
            print(f"\n【{q}】")
            for name in names:
                r = next(x for x in results if x["q"] == q and x["variant"] == name)
                print(f"  {name:12s} {r['avg']:4.1f}  {r['answer'][:90].replace(chr(10), ' ')}")

    # ---------- 存盘 ----------
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "是否在知识库内", "变体"] + DIMS + ["平均分", "回答"])
        for r in results:
            w.writerow([r["q"], r["in_kb"], r["variant"]]
                       + [r["scores"][d] for d in DIMS] + [f"{r['avg']:.2f}", r["answer"]])
    print(f"\n明细已保存 -> {OUT}")


if __name__ == "__main__":
    main()
