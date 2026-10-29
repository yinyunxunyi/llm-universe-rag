# 个人知识库助手（llm-universe 实战）
> 本项目基于 [Datawhale llm-universe](https://github.com/datawhalechina/llm-universe) 教程完成，教材原文见 `README.md`。
> 本仓库中的 `README_我的项目.md`、`实验记录.md`、各 `step*.py` 与 `.csv` 结果文件为本人实现、实验与结论。


一个基于 RAG 的本地知识库问答系统：把文档丢进 `data_base/knowledge_db/`，它就能**只依据这些文档**回答问题，答不出来就说"资料中未提及"，并在网页上连续对话。

---

## 1. 用了什么（可复现的关键信息）

| 项目 | 选择 | 说明 |
|---|---|---|
| 生成模型 | **`deepseek-ai/DeepSeek-V3`** | 走硅基流动，`temperature=0` |
| 向量模型 | **`BAAI/bge-m3`** | 1024 维，走硅基流动 |
| 裁判模型（评测用） | **`deepseek-ai/DeepSeek-V4.1-Flash`** | 必须比被测模型强，否则会给自己打高分 |
| 向量库 | **Chroma 1.5.9** | 落盘于 `data_base/vector_db/chroma/` |
| Python | **3.13**（conda 环境 `llm-universe`） | 见第 5 节版本坑 |
| **chunk_size / overlap** | **500 / 50** | `RecursiveCharacterTextSplitter` |
| **top_k** | **3** | 每次检索 3 段原文 |
| **验证集平均分** | **4.41 / 5**（20 条，库内 15 + 库外 5） | 优化前 3.80 |

### 依赖版本（钉死，别随手升级）

```
pydantic==2.10.6  langchain==0.3.27  langchain-core==0.3.86
langchain-community==0.3.27  langchain-text-splitters==0.3.11
langchain-openai==0.3.35  langchain-chroma==0.2.4  chromadb==1.5.9
pymupdf>=1.24,<2  streamlit==1.66.1  python-dotenv
```

---

## 2. 十五分钟跑起来

```bash
# 1. 建环境（Python 3.13）
conda create -n llm-universe python=3.13 -y

# 2. 装依赖（用上面那串，不要用项目自带的 requirements.txt）
D:\anconda\envs\llm-universe\python.exe -m pip install <上面的包>

# 3. 在项目根目录建 .env，至少这四行
SILICONFLOW_API_KEY = "sk-你的Key"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SF_EMBEDDING_MODEL = "BAAI/bge-m3"
SF_LLM_MODEL = "deepseek-ai/DeepSeek-V3"

# 4. 冒烟测试：应打印「维度 1024」+ 一句中文回复
D:\anconda\envs\llm-universe\python.exe check_env.py

# 5. 加载 → 清洗 → 切分（产出 chunks.pkl，788 片）
D:\anconda\envs\llm-universe\python.exe step4a_chunk.py

# 6. 建向量库（只能跑一次！重复跑会追加数据）
D:\anconda\envs\llm-universe\python.exe step4b_build.py

# 7. 检索验证（建库完先确认检索是准的，再接大模型）
D:\anconda\envs\llm-universe\python.exe step4c_search.py

# 8. 启动网页
D:\anconda\envs\llm-universe\Scripts\streamlit.exe run app.py
```

> 本机有 4 份 Python，所以所有命令都用**完整路径**调用，不依赖 `conda activate`。

---

## 3. 文件说明

| 文件 | 作用 |
|---|---|
| `check_env.py` | 冒烟测试：密钥 / embedding 维度 / LLM / 账号可用模型 |
| `step4a_chunk.py` | 加载 PDF/MD/TXT → 清洗 → 切分 → `chunks.pkl` |
| `step4b_build.py` | 分批写入 Chroma（含"防重复写入"保护） |
| `step4c_search.py` | 检索验证，可指定查询：`step4c_search.py "什么是查全率"` |
| `step5_qa.py` | 单轮问答 + 「裸模型 vs 加知识库」对照 |
| `step5_chat.py` | 多轮对话（含问题压缩），`-i` 进交互模式 |
| `app.py` | Streamlit 网页版 |
| `eval_set.csv` | 20 条验证集（15 库内 + 5 库外） |
| `step7_eval.py` | 用 LLM 裁判跑验证集，输出 7 维度分 |
| `step7_optimize.py` | prompt A/B 实验 |
| `prompt_ab.csv` | 40 条评测明细（20 题 × P1/P2 两个 prompt，Excel 直接打开不乱码） |
| `ab_full.log` / `eval_baseline.log` | 评测原始日志，可作为分数来源的证据 |

---

## 4. 评测结果（这是本项目最该看的部分）

| 变体 | 库内 15 条 | 库外 5 条 | 总体 | 拒答率 |
|---|---|---|---|---|
| P1 基线 | 4.55 | 1.54 | 3.80 | **0 / 5** |
| **P2 材料边界（当前采用）** | 4.45 | **4.29** | **4.41** | **5 / 5** |

**结论**：总体 +0.61，全部来自"不再编造"。库内 −0.10 在裁判抖动范围内（±0.1），可视为无变化。

核心问题是 prompt 里那句"如果你不知道答案就说不知道"——它约束的是**模型的认知**，而这道题的答案模型其实知道，所以它从不拒答。改成"如果**资料中**没有相关信息"后，判断依据从"我知不知道"变成"这段文字里有没有"，拒答率立刻从 0 变成 100%。

详细实验过程见 `实验记录.md`。

---

## 5. 踩过的坑（照抄省一天）

| 现象 | 原因 | 处理 |
|---|---|---|
| pip 去下 `numpy-1.26.4.tar.gz` 然后报 metadata 错误 | 装了 `unstructured`（拖 spacy+numba，3.13 上没 numpy 轮子） | md/txt 改用 `TextLoader` |
| `chroma-hnswlib` 编译失败 | chromadb 被降到 <1.0（Windows 轮子只到 cp312） | 钉 `chromadb==1.5.9` + `langchain-chroma==0.2.4` |
| `AttributeError: 'Collection' object has no attribute 'persist'` | 用了教材的 `vectordb.persist()` | chromadb 1.x 自动落盘，删掉这行 |
| 终端跑 LLM 像卡死 | `invoke()` 一次性返回，38 秒内屏幕无任何输出 | 改用 `.stream()`，首字 1 秒 |
| prompt 里的 JSON 示例报 `KeyError: missing variables` | LangChain 把 `{...}` 当模板变量 | 字面花括号写成 `{{ }}` |
| 多轮对话"改写问题"却输出了答案 | 历史以 human/ai 消息传入，模型模仿上一轮答案风格 | 历史降级为 system 里的纯文本 |
| 查 `few-shot` 返回"压缩感知" | 语料里该词出现 **0 次**，检索器被迫给一个"最不坏"的 | 见下方"已知限制" |

---

## 6. 已知限制（诚实地写出来）

1. **默认语料不适合定义类问题。** 南瓜书是《机器学习公式详解》，通篇推导公式、几乎不"定义"概念。问"什么是 PCA"检索效果很差，问"查准率和查全率怎么权衡"就好得多。
2. **`easy_rl/强化学习入门指南.txt` 是语音转写稿**，含大量错字（"引用方向"=研究方向），属于噪音源，可考虑移除。
3. **向量检索不会说"我没有"。** 库里没有答案时，它仍会返回 top-k 个"最不坏"的片段，靠 LLM 判断力兜底。因此 prompt 必须明确边界。
4. **裁判有 ±0.1 抖动**，同一份代码两次跑分数不完全相同。提升幅度小于 0.2 的一律不算数。
5. **top-3 里常有噪音**（封面页、无关公式）。实测"什么是南瓜书"的 3 段里只有 1 段有用。

---

## 7. 下一步（按性价比排序）

> 前两条已经做完了，结论见 `实验记录.md`：**rerank 零收益（不做）**，**query 改写 +0.60（有效，待接入 UI）**。

1. **chunk_size 对比实验**：300 / 500 / 800 各建一次库，看平均分与落差变化（要重建 3 次库）
2. **top_k 敏感性**：1 / 3 / 5 / 8，不用重建库，改参数即可
3. **把 query 改写接进 `app.py`**：作为侧边栏开关，因为会给首字延迟加 1~2 秒
4. **换语料**：把 `data_base/knowledge_db/` 换成自己领域的文档——这才是"个人知识库"的本意
5. **在多文档语料下重测 rerank**：本书单语料下它无效，但多文档竞争场景下结论可能反转
