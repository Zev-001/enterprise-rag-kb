# 企业知识库问答系统（RAG）

> 阶段 5 作品集 · 项目 1
> 把公司文档（PDF / Word / TXT / Markdown）喂进去，员工用大白话提问，系统**只根据文档回答**，并附上「来自哪份文件哪一段」的可点击溯源。问到资料外的问题会老老实实说「资料里没提到」，绝不瞎编。

---

## 一、这个系统能做什么（小白版）

想象一个「不会撒谎的公司百事通」：

- 📄 **上传文档**：把员工手册、产品说明、规章制度拖进系统，自动解析 + 切块。
- 💬 **自然语言提问**：「年假怎么请？」「报销上限多少？」直接问，不用翻文档。
- 🔗 **答案可溯源**：每条答案后面标 `[1][2]`，点开就知道来自哪份文件哪一段。
- 🛡️ **防幻觉闸门**：资料里没有的内容，系统会回答「资料里没提到」，而不是编一个看似合理的答案。
- 🧱 **双检索兜底**：语义检索（中文 bge 模型）+ 关键词检索（TF-IDF）双保险，一个挂了自动切另一个。

---

## 二、技术架构（一句话版）

```
你上传的文档
   ↓ 解析（PDF/Word/TXT/MD → 纯文本）
   ↓ 切块（带重叠，避免一句话被切断）
   ↓ 向量化（bge 中文模型）→ 建索引
        ↓
你提问 ──→ 检索最相关的几段 ──→ 拼进「防幻觉提示词」 ──→ 大模型作答
                                          ↓
                              答案 + 出处 [1][2] 返回前端
```

| 模块 | 文件 | 作用 |
|------|------|------|
| 核心引擎 | `rag_core.py` | 切块、双检索、防幻觉问答、事实核查闸门 |
| 文档解析 | `ingest_docs.py` | 把上传的 PDF/Word/TXT/MD 变成纯文本再切块入库 |
| Web 应用 | `app.py` | Flask 接口：上传、问答、统计；托管前端页面 |
| 前端页面 | `templates/index.html` + `static/style.css` | 聊天式界面：上传区 + 对话区 + 引用卡片 |
| 演示入库 | `demo_ingest.py` | 把示例资料喂进知识库，方便一上来就有东西可问 |
| 鉴权 / ACL | `auth.py` | **P0 补强**：token 鉴权 + 知识库 namespace 权限隔离（D1/D5） |
| 量化评测 | `eval_rag.py` | **P0 补强**：内置中文问答金标准集，离线验检索召回率与可分性（D6） |
| 会话存储 | `session_store.py` | **P1 补强**：文件级多轮会话（重启不丢，TTL 清理），支撑查询改写（D10） |
| LLMOps 日志 | `rag_log.py` | **P1 补强**：问答落 JSONL + 指标聚合（延迟/拒答率/Top 问题）（D11） |
| 持久化向量库 | `vector_store.py` | **P1 补强**：FAISS 索引落盘 + 签名校验，免每次重建（D12） |
| GraphRAG | `graph_rag.py` | **P1 补强**：实体抽取 + 轻量图谱多跳检索，`retrieval=graph` 启用（D7） |
| Agentic RAG | `agentic_rag.py` | **P1 补强**：ReAct 式自问自答，`mode=agent` 启用（D8） |
| 企业连接器 | `connectors.py` | **P1 补强**：Connector 基类 + 本地目录 / 网页连接器（D9） |
| 语义分块 | `chunker.py` | **P2 补强**：语义/命题/模板多模式切块，`RAG_CHUNK_MODE` 切换（D13） |
| 上下文压缩 | `context.py` | **P2 补强**：检索后合并去重压缩上下文，控 token 成本（D14） |
| 答案缓存 | `cache.py` | **P2 补强**：同问秒回（1h TTL），缓存 key 区分检索模式（D15） |
| 置信度评分 | `confidence.py` | **P2 补强**：检索分+引用+拒答占比算可信度，前端徽章展示（D16） |
| 跨模态 | `multimodal.py` | **P2 补强**：表格/图片(OCR) 结构化入库，统一走一套检索（D17） |
| 答案质检 | `schema_qc.py` | **对标 GEOFlow 质量门禁**：答案质检强制 JSON Schema 输出 + 证据编号回验，替代拒答正则（详见下节） |
| 引用内容回验 | `evidence_verify.py` | **对标 GEOFlow validateEvidenceSnapshot**：拿引用块全文校验每条断言，抓「张冠李戴」型引用幻觉，只降不升（D19，详见下节） |
| 召回前元数据过滤 | `meta_filter.py` | **对标 GEOFlow 知识治理**：生效日期(effective_date) 时效 + 审核状态(review_status) 治理——过期/未生效/未审核/作废的块在【进入检索之前】就被拦掉，不靠 prompt 叮嘱（D20，详见下节） |

**技术选型理由（面试常问）**
- `sentence-transformers` 的 `bge-small-zh-v1.5`：中文语义检索效果好的轻量模型，本地跑不联网也能检索。
- `faiss-cpu`：毫秒级向量检索，比暴力遍历快几个数量级。
- `TF-IDF` 兜底：语义模型偶尔抽风时，关键词匹配还能顶上，保证「不会彻底失忆」。
- 防幻觉核心来自 day17 已经验证过的 `grounded_ask` + `claim_verdict` 三道闸门，这里泛化成「任意文档」版本。

---

## 三、运行环境准备

### 1. 准备 Python 虚拟环境（推荐用托管 venv）
```bash
# 用 WorkBuddy 托管的 python 建 venv（Windows 路径）
C:/Users/1/.workbuddy/binaries/python/versions/3.13.12/python.exe -m venv C:/Users/1/.workbuddy/binaries/python/envs/default
PY="C:/Users/1/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
$PY -m pip install -r requirements.txt
```

### 2. 准备 API Key
在项目根目录的 `.env` 里放一行（只放 key 名和值，本文件不提交）：
```
DEEPSEEK_API_KEY=你的真实key
```
代码会自动读这个文件，API key 不会写进任何代码。

### 3. 首次运行会下载中文模型（约 90MB，走镜像）
已下载过会直接复用。若缓存损坏，删掉 `~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5` 重下即可。

---

## 四、三种使用方式

### 方式 A：命令行直接玩（最快验证）
```bash
cd D:/Workbody/AI学习/AI转行计划/enterprise_rag
PY="C:/Users/1/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# 1) 先灌示例资料（AI 新闻 jsonl）
$PY demo_ingest.py

# 2) 提问（会自动检索 + 调大模型 + 溯源）
$PY rag_core.py ask "阿里开源的 Qwen 模型总参数和激活参数分别是多少？"

# 3) 只测检索、不调大模型（离线、省 key）
RAG_TEST=1 $PY rag_core.py ask "Qwen 相关的内容有哪些？"
```

### 方式 B：启动网页界面（推荐体验）
```bash
$PY app.py
# 浏览器打开 http://127.0.0.1:5000
# ① 点「选择文件」拖入公司文档（可多选）
# ② 等右上角提示「已入库 N 段」
# ③ 在对话框提问，答案下方会显示引用卡片
```

### 方式 C：当成库调用（你自己的程序）
```python
import rag_core as r
r.ingest_file("员工手册.pdf", kb="default")   # 入库
out = r.grounded_ask("年假怎么请？", kb="default")  # 问答
print(out["answer"])
for s in out["sources"]:
    print(s["filename"], s["chunk_id"], s["score"])
```

---

## 五、效果验证结论（已实测）

| 测试项 | 结果 |
|--------|------|
| 中文语义检索 | ✅ 相关度 0.747（远胜 TF-IDF 0.179） |
| 资料内问题溯源 | ✅ 正确标 `[1][2]` 到对应文档段落 |
| 资料外问题守幻觉 | ✅ 「公司年假多少天」答「资料里没提到」，不编造 |
| 多格式上传 | ✅ PDF / Word / TXT / MD 均可；中文文件名正常 |
| Web 整链 | ✅ /api/stats、/api/upload、/api/ask 全部跑通 |

---

## 六、目录结构

```
enterprise_rag/
├── rag_core.py          # 核心引擎（切块/检索/防幻觉/溯源）
├── ingest_docs.py       # 文档解析（PDF/Word/TXT/MD）
├── app.py               # Flask Web 服务
├── demo_ingest.py       # 示例资料入库脚本
├── requirements.txt     # 依赖清单
├── 需求文档.md           # 需求 + 架构设计说明
├── templates/
│   └── index.html       # 聊天前端
├── static/
│   └── style.css        # 前端样式
├── data/                # 知识库存放处（kb_default.jsonl，运行后生成）
└── .env                 # 你的 API key（不提交，见 .gitignore）
```

---

## 七、P0 补强记录（对标高端系统的差距收敛）

> 依据 `RAG调研与缺陷分析.md` 的缺陷清单，本轮补齐了 6 项 P0（D1–D6）。

| 缺陷 | 内容 | 本轮补强 | 文件 |
|------|------|----------|------|
| D2 无 Rerank | 单路取 top_k，易漏相关块 | RRF 跨「语义+关键词」两路融合重排；**保留原始相似度做阈值守门**，避免稀释防幻觉闸门 | `rag_core.py` `rerank_fusion` |
| D3 文档解析 | 仅读 PDF 文字层，表格/扫描件丢失 | `pdfplumber` **表格抽取**单列成块；扫描件无文字层时优雅尝试 OCR（pytesseract 可选，缺依赖不中断） | `ingest_docs.py` |
| D4 单 LLM 硬编码 | 只能 deepseek-chat | `MODEL` / `API_URL` 改读环境变量（`RAG_LLM_MODEL` / `RAG_LLM_API_URL`），可切任意 OpenAI 兼容端点（模型路由第一步） | `rag_core.py` |
| D1 无 ACL / D5 无鉴权 | 谁都能访问任意知识库 | `auth.py`：token 鉴权 + namespace 白名单；admin token 落 `acl.json` 持久化；前端加 token 输入框 | `auth.py` + `app.py` + `index.html` |
| D6 无评测体系 | 改了不知变好变坏 | `eval_rag.py`：内置中文问答金标准集，离线验检索召回率 + 资料内/外可分性；`--live` 真机复验拒答话术 | `eval_rag.py` |

### P1 补强（差异化能力，对标高端系统，2026-08-31）

> 阶段 A（P0）收敛底线后，本轮补齐 6 项 P1（D7–D12）。实现原则：**零额外重依赖、可离线、等价对标**——用自研轻量 ReAct 替代 LangGraph、落盘 FAISS 替代 Qdrant，不引入 Docker / 外部服务即可讲高端故事。

| 缺陷 | 内容 | 本轮补强 | 文件 |
|------|------|----------|------|
| D7 无 GraphRAG | 扁平 chunk 检索，答不了多跳/关系问题 | `graph_rag.py`：实体抽取 + 共现图谱（纯 dict 零依赖）；`/api/ask?retrieval=graph` 多跳检索 | `graph_rag.py` |
| D8 无 Agentic RAG | 单趟检索，复杂问题遗漏 | `agentic_rag.py`：ReAct 自问自答（拆子问题→逐条检索作答→综合）；`/api/ask?mode=agent` | `agentic_rag.py` |
| D9 无连接器 | 只能本地传文件 | `connectors.py`：基类 + `LocalFolderConnector` / `WebPageConnector`；`/api/connector/ingest`（admin）入库 | `connectors.py` |
| D10 多轮上下文 | 每轮独立、不接上文 | `session_store.py` + `rewrite_query`：文件级会话 + 查询改写（指代消解）；`/api/ask` 带 `session_id` 连续追问 | `session_store.py` + `rag_core.py` |
| D11 无 LLMOps | 上线黑盒、无日志 | `rag_log.py`：问答落 JSONL；`/api/metrics` 聚合（延迟/拒答率/Top 问题/按库分布） | `rag_log.py` |
| D12 无生产向量库 | 内存索引、库大就慢 | `vector_store.py`：FAISS 落盘 + 签名校验，库不变免重建、变了自动失效 | `vector_store.py` |

**P1 新增接口速查**
```bash
# 多轮连续追问（带 session_id 即承接上文）
curl -X POST "http://127.0.0.1:5000/api/ask?token=$TK" -H 'Content-Type: application/json' \
  -d '{"question":"那报销呢","namespace":"default","session_id":"上轮返回的sid"}'

# GraphRAG 多跳检索
curl -X POST ... -d '{"question":"年假怎么算","retrieval":"graph"}'

# Agentic RAG 多步推理
curl -X POST ... -d '{"question":"对比年假和调休政策","mode":"agent"}'

# 从连接器拉数据入库（仅 admin）
curl -X POST "http://127.0.0.1:5000/api/connector/ingest?token=$ADMIN" -H 'Content-Type: application/json' \
  -d '{"type":"local_folder","source":"D:/docs","namespace":"team"}'

# 看运维指标
curl "http://127.0.0.1:5000/api/metrics?token=$TK"
```

**运行评测（不烧 token）**
```bash
$PY eval_rag.py            # 离线：检索召回率 + 可分性（默认 100% 召回）
$PY eval_rag.py --live     # 真机：连「资料里没提到」拒答话术一起验（需 DEEPSEEK_API_KEY）
```

### P2 补强（体验层，对标高端系统的"感觉"，2026-08-31）

> 阶段 C（P2）补齐 5 项体验（D13–D17）。原则不变：**零额外重依赖、可离线、等价对标**——语义分块用句群聚类而不是引第三方 LLM、流式走 SSE 而不是引 WebSocket 服务。

| 缺陷 | 能力 | 落地 | 关键实现 |
|---|---|---|---|
| D13 语义分块 | 四种切块模式 | `chunker.py` | legacy 按段硬切 / semantic 句群聚类 / proposition 命题切 / template 表格模板切；`RAG_CHUNK_MODE` 切换，默认 semantic |
| D14 上下文压缩 | 检索后瘦身 | `context.py` | 相邻块合并 + 近似重复去重，控 prompt token 成本，压缩统计随响应返回 |
| D15 流式响应 + 缓存 | 边想边答 | `cache.py` + `ask_llm_stream` | SSE 逐 token 推流（前端打字机效果）；答案缓存 1h TTL，key 含检索模式防互相污染。**D21 内容指纹失效**：缓存条目绑定知识库指纹（`rag_core.kb_fingerprint`，文件字节 md5 + mtime 缓存），知识一变旧答案自动作废，无需手动清缓存 |
| D16 置信度评分 | 答案可信度 | `confidence.py` | 检索分 + 引用标注 + 按句拒答占比算分，输出 confidence/level/reason，前端徽章展示 |
| D17 多语言/跨模态 | 双语 + 表格图片 | `multimodal.py` + 提示词 | 表格/图片(OCR) 结构化入库走同一检索；系统提示词要求「用提问的语言回答」 |

**P2 新增接口速查**
```bash
# 流式问答（SSE 打字机效果）
curl -N -X POST "http://127.0.0.1:5000/api/ask?token=$TK" -H 'Content-Type: application/json' \
  -d '{"question":"年假有几天？","namespace":"demo","stream":true}'

# 表格入库（D17，仅 admin）
curl -X POST "http://127.0.0.1:5000/api/multimodal/ingest?token=$TK" -H 'Content-Type: application/json' \
  -d '{"type":"table","namespace":"demo","filename":"报销标准.csv","title":"报销标准表","headers":["项目","上限"],"rows":[["住宿","500元/晚"]]}'

# 缓存管理（GET 统计 / DELETE 清缓存，仅 admin）
curl "http://127.0.0.1:5000/api/cache?token=$TK"
curl -X DELETE "http://127.0.0.1:5000/api/cache?token=$TK&namespace=demo"
```

**前端体验**：打字机流式输出、置信度徽章（high/medium/low）、**质检徽章（有据可答/部分有据/资料未提及/无依据·疑似幻觉）**、缓存命中标记、检索模式切换（默认/图谱）、跨模态来源卡片。

### 答案质检：JSON Schema 硬约束（借鉴 GEOFlow 质量门禁，2026-08-31）

> 改造前，「这个答案可不可信」全靠 `confidence.py` 里的**中文正则**猜：
> `_REFUSE_RE = re.compile(r"资料里没提到|没有检索到|我不了解|无法确定|没有找到")`
> 三个硬伤：**换个说法就漏判**、**英文场景直接瞎**（D17 做了多语言，拒答检测却只有中文词表）、
> **引用从不校验**（模型编一个 `[3]`，实际只召回 2 段，照样高分通过）。

**做法**：让模型自己结构化呈堂证供，再用 schema 硬性校验（GEOFlow 的 `ArticleQualityReviewerAgent`
同款思路）。

```json
{
  "status": "answered|partial|refused|unsupported",
  "confidence": 0.0,
  "claims": [
    {"text": "年假有5天", "evidence_keys": [1], "evidence_status": "supported|weak|unsupported"}
  ],
  "missing": "资料里缺少什么",
  "reasons": ["判定理由"]
}
```

**三道硬校验**（不靠模型自觉，`schema_qc.validate_qc`）：

| 层 | 检查什么 | 失败后果 |
|---|---|---|
| ① schema | required / 类型 / status 四态枚举 / confidence 值域 | 记 errors，尽力修复后继续 |
| ② 证据回验 | `evidence_keys` 超出实际召回数 → 强制 `unsupported` 并剔除越界编号 | **抓引用幻觉** |
| ③ 一致性 | `status=answered` 但无一条 `supported` → 降级 `unsupported` | 抓自相矛盾 |

**置信度融合**（`confidence.score_confidence(..., qc=qc)`）：拿到结构化台账后，落地性
（groundedness）改由**证据比例**算（supported 计 1、weak 计 0.5），不再猜拒答；并按 status
设天花板：`refused ≤ 0.15`、`unsupported ≤ 0.30`、`partial ≤ 0.60`。模型自评占三成权重。

**兜底原则**：解析失败 / 校验失败 / 调用失败 → 一律回退正则启发式，并标
`mode=heuristic_fallback`，**保证不比改造前更差**。回退模式下评分逻辑完全不变（自测已锁回归）。

**实测对比**（真实 DeepSeek 调用，同一答案）：

| 场景 | 旧（拒答正则） | 新（JSON Schema 质检） |
|---|---|---|
| 「公司提供住房补贴吗？」<br>答：*资料里没提到住房补贴。不过《报销标准表》里提到住宿报销上限是500元/晚 [2]。* | **low 0.15**<br>首句含「资料里没提到」→ 整段判拒答<br>（实际拒答只占 22% 篇幅） | **medium 0.60**<br>3 条断言：2 有据 + 1 无据 → partial<br>并给出「资料缺口：未明确提及住房补贴」 |
| 英文提问（正则的死穴） | 中文词表匹配不到，拒答检测失效 | 正常判 `answered`，理由为英文 |

**开关**：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `RAG_QC_MODE` | `json` | `off` = 完全关闭退回纯启发式（省一次调用） |
| `RAG_QC_ACTION` | `warn` | `block` = `unsupported` 答案就地拦截，换成拒答话术（对齐 GEOFlow「不合格留草稿」） |

**成本优化**：`hits=0`（没召回任何资料）直接短路判拒答，不浪费质检调用；流式场景质检在
打字机结束后再跑，不挡首字延迟。

**自测**：`python test_schema_qc.py`（41 项离线断言，无需 API Key），覆盖六种「模型不乖」的
输出形态、三层校验、置信度融合与回归、回退分支、门禁动作。

### 引用内容级回验：抓「张冠李戴」（借鉴 GEOFlow validateEvidenceSnapshot，2026-08-31）

D18 的编号越界校验抓不到这种幻觉：答案说「年假有 5 天 [2]」，而 [2] 那块讲的是报销——
编号合法、LLM 质检员也可能看走眼。`evidence_verify.py` 在质检后追加**内容级回验**：

1. 对每条 claim，拿它引用块的**全文**（snippet 200 字会截断依据，`prepare_ask` 已补全文字段）
   做相似度校验，三层评分：归一化子串精确命中 → 字符 bigram 重叠（零依赖兜底）→
   bge 向量余弦（复用检索模型，`HF_HUB_OFFLINE=1` 离线加载，不依赖运行时网络）；
2. **双信号判据**（按实测校准：相关对 cos≈0.73 / 张冠李戴对 cos≈0.50 / 无关 <0.40，
   bge 对「短句 vs 长块」分数虚高，单看绝对线会误判）：cos ≥0.62 或 n-gram ≥0.35 →
   supported；cos ≥0.42 或 n-gram ≥0.15 → weak；否则 unsupported；
3. **只降不升**：回验只能把 LLM 判的 supported→weak→unsupported 往下修，绝不往上修——
   宁可错杀为 weak，不把幻觉洗白；修正写入 `reasons` 与 `verification` 明细，
   并重算 `summary`（置信度融合自动吃到新证据比例）。

**实测**（真实 DeepSeek 调用）：把「年假有5天」的引用人为换到报销块 → 回验判
`weak（score=0.483）` 并写明「依据偏弱」；正常引用（exact 1.0）不受影响。
回归：`test_evidence_verify.py` 33 项（含真实语义路径）、P2 自测 35/35、D18 自测 41/41、
eval 检索层 100 分。开关：`RAG_EVIDENCE_VERIFY=off` 关闭（默认 on，异常自动跳过不伤主流程）。

### 召回前元数据过滤：过期/未审核的块连候选都不进（借鉴 GEOFlow 知识治理，2026-08-31）

D19 及之前的质检都在「答案已经生成之后」兜底。GEOFlow 给的另一个启发是**治理前移**：
企业知识库不是「进了库就能用」——明年才生效的薪级表、上个月已作废的报销标准、
还没定稿的草稿，检索得再准也不该拿来回答今天的提问。`meta_filter.py`（D20）把这道门
挪到**召回之前**：

1. **代码层硬过滤，不靠 prompt 叮嘱**：块带 `effective_date`（生效日期）/
   `effective_until`/`expires_at`（失效日期）/ `review_status`（审核状态）元数据时，
   未生效/已过期/草稿/待审核/已作废的块在进入语义索引与 TF-IDF **之前**就被剔除——
   两路召回天然看不到它们，GraphRAG 检索同样按可见块集合收口；
2. **向后兼容**：不写治理字段的块（老库）行为完全不变；`review_status` 写了但值不认识
   → 保守当未审核拦下（fail-closed，治理场景宁可误拦）；
3. **边界语义**：当天生效算生效、当天到期仍可用（含边界），日期支持
   `2026-08-31` / `2026/8/31` / `2026年8月31日` / `20260831` 等多格式；
4. **治理可观测**：过滤不搞黑箱——`/api/stats` 返回库里拦了多少块（前端知识库状态栏
   显示「🛡 已拦截 N 块」），每次问答响应带 `meta_filtered`（拦了哪些块、各什么原因），
   CLI/日志同步打印；全被拦时检索为空，交给防幻觉闸门走「资料里没提到」；
5. **入库即声明**：上传文档时可带治理字段（`{"filename":..., "effective_date":"2026-09-01",
   "review_status":"draft"}`），自动下沉到每块 meta。

开关：`RAG_META_FILTER=off` 退回不过滤（对照评测）；`RAG_TODAY=2026-08-31` 可指定
「今天」（时间旅行演示：同一份库，把今天拨到生效日之后，被拦的块立刻可见）。
自测：`test_meta_filter.py` 43 项全过（离线零依赖）；P2 35/35、D18 41/41、
D19 30/30、eval 100 分无回归。

**启用鉴权**
```bash
# 设环境变量指定 admin token（不设则首次运行自动生成并写入 acl.json）
export RAG_API_TOKEN=你的强口令
$PY app.py
# 网页右上角「访问令牌」框粘贴该 token；API 也可带 ?token= 或 Header Authorization: Bearer
```

---

## 八、已知边界 / 下一步可扩展（P2 已收）

- ✅ **P0 + P1 + P2 全收敛**：Rerank / 表格抽取 / 模型可配置 / 鉴权+ACL / 量化评测（P0）；GraphRAG / Agentic RAG / 连接器 / 多轮上下文 / LLMOps / 持久化向量库（P1）；语义分块 / 上下文压缩 / 流式+缓存 / 置信度评分 / 多语言跨模态（P2）均已落地并实测。另补 **JSON Schema 硬约束质检**（对标 GEOFlow 质量门禁）。
- ⚠️ 当前 ACL 是「应用层 token + namespace 白名单」，企业级应**镜像源系统 ACL**（Glean / M365 Purview 那种），数据模型已留扩展位（namespace 即隔离边界）。
- ⚠️ OCR 需另装 `pytesseract` + 系统 `Tesseract` + `poppler`，未装时扫描件仅提示、不阻断入库。
- ⚠️ 大文件（>50MB PDF）解析较慢，可加「后台任务 + 进度条」。
- ✅ **额外补强**：答案质检改 JSON Schema 硬约束（`schema_qc.py`）+ 引用内容级回验（`evidence_verify.py`）+ 召回前元数据过滤（`meta_filter.py`）+ 内容指纹缓存失效（`cache.py` + `kb_fingerprint`），四件套借鉴 GEOFlow 质量门禁/知识治理——替代拒答正则、抓引用幻觉、过期/未审核资料进不了检索、知识一变旧答案自动作废。
- 💡 **下一步可做**：把 Qdrant / Langfuse 当成可选后端替换 `vector_store` / `rag_log` 的轻量实现；引用块加 `content_hash` 发布时回验；缓存命中/失效指标进 LLMOps 看板。
- 💡 可接 day22 的 `xhs-cover-gen` 思路，把「知识库问答」包装成可被外部调用的服务。
