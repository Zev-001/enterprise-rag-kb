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

## 七、已知边界 / 下一步可扩展

- ⚠️ 当前是单知识库 `default`，多租户隔离（不同公司互不看到）可在 `rag_core.load_kb(kb)` 的 `kb` 参数上扩展。
- ⚠️ 大文件（>50MB PDF）解析较慢，可加「后台任务 + 进度条」。
- ⚠️ 没有登录鉴权，公网部署前必须加（否则谁都能问你的内部文档）。
- 💡 可接 day22 的 `xhs-cover-gen` 思路，把「知识库问答」包装成可被外部调用的服务。
