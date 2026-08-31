"""
rag_core.py —— 企业知识库问答系统的「检索 + 防幻觉」核心引擎。

这是从 day17 的 dailyrag 演化来的「企业级」版本。day17 只吃写死的 AI 资讯元组，
现在泛化成：任意格式的**公司文档**都能上传、切块、检索、基于文档作答并标出来源。

为什么不直接用 dailyrag？
    dailyrag 的元数据写死成 (date, source, url, title)，只适合「按天归档的新闻」。
    企业要的是「上传一份员工手册 / 产品说明书就能问」，元数据换成
    (doc_id, filename, title)。算法本身（切块、双检索、防幻觉、事实核查）是通用金子，
    原样复用，只改数据模型 + 加多知识库隔离。

防幻觉纪律（和 dailyrag 一样，是整个系统唯一真正挡住幻觉的东西）：
    【硬规矩】只能使用【参考资料】里的字，资料里没提到就直接说「资料里没提到」，
    不许编、不许推测、不许用训练记忆。

依赖
    sentence-transformers + faiss-cpu（语义检索，推荐）
    只有标准库也能跑（TF-IDF + 中文二元组兜底）
    requests（调大模型）

环境变量
    DEEPSEEK_API_KEY   调大模型用
    HF_ENDPOINT        HuggingFace 镜像（国内防超时），默认 https://hf-mirror.com
    RAG_TEST=1         离线模式：不调 LLM，只返回检索命中，供零成本自查
    RAG_BACKEND=tfidf  强制走 TF-IDF（不想拉模型时）

位置
    D:/Workbody/AI学习/AI转行计划/enterprise_rag/
"""

import json
import math
import os
import re
from collections import Counter

import requests

# D13/D14/D15/D16/D17 补强（P2 / 阶段C）：新模块均零额外依赖
import chunker as _chunker
import context as _context
import cache as _cache
import confidence as _confidence
import multimodal as _mm
# 借鉴 GEOFlow 质量门禁：答案质检走 JSON Schema 硬约束
import schema_qc as _qc
# D20（GEOFlow 升级③）：召回前元数据过滤（时效 effective_date + 审核态 review_status）
import meta_filter as _mf

# ---------------------------------------------------------------- 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 知识库存这里：每个「知识库(namespace)」一个 jsonl 文件，实现多库隔离
KB_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(KB_DIR, exist_ok=True)

# D4 补强（P0）：模型可配置，迈出「模型路由」第一步。
#   默认 DeepSeek；想换模型/厂商，只需设环境变量，不必改代码：
#     RAG_LLM_API_URL  自定义 chat/completions 端点（OpenAI / 通义 / 本地 vLLM 都兼容）
#     RAG_LLM_MODEL    自定义模型名（如 gpt-4o-mini / qwen-plus / 本地模型名）
API_URL = os.environ.get("RAG_LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

DEFAULT_TOP_K = 4
# 相关性阈值。低于它的块当「没查到」——否则 tfidf 兜底会把一堆 0.01 分的噪声块
# 塞进上下文，AI 看到「查到了」就去编答，正好毁掉防幻觉闸门。
RELEVANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------- 工具：读 .env
def _load_env(dotenv_path=None):
    """从指定 .env 读环境变量（不覆盖已存在的 shell 变量之外，按 agent.py 的修复逻辑：
    .env 总是覆盖 session 变量，避免旧 key 压住新 key 导致 401）。"""
    if dotenv_path is None:
        # 默认找项目上层「AI学习」根的 .env（和 agent.py 同源）
        dotenv_path = os.path.join(BASE_DIR, "..", "..", ".env")
    dotenv_path = os.path.abspath(dotenv_path)
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            # .env 总是覆盖，确保换 key 后立刻生效
            os.environ[k] = v


def _clean_key(raw):
    raw = (raw or "").strip().strip("'\"")
    if any(ord(c) > 127 for c in raw):
        raise ValueError(
            "API key 含有非 ASCII 字符 —— 去平台重新复制一个纯英文的 sk-... key，"
            "别带空格和中文。"
        )
    return raw


def _key():
    """每次调大模型时实时读环境变量（先确保 .env 已加载）。"""
    _load_env()
    return _clean_key(os.environ.get("DEEPSEEK_API_KEY", ""))


# ---------------------------------------------------------------- 切块
def chunk_text(text, max_chars=320, overlap=64, mode="semantic"):
    """按模式切块。D13：默认走语义分块（不切断句子、识别章节）；mode="legacy"
    保留旧行为（固定字数硬切 + 重叠），保证向后兼容。"""
    return _chunker.chunk_text(text, mode=mode, max_chars=max_chars, overlap=overlap)


# ---------------------------------------------------------------- 入库
def _kb_path(namespace="default"):
    return os.path.join(KB_DIR, "kb_{0}.jsonl".format(namespace))


def ingest_documents(docs, namespace="default", kb_path=None, chunk_mode="semantic"):
    """docs = [{"filename": "...", "title": "...", "text": "..."}, ...]
    切块并打元数据 (doc_id, filename, title, chunk_id, text)，写入知识库。
    返回写入的块数。

    D13：新增 chunk_mode（semantic | legacy | proposition | template），
    默认语义分块；doc 里也可带 `modality`/`meta` 字段走 D17 跨模态入库。
    D20：doc 可带治理字段 `effective_date`（生效日期）/ `effective_until`
    （失效日期）/ `review_status`（审核状态），会下沉到每块的 meta，
    供召回前过滤（meta_filter）判定时效与审核态。
    """
    if kb_path is None:
        kb_path = _kb_path(namespace)
    rows = []
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    seen = set()
    added = 0
    for doc in docs:
        fn = doc.get("filename", "unknown")
        title = doc.get("title", fn)
        text = doc.get("text", "").strip()
        if not text:
            continue
        doc_id = re.sub(r"\W+", "", fn)[:40] or "doc"
        # D17：doc 可带 modality/meta（table / image），此时 text 已是渲染好的检索文本
        modality = doc.get("modality", "text")
        meta = doc.get("meta") or {}
        # D20：doc 顶层治理字段下沉到 meta（doc.meta 里写了的同名键优先，不覆盖）
        for k in ("effective_date", "effective_until", "expires_at", "review_status"):
            if doc.get(k) and not meta.get(k):
                meta[k] = doc[k]
        for i, ch in enumerate(chunk_text(text, mode=chunk_mode)):
            cid = "{0}_{1}".format(doc_id, i)
            if cid in seen:
                continue
            seen.add(cid)
            row = {
                "doc_id": doc_id,
                "filename": fn,
                "title": title.strip(),
                "chunk_id": cid,
                "text": ch,
                "modality": modality,
                # D13：记录切块方法，方便溯源与评测
                "chunk_method": chunk_mode,
            }
            if meta:
                row["meta"] = meta
            rows.append(row)
            added += 1
    with open(kb_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # P1：知识库变了 → 让向量库索引 / 图谱缓存失效，下次查询自动重建（不读到旧数据）
    try:
        import vector_store as vs
        vs.drop(namespace)
    except Exception:
        pass
    try:
        import graph_rag as gr
        gr.clear_cache(namespace)
    except Exception:
        pass
    return added


def load_kb(namespace="default", kb_path=None):
    if kb_path is None:
        kb_path = _kb_path(namespace)
    rows = []
    if not os.path.exists(kb_path):
        return rows
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def kb_stats(namespace="default"):
    rows = load_kb(namespace)
    if not rows:
        return {"rows": 0, "chunks": 0, "docs": [], "meta": _mf.filter_stats([])}
    docs = sorted({r["filename"] for r in rows})
    # D20：治理健康度——过期/未生效/未审核/作废各拦了多少
    return {"rows": len(rows), "chunks": len(rows), "docs": docs,
            "namespace": namespace, "meta": _mf.filter_stats(rows)}


# -------------------------------------------- D20：召回前元数据过滤（治理门禁）
# 最近一次检索的过滤统计（可观测：拦了几块、各什么原因），prepare_ask 透出给前端
_LAST_META_FILTER = {"hidden": 0, "reasons": []}


def _filter_rows(rows):
    """D20 统一过滤入口：过期/未生效/未审核/作废的块【连候选都不进】。

    返回可见块列表；过滤统计记入 _LAST_META_FILTER，供 answer 响应透出。
    GEOFlow 的核心思想：治理规则必须发生在代码层，不能指望 prompt 叮嘱 LLM。
    """
    global _LAST_META_FILTER
    visible, blocked = _mf.visible_rows(rows)
    _LAST_META_FILTER = {
        "hidden": len(blocked),
        "reasons": [{"chunk_id": r.get("chunk_id", ""), "title": r.get("title", ""),
                     "state": st, "reason": why}
                    for r, st, why in blocked[:10]],  # 最多透出 10 条，防响应膨胀
    }
    return visible


def last_meta_filter():
    return dict(_LAST_META_FILTER)


# -------------------------------------------- 检索：embedding 主，TF-IDF 兜底 --
def _build_index(chunks, namespace="default", chunk_ids=None):
    if os.environ.get("RAG_BACKEND") == "tfidf":
        return "tfidf", _TfidfIndex(chunks)
    # D12 补强（P1）：优先用持久化向量库（落盘索引，免每次重建）
    try:
        import vector_store as vs
        if chunk_ids is None:
            chunk_ids = [str(i) for i in range(len(chunks))]
        res = vs.build_or_load(namespace, chunks, chunk_ids)
        if res:
            return res
    except Exception:
        pass
    # 回落：内存重建（保持原行为，确保缺依赖也能跑）
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss

        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        model_name = "BAAI/bge-small-zh-v1.5"
        model = SentenceTransformer(model_name)
        vecs = model.encode(chunks, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return "semantic", (model, index)
    except Exception:
        # 任何失败（缺依赖/缓存坏/网络超时）一律回落 tfidf，绝不打断检索
        return "tfidf", _TfidfIndex(chunks)


class _TfidfIndex:
    """纯标准库 TF-IDF，语义弱但够关键词题用。离线也能跑。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [self._tok(c) for c in chunks]
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)
        self.idf = {w: math.log((self.n + 1) / (c + 1)) + 1 for w, c in self.df.items()}
        self.tfs = []
        for d in self.docs:
            c = Counter(d)
            total = len(d) or 1
            self.tfs.append({w: (n / total) for w, n in c.items()})

    def query(self, text, top_k=4):
        q = self._tok(text)
        scores = []
        for i, d in enumerate(self.docs):
            s = sum(self.idf.get(w, 0) * self.tfs[i].get(w, 0) for w in q if w in self.tfs[i])
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tok(text):
        out, cur = [], ""
        for ch in text:
            if ch.isalnum():
                cur += ch
            else:
                cur = cur.strip()
                if len(cur) >= 2:
                    out.append(cur.lower())
                cur = ""
        if len(cur.strip()) >= 2:
            out.append(cur.strip().lower())
        chs = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        for i in range(len(chs) - 1):
            out.append("".join(chs[i:i + 2]))
        return out


_INDEX_KEY = None
_INDEX_OBJ = None
_BACKEND = None


def _ensure_index(rows, namespace="default"):
    global _INDEX_KEY, _INDEX_OBJ, _BACKEND
    key = (namespace, tuple(r["chunk_id"] for r in rows))
    if not rows:
        return "tfidf", _TfidfIndex([])
    if _INDEX_KEY == key and _INDEX_OBJ is not None:
        return _BACKEND, _INDEX_OBJ
    backend, index = _build_index([r["text"] for r in rows], namespace,
                                  [r["chunk_id"] for r in rows])
    _INDEX_KEY = key
    _INDEX_OBJ = index
    _BACKEND = backend
    return backend, index


# -------------------------------------------- D2 补强（P0）：RRF 融合重排
def retrieve_candidates(query, top_k=DEFAULT_TOP_K, namespace="default",
                        kb_path=None, candidate_factor=3):
    """两路各多召回候选（top_k*candidate_factor），交给融合重排。

    语义检索擅长「意思相近但用词不同」，TF-IDF 擅长「精确关键词命中」。
    两路都只取 top_k 容易漏掉对方高 relevant 的结果，所以先各取宽一些，
    再在 rerank_fusion 里对齐排序。

    D20：两路召回共用同一个可见块集合——过期/未审核的块在【进入检索之前】
    就被 _filter_rows 挡掉，语义索引与 TF-IDF 天然看不到它们。
    """
    rows = load_kb(namespace, kb_path)
    if not rows:
        return [], []
    rows = _filter_rows(rows)
    if not rows:
        return [], []
    backend, index = _ensure_index(rows, namespace)
    sem_pairs = []
    if backend == "semantic":
        try:
            model, faiss_index = index
            import numpy as np
            vecs = model.encode([query], normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype("float32")
            cand = max(top_k * candidate_factor, 8)
            d, idxs = faiss_index.search(vecs, cand)
            for s, i in zip(d[0], idxs[0]):
                if i < 0 or i >= len(rows) or s <= 0:
                    continue
                sem_pairs.append((float(s), rows[i]))
            sem_pairs.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            sem_pairs = []
    tfidf_idx = _TfidfIndex([r["text"] for r in rows])
    tfidf_pairs = [(s, rows[i]) for s, i in tfidf_idx.query(query, max(top_k * candidate_factor, 8))]
    return sem_pairs, tfidf_pairs


def rerank_fusion(sem_pairs, tfidf_pairs, top_k=DEFAULT_TOP_K, k=60):
    """RRF（Reciprocal Rank Fusion）跨两路召回做重排。

    高端系统用 cross-encoder 重排（如 bge-reranker）；我们离线优先用 RRF——
    不引新模型、不烧算力，就能把「语义相关但关键词弱」和「关键词命中但语义偏」
    两类结果对齐排序，比单路直接取 top_k 精度更高。后续接 bge-reranker 时
    只需替换此函数（接口不变：喂两路候选，吐排好序的 top_k）。

    关键：融合分只用于「排序」，最终返回的得分仍用「原始最高相似度」
    （语义/关键词两路里较大的那个）。这样 grounded_ask 里的
    RELEVANCE_THRESHOLD 守门照常生效——不会因重排把一堆噪声块当命中，
    否则防幻觉闸门会被稀释（这是加 rerank 时最容易踩的坑，已在此规避）。
    """
    fused = {}
    for pairs in (sem_pairs, tfidf_pairs):
        for rank, (s, r) in enumerate(pairs, 1):
            cid = r["chunk_id"]
            entry = fused.setdefault(cid, [0.0, 0.0, r])  # [rrf分, 原始最高相似度, 块]
            entry[0] += 1.0 / (k + rank)
            entry[1] = max(entry[1], s)
    ranked = sorted(fused.values(), key=lambda x: -x[0])
    return [(sc, r) for _, sc, r in ranked[:top_k]]


def retrieve(query, top_k=DEFAULT_TOP_K, namespace="default", kb_path=None,
             rerank=True, candidate_factor=3):
    """返回 [(块 dict, 得分)]，默认经 RRF 融合重排（D2 补强）。

    设 rerank=False 可退回「单路直接取 top_k」的旧行为，方便对照评测。
    """
    sem_pairs, tfidf_pairs = retrieve_candidates(
        query, top_k=top_k, namespace=namespace, kb_path=kb_path,
        candidate_factor=candidate_factor)
    if rerank and (sem_pairs or tfidf_pairs):
        return rerank_fusion(sem_pairs, tfidf_pairs, top_k=top_k)
    return sem_pairs or tfidf_pairs


# ------------------------------------------------ 防幻觉 prompt
SYSTEM_GROUND = (
    "你是一个企业知识库问答助手，负责根据【参考资料】回答员工/用户关于公司内部文档的问题。"
    "【硬规矩】只能使用【参考资料】里的字，一个字都不能自己发挥。"
    "参考资料里没提到的，直接说「资料里没提到」，不许编、不许推测、不许用你训练时记得的东西。"
    "参考资料说不清楚，就照实说不清楚，不许替它圆。"
    "回答要简洁、口语化，像真人同事在解答。如果资料能回答，请在相关句子后用 [n] 标注引用了第几段资料。"
    # D17 多语言：用提问的语言回答，别默认中文
    "【语言】用用户提问时的语言回答：中文问题用中文，英文问题用英文，别自动翻译。"
    "最后可以加一句引导继续提问的话。"
)

USER_GROUND = (
    "【问题】{question}\n\n"
    "【参考资料】（每段前的 [n] 是资料编号，回答时请对应引用）\n{ctx}\n\n"
    "请只根据上面的参考资料回答。资料里没有就说资料里没提到。"
)


def _build_context(pairs, cap_chars=1200):
    """把命中块拼成带 [n] 编号的上下文，方便回答里引用。"""
    out, used = [], 0
    for n, (s, r) in enumerate(pairs, 1):
        line = "[{0}]《{1}》{2}".format(n, r.get("title", r.get("filename", "")), r["text"])
        if used + len(line) > cap_chars:
            # 超出预算的块仍保留前若干个，保证有料可答；硬截断文本而非整块丢弃
            remaining = cap_chars - used
            if remaining < 80:
                break
            line = line[:remaining] + "…"
        out.append(line)
        used += len(line)
    return "\n".join(out)


def ask_llm_raw(system, user, temperature=0.3, max_tokens=700):
    """底层 LLM 调用，system/user 都可自定义（D8 Agentic / D10 改写复用）。

    max_tokens 可覆盖：质检（schema_qc）要输出完整 JSON 台账，默认 700 可能截断。
    """
    key = _key()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，无法调大模型。")
    # D4 补强：每次调用时实时读环境变量，支持运行时切换模型/端点（模型路由）
    api_url = os.environ.get("RAG_LLM_API_URL", API_URL)
    model = os.environ.get("RAG_LLM_MODEL", MODEL)
    resp = requests.post(
        api_url,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def ask_llm(prompt, temperature=0.3):
    """默认用防幻觉 system 提示作答。"""
    return ask_llm_raw(SYSTEM_GROUND, prompt, temperature=temperature)

def ask_llm_stream(system, user, temperature=0.3):
    """D15 流式作答（P2 / 阶段C）。逐 token yield，让前端能边想边打，长答案不卡顿。
    端点不支持流式 / 出错时退化为 ask_llm_raw 整句，不中断。
    """
    key = _key()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，无法调大模型。")
    api_url = os.environ.get("RAG_LLM_API_URL", API_URL)
    model = os.environ.get("RAG_LLM_MODEL", MODEL)
    try:
        resp = requests.post(
            api_url,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": 700,
                "stream": True,
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except Exception:
                continue
    except Exception:
        # 流式失败（端点不支持 / 网络抖动）→ 退化成整句，保证接口不挂
        yield ask_llm_raw(system, user, temperature=temperature)




def rewrite_query(question, history):
    """D10 补强（P1）：把追问改写成可独立检索的完备问题。

    history = [{"q":..., "a":...}, ...]。高端系统都做这一步，否则「那加班餐补呢」
    这种承接上一轮的问题检索不到任何东西。
      - 含指代代词 / 过短 → 用上一轮问题补全
      - 在线时让 LLM 改写成独立问题；离线（RAG_TEST）直接拼接上一轮问题
    """
    if not history:
        return question
    last = history[-1]
    last_q = (last.get("q") or "").strip()
    last_a = (last.get("a") or "").strip()
    # 仅在「明显承接上一轮」时改写：含指代代词（它/这个/前者…）或追问词（为什么）。
    # 不含这些的完整问题（如「报销上限多少」）保持原样，避免画蛇添足反而干扰检索。
    pronouns = ("它", "他", "她", "这个", "那个", "这些", "那些", "前者", "后者",
                "上面", "前面", "上述", "这", "那", "为什么")
    if any(p in question for p in pronouns):
        combined = (last_q + " " + question).strip() if last_q else question
        if os.environ.get("RAG_TEST") == "1":
            return combined
        try:
            sys_p = ("你是查询改写助手。用户正在多轮追问，请把本轮问题改写成一个"
                     "脱离上下文也能独立检索的完整问题。只输出改写后的问题，不要解释。")
            user_p = "上一轮问题：{0}\n上一轮回答：{1}\n本轮追问：{2}".format(
                last_q, last_a, question)
            rewritten = ask_llm_raw(sys_p, user_p, temperature=0.1).strip()
            return rewritten or combined
        except Exception:
            return combined
    return question


def prepare_ask(question, top_k=DEFAULT_TOP_K, namespace="default",
                history=None, retrieval="default", modalities=None,
                use_cache=True):
    """检索 + 改写 + 压缩 + 模态过滤 + 拼 prompt。【不调模型】。

    拆出来的目的是让「普通问答」和「流式问答」共用同一套检索/压缩逻辑，
    只差最后一步：前者调 ask_llm 整句，后者调 ask_llm_stream 逐 token。

    返回 dict：{prompt, sources, effective_question, backend, hits, scores,
                compress, from_cache, retrieval, namespace}
    缓存命中时 prompt=None、from_cache=True，调方直接拿 answer 用。
    """
    effective_q = rewrite_query(question, history) if history else question

    # D15：先查缓存（命中则完全跳过检索与调模型；key 区分检索模式）
    if use_cache:
        cached = _cache.cache_get(namespace, effective_q, mode=retrieval)
        if cached:
            d = dict(cached)
            d["from_cache"] = True
            d["prompt"] = None
            return d

    # D7：检索模式
    if retrieval == "graph":
        try:
            import graph_rag as gr
            pairs = gr.retrieve_graph(effective_q, namespace=namespace, top_k=top_k)
            # 图谱抽不到实体 / 实体无匹配 → 回退语义检索，避免小库空手而归
            if not pairs:
                pairs = retrieve(effective_q, top_k=top_k, namespace=namespace)
        except Exception:
            pairs = retrieve(effective_q, top_k=top_k, namespace=namespace)
    else:
        pairs = retrieve(effective_q, top_k=top_k, namespace=namespace)

    # D17：按模态过滤（在阈值之前， modalities 之外的块连候选都不进）
    pairs = _mm.filter_by_modalities(pairs, modalities)
    pairs = [(s, r) for s, r in pairs if s >= RELEVANCE_THRESHOLD]

    # D14：压缩/去重上下文（替代原来的 _build_context 硬截断）
    ctx, compress_info = _context.compress_context(pairs)
    prompt = USER_GROUND.format(
        question=effective_q, ctx=ctx or "（没有检索到相关资料）")

    sources = [
        {"n": i + 1, "filename": r.get("filename", ""), "title": r.get("title", ""),
         "score": round(s, 3), "snippet": r["text"][:200],
         "text": r["text"],  # D19：全文，供引用内容回验（snippet 200 字会截断依据）
         "modality": r.get("modality", "text")}
        for i, (s, r) in enumerate(pairs)]

    return {
        "prompt": prompt,
        "sources": sources,
        "scores": [s for s, _ in pairs],
        "effective_question": effective_q,
        "backend": _BACKEND or "tfidf",
        "hits": len(pairs),
        "compress": compress_info,
        "from_cache": False,
        "retrieval": retrieval,
        "namespace": namespace,
        # D20：治理可观测——本次检索拦了哪些过期/未审核块
        "meta_filtered": _LAST_META_FILTER,
    }


def grounded_ask(question, top_k=DEFAULT_TOP_K, namespace="default",
                 verbose=True, return_sources=True, history=None,
                 retrieval="default", modalities=None, use_cache=True):
    """检索 → 提示 → 大模型回答。返回 dict，含 answer 与 sources(可点击溯源)。

    D10：传 history（多轮历史）会自动改写追问，让承接上一轮的问题也能检索到。
    D7 ：retrieval="graph" 走 GraphRAG 多跳检索（默认仍是语义+关键词融合）。
    D14：上下文先压缩/去重，再送进 prompt。
    D15：use_cache=True 时相同问题命中缓存，不重复检索+调模型。
    D16：额外输出 confidence / confidence_level / confidence_reason。
    D17：modalities 可限定只检索 text/table/image 中的某几种模态。
    D20：召回前元数据过滤——过期/未生效/未审核/作废的块不进检索，
         响应里带 meta_filtered 说明拦了什么（治理可观测）。
    """
    prepared = prepare_ask(
        question, top_k=top_k, namespace=namespace, history=history,
        retrieval=retrieval, modalities=modalities, use_cache=use_cache)

    if prepared["from_cache"]:
        out = dict(prepared)
        out.setdefault("from_cache", True)
        out.setdefault("meta_filtered", _LAST_META_FILTER)
        if verbose:
            print("📦 命中缓存，跳过检索与调模型：" + question)
        return out

    prompt = prepared["prompt"]
    if os.environ.get("RAG_TEST") == "1":
        answer = "（RAG_TEST=1 离线模式，未调用大模型）"
    else:
        answer = ask_llm(prompt)

    # D16：先用启发式算一版置信度，作为质检失败时的兜底基线
    conf = _confidence.score_confidence(
        [(s, {}) for s in prepared["scores"]], answer,
        threshold=RELEVANCE_THRESHOLD)

    # JSON Schema 硬约束质检（默认开；RAG_QC_MODE=off 可关）
    # 拿到结构化证据台账后，用证据比例重算落地性，不再靠拒答正则猜
    qc = _qc.run_qc(prepared["effective_question"], answer, prepared["sources"],
                    hits=prepared["hits"], level=conf["level"], verbose=verbose)
    qc_action = "warn"
    if qc and (qc.get("mode", "") or "").startswith("json"):
        conf = _confidence.score_confidence(
            [(s, {}) for s in prepared["scores"]], answer,
            threshold=RELEVANCE_THRESHOLD, qc=qc)
        # 门禁动作：RAG_QC_ACTION=block 时把无依据答案拦下来
        answer, qc_action = _qc.apply_action(qc, answer)

    # D15：缓存真实答案（含置信度，命中时一起复用）。离线占位答案不缓存。
    if os.environ.get("RAG_TEST") != "1":
        try:
            _cache.cache_set(prepared["namespace"], prepared["effective_question"],
                mode=retrieval, value={
                "answer": answer,
                "sources": prepared["sources"],
                "backend": prepared["backend"],
                "hits": prepared["hits"],
                "effective_question": prepared["effective_question"],
                "retrieval": prepared["retrieval"],
                "compress": prepared["compress"],
                "confidence": conf["confidence"],
                "confidence_level": conf["level"],
                "confidence_reason": conf["reason"],
                "confidence_factors": conf["factors"],
                "qc": qc,
                "qc_action": qc_action,
            })
        except Exception:
            pass

    out = {"question": question, "answer": answer, "sources": prepared["sources"],
           "backend": prepared["backend"], "hits": prepared["hits"],
           "effective_question": prepared["effective_question"],
           "retrieval": retrieval,
           "confidence": conf["confidence"],
           "confidence_level": conf["level"],
           "confidence_reason": conf["reason"],
           "confidence_factors": conf["factors"],
           "qc": qc, "qc_action": qc_action,
           "compress": prepared["compress"], "from_cache": False}
    if verbose:
        print("=" * 70)
        print("❓ " + question)
        if prepared["effective_question"] != question:
            print("🔄 改写后检索问句：" + prepared["effective_question"])
        print("🔍 命中 {0} 块（后端：{1}｜模式：{2}）".format(
            prepared["hits"], prepared["backend"], retrieval))
        mf = prepared.get("meta_filtered") or {}
        if mf.get("hidden"):
            print("🛡 治理过滤：{0} 块过期/未生效/未审核/作废，召回前已拦截".format(mf["hidden"]))
        for s in prepared["sources"]:
            print("  [{0}] {1}（{2}｜{3}）".format(
                s["n"], s["title"], s["filename"], s["score"]))
        print("💬 " + answer)
        print("📊 置信度：{0}（{1}）— {2}".format(
            conf["confidence"], conf["level"], "；".join(conf["reason"])))
        if qc:
            s = qc.get("summary") or {}
            print("🧪 质检：{0}（模式 {1}｜断言 {2} 条：{3} 有据 / {4} 弱据 / {5} 无据）".format(
                qc.get("status"), qc.get("mode"), s.get("claims", 0),
                s.get("supported", 0), s.get("weak", 0), s.get("unsupported", 0)))
            if qc.get("errors"):
                print("   ⚠️ 校验问题：" + "；".join(qc["errors"][:3]))
    return out


# ----------------------------------------------------------------- 事实核查闸门
SPAN_STOP = {
    "今天", "今日", "发布", "宣布", "上线", "公测", "开源", "模型", "参数",
    "架构", "技术", "系统", "视频", "生成", "落地", "正式", "支持",
    "接口", "API", "公司", "我们", "文档", "相关", "内容", "信息",
}


def _salient(claim):
    salient = []
    cur = []
    for ch in claim:
        if "\u4e00" <= ch <= "\u9fff":
            cur.append(ch)
        else:
            if len(cur) >= 2:
                salient.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        salient.append("".join(cur))
    salient.extend(
        w for w in re.findall(r"\d[\dKk%±\.]+", claim)
        if re.search(r"\d{3,}", w) or w[-1] in "KkMB%±"
    )
    salient.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-\.]{1,}", claim))
    return {w for w in salient if w not in SPAN_STOP and len(w) > 1}


def claim_verdict(claim, namespace="default", top_k=8):
    """三道闸门，把一条宣称收敛成四种状态：
    ✅出处可查 / 🟡需补标注 / 🚨查到的是另一件事 / ❌资料里没提到
    企业场景主要用来核对「对外答复/制度引用」是否有文档撑腰。
    """
    pairs = retrieve(claim, top_k=top_k, namespace=namespace)
    candidates = [(i + 1, s, c) for i, (s, c) in enumerate(pairs)
                  if c is not None and s >= RELEVANCE_THRESHOLD]
    if not candidates:
        return "❌", "资料里没提到", 0.0, None, 0

    need = _salient(claim)
    if need:
        if not any(_has(c, need) for _, _, c in candidates):
            rank, s, ch = candidates[0]
            return ("🚨", "查到了同主题的块，但找不到该宣称里的具体内容", s, ch, rank)

    alpha_want = {w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-\.]{1,}", claim)
                  if w not in SPAN_STOP and len(w) > 1}
    digit_want = {w for w in re.findall(r"\d[\dKk%±\.]+", claim)
                  if re.search(r"\d{3,}", w) or w[-1] in "KkMB%±"}
    must = alpha_want | digit_want

    def _has(chunk, words):
        text = (chunk or {}).get("text") or ""
        return {w for w in words if w in text}

    winners = [(r, s, c) for r, s, c in candidates if (not must) or _has(c, must) == must]
    if must and not winners:
        anywhere = set().union(*(_has(c, must) for _, _, c in candidates)) or set()
        missing = [w for w in must if w not in anywhere]
        rank, s, ch = candidates[0]
        return ("🚨", "缺了宣称里写明的实体/数字：" + "、".join(missing[:6])
                + "（窗口里没有任何块覆盖全句，不能当出处）", s, ch, rank)

    if not must:
        winners = candidates

    rank, s, chunk = sorted(winners, key=lambda t: -t[1])[0]
    return "✅", "出处可查", s, chunk, rank


# ----------------------------------------------------------------- CLI 自检
def _main():
    import sys
    args = sys.argv[1:]
    if not args or args[0] == "stats":
        ns = args[1] if len(args) > 1 else "default"
        st = kb_stats(ns)
        print("知识库[{0}] 共 {1} 个块，{2} 份文档".format(ns, st["chunks"], len(st["docs"])))
        for d in st["docs"]:
            print("  -", d)
        return
    if args[0] == "ask":
        q = " ".join(args[1:])
        grounded_ask(q)
        return
    grounded_ask(" ".join(args))


if __name__ == "__main__":
    _main()
