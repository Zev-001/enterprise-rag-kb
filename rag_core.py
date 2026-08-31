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

# ---------------------------------------------------------------- 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 知识库存这里：每个「知识库(namespace)」一个 jsonl 文件，实现多库隔离
KB_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(KB_DIR, exist_ok=True)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
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
def chunk_text(text, max_chars=320, overlap=64):
    """按段落切块，长段硬切并留重叠。通用版：不假设文档结构。"""
    text = (text or "").strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(cur) + len(b) <= max_chars:
            cur = (cur + "\n" + b).strip() if cur else b
        else:
            if cur:
                chunks.append(cur)
            if len(b) > max_chars:
                for i in range(0, len(b), max_chars - overlap):
                    chunks.append(b[i:i + max_chars])
                cur = ""
            else:
                cur = b
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- 入库
def _kb_path(namespace="default"):
    return os.path.join(KB_DIR, "kb_{0}.jsonl".format(namespace))


def ingest_documents(docs, namespace="default", kb_path=None):
    """docs = [{"filename": "...", "title": "...", "text": "..."}, ...]
    切块并打元数据 (doc_id, filename, title, chunk_id, text)，写入知识库。
    返回写入的块数。
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
        for i, ch in enumerate(chunk_text(text)):
            cid = "{0}_{1}".format(doc_id, i)
            if cid in seen:
                continue
            seen.add(cid)
            rows.append({
                "doc_id": doc_id,
                "filename": fn,
                "title": title.strip(),
                "chunk_id": cid,
                "text": ch,
            })
            added += 1
    with open(kb_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
        return {"rows": 0, "chunks": 0, "docs": []}
    docs = sorted({r["filename"] for r in rows})
    return {"rows": len(rows), "chunks": len(rows), "docs": docs, "namespace": namespace}


# -------------------------------------------- 检索：embedding 主，TF-IDF 兜底 --
def _build_index(chunks):
    if os.environ.get("RAG_BACKEND") == "tfidf":
        return "tfidf", _TfidfIndex(chunks)
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


def _ensure_index(rows):
    global _INDEX_KEY, _INDEX_OBJ, _BACKEND
    key = tuple(r["chunk_id"] for r in rows)
    if not rows:
        return "tfidf", _TfidfIndex([])
    if _INDEX_KEY == key and _INDEX_OBJ is not None:
        return _BACKEND, _INDEX_OBJ
    backend, index = _build_index([r["text"] for r in rows])
    _INDEX_KEY = key
    _INDEX_OBJ = index
    _BACKEND = backend
    return backend, index


def retrieve(query, top_k=DEFAULT_TOP_K, namespace="default", kb_path=None):
    """返回 [(块 dict, 得分)]。"""
    rows = load_kb(namespace, kb_path)
    if not rows:
        return []
    backend, index = _ensure_index(rows)
    if backend == "semantic":
        try:
            model, faiss_index = index
            import numpy as np
            vecs = model.encode([query], normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype("float32")
            d, idxs = faiss_index.search(vecs, top_k)
            pairs = []
            for s, i in zip(d[0], idxs[0]):
                if i < 0 or i >= len(rows):
                    continue
                if s <= 0:
                    continue
                pairs.append((float(s), rows[i]))
            pairs.sort(key=lambda x: x[0], reverse=True)
            return pairs
        except Exception:
            pass
    idx = _TfidfIndex([r["text"] for r in rows])
    out = []
    for s, i in idx.query(query, top_k):
        out.append((s, rows[i]))
    return out


# ------------------------------------------------ 防幻觉 prompt
SYSTEM_GROUND = (
    "你是一个企业知识库问答助手，负责根据【参考资料】回答员工/用户关于公司内部文档的问题。"
    "【硬规矩】只能使用【参考资料】里的字，一个字都不能自己发挥。"
    "参考资料里没提到的，直接说「资料里没提到」，不许编、不许推测、不许用你训练时记得的东西。"
    "参考资料说不清楚，就照实说不清楚，不许替它圆。"
    "回答要简洁、口语化，像真人同事在解答。如果资料能回答，请在相关句子后用 [n] 标注引用了第几段资料。"
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


def ask_llm(prompt, temperature=0.3):
    key = _key()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，无法调大模型。")
    resp = requests.post(
        API_URL,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_GROUND},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 700,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def grounded_ask(question, top_k=DEFAULT_TOP_K, namespace="default",
                 verbose=True, return_sources=True):
    """检索 → 提示 → 大模型回答。返回 dict，含 answer 与 sources(可点击溯源)。"""
    pairs = retrieve(question, top_k=top_k, namespace=namespace)
    pairs = [(s, r) for s, r in pairs if s >= RELEVANCE_THRESHOLD]
    ctx = _build_context(pairs)
    prompt = USER_GROUND.format(question=question, ctx=ctx or "（没有检索到相关资料）")

    sources = [
        {"n": i + 1, "filename": r.get("filename", ""), "title": r.get("title", ""),
         "score": round(s, 3), "snippet": r["text"][:200]}
        for i, (s, r) in enumerate(pairs)
    ]

    if os.environ.get("RAG_TEST") == "1":
        answer = "（RAG_TEST=1 离线模式，未调用大模型）"
    else:
        answer = ask_llm(prompt)

    out = {"question": question, "answer": answer, "sources": sources,
           "backend": _BACKEND or "tfidf", "hits": len(pairs)}
    if verbose:
        print("=" * 70)
        print("❓ " + question)
        print("🔍 命中 {0} 块（后端：{1}）".format(len(pairs), out["backend"]))
        for s in sources:
            print("  [{0}] {1}（{2}｜{3}）".format(s["n"], s["title"], s["filename"], s["score"]))
        print("💬 " + answer)
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
