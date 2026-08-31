"""
evidence_verify.py —— 引用内容级回验（D19，借鉴 GEOFlow validateEvidenceSnapshot）。

## 为什么要这个模块

D18 的 schema_qc 解决了两件事：编号越界（引用 [9] 但只召回 2 段）和
模型自报的证据台账。但还剩一个漏网之鱼——**张冠李戴**：

    答案说「年假有 5 天 [2]」，而 [2] 那块讲的是报销标准。

编号合法（没越界）、模型自己也标了 supported，但引用的块内容根本
支撑不了这句话。质检员（LLM）也可能犯这个错，因为它看到的资料
是 snippet 截断的，或者干脆就是看走眼。

GEOFlow 的做法是 validateEvidenceSnapshot：**发布前拿引用块的真实
内容回验一遍**。本模块把这套搬到 RAG 问答上——

    对每条 claim，拿它引用的块的【全文】做内容相似度校验，
    相似度不够就把 LLM 给的 evidence_status 往下修。

## 三层评分（快 → 慢，本地零 API 调用）

  ① content_hash 精确命中：归一化后 claim 是块的子串 → 1.0
  ② 字符 bigram 重叠（containment）：零依赖兜底，中文友好
  ③ bge 向量余弦：复用 vector_store._MODEL（检索用的同一个模型），
     归一化向量点积即余弦；模型不可用自动退回 ②

最终 score = max(各层得分)。阈值：
  语义可用   ≥0.50 supported / ≥0.30 weak / 否则 unsupported
  仅 n-gram  ≥0.35 supported / ≥0.15 weak / 否则 unsupported
  （n-gram 对缩写/换说法偏保守，阈值放低）

## 安全设计（三条铁律）

  1. **只降不升**：回验只能把 LLM 判的 supported→weak→unsupported
     往下修，绝不往上修——宁可错杀为 weak，不把幻觉洗白。
  2. **回验失败不伤主流程**：模型加载失败/异常 → 返回原 qc 原样。
  3. **可关**：RAG_EVIDENCE_VERIFY=off 完全跳过（默认 on）。

## 依赖

纯标准库 + 可选 sentence_transformers（检索已在用，不新增依赖）。
"""
import hashlib
import os
import re

# ----------------------------------------------------------------- 块指纹
_PUNCT_RE = re.compile(
    r"""[\s，。、；：？！,.:;?!()\[\]{}【】《》<>—_·…“”‘’"'\-]""")


def normalize(text):
    """比对前先归一化：去空白标点、全角转半角、转小写。"""
    if not text:
        return ""
    s = str(text)
    # 全角字母数字转半角
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    s = "".join(out)
    s = _PUNCT_RE.sub("", s)
    return s.lower()


def content_hash(text):
    """块/断言的内容指纹（归一化后 md5 前 16 位）。

    同一内容换行、空格、全半角、标点差异 → 同一指纹。
    用途：精确匹配快路径 + 未来的缓存指纹失效（GEOFlow input_fingerprint 同思想）。
    """
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------- n-gram 重叠
def _char_bigrams(s):
    """字符 bigram 集合（已归一化的串），中文比分词友好且零依赖。"""
    n = normalize(s)
    if len(n) < 2:
        return {n} if n else set()
    return {n[i:i + 2] for i in range(len(n) - 1)}


def ngram_support(claim, chunk):
    """claim 的 bigram 有多少比例出现在 chunk 里（containment，0–1）。

    用 containment 而非 jaccard：claim 是块的浓缩转述，块远长于 claim，
    jaccard 会被块长稀释成虚低。
    """
    cb = _char_bigrams(claim)
    if not cb:
        return 0.0
    kb = _char_bigrams(chunk)
    if not kb:
        return 0.0
    return len(cb & kb) / float(len(cb))


# ------------------------------------------------------------- 向量余弦
def semantic_sim(claim, chunk):
    """bge 归一化向量点积 = 余弦。模型不可用返回 None（退回 n-gram）。

    复用 vector_store._MODEL：检索用什么模型，回验就用什么模型，
    分数口径一致，且不新增任何依赖/内存。
    加载前强制 HF 离线：模型已在本地缓存（检索用过），避免运行时
    联网校验版本被代理/断网卡死（沙箱实测 502）。
    """
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import vector_store as _vs
        model = getattr(_vs, "_MODEL", None)
        if model is None:
            # 检索没跑过时模型还没加载；惰性加载一次
            from sentence_transformers import SentenceTransformer
            model = _vs._MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        pair = [normalize(claim) or claim, normalize(chunk) or chunk]
        vecs = model.encode(pair, normalize_embeddings=True,
                            show_progress_bar=False)
        a, b = vecs[0], vecs[1]
        return float(sum(x * y for x, y in zip(a, b)))
    except Exception:
        return None


# ------------------------------------------------------------- 单条断言回验
def verify_claim(claim_text, chunk_texts):
    """回验一条断言对若干候选块的支持度。

    返回 {"score": 综合展示分, "cos": 余弦或None, "ngram": 重叠度,
          "method": str, "best_chunk": int(1 起, 0=无)}

    注意 cos 与 ngram **分开保留、不做 max 合并**：bge 对「短句 vs 长块」
    分数虚高（实测张冠李戴对也能到 0.50），单看绝对线会误判；
    判定走 _verdict 的双信号规则。
    """
    best = {"score": 0.0, "cos": None, "ngram": 0.0,
            "method": "none", "best_chunk": 0}
    if not chunk_texts:
        return best
    used_sem = False
    for i, ch in enumerate(chunk_texts, start=1):
        if not ch:
            continue
        # 快路径：归一化子串精确命中
        nc, nch = normalize(claim_text), normalize(ch)
        if nc and nc in nch:
            return {"score": 1.0, "cos": None, "ngram": 1.0,
                    "method": "exact", "best_chunk": i}
        ng = ngram_support(claim_text, ch)
        cos = semantic_sim(claim_text, ch)
        if cos is not None:
            used_sem = True
        score = max(ng, cos or 0.0)
        better = (score > best["score"] or best["method"] == "none")
        if better:
            best = {"score": round(score, 3), "cos": cos, "ngram": ng,
                    "method": "semantic+ngram" if used_sem else "ngram",
                    "best_chunk": i}
    return best


def _verdict(cos, ng, has_sem):
    """双信号判据。

    - 仅 n-gram（模型不可用）：绝对线，保守阈值
    - 语义可用：cos 主判，ngram 做强补充（任一命中即可）
      阈值按实测校准：相关对 ~0.73，张冠李戴对 ~0.50，无关对 <0.40
    """
    if not has_sem:
        return ("supported" if ng >= 0.35
                else "weak" if ng >= 0.15 else "unsupported")
    if cos is not None and cos >= 0.62:
        return "supported"
    if ng >= 0.35:
        return "supported"
    if (cos is not None and cos >= 0.42) or ng >= 0.15:
        return "weak"
    return "unsupported"


# ------------------------------------------------------------- 整体回验
def reverify(qc, sources, verbose=False):
    """对 schema_qc 产出的 qc 做引用内容级回验（只降不升）。

    - sources：prepare_ask 的 sources（每条含 text 全文或 snippet）
    - 返回 (qc, changed)；失败/关闭时返回 (原 qc, False) 不伤主流程
    """
    if not qc or not isinstance(qc, dict):
        return qc, False
    if os.environ.get("RAG_EVIDENCE_VERIFY", "1").strip().lower() == "off":
        return qc, False
    claims = qc.get("claims") or []
    if not claims or not sources:
        return qc, False

    try:
        changed = False
        ver_rows = []
        downgrade_reasons = []
        for c in claims:
            keys = c.get("evidence_keys") or []
            if not keys:
                ver_rows.append({"claim": c["text"], "score": None,
                                 "note": "无引用编号，跳过"})
                continue
            # 引用块的全文（优先 text，退化 snippet）
            chunks = []
            for k in keys:
                if 1 <= k <= len(sources):
                    s = sources[k - 1]
                    chunks.append(s.get("text") or s.get("snippet") or "")
                else:
                    chunks.append("")  # 越界（schema_qc 已处理），给空串
            v = verify_claim(c["text"], chunks)
            verdict = _verdict(v["cos"], v["ngram"],
                               v["method"] in ("semantic+ngram", "exact"))
            llm_status = c.get("evidence_status")
            row = {"claim": c["text"], "score": v["score"],
                   "method": v["method"], "verdict": verdict,
                   "llm_said": llm_status}
            # 铁律：只降不升
            rank = {"supported": 2, "weak": 1, "unsupported": 0}
            if llm_status in rank and rank[verdict] < rank[llm_status]:
                c["evidence_status"] = verdict
                changed = True
                msg = ("引用内容回验：断言「{0}」{1}（score={2}），"
                       "由 {3} 降为 {4}".format(
                           c["text"][:30], "内容不符" if verdict == "unsupported"
                           else "依据偏弱", v["score"], llm_status, verdict))
                row["downgraded"] = msg
                downgrade_reasons.append(msg)
            ver_rows.append(row)

        qc["verification"] = {"rows": ver_rows,
                              "downgraded": len(downgrade_reasons)}
        if changed:
            qc["reasons"] = (qc.get("reasons") or []) + downgrade_reasons
            qc["valid"] = False  # 有修正，标记被回验改过（json_repaired 语义）
            qc["mode"] = "json_reverified"
            # summary 重算（evidence_ratio 变了，置信度融合吃这个）
            import schema_qc as _sq
            qc["summary"] = _sq.summarize(qc)
        if verbose:
            print("🔍 引用回验：{0} 条断言，降级 {1} 条".format(
                len(ver_rows), len(downgrade_reasons)))
        return qc, changed
    except Exception as e:
        # 任何异常都不伤主流程：原样返回
        qc.setdefault("verification", {"error": str(e)})
        return qc, False
