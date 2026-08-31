"""
schema_qc.py —— JSON Schema 硬约束质检（借鉴 GEOFlow 的 AI 质量门禁）。

## 为什么要这个模块

原来的「这个答案可不可信」全靠 confidence.py 里的**中文正则**猜：

    _REFUSE_RE = re.compile(r"资料里没提到|没有检索到|我不了解|无法确定|没有找到")

三个硬伤：
  1. 换个说法就漏判 —— 「现有材料无法支撑这个结论」匹配不到；
  2. 英文场景直接瞎 —— D17 做了多语言，但拒答检测还是只有中文词表；
  3. 引用从不校验 —— 模型编一个 [3]（实际只召回 2 段）也照常高分通过。

GEOFlow 的做法是**让模型自己结构化呈堂证供**，再用 schema 硬性校验：
质检结果必须是 JSON，含 claim / evidence_keys / evidence_status 三态，
不合格就进不了发布流程。本模块把这套搬到 RAG 问答上。

## 质检 JSON 契约（QC_SCHEMA）

{
  "status": "answered" | "partial" | "refused" | "unsupported",
      answered    完全有据可答
      partial     答了一部分，另一部分在资料外
      refused     资料里没有，整体拒答
      unsupported 答案有实质内容，但没有证据支撑（最危险 = 幻觉嫌疑）
  "confidence": 0.0–1.0,          模型自评
  "claims": [                     逐条断言的证据台账
    {
      "text": "年假有5天",
      "evidence_keys": [1],       引用了第几段资料（必须落在 1..n_sources）
      "evidence_status": "supported" | "weak" | "unsupported"
    }
  ],
  "missing": "资料里缺少xxx",     可选，说明缺什么
  "reasons": ["..."]              可选，判定理由
}

## 三道硬校验（不靠模型自觉）

  ① schema 校验：required / 类型 / 枚举 / 值域，缺一个字段就失败
  ② 证据回验：evidence_keys 越界（超出实际召回数）→ 该条 claim 强制
     降为 unsupported，并记「引用编号越界」理由（抓引用幻觉）
  ③ 一致性校验：status=answered 但 claims 全 unsupported → 判为自相矛盾，
     降级为 unsupported

任一步失败 = 结构化质检不可用 → 自动回退到正则启发式（mode 标
heuristic_fallback），**保证不比改造前更差**。

## 开关

  RAG_QC_MODE    json（默认，走 LLM 结构化质检）
                 off（完全关闭，退回纯启发式）
  RAG_QC_ACTION  warn（默认，只打标）
                 block（status=unsupported 时拦截答案，换成拒答话术）

## 依赖

纯标准库，零新依赖。校验器自己写（比引 jsonschema 更容易给出中文错误），
JSON_SCHEMA_TEXT 里附等价的 JSON Schema 文本，方便对外/面试讲契约。
"""
import json
import os
import re

# ----------------------------------------------------------------- 契约定义
STATUS_ENUM = ("answered", "partial", "refused", "unsupported")
CLAIM_STATUS_ENUM = ("supported", "weak", "unsupported")

# 等价的 JSON Schema 文本（给文档/对外契约用；运行时校验走下面的 Python 校验器）
JSON_SCHEMA_TEXT = json.dumps({
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RagAnswerQualityCheck",
    "type": "object",
    "required": ["status", "confidence", "claims", "reasons"],
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": list(STATUS_ENUM)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "evidence_keys", "evidence_status"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_keys": {"type": "array",
                                      "items": {"type": "integer", "minimum": 1}},
                    "evidence_status": {"type": "string",
                                        "enum": list(CLAIM_STATUS_ENUM)},
                },
            },
        },
        "missing": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}, ensure_ascii=False, indent=2)


SYSTEM_QC = (
    "你是企业知识库问答的质检员。给你【问题】、【参考资料】和【待检答案】，"
    "你要判断答案里的每句话是不是真能从资料里找到依据。\n"
    "【输出格式】只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块、不要前缀后缀。\n"
    "【字段要求】\n"
    "- status: 四选一 —— answered(完全有据)/partial(部分有据)/refused(资料里没有)/"
    "unsupported(有实质内容但找不到依据)\n"
    "- confidence: 0 到 1 之间的小数，你对这个答案的整体可信度\n"
    "- claims: 数组，把答案拆成若干条断言，每条给 text(断言原文)、"
    "evidence_keys(依据的资料编号数组，编号从 1 开始，只填真正能支撑它的编号，"
    "没有就填空数组)、evidence_status(supported/weak/unsupported)\n"
    "- missing: 一句话说明资料里缺少什么（没有就空字符串）\n"
    "- reasons: 判定理由，字符串数组\n"
    "【铁律】evidence_keys 只能填【参考资料】里真实存在的编号；"
    "资料里没有的内容，evidence_keys 必须是空数组、evidence_status 必须是 unsupported。"
)

USER_QC = (
    "【问题】{question}\n\n"
    "【参考资料】（共 {n} 段，编号 1 到 {n}）\n{ctx}\n\n"
    "【待检答案】\n{answer}\n\n"
    "请只输出质检 JSON。"
)


# --------------------------------------------------------------- 严格解析
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text):
    """从模型输出里抠出 JSON 对象。

    模型常见的四种「不乖」都要兜住：
      1. 直接就是纯 JSON
      2. 用 ```json 代码块包着
      3. 前后带一句废话（「好的，这是质检结果：」/「希望有帮助」）
      4. 尾随逗号 / 中文引号
    返回 (obj, err)：成功 err=None；失败 obj=None。
    """
    if text is None:
        return None, "empty"
    raw = text.strip()

    candidates = []
    # 1) 整段
    candidates.append(raw)
    # 2) 代码块
    for m in _FENCE_RE.finditer(raw):
        candidates.append(m.group(1).strip())
    # 3) 首个 { 到最后一个 }
    i, j = raw.find("{"), raw.rfind("}")
    if i >= 0 and j > i:
        candidates.append(raw[i:j + 1])

    for cand in candidates:
        obj = _loads_lenient(cand)
        if isinstance(obj, dict):
            return obj, None
    return None, "no_json_object"


def _loads_lenient(s):
    """宽松 json.loads：吃掉尾随逗号、中文引号、BOM。"""
    if not s:
        return None
    s = s.strip().lstrip("\ufeff")
    # 中文引号 → 英文引号（模型偶发）
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(s)
    except Exception:
        pass
    # 尾随逗号：, } 或 , ]
    s2 = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except Exception:
        return None


# --------------------------------------------------------------- schema 校验
def validate_qc(obj, n_sources=0):
    """校验质检 JSON。返回 (qc, errors)；qc 为规范化后的对象（errors 非空时也可返回尽力修复版）。

    校验分三层：schema → 证据回验 → 一致性。见模块 docstring。
    """
    errors = []
    if not isinstance(obj, dict):
        return None, ["根节点不是 JSON 对象"]

    qc = {}

    # ---- status
    st = obj.get("status")
    if not isinstance(st, str):
        errors.append("status 缺失或不是字符串")
        st = None
    else:
        st = st.strip().lower()
        if st not in STATUS_ENUM:
            errors.append("status 取值非法：{0}（应为 {1}）".format(st, "/".join(STATUS_ENUM)))
            st = None
    qc["status"] = st

    # ---- confidence
    cf = obj.get("confidence")
    if isinstance(cf, bool) or not isinstance(cf, (int, float)):
        errors.append("confidence 缺失或不是数字")
        cf = None
    else:
        cf = float(cf)
        if cf < 0 or cf > 1:
            errors.append("confidence 越界：{0}（应在 0–1）".format(cf))
            cf = max(0.0, min(1.0, cf))
        cf = round(cf, 3)
    qc["confidence"] = cf

    # ---- claims
    raw_claims = obj.get("claims")
    if not isinstance(raw_claims, list):
        errors.append("claims 缺失或不是数组")
        raw_claims = []
    claims = []
    for idx, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            errors.append("claims[{0}] 不是对象".format(idx))
            continue
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append("claims[{0}].text 缺失或为空".format(idx))
            continue
        keys = c.get("evidence_keys")
        if keys is None:
            keys = []
        if not isinstance(keys, list):
            errors.append("claims[{0}].evidence_keys 不是数组".format(idx))
            keys = []
        norm_keys = []
        for k in keys:
            try:
                ki = int(k)
            except Exception:
                errors.append("claims[{0}].evidence_keys 含非整数：{1}".format(idx, k))
                continue
            norm_keys.append(ki)
        cst = c.get("evidence_status")
        if not isinstance(cst, str):
            errors.append("claims[{0}].evidence_status 缺失".format(idx))
            cst = None
        else:
            cst = cst.strip().lower()
            if cst not in CLAIM_STATUS_ENUM:
                errors.append(
                    "claims[{0}].evidence_status 取值非法：{1}".format(idx, cst))
                cst = None
        claims.append({"text": text.strip(), "evidence_keys": norm_keys,
                       "evidence_status": cst})
    qc["claims"] = claims

    # ---- missing / reasons
    missing = obj.get("missing", "")
    if not isinstance(missing, str):
        errors.append("missing 不是字符串")
        missing = str(missing)
    qc["missing"] = missing.strip()

    reasons = obj.get("reasons")
    if reasons is None:
        reasons = []
    if not isinstance(reasons, list):
        errors.append("reasons 不是数组")
        reasons = [str(reasons)]
    qc["reasons"] = [str(r).strip() for r in reasons if str(r).strip()]

    # ---- ② 证据回验（抓引用幻觉）
    if n_sources > 0:
        for c in claims:
            bad = [k for k in c["evidence_keys"] if k < 1 or k > n_sources]
            if bad:
                c["evidence_status"] = "unsupported"
                c["evidence_keys"] = [k for k in c["evidence_keys"]
                                      if 1 <= k <= n_sources]
                errors.append(
                    "claims 引用编号越界：{0}（资料共 {1} 段）".format(bad, n_sources))
                qc["reasons"].append(
                    "引用编号 {0} 越界（实际资料共 {1} 段），该断言判为无依据".format(
                        bad, n_sources))

    # ---- ③ 一致性：status=answered 但全部无据 → 自相矛盾
    if st == "answered" and claims:
        supported = [c for c in claims if c["evidence_status"] == "supported"]
        if not supported:
            qc["status"] = "unsupported"
            errors.append("status=answered 但没有任何 supported 断言，降级为 unsupported")
            qc["reasons"].append("自称有据但无一条断言被资料支撑，判为无依据")

    qc["errors"] = errors
    qc["valid"] = not errors
    return qc, errors


# --------------------------------------------------------------- 汇总统计
def summarize(qc):
    """从 claims 台账算出证据比例，供置信度融合用。"""
    if not qc:
        return {"claims": 0, "supported": 0, "weak": 0, "unsupported": 0,
                "evidence_ratio": 0.0, "cited": False}
    claims = qc.get("claims") or []
    sup = len([c for c in claims if c.get("evidence_status") == "supported"])
    weak = len([c for c in claims if c.get("evidence_status") == "weak"])
    un = len([c for c in claims if c.get("evidence_status") == "unsupported"])
    total = len(claims)
    ratio = (sup + 0.5 * weak) / float(total) if total else 0.0
    cited = any(c.get("evidence_keys") for c in claims)
    return {"claims": total, "supported": sup, "weak": weak,
            "unsupported": un, "evidence_ratio": round(ratio, 3),
            "cited": cited}


# --------------------------------------------------------------- 启发式回退
def heuristic_qc(answer, level=None, hits=0):
    """正则启发式兜底（改造前的行为），供 JSON 模式失败时使用。

    返回值结构与 run_qc 一致，mode="heuristic_fallback"，让调用方无感。
    """
    import confidence as _conf
    answer = answer or ""
    if hits == 0:
        status = "refused"
    elif _conf._is_full_refusal(answer):
        status = "refused"
    elif _conf._refusal_ratio(answer) > 0:
        status = "partial"
    else:
        status = "answered"
    conf = {"low": 0.15, "medium": 0.55, "high": 0.85}.get(level or "medium", 0.55)
    return {
        "status": status,
        "confidence": conf,
        "claims": [],
        "missing": "",
        "reasons": ["结构化质检不可用，已回退到正则启发式判断"],
        "errors": [],
        "valid": False,
        "mode": "heuristic_fallback",
        "summary": summarize(None),
    }


REFUSAL_TEMPLATE = (
    "资料里没有能支撑这个回答的内容，我不给没有依据的答案。\n"
    "（质检判定：{status}｜{reason}）\n"
    "补充资料或换个问法再来问我。"
)


def apply_action(qc, answer):
    """门禁动作：block 模式下把无依据的答案拦下来（对齐 GEOFlow「不合格留草稿」）。"""
    action = os.environ.get("RAG_QC_ACTION", "warn").strip().lower()
    if action != "block":
        return answer, action
    if not qc or qc.get("status") != "unsupported":
        return answer, action
    reason = (qc.get("reasons") or ["答案中的断言在资料里找不到依据"])[0]
    return REFUSAL_TEMPLATE.format(status="unsupported", reason=reason), action


# --------------------------------------------------------------- 主入口
def run_qc(question, answer, sources, hits=0, level=None, verbose=False):
    """对一次问答做结构化质检。返回 qc dict（永不抛异常）。

    - RAG_QC_MODE=off  → 直接走启发式，不调模型
    - hits == 0        → 必然拒答，不必花钱问模型
    - RAG_TEST=1       → 离线模式，不调模型
    - 调模型/解析/校验任一失败 → 回退启发式，errors 里留证据
    """
    mode_env = os.environ.get("RAG_QC_MODE", "json").strip().lower()
    n = len(sources or [])

    if mode_env == "off":
        return heuristic_qc(answer, level=level, hits=hits)
    if hits == 0 or not sources:
        q = heuristic_qc(answer, level=level, hits=0)
        q["mode"] = "short_circuit"
        q["reasons"] = ["没有检索到任何相关资料，直接判拒答"]
        return q
    if os.environ.get("RAG_TEST") == "1":
        return heuristic_qc(answer, level=level, hits=hits)

    ctx = "\n".join(
        "[{0}]《{1}》{2}".format(s.get("n", i + 1), s.get("title", s.get("filename", "")),
                                s.get("snippet", ""))
        for i, s in enumerate(sources))

    try:
        import rag_core
        raw = rag_core.ask_llm_raw(
            SYSTEM_QC,
            USER_QC.format(question=question, n=n, ctx=ctx, answer=answer),
            temperature=0.0,
            max_tokens=int(os.environ.get("RAG_QC_MAX_TOKENS", 900)),
        )
    except Exception as e:
        q = heuristic_qc(answer, level=level, hits=hits)
        q["errors"] = ["质检调用失败：{0}".format(e)]
        return q

    obj, perr = extract_json(raw)
    if obj is None:
        q = heuristic_qc(answer, level=level, hits=hits)
        q["errors"] = ["质检输出不是合法 JSON（{0}）".format(perr)]
        q["raw_head"] = (raw or "")[:200]
        return q

    qc, errors = validate_qc(obj, n_sources=n)
    if qc is None:
        q = heuristic_qc(answer, level=level, hits=hits)
        q["errors"] = errors or ["质检 JSON 校验失败"]
        return q

    qc["mode"] = "json" if not errors else "json_repaired"

    # D19：引用内容级回验（张冠李戴检测）——拿引用块全文校验每条断言，
    # 只降不升。失败/关闭时原样返回，不伤主流程。
    try:
        import evidence_verify
        qc, _ = evidence_verify.reverify(qc, sources, verbose=verbose)
    except Exception:
        pass

    qc["summary"] = summarize(qc)
    if verbose:
        print("🧪 质检：status={0} conf={1} claims={2}（{3}）".format(
            qc.get("status"), qc.get("confidence"),
            qc["summary"]["claims"], qc["mode"]))
    return qc
