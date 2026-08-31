"""
meta_filter.py —— D20 召回前元数据过滤（对标 GEOFlow 的 effective_date / review_status 治理）。

解决什么问题？
    企业知识库不是「进了库就能用」的：
      - 制度文件有生效日期：明年才生效的薪级表、上个月已作废的报销标准，
        检索得再准也不该拿来回答今天的提问；
      - 文档有审核状态：草稿、待审核的版本混在库里，一旦被检索命中，
        AI 会一本正经地引用一份「还没定稿」的制度——这在企业里是事故。
    GEOFlow 的做法是在召回前按元数据把这类块直接挡在门外（而不是召回后
    靠 prompt 叮嘱「注意时效」——LLM 不可靠，门禁必须发生在代码层）。

设计原则
    1. 代码层硬过滤：过期/未生效/未审核/已作废的块【连候选都不进】，
       而不是召回后降权——治理类规则不能有概率性放过。
    2. 缺元数据 = 放行：老库（没有任何 effective_date / review_status 的块）
       行为完全不变，向后兼容；只有明确写了「未生效/已过期/未审核/作废」
       的块才被拦。
    3. 审核状态 fail-closed：写了 review_status 但值不认识（拼错、自定义），
       保守当「未审核」拦下，并在 reason 里说明——治理场景宁可误拦。
    4. 可观测：每次过滤留下 {hidden, reasons} 统计，前端/日志能看到
       「拦了什么、为什么拦」，不搞黑箱。
    5. 总开关：RAG_META_FILTER=off 一键退回旧行为（对照评测用）。

元数据约定（块级 row 或 row["meta"] 里，键名中英文都认）
    effective_date / 生效日期     生效起始日，晚于今天 → 未生效
    effective_until / expires_at / 失效日期 / 过期日期   失效日，早于今天 → 已过期
    review_status / 审核状态      approved|published|已审核|已发布 → 放行
                                  draft|pending|草稿|待审核|审核中 → 拦
                                  deprecated|retired|作废|已作废|废止 → 拦
                                  其他未知值 → 保守拦（fail-closed）

依赖：仅标准库（datetime）。
"""

import datetime as _dt
import os

# ---------------------------------------------------------------- 配置
# 当天日期可用环境变量覆盖（测试/演示「时间旅行」用），默认真实今天
DATE_ENV = "RAG_TODAY"

_REVIEW_OK = {"approved", "approved", "published", "publish", "released",
              "已审核", "审核通过", "已发布", "已生效", "发布", "生效", "正式"}
_REVIEW_DRAFT = {"draft", "pending", "reviewing", "in_review", "wip",
                 "草稿", "待审核", "审核中", "未审核", "拟定", "预发布"}
_REVIEW_DEAD = {"deprecated", "retired", "obsolete", "revoked",
                "作废", "已作废", "废止", "已废止", "停用", "已停用", "失效"}

# 日期字段的全部别名：{标准名: [别名...]}
_DATE_KEYS = {
    "effective_date": ["effective_date", "生效日期", "生效日", "发布日期", "publish_date"],
    "effective_until": ["effective_until", "expires_at", "expire_date", "expiry_date",
                        "失效日期", "过期日期", "失效日", "有效期至", "有效期至："],
}


def enabled():
    """总开关：RAG_META_FILTER=off 时整个模块退化为不过滤。"""
    return os.environ.get("RAG_META_FILTER", "").lower() != "off"


def today():
    """「今天」从哪来：RAG_TODAY 环境变量可覆盖（演示/测试时间旅行），默认真实今天。"""
    raw = os.environ.get(DATE_ENV, "").strip()
    if raw:
        d = parse_date(raw)
        if d:
            return d
    return _dt.date.today()


def parse_date(value):
    """多格式日期解析，返回 datetime.date；认不出返回 None（不抛异常）。

    支持：date/datetime 对象、2026-08-31、2026/8/31、2026.8.31、
    2026年8月31日、20260831、2026-08-31 12:00（取日期部分）。
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    s = str(value).strip()
    if not s:
        return None
    # 中文日期 → 规范分隔符
    s = s.replace("年", "-").replace("月", "-").replace("日", "").replace("号", "")
    s = s.replace(".", "-").replace("/", "-")
    s = s.split(" ")[0].split("T")[0]  # 去掉时间部分
    # 20260831 紧凑格式
    if s.isdigit() and len(s) == 8:
        s = "{0}-{1}-{2}".format(s[:4], s[4:6], s[6:])
    try:
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 3:
            return _dt.date(*parts)
    except (ValueError, TypeError):
        return None
    return None


def get_meta(row):
    """合并块级元数据：row 顶层字段优先级低于 row["meta"]（meta 是显式声明的，后写覆盖）。"""
    merged = {}
    for k, v in (row or {}).items():
        if k not in ("meta",):
            merged[k] = v
    meta = (row or {}).get("meta") or {}
    if isinstance(meta, dict):
        merged.update(meta)
    return merged


def _pick(meta, std_key):
    """按标准名+别名取第一个非空值。"""
    for key in [std_key] + _DATE_KEYS[std_key]:
        v = meta.get(key)
        if v not in (None, ""):
            return v
    return None


def row_status(row, ref_date=None):
    """判定单个块的状态，返回 (state, reason)。

    state ∈ ok | not_effective | expired | draft | deprecated
    判定顺序：审核态（身份问题，一票否决）→ 未生效 → 已过期。
    """
    if not enabled():
        return "ok", ""
    if ref_date is None:
        ref_date = today()
    meta = get_meta(row)

    # 1) 审核状态：写了才判；没写 = 老数据，放行（向后兼容）
    rs = meta.get("review_status") or meta.get("审核状态") or meta.get("status")
    if rs not in (None, ""):
        val = str(rs).strip().lower()
        if val in _REVIEW_OK:
            pass
        elif val in _REVIEW_DRAFT:
            return "draft", "审核状态为「{0}」，未发布".format(rs)
        elif val in _REVIEW_DEAD:
            return "deprecated", "审核状态为「{0}」，已作废".format(rs)
        else:
            # fail-closed：认不出的状态保守拦截
            return "draft", "审核状态「{0}」不在已知白名单，保守拦截".format(rs)

    # 2) 未生效：生效日期晚于今天（当天生效算生效，含边界）
    ed = parse_date(_pick(meta, "effective_date"))
    if ed and ed > ref_date:
        return "not_effective", "生效日期 {0} 晚于今天 {1}，尚未生效".format(ed, ref_date)

    # 3) 已过期：失效日期早于今天（当天到期仍可用，含边界）
    ud = parse_date(_pick(meta, "effective_until"))
    if ud and ud < ref_date:
        return "expired", "失效日期 {0} 早于今天 {1}，已过期".format(ud, ref_date)

    return "ok", ""


def visible_rows(rows, ref_date=None):
    """过滤入口：返回 (可见块列表, 被拦列表)。

    被拦列表元素为 (row, state, reason)，供上层做可观测展示。
    """
    if not enabled():
        return list(rows or []), []
    visible, blocked = [], []
    for r in rows or []:
        state, reason = row_status(r, ref_date)
        if state == "ok":
            visible.append(r)
        else:
            blocked.append((r, state, reason))
    return visible, blocked


def filter_stats(rows, ref_date=None):
    """统计一个知识库的治理健康度（kb_stats / 前端展示用）。"""
    if not enabled():
        return {"total": len(rows or []), "hidden": 0}
    if ref_date is None:
        ref_date = today()
    counts = {"expired": 0, "not_effective": 0, "draft": 0, "deprecated": 0}
    visible, blocked = visible_rows(rows, ref_date)
    for _, state, _ in blocked:
        counts[state] = counts.get(state, 0) + 1
    return {"total": len(rows or []), "hidden": len(blocked), **counts}
