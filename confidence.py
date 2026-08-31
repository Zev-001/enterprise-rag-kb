"""
confidence.py —— D16 答案置信度评分（P2 / 阶段 C）。

原来的 `grounded_ask` 不回答「这个答案有多可信」，前端无法提示风险（报告 D16）。
这里合成 0–1 分 + 等级 + 可读理由，三个因子全部可解释：

  retrieval  相关性 —— 命中块平均分相对「高相关」基准的比值（阈值*6）
  coverage   覆盖度 —— 命中数 / 期望数（期望 3 块起）
  groundedness 落地性 —— 是否拒答 / 是否显式引用来源 [n] / 来源数量

分数是启发式的、可解释的，不是统计置信度——它的作用是让前端能标红/标绿，
让「这个答案还是再确认一下」变成可程序化判断的事。
"""
import re

_REFUSE_RE = re.compile(r"资料里没提到|没有检索到|我不了解|无法确定|没有找到")
_CITE_RE = re.compile(r"\[\d+\]")


def _split_sentences(answer):
    return [p for p in re.split(r"(?<=[。！？\n])", answer or "") if p.strip()]


def _refusal_ratio(answer):
    """拒答句占全文的字符比例（0–1）。整段拒答 → 1.0；无拒答 → 0.0。

    按句切分后再算，避免「答对了 + 补一句资料外说明」被整段误判为拒答。
    """
    parts = _split_sentences(answer)
    if not parts:
        return 0.0
    total = float(sum(len(p) for p in parts)) or 1.0
    refuse_chars = float(sum(len(p) for p in parts if _REFUSE_RE.search(p)))
    return refuse_chars / total


def _is_full_refusal(answer):
    """是否「整体拒答」。

    只算字符占比不够稳——模型常在给出正确答案后补一句「资料里没提到其他
    情况」，那是有依据的答案，不该整段打成低可信。真实的拒答有个稳定特征：
    **第一句就拒答**（先说没有，再解释）。所以判据是：
      首句即拒答 → 整体拒答
      或拒答篇幅 ≥85% → 整体拒答（几乎全篇都是拒答话术）
    其余带拒答词的情况一律算「部分拒答」：答了一部分，另一部分在资料外。
    """
    parts = _split_sentences(answer)
    if not parts:
        return False
    if _REFUSE_RE.search(parts[0]):
        return True
    return _refusal_ratio(answer) >= 0.85


def score_confidence(pairs, answer, threshold=0.05):
    """pairs = [(score, row), ...]；answer = 模型回答文本。
    返回 dict：{confidence, level, factors, reason}
    """
    answer = answer or ""
    n = len(pairs)
    if n == 0:
        return {
            "confidence": 0.0, "level": "low",
            "factors": {"retrieval": 0.0, "coverage": 0.0, "groundedness": 0.0,
                        "max_score": 0.0, "mean_score": 0.0, "hits": 0},
            "reason": ["没有检索到任何相关块，答案不可信"],
        }

    scores = [s for s, _ in pairs]
    max_s = max(scores)
    mean_s = sum(scores) / float(n)

    # 相关性：平均分相对「高相关」基准（阈值*6）的比值，上限 1
    retrieval = min(1.0, mean_s / max(threshold * 6.0, 1e-6))
    # 覆盖：期望 3 块起
    coverage = min(1.0, n / 3.0)
    # 落地性
    # 注意：不能只凭「出现拒答词」就判拒答——模型常在给出正确答案后补一句
    # 「资料里没提到其他情况」，整句判拒答会把好答案误杀成低可信。
    # 这里按句算拒答占比：只有拒答占了主体（≥60% 篇幅）才算真拒答；
    # 部分拒答（答了一部分、另一部分资料外）单独折中处理。
    ratio = _refusal_ratio(answer)
    cited = bool(_CITE_RE.search(answer))
    if _is_full_refusal(answer):
        refused = True
        partial = False
        grounded = 0.1
    elif ratio > 0:
        refused = False
        partial = True
        grounded = 0.7 if cited else 0.5   # 答了一部分，另一部分资料外
    elif cited:
        refused = False
        partial = False
        grounded = 0.95
    elif n >= 2:
        refused = False
        partial = False
        grounded = 0.6   # 有多个来源但未显式引用
    else:
        refused = False
        partial = False
        grounded = 0.4   # 单一来源且未引用

    conf = 0.35 * retrieval + 0.25 * coverage + 0.40 * grounded
    if refused:
        conf = min(conf, 0.15)
    conf = max(0.0, min(1.0, conf))

    level = "high" if conf >= 0.7 else ("medium" if conf >= 0.4 else "low")

    reason = []
    if retrieval >= 0.8:
        reason.append("命中块相关性高")
    elif retrieval >= 0.5:
        reason.append("命中块相关性中等")
    else:
        reason.append("命中块相关性偏低")
    if coverage >= 0.9:
        reason.append("覆盖充分（{0} 块）".format(n))
    else:
        reason.append("覆盖一般（{0} 块）".format(n))
    if refused:
        reason.append("整体拒答，视为低可信")
    elif partial:
        reason.append("部分拒答（约 {0:.0%} 篇幅为拒答），其余有依据".format(ratio))
    elif cited:
        reason.append("答案显式引用了来源")
    else:
        reason.append("答案未显式引用来源")

    return {
        "confidence": round(conf, 3),
        "level": level,
        "factors": {
            "retrieval": round(retrieval, 3),
            "coverage": round(coverage, 3),
            "groundedness": round(grounded, 3),
            "max_score": round(max_s, 4),
            "mean_score": round(mean_s, 4),
            "hits": n,
        },
        "reason": reason,
    }