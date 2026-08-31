"""
agentic_rag.py —— Agentic RAG 多步推理（P1 补强：D8 自主式检索）。

高端系统（RAGFlow Agentic / M365 Copilot Agent / Vertex Agent Engine）让模型
「先规划、再检索、再反思」，而不是一锤子把问题丢给检索。这能答好需要拼多块信息
的复杂问题（如「对比 A 和 B 的报销政策差异」）。

这里做轻量 ReAct：
  Step1 拆解：让 LLM 把原问题拆成 2-3 个子问题（plan）。
  Step2 检索+作答：每个子问题走一次 grounded 检索，让 LLM 用资料简要答。
  Step3 综合：把子答案 + 原始来源喂回 LLM，产出最终答案 + 完整引用。

离线（RAG_TEST=1）：不做多步分解，直接退化为单次 grounded 作答并标注，
保证不烧 token、不报错。
"""

import os

import rag_core as rc

SYS_DECOMPOSE = (
    "你是一个问题分析助手。把用户的问题拆成 2-3 个可以分别用「资料检索」回答的子问题。"
    "每行一个，用「Q:」开头，不要解释。如果原问题已经很简单不需要拆解，就只输出一行 Q: 原问题。"
)
SYS_SUBANS = (
    "你依据下列【参考资料】简要回答这个问题。资料没提到的就写「资料里没提到」。"
    "只输出答案，不要发挥。"
)
SYS_SYNTH = (
    "你是企业知识库问答助手。下面是围绕一个问题的多个子答案和它们的引用来源。"
    "请综合成一段连贯的最终答复，引用时标注 [n]，资料里没提到的明确说没提到。"
)


def _call(system, user, temperature=0.3):
    return rc.ask_llm_raw(system, user, temperature=temperature)


def agentic_ask(question, namespace="default", top_k=rc.DEFAULT_TOP_K, verbose=True):
    """返回 dict：answer / sources / steps（推理轨迹）/ backend。"""
    offline = os.environ.get("RAG_TEST") == "1"

    if offline:
        out = rc.grounded_ask(question, top_k=top_k, namespace=namespace,
                              verbose=False)
        out["steps"] = [{"phase": "agentic(offline)",
                         "note": "RAG_TEST=1 未做多步分解，退化为单次检索"}]
        out["mode"] = "agent"
        return out

    # Step1：拆解
    sub_qs = []
    try:
        dec = _call(SYS_DECOMPOSE, "用户问题：" + question)
        for line in dec.splitlines():
            line = line.strip()
            if line.startswith("Q:") or line.startswith("q:"):
                sub_qs.append(line[2:].strip())
    except Exception:
        sub_qs = []
    if not sub_qs:
        sub_qs = [question]

    # Step2：逐子问题检索作答
    steps = []
    all_pairs = []  # 收集所有命中块用于最终溯源
    sub_answers = []
    for i, sq in enumerate(sub_qs[:3], 1):
        so = rc.grounded_ask(sq, top_k=top_k, namespace=namespace, verbose=False)
        sub_answers.append("子问题{0}：{1}\n答：{2}".format(i, sq, so["answer"]))
        for s in so["sources"]:
            all_pairs.append((s["score"], {"filename": s["filename"],
                                           "title": s["title"],
                                           "text": s.get("snippet", "")}))
        steps.append({"phase": "sub_q{0}".format(i), "question": sq,
                      "answer": so["answer"], "hits": so["hits"]})

    # Step3：综合
    ctx = "\n\n".join(sub_answers)
    try:
        answer = _call(SYS_SYNTH, "【子答案汇总】\n" + ctx)
    except Exception:
        answer = "（综合失败）" + sub_answers[0]

    # 溯源去重
    seen = set()
    sources = []
    all_pairs.sort(key=lambda x: -x[0])
    for s, r in all_pairs:
        key = r.get("text", "")[:80]
        if key in seen:
            continue
        seen.add(key)
        sources.append({"filename": r.get("filename", ""), "title": r.get("title", ""),
                        "score": round(s, 3), "snippet": r.get("text", "")[:200]})
        if len(sources) >= 6:
            break

    out = {"question": question, "answer": answer, "sources": sources,
           "backend": rc._BACKEND or "tfidf", "hits": len(sources),
           "steps": steps, "mode": "agent"}
    if verbose:
        print("=" * 70)
        print("🤖 [agentic] " + question)
        for st in steps:
            print("  · {0}: {1}（命中 {2}）".format(st["phase"], st["question"], st["hits"]))
        print("💬 " + answer)
    return out
