"""
rag_log.py —— LLMOps 日志与指标（P1 补强：D11 可观测性）。

高端 RAG 平台（Dify / Vertex AI / Glean）都有「对话日志 + 性能看板」，
否则你根本不知道：哪些问题答不上来、哪段知识没人问、平均延迟多少、换模型后有没有变差。
这里用 JSONL 单行日志 + 聚合指标，零依赖、可离线、可直接喂给看板。

字段（每行一条）：
  ts        时间戳
  namespace 知识库
  question  问题
  mode      检索/推理模式（default / graph / agent）
  backend   实际检索后端（semantic / tfidf）
  hits      命中块数
  latency   端到端耗时（秒）
  ans_len   答案字数
  src_count 引用来源数
  refused   是否触发拒答（资料里没提到）
"""

import json
import os
import time
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_PATH = os.path.join(DATA_DIR, "rag_log.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)


def log_event(**fields):
    """追加一条日志。所有字段原样写进 JSONL。"""
    fields.setdefault("ts", time.time())
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _iter(limit=None):
    if not os.path.exists(LOG_PATH):
        return
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    if limit:
        lines = lines[-limit:]
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def recent(limit=50):
    return list(_iter(limit=limit))


def get_metrics():
    """聚合指标：总数、平均/中位/分位延迟、拒答率、Top 问题、按库分布。"""
    rows = list(_iter())
    if not rows:
        return {"total": 0}
    total = len(rows)
    lats = sorted(r.get("latency", 0) for r in rows if "latency" in r)
    refused = sum(1 for r in rows if r.get("refused"))
    ns_counter = Counter(r.get("namespace", "default") for r in rows)
    q_counter = Counter(r.get("question", "")[:40] for r in rows)
    mode_counter = Counter(r.get("mode", "default") for r in rows)

    def pct(p):
        if not lats:
            return 0
        idx = min(len(lats) - 1, int(len(lats) * p))
        return round(lats[idx], 3)

    return {
        "total": total,
        "latency_avg": round(sum(lats) / len(lats), 3) if lats else 0,
        "latency_p50": pct(0.5),
        "latency_p95": pct(0.95),
        "refuse_rate": round(refused / total, 3),
        "by_namespace": dict(ns_counter),
        "by_mode": dict(mode_counter),
        "top_questions": q_counter.most_common(10),
    }
