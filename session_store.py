"""
session_store.py —— 多轮对话会话存储（P1 补强：D10 多轮上下文）。

高端企业问答系统都支持「多轮追问、上下文连续」（Glean / M365 Copilot 标配），
否则问完「年假几天」再问「那加班餐补呢」就接不上。这里用文件级 JSON 存储会话，
比纯内存更稳（重启不丢），又比数据库轻（适合原型 / 作品集）。

设计要点：
  - 每个 session 存历史 [{q, a, sources}]，前端把 session_id 一路带下去即可连续对话。
  - 历史只用于「查询改写」和「上下文补全」，绝不偷偷把历史当检索资料——防幻觉纪律不变。
  - 提供 TTL 清理（默认 7 天）防止磁盘无限涨。
"""

import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SESSIONS_PATH = os.path.join(DATA_DIR, "sessions.json")
SESSION_TTL = 7 * 24 * 3600  # 7 天


def _load():
    if not os.path.exists(SESSIONS_PATH):
        return {}
    try:
        with open(SESSIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(all_s):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_s, f, ensure_ascii=False, indent=2)


def new_session():
    """新建会话，返回 session_id。"""
    import secrets
    sid = "sess-" + secrets.token_hex(8)
    all_s = _load()
    all_s[sid] = {"created": time.time(), "updated": time.time(), "history": []}
    _save(all_s)
    return sid


def append(sid, q, a, sources=None, namespace="default"):
    """把一问一答追加进会话历史。"""
    all_s = _load()
    s = all_s.get(sid)
    if s is None:
        s = {"created": time.time(), "updated": time.time(), "history": []}
        all_s[sid] = s
    s["history"].append({
        "q": q,
        "a": a,
        "namespace": namespace,
        "sources": sources or [],
    })
    # 只保留最近 20 轮，避免上下文无限膨胀
    if len(s["history"]) > 20:
        s["history"] = s["history"][-20:]
    s["updated"] = time.time()
    all_s[sid] = s
    _save(all_s)


def get_history(sid):
    """返回该会话的 [{q, a, sources}] 列表；不存在返回空表。"""
    s = _load().get(sid)
    return s["history"] if s else []


def clear(sid):
    all_s = _load()
    if sid in all_s:
        del all_s[sid]
        _save(all_s)
        return True
    return False


def list_sessions():
    all_s = _load()
    return [
        {"sid": k, "turns": len(v.get("history", [])),
         "updated": v.get("updated", 0)}
        for k, v in all_s.items()
    ]


def cleanup_expired():
    """清掉超过 TTL 的会话，返回清理数。"""
    all_s = _load()
    now = time.time()
    before = len(all_s)
    all_s = {k: v for k, v in all_s.items()
             if now - v.get("updated", 0) < SESSION_TTL}
    _save(all_s)
    return before - len(all_s)
