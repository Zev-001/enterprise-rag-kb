"""
cache.py —— D15 答案缓存（P2 / 阶段 C）。

相同问题重复检索 + 重复调大模型是纯浪费：员工问「年假怎么算」可能一天问十次。
按「namespace + 规范化问题」缓存答案，TTL 过期自动失效；命中时直接复用，不重算。
零依赖（标准库）。缓存文件落在 data/answer_cache.json（.gitignore 已忽略 data/）。
"""
import os
import json
import hashlib
import re
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "data", "answer_cache.json")
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
_lock = threading.Lock()

DEFAULT_TTL = 3600  # 1 小时

_NORM_RE = re.compile(r"\s+|[^\w\u4e00-\u9fff]")


def _normalize(question):
    return _NORM_RE.sub("", (question or "")).lower()


def _key(namespace, question, mode=None):
    raw = "{0}|{1}|{2}".format(namespace, mode or "", _normalize(question))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def cache_get(namespace, question, mode=None, ttl=DEFAULT_TTL):
    """返回缓存的答案 dict，未命中或过期返回 None。mode 用于区分检索模式（default/graph），
    避免不同模式的答案互相污染缓存。"""
    k = _key(namespace, question, mode)
    with _lock:
        cache = _load()
        item = cache.get(k)
        if not item:
            return None
        if (time.time() - item.get("ts", 0)) > item.get("ttl", ttl):
            cache.pop(k, None)
            _save(cache)
            return None
        return item.get("value")


def cache_set(namespace, question, value, mode=None, ttl=DEFAULT_TTL):
    """缓存一份答案。value 是可 JSON 序列化的 dict（answer/sources/...）。
    mode 与 cache_get 保持一致，确保读写落在同一 key。"""
    k = _key(namespace, question, mode)
    with _lock:
        cache = _load()
        cache[k] = {"ts": time.time(), "ttl": ttl, "namespace": namespace,
                    "question": question, "value": value}
        _save(cache)
    return True


def cache_invalidate(namespace=None):
    """清缓存：指定 namespace 只清该库，None 清全部（入库后建议调一次）。"""
    with _lock:
        cache = _load()
        if namespace is None:
            cache.clear()
        else:
            for k in list(cache.keys()):
                if cache[k].get("namespace") == namespace:
                    del cache[k]
        _save(cache)
    return True


def cache_stats():
    """缓存使用情况：总数 / 有效数 / 涉及的知识库。"""
    with _lock:
        cache = _load()
        now = time.time()
        active = [v for v in cache.values()
                  if (now - v.get("ts", 0)) <= v.get("ttl", DEFAULT_TTL)]
        return {"total": len(cache), "active": len(active),
                "namespaces": sorted({v.get("namespace") for v in cache.values()})}