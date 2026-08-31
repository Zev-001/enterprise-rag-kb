# -*- coding: utf-8 -*-
"""D21 内容指纹缓存失效 —— 自测（离线零依赖，不调模型）。

跑法：python test_cache_fingerprint.py
覆盖：读写命中 / 知识库一变旧答案当场失效 / 无指纹旧条目宁失效不答旧 /
      rag_core.kb_fingerprint 稳定性与变化 / prepare_ask 端到端透出指纹。
"""
import os
import sys
import json
import tempfile
import traceback

import cache as _cache

# 隔离：把缓存文件指到临时目录，不污染真实 data/
_TMP = tempfile.mkdtemp(prefix="fp_")
_cache.CACHE_PATH = os.path.join(_TMP, "answer_cache.json")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  ok  " + name)
    else:
        FAIL += 1
        print("  FAIL " + name + ("  | " + str(detail) if detail else ""))


# ------------------------------------------------- 1. 基础读写（带指纹）
print("[1] 基础读写：同指纹命中")
_cache.cache_invalidate()
_cache.cache_set("ns1", "年假有几天", {"answer": "5天"}, kb_fingerprint="FP_A")
r = _cache.cache_get("ns1", "年假有几天", kb_fingerprint="FP_A")
check("同指纹命中", r is not None and r["answer"] == "5天", r)
r = _cache.cache_get("ns1", "  年假，有几天! ", kb_fingerprint="FP_A")
check("问题规范化仍命中", r is not None)
check("stats 记录指纹", _cache.cache_stats()["kb_fingerprints"] == ["FP_A"])

# ------------------------------------------------- 2. 指纹变化 → 自动失效
print("[2] 核心场景：知识库一变，旧答案当场作废")
_cache.cache_set("ns2", "报销标准多少", {"answer": "每晚300元"},
                 kb_fingerprint="OLD_FP")
r = _cache.cache_get("ns2", "报销标准多少", kb_fingerprint="OLD_FP")
check("旧指纹命中", r is not None and r["answer"] == "每晚300元")
# 知识库变了（制度改版）→ 新指纹
r = _cache.cache_get("ns2", "报销标准多少", kb_fingerprint="NEW_FP")
check("新指纹未命中（旧答案不外泄）", r is None)
# 且旧条目已被主动销毁，不是躺着等 TTL
raw = json.load(open(_cache.CACHE_PATH, encoding="utf-8"))
check("失效条目被物理删除", len(raw) == 1, list(raw.keys()))  # 只剩 ns1 那条

# ------------------------------------------------- 3. 老条目（无指纹）宁失效不答旧
print("[3] 兼容与安全：没存指纹的老条目，传入指纹的调用视为失效")
_cache.cache_set("ns3", " OLD ", {"answer": "老格式答案"})  # 未传指纹
r = _cache.cache_get("ns3", "old")  # 旧式调用（双方都无指纹）不受影响
check("旧式调用（双方都无指纹）仍命中", r is not None)
r = _cache.cache_get("ns3", "old", kb_fingerprint="ANY")
check("老条目 + 新调用方式 → 失效（物理删除）", r is None)
r2 = _cache.cache_get("ns3", "old")
check("失效后旧式调用也不再命中", r2 is None)

# ------------------------------------------------- 4. TTL 仍正常
print("[4] TTL 过期逻辑不受影响")
_cache.cache_set("ns4", "q", {"answer": "x"}, kb_fingerprint="F", ttl=-1)
r = _cache.cache_get("ns4", "q", kb_fingerprint="F")
check("TTL 过期未命中", r is None)

# ------------------------------------------------- 5. rag_core.kb_fingerprint
print("[5] rag_core.kb_fingerprint：稳定 + 敏感 + mtime 缓存")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import rag_core as rc

tmp2 = tempfile.mkdtemp(prefix="fpcb_")
kb = os.path.join(tmp2, "kb_fpns.jsonl")
check("空库指纹=empty", rc.kb_fingerprint("fpns", kb_path=kb) == "empty")
rc.ingest_documents(
    [{"filename": "报销.txt", "text": "差旅住宿报销每晚400元"}],
    namespace="fpns", kb_path=kb)
fp1 = rc.kb_fingerprint("fpns", kb_path=kb)
fp1b = rc.kb_fingerprint("fpns", kb_path=kb)
check("同库指纹稳定（mtime 缓存复用）", fp1 == fp1b and fp1 not in ("", "empty"))
# 内容变了 → 指纹变（覆盖写一份新内容）
with open(kb, "w", encoding="utf-8") as f:
    f.write(json.dumps({"filename": "报销.txt", "chunk_id": 0,
                        "text": "差旅住宿报销每晚500元"}, ensure_ascii=False) + "\n")
# 强制绕过 mtime 缓存（内容变了但 mtime 粒度可能不变）
rc._FP_CACHE.pop(kb, None)
fp2 = rc.kb_fingerprint("fpns", kb_path=kb)
check("内容变了指纹变", fp2 != fp1, (fp1, fp2))

# ------------------------------------------------- 6. 端到端：prepare_ask 透出指纹
print("[6] 端到端：prepare_ask 带指纹，缓存随库失效")
rc.ingest_documents(
    [{"filename": "现行报销.txt", "text": "差旅住宿报销标准为每晚400元",
      "review_status": "approved"}],
    namespace="fpns", kb_path=kb)
# prepare_ask 按 namespace 走默认 data 目录 → 把 KB_DIR 指到临时目录
rc.KB_DIR = tmp2
prepared = rc.prepare_ask("差旅住宿报销标准每晚多少钱",
                          namespace="fpns", use_cache=False)
check("prepare_ask 透出 kb_fp",
      prepared.get("kb_fp") == rc.kb_fingerprint("fpns", kb_path=kb))

# 用 cache 模块直接模拟「入库前缓存 → 入库后失效」闭环
_cache.cache_set("fpns", "差旅住宿报销标准每晚多少钱",
                 {"answer": "旧版答案 300 元"},
                 mode="default", kb_fingerprint="FAKE_OLD")
hit = _cache.cache_get("fpns", "差旅住宿报销标准每晚多少钱", mode="default",
                       kb_fingerprint=prepared["kb_fp"])
check("真实新指纹下，旧缓存条目失效", hit is None)

print()
print("通过 {0} / 失败 {1}".format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
