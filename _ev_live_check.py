"""
_ev_live_check.py —— D19 引用回验在线实测（真调 LLM + 真检索）。

场景：
  A. 正常问答 → 质检 + 回验全链路，看 verification 明细
  B. 人为张冠李戴：把 claims 的引用编号换到内容无关的块上，
     重跑 reverify，验证「只降不升」能把 LLM 的 supported 修下来
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import rag_core as rc
import evidence_verify as ev

print("=== [A] 正常问答：年假 ===")
out = rc.grounded_ask("公司年假有几天？", namespace="demo", verbose=False)
print("答案   :", (out.get("answer") or "")[:150])
qc = out.get("qc") or {}
print("质检   : status={0} mode={1} conf={2}".format(
    qc.get("status"), qc.get("mode"), qc.get("confidence")))
for c in (qc.get("claims") or []):
    print("  claim:「{0}」 keys={1} → {2}".format(
        c["text"][:40], c["evidence_keys"], c["evidence_status"]))
ver = qc.get("verification") or {}
print("回验   : 降级 {0} 条".format(ver.get("downgraded", 0)))
for r in (ver.get("rows") or []):
    print("  row : score={0} method={1} verdict={2} (llm_said={3})".format(
        r.get("score"), r.get("method"), r.get("verdict"), r.get("llm_said")))

print()
print("=== [B] 人为张冠李戴：把引用换到无关块 ===")
if not (qc.get("claims") or []):
    print("（A 没产出 claims，跳过）")
    sys.exit(0)
sources = rc.prepare_ask("公司年假有几天？", namespace="demo")["sources"]
print("召回块 :", [(s["n"], s["title"], s["snippet"][:30]) for s in sources])
import copy
qc_bad = copy.deepcopy(qc)
n = len(sources)
# 找到原文引用的块，把它换成「下一个块」（内容不同 → 张冠李戴）
swapped = False
for c in qc_bad["claims"]:
    if c["evidence_keys"] and n >= 2:
        old = c["evidence_keys"][0]
        new = old % n + 1  # 1→2, 2→1, ...：换到另一块
        if new != old:
            c["evidence_keys"] = [new]
            swapped = True
if not swapped:
    print("（只有一块或无法换引，跳过 B）")
    sys.exit(0)
qc_fixed, changed = ev.reverify(qc_bad, sources)
print("换引后回验 changed =", changed)
for c in qc_fixed["claims"]:
    print("  claim:「{0}」 keys={1} → {2}".format(
        c["text"][:40], c["evidence_keys"], c["evidence_status"]))
print("降级理由:", qc_fixed["reasons"][-1] if qc_fixed["reasons"] else "无")
print()
print(">>> 结论：引用错块被回验降级 ✅" if changed else ">>> ⚠️ 换引未被降级，需查阈值")
