"""
test_schema_qc.py —— JSON Schema 硬约束质检的离线自测（不需要 API Key）。

跑法：
    python test_schema_qc.py

覆盖：
  1. extract_json 六种「模型不乖」的输出形态
  2. validate_qc 三层校验（schema / 证据回验 / 一致性）
  3. summarize 证据比例
  4. score_confidence 与 qc 的融合 + 不传 qc 的回归
  5. heuristic_qc 回退分支
  6. apply_action 门禁动作
  7. run_qc 的三种短路（off / RAG_TEST / hits=0）
"""
import os
import sys

os.environ.setdefault("RAG_TEST", "1")   # 防止误触发真实模型调用

import confidence as conf_mod
import schema_qc as qc

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  ✅ " if cond else "  ❌ ") + name + ((" — " + str(extra)) if extra else ""))


GOOD = {
    "status": "answered",
    "confidence": 0.9,
    "claims": [
        {"text": "年假有5天", "evidence_keys": [1], "evidence_status": "supported"},
        {"text": "需提前申请", "evidence_keys": [1, 2], "evidence_status": "weak"},
    ],
    "missing": "",
    "reasons": ["资料第1段写明年假天数"],
}

print("\n[1] extract_json —— 模型不乖的六种输出")
o, e = qc.extract_json('{"status": "answered"}')
check("纯 JSON", o == {"status": "answered"}, e)

o, _ = qc.extract_json('```json\n{"status": "refused"}\n```')
check("markdown 代码块", o and o["status"] == "refused")

o, _ = qc.extract_json('好的，这是质检结果：\n{"status": "partial"}\n希望能帮到你')
check("前后带废话", o and o["status"] == "partial")

o, e = qc.extract_json('我无法输出 JSON')
check("完全没有 JSON → 返回 None", o is None, e)

o, _ = qc.extract_json('{"status": "answered", "claims": [],}')
check("尾随逗号", o is not None)

o, _ = qc.extract_json('{"status": “answered”}')
check("中文引号", o is not None and o.get("status") == "answered")

print("\n[2] validate_qc —— schema 层")
qcobj, errs = qc.validate_qc(GOOD, n_sources=2)
check("合法对象 → valid", qcobj["valid"] and not errs, errs)

_, errs = qc.validate_qc({"confidence": 0.5}, n_sources=2)
check("缺 status → 报错", any("status" in x for x in errs))

_, errs = qc.validate_qc({"status": "maybe", "confidence": 0.5}, n_sources=2)
check("status 非法枚举 → 报错", any("非法" in x for x in errs))

o, errs = qc.validate_qc({"status": "answered", "confidence": 3.7}, n_sources=2)
check("confidence 越界 → clamp 到 1.0", o["confidence"] == 1.0)

_, errs = qc.validate_qc({"status": "answered", "confidence": 0.5, "claims": "x"},
                         n_sources=2)
check("claims 不是数组 → 报错", any("claims" in x for x in errs))

print("\n[2b] validate_qc —— 证据回验层（抓引用幻觉）")
HALLUC = {
    "status": "answered", "confidence": 0.9,
    "claims": [{"text": "年假15天", "evidence_keys": [7],
                "evidence_status": "supported"}],
    "reasons": [],
}
o, errs = qc.validate_qc(HALLUC, n_sources=2)
check("引用编号越界 → 强制 unsupported",
      o["claims"][0]["evidence_status"] == "unsupported")
check("越界编号被剔除", o["claims"][0]["evidence_keys"] == [])
check("越界记入 reasons", any("越界" in r for r in o["reasons"]), o["reasons"])

print("\n[2c] validate_qc —— 一致性层")
o, errs = qc.validate_qc({
    "status": "answered", "confidence": 0.9,
    "claims": [{"text": "xxx", "evidence_keys": [], "evidence_status": "unsupported"}],
    "reasons": []}, n_sources=2)
check("answered 但无 supported → 降级 unsupported", o["status"] == "unsupported")

print("\n[3] summarize —— 证据比例")
s = qc.summarize(qc.validate_qc(GOOD, n_sources=2)[0])
check("claims 计数", s["claims"] == 2, s)
check("supported/weak 计数", s["supported"] == 1 and s["weak"] == 1, s)
check("证据比例 = (1 + 0.5*1)/2 = 0.75", s["evidence_ratio"] == 0.75, s)
check("cited=True", s["cited"] is True)

print("\n[4] score_confidence × qc 融合")
pairs = [(0.4, {}), (0.35, {}), (0.3, {})]

base = conf_mod.score_confidence(pairs, "年假有5天。[1]", threshold=0.05)
check("不传 qc → 与改造前一致（回归基线 high）", base["level"] == "high", base["confidence"])
check("不传 qc → qc_mode 为空", base.get("qc_mode") is None)

q_good = dict(qc.validate_qc(GOOD, n_sources=2)[0], mode="json",
              summary=qc.summarize(qc.validate_qc(GOOD, n_sources=2)[0]))
f_good = conf_mod.score_confidence(pairs, "年假有5天。[1]", threshold=0.05, qc=q_good)
check("全有据 → 仍是 high", f_good["level"] == "high", f_good["confidence"])
check("qc_mode 标记为 json", f_good.get("qc_mode") == "json")
check("理由改用质检口径", any("质检判定" in r for r in f_good["reason"]), f_good["reason"])

UNSUP = {"status": "unsupported", "confidence": 0.2,
         "claims": [{"text": "年假15天", "evidence_keys": [],
                     "evidence_status": "unsupported"}],
         "missing": "缺年假天数", "reasons": ["资料里找不到天数"],
         "errors": [], "valid": True, "mode": "json",
         "summary": {"claims": 1, "supported": 0, "weak": 0, "unsupported": 1,
                     "evidence_ratio": 0.0, "cited": False}}
f_un = conf_mod.score_confidence(pairs, "年假有15天。", threshold=0.05, qc=UNSUP)
check("无依据 → 天花板 0.30 生效", f_un["confidence"] <= 0.30, f_un["confidence"])
check("无依据 → 等级 low", f_un["level"] == "low")

REF = dict(UNSUP, status="refused", confidence=0.05)
f_ref = conf_mod.score_confidence(pairs, "资料里没提到。", threshold=0.05, qc=REF)
check("拒答 → 天花板 0.15 生效", f_ref["confidence"] <= 0.15, f_ref["confidence"])
check("拒答理由含 整体拒答", any("拒答" in r for r in f_ref["reason"]))

PAR = dict(q_good, status="partial", confidence=0.6)
f_par = conf_mod.score_confidence(pairs, "年假有5天，其他没提。", threshold=0.05, qc=PAR)
check("部分有据 → 天花板 0.60", f_par["confidence"] <= 0.60, f_par["confidence"])

# 回退模式不该接管评分（否则就是拿猜测覆盖真信号）
fallback = dict(UNSUP, mode="heuristic_fallback")
f_fb = conf_mod.score_confidence(pairs, "年假有5天。[1]", threshold=0.05, qc=fallback)
check("回退模式 → 不接管，等同不传 qc",
      abs(f_fb["confidence"] - base["confidence"]) < 1e-6,
      (f_fb["confidence"], base["confidence"]))

print("\n[5] heuristic_qc —— 回退分支")
h = qc.heuristic_qc("随便什么答案", level="high", hits=0)
check("hits=0 → refused", h["status"] == "refused")
h = qc.heuristic_qc("资料里没提到这个。", level="low", hits=3)
check("整段拒答 → refused", h["status"] == "refused")
h = qc.heuristic_qc("年假有5天。[1] 资料里没提到其他情况。", level="high", hits=3)
check("答对+补一句 → partial", h["status"] == "partial", h["status"])
h = qc.heuristic_qc("年假有5天。[1]", level="high", hits=3)
check("正常答案 → answered", h["status"] == "answered")
check("回退标 mode=heuristic_fallback", h["mode"] == "heuristic_fallback")

print("\n[6] apply_action —— 门禁")
ans = "年假有15天。"
os.environ["RAG_QC_ACTION"] = "warn"
out, act = qc.apply_action(UNSUP, ans)
check("warn 模式不改答案", out == ans and act == "warn")
os.environ["RAG_QC_ACTION"] = "block"
out, act = qc.apply_action(UNSUP, ans)
check("block + unsupported → 拦下", "没有依据" in out and act == "block")
out, act = qc.apply_action(q_good, ans)
check("block + answered → 放行", out == ans)
os.environ["RAG_QC_ACTION"] = "warn"

print("\n[7] run_qc —— 短路分支（不调模型）")
os.environ["RAG_QC_MODE"] = "off"
r = qc.run_qc("q", "a", [{"n": 1, "snippet": "x"}], hits=1)
check("RAG_QC_MODE=off → 不调模型直接回退", r["mode"] == "heuristic_fallback")
os.environ["RAG_QC_MODE"] = "json"

r = qc.run_qc("q", "a", [], hits=0)
check("hits=0 → 短路拒答（省一次调用）",
      r["status"] == "refused" and r["mode"] == "short_circuit")

os.environ["RAG_TEST"] = "1"
r = qc.run_qc("q", "年假有5天", [{"n": 1, "snippet": "x"}], hits=1)
check("RAG_TEST=1 离线 → 不调模型", r["mode"] == "heuristic_fallback")

print("\n" + "=" * 60)
print("通过 {0} / 失败 {1}".format(len(PASS), len(FAIL)))
if FAIL:
    print("失败项：")
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
print("🎉 schema_qc 离线自测全过")
