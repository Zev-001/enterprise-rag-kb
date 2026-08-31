"""
test_evidence_verify.py —— D19 引用内容级回验的离线自测（无需 API Key）。

跑法：python test_evidence_verify.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RAG_TEST", "1")  # 离线，杜绝任何真实调用

import evidence_verify as ev

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  {0} {1}{2}".format("✅" if cond else "❌", name,
                                (" — " + detail) if detail and not cond else ""))


# ------------------------------------------------------------- ① 归一化与指纹
print("[1] normalize / content_hash")
check("全角→半角", ev.normalize("ＡＢＣ１２３") == "abc123")
check("去空白标点", ev.normalize("年假，有 5 天！") == ev.normalize("年假有5天"))
check("大小写归一", ev.normalize("Annual Leave") == ev.normalize("annual leave"))
check("指纹稳定", ev.content_hash("年假有5天") == ev.content_hash("年假 有 5 天。"))
check("指纹可区分", ev.content_hash("年假5天") != ev.content_hash("报销500元"))

# ------------------------------------------------------------- ② n-gram
print("[2] ngram_support")
chunk_nian = "员工入职满一年的，每年享有5天带薪年假，需提前3个工作日申请。"
chunk_bao = "出差住宿报销上限为每晚500元，需提供发票及审批单。"
check("相同内容=1", ev.ngram_support("每年享有5天带薪年假", chunk_nian) == 1.0)
check("浓缩转述>0.4", ev.ngram_support("年假有5天", chunk_nian) > 0.4,
      str(ev.ngram_support("年假有5天", chunk_nian)))
check("无关内容≈0", ev.ngram_support("年假有5天", chunk_bao) < 0.15,
      str(ev.ngram_support("年假有5天", chunk_bao)))

# ------------------------------------------------------------- ③ verify_claim
print("[3] verify_claim（强制 n-gram 路径，保证离线可测）")
_orig_sem = ev.semantic_sim
ev.semantic_sim = lambda a, b: None  # 屏蔽向量路径
r = ev.verify_claim("年假有5天", [chunk_bao, chunk_nian])
check("命中正确块", r["best_chunk"] == 2, str(r))
r2 = ev.verify_claim("需提供发票及审批单", [chunk_bao])
check("精确子串→exact 1.0", r2["method"] == "exact" and r2["score"] == 1.0, str(r2))
check("空候选不炸", ev.verify_claim("x", [])["method"] == "none")
ev.semantic_sim = _orig_sem

# ------------------------------------------------------------- ④ 阈值（双信号）
print("[4] _verdict 双信号阈值")
# 仅 n-gram 路径
check("ngram supported", ev._verdict(None, 0.4, False) == "supported")
check("ngram weak", ev._verdict(None, 0.25, False) == "weak")
check("ngram unsupported", ev._verdict(None, 0.05, False) == "unsupported")
# 语义路径（阈值按实测校准：相关对~0.73 / 张冠李戴对~0.50 / 无关<0.40）
check("语义：相关对 supported", ev._verdict(0.73, 0.5, True) == "supported")
check("语义：ngram 补刀 supported", ev._verdict(0.55, 0.4, True) == "supported")
check("语义：张冠李戴对只给 weak", ev._verdict(0.50, 0.08, True) == "weak",
      "关键校准点：bge 虚高的 0.50 不能算 supported")
check("语义：弱相关 weak", ev._verdict(0.45, 0.05, True) == "weak")
check("语义：无关 unsupported", ev._verdict(0.30, 0.05, True) == "unsupported")
check("语义：双低 unsupported", ev._verdict(0.40, 0.10, True) == "unsupported")

# ------------------------------------------------------------- ⑤ reverify 集成
print("[5] reverify 集成（张冠李戴检测）")
sources = [
    {"n": 1, "title": "年假", "snippet": chunk_nian[:200], "text": chunk_nian},
    {"n": 2, "title": "报销", "snippet": chunk_bao[:200], "text": chunk_bao},
]
# 场景 A：模型张冠李戴——年假断言引了 [2]（报销块）
qc_a = {"status": "answered", "confidence": 0.9,
        "claims": [{"text": "年假有5天", "evidence_keys": [2],
                    "evidence_status": "supported"}],
        "missing": "", "reasons": [], "errors": []}
ev.semantic_sim = lambda a, b: None
qc_a2, ch_a = ev.reverify(qc_a, sources)
check("张冠李戴被抓", ch_a and qc_a2["claims"][0]["evidence_status"] != "supported",
      str(qc_a2["claims"][0]))
check("降级理由已写", any("回验" in r for r in qc_a2["reasons"]), str(qc_a2["reasons"]))
check("mode 标记", qc_a2["mode"] == "json_reverified", str(qc_a2.get("mode")))
check("summary 重算", qc_a2["summary"]["supported"] == 0, str(qc_a2["summary"]))

# 场景 B：引用正确 → 不动
qc_b = {"status": "answered", "confidence": 0.9,
        "claims": [{"text": "年假有5天", "evidence_keys": [1],
                    "evidence_status": "supported"}],
        "missing": "", "reasons": [], "errors": []}
qc_b2, ch_b = ev.reverify(qc_b, sources)
check("正确引用不动", not ch_b and qc_b2["claims"][0]["evidence_status"] == "supported")

# 场景 C：只降不升——LLM 说 unsupported 但内容其实匹配，不往上修
qc_c = {"status": "partial", "confidence": 0.5,
        "claims": [{"text": "每年享有5天带薪年假", "evidence_keys": [1],
                    "evidence_status": "unsupported"}],
        "missing": "", "reasons": [], "errors": []}
qc_c2, ch_c = ev.reverify(qc_c, sources)
check("只降不升", not ch_c and qc_c2["claims"][0]["evidence_status"] == "unsupported",
      str(qc_c2["claims"][0]))

# 场景 D：无 text 只有 snippet（旧数据兼容）
sources_snip = [{"n": 1, "title": "年假", "snippet": chunk_nian}]
qc_d = {"status": "answered", "confidence": 0.9,
        "claims": [{"text": "年假有5天", "evidence_keys": [1],
                    "evidence_status": "supported"}],
        "missing": "", "reasons": [], "errors": []}
qc_d2, ch_d = ev.reverify(qc_d, sources_snip)
check("snippet 兜底可用", not ch_d and qc_d2["claims"][0]["evidence_status"] == "supported")

# 场景 E：开关 off → 原样（注意：qc_a 已被场景 A 原地降级，这里必须新建）
os.environ["RAG_EVIDENCE_VERIFY"] = "off"
qc_e = {"status": "answered", "confidence": 0.9,
        "claims": [{"text": "年假有5天", "evidence_keys": [2],
                    "evidence_status": "supported"}],
        "missing": "", "reasons": [], "errors": []}
qc_e2, ch_e = ev.reverify(qc_e, sources)
check("off 开关跳过", not ch_e and qc_e2["claims"][0]["evidence_status"] == "supported")
os.environ["RAG_EVIDENCE_VERIFY"] = "1"
ev.semantic_sim = _orig_sem

# 场景 F：异常不伤主流程
print("[6] 健壮性")
qc_f, ch_f = ev.reverify(None, sources)
check("qc=None 原样", qc_f is None and not ch_f)
qc_g, ch_g = ev.reverify({"claims": [{"text": "x", "evidence_keys": [99],
                                      "evidence_status": "supported"}]}, sources)
check("越界 key 不炸", ch_g and qc_g["claims"][0]["evidence_status"] == "unsupported",
      str(qc_g["claims"][0]))

# ------------------------------------------------------------- ⑥ 语义路径（真实模型，opt-in）
# bge 模型加载吃内存且慢，默认跳过（n-gram 兜底已覆盖主逻辑）。
# 要真实跑语义路径：PowerShell 里 $env:RAG_EVIDENCE_TEST_SEMANTIC="1" 后再跑本测试。
print("[7] semantic_sim（真实 bge 模型，默认跳过）")
if os.environ.get("RAG_EVIDENCE_TEST_SEMANTIC") == "1":
    try:
        s_good = ev.semantic_sim("年假有5天", chunk_nian)
        s_bad = ev.semantic_sim("年假有5天", chunk_bao)
        if s_good is None:
            print("  ⏭️ 模型不可用，跳过（n-gram 兜底已覆盖）")
        else:
            check("语义：相关块分更高", s_good > s_bad,
                  "good={0} bad={1}".format(s_good, s_bad))
            check("语义：相关块过 supported 线", s_good >= 0.62, str(s_good))
            check("语义：无关块过不了 supported 线", s_bad < 0.62, str(s_bad))
    except Exception as e:
        print("  ⏭️ 语义模型加载失败（{0}），跳过——n-gram 兜底已覆盖".format(e))
else:
    print("  ⏭️ 未设 RAG_EVIDENCE_TEST_SEMANTIC=1，跳过真实模型加载")

# ------------------------------------------------------------- 汇总
print()
print("=" * 46)
print("结果：{0} 过 / {1} 挂".format(len(PASS), len(FAIL)))
if FAIL:
    print("失败项：" + "; ".join(FAIL))
    sys.exit(1)
print("全部通过 ✅")
