"""在线实测：真调 DeepSeek，验证 JSON Schema 质检在真实链路上的表现。跑完即删。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("RAG_TEST", None)

import rag_core as rc
import schema_qc as qc

print("=== 知识库状态 ===")
try:
    print(rc.kb_stats("demo"))
except Exception as e:
    print("demo:", e)
try:
    print(rc.kb_stats("default"))
except Exception as e:
    print("default:", e)

NS = "demo"
CASES = [
    ("有据问答（中文）", "公司年假有几天？"),
    ("拒答（中文，资料里没有）", "公司提供住房补贴吗？"),
    ("拒答（英文 —— 正则的死穴）", "How many vacation days does the company offer? answer in English"),
]

for tag, q in CASES:
    print("\n" + "=" * 70)
    print("【{0}】{1}".format(tag, q))
    try:
        out = rc.grounded_ask(q, namespace=NS, verbose=False)
    except Exception as e:
        print("  ❌ 失败：", e)
        continue
    print("  命中：{0} 块".format(out.get("hits")))
    print("  答案：{0}".format((out.get("answer") or "")[:220]))
    print("  置信度：{0}（{1}）".format(out.get("confidence"), out.get("confidence_level")))
    q = out.get("qc") or {}
    s = q.get("summary") or {}
    print("  质检：status={0} mode={1} 断言 {2} 条（{3}有据/{4}弱/{5}无据）".format(
        q.get("status"), q.get("mode"), s.get("claims", 0),
        s.get("supported", 0), s.get("weak", 0), s.get("unsupported", 0)))
    print("  理由：{0}".format("；".join(out.get("confidence_reason") or [])[:300]))
    if q.get("errors"):
        print("  校验问题：{0}".format(q["errors"][:2]))
    if q.get("missing"):
        print("  资料缺口：{0}".format(q["missing"]))

print("\n" + "=" * 70)
print("【引用越界捕获】构造：资料只有 2 段，答案却引用 [5]")
fake_sources = [
    {"n": 1, "title": "hr.txt", "filename": "hr.txt", "snippet": "年假5天"},
    {"n": 2, "title": "hr.txt", "filename": "hr.txt", "snippet": "报销需发票"},
]
r = qc.run_qc("公司年假和股票期权？", "年假有5天[1]，股票期权每人1000股[5]。",
              fake_sources, hits=2)
print("  status =", r.get("status"), "| mode =", r.get("mode"))
for c in r.get("claims", []):
    print("   - {0} → keys={1} {2}".format(
        c["text"][:30], c["evidence_keys"], c["evidence_status"]))
print("  reasons:", r.get("reasons"))
print("  errors :", r.get("errors"))
