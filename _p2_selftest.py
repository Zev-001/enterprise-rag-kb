# -*- coding: utf-8 -*-
"""P2（阶段C）自测：D13 切块 / D14 压缩 / D15 缓存 / D16 置信度 / D17 跨模态。
纯离线（RAG_TEST=1），不调大模型。测完即可删除。
"""
import os
os.environ["RAG_TEST"] = "1"
os.environ["RAG_BACKEND"] = "tfidf"  # 演示库太小，语义模型不稳，用 tfidf 兜底

import json
import rag_core as rc
import chunker
import context as ctx_mod
import cache
import confidence as conf_mod
import multimodal as mm

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [FAIL] ") + name + (("  " + extra) if extra and not cond else ""))


# ---------------- D13 语义分块 ----------------
print("== D13 语义分块 ==")
manual = """# 员工手册
## 第一章 总则
第一条 本手册适用于公司全体正式员工。
第二条 员工应遵守公司规章制度，保守商业秘密。

## 第二章 考勤与休假
第十五条 年假每年最多 5 天，需提前申请。
第十六条 加班餐补每晚 30 元，凭发票报销。"""
for mode in ("semantic", "proposition", "template", "legacy"):
    cs = chunker.chunk_text(manual, mode=mode)
    check("chunk mode=%s 返回 %d 块" % (mode, len(cs)), len(cs) >= 1)
# semantic 能识别章节标题作为新块起点（两个章节分到不同块）
sem = chunker.chunk_text(manual, mode="semantic")
check("semantic 识别章节边界（第二章在另一块）",
      any("第二章" in c for c in sem) and any("第一章" in c for c in sem))
print("  semantic 前两块：")
for c in sem[:2]:
    print("   -", repr(c[:60]))

# ---------------- D14 上下文压缩/去重 ----------------
print("== D14 上下文压缩/去重 ==")
# kb_demo 里有 5 条近似重复块
rows = rc.load_kb("demo")
pairs = []
for r in rows:
    pairs.append((0.9, r))
ctx, info = ctx_mod.compress_context(pairs, cap_chars=200)
check("压缩把 5 块去重后 <= 5 块", info["chunks_in_ctx"] <= 5)
check("识别出重复块 merged>0", info["merged"] >= 1, str(info))
check("上下文非空", len(ctx) > 0)
print("   info:", info)

# ---------------- D15 缓存 ----------------
print("== D15 答案缓存 ==")
cache.cache_invalidate("demo")
r = cache.cache_get("demo", "年假有几天")
check("未命中返回 None", r is None)
cache.cache_set("demo", "年假有几天", {"answer": "5天", "sources": []})
r = cache.cache_get("demo", "年假有几天")
check("命中缓存", r is not None and r.get("answer") == "5天")
r2 = cache.cache_get("demo", "  年假有几天  ")  # 规范化：空白/标点不敏感
check("规范化后命中", r2 is not None, str(r2))
st = cache.cache_stats()
check("stats 正确", st["total"] >= 1 and st["active"] >= 1, str(st))
cache.cache_invalidate("demo")
check("清空后未命中", cache.cache_get("demo", "年假有几天") is None)

# ---------------- D16 置信度 ----------------
print("== D16 置信度评分 ==")
c1 = conf_mod.score_confidence([(0.9, {}), (0.8, {})], "根据资料[1]，年假有5天。")
check("高相关+引用 → high", c1["level"] == "high", str(c1))
c2 = conf_mod.score_confidence([(0.1, {})], "参考资料里没提到。")
check("拒答 → low", c2["level"] == "low", str(c2))
c3 = conf_mod.score_confidence([], "任何答案")
check("无命中 → confidence=0", c3["confidence"] == 0.0, str(c3))
c4 = conf_mod.score_confidence([(0.3, {})], "年假有5天。")
check("0-1 之间", 0.0 <= c4["confidence"] <= 1.0, str(c4))
# 回归用例：答对了 + 补一句「资料里没提到其他情况」，不能整段判成拒答
c5 = conf_mod.score_confidence([(0.9, {}), (0.8, {})],
                               "年假有5天。[1] 资料里没提到其他情况。")
check("答对+补拒答句 不算整体拒答", c5["level"] != "low", str(c5))
print("   c5:", c5["confidence"], c5["level"], c5["reason"])
c6 = conf_mod.score_confidence([(0.1, {})],
                               "资料里没提到年假天数。资料里也没提到报销标准。")
check("整段拒答仍判 low", c6["level"] == "low", str(c6))

# ---------------- D17 跨模态 ----------------
print("== D17 跨模态 ==")
check("语言识别 zh", mm.language_of("年假有几天") == "zh")
check("语言识别 en", mm.language_of("How many days") == "en")
tbl = mm.render_table(["项目", "金额"], [["餐补", "30元"], ["交通", "50元"]])
check("表格渲染成 markdown", "| 项目 |" in tbl and "餐补" in tbl, tbl[:80])
tc = mm.table_chunk(["项目", "金额"], [["餐补", "30元"]], "报销表", "baoxiao.csv", "bx", 0)
check("table chunk 有 modality", tc["modality"] == "table" and tc["meta"]["n_rows"] == 1)
ic = mm.image_chunk("这是一张工牌照片", "工牌", "card.png", "cp", 0, image_path="x.png")
check("image chunk 有 modality", ic["modality"] == "image" and ic["meta"]["has_b64"] is False)
filt = mm.filter_by_modalities([(0.9, tc), (0.8, ic)], ["table"])
check("模态过滤只留 table", len(filt) == 1 and filt[0][1]["modality"] == "table")

# ---------------- 端到端：grounded_ask 离线 ----------------
print("== 端到端 grounded_ask（离线） ==")
out = rc.grounded_ask("年假有几天", namespace="demo", verbose=False)
check("离线作答", "RAG_TEST" in out["answer"], out["answer"])
check("置信度字段存在", "confidence" in out and "confidence_level" in out)
check("compress 字段存在", "compress" in out and "chunks_in_ctx" in out["compress"])
check("from_cache 默认 False", out.get("from_cache") is False)
print("   confidence:", out["confidence"], out["confidence_level"], out["confidence_reason"])

# 缓存命中路径：离线占位答案不入库（缓存占位没意义），这里手工灌一条真实缓存再测
cache.cache_set("demo", "年假有几天", mode="default", value={
    "answer": "年假每年最多 5 天[1]。", "sources": [],
    "effective_question": "年假有几天", "retrieval": "default",
    "compress": {"chunks_in_ctx": 1, "merged": 0, "deduped": 0, "chars": 20},
    "confidence": 0.82, "confidence_level": "high",
    "confidence_reason": ["命中 3 块高相关资料"], "confidence_factors": {},
})
out2 = rc.grounded_ask("年假有几天", namespace="demo", verbose=False, use_cache=True)
check("灌缓存后命中（跳过检索+调模型）", out2.get("from_cache") is True, str(out2.get("from_cache")))
check("缓存命中也带置信度", out2.get("confidence") == 0.82, str(out2.get("confidence")))
out3 = rc.grounded_ask("年假有几天", namespace="demo", verbose=False, use_cache=False)
check("use_cache=False 绕过缓存", out3.get("from_cache") is False)
cache.cache_invalidate("demo")

# prepare_ask 不调模型
prep = rc.prepare_ask("报销上限多少", namespace="demo")
check("prepare_ask 返回 prompt", bool(prep.get("prompt")))
check("prepare_ask 有 scores", isinstance(prep.get("scores"), list))

# 流式接口存在且为生成器
gen = rc.ask_llm_stream("x", "y")
check("ask_llm_stream 是生成器", hasattr(gen, "__next__"))

print("\n===== P2 自测结果 =====")
print("通过: %d / 失败: %d" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项:", FAIL)
    raise SystemExit(1)
print("全部通过 ✅")