"""
test_meta_filter.py —— D20 召回前元数据过滤的自测（离线，零依赖，不调模型）。

覆盖：
  1) parse_date 多格式解析
  2) row_status 四态判定 + 边界（当天生效/当天到期都算可用）+ fail-closed
  3) visible_rows 拆分与 reason 可观测
  4) filter_stats 治理统计
  5) RAG_META_FILTER=off 总开关退回旧行为
  6) ingest doc 级治理字段下沉 meta
  7) 端到端 retrieve：过期/草稿块拦在召回前，正常块照常命中
  8) GraphRAG 检索同样收口

运行：python test_meta_filter.py
"""

import datetime as _dt
import json
import os
import sys
import tempfile

# 必须在 import rag_core 之前：测试全程走 TF-IDF，不拉 bge 模型
os.environ.setdefault("RAG_BACKEND", "tfidf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 固定「今天」，测试可复现（不用真实日期，免得测试随时间腐烂）
TODAY = "2026-08-31"
os.environ.setdefault("RAG_TODAY", TODAY)

import meta_filter as mf
import rag_core as rc

_PASSED = 0
_FAILED = []


def check(name, cond, detail=""):
    global _PASSED
    if cond:
        _PASSED += 1
        print("  ✅ {0}".format(name))
    else:
        _FAILED.append(name)
        print("  ❌ {0} {1}".format(name, detail))


# --------------------------------------------------------------- 1. parse_date
def test_parse_date():
    print("[1] parse_date 多格式解析")
    d = _dt.date(2026, 8, 31)
    check("ISO", mf.parse_date("2026-08-31") == d)
    check("斜杠", mf.parse_date("2026/8/31") == d)
    check("点分", mf.parse_date("2026.8.31") == d)
    check("中文", mf.parse_date("2026年8月31日") == d)
    check("紧凑8位", mf.parse_date("20260831") == d)
    check("带时间取日期", mf.parse_date("2026-08-31 12:00:00") == d)
    check("datetime对象", mf.parse_date(_dt.datetime(2026, 8, 31, 8, 0)) == d)
    check("垃圾输入返回None", mf.parse_date("下个月生效") is None)
    check("空输入返回None", mf.parse_date("") is None and mf.parse_date(None) is None)


# --------------------------------------------------------------- 2. row_status
def test_row_status():
    print("[2] row_status 四态判定")
    today = _dt.date(2026, 8, 31)
    ok, _ = mf.row_status({"text": "x"})
    check("无元数据放行（向后兼容）", ok == "ok")

    st, why = mf.row_status({"text": "x", "effective_until": "2026-08-30"}, today)
    check("已过期（失效日=昨天）", st == "expired", why)
    st, why = mf.row_status({"text": "x", "expires_at": "2025-01-01"}, today)
    check("expires_at 别名", st == "expired", why)
    st, why = mf.row_status({"text": "x", "失效日期": "2026-07-01"}, today)
    check("中文键别名", st == "expired", why)

    st, why = mf.row_status({"text": "x", "effective_date": "2026-09-01"}, today)
    check("未生效（生效日=明天）", st == "not_effective", why)

    # 边界：当天生效 / 当天到期 都算可用（含边界）
    st, _ = mf.row_status({"text": "x", "effective_date": "2026-08-31"}, today)
    check("当天生效=可用", st == "ok")
    st, _ = mf.row_status({"text": "x", "effective_until": "2026-08-31"}, today)
    check("当天到期=可用", st == "ok")

    st, why = mf.row_status({"text": "x", "review_status": "draft"}, today)
    check("草稿拦截", st == "draft", why)
    st, _ = mf.row_status({"text": "x", "review_status": "待审核"}, today)
    check("中文待审核拦截", st == "draft")
    st, _ = mf.row_status({"text": "x", "review_status": "deprecated"}, today)
    check("作废拦截", st == "deprecated")
    st, _ = mf.row_status({"text": "x", "review_status": "approved"}, today)
    check("已审核放行", st == "ok")
    st, why = mf.row_status({"text": "x", "review_status": "wip_v2?"}, today)
    check("未知状态 fail-closed", st == "draft", why)

    # 一票否决顺序：审核态优先于时效
    st, _ = mf.row_status(
        {"text": "x", "review_status": "draft", "effective_until": "2020-01-01"}, today)
    check("审核态优先于时效", st == "draft")

    # meta 里的字段同样生效，且 meta 覆盖顶层
    st, _ = mf.row_status({"text": "x", "meta": {"effective_until": "2026-08-01"}}, today)
    check("meta 字段生效", st == "expired")
    st, _ = mf.row_status(
        {"text": "x", "effective_until": "2099-01-01", "meta": {"effective_until": "2026-08-01"}},
        today)
    check("meta 覆盖顶层", st == "expired")


# --------------------------------------------------------------- 3/4. 过滤与统计
def test_visible_and_stats():
    print("[3] visible_rows 拆分")
    rows = [
        {"chunk_id": "a", "text": "现行制度", "review_status": "approved"},
        {"chunk_id": "b", "text": "过期制度", "effective_until": "2026-01-01"},
        {"chunk_id": "c", "text": "草稿", "review_status": "draft"},
        {"chunk_id": "d", "text": "新制度", "effective_date": "2026-08-31"},
    ]
    visible, blocked = mf.visible_rows(rows)
    check("可见块=2且保序", [r["chunk_id"] for r in visible] == ["a", "d"])
    check("拦截块=2", len(blocked) == 2)
    states = {r["chunk_id"]: st for r, st, _ in blocked}
    check("拦截原因可观测", states == {"b": "expired", "c": "draft"})

    print("[4] filter_stats 治理统计")
    st = mf.filter_stats(rows)
    check("total=4", st["total"] == 4)
    check("hidden=2", st["hidden"] == 2)
    check("expired=1 draft=1", st["expired"] == 1 and st["draft"] == 1)


# --------------------------------------------------------------- 5. 总开关
def test_off_switch():
    print("[5] RAG_META_FILTER=off 总开关")
    rows = [{"chunk_id": "b", "text": "过期制度", "effective_until": "2026-01-01"}]
    os.environ["RAG_META_FILTER"] = "off"
    try:
        visible, blocked = mf.visible_rows(rows)
        check("开关关=全放行", len(visible) == 1 and not blocked)
        st, _ = mf.row_status(rows[0])
        check("row_status 也放行", st == "ok")
    finally:
        os.environ.pop("RAG_META_FILTER", None)
    visible, blocked = mf.visible_rows(rows)
    check("开关恢复=继续拦", len(visible) == 0 and len(blocked) == 1)


# --------------------------------------------------------------- 6. ingest 下沉
def test_ingest_meta():
    print("[6] ingest doc 级治理字段下沉 meta")
    tmp = tempfile.mkdtemp(prefix="mf_")
    kb = os.path.join(tmp, "kb_ingest_test.jsonl")
    added = rc.ingest_documents([
        {"filename": "新薪级表.txt", "text": "2026年9月生效的新薪级表内容",
         "effective_date": "2026-09-01"},
        {"filename": "旧报销标准.txt", "text": "已经作废的旧报销标准",
         "review_status": "deprecated"},
    ], namespace="ingest_test", kb_path=kb)
    check("写入2块", added == 2)
    rows = rc.load_kb(kb_path=kb)
    m0 = rows[0].get("meta", {})
    m1 = rows[1].get("meta", {})
    check("effective_date 下沉", m0.get("effective_date") == "2026-09-01")
    check("review_status 下沉", m1.get("review_status") == "deprecated")
    # doc.meta 里显式写了的同名键优先，不被顶层覆盖
    kb2 = os.path.join(tmp, "kb_ingest_test2.jsonl")
    rc.ingest_documents([
        {"filename": "x.txt", "text": "内容", "effective_date": "2099-01-01",
         "meta": {"effective_date": "2026-01-01"}},
    ], namespace="ingest_test2", kb_path=kb2)
    check("doc.meta 显式键优先", rc.load_kb(kb_path=kb2)[0]["meta"]["effective_date"] == "2026-01-01")


# --------------------------------------------------------------- 7. 端到端 retrieve
def test_e2e_retrieve():
    print("[7] 端到端 retrieve：过期/草稿块拦在召回前")
    ns = "mf_e2e_test"
    kb = rc._kb_path(ns)  # prepare_ask 按 namespace 走 data 目录，测完清理
    try:
        rc.ingest_documents([
            {"filename": "现行报销.txt", "text": "差旅住宿报销标准为每晚400元",
             "review_status": "approved"},
            {"filename": "过期报销.txt", "text": "差旅住宿报销标准为每晚300元",
             "effective_until": "2026-08-30"},  # 昨天到期
            {"filename": "草稿报销.txt", "text": "差旅住宿报销标准拟调整为每晚500元",
             "review_status": "draft"},
        ], namespace=ns)

        pairs = rc.retrieve("差旅住宿报销标准每晚多少钱", namespace=ns)
        titles = {p[1]["filename"] for p in pairs}
        check("只命中现行制度", titles == {"现行报销.txt"}, str(titles))
        mf_info = rc.last_meta_filter()
        check("过滤统计 hidden=2", mf_info["hidden"] == 2, str(mf_info))
        check("拦截原因齐全",
              {x["state"] for x in mf_info["reasons"]} == {"expired", "draft"})

        # 全部被拦 → 检索为空，AI 会走「资料里没提到」防幻觉路径
        rc.ingest_documents([
            {"filename": "未来文件.txt", "text": "明年的预算规划",
             "effective_date": "2027-01-01"},
        ], namespace="mf_e2e_all")
        pairs2 = rc.retrieve("明年预算规划", namespace="mf_e2e_all")
        check("全拦=空手而归（交给防幻觉闸门）", pairs2 == [])

        # prepare_ask：meta_filtered 透出（不调 LLM，只看 prepare 层）
        prepared = rc.prepare_ask("差旅住宿报销标准每晚多少钱", namespace=ns,
                                  use_cache=False)
        check("prepare_ask 透出 meta_filtered",
              (prepared.get("meta_filtered") or {}).get("hidden") == 2)
    finally:
        for n in (ns, "mf_e2e_all"):
            p = rc._kb_path(n)
            if os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------- 8. GraphRAG
def test_graph_filter():
    print("[8] GraphRAG 同样收口")
    ns = "mf_graph_test"
    kb = rc._kb_path(ns)  # 图谱按 namespace 走 data 目录，测完清理
    try:
        rc.ingest_documents([
            {"filename": "现行年假.txt", "text": "员工年假制度为每年5个工作日",
             "review_status": "approved"},
            {"filename": "草稿年假.txt", "text": "员工年假制度拟调整为每年10个工作日",
             "review_status": "draft"},
        ], namespace=ns)
        import graph_rag as gr
        pairs = gr.retrieve_graph("年假制度几天", namespace=ns)
        titles = {p[1]["filename"] for p in pairs}
        check("图谱只返回已审核块", titles == {"现行年假.txt"}, str(titles))
    finally:
        if os.path.exists(kb):
            os.remove(kb)
        try:
            import graph_rag as gr
            gr.clear_cache(ns)
        except Exception:
            pass


def main():
    test_parse_date()
    test_row_status()
    test_visible_and_stats()
    test_off_switch()
    test_ingest_meta()
    test_e2e_retrieve()
    test_graph_filter()
    print("=" * 60)
    print("通过 {0} 项".format(_PASSED))
    if _FAILED:
        print("失败 {0} 项：{1}".format(len(_FAILED), _FAILED))
        sys.exit(1)
    print("ALL PASS ✅")


if __name__ == "__main__":
    main()
