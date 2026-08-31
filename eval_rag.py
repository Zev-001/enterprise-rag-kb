"""
eval_rag.py —— 量化评测体系（P0-D6 补强）。

之前没有评测体系，等于蒙眼开车：改了 rerank / 换了模型，到底变好还是变坏全靠感觉。
这个脚本内置一份**中文问答金标准集**，覆盖两类问题：

  A. 应答对（in_scope）：资料里真有答案 → 期望「检索能召回相关块」（召回率）
  B. 应拒答（out_of_scope）：资料里没有 → 期望「检索 0 命中 / 低于阈值」，
     系统才会老实说「资料里没提到」（防幻觉闸门能真正关上）

运行方式：
  # 完全离线（只验检索层，不烧 token，默认）
  python eval_rag.py

  # 指定知识库
  python eval_rag.py --namespace default

  # 真实调大模型，连「拒答话术」一起验（需要 DEEPSEEK_API_KEY）
  python eval_rag.py --live

输出：
  - 控制台评分表（召回率 / 拒答识别率 / 总分）
  - eval_report.json（可进 CI、可对比每次改动前后的曲线）

设计取舍（给小白）：离线模式验的是「检索层」这一最客观、最该先固定的环节；
LLM 拒答话术由 --live 在真机上复验。两步分开，既不浪费 key，又能持续盯住回归。
"""

import argparse
import json
import os
import sys

# 让脚本能直接 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import ingest_documents, kb_stats, retrieve, grounded_ask, _kb_path, RELEVANCE_THRESHOLD

EVAL_NS = "eval_tmp"

# ----------------------------------------------------- 内置金标准语料（示例科技 员工手册）
SAMPLE_DOCS = [
    {
        "filename": "员工手册.txt",
        "title": "员工手册",
        "text": (
            "年假制度：正式员工每年享有 5 天带薪年假，入职满一年后可递增。\n"
            "报销制度：国内出差住宿报销上限为每晚 500 元，交通据实报销，单次差旅"
            "总报销上限 5000 元，超出需部门负责人审批。\n"
            "餐补：工作日加班至 20:00 后可领 20 元加班餐补，每月上限 30 次。\n"
            "试用期：新入职员工试用期为 3 个月，试用期内不享受年终奖。\n"
            "考勤：工作日 9:30 上班，18:30 下班，午休 1 小时。"
        ),
    },
    {
        "filename": "产品白皮书.txt",
        "title": "产品白皮书",
        "text": (
            "本产品为企业知识库问答系统，基于检索增强生成（RAG）构建。\n"
            "支持 PDF、Word、TXT、Markdown 四种文档格式入库。\n"
            "核心能力是「只依据资料回答、答案可溯源、资料外诚实拒答」。"
        ),
    },
]

# in_scope：资料里有明确答案，期望检索命中
IN_SCOPE = [
    "年假有几天？",
    "报销上限是多少？",
    "加班餐补多少钱？",
    "试用期几个月？",
    "支持哪些文档格式？",
]

# out_of_scope：资料里没有。分两类，评测逻辑不同（重要！）
#   "distant"  —— 完全无关（如写代码/体育），语义分近 0，离线阈值就能触发拒答 ✅
#   "adjacent" —— 公司相关但资料外（如上市/星座），bge 给中等分（~0.3），
#                单靠阈值抓不住，真正的拒答靠 LLM 指令 → 必须 --live 真机复验
OUT_OF_SCOPE = [
    ("公司明年会在哪上市？", "adjacent"),
    ("创始人的星座是什么？", "adjacent"),
    ("如何用 Python 写快速排序？", "distant"),
    ("今天 NBA 总决赛谁赢了？", "distant"),
]


def _prepare():
    """把示例语料灌进临时知识库，返回是否成功。"""
    if kb_stats(EVAL_NS)["rows"] == 0:
        added = ingest_documents(SAMPLE_DOCS, namespace=EVAL_NS)
        print("[eval] 已灌入示例语料，共 {0} 个知识块（namespace={1}）".format(added, EVAL_NS))
    else:
        print("[eval] 复用已存在的示例语料（namespace={0}）".format(EVAL_NS))


def _cleanup():
    p = _kb_path(EVAL_NS)
    if os.path.exists(p):
        os.remove(p)
        print("[eval] 已清理临时知识库 {0}".format(EVAL_NS))


def run(verbose=True, live=False):
    _prepare()
    report = {"in_scope": [], "out_of_scope": [], "metrics": {}}

    # 离线模式（默认）：设 RAG_TEST=1，只验「检索 + 阈值守门」这一最客观环节，
    # 不烧 token。--live 时才真调大模型复验拒答话术。
    if not live:
        os.environ["RAG_TEST"] = "1"

    # ---- A. 检索召回率（in_scope：经阈值过滤后应有来源块）----
    in_pass = 0
    in_scores = []
    for q in IN_SCOPE:
        out = grounded_ask(q, namespace=EVAL_NS, verbose=False)
        hits = len(out["sources"])
        passed = hits >= 1
        in_pass += int(passed)
        # 取最高分，后面算「资料内/资料外可分性」
        best = max([s["score"] for s in out["sources"]], default=0.0)
        in_scores.append(best)
        if verbose:
            print("  [{0}] 命中 {1} 块（最高相关度 {2}）  {3}".format(
                "✅" if passed else "❌", hits, best, q))
        report["in_scope"].append({"q": q, "hits": hits, "pass": passed})

    # ---- B. 应拒答识别（out_of_scope）----
    # 离线下 bge 绝对分在小库里会虚高，阈值法对 OOS 不可靠；这里统一记录最高分，
    # 并把「资料外拒答」的真机验证留给 --live（看 LLM 是否说了「资料里没提到」）。
    oos_adjacent = 0
    oos_live_pass = 0
    oos_total = len(OUT_OF_SCOPE)
    for q, kind in OUT_OF_SCOPE:
        raw = retrieve(q, top_k=4, namespace=EVAL_NS)
        best = max([s for s, _ in raw], default=0.0)
        if kind == "adjacent":
            oos_adjacent += 1
        if verbose:
            tag = "⚠️ adjacent" if kind == "adjacent" else "ℹ️ distant"
            print("  [{0}] 最高相关度 {1}（拒答以 --live 为准）  {2}".format(tag, best, q))
        entry = {"q": q, "kind": kind, "best_score": best, "pass": None}
        if live:
            real = grounded_ask(q, namespace=EVAL_NS, verbose=False)
            refused = "资料里没提到" in real["answer"]
            entry["llm_refused"] = refused
            oos_live_pass += int(refused)
            if verbose:
                print("        → LLM {0}：「{1}」".format(
                    "已拒答" if refused else "⚠️未拒答", real["answer"][:36]))
        report["out_of_scope"].append(entry)

    recall = in_pass / len(IN_SCOPE)
    # 可分性：资料内最低分 - 资料外(adjacent)最高分；>0 说明阈值方向上能分开
    sep = (min(in_scores) - max([e["best_score"] for e in report["out_of_scope"]
                                if e["kind"] == "adjacent"], default=0)) if in_scores else 0

    if live:
        oos = oos_live_pass / oos_total if oos_total else 0
        score = round((recall * 0.5 + oos * 0.5) * 100, 1)
    else:
        # 离线：只确定性地给「检索召回率」打分；OOS 拒答单独标注（见 --live）
        score = round(recall * 100, 1)

    report["metrics"] = {
        "retrieval_recall": recall,
        "separability": round(sep, 3),
        "score": score,
        "live": live,
    }

    if verbose:
        print("-" * 56)
        print("📊 检索召回率（应答对）      ：{0}/{1} = {2:.0%}".format(
            in_pass, len(IN_SCOPE), recall))
        print("📊 资料内/外可分性 gap        ：{0:.3f}（>0 表示阈值方向上能分开）".format(sep))
        if oos_adjacent:
            print("📊 资料外拒答               ：全部 OOS 需 `python eval_rag.py --live` 真机复验"
                  "（小库下 bge 绝对分虚高，阈值法不可靠，拒答靠 LLM 指令）")
        print("🏁 评测总分（离线=检索层）    ：{0} 分".format(score))
        if recall >= 0.999:
            print("✅ 检索层达标：可放心迭代 rerank / 换模型。资料外拒答见 --live。")
        else:
            print("⚠️ 检索召回未达标：先查为什么相关块没被召回。")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _main():
    global EVAL_NS
    ap = argparse.ArgumentParser(description="企业 RAG 量化评测")
    ap.add_argument("--namespace", default=EVAL_NS, help="要评测的知识库（默认临时库）")
    ap.add_argument("--live", action="store_true",
                    help="真实调用大模型，连拒答话术一起验（需要 DEEPSEEK_API_KEY）")
    ap.add_argument("--keep", action="store_true", help="评测后保留临时知识库")
    args = ap.parse_args()

    EVAL_NS = args.namespace
    try:
        run(verbose=True, live=args.live)
    finally:
        if not args.keep and args.namespace == "eval_tmp":
            _cleanup()


if __name__ == "__main__":
    _main()
