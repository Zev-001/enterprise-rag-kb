"""
demo_ingest.py —— 用 dailyrag 的 AI 新闻当「公司文档」演示素材，灌进企业知识库。

真实使用时不需要这个脚本：直接在网页上传你自己的 PDF/Word 即可。
这个脚本只是为了让 demo 一键可复现（不用手造文档）。

运行：
    python demo_ingest.py
"""

import json
import os

from rag_core import ingest_documents, kb_stats

# dailyrag 的演示素材（6 篇真实 AI 资讯，当「公司文档」用）
NEWS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dailyrag", "news_2026-08-27.jsonl"
)


def main():
    if not os.path.exists(NEWS_PATH):
        print("找不到演示素材：", NEWS_PATH)
        return
    docs = []
    with open(NEWS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append({
                "filename": row["title"] + ".md",
                "title": row["title"],
                "text": row["body"],
            })
    added = ingest_documents(docs, namespace="default")
    st = kb_stats("default")
    print("✅ 演示入库完成：{0} 篇文档 → {1} 个知识块".format(len(docs), added))
    print("   当前知识库：{0} 份文档，{1} 个块".format(len(st["docs"]), st["chunks"]))


if __name__ == "__main__":
    main()
