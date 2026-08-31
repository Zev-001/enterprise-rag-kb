"""
ingest_docs.py —— 把上传的公司文档解析成纯文本，再喂给 rag_core 切块入库。

支持格式：
    .pdf    pdfplumber（能读文字层的 PDF；扫描件图片 PDF 读不出字，会提示）
    .docx   python-docx（Word 2007+；老 .doc 不支持）
    .txt    UTF-8 纯文本
    .md     Markdown，先去标记符号再存（保留可读文字）

设计取舍（给小白看的）：
    1. 只存「原文文字」，不存摘要。摘要已经过模型加工，再把摘要当资料检索是二级转述。
       存原文，溯源能追到具体哪份文件哪一段。
    2. 每块带 doc_id / filename / title 元数据，回答时能说「这句话来自《员工手册》第 3 段」。
    3. 大文件截断到 5 万字上限，避免一次塞太多把上下文撑爆（企业真实文档常很长）。
"""

import os

from rag_core import ingest_documents, kb_stats

MAX_CHARS = 50_000  # 单份文档最多摄入字数，超出截断


def _read_pdf(path):
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
    return "\n\n".join(parts)


def _read_docx(path):
    from docx import Document
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    # 表格也抓出来，制度/手册常放表格里
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_md(path):
    import re
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # 去掉 Markdown 标记，保留可读文字
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)          # 标题 #
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)                # 粗体
    text = re.sub(r"\*(.*?)\*", r"\1", text)                    # 斜体
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.S)  # 行内/代码块
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)                 # 图片
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)          # 链接保留文字
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.M)      # 列表符
    return text


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_txt,
    ".md": _read_md,
}


def parse_file(path):
    """解析单个文件，返回 (filename, text) 或抛错。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _READERS:
        raise ValueError("不支持的格式：{0}（支持 pdf/docx/txt/md）".format(ext))
    text = _READERS[ext](path)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return os.path.basename(path), text.strip()


def ingest_file(path, namespace="default", title=None):
    """解析并入库单个文件，返回 {filename, chars, chunks_added}。"""
    filename, text = parse_file(path)
    if not text:
        raise ValueError("从 {0} 没提取到任何文字（可能是扫描件 PDF，需要 OCR）".format(filename))
    docs = [{"filename": filename, "title": title or filename, "text": text}]
    added = ingest_documents(docs, namespace=namespace)
    return {"filename": filename, "chars": len(text), "chunks_added": added}


def ingest_paths(paths, namespace="default"):
    """批量入库，返回结果列表（含错误）。"""
    results = []
    for p in paths:
        try:
            r = ingest_file(p, namespace=namespace)
            results.append({"ok": True, **r})
        except Exception as e:
            results.append({"ok": False, "filename": os.path.basename(p), "error": str(e)})
    return results


# ----------------------------------------------------------------- CLI
def _main():
    import sys
    import glob as _glob
    args = sys.argv[1:]
    if not args:
        print("用法：python ingest_docs.py 文件1.pdf 文件2.docx ...")
        return
    paths = []
    for a in args:
        paths.extend(_glob.glob(a)) if any(c in a for c in "*?") else paths.append(a)
    res = ingest_paths(paths)
    for r in res:
        if r["ok"]:
            print("✅ {0}：{1} 字 → 入库 {2} 块".format(r["filename"], r["chars"], r["chunks_added"]))
        else:
            print("❌ {0}：{1}".format(r["filename"], r["error"]))
    st = kb_stats()
    print("--- 当前知识库：{0} 份文档，{1} 块 ---".format(len(st["docs"]), st["chunks"]))


if __name__ == "__main__":
    _main()
