"""
multimodal.py —— D17 多语言 / 跨模态（P2 / 阶段 C）。

原来的系统只吃中文纯文本（报告 D17，与 D3 同源）。这里补齐三块：

  - 模态：text / table / image 三种块共用同一套检索（差异只在 row 的
    `modality` / `meta` 字段，检索算法不动）
  - 表格：结构化保留（headers/rows），检索用渲染后的 Markdown 文本，
    回答时可整表引用（比「把表格拍平成一段话」靠谱得多）
  - 图片：OCR 文本可检索；需要看图时再调视觉模型（RAG_VISION=1 且端点支持，
    默认关，零额外依赖）
  - 多语言：识别提问语言，让模型用同一种语言回答（中文问中文、英文问英文）

零额外依赖（pytesseract 仍是可选）。
"""
import os
import re

MODALITIES = ("text", "table", "image")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def language_of(text):
    """粗识别提问语言：zh / en / other / unknown。"""
    if not text:
        return "unknown"
    if _CJK_RE.search(text):
        return "zh"
    if _LATIN_RE.search(text):
        return "en"
    return "other"


# ---------------------------------------------------------------- 表格
def render_table(headers, rows):
    """把表格渲染成 Markdown 文本，供检索与引用。"""
    headers = [str(h) for h in (headers or [])]
    rows = [[str(c) for c in r] for r in (rows or [])]
    if not headers and not rows:
        return ""
    width = max([len(headers)] + [len(r) for r in rows] or [0])
    headers = headers + [""] * (max(0, width - len(headers)))
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * width) + "|"]
    for r in rows:
        r = r + [""] * (max(0, width - len(r)))
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def table_chunk(headers, rows, title, filename, doc_id, idx):
    """构造一个 table 模态的 chunk（含结构化 meta）。"""
    cid = "{0}_t{1}".format(doc_id, idx)
    return {
        "doc_id": doc_id, "filename": filename, "title": title,
        "chunk_id": cid, "text": render_table(headers, rows),
        "modality": "table",
        "meta": {"headers": headers, "rows": rows, "n_rows": len(rows)},
    }


# ---------------------------------------------------------------- 图片
def image_chunk(ocr_text, title, filename, doc_id, idx, image_path=None, image_b64=None):
    """构造一个 image 模态的 chunk。OCR 文本即检索文本；原图存在 meta 里备用。"""
    cid = "{0}_img{1}".format(doc_id, idx)
    return {
        "doc_id": doc_id, "filename": filename, "title": title,
        "chunk_id": cid, "text": ocr_text or "",
        "modality": "image",
        "meta": {"image_path": image_path, "has_b64": bool(image_b64),
                 "ocr_len": len(ocr_text or "")},
    }


def ocr_image_file(path):
    """可选 OCR：缺 pytesseract / PIL 时返回空字符串，不中断入库。"""
    try:
        import pytesseract
        from PIL import Image
        return (pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng") or "").strip()
    except Exception:
        return ""


def describe_image(image_b64, model=None):
    """视觉描述：仅在 RAG_VISION=1 时调用，需要端点支持 vision；离线返回空。"""
    if os.environ.get("RAG_VISION") != "1" or not image_b64:
        return ""
    try:
        import requests
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        url = os.environ.get("RAG_LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
        resp = requests.post(url, timeout=90,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json={"model": model or os.environ.get("RAG_LLM_MODEL", "deepseek-chat"),
                  "messages": [{"role": "user", "content": [
                      {"type": "text",
                       "text": "用一句话描述这张图里能看到的内容，只说看得见的，不要推测。"},
                      {"type": "image_url",
                       "image_url": {"url": "data:image/png;base64," + image_b64}}]}],
                  "max_tokens": 200})
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- 检索过滤
def filter_by_modalities(pairs, modalities):
    """pairs = [(score, row)]；按 modalities 过滤（None / 空 = 全部模态）。"""
    if not modalities:
        return pairs
    wanted = set(modalities)
    return [(s, r) for s, r in pairs if r.get("modality", "text") in wanted]