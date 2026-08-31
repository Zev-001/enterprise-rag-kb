"""
chunker.py —— D13 语义分块（P2 / 阶段 C）。

原来的 `rag_core.chunk_text` 只有「固定 320 字硬切」一种策略（报告 D13）。
高端系统都按文档结构切块：法律/手册按「条」、手册按「章节」、通用按「语义句群」。
这里提供 4 种模式，零额外依赖，全部可离线。

模式
    legacy     原样保留旧行为（固定字数硬切 + 重叠），向后兼容
    semantic   按段落 → 句子边界切，不切断句子；章节标题作为新块起点（默认）
    proposition 每个句子独立成块（命题级），适合事实型问答
    template   按「条/款」切（中文法条 `第X条`、编号列表），适合手册/规章

切块只是改变了「资料怎么切成小段」，不影响检索算法与防幻觉纪律。
"""
import re

# 章节标题：markdown / 中文「第X章」/ 编号列表 /【小节】
_SECTION_RE = re.compile(
    r"^(#{1,4}\s+|第[一二三四五六七八九十\d]+[章节部]\s*|"
    r"(\d+\.)+[0-9]*\s*|【[^】]{1,20}】\s*)"
)
# 条款边界：中文法条 / 数字或中文编号
_CLAUSE_RE = re.compile(r"^第[一二三四五六七八九十\d]+条\s*|^(\d+|[一二三四五六七八九十]+)[\.\、\s]")
# 中文句末标点
_SENT_SPLIT_RE = re.split


def _split_paragraphs(text):
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _split_sentences(text):
    """按 。！？ 切句，保留标点；段内换行先并成一句。"""
    s = (text or "").replace("\r", "").replace("\n", "")
    parts = re.split(r"(?<=[。！？])", s)
    return [p.strip() for p in parts if p.strip()]


def _detect_sections(paragraphs):
    """把段落序列切成 (heading, [body paragraphs])。无标题时 heading=None。"""
    sections, cur_head, cur_body = [], None, []
    for p in paragraphs:
        first = p.split("\n", 1)[0].strip()
        if _SECTION_RE.match(first):
            if cur_body or cur_head is not None:
                sections.append((cur_head, cur_body))
            cur_head = first
            rest = p[len(first):].strip()
            cur_body = [rest] if rest else []
        else:
            cur_body.append(p)
    if cur_body or cur_head is not None:
        sections.append((cur_head, cur_body))
    return sections


def _hard_cut(s, max_chars, overlap):
    out = []
    for i in range(0, len(s), max_chars - overlap):
        out.append(s[i:i + max_chars])
    return out


def _semantic_chunk(body, max_chars, overlap):
    """按句子边界累积成块，不切断句子；长句本身超长才硬切。"""
    sentences = []
    for b in body:
        sentences.extend(_split_sentences(b))
    chunks, cur = [], []
    for s in sentences:
        trial = "".join(cur + [s])
        if len(trial) <= max_chars or not cur:
            cur.append(s)
        else:
            chunks.append("".join(cur).strip())
            carry = cur[-2:] if len(cur) >= 2 else cur
            cur = list(carry)
            if len("".join(cur + [s])) > max_chars:
                if len(s) > max_chars:
                    chunks.extend(_hard_cut(s, max_chars, overlap))
                    cur = []
                else:
                    cur = [s]
            else:
                cur.append(s)
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c and c.strip()]


def _proposition_chunk(body, max_chars):
    """命题级：每个句子独立成块，按 max_chars 归并。"""
    sentences = []
    for b in body:
        sentences.extend(_split_sentences(b))
    chunks, cur = [], []
    for s in sentences:
        if len("".join(cur + [s])) <= max_chars:
            cur.append(s)
        else:
            if cur:
                chunks.append("".join(cur).strip())
            cur = [s]
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c and c.strip()]


def _template_chunk(head, body, max_chars):
    """按「条」切：法条/编号行作为块边界。无条款标记时回退到语义切块。"""
    units, cur_h, cur_t = [], head, ""
    for b in body:
        first = b.split("\n", 1)[0].strip()
        if _CLAUSE_RE.match(first) and (cur_t or cur_h is not None):
            units.append((cur_h, cur_t.strip()))
            rest = b[len(first):].strip()
            cur_h, cur_t = first, rest
        else:
            cur_t = (cur_t + "\n" + b).strip() if cur_t else b
    if cur_t.strip() or cur_h is not None:
        units.append((cur_h, cur_t.strip()))

    if not units:
        return [(None, c) for c in _semantic_chunk(body, max_chars, 0)]

    out = []
    for h, t in units:
        if not t:
            continue
        if len(t) <= max_chars:
            out.append((h, t))
        else:
            for i, c in enumerate(_semantic_chunk([t], max_chars, 64)):
                out.append((h + ("（续）" if i else ""), c))
    return out


def _legacy_chunk(text, max_chars=320, overlap=64):
    """原 `rag_core.chunk_text` 完整保留，保证向后兼容。"""
    blocks = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(cur) + len(b) <= max_chars:
            cur = (cur + "\n" + b).strip() if cur else b
        else:
            if cur:
                chunks.append(cur)
            if len(b) > max_chars:
                chunks.extend(_hard_cut(b, max_chars, overlap))
                cur = ""
            else:
                cur = b
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text, mode="semantic", max_chars=320, overlap=64):
    """按模式切块，返回文本块列表（不含元数据，元数据由调用方加）。

    text       文档正文
    mode       legacy | semantic | proposition | template
    max_chars  单块上限（ proposition 模式下为句群上限）
    overlap    硬切时的重叠字数（ semantic / legacy 生效）
    """
    text = (text or "").strip()
    if not text:
        return []
    if mode == "legacy":
        return _legacy_chunk(text, max_chars, overlap)

    paragraphs = _split_paragraphs(text)
    sections = _detect_sections(paragraphs) or [(None, paragraphs)]
    chunks = []
    for head, body in sections:
        if mode == "proposition":
            cs = [(head, c) for c in _proposition_chunk(body, max_chars)]
        elif mode == "template":
            cs = _template_chunk(head, body, max_chars)
        else:  # semantic
            cs = [(head, c) for c in _semantic_chunk(body, max_chars, overlap)]
        for h, c in cs:
            if h:
                c = (h.strip() + "\n" + c).strip() if c else h.strip()
            if c:
                chunks.append(c)
    return chunks


def chunk_metadata(mode):
    """返回切块方法元数据，随 chunk 一起入库，方便溯源与评测。"""
    return {"chunk_method": mode}