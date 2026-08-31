"""
context.py —— D14 上下文压缩 / 去重（P2 / 阶段 C）。

原来的 `rag_core._build_context` 只按字符数硬截断（报告 D14）：重复块全塞进上下文、
超预算的整块丢掉，既浪费 token 又容易丢关键信息。高端系统会：
  ① 去重近似重复块（同一文档被多路检索命中时尤其常见）
  ② 预算内优先保留整块，只在最后一块截断，避免半截块挤掉后面的料
  ③ 压缩后返回信息，供置信度与日志使用

零依赖（标准库正则）。压缩不改语义，只是去掉冗余。
"""
import re

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def _tokens(s):
    return set(_TOKEN_RE.findall(s or ""))


def _jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def compress_context(pairs, cap_chars=1200, dedup_threshold=0.85):
    """把命中块压缩成带 [n] 编号的上下文。

    pairs = [(score, row), ...]（已按相关性从高到低排好）
    返回 (context_str, info)。
      info = {
        compressed : 是否发生了压缩/去重（用于日志标记）
        merged     : 被当作重复合并掉的块数
        dropped    : 因预算丢弃的块数
        chunks_in_ctx : 实际写进上下文的块数
        total_hits : 原始命中数
        ctx_chars  : 上下文字符数
      }
    """
    # ① 去重（保留先出现的，即相关性更高的那块）
    kept, merged = [], 0
    for s, r in pairs:
        dup = False
        for _, rk in kept:
            if _jaccard(r.get("text", ""), rk.get("text", "")) >= dedup_threshold:
                dup = True
                break
        if dup:
            merged += 1
        else:
            kept.append((s, r))

    # ② 预算内优先整块；装不下的一块截一刀就收
    out, used = [], 0
    for n, (s, r) in enumerate(kept, 1):
        title = r.get("title") or r.get("filename") or ""
        line = "[{0}]《{1}》{2}".format(n, title, r.get("text", ""))
        if used + len(line) > cap_chars:
            remaining = cap_chars - used
            if remaining < 80:
                break
            out.append(line[:remaining] + "…")
            used += remaining
            break
        out.append(line)
        used += len(line)

    ctx = "\n".join(out)
    dropped = len(kept) - len(out)
    info = {
        "compressed": (merged > 0) or (len(out) != len(pairs)) or (dropped > 0),
        "merged": merged,
        "dropped": dropped,
        "chunks_in_ctx": len(out),
        "total_hits": len(pairs),
        "ctx_chars": len(ctx),
    }
    return ctx, info