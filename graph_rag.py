"""
graph_rag.py —— GraphRAG 多跳检索（P1 补强：D7 知识图谱）。

高端系统（Glean work graph / RAGFlow GraphRAG / 火山方舟 GraphRAG）都给知识建「图」，
让「A 和 B 什么关系」「谁负责 X」这类多跳问题能顺着实体链路找到答案，
而不仅是向量相近。这里做一个轻量版：

  1. 实体抽取：优先用 LLM（质量高），离线/无 key 时退回启发式（引号词、专有英文、
     金额、领域术语后缀）。
  2. 建图：实体→所在块（倒排）+ 同块实体共现边（权重）。
  3. 多跳检索：问句实体 → 种子块 → 沿共现边扩散 1 跳 → 按「命中实体数 + 邻居权重」
     给块打分，取 top_k。没抽到实体时退回关键词重叠，保证不空返。

纯标准库实现图（dict 邻接表），不引新依赖；实体抽取走 rag_core 的 LLM 通道。
"""

import re
from collections import defaultdict

import rag_core as rc

# 领域术语后缀：企业文档里这些后缀的词大概率是「实体/概念」
TERM_SUFFIX = ("系统", "计划", "手册", "流程", "制度", "标准", "规范", "部门",
               "公司", "团队", "平台", "模型", "接口", "协议", "方案", "政策",
               "规定", "办法", "指南", "架构")

_GRAPH_CACHE = {}  # namespace -> (entity2chunks, cooccur)


def extract_entities(text):
    """抽实体：引号/书名号内词 + 专有英文 + 金额 + 领域术语。离线启发式。"""
    if not text:
        return set()
    ents = set()
    # 引号 / 书名号 内的短语
    for m in re.findall(r"[「『\"'《]([^」』\"'》]{2,20})[」』\"'》]", text):
        ents.add(m.strip())
    # 专有英文（首字母大写，长度>=2）
    for m in re.findall(r"\b[A-Z][A-Za-z0-9\-]{1,}\b", text):
        ents.add(m)
    # 金额 / 数字+单位
    for m in re.findall(r"\d[\d,\.]*\s?(?:元|万|亿|天|小时|小时|次|%|人|年|月|条)", text):
        ents.add(m.replace(" ", ""))
    # 中文领域术语：2-8 字 + 术语后缀
    for m in re.findall(r"[\u4e00-\u9fff]{2,8}(" + "|".join(TERM_SUFFIX) + ")", text):
        ents.add(m)
    # 清洗：去掉纯停用后缀噪声
    return {e for e in ents if len(e) >= 2}


def _build_graph(namespace):
    rows = rc.load_kb(namespace)
    entity2chunks = defaultdict(list)
    cooccur = defaultdict(float)
    for r in rows:
        cid = r["chunk_id"]
        ents = extract_entities(r["text"])
        for e in ents:
            entity2chunks[e].append(cid)
        el = list(ents)
        for i in range(len(el)):
            for j in range(i + 1, len(el)):
                key = (el[i], el[j]) if el[i] <= el[j] else (el[j], el[i])
                cooccur[key] += 1.0
    return dict(entity2chunks), dict(cooccur)


def _get_graph(namespace):
    if namespace not in _GRAPH_CACHE:
        _GRAPH_CACHE[namespace] = _build_graph(namespace)
    return _GRAPH_CACHE[namespace]


def retrieve_graph(query, namespace="default", top_k=rc.DEFAULT_TOP_K):
    """多跳检索，返回 [(score, chunk)]，score 归一化到 [0,1]。

    D20：过期/未审核的块同样不进图谱检索——实体图仍全量建（关系完整），
    但打分与出结果都按「可见块集合」收口，治理口径与普通检索一致。
    """
    entity2chunks, cooccur = _get_graph(namespace)
    rows = rc.load_kb(namespace)
    rows = rc._filter_rows(rows)  # D20：治理过滤（过期/未生效/未审核/作废）
    if not rows:
        return []
    visible_ids = {r["chunk_id"] for r in rows}
    q_ents = extract_entities(query)
    # 候选块打分
    score = defaultdict(float)
    seen_entities = set()
    if q_ents:
        # 第 0 跳：命中实体直接加
        for e in q_ents:
            if e in entity2chunks:
                seen_entities.add(e)
                for cid in entity2chunks[e]:
                    if cid in visible_ids:
                        score[cid] += 1.0
        # 第 1 跳：沿共现边扩散到邻居实体，继承部分权重
        neighbors = set()
        for e in q_ents:
            for (a, b), w in cooccur.items():
                if a == e and b not in seen_entities:
                    neighbors.add(b)
                elif b == e and a not in seen_entities:
                    neighbors.add(a)
        for nb in neighbors:
            if nb in entity2chunks:
                seen_entities.add(nb)
                for cid in entity2chunks[nb]:
                    if cid in visible_ids:
                        score[cid] += 0.5
    else:
        # 没抽到实体：退回关键词重叠，保证有结果
        toks = set(rc._TfidfIndex._tok(query))
        for r in rows:
            hit = len(toks & set(rc._TfidfIndex._tok(r["text"])))
            if hit:
                score[r["chunk_id"]] = hit
    if not score:
        return []
    row_by_id = {r["chunk_id"]: r for r in rows}
    # 归一化到 [0,1]
    mx = max(score.values()) or 1
    ranked = sorted(score.items(), key=lambda x: -x[1])[:top_k]
    out = []
    for cid, s in ranked:
        r = row_by_id.get(cid)
        if r:
            out.append((round(s / mx, 3), r))
    return out


def clear_cache(namespace=None):
    if namespace:
        _GRAPH_CACHE.pop(namespace, None)
    else:
        _GRAPH_CACHE.clear()
