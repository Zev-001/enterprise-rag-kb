"""
vector_store.py —— 持久化向量库（P1 补强：D12 生产级检索底座）。

P0 之前的检索每次都「重新加载 bge 模型 + 重建 FAISS 索引」——知识库一大就慢，
且进程重启必重来，没法扛生产流量。这里把 FAISS 索引落盘（data/idx_<ns>.faiss）
并带签名校验：知识库没变就直接读盘，变了才重建。

设计：
  - 索引文件按 namespace 哈希隔离，元数据（meta）存 chunk_id 签名用于判断「是否过期」。
  - 语义检索不可用（缺依赖/没网）时返回 None，调用方自动回落 TF-IDF，绝不打断检索。
  - 模型在模块内缓存，避免重复加载。
"""

import hashlib
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

_MODEL = None  # 模块级缓存 bge 模型


def _idx_paths(namespace):
    h = hashlib.md5(namespace.encode("utf-8")).hexdigest()[:12]
    return (os.path.join(DATA_DIR, "idx_{0}.faiss".format(h)),
            os.path.join(DATA_DIR, "meta_{0}.faiss.json".format(h)))


def _signature(chunk_ids):
    return hashlib.md5("|".join(chunk_ids).encode("utf-8")).hexdigest()


def build_or_load(namespace, texts, chunk_ids):
    """返回 ("semantic", (model, faiss_index))；失败时返回 None（调用方回落 tfidf）。"""
    global _MODEL
    idx_path, meta_path = _idx_paths(namespace)
    sig = _signature(chunk_ids)
    # 命中缓存：索引文件存在且 chunk 签名一致 → 直接读盘
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("sig") == sig and len(meta.get("ids", [])) == len(chunk_ids):
                import faiss
                index = faiss.read_index(idx_path)
                if _MODEL is None:
                    from sentence_transformers import SentenceTransformer
                    _MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
                return "semantic", (_MODEL, index)
        except Exception:
            pass
    # 未命中：构建并落盘
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if _MODEL is None:
            _MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        vecs = _MODEL.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "ids": list(chunk_ids), "dim": vecs.shape[1]},
                      f, ensure_ascii=False)
        return "semantic", (_MODEL, index)
    except Exception:
        return None


def drop(namespace):
    """知识库被清空/重建时，删掉对应索引缓存，避免读到旧的。"""
    for p in _idx_paths(namespace):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
