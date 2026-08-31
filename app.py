"""
app.py —— 企业知识库问答系统的 Web 服务（Flask）。

提供三个能力：
    1. 上传公司文档（PDF/Word/TXT/MD）→ 自动解析切块入库
    2. 在网页里提问 → 基于已上传文档作答，并标出来源（可点击溯源）
    3. 查看知识库状态

零前端构建：用原生 HTML/CSS/JS 写聊天界面，不需要 npm/打包，小白也能看懂。

运行：
    set DEEPSEEK_API_KEY=你的key   （或用项目上层 .env，代码会自动读）
    python app.py
    浏览器打开 http://127.0.0.1:5000
"""

import os
import re
import tempfile
import json
import time

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

import rag_core as rc
from rag_core import kb_stats, grounded_ask, _load_env
from ingest_docs import ingest_paths
import auth as auth_mod
import session_store as sess_mod
import rag_log as log_mod
import agentic_rag as agentic_mod
import connectors as conn_mod
import multimodal as mm_mod
import cache as cache_mod
import confidence as conf_score_mod
import schema_qc as qc_mod

# 启动即加载 .env（和 agent.py 同一份），确保换 key 立刻生效
_load_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(BASE_DIR, "templates")
STATIC = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)

# D1+D5 补强：首次运行把 admin token 落盘，保证重启不变
auth_mod.ensure_acl_file()


def _guard(namespace):
    """鉴权守卫：取出 token → 校验该账号能否访问此 namespace。

    通过返回 None；不通过返回 (resp, code)，路由里 `if g: return g` 即可。
    """
    token = auth_mod.extract_token(request)
    ok, why = auth_mod.authorized(token, namespace)
    if not ok:
        return jsonify({"ok": False, "error": why}), 403
    return None


def _is_admin():
    token = auth_mod.extract_token(request)
    ok, identity = auth_mod.authorized(token, "*")
    return ok and identity == "admin"


@app.route("/")
def index():
    return send_from_directory(TEMPLATES, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC, filename)


@app.route("/api/upload", methods=["POST"])
def upload():
    """接收上传的文件，解析切块入库。支持多文件。"""
    namespace = request.form.get("namespace", "default")
    g = _guard(namespace)
    if g:
        return g
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "没有收到文件"}), 400
    paths = []
    tmpdir = tempfile.mkdtemp()
    for f in files:
        if not f.filename:
            continue
        dest = os.path.join(tmpdir, f.filename)
        f.save(dest)
        paths.append(dest)
    results = ingest_paths(paths, namespace=namespace)
    st = kb_stats(namespace)
    return jsonify({
        "ok": True,
        "results": results,
        "kb": {"docs": len(st["docs"]), "chunks": st["chunks"]},
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    namespace = data.get("namespace", "default")
    mode = data.get("mode", "default")        # default | agent（D8）
    retrieval = data.get("retrieval", "default")  # default | graph（D7）
    session_id = data.get("session_id")       # D10 多轮
    modalities = data.get("modalities")      # D17 跨模态：text|table|image
    use_cache = bool(data.get("use_cache", True))  # D15 缓存
    stream = bool(data.get("stream"))        # D15 流式
    g = _guard(namespace)
    if g:
        return g
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400

    # D15：流式走独立生成器（SSE），普通问答走整句
    if stream:
        return _ask_stream(question, namespace=namespace, mode=mode,
                           retrieval=retrieval, session_id=session_id,
                           modalities=modalities)

    # D10：从会话取历史（用于查询改写）
    history = sess_mod.get_history(session_id) if session_id else None
    t0 = time.time()
    try:
        if mode == "agent":
            out = agentic_mod.agentic_ask(question, namespace=namespace, verbose=False)
        else:
            out = grounded_ask(question, namespace=namespace, verbose=False,
                               history=history, retrieval=retrieval,
                               modalities=modalities, use_cache=use_cache)
    except ValueError as e:
        # 通常是没设 API key
        return jsonify({"ok": False, "error": str(e)}), 500

    latency = round(time.time() - t0, 3)
    refused = "资料里没提到" in (out.get("answer") or "")
    # D11：落日志
    log_mod.log_event(
        namespace=namespace, question=question, mode=mode if mode != "default" else retrieval,
        backend=out.get("backend"), hits=out.get("hits", 0),
        latency=latency, ans_len=len(out.get("answer") or ""),
        src_count=len(out.get("sources") or []), refused=refused,
        cached=bool(out.get("from_cache")),
    )
    # D10：把这一轮写进会话（没有 session_id 也建一个，方便前端连续追问）
    if not session_id:
        session_id = sess_mod.new_session()
    sess_mod.append(session_id, question, out.get("answer", ""), out.get("sources"), namespace)

    return jsonify({
        "ok": True,
        "answer": out["answer"],
        "sources": _public_sources(out["sources"]),
        "backend": out["backend"],
        "hits": out["hits"],
        "session_id": session_id,
        "mode": mode if mode != "default" else retrieval,
        "effective_question": out.get("effective_question"),
        "steps": out.get("steps"),
        # D16 置信度
        "confidence": out.get("confidence"),
        "confidence_level": out.get("confidence_level"),
        "confidence_reason": out.get("confidence_reason"),
        # JSON Schema 硬约束质检（schema_qc，借鉴 GEOFlow 质量门禁）
        "qc": out.get("qc"),
        "qc_action": out.get("qc_action", "warn"),
        # D14 上下文压缩 / D15 缓存
        "compress": out.get("compress"),
        "from_cache": bool(out.get("from_cache")),
    })


def _sse(obj):
    """SSE 帧：data: {json}\n\n"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _public_sources(sources):
    """API 出口过滤：剥掉 sources 里的全文 text 字段（D19 回验内部用，不外发）。

    前端只需要 n/title/score/snippet/modality；全文可能几 KB/块，不该出网。
    """
    if not sources:
        return sources
    out = []
    for s in sources:
        if isinstance(s, dict):
            s = {k: v for k, v in s.items() if k != "text"}
        out.append(s)
    return out


def _ask_stream(question, namespace="default", mode="default", retrieval="default",
                session_id=None, modalities=None):
    """D15 流式问答：以 SSE 逐 token 推给前端，边想边打，长答案不卡顿。
    agent 模式暂不支持流式，退回整句一次性下发。
    """
    history = sess_mod.get_history(session_id) if session_id else None
    t0 = time.time()

    def gen():
        nonlocal session_id
        out = None
        try:
            if mode == "agent":
                out = agentic_mod.agentic_ask(question, namespace=namespace, verbose=False)
                yield _sse({"type": "delta", "token": out.get("answer", "")})
            else:
                prepared = rc.prepare_ask(
                    question, namespace=namespace, history=history,
                    retrieval=retrieval, modalities=modalities, use_cache=True)
                if prepared.get("from_cache"):
                    cached_ans = prepared.get("answer", "") or ""
                    yield _sse({"type": "delta", "token": cached_ans, "cached": True})
                    out = dict(prepared)
                else:
                    full = []
                    for tok in rc.ask_llm_stream(rc.SYSTEM_GROUND, prepared["prompt"]):
                        full.append(tok)
                        yield _sse({"type": "delta", "token": tok})
                    answer = "".join(full)
                    conf = conf_score_mod.score_confidence(
                        [(s, {}) for s in prepared.get("scores", [])], answer,
                        threshold=rc.RELEVANCE_THRESHOLD)
                    # JSON Schema 硬约束质检：流式打印结束后再判，不挡打字机体验
                    qc = qc_mod.run_qc(prepared["effective_question"], answer,
                                       prepared["sources"], hits=prepared["hits"],
                                       level=conf["level"])
                    qc_action = "warn"
                    if qc and (qc.get("mode", "") or "").startswith("json"):
                        conf = conf_score_mod.score_confidence(
                            [(s, {}) for s in prepared.get("scores", [])], answer,
                            threshold=rc.RELEVANCE_THRESHOLD, qc=qc)
                        # block 模式：无依据答案就地拦截（前端收到后整体替换显示）
                        answer, qc_action = qc_mod.apply_action(qc, answer)
                    try:
                        cache_mod.cache_set(prepared["namespace"],
                                            prepared["effective_question"],
                                            mode=prepared["retrieval"], value={
                                                "answer": answer,
                                                "sources": prepared["sources"],
                                                "backend": prepared["backend"],
                                                "hits": prepared["hits"],
                                                "effective_question": prepared["effective_question"],
                                                "retrieval": prepared["retrieval"],
                                                "compress": prepared["compress"],
                                                "confidence": conf["confidence"],
                                                "confidence_level": conf["level"],
                                                "confidence_reason": conf["reason"],
                                                "confidence_factors": conf["factors"],
                                                "qc": qc,
                                                "qc_action": qc_action,
                                            })
                    except Exception:
                        pass
                    out = {"question": question, "answer": answer,
                           "sources": prepared["sources"],
                           "backend": prepared["backend"],
                           "hits": prepared["hits"],
                           "effective_question": prepared["effective_question"],
                           "retrieval": retrieval,
                           "confidence": conf["confidence"],
                           "confidence_level": conf["level"],
                           "confidence_reason": conf["reason"],
                           "confidence_factors": conf["factors"],
                           "qc": qc, "qc_action": qc_action,
                           "compress": prepared["compress"],
                           "from_cache": False}
        except ValueError as e:
            yield _sse({"type": "error", "error": str(e)})
            return
        except Exception as e:
            yield _sse({"type": "error", "error": "流式失败：" + str(e)})
            return

        latency = round(time.time() - t0, 3)
        answer = out.get("answer", "")
        refused = "资料里没提到" in answer
        log_mod.log_event(
            namespace=namespace, question=question,
            mode=mode if mode != "default" else retrieval,
            backend=out.get("backend"), hits=out.get("hits", 0),
            latency=latency, ans_len=len(answer),
            src_count=len(out.get("sources") or []), refused=refused,
            cached=bool(out.get("from_cache")),
        )
        if not session_id:
            session_id = sess_mod.new_session()
        sess_mod.append(session_id, question, answer, out.get("sources"), namespace)
        yield _sse({
            "type": "done", "ok": True, "answer": answer,
            "sources": _public_sources(out.get("sources")), "backend": out.get("backend"),
            "hits": out.get("hits"), "session_id": session_id,
            "mode": mode if mode != "default" else retrieval,
            "effective_question": out.get("effective_question"),
            "confidence": out.get("confidence"),
            "confidence_level": out.get("confidence_level"),
            "confidence_reason": out.get("confidence_reason"),
            "qc": out.get("qc"),
            "qc_action": out.get("qc_action", "warn"),
            "compress": out.get("compress"),
            "from_cache": bool(out.get("from_cache")),
        })

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


@app.route("/api/multimodal/ingest", methods=["POST"])
def multimodal_ingest():
    """D17 跨模态入库（仅 admin）。支持 table / image 两种模态，共用同一套检索。
    table: {"type":"table","headers":[...],"rows":[[...],...],"title":"...","filename":"...","namespace":"..."}
    image: {"type":"image","image_b64":"..."/"image_path":"...","title":"...","filename":"...","namespace":"..."}
    """
    if not _is_admin():
        return jsonify({"ok": False, "error": "该接口仅 admin 可用"}), 403
    data = request.get_json(force=True, silent=True) or {}
    mtype = data.get("type")
    namespace = data.get("namespace", "default")
    filename = data.get("filename", "multimodal")
    title = data.get("title", filename)
    doc_id = re.sub(r"\W+", "", filename)[:40] or "doc"
    try:
        if mtype == "table":
            chunk = mm_mod.table_chunk(
                data.get("headers") or [], data.get("rows") or [],
                title, filename, doc_id, 0)
        elif mtype == "image":
            ocr = ""
            if data.get("image_path"):
                ocr = mm_mod.ocr_image_file(data["image_path"])
            chunk = mm_mod.image_chunk(
                ocr, title, filename, doc_id, 0,
                image_path=data.get("image_path"), image_b64=data.get("image_b64"))
        else:
            return jsonify({"ok": False, "error": "未知模态类型：{0}".format(mtype)}), 400
        added = rc.ingest_documents([chunk], namespace=namespace)
        return jsonify({"ok": True, "chunks_added": added,
                        "modality": chunk["modality"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cache", methods=["GET", "DELETE"])
def cache_api():
    """D15 缓存管理：GET 看统计，DELETE 清缓存（仅 admin）。"""
    if request.method == "DELETE":
        if not _is_admin():
            return jsonify({"ok": False, "error": "该接口仅 admin 可用"}), 403
        ns = request.args.get("namespace")
        cache_mod.cache_invalidate(namespace=ns)
        return jsonify({"ok": True, "invalidated_namespace": ns})
    g = _guard("default")
    if g:
        return g
    return jsonify({"ok": True, **cache_mod.cache_stats()})

@app.route("/api/session", methods=["GET", "POST"])
def session_api():
    if request.method == "POST":
        sid = (request.get_json(force=True, silent=True) or {}).get("session_id")
        if not sid:
            return jsonify({"ok": False, "error": "缺 session_id"}), 400
        ok = sess_mod.clear(sid)
        return jsonify({"ok": ok})
    # GET：列出会话（带 token 即可，方便调试）
    return jsonify({"ok": True, "sessions": sess_mod.list_sessions()})


@app.route("/api/logs")
def logs_api():
    g = _guard("default")
    if g:
        return g
    limit = int(request.args.get("limit", 50))
    return jsonify({"ok": True, "logs": log_mod.recent(limit)})


@app.route("/api/metrics")
def metrics_api():
    g = _guard("default")
    if g:
        return g
    return jsonify({"ok": True, **log_mod.get_metrics()})


@app.route("/api/connector/ingest", methods=["POST"])
def connector_ingest():
    """D9：从连接器拉数据入库（仅 admin）。支持 local_folder / web_page。"""
    if not _is_admin():
        return jsonify({"ok": False, "error": "该接口仅 admin 可用"}), 403
    data = request.get_json(force=True, silent=True) or {}
    ctype = data.get("type")
    source = data.get("source", "")
    namespace = data.get("namespace", "default")
    if ctype == "local_folder":
        c = conn_mod.LocalFolderConnector(source)
    elif ctype == "web_page":
        c = conn_mod.WebPageConnector(source)
    else:
        return jsonify({"ok": False, "error": "未知连接器类型：{0}".format(ctype)}), 400
    try:
        res = c.ingest(namespace=namespace)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, **res})


@app.route("/api/stats")
def stats():
    namespace = request.args.get("namespace", "default")
    g = _guard(namespace)
    if g:
        return g
    st = kb_stats(namespace)
    return jsonify({
        "ok": True,
        "docs": st["docs"],
        "chunks": st["chunks"],
        "namespace": namespace,
    })


if __name__ == "__main__":
    # debug=False 生产更稳；端口默认 5000，可用 $env:PORT=5001 覆盖（5000 被占时绕开）
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
