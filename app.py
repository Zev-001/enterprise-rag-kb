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
import tempfile

from flask import Flask, request, jsonify, send_from_directory

from rag_core import kb_stats, grounded_ask, _load_env
from ingest_docs import ingest_paths

# 启动即加载 .env（和 agent.py 同一份），确保换 key 立刻生效
_load_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(BASE_DIR, "templates")
STATIC = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)


@app.route("/")
def index():
    return send_from_directory(TEMPLATES, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC, filename)


@app.route("/api/upload", methods=["POST"])
def upload():
    """接收上传的文件，解析切块入库。支持多文件。"""
    files = request.files.getlist("files")
    namespace = request.form.get("namespace", "default")
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
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400
    try:
        out = grounded_ask(question, namespace=namespace, verbose=False)
    except ValueError as e:
        # 通常是没设 API key
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "answer": out["answer"],
        "sources": out["sources"],
        "backend": out["backend"],
        "hits": out["hits"],
    })


@app.route("/api/stats")
def stats():
    namespace = request.args.get("namespace", "default")
    st = kb_stats(namespace)
    return jsonify({
        "ok": True,
        "docs": st["docs"],
        "chunks": st["chunks"],
        "namespace": namespace,
    })


if __name__ == "__main__":
    # debug=False 生产更稳；端口 5000 是 Flask 默认，方便记忆
    app.run(host="127.0.0.1", port=5000, debug=False)
