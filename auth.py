"""
auth.py —— 轻量鉴权 + 知识库 ACL（权限隔离）。P0 补强：D5 无鉴权 / D1 无 ACL。

之前任何人都能访问任意知识库（甚至公网谁都能问你的内部文档）。这里加一层
真实但轻量的护栏，适合个人/小团队原型与作品集演示：

  1. 所有 API 调用必须带 token（Header `Authorization: Bearer <token>`
     或 URL `?token=<token>`，或 JSON 体 `{"token": "..."}`）。
  2. `acl.json` 把「账号 → 允许访问的 namespace 列表」写清楚：
       - admin token：可访问全部 namespace（`["*"]`）
       - 普通用户：只能访问被授权的 namespace，访问别的库直接 403
  3. 匿名访问默认关闭。本地调试可设环境变量 `RAG_ALLOW_ANON=1` 临时放开。
  4. admin token 来源：环境变量 `RAG_API_TOKEN`；没设则启动时生成并写入
     `acl.json` 持久化（避免每次重启都变）。

多租户/企业级做法提示（给面试官看）：
  真正的企业权限应「镜像源系统 ACL」（Glean / M365 Purview 那种），
  本模块是「应用层 token + namespace 白名单」的轻量实现，
  已在数据模型上留好扩展位（namespace 即隔离边界）。
"""

import json
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACL_PATH = os.path.join(BASE_DIR, "acl.json")


def default_admin_token():
    env = os.environ.get("RAG_API_TOKEN")
    if env:
        return env
    return "dev-" + secrets.token_hex(12)


def load_acl():
    """返回 {admin: <token>, users: {<账号>: {"token":..., "namespaces":[...]}}}。"""
    if os.path.exists(ACL_PATH):
        try:
            with open(ACL_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("admin", default_admin_token())
                data.setdefault("users", {})
                return data
        except Exception:
            pass
    return {"admin": default_admin_token(), "users": {}}


def ensure_acl_file():
    """首次运行：把 admin token 落盘，保证重启不变。"""
    if os.path.exists(ACL_PATH):
        return
    tok = default_admin_token()
    with open(ACL_PATH, "w", encoding="utf-8") as f:
        json.dump({"admin": tok, "users": {}}, f, ensure_ascii=False, indent=2)
    if os.environ.get("RAG_API_TOKEN"):
        print("[auth] 使用环境变量 RAG_API_TOKEN 作为 admin token")
    else:
        print("[auth] 已生成 admin token 并写入 acl.json（首次运行）：", tok)


def extract_token(req):
    """从 Flask request 里取 token：Header > URL 参数 > JSON 体。"""
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    t = req.args.get("token") or req.form.get("token")
    if t:
        return t
    try:
        body = req.get_json(silent=True) or {}
        return body.get("token")
    except Exception:
        return None


def authorized(token, namespace):
    """返回 (ok: bool, reason: str)。

    ok=True 时 reason 是身份（"admin" / 用户名）；ok=False 时是拒绝原因。
    """
    if os.environ.get("RAG_ALLOW_ANON") == "1":
        return True, "anon"
    acl = load_acl()
    admin = acl.get("admin")
    if token and admin and token == admin:
        return True, "admin"
    users = acl.get("users", {})
    for uname, u in users.items():
        if u.get("token") == token:
            ns_list = u.get("namespaces", [])
            if "*" in ns_list or namespace in ns_list:
                return True, uname
            return False, "该账号无权访问知识库 [{0}]".format(namespace)
    return False, "缺少有效 token（Header 带 `Authorization: Bearer <token>`，或 URL 加 `?token=`）"
