"""
connectors.py —— 企业连接器（P1 补强：D9 数据源生态）。

高端系统（Glean 100+ 连接器 / M365 接 SharePoint / Coze 接飞书）的护城河就是
「能直接从你已经在用的系统拉数据」。这里做一个可扩展的连接器框架 + 两个开箱即用的实现：

  - LocalFolderConnector：扫本地目录，把 pdf/docx/txt/md 批量入库（演示私有文档同步）。
  - WebPageConnector：抓单个网页正文入库（演示 SaaS/知识站同步；缺 bs4 用正则兜底）。

框架是「基类 + fetch() 返回文档列表」，以后接 Confluence/Notion/飞书，
只要写一个子类实现 fetch() 即可，ingest 逻辑完全复用。
"""

import os

import ingest_docs as ing
import rag_core as rc


class Connector:
    """所有连接器的基类。子类实现 fetch() 返回 [{"filename","title","text"}]。"""

    name = "base"

    def fetch(self):
        raise NotImplementedError

    def ingest(self, namespace="default"):
        docs = self.fetch()
        if not docs:
            return {"ok": False, "error": "没有取到任何内容", "docs": 0}
        added = rc.ingest_documents(docs, namespace=namespace)
        # 连接器写入后，图缓存 / 向量缓存要失效
        try:
            import graph_rag as gr
            gr.clear_cache(namespace)
        except Exception:
            pass
        try:
            import vector_store as vs
            vs.drop(namespace)
        except Exception:
            pass
        return {"ok": True, "docs": len(docs), "chunks_added": added}


class LocalFolderConnector(Connector):
    """扫描本地目录，解析所有支持格式的文件。"""

    name = "local_folder"

    def __init__(self, folder, exts=(".pdf", ".docx", ".txt", ".md")):
        self.folder = folder
        self.exts = exts

    def fetch(self):
        docs = []
        if not os.path.isdir(self.folder):
            raise ValueError("目录不存在：{0}".format(self.folder))
        for root, _, files in os.walk(self.folder):
            for fn in files:
                if fn.lower().endswith(self.exts):
                    path = os.path.join(root, fn)
                    try:
                        filename, text = ing.parse_file(path)
                        if text:
                            docs.append({"filename": filename, "title": filename, "text": text})
                    except Exception:
                        continue
        return docs


class WebPageConnector(Connector):
    """抓取单个网页正文（演示 SaaS/文档站同步）。"""

    name = "web_page"

    def __init__(self, url):
        self.url = url

    def fetch(self):
        import requests
        resp = requests.get(self.url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
        title, text = _extract(html, self.url)
        return [{"filename": title, "title": title, "text": text}]


def _extract(html, url):
    """优先 bs4，否则正则兜底抽正文。返回 (title, text)。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        # 正则兜底：去 script/style，去标签，压 blank
        html = re.sub(r"(?is)<(script|style).*?</\1>", "", html)
        title_m = re.search(r"(?i)<title>(.*?)</title>", html)
        title = title_m.group(1).strip() if title_m else url
        text = re.sub(r"(?s)<[^>]+>", "\n", html)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)


import re  # 放模块尾，保证 _extract 用到
