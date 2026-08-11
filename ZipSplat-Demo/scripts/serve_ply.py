"""本地 PLY 查看服务:SuperSplat 编辑器 + 任务输出文件。

用法:
    python serve_ply.py [task_id] [--port 8765]

打开 http://localhost:<port>/?url=/files/<task_id>/scene.ply
根路径为 supersplat-editor/dist(可直接拖拽加载任意 .ply)。
"""

import argparse
import http.server
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "supersplat-editor" / "dist"
OUTPUTS = ROOT / "data" / "outputs"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".ply": "application/octet-stream",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def end_headers(self):
        # SuperSplat 渲染所需跨源隔离头
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_GET(self):
        # /files/<task_id>/<name> → data/outputs/<task_id>/<name>
        if self.path.startswith("/files/"):
            rel = self.path[len("/files/"):].strip("/")
            if "/" not in rel:
                self.send_error(404); return
            task_id, name = rel.split("/", 1)
            p = OUTPUTS / task_id / name
            if not p.is_file() or name.rsplit(".", 1)[-1].lower() not in ("ply", "npy"):
                self.send_error(404, f"文件不存在: {task_id}/{name}"); return
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(p.suffix.lower(),
                             "application/octet-stream"))
            self.send_header("Content-Length", str(p.stat().st_size))
            self.end_headers()
            with open(p, "rb") as f:
                self.wfile.write(f.read())
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description="本地 PLY 查看服务")
    parser.add_argument("task_id", nargs="?", default="test123")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ply = OUTPUTS / args.task_id / "scene.ply"
    if not ply.exists():
        print(f"警告: {ply} 不存在", file=sys.stderr)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/?url=/files/{args.task_id}/scene.ply"
        print(f"SuperSplat 编辑器: http://127.0.0.1:{args.port}/")
        print(f"直接打开 PLY:     {url}")
        print("按 Ctrl+C 停止服务")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
