"""Static debugger UI + REST/SSE API (stdlib only)."""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pytorch_encoder.debug_ui.event_hub import publish, subscribe, unsubscribe
from pytorch_encoder.paths import DEFAULT_BENCHMARK, DEFAULT_TRACE

STATIC = Path(__file__).resolve().parent / "static"


class DebugUIHandler(BaseHTTPRequestHandler):
    server_version = "PyTorchEncoderDebugUI/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._send_file(STATIC / "index.html")
        if route.startswith("/static/"):
            rel = route[len("/static/") :]
            target = (STATIC / rel).resolve()
            if not str(target).startswith(str(STATIC.resolve())):
                self.send_error(403)
                return
            return self._send_file(target)

        if route == "/api/benchmark":
            path = getattr(self.server, "benchmark_path", DEFAULT_BENCHMARK)
            if path.is_file():
                return self._send_json(json.loads(path.read_text(encoding="utf-8")))
            return self._send_json({"error": "benchmark not found", "path": str(path)}, 404)

        if route == "/api/trace":
            path = getattr(self.server, "trace_path", DEFAULT_TRACE)
            if not path.is_file():
                return self._send_json({"steps": [], "path": str(path)})
            steps = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
            return self._send_json({"steps": steps, "path": str(path)})

        if route == "/api/manifest":
            path = getattr(self.server, "manifest_path", None)
            if path and path.is_file():
                return self._send_json(json.loads(path.read_text(encoding="utf-8")))
            return self._send_json({})

        if route == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = subscribe()
            try:
                self.wfile.write(b"event: hello\ndata: {\"status\":\"connected\"}\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    chunk = f"event: step\ndata: {msg}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                unsubscribe(q)
            return

        self.send_error(404)


def start_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    benchmark_path: Path | None = None,
    trace_path: Path | None = None,
    manifest_path: Path | None = None,
    daemon: bool = True,
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), DebugUIHandler)
    httpd.benchmark_path = benchmark_path or DEFAULT_BENCHMARK
    httpd.trace_path = trace_path or DEFAULT_TRACE
    httpd.manifest_path = manifest_path
    thread = threading.Thread(target=httpd.serve_forever, name="debug-ui-http", daemon=daemon)
    thread.start()
    publish({"type": "server", "url": f"http://{host}:{port}/"})
    return httpd
