"""Local HTTP API used by the notebook. No CLI or web UI is required."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import Config
from .evaluate import render_markdown
from .runner import Runner, RunnerError


class Api:
    def __init__(self, runner: Runner):
        self.runner = runner

    def get(self, path: str, query: dict[str, list[str]]) -> tuple[int, Any]:
        if path == "/health":
            return 200, {"ok": True}
        if path == "/runs":
            return 200, self.runner.list_runs()
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "runs":
            run_id = parts[1]
            if len(parts) == 2:
                return 200, self.runner.state(run_id)
            if parts[2] == "events":
                after = int(query.get("after", ["0"])[0])
                return 200, self.runner.events(run_id, after)
            if parts[2] == "report":
                return 200, self.runner.report(run_id)
            if parts[2] == "report-markdown":
                return 200, {"markdown": render_markdown(self.runner.report(run_id))}
        return 404, {"error": "not found"}

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        parts = [part for part in path.split("/") if part]
        if parts == ["runs"]:
            return 202, self.runner.start(body or {})
        if len(parts) == 3 and parts[0] == "runs":
            run_id = parts[1]
            if parts[2] == "stop":
                return 202, self.runner.stop(run_id)
            if parts[2] == "reevaluate":
                return 200, self.runner.reassess(run_id)
        return 404, {"error": "not found"}


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                status, payload = api.get(parsed.path, parse_qs(parsed.query))
            except RunnerError as exc:
                status, payload = 409, {"error": str(exc)}
            except (OSError, json.JSONDecodeError) as exc:
                status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
            self._send(status, payload)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON body"})
                return
            try:
                status, payload = api.post(parsed.path, body)
            except RunnerError as exc:
                status, payload = 409, {"error": str(exc)}
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
            self._send(status, payload)

    return Handler


def serve(cfg: Config, host: str = "127.0.0.1", port: int | None = None) -> None:
    runner = Runner(cfg)
    server = ThreadingHTTPServer((host, port or cfg.port), make_handler(Api(runner)))
    server.daemon_threads = True
    print(f"pipeline runner listening on http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            runner.shutdown()
        finally:
            server.server_close()
