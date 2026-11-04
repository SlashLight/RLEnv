"""Local tool gateway: the pi extension's only path into the sandbox."""

from __future__ import annotations

import base64
import json
import secrets
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import docker
from .runlog import EventLog
from .sandbox import Sandbox, SandboxError, SandboxInfrastructureError


class Gateway:
    def __init__(self, sandbox: Sandbox, log: EventLog, output_limit: int):
        self.sandbox = sandbox
        self.log = log
        self.output_limit = output_limit
        self.token = secrets.token_urlsafe(32)
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # noqa: D401 - silence stderr
                return

            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if self.headers.get("X-Pipeline-Token") != gateway.token:
                    self._reply(403, {"error": "forbidden"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    request = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._reply(400, {"error": "invalid json"})
                    return
                started = time.monotonic()
                op = self.path.rsplit("/", 1)[-1]
                try:
                    payload, audit = gateway.handle(op, request)
                except FileNotFoundError as exc:
                    gateway.log.publish("sandbox", "tool_error", {
                        "op": op, "path": request.get("path"), "error": "not_found",
                        "infrastructure_error": False,
                    })
                    self._reply(404, {"error": f"not found: {exc}"})
                    return
                except SandboxError as exc:
                    gateway.log.publish("sandbox", "tool_error", {
                        "op": op, "path": request.get("path"), "error": str(exc),
                        "infrastructure_error": isinstance(exc, SandboxInfrastructureError),
                    })
                    self._reply(500, {"error": str(exc)})
                    return
                except Exception as exc:  # keep the agent loop alive, record the failure
                    gateway.log.publish("sandbox", "tool_error", {
                        "op": op, "path": request.get("path"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "infrastructure_error": isinstance(exc, (docker.DockerError, subprocess.TimeoutExpired)),
                    })
                    self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                audit["duration_ms"] = round((time.monotonic() - started) * 1000)
                gateway.log.publish("sandbox", "tool_exec", audit)
                self._reply(200, payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def handle(self, op: str, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        sandbox = self.sandbox
        if op == "read":
            path = str(request["path"])
            data = sandbox.read_file(path)
            return {"data": base64.b64encode(data).decode("ascii")}, {"op": op, "path": path,
                                                                       "bytes": len(data)}
        if op == "access":
            path = str(request["path"])
            return {"ok": sandbox.access(path)}, {"op": op, "path": path}
        if op == "mimetype":
            path = str(request["path"])
            mime = sandbox.detect_image_mime(path)
            return {"mime": mime}, {"op": op, "path": path, "mime": mime}
        if op == "write":
            path = str(request["path"])
            data = base64.b64decode(request.get("data", ""))
            sandbox.write_file(path, data)
            return {"ok": True}, {"op": op, "path": path, "bytes": len(data)}
        if op == "mkdir":
            path = str(request["path"])
            sandbox.mkdir(path)
            return {"ok": True}, {"op": op, "path": path}
        if op == "bash":
            command = str(request["command"])
            cwd = request.get("cwd")
            result = sandbox.bash(command, cwd, request.get("timeout"))
            audit = {"op": op, "command": command, "cwd": cwd, "exit_code": result["exit_code"],
                     "timed_out": result["timed_out"], "truncated": result["truncated"],
                     "output": result["output"]}
            payload = {key: result[key] for key in ("exit_code", "timed_out", "output",
                                                    "truncated", "duration_ms")}
            if "syntax_check" in result:
                audit["syntax_check"] = result["syntax_check"]
                payload["syntax_check"] = result["syntax_check"]
            return payload, audit
        raise SandboxError(f"unknown tool operation: {op}")
