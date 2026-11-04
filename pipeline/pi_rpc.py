"""Trusted pi process: clean profile, RPC over JSONL, no host tool fallback."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .config import Config
from .runlog import EventLog


class PiError(RuntimeError):
    pass


class PiSession:
    def __init__(self, cfg: Config, log: EventLog, run_dir: Path, extension_path: Path,
                 gateway_url: str, gateway_token: str, provider: str, model: str,
                 thinking: str):
        self.cfg = cfg
        self.log = log
        self.run_dir = Path(run_dir)
        self.extension_path = Path(extension_path)
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self.provider = provider
        self.model = model
        self.thinking = thinking
        self.process: subprocess.Popen[bytes] | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._settled = threading.Event()
        self._exited = threading.Event()
        self._stop_requested = threading.Event()
        self.settled_aborted = False
        self.protocol_error: str | None = None
        self.exit_code: int | None = None
        self._stderr_handle = None
        self._agent_dir: Path | None = None
        self.api_error: str | None = None
        self.assistant_stop_reasons: list[str] = []
        self.saw_tool_call = False
        self._readers: list[threading.Thread] = []

    # -- process ---------------------------------------------------------
    def _env(self) -> dict[str, str]:
        keep = ("PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env.update({
            "HOME": str(self.run_dir / "pi-home"),
            "TMPDIR": str(self.run_dir / "pi-tmp"),
            "PIPELINE_GATEWAY_URL": self.gateway_url,
            "PIPELINE_GATEWAY_TOKEN": self.gateway_token,
            "NO_COLOR": "1",
            "CI": "1",
        })
        (self.run_dir / "pi-home").mkdir(exist_ok=True)
        (self.run_dir / "pi-tmp").mkdir(exist_ok=True)
        return env

    def _prepare_auth_dir(self, env: dict[str, str]) -> None:
        if self.cfg.resolved_api_key() or not self.cfg.auth_file:
            return
        source = Path(self.cfg.auth_file)
        if not source.is_file():
            raise PiError(f"auth file not found: {source}")
        self._agent_dir = Path(tempfile.mkdtemp(prefix="pipeline-pi-auth-"))
        shutil.copyfile(source, self._agent_dir / "auth.json")
        os.chmod(self._agent_dir / "auth.json", 0o600)
        env["PI_CODING_AGENT_DIR"] = str(self._agent_dir)

    def _argv(self) -> list[str]:
        pi = shutil.which("pi")
        if not pi:
            raise PiError("pi executable not found in PATH")
        argv = [
            pi, "--mode", "rpc", "--no-session",
            "--no-extensions", "-e", str(self.extension_path),
            "--no-skills", "--no-prompt-templates", "--no-themes",
            "--no-context-files", "--no-mcp", "--no-approve", "--offline",
            "--no-builtin-tools", "--tools", "read,write,edit,bash",
            "--provider", self.provider, "--model", self.model,
            "--thinking", self.thinking,
        ]
        key = self.cfg.resolved_api_key()
        if key:
            argv += ["--api-key", key]
        return argv

    def start(self) -> None:
        env = self._env()
        self._prepare_auth_dir(env)
        (self.run_dir / "pi-cwd").mkdir(exist_ok=True)
        self._stderr_handle = (self.run_dir / "raw" / "pi.stderr.log").open("ab")
        self.process = subprocess.Popen(
            self._argv(),
            cwd=self.run_dir / "pi-cwd",
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            bufsize=0,
            start_new_session=True,
        )
        for target, name in ((self._read_stdout, "pi-stdout"), (self._watch_exit, "pi-exit")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._readers.append(thread)
            thread.start()
        time.sleep(0.5)
        if self._exited.is_set():
            raise PiError(
                f"pi exited during startup ({self.exit_code}); see raw/pi.stderr.log"
            )
        # No automatic retries inside the run; compaction stays allowed.
        self.command("set_auto_retry", enabled=False, timeout=10)
        self.command("set_auto_compaction", enabled=True, timeout=10)
        state = self.command("get_state", timeout=10)
        self.log.publish("runner", "pi_started", {"state": state.get("data")})

    def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        for raw in self.process.stdout:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                self.protocol_error = f"invalid JSON from pi: {exc}"
                self.log.publish("runner", "protocol_error", {"line": line[:2000].decode("utf-8", "replace")})
                continue
            if not isinstance(record, dict):
                continue
            raw_line = self.log.raw_record(record)
            self.log.publish("pi", str(record.get("type", "unknown")), record, raw_line=raw_line)
            self._inspect(record)

    def _inspect(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        if kind == "response":
            request_id = record.get("id")
            if request_id:
                with self._pending_lock:
                    pending = self._pending.pop(request_id, None)
                if pending:
                    pending["response"] = record
                    pending["event"].set()
        elif kind == "agent_settled":
            self.settled_aborted = bool(record.get("aborted"))
            self._settled.set()
        elif kind == "tool_execution_start":
            self.saw_tool_call = True
        elif kind == "message_end":
            message = record.get("message") or {}
            if message.get("role") == "assistant":
                reason = message.get("stopReason")
                if reason:
                    self.assistant_stop_reasons.append(str(reason))
                    if reason == "error":
                        self.api_error = json.dumps(message.get("errorMessage") or "model error")
        elif kind == "auto_retry_end" and record.get("success") is False:
            self.api_error = str(record.get("finalError") or "retry failed")
        elif kind == "extension_error":
            self.api_error = f"extension error: {record.get('error')}"

    def _watch_exit(self) -> None:
        assert self.process
        self.exit_code = self.process.wait()
        self._exited.set()
        with self._pending_lock:
            for pending in self._pending.values():
                pending["event"].set()
            self._pending.clear()
        self.log.publish("runner", "pi_exited", {"exit_code": self.exit_code,
                                                 "stop_requested": self._stop_requested.is_set()})

    # -- commands --------------------------------------------------------
    def command(self, kind: str, timeout: float = 30, **fields: Any) -> dict[str, Any]:
        if not self.process or not self.process.stdin:
            raise PiError("pi is not running")
        request_id = f"{kind}-{time.time_ns()}"
        event = threading.Event()
        slot: dict[str, Any] = {"event": event}
        with self._pending_lock:
            self._pending[request_id] = slot
        payload = {"id": request_id, "type": kind, **fields}
        try:
            self.process.stdin.write(json.dumps(payload).encode() + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._pending.pop(request_id, None)
            raise PiError(f"cannot write to pi: {exc}") from exc
        if not event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise PiError(f"pi command {kind} timed out")
        response = slot.get("response")
        if response is None:
            raise PiError(f"pi exited while handling {kind}")
        return response

    def prompt(self, text: str) -> dict[str, Any]:
        return self.command("prompt", message=text, timeout=30)

    def wait_settled(self, deadline: float,
                     stop_event: threading.Event | None = None) -> str:
        while True:
            if stop_event is not None and stop_event.is_set():
                self.request_stop()
            # Settled means no work remains, not that the attempt succeeded.
            if self.api_error:
                return "error"
            if self._stop_requested.is_set():
                return "stopped"
            if self._settled.is_set():
                return "settled"
            if self._exited.is_set():
                return "error"
            if time.monotonic() >= deadline:
                return "time_limit"
            time.sleep(0.1)

    def request_stop(self) -> None:
        self._stop_requested.set()

    def close(self) -> None:
        """Cooperative abort, then bounded process-group shutdown and profile cleanup."""
        try:
            self._close_process()
        finally:
            self._cleanup_profile()

    def _close_process(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            for kind in ("clear_queue",):
                try:
                    self.command(kind, timeout=5)
                except PiError:
                    pass
            try:
                self.command("abort", timeout=5)
            except PiError:
                pass
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    self.process.wait(timeout=5)
        self._signal_group(signal.SIGKILL)
        for thread in self._readers:
            thread.join(timeout=5)

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def _cleanup_profile(self) -> None:
        if self._stderr_handle:
            try:
                self._stderr_handle.close()
            except OSError:
                pass
        if self._agent_dir:
            shutil.rmtree(self._agent_dir, ignore_errors=True)
            self._agent_dir = None
