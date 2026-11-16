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
                 thinking: str, *, local: bool = True):
        self.cfg = cfg
        self.log = log
        self.run_dir = Path(run_dir)
        self.extension_path = Path(extension_path)
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self.provider = provider
        self.model = model
        self.thinking = thinking
        self.local = local
        self.submitted = threading.Event()
        self.budget_stop_reason: str | None = None
        self._budget_ready = threading.Event()
        self.usage_path = self.run_dir / "usage.json"
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
        if self.local:
            env.update({
                "PIPELINE_LOCAL": "1",
                "PIPELINE_TOKEN_BUDGET": str(self.cfg.limits.token_budget),
                "PIPELINE_USAGE_PATH": str(self.usage_path.resolve()),
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
        if self.local:
            try:
                version = subprocess.run([pi, "--version"], capture_output=True, text=True,
                                         timeout=10, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError) as exc:
                raise PiError(f"cannot verify pinned pi build: {exc}") from exc
            if version != "1.5.2":
                raise PiError(f"local token accounting requires pi 1.5.2, found {version!r}")
        argv = [
            pi, "--mode", "rpc", "--no-session",
            "--no-extensions", "-e", str(self.extension_path),
            "--no-skills", "--no-prompt-templates", "--no-themes",
            "--no-context-files", "--no-mcp", "--no-approve", "--offline",
            "--no-builtin-tools", "--tools", "read,write,edit,bash",
            "--provider", self.provider, "--model", self.model,
            "--thinking", self.thinking,
        ]
        if self.local:
            argv += ["-e", str(Path(__file__).parent / "extensions" / "token_budget.ts")]
        key = self.cfg.resolved_api_key()
        if key:
            argv += ["--api-key", key]
        return argv

    def start(self) -> None:
        env = self._env()
        self._prepare_auth_dir(env)
        if self.local:
            self.usage_path.write_text(json.dumps({
                "schema": "evaluator-pipeline/usage/v1", "ready": False,
                "token_budget": self.cfg.limits.token_budget, "total_tokens": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "requests": 0,
                "usage_reliable": True, "stop_reason": None,
                "cost": {"kind": "unknown", "value": None, "currency": "USD"},
            }) + "\n", encoding="utf-8")
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
        retry = self.command("set_auto_retry", enabled=False, timeout=10)
        compaction = self.command("set_auto_compaction", enabled=True, timeout=10)
        if self.local and (not retry.get("success") or not compaction.get("success")):
            raise PiError("cannot configure trusted pi retry/compaction policy")
        state = self.command("get_state", timeout=10)
        if self.local:
            if (not self._budget_ready.wait(10) or self.api_error
                    or not state.get("success")):
                raise PiError(self.api_error or "token budget extension did not become ready")
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
        if self.local and kind == "extension_ui_request" and record.get("method") == "notify":
            message = record.get("message")
            if isinstance(message, str) and message.startswith("PIPELINE_BUDGET:"):
                try:
                    payload = json.loads(message.removeprefix("PIPELINE_BUDGET:"))
                    if not isinstance(payload, dict):
                        raise ValueError("budget event must be an object")
                except (ValueError, TypeError) as exc:
                    self.api_error = f"invalid budget extension event: {exc}"
                    return
                record = {"type": "pipeline_budget", **payload}
                kind = "pipeline_budget"
        if kind == "pipeline_budget" and self.local:
            usage = record.get("usage") or {}
            self.log.publish("budget", str(record.get("event", "unknown")), usage)
            if record.get("event") == "ready" and usage.get("ready") is True:
                self._budget_ready.set()
            reason = usage.get("stop_reason")
            if reason in ("token_limit", "usage_unknown"):
                self.budget_stop_reason = reason
                if reason == "usage_unknown":
                    self.api_error = "provider usage is unknown; further requests are forbidden"
            return
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
            if self.local and self.submitted.is_set():
                return "submitted"
            if self.local and self.budget_stop_reason:
                return self.budget_stop_reason
            if stop_event is not None and stop_event.is_set():
                self.request_stop()
            # Settled means no work remains, not that the attempt succeeded.
            if self.api_error:
                return "error"
            if self._stop_requested.is_set():
                return "stopped"
            if self._settled.is_set():
                if not self.local:
                    return "settled"
                if time.monotonic() < deadline:
                    self._settled.clear()
                    response = self.prompt(
                        "Continue research and improve the solution. Only explicit submission "
                        "ends research; use the submit tool when the final artifacts are ready."
                    )
                    if response.get("success") is False:
                        self.api_error = str(response.get("error") or "continuation failed")
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
            if self.local and self.usage_path.exists():
                usage = json.loads(self.usage_path.read_text(encoding="utf-8"))
                if not usage.get("usage_reliable"):
                    usage["stop_reason"] = "usage_unknown"
                    usage["cost"] = {"kind": "unknown", "value": None, "currency": "USD"}
                    self.budget_stop_reason = "usage_unknown"
                    self.usage_path.write_text(json.dumps(usage) + "\n", encoding="utf-8")
                    self.log.publish("runner", "budget_usage_unknown", usage)

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
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, sig)
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
