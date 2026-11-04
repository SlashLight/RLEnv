"""One-shot tool container and submission snapshot collection."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from . import docker
from .config import Config
from .runlog import EventLog
from .task import Task
from .util import ProcessResult, tree_manifest


# The checker runs before any agent command, with no imports or startup files from it.
_BASH_SYNTAX_CHECK = r'''
import json, subprocess, sys
try:
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-O", "extglob", "-n", "-c", sys.argv[1]],
        env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=float(sys.argv[2]),
    )
    verdict = {0: "valid", 2: "invalid"}.get(result.returncode, "check_error")
    evidence = {"verdict": verdict, "parser_exit_code": result.returncode}
except (OSError, subprocess.TimeoutExpired):
    evidence = {"verdict": "check_error", "reason": "parser_unavailable_or_timeout"}
print(json.dumps(evidence))
'''


# Only this supervisor owns the Docker stdout pipe; Bash gets a separate pipe.
_BASH_RUN = r'''
import base64, ctypes, json, os, selectors, signal, subprocess, sys, time
# Deny same-uid children access to supervisor descriptors through /proc.
if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
    raise OSError("cannot protect supervisor descriptors")
budget, limit = float(sys.argv[2]), int(sys.argv[3])
proc = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-O", "extglob", "-c", sys.argv[1]],
                        env={k: v for k, v in os.environ.items()
                             if k not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}},
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, start_new_session=True)
end = time.monotonic() + budget
output = bytearray()
truncated = timed_out = killed = False

def kill_group(sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass

with selectors.DefaultSelector() as selector:
    os.set_blocking(proc.stdout.fileno(), False)
    selector.register(proc.stdout, selectors.EVENT_READ)
    while selector.get_map() or proc.poll() is None:
        now = time.monotonic()
        if now >= end and not timed_out:
            timed_out = True
            kill_group(signal.SIGTERM)
        if timed_out and now >= end + 5 and not killed:
            kill_group(signal.SIGKILL)
            killed = True
        if timed_out and now >= end + 6:
            truncated |= bool(selector.get_map())
            break
        for key, _ in selector.select(0.05):
            data = os.read(key.fd, 65536)
            if not data:
                selector.unregister(key.fileobj)
                continue
            room = limit - len(output)
            output.extend(data[:room])
            truncated |= len(data) > room
if timed_out:
    kill_group(signal.SIGKILL)
proc.stdout.close()
code = proc.wait(timeout=1)
print(json.dumps({"exit_code": 124 if timed_out else (code if code >= 0 else 128 - code),
                  "timed_out": timed_out, "truncated": truncated,
                  "output_b64": base64.b64encode(output).decode("ascii")}))
'''


class SandboxError(RuntimeError):
    pass


class SandboxInfrastructureError(SandboxError):
    pass


MIME_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class Sandbox:
    """All agent tool operations execute here; there is no host fallback."""

    def __init__(self, cfg: Config, run_id: str, workspace: Path, log: EventLog,
                 pipeline_dir: Path):
        self.cfg = cfg
        self.run_id = run_id
        self.workspace = workspace
        self.log = log
        self.pipeline_dir = pipeline_dir
        self.network = f"pipeline-{run_id.lower()}"
        self.container = f"pipeline-{run_id.lower()}-tools"
        self.tools_image = ""
        self.tools_image_id = ""
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.tools_image_id = docker.ensure_tools_image(self.cfg, self.pipeline_dir)
        self.tools_image = self.cfg.docker.tools_image
        docker.create_network(self.network)
        docker.start_tools_container(
            self.tools_image, self.network, self.container, self.workspace, self.cfg
        )
        self.started = True

    def stop(self) -> None:
        """Terminate all tool processes before copying evidence from the container."""
        docker.stop_container(self.container)
        self.stopped = True

    def cleanup(self) -> list[str]:
        errors: list[str] = []
        try:
            docker.remove_container(self.container)
        except docker.DockerError as exc:
            errors.append(f"tools container: {exc}")
        try:
            docker.remove_network(self.network)
        except docker.DockerError as exc:
            errors.append(f"network: {exc}")
        return errors

    def _exec(self, argv: list[str], timeout: float, workdir: str | None = None,
              input_data: bytes | None = None, output_limit: int | None = None) -> ProcessResult:
        return docker.exec_in(self.container, argv, timeout=timeout, workdir=workdir,
                              input_data=input_data,
                              output_limit=(self.cfg.tool_output_limit_bytes
                                            if output_limit is None else output_limit))

    def read_file(self, path: str) -> bytes:
        result = self._exec(["cat", "--", path], timeout=30)
        if result.returncode != 0:
            if b"No such file" in result.stderr or b"cannot open" in result.stderr:
                raise FileNotFoundError(path)
            raise SandboxError(f"read failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        if result.stdout_truncated:
            raise SandboxError("read exceeds output limit; use bash with head or a line range")
        return result.stdout

    def access(self, path: str) -> bool:
        return self._exec(["test", "-r", path], timeout=15).returncode == 0

    def detect_image_mime(self, path: str) -> str | None:
        result = self._exec(["head", "-c", "16", "--", path], timeout=15)
        if result.returncode != 0:
            return None
        out = result.stdout
        for magic, mime in MIME_MAGIC:
            if out.startswith(magic):
                return mime
        if out[:4] == b"RIFF" and out[8:12] == b"WEBP":
            return "image/webp"
        return None

    def write_file(self, path: str, data: bytes) -> None:
        if len(data) > self.cfg.submission_limit_bytes:
            raise SandboxError(
                f"write rejected: {len(data)} bytes exceeds {self.cfg.submission_limit_bytes}"
            )
        result = self._exec(["sh", "-c", 'cat > "$1"', "sh", path], timeout=30,
                            input_data=data)
        if result.returncode != 0:
            raise SandboxError(f"write failed: {result.stderr.decode('utf-8', 'replace').strip()}")

    def mkdir(self, path: str) -> None:
        result = self._exec(["mkdir", "-p", "--", path], timeout=15)
        if result.returncode != 0:
            raise SandboxError(f"mkdir failed: {result.stderr.decode('utf-8', 'replace').strip()}")

    def _bash_syntax(self, executable_command: str, budget: float) -> dict[str, Any]:
        # shortcut: parses only top-level source, not runtime eval or sourced files.
        evidence = {
            "verdict": "check_error", "language": "bash", "parser": "/bin/bash -O extglob -n",
            "source_sha256": hashlib.sha256(executable_command.encode()).hexdigest(),
            "scope": "tool_input_preflight", "container": self.container,
            "evidence_version": 3,
        }
        try:
            result = self._exec(
                ["/usr/bin/timeout", "-k", "1s", f"{budget + 1:g}s",
                 "/usr/local/bin/python", "-I", "-S", "-c", _BASH_SYNTAX_CHECK,
                 executable_command, str(budget)],
                timeout=budget + 5, workdir="/",
            )
            if result.returncode != 0 or result.stdout_truncated:
                return {**evidence, "reason": "checker_failed"}
            parsed = json.loads(result.stdout)
            if (not isinstance(parsed, dict)
                    or parsed.get("verdict") not in ("valid", "invalid", "check_error")):
                return {**evidence, "reason": "invalid_checker_output"}
            return {**evidence, **parsed}
        except (docker.DockerError, OSError, ValueError):
            return {**evidence, "reason": "checker_unavailable"}

    def bash(self, command: str, cwd: str | None, timeout: float | None) -> dict[str, Any]:
        budget = min(float(timeout or self.cfg.limits.tool_seconds),
                     self.cfg.limits.tool_seconds)
        started = time.monotonic()
        # `exec 2>&1` keeps the interleaving that the agent sees.
        executable_command = f"exec 2>&1\n{command}"
        syntax_check = self._bash_syntax(executable_command, min(5.0, budget))
        # Reject invalid tool input before any effects: this is an admission check,
        # not a claim about arbitrary dynamically constructed shell programs.
        if syntax_check["verdict"] == "invalid":
            return {"syntax_check": syntax_check, "exit_code": 2, "timed_out": False,
                    "output": "[pipeline: bash syntax preflight rejected the command; nothing executed]",
                    "truncated": False, "duration_ms": round((time.monotonic() - started) * 1000)}
        if syntax_check["verdict"] != "valid":
            raise SandboxInfrastructureError("bash syntax checker unavailable; command not executed")
        limit = self.cfg.tool_output_limit_bytes
        result = self._exec(
            ["/usr/local/bin/python", "-I", "-S", "-c", _BASH_RUN,
             executable_command, str(budget), str(limit)],
            timeout=budget + 15, workdir=cwd or "/workspace",
            output_limit=4 * ((limit + 2) // 3) + 1024,
        )
        if result.returncode or result.stdout_truncated or result.stderr_truncated:
            raise SandboxInfrastructureError("bash supervisor failed; execution status unavailable")
        try:
            receipt = json.loads(result.stdout)
            if (not isinstance(receipt, dict)
                    or type(receipt.get("exit_code")) is not int
                    or type(receipt.get("timed_out")) is not bool
                    or type(receipt.get("truncated")) is not bool
                    or not isinstance(receipt.get("output_b64"), str)):
                raise ValueError("invalid supervisor receipt")
            output = base64.b64decode(receipt["output_b64"], validate=True)
            if len(output) > limit:
                raise ValueError("supervisor output exceeds limit")
        except (ValueError, TypeError) as exc:
            raise SandboxInfrastructureError("bash supervisor receipt unavailable") from exc
        text = output.decode("utf-8", "replace")
        if receipt["truncated"]:
            text += "\n[pipeline: output truncated]"
        return {
            "syntax_check": syntax_check,
            "exit_code": receipt["exit_code"],
            "timed_out": receipt["timed_out"],
            "output": text,
            "truncated": receipt["truncated"],
            "duration_ms": round((time.monotonic() - started) * 1000),
        }

    def snapshot_submission(self, task: Task, dest: Path) -> dict[str, Any]:
        """Copy from a stopped container, preserving links for validation."""
        if not self.stopped:
            raise SandboxError("submission snapshot requires a stopped container")
        src = task.submission_dir
        result: dict[str, Any] = {
            "present": False,
            "container_path": src,
            "files": [],
            "violations": [],
            "health": "ok",
        }
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="submission-copy-", dir=dest.parent) as staging:
            try:
                # No '/.' or '-L': even a symlink at the submission root stays a link.
                docker.copy_from_container(self.container, src, Path(staging), timeout=120)
            except docker.DockerError as exc:
                if f"Could not find the file {src} in container" in str(exc):
                    return result
                result["health"] = "error"
                result["violations"].append({"kind": "copy_failed", "detail": str(exc)})
                return result
            copied = Path(staging) / Path(src).name
            result["present"] = True
            if copied.is_symlink() or not copied.is_dir():
                result["violations"].append({"kind": "not_a_regular_file", "path": ".",
                                             "type": "invalid_submission_root"})
                return result
            # Inspect the stopped-container copy without following links. Keep
            # violations as evidence, but never promote unsafe entries into the
            # saved submission (nor silently make an invalid replay valid).
            manifest = tree_manifest(copied)
            for entry in manifest:
                if entry["type"] != "file":
                    result["violations"].append({
                        "kind": "not_a_regular_file", "path": entry["path"],
                        "type": entry["type"],
                    })
                    (copied / entry["path"]).unlink()
            shutil.move(str(copied), dest)
        manifest = [entry for entry in manifest if entry["type"] == "file"]
        total = sum(entry.get("bytes", 0) for entry in manifest)
        if total > self.cfg.submission_limit_bytes:
            result["violations"].append(
                {"kind": "submission_too_large", "bytes": total,
                 "limit": self.cfg.submission_limit_bytes}
            )
        result["files"] = manifest
        result["bytes"] = total
        return result
