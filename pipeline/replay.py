"""Fresh-instance replay: a new flag, a new service container, no model."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from . import docker
from .config import Config
from .evaluate import ReplayResult
from .task import Task, prepare_build_context
from .util import read_json, run_bounded


_SYNTAX_PREFIX = "pipeline-replay-v2:"
_SYNTAX_HEADER_LIMIT = 4096
# The script never owns the supervisor stdout descriptor, even through /proc.
_PYTHON_SYNTAX_AND_RUN = r'''
import base64, ctypes, hashlib, json, os, selectors, signal, subprocess, sys, time
if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
    raise OSError("cannot protect supervisor descriptors")
url, budget, limit = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
check = {"language": "python", "parser": "compile", "path": "solution.py"}
try:
    with open("solution.py", "rb") as source:
        data = source.read()
    check["source_sha256"] = hashlib.sha256(data).hexdigest()
    compile(data, "solution.py", "exec", dont_inherit=True)
    check["verdict"] = "valid"
except SyntaxError:
    check.update(verdict="invalid", reason="syntax_error")
except (OSError, ValueError, MemoryError, RecursionError):
    check.update(verdict="check_error", reason="parser_failed")
proc = subprocess.Popen([sys.executable, "solution.py", url], stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
end = time.monotonic() + budget
buffers = {"stdout": bytearray(), "stderr": bytearray()}
cut = {"stdout": False, "stderr": False}
timed_out = False
def kill_group():
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
with selectors.DefaultSelector() as selector:
    for name in buffers:
        pipe = getattr(proc, name)
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    while selector.get_map() or proc.poll() is None:
        if time.monotonic() >= end and not timed_out:
            timed_out = True
            kill_group()
        if timed_out and time.monotonic() >= end + 1:
            for key in selector.get_map().values():
                cut[key.data] = True
            break
        for key, _ in selector.select(0.05):
            chunk = os.read(key.fd, 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            room = limit - len(buffers[key.data])
            buffers[key.data].extend(chunk[:room])
            cut[key.data] |= len(chunk) > room
kill_group()
code = proc.wait(timeout=1)
print("pipeline-replay-v2:" + json.dumps({
    "exit_code": code if code >= 0 else 128 - code, "timed_out": timed_out,
    "syntax_check": check,
    **{name + "_b64": base64.b64encode(value).decode("ascii") for name, value in buffers.items()},
    **{name + "_truncated": value for name, value in cut.items()},
}))
'''


class DockerReplayEnvironment:
    def __init__(self, cfg: Config, task: Task, run_dir: Path):
        self.cfg = cfg
        self.task = task
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.attempt_id = uuid.uuid4().hex[:16]
        prefix = f"pipeline-{self.run_id.lower()}-replay-{self.attempt_id}"
        self.network = prefix
        self.service_name = prefix + "-task"
        self.run_name = prefix + "-run"
        self.build_dir = self.run_dir / "private" / ("replay-context-" + self.attempt_id)
        self.submission_copy = self.run_dir / "private" / ("replay-submission-" + self.attempt_id)
        self.image_tag = f"pipeline-task-{self.run_id.lower()}-replay-{self.attempt_id}"
        self.image_id: str | None = None
        self.tools_image = ""
        self.url = ""

    def prepare(self, flag: str) -> str:
        with docker.deadline(time.monotonic() + self.cfg.limits.prepare_seconds):
            return self._prepare(flag)

    def _prepare(self, flag: str) -> str:
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.parent.mkdir(parents=True, exist_ok=True)
        prepare_build_context(self.task, flag, self.build_dir)
        self.image_id = docker.build_task_image(self.build_dir, self.image_tag)
        passport = read_json(self.run_dir / "passport.json")
        pinned = passport.get("environment", {}).get("tools_image_id")
        if not isinstance(pinned, str) or not pinned.startswith("sha256:"):
            raise ValueError("Exploration tools image has no immutable sha256 identifier.")
        if docker.image_id(pinned) != pinned:
            raise docker.DockerError("Pinned exploration tools image is unavailable.")
        self.tools_image = pinned
        docker.create_network(self.network, isolated=True)
        docker.start_service(
            self.image_id, self.network, self.service_name,
            str(self.task.service["hostname"]), self.cfg,
        )
        port = int(self.task.service.get("port", 5000))
        readiness = self.task.service.get("readiness", {})
        path = readiness.get("path", "/")
        status = int(readiness.get("status", 200))
        self.url = f"http://{self.task.service['hostname']}:{port}"
        docker.check_ready(
            self.network, self.url + path, status, self.tools_image,
            self.cfg.docker.service_start_seconds,
        )
        return self.url

    def run(self, submission_dir: Path, url: str, timeout: float) -> ReplayResult:
        if self.submission_copy.exists():
            shutil.rmtree(self.submission_copy)
        self.submission_copy.mkdir(parents=True)
        source = submission_dir / "solution.py"
        if submission_dir.is_symlink() or source.is_symlink() or not source.is_file():
            raise ValueError("Replay requires a regular solution.py without links.")
        if source.stat().st_nlink != 1:
            raise ValueError("Replay refuses hard-linked scripts.")
        shutil.copyfile(source, self.submission_copy / "solution.py")
        command = [
            "docker", "run", "--name", self.run_name,
            "--network", self.network,
            "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=64m,mode=1777",
            "-v", f"{self.submission_copy.resolve()}:/submission:ro",
            "-w", "/submission",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "128",
            "--memory", self.cfg.docker.tools_memory,
            "--cpus", "1",
            "--entrypoint", "/usr/local/bin/python",
            self.tools_image, "-I", "-S", "-c", _PYTHON_SYNTAX_AND_RUN,
            url, str(timeout), str(self.cfg.tool_output_limit_bytes),
        ]
        started = time.monotonic()
        limit = self.cfg.tool_output_limit_bytes
        stdout = stderr = ""
        state = {}
        receipt = None
        error = None
        transport_truncated = False
        try:
            proc = run_bounded(command, timeout=timeout + 20,
                               output_limit=4 * limit + _SYNTAX_HEADER_LIMIT)
            transport_truncated = proc.stdout_truncated
            stderr = proc.stderr.decode("utf-8", "replace")
            wire = proc.stdout.decode("utf-8", "strict")
            stdout = wire
            if wire.startswith(_SYNTAX_PREFIX) and not transport_truncated:
                receipt = json.loads(wire[len(_SYNTAX_PREFIX):])
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", "replace")
            stderr = (exc.stderr or b"").decode("utf-8", "replace")
            error = "Docker watchdog expired; no script timeout is proven."
        except (ValueError, UnicodeError):
            error = "Replay supervisor receipt is invalid."
        try:
            inspected = docker.docker("inspect", "--format", "{{json .State}}",
                                      self.run_name, timeout=15)
            task_inspected = docker.docker("inspect", "--format", "{{json .State}}",
                                           self.service_name, timeout=15)
        except docker.DockerError as exc:
            return ReplayResult(None, stdout, stderr,
                                round((time.monotonic() - started) * 1000), command,
                                infrastructure_error=True, check_error=str(exc),
                                environment={"tools_image": self.tools_image,
                                             "resource_attempt_id": self.attempt_id})
        try:
            state = json.loads(inspected.stdout)
            task_state = json.loads(task_inspected.stdout)
            if not isinstance(state, dict) or not isinstance(task_state, dict):
                raise ValueError("container state must be an object")
        except (ValueError, TypeError):
            raise docker.DockerError("Docker returned an invalid container state.") from None
        environment = {
            "fresh_instance": True, "new_flag": "configured, not recorded",
            "task_image_id": self.image_id, "tools_image": self.tools_image,
            "network": self.network, "container_state": state,
            "task_container_state": task_state,
            "resource_attempt_id": self.attempt_id,
        }
        common: dict[str, Any] = dict(duration_ms=round((time.monotonic() - started) * 1000),
                                      command=command, environment=environment)
        if state.get("Error"):
            return ReplayResult(None, stdout, stderr, **common, infrastructure_error=True,
                                check_error="Docker confirmed a container runtime failure.")
        if state.get("OOMKilled") is True:
            return ReplayResult(state.get("ExitCode"), stdout, stderr, **common, oom_killed=True)
        if state.get("Status") != "exited":
            error = error or "Replay container did not reach a known completed state."
        if state.get("ExitCode") != 0:
            error = error or "Replay supervisor exited abnormally; child failure is not proven."
        try:
            if error or not isinstance(receipt, dict):
                raise ValueError(error or "Replay supervisor receipt is missing.")
            if (type(receipt.get("exit_code")) is not int
                    or type(receipt.get("timed_out")) is not bool
                    or any(type(receipt.get(name + "_truncated")) is not bool
                           for name in ("stdout", "stderr"))
                    or not isinstance(receipt.get("syntax_check"), dict)):
                raise ValueError("Replay supervisor receipt has invalid fields.")
            out_bytes = base64.b64decode(receipt["stdout_b64"], validate=True)
            err_bytes = base64.b64decode(receipt["stderr_b64"], validate=True)
            if len(out_bytes) > limit or len(err_bytes) > limit:
                raise ValueError("Replay supervisor exceeded its output bound.")
            stdout = out_bytes.decode("utf-8", "replace")
            stderr = err_bytes.decode("utf-8", "replace")
        except (ValueError, KeyError, TypeError) as exc:
            return ReplayResult(None, stdout, stderr, **common, check_error=str(exc))
        return ReplayResult(
            receipt["exit_code"], stdout, stderr, **common,
            timed_out=receipt["timed_out"], stdout_truncated=receipt["stdout_truncated"],
            stderr_truncated=receipt["stderr_truncated"], syntax_check=receipt["syntax_check"],
        )

    def close(self) -> list[str]:
        errors: list[str] = []
        try:
            docker.remove_container(self.run_name)
        except docker.DockerError as exc:
            errors.append(f"replay container: {exc}")
        try:
            docker.remove_container(self.service_name)
        except docker.DockerError as exc:
            errors.append(f"replay service: {exc}")
        try:
            docker.remove_network(self.network)
        except docker.DockerError as exc:
            errors.append(f"replay network: {exc}")
        try:
            docker.remove_image(self.image_tag)
        except docker.DockerError as exc:
            errors.append(f"replay image: {exc}")
        for path in (self.build_dir, self.submission_copy):
            try:
                if path.exists():
                    shutil.rmtree(path)
            except OSError as exc:
                errors.append(f"cleanup {path.name}: {exc}")
        return errors
