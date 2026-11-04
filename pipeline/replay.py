"""Fresh-instance replay: a new flag, a new service container, no model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import docker
from .config import Config
from .evaluate import ReplayResult
from .task import Task, prepare_build_context
from .util import copy_tree, run_bounded


_SYNTAX_PREFIX = "pipeline-syntax-v1:"
_SYNTAX_HEADER_LIMIT = 1024
# -I -S prevents submission modules/sitecustomize from influencing the parser.
# Flush the receipt before exec: subsequent untrusted stdout cannot replace it.
_PYTHON_SYNTAX_AND_RUN = r'''
import hashlib, json, os, sys
check = {"language": "python", "parser": "compile", "python": sys.version.split()[0],
         "path": "solution.py", "scope": "final_submission"}
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
print("pipeline-syntax-v1:" + json.dumps(check), flush=True)
os.execv(sys.executable, [sys.executable, "solution.py", sys.argv[1]])
'''


class DockerReplayEnvironment:
    def __init__(self, cfg: Config, task: Task, run_dir: Path):
        self.cfg = cfg
        self.task = task
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.network = f"pipeline-{self.run_id.lower()}-replay"
        self.service_name = f"pipeline-{self.run_id.lower()}-replay-task"
        self.run_name = f"pipeline-{self.run_id.lower()}-replay-run"
        self.build_dir = self.run_dir / "private" / "replay-context"
        self.submission_copy = self.run_dir / "private" / "replay-submission"
        self.image_tag = f"pipeline-task-{self.run_id.lower()}-replay"
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
        self.tools_image = self.cfg.docker.tools_image
        docker.ensure_tools_image(self.cfg, Path(__file__).parent)
        docker.create_network(self.network)
        docker.start_service(
            self.image_tag, self.network, self.service_name,
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
        copy_tree(submission_dir, self.submission_copy)
        command = [
            "docker", "run", "--name", self.run_name,
            "--network", self.network,
            "--read-only", "--tmpfs", "/tmp:rw,exec,size=64m,mode=1777",
            "-v", f"{self.submission_copy.resolve()}:/submission:rw",
            "-w", "/submission",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "128",
            "--memory", self.cfg.docker.tools_memory,
            "--cpus", "1",
            "--entrypoint", "timeout",
            self.tools_image,
            "-k", "5s", f"{timeout:g}s",
            "/usr/local/bin/python", "-I", "-S", "-c", _PYTHON_SYNTAX_AND_RUN, url,
        ]
        started = time.monotonic()
        timed_out = False
        exit_code: int | None = None
        stdout_cut = stderr_cut = False
        try:
            proc = run_bounded(command, timeout=timeout + 20,
                               output_limit=self.cfg.tool_output_limit_bytes + _SYNTAX_HEADER_LIMIT)
            stdout_cut = proc.stdout_truncated
            stderr_cut = proc.stderr_truncated
            stdout = proc.stdout.decode("utf-8", "replace")
            stderr = proc.stderr.decode("utf-8", "replace")
            # A CLI exit code is not necessarily the script's exit code (e.g. 125).
            # Keep the container until inspection so even a script exiting 125 is
            # distinguishable from a failed Docker launch.
            inspected = docker.docker("inspect", "--format", "{{json .State}}",
                                      self.run_name, timeout=15)
            state = json.loads(inspected.stdout)
            if state.get("Status") != "exited" or state.get("Error") or state.get("OOMKilled"):
                raise docker.DockerError(f"replay container failed: {state.get('Error') or stderr or state}")
            exit_code = int(state["ExitCode"])
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout_cut = getattr(exc, "stdout_truncated", False)
            stderr_cut = getattr(exc, "stderr_truncated", False)
            stdout = (exc.stdout or b"").decode("utf-8", "replace")
            stderr = (exc.stderr or b"").decode("utf-8", "replace")
            docker.remove_container(self.run_name)
        syntax_check = {"verdict": "check_error", "language": "python",
                        "reason": "missing_parser_receipt"}
        header, separator, script_stdout = stdout.partition("\n")
        if separator and header.startswith(_SYNTAX_PREFIX):
            try:
                parsed = json.loads(header[len(_SYNTAX_PREFIX):])
                if (not isinstance(parsed, dict)
                        or parsed.get("verdict") not in ("valid", "invalid", "check_error")):
                    raise ValueError("invalid parser receipt")
                syntax_check = parsed
                stdout = script_stdout
            except (ValueError, TypeError):
                syntax_check["reason"] = "invalid_parser_receipt"
        # The extra capture space is exclusively for the trusted receipt.
        limit = self.cfg.tool_output_limit_bytes
        out_bytes, err_bytes = stdout.encode(), stderr.encode()
        stdout_cut |= len(out_bytes) > limit
        stderr_cut |= len(err_bytes) > limit
        stdout = out_bytes[:limit].decode("utf-8", "replace")
        stderr = err_bytes[:limit].decode("utf-8", "replace")
        duration = time.monotonic() - started
        if exit_code in (124, 137):
            timed_out = True
        return ReplayResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=round(duration * 1000),
            command=command,
            timed_out=timed_out,
            stdout_truncated=stdout_cut,
            stderr_truncated=stderr_cut,
            syntax_check=syntax_check,
            environment={
                "fresh_instance": True,
                "new_flag": "configured, not recorded",
                "task_image_id": self.image_id,
                "tools_image": self.tools_image,
                "network": self.network,
                "python": "3.11 (tools image)",
            },
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
