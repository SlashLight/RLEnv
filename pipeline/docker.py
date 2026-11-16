"""Thin Docker control plane for task services and the one-shot tool container."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterable, cast

from .config import Config
from .util import ProcessResult, run_bounded


class DockerError(RuntimeError):
    pass


_deadline: ContextVar[float | None] = ContextVar("docker_deadline", default=None)


@contextmanager
def deadline(expires_at: float):
    """Bound every Docker subprocess in this preparation stage by one budget."""
    token = _deadline.set(expires_at)
    try:
        yield
    finally:
        _deadline.reset(token)


def docker(*args: str, input_data: bytes | None = None, timeout: float | None = None,
           check: bool = True, output_limit: int | None = None
           ) -> subprocess.CompletedProcess[bytes] | ProcessResult:
    expires_at = _deadline.get()
    if expires_at is not None:
        remaining = expires_at - time.monotonic()
        if remaining <= 0:
            raise DockerError("preparation deadline exceeded")
        timeout = min(timeout, remaining) if timeout is not None else remaining
    try:
        if output_limit is not None:
            if timeout is None:
                raise ValueError("bounded Docker output requires an explicit timeout")
            proc = run_bounded(["docker", *args], input_data=input_data,
                               timeout=timeout, output_limit=output_limit)
        else:
            proc = subprocess.run(
                ["docker", *args], input=input_data,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
    except FileNotFoundError as exc:
        raise DockerError("docker executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"docker {args[0]} timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        raise DockerError(
            f"docker {' '.join(args[:2])} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:2000]}"
        )
    return proc


def image_id(image: str) -> str | None:
    proc = docker("image", "inspect", image, "--format", "{{.Id}}", check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip()


def ensure_tools_image(cfg: Config, pipeline_dir: Path) -> str:
    existing = image_id(cfg.docker.tools_image)
    if existing:
        return existing
    dockerfile = pipeline_dir / "tools.Dockerfile"
    if not dockerfile.exists():
        raise DockerError(f"tools image missing and {dockerfile} not found")
    docker("build", "-q", "-t", cfg.docker.tools_image, "-f", str(dockerfile), str(pipeline_dir),
           timeout=900)
    built = image_id(cfg.docker.tools_image)
    if not built:
        raise DockerError("tools image build produced no image")
    return built


def build_task_image(context_dir: Path, tag: str, timeout: float = 600) -> str:
    docker("build", "-q", "-t", tag, str(context_dir), timeout=timeout)
    built = image_id(tag)
    if not built:
        raise DockerError("task image build produced no image")
    return built


def create_network(name: str, *, isolated: bool = False) -> None:
    # Isolated gateway mode gives the internal bridge no host-side gateway address.
    options = ("--opt", "com.docker.network.bridge.gateway_mode_ipv4=isolated") if isolated else ()
    proc = docker("network", "create", "--driver", "bridge", "--internal", *options, name, check=False)
    if proc.returncode != 0 and b"already exists" not in proc.stderr:
        raise DockerError(proc.stderr.decode("utf-8", "replace").strip())


def _remove(args: tuple[str, ...], absent_messages: tuple[bytes, ...]) -> None:
    proc = docker(*args, check=False, timeout=30)
    if proc.returncode and not any(message in proc.stderr.lower() for message in absent_messages):
        raise DockerError(f"docker {' '.join(args)} failed: "
                          f"{proc.stderr.decode('utf-8', 'replace').strip()[:2000]}")


def remove_network(name: str) -> None:
    _remove(("network", "rm", name),
            (b"no such network", f"network {name} not found".encode()))


def remove_image(name: str) -> None:
    _remove(("rmi", "-f", name), (b"no such image",))


def start_service(image: str, network: str, name: str, hostname: str, cfg: Config) -> str:
    remove_container(name)
    proc = docker(
        "run", "-d", "--name", name, "--network", network, "--hostname", hostname,
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--pids-limit", "128",
        "--memory", cfg.docker.task_memory,
        "--cpus", str(cfg.docker.task_cpus),
        image,
        timeout=60,
    )
    return proc.stdout.decode().strip()


def start_tools_container(image: str, network: str, name: str, workspace: Path, cfg: Config) -> str:
    workspace = Path(workspace).resolve()
    remove_container(name)
    proc = docker(
        "run", "-d", "--name", name, "--network", network, "--hostname", "agent-tools",
        "--read-only",
        "--tmpfs", "/tmp:rw,exec,nosuid,size=128m,mode=1777",
        "-v", f"{workspace}:/workspace:rw",
        "-w", "/workspace",
        "-e", "HOME=/tmp",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--pids-limit", "256",
        "--memory", cfg.docker.tools_memory,
        "--cpus", "1",
        "--entrypoint", "sleep",
        image,
        "infinity",
        timeout=60,
    )
    return proc.stdout.decode().strip()


def remove_container(name: str) -> None:
    _remove(("rm", "-f", name), (b"no such container",))


def stop_container(name: str) -> None:
    docker("stop", "--time", "1", name, timeout=30)


def exec_in(container: str, argv: Iterable[str], timeout: float,
            workdir: str | None = None, input_data: bytes | None = None,
            output_limit: int = 262144) -> ProcessResult:
    args = ["exec", "-i"]
    if workdir:
        args += ["-w", workdir]
    args += [container, *argv]
    return cast(ProcessResult, docker(*args, input_data=input_data, timeout=timeout, check=False,
                                     output_limit=output_limit))


def check_ready(network: str, url: str, expected_status: int, tools_image: str,
                timeout: float) -> None:
    code = (
        "import sys,urllib.request\n"
        f"u=urllib.request.urlopen(sys.argv[1], timeout={max(1.0, timeout / 2)})\n"
        f"sys.exit(0 if u.status == {int(expected_status)} else 1)\n"
    )
    name = f"pipeline-readiness-{uuid.uuid4().hex[:12]}"
    try:
        proc = docker(
            "run", "--rm", "--name", name, "--network", network,
            "--entrypoint", "python", tools_image,
            "-c", code, url, timeout=timeout, check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).decode("utf-8", "replace").strip()[:1000]
            raise DockerError(f"readiness check failed for {url}: {detail or 'no output'}")
    finally:
        # Killing the Docker CLI on deadline does not kill its container.
        # Cleanup has its own bounded timeout, even after preparation expired.
        token = _deadline.set(None)
        try:
            remove_container(name)
        finally:
            _deadline.reset(token)


def copy_from_container(container: str, src: str, dest: Path, timeout: float = 60) -> None:
    docker("cp", f"{container}:{src}", str(dest), timeout=timeout)



