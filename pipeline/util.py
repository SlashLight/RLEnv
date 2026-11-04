"""Small filesystem/time helpers shared by the pipeline modules."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def run_bounded(command: list[str], *, timeout: float, output_limit: int,
                input_data: bytes | None = None) -> ProcessResult:
    """Drain both pipes, keeping only a bounded prefix of each in memory."""
    if timeout <= 0 or output_limit <= 0:
        raise ValueError("timeout and output_limit must be positive")
    expires = time.monotonic() + timeout
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    cut = {"stdout": False, "stderr": False}
    pending = memoryview(input_data or b"")
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True) as proc:
        with selectors.DefaultSelector() as selector:
            for name in buffers:
                pipe = getattr(proc, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            if pending:
                os.set_blocking(proc.stdin.fileno(), False)
                selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                proc.stdin.close()
            try:
                while selector.get_map():
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(remaining):
                        pipe, name = key.fileobj, key.data
                        if name == "stdin":
                            try:
                                pending = pending[os.write(pipe.fileno(), pending[:65536]):]
                            except BrokenPipeError:
                                pending = memoryview(b"")
                            if not pending:
                                selector.unregister(pipe)
                                pipe.close()
                            continue
                        chunk = os.read(pipe.fileno(), 65536)
                        if not chunk:
                            selector.unregister(pipe)
                            pipe.close()
                            continue
                        room = output_limit - len(buffers[name])
                        buffers[name].extend(chunk[:room])
                        cut[name] |= len(chunk) > room
                proc.wait(timeout=max(0, expires - time.monotonic()))
            except BaseException as exc:
                # Kill the local CLI and any children holding its pipes open.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    exc.output = bytes(buffers["stdout"])
                    exc.stderr = bytes(buffers["stderr"])
                    exc.stdout_truncated = cut["stdout"]
                    exc.stderr_truncated = cut["stderr"]
                raise
    return ProcessResult(proc.returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]),
                         cut["stdout"], cut["stderr"])


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file so readers never see a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    """Describe regular files under root without following symlinks."""
    root = Path(root)
    entries: list[dict[str, Any]] = []
    if not root.exists():
        return entries
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(dirnames):
            path = Path(dirpath) / name
            if path.is_symlink():
                entries.append({"path": str(path.relative_to(root)), "type": "symlink",
                                "target": os.readlink(path)})
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                entries.append({"path": rel, "type": "symlink", "target": os.readlink(path)})
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "type": "file", "bytes": path.stat().st_size,
                                "sha256": sha256_file(path)})
            else:
                entries.append({"path": rel, "type": "special"})
    return entries


def copy_tree(src: Path, dest: Path) -> None:
    """Copy regular files and directories, refusing links and special files."""
    import shutil

    src = Path(src)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        rel = Path(dirpath).relative_to(src)
        target = dest / rel
        target.mkdir(parents=True, exist_ok=True)
        for dirname in dirnames:
            path = Path(dirpath) / dirname
            if path.is_symlink():
                raise ValueError(f"refusing symlink in task materials: {path}")
        for filename in filenames:
            path = Path(dirpath) / filename
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f"refusing non-regular task material: {path}")
            shutil.copyfile(path, target / filename)


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[pipeline: output truncated at {limit} characters]", True
