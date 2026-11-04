"""Versioned task manifest: author materials, provenance, service definition."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import copy_tree, sha256_file
from .milestones import inject_canaries, validate_milestones


class TaskError(RuntimeError):
    pass


@dataclass
class Task:
    id: str
    path: Path
    manifest: dict[str, Any]

    @property
    def dist_dir(self) -> Path:
        return self.path / self.manifest.get("dist", "dist")

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    @property
    def agent_files(self) -> list[str]:
        return list(self.manifest.get("agent_files", []))

    @property
    def flag_file(self) -> str:
        return self.manifest.get("flag_file", "app/flag.txt")

    @property
    def submission_dir(self) -> str:
        return self.manifest.get("submission_dir", "/workspace/submission")

    @property
    def service(self) -> dict[str, Any]:
        return self.manifest["service"]

    def verify_materials(self) -> None:
        """Fail closed if the pinned author files changed on disk."""
        expected = self.manifest.get("files_sha256", {})
        for rel, digest in expected.items():
            path = self.dist_dir / rel
            if not path.is_file():
                raise TaskError(f"task material missing: {rel}")
            if sha256_file(path) != digest:
                raise TaskError(f"task material changed: {rel}")

    def prompt_text(self) -> str:
        return str(self.manifest.get("prompt", self.manifest.get("description", ""))).strip()


def load_task(task_id: str, tasks_dir: Path) -> Task:
    path = Path(tasks_dir) / task_id
    manifest_path = path / "task.yaml"
    if not manifest_path.exists():
        raise TaskError(f"unknown task {task_id!r}: {manifest_path} not found")
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    if not isinstance(manifest, dict) or manifest.get("id") != task_id:
        raise TaskError(f"invalid task manifest: {manifest_path}")
    task = Task(id=task_id, path=path, manifest=manifest)
    task.verify_materials()
    try:
        validate_milestones(manifest, task.dist_dir)
    except ValueError as exc:
        raise TaskError(str(exc)) from exc
    return task


def generate_flag(prefix: str = "SEKAI") -> str:
    return f"{prefix}{{{secrets.token_hex(16)}}}"


def prepare_agent_workspace(task: Task, workspace: Path) -> None:
    """Copy the author distribution the agent is allowed to see."""
    import shutil

    workspace.mkdir(parents=True, exist_ok=True)
    for rel in task.agent_files:
        src = task.dist_dir / rel
        if src.is_dir():
            copy_tree(src, workspace / rel)
        elif src.is_file() and not src.is_symlink():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        else:
            raise TaskError(f"agent material missing or not a regular file/dir: {rel}")
    (workspace / "submission").mkdir(exist_ok=True)


def prepare_build_context(task: Task, flag: str, dest: Path,
                          milestone_run_dir: Path | None = None) -> Path:
    """Copy the author dist and replace the placeholder with this run's flag."""
    copy_tree(task.dist_dir, dest)
    flag_path = dest / task.flag_file
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(flag, encoding="utf-8")
    inject_canaries(task.manifest, dest, milestone_run_dir)
    return dest
