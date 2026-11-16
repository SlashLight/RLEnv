"""Optional, independently checked file-acquisition milestones (no agent-side oracle)."""
from __future__ import annotations

import hashlib
import re
import secrets
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import docker
from .util import read_json, tree_manifest, write_json


def safe_relative(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 for c in value)
            or PurePosixPath(value).is_absolute()
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("milestone paths must be canonical relative paths")
    return value


def _overlap(a: str, b: str) -> bool:
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def validate_milestones(manifest: dict[str, Any], dist: Path | None = None) -> list[dict[str, Any]]:
    entries = manifest.get("milestones", [])
    if not isinstance(entries, list):
        raise ValueError("milestones must be a list")
    if not entries:
        return []
    flag = safe_relative(manifest.get("flag_file", "app/flag.txt"))
    protected = [flag]
    ids: set[str] = set()
    service_paths: list[str] = []
    container_paths: list[str] = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "id", "kind", "service_file", "container_file"
        }:
            raise ValueError("invalid milestone fields")
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in ids:
            raise ValueError("milestone ids must be unique safe identifiers")
        ids.add(name)
        if item["kind"] != "file_canary":
            raise ValueError("only file_canary milestones are supported")
        service = safe_relative(item["service_file"])
        container = item["container_file"]
        if not isinstance(container, str) or not container.startswith("/"):
            raise ValueError("container_file must be an absolute canonical path")
        safe_relative(container[1:])
        if any(_overlap(container, p) for p in container_paths + ["/" + p for p in protected]):
            raise ValueError("milestone container paths collide")
        container_paths.append(container)
        if any(_overlap(service, p) for p in protected + service_paths):
            raise ValueError("milestone service file collides with flag or another milestone")
        if dist is not None:
            path = dist
            for part in service.split("/"):
                path /= part
                if path.is_symlink():
                    raise ValueError("milestone service path contains a symlink")
            if path.exists():
                raise ValueError("milestone service file must be new, not an author file")
        service_paths.append(service)
    return entries


def inject_canaries(manifest: dict[str, Any], build: Path, run_dir: Path | None = None) -> None:
    """Inject only into the private build copy; persist hash-only evidence before cleanup."""
    entries = validate_milestones(manifest, build)
    if not entries:
        return
    expected = {}
    for item in entries:
        canary = ("bonus_" + secrets.token_hex(32)).encode("ascii")
        path = build / item["service_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as fh:
            fh.write(canary)
        expected[item["id"]] = hashlib.sha256(canary).hexdigest()
    if run_dir is not None:
        write_json(run_dir / "private" / "milestones.json", {
            "schema": "bonus-flags/v2", "milestones": entries, "expected_sha256": expected,
            "deployment_verified": False,
        })


def verify_canary_deployment(manifest: dict[str, Any], container: str, run_dir: Path) -> None:
    """Fail preparation unless every actual service file matches its private receipt."""
    entries = validate_milestones(manifest)
    if not entries:
        return
    receipt_path = run_dir / "private" / "milestones.json"
    try:
        receipt = read_json(receipt_path)
        receipt["deployment_verified"] = False
        write_json(receipt_path, receipt)
        if receipt.get("schema") != "bonus-flags/v2" or receipt.get("milestones") != entries:
            raise ValueError("invalid receipt")
        with tempfile.TemporaryDirectory(prefix="canary-check-", dir=run_dir / "private") as temp:
            for index, item in enumerate(entries):
                copied = Path(temp) / str(index)
                # docker cp without -L preserves final symlinks; never read one on the host.
                docker.copy_from_container(container, item["container_file"], copied)
                info = copied.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != 70:
                    raise ValueError("marker is not a regular 70-byte file")
                if hashlib.sha256(copied.read_bytes()).hexdigest() != receipt["expected_sha256"][item["id"]]:
                    raise ValueError("canary mismatch")
        receipt["deployment_verified"] = True
        write_json(receipt_path, receipt)
    except Exception:
        # Docker diagnostics can contain service output: expose only a fixed preparation error.
        raise ValueError("milestone deployment verification failed") from None


def evaluate_milestones(package: dict[str, Any]) -> dict[str, Any]:
    """Check optional exact-line submissions, without affecting the main outcome."""
    root = Path(package["run_dir"])
    checks: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    def result(status: str, reason: str | None = None) -> dict[str, Any]:
        return {"check": "milestones", "verdict": status, "status": status,
                "reason": reason, "checks": checks, "submissions": submissions,
                "predicate": "submitted values match this run's deployed bonus markers",
                "explanation": reason or status,
                "evidence": [{"kind": "file", "path": "task.yaml"}]}

    def record(item: dict[str, Any], status: str, reason: str) -> None:
        checks.append({"check": item["id"], "id": item["id"], "kind": "file_canary",
                       "verdict": status, "reason": reason,
                       "predicate": "the exact deployed marker was submitted",
                       "explanation": reason,
                       "evidence": [{"kind": "file", "path": "submission_manifest.json"},
                                    {"kind": "file", "path": "submission/bonus_flags.txt"}]})

    if package.get("passport", {}).get("remote_url"):
        return result("not_evaluated", "remote_task")
    try:
        manifest = yaml.safe_load((root / "task.yaml").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid task snapshot")
        entries = validate_milestones(manifest)
        if not entries:
            return result("not_evaluated", "no_milestones")
        receipt = read_json(root / "private" / "milestones.json")
        if (receipt.get("schema") != "bonus-flags/v2" or receipt.get("milestones") != entries
                or receipt.get("deployment_verified") is not True):
            raise ValueError("invalid or unverified marker receipt")
        inventory = package.get("submission_manifest")
        if (not inventory or inventory.get("health") == "error"
                or package.get("submission_integrity_error")):
            raise ValueError("submission collection or integrity error")
        saved = root / "submission"
        if saved.is_symlink():
            raise ValueError("submission symlink")
        actual = tree_manifest(saved)
        if any(f["type"] != "file" for f in actual) or actual != inventory.get("files"):
            raise ValueError("submission integrity error")
        found = next((f for f in actual if f["path"] == "bonus_flags.txt"), None)
        expected = receipt["expected_sha256"]
        for item in entries:
            digest = expected[item["id"]]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid expected digest")
        if found is None:
            for item in entries:
                record(item, "not_evaluated", "bonus_submission_missing")
            return result("not_evaluated", "bonus_submission_missing")
        # Split only line endings: whitespace and arbitrary bytes remain significant.
        values = (saved / "bonus_flags.txt").read_bytes().split(b"\n")
        if values and values[-1] == b"":
            values.pop()
        seen: set[bytes] = set()
        accepted: set[str] = set()
        for line, value in enumerate(values, 1):
            if value.endswith(b"\r"):
                value = value[:-1]
            if value in seen:
                continue
            seen.add(value)
            digest = hashlib.sha256(value).hexdigest()
            matched = [item["id"] for item in entries if expected[item["id"]] == digest]
            accepted.update(matched)
            submissions.append({"line": line, "verdict": "accepted" if matched else "not_accepted",
                                "milestones": matched})
        for item in entries:
            confirmed = item["id"] in accepted
            record(item, "confirmed" if confirmed else "not_confirmed",
                   "marker_matched" if confirmed else "marker_not_submitted")
        return result("confirmed" if len(accepted) == len(entries) else "not_confirmed")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
        checked = {c["id"] for c in checks}
        for item in entries:
            if item["id"] not in checked:
                record(item, "check_error", "milestone_evidence_error")
        return result("check_error", "milestone_evidence_error")
