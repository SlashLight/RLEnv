import json
import shutil
from pathlib import Path

import pytest

from pipeline.milestones import evaluate_milestones, validate_milestones, verify_canary_deployment
from pipeline.task import Task, load_task, prepare_agent_workspace, prepare_build_context
from pipeline.util import tree_manifest, write_json


def definition():
    return {"id": "private-file", "kind": "file_canary", "service_file": "secrets/probe.txt",
            "container_file": "/opt/secrets/probe.txt"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    dist = tmp_path / "task" / "dist"
    dist.mkdir(parents=True)
    (dist / "app.py").write_text("print('public')")
    manifest = {"agent_files": ["app.py"], "flag_file": "flag.txt",
                "milestones": [definition(), {**definition(), "id": "second",
                "service_file": "secrets/second.txt", "container_file": "/opt/secrets/second.txt"}]}
    task = Task("test", dist.parent, manifest)
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "task.yaml", manifest)
    build = run / "private" / "build-context"
    prepare_build_context(task, "FLAG", build, milestone_run_dir=run)

    def copy(container, source, dest):
        shutil.copyfile(build / "secrets" / Path(source).name, dest)

    monkeypatch.setattr("pipeline.milestones.docker.copy_from_container", copy)
    verify_canary_deployment(manifest, "task-service", run)
    (run / "submission").mkdir()
    return task, build, {"run_dir": run, "passport": {}}


def collect(package):
    package["submission_manifest"] = {"health": "ok", "present": True,
                                      "files": tree_manifest(package["run_dir"] / "submission")}


def test_mixed_values_exact_deduplicated_and_fresh(prepared, tmp_path):
    task, build, package = prepared
    run = package["run_dir"]
    marker = (build / "secrets/probe.txt").read_bytes()
    assert marker.startswith(b"bonus_") and len(marker) == 70
    prepare_agent_workspace(task, run / "workspace")
    assert not (run / "workspace/secrets").exists()
    assert not (task.dist_dir / "secrets").exists()
    second = tmp_path / "second"
    prepare_build_context(task, "FLAG2", second)
    old = (second / "secrets/probe.txt").read_bytes()
    assert old != marker
    receipt = run / "private/milestones.json"
    assert marker not in receipt.read_bytes()
    assert receipt.stat().st_mode & 0o777 == 0o600
    (run / "submission/bonus_flags.txt").write_bytes(
        old + b"\nwrong\n" + marker + b"\r\n" + marker + b"\n " + marker + b"\n")
    collect(package)
    shutil.rmtree(build)
    result = evaluate_milestones(package)
    assert [c["verdict"] for c in result["checks"]] == ["confirmed", "not_confirmed"]
    assert [s["verdict"] for s in result["submissions"]] == [
        "not_accepted", "not_accepted", "accepted", "not_accepted"]
    assert marker.decode() not in json.dumps(result)
    assert "points" not in json.dumps(result)
    assert evaluate_milestones(package) == result
    # Milestone evaluation consumes evidence, not the main flag/replay outcome.
    package["flag_check"] = {"verdict": "confirmed"}
    assert evaluate_milestones(package) == result
    assert package["flag_check"] == {"verdict": "confirmed"}


def test_all_values_in_any_order(prepared):
    _, build, package = prepared
    values = [(build / "secrets" / name).read_bytes() for name in ("second.txt", "probe.txt")]
    (package["run_dir"] / "submission/bonus_flags.txt").write_bytes(b"\n".join(values))
    collect(package)
    assert evaluate_milestones(package)["status"] == "confirmed"


def test_missing_submission_and_no_checks_are_distinct(prepared, tmp_path):
    _, _, package = prepared
    collect(package)
    result = evaluate_milestones(package)
    assert result["reason"] == "bonus_submission_missing"
    assert all(c["verdict"] == "not_evaluated" for c in result["checks"])
    write_json(tmp_path / "task.yaml", {})
    result = evaluate_milestones({"run_dir": tmp_path})
    assert result["reason"] == "no_milestones" and result["checks"] == []
    assert evaluate_milestones({"run_dir": tmp_path, "passport": {"remote_url": "url"}})[
        "reason"] == "remote_task"


@pytest.mark.parametrize("failure", ["receipt", "inventory", "changed", "link"])
def test_technical_errors_per_subtask(prepared, failure):
    _, _, package = prepared
    root = package["run_dir"]
    collect(package)
    if failure == "receipt":
        (root / "private/milestones.json").unlink()
    elif failure == "inventory":
        package["submission_manifest"]["health"] = "error"
    elif failure == "changed":
        (root / "submission/bonus_flags.txt").write_text("unknown")
    else:
        (root / "submission/bonus_flags.txt").symlink_to("/etc/passwd")
        collect(package)
    result = evaluate_milestones(package)
    assert result["status"] == "check_error"
    assert all(c["verdict"] == "check_error" for c in result["checks"])


@pytest.mark.parametrize("failure", ["wrong", "missing", "link", "directory", "docker_error"])
def test_deployment_failure_is_invalid_and_private(prepared, monkeypatch, failure):
    task, build, package = prepared

    def copy(container, source, dest):
        if failure == "wrong":
            dest.write_text("0" * 70)
        elif failure == "link":
            dest.symlink_to(build / "secrets/probe.txt")
        elif failure == "directory":
            dest.mkdir()
        elif failure == "docker_error":
            raise RuntimeError((build / "secrets/probe.txt").read_text())

    monkeypatch.setattr("pipeline.milestones.docker.copy_from_container", copy)
    with pytest.raises(ValueError, match="^milestone deployment verification failed$"):
        verify_canary_deployment(task.manifest, "service", package["run_dir"])
    receipt = json.loads((package["run_dir"] / "private/milestones.json").read_text())
    assert receipt["deployment_verified"] is False
    collect(package)
    assert evaluate_milestones(package)["status"] == "check_error"
    assert not list((package["run_dir"] / "private").glob("canary-check-*"))


@pytest.mark.parametrize("verified", [False, None, "true", 1])
def test_requires_boolean_deployment_confirmation(prepared, verified):
    _, _, package = prepared
    receipt_path = package["run_dir"] / "private/milestones.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["deployment_verified"] = verified
    write_json(receipt_path, receipt)
    collect(package)
    result = evaluate_milestones(package)
    assert result["status"] == "check_error"
    assert all(c["verdict"] == "check_error" for c in result["checks"])


@pytest.mark.parametrize("field,value", [
    ("service_file", "../escape"), ("service_file", "/etc/secret"),
    ("service_file", "app.py"), ("service_file", "flag.txt"),
    ("weight", 0.3), ("submission_file", "proof.txt"),
    ("kind", "shell"), ("id", "../bad"),
    ("container_file", "relative"), ("container_file", "/"),
    ("container_file", "//etc/a"), ("container_file", "/a/../b"),
])
def test_invalid_config(tmp_path, field, value):
    (tmp_path / "app.py").write_text("author material")
    with pytest.raises(ValueError):
        validate_milestones({"flag_file": "flag.txt",
                             "milestones": [{**definition(), field: value}]}, tmp_path)


def test_existing_task_marker_available_through_static_route(tmp_path, monkeypatch):
    task = load_task("my-flask-app", Path(__file__).resolve().parents[1] / "tasks")
    build = prepare_build_context(task, "FLAG", tmp_path / "build", milestone_run_dir=tmp_path)
    workspace = tmp_path / "workspace"
    prepare_agent_workspace(task, workspace)
    marker = build / "app/static/diagnostics.txt"
    assert marker.read_bytes().startswith(b"bonus_")
    assert not (workspace / "app/static/diagnostics.txt").exists()
    # Flask serves this directory at /static; verify the same deployed path.
    def copy(container, source, dest):
        assert source == "/app/static/diagnostics.txt"
        shutil.copyfile(marker, dest)
    monkeypatch.setattr("pipeline.milestones.docker.copy_from_container", copy)
    verify_canary_deployment(task.manifest, "service", tmp_path)
    (tmp_path / "submission").mkdir()
    shutil.copyfile(marker, tmp_path / "submission/bonus_flags.txt")
    package = {"run_dir": tmp_path}
    write_json(tmp_path / "task.yaml", task.manifest)
    collect(package)
    assert evaluate_milestones(package)["status"] == "confirmed"
