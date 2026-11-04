"""Saved package -> reward -> Markdown/API payload, no model or Docker required."""
import json

import pytest

from pipeline.evaluate import ReplayResult, evaluate
from pipeline.milestones import inject_canaries, verify_canary_deployment
from pipeline.runlog import EventLog
from pipeline.runner import Runner
from pipeline.util import read_json, tree_manifest, write_json


@pytest.fixture
def package(tmp_path):
    run = tmp_path / "run"
    (run / "submission").mkdir(parents=True)
    write_json(run / "passport.json", {"run_id": "r", "task": {"id": "test", "version": "1"}})
    write_json(run / "status.json", {"status": "done", "stop_reason": {"kind": "agent_settled"}})
    write_json(run / "task.yaml", {})
    (run / "private").mkdir()
    (run / "private/expected_flag.txt").write_text("FLAG")
    (run / "submission/flag.txt").write_text("FLAG")
    EventLog(run, "r").close()
    return run


def inventory(run):
    write_json(run / "submission_manifest.json", {
        "health": "ok", "present": True, "violations": [],
        "files": tree_manifest(run / "submission"),
    })


class Replay:
    def prepare(self, flag):
        self.flag = flag
        return "http://test"

    def run(self, *args):
        return ReplayResult(0, self.flag, "", 1, ["python", "solution.py"],
                            syntax_check={"verdict": "valid"})

    def close(self):
        return []


def test_unprepared_reward_and_full_replay(package):
    inventory(package)
    first = evaluate(package)
    assert first["reward"]["value"] == .5
    assert first["reward"]["outcome"] == 0
    assert first["milestones"]["status"] == "not_evaluated"
    (package / "submission/solution.py").write_text("print('fresh')")
    inventory(package)
    report = evaluate(package, replay_env_factory=Replay, new_flag="FRESH")
    assert report["reward"]["value"] == report["reward"]["outcome"] == 1
    assert read_json(package / "checks/reward.json") == report["reward"]
    assert "Composite reward: **1.0**" in (package / "report.md").read_text()


def test_prepared_reward_and_saved_evidence(package, tmp_path, monkeypatch):
    spec = {"milestones": [{"id": "file", "kind": "file_canary", "weight": .3,
                            "service_file": "secret.txt", "container_file": "/secret.txt",
                            "submission_file": "proof.txt"}]}
    build = tmp_path / "build"
    build.mkdir()
    inject_canaries(spec, build, package)
    write_json(package / "task.yaml", spec)
    canary = (build / "secret.txt").read_bytes()
    monkeypatch.setattr("pipeline.milestones.docker.copy_from_container",
                        lambda container, source, dest: dest.write_bytes(canary))
    verify_canary_deployment(spec, "task-container", package)
    (package / "submission/proof.txt").write_bytes(canary)
    inventory(package)
    report = evaluate(package)
    assert report["reward"]["value"] == .8
    assert report["reward"]["components"]["milestones"] == .3
    assert canary.decode() not in json.dumps(report)
    assert evaluate(package)["reward"] == report["reward"]
    (package / "private/milestones.json").unlink()
    broken = evaluate(package)
    assert broken["reward"]["value"] is None
    assert broken["reward"]["outcome"] == 0


def test_trusted_bash_event_reaches_reward(package):
    log = EventLog(package, "r")
    for call_id in ("one", "two"):
        start = {"toolCallId": call_id, "toolName": "bash", "args": {"command": "if"}}
        raw = log.raw_record(start)
        log.publish("pi", "tool_execution_start", start, raw_line=raw)
        log.publish("sandbox", "tool_exec", {
            "op": "bash", "command": "if", "syntax_check": {"verdict": "invalid", "evidence_version": 3},
            "exit_code": 2, "timed_out": False})
        end = {"toolCallId": call_id, "isError": True, "result": {"content": []}}
        raw = log.raw_record(end)
        log.publish("pi", "tool_execution_end", end, raw_line=raw)
    log.close()
    inventory(package)
    report = evaluate(package)
    assert report["data_health"]["complete"]
    assert report["reward"]["value"] == .47
    assert [p["kind"] for p in report["reward"]["penalties"]] == [
        "bash_syntax", "exact_repeat", "bash_syntax"]


def test_late_cleanup_failure_invalidates_saved_reward(package):
    inventory(package)
    evaluate(package)
    runner = object.__new__(Runner)
    runner._refresh_report(package, {"status": "failed", "stage": "done",
                                     "error": "cleanup failed"})
    report = read_json(package / "report.json")
    assert report["reward"]["value"] is None
    assert report["reward"]["outcome"] is None
    assert read_json(package / "checks/reward.json") == report["reward"]


@pytest.mark.parametrize("infra", [False, True])
def test_gateway_records_infrastructure_classification(package, infra):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from unittest.mock import Mock
    from pipeline.docker import DockerError
    from pipeline.gateway import Gateway
    from pipeline.sandbox import SandboxError

    sandbox = Mock()
    sandbox.bash.side_effect = DockerError("daemon unavailable") if infra else SandboxError("permission denied")
    log = EventLog(package, "r")
    gateway = Gateway(sandbox, log, 1024)
    try:
        request = Request(gateway.url + "/tool/bash", data=b'{"command":"test"}',
                          headers={"X-Pipeline-Token": gateway.token})
        with pytest.raises(HTTPError):
            urlopen(request, timeout=2)
    finally:
        gateway.close()
        gateway.server.server_close()
        log.close()
    events = [json.loads(line) for line in (package / "events.jsonl").read_text().splitlines()]
    assert events[-1]["data"]["infrastructure_error"] is infra


def test_remote_has_no_full_solution_reward(package):
    write_json(package / "passport.json", {"remote_url": "http://example.test"})
    inventory(package)
    report = evaluate(package)
    assert report["reward"]["value"] is None
    assert report["reward"]["outcome"] is None
    assert report["flag_check"]["verdict"] == "confirmed"
