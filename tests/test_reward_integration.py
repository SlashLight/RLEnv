"""Saved local check contracts and unchanged remote reward payload."""
import json

import pytest

from pipeline.evaluate import ReplayResult, evaluate
from pipeline.runlog import EventLog
from pipeline.runner import Runner
from pipeline.util import read_json, tree_manifest, write_json


@pytest.fixture
def package(tmp_path):
    run = tmp_path / "run"
    (run / "submission").mkdir(parents=True)
    write_json(run / "passport.json", {"schema": "evaluator-pipeline/passport/v2",
                                        "run_id": "r", "task": {"id": "test", "version": "1"}})
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


def test_local_report_checks_replace_scores(package):
    inventory(package)
    first = evaluate(package)
    assert first["flag_check"]["verdict"] == "confirmed"
    assert first["script_check"]["reason"] == "script_missing"
    assert first["solution"]["solved"] is False
    assert first["milestones"]["status"] == "not_evaluated"
    (package / "submission/solution.py").write_text("print('fresh')")
    inventory(package)
    report = evaluate(package, replay_env_factory=Replay, new_flag="FRESH")
    assert report["solution"]["solved"] is True
    assert "reward" not in report and not (package / "checks/reward.json").exists()
    assert read_json(package / "checks/replay.json") == report["script_check"]
    assert "Composite reward" not in (package / "report.md").read_text()


def test_recording_failure_does_not_cancel_confirmed_checks(package):
    (package / "submission/solution.py").write_text("pass")
    inventory(package)
    (package / "events.jsonl").write_text("{broken\n")
    report = evaluate(package, replay_env_factory=Replay, new_flag="FRESH")
    assert report["data_health"]["complete"] is False
    assert report["solution"]["solved"] is True


def test_late_cleanup_failure_preserves_saved_check_verdicts(package):
    (package / "submission/solution.py").write_text("pass")
    inventory(package)
    evaluate(package, replay_env_factory=Replay, new_flag="FRESH")
    runner = object.__new__(Runner)
    runner._refresh_report(package, {"status": "done", "stage": "done",
                                     "cleanup_errors": ["network in use"]})
    report = read_json(package / "report.json")
    assert report["solution"]["solved"] is True
    assert report["pipeline"]["cleanup_errors"] == ["network in use"]
    assert "reward" not in report

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
