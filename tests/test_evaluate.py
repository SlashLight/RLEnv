"""Package -> report boundary tests for the evaluator.

They use controlled saved packages and a fake replay environment; no model and
no Docker are involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.evaluate import ReplayResult, evaluate
from pipeline.runlog import EventLog
from pipeline.util import tree_manifest, write_json

EXPECTED = "SEKAI{expected_flag}"
NEW_FLAG = "SEKAI{new_fresh_flag}"


def event(seq: int, kind: str, data=None, source: str = "pi", **extra):
    item = {
        "run_id": "r1",
        "seq": seq,
        "ts": "2026-10-24T10:00:00.000Z",
        "source": source,
        "type": kind,
        "data": data,
    }
    item.update(extra)
    return item


def make_package(tmp_path: Path, *, expected: str | None = EXPECTED,
                 flag: bytes | None = None, script: str | None = None,
                 extra_files: dict[str, str] | None = None,
                 manifest: dict | None = None,
                 events: list[dict] | None = None) -> Path:
    run_dir = tmp_path / "r1"
    (run_dir / "private").mkdir(parents=True)
    if expected is not None:
        (run_dir / "private" / "expected_flag.txt").write_text(expected + "\n")
    write_json(run_dir / "status.json", {
        "run_id": "r1", "status": "done", "stage": "done",
        "stop_reason": {"kind": "agent_settled"},
    })
    write_json(run_dir / "passport.json", {
        "run_id": "r1",
        "task": {"id": "my-flask-app", "version": "v1"},
        "agent": {"provider": "p", "model": "m"},
    })
    submission = run_dir / "submission"
    need_submission = flag is not None or script is not None or extra_files
    if need_submission:
        submission.mkdir()
        if flag is not None:
            (submission / "flag.txt").write_bytes(flag)
        if script is not None:
            (submission / "solution.py").write_text(script)
        for name, content in (extra_files or {}).items():
            path = submission / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        if manifest is None:
            manifest = {
                "present": True, "health": "ok", "violations": [],
                "files": tree_manifest(submission),
            }
        else:
            manifest = {**manifest, "files": tree_manifest(submission)}
    if manifest is not None:
        write_json(run_dir / "submission_manifest.json", manifest)
    if events:
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
            for item in events:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return run_dir


class FakeEnv:
    def __init__(self, *, result: ReplayResult | None = None, mode: str = "result",
                 prepare_error: Exception | None = None, cleanup_errors=None):
        self.result = result
        self.mode = mode
        self.prepare_error = prepare_error
        self.cleanup_errors = cleanup_errors or []
        self.prepared_with: str | None = None
        self.ran = False
        self.closed = False

    def prepare(self, flag: str) -> str:
        self.prepared_with = flag
        if self.prepare_error:
            raise self.prepare_error
        return "http://fake-service/"

    def run(self, submission_dir: Path, url: str, timeout: float) -> ReplayResult:
        self.ran = True
        if self.mode == "local":
            try:
                proc = subprocess.run(
                    [sys.executable, str(submission_dir / "solution.py"), url],
                    cwd=submission_dir, capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                return ReplayResult(None, exc.stdout or "", exc.stderr or "", 1,
                                    ["python", "solution.py", url], timed_out=True)
            return ReplayResult(proc.returncode, proc.stdout, proc.stderr, 1,
                                ["python", "solution.py", url])
        assert self.result is not None
        return self.result

    def close(self):
        self.closed = True
        return self.cleanup_errors


def run_eval(run_dir: Path, env: FakeEnv | None = None, **kwargs):
    return evaluate(
        run_dir,
        replay_env_factory=(lambda: env) if env is not None else None,
        replay_timeout=kwargs.pop("replay_timeout", 2),
        new_flag=kwargs.pop("new_flag", NEW_FLAG),
        **kwargs,
    )


# -- chronology ----------------------------------------------------------

def test_call_and_result_are_linked_and_rendered(tmp_path):
    run_dir = make_package(
        tmp_path,
        manifest={"present": False, "health": "ok", "violations": [], "files": []},
        events=[
            event(1, "agent_start", {}),
            event(2, "tool_execution_start", {
                "toolCallId": "call-1", "toolName": "bash",
                "args": {"command": "curl -s http://flask-app:5000/"},
            }),
            event(3, "tool_execution_end", {
                "toolCallId": "call-1", "toolName": "bash", "durationMs": 42,
                "result": {"content": [{"type": "text", "text": "hello"}]},
                "isError": False,
            }),
            event(4, "message_end", {"message": {
                "role": "assistant", "content": [{"type": "text", "text": "I tried the service."}],
            }}),
        ],
    )
    report = run_eval(run_dir, execute_replay=False)
    calls = [item for item in report["chronology"] if item["kind"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["start_event"] == 2
    assert calls[0]["end_event"] == 3
    assert calls[0]["status"] == "completed"
    assert calls[0]["result_text"] == "hello"
    assert calls[0]["duration_ms"] == 42
    assert report["data_health"]["unfinished_calls"] == []
    assert report["flag_check"]["verdict"] == "not_confirmed"
    assert report["flag_check"]["reason"] == "flag_missing"
    assert report["script_check"]["verdict"] == "not_confirmed"
    assert report["script_check"]["reason"] == "script_missing"
    assert report["solution"]["status"] == "not_confirmed"
    markdown = (run_dir / "report.md").read_text()
    assert "curl -s http://flask-app:5000/" in markdown
    assert "result [3]" in markdown


def test_unfinished_call_truncation_and_recording_loss_are_visible(tmp_path):
    run_dir = make_package(
        tmp_path,
        manifest={"present": False, "health": "ok", "violations": [], "files": []},
        events=[
            event(1, "tool_execution_start", {
                "toolCallId": "call-open", "toolName": "bash", "args": {"command": "sleep 100"},
            }),
            event(2, "message_end", {"message": {
                "role": "assistant", "content": [{"type": "text", "text": "x" * 5000}],
            }}, truncated=True, truncation={"truncated_strings": 1}),
        ],
    )
    write_json(run_dir / "raw" / "recording_health.json", {
        "errors": [{"stream": "events", "error": "OSError: disk full"}],
    })
    report = run_eval(run_dir, execute_replay=False)
    assert report["data_health"]["unfinished_calls"] == [{"call_id": "call-open", "event": 1}]
    assert 2 in report["data_health"]["truncated_events"]
    assert report["data_health"]["complete"] is False
    assert report["chronology"][0]["status"] == "incomplete"
    markdown = (run_dir / "report.md").read_text()
    assert "unfinished call" in markdown
    assert "recording error" in markdown
    assert "(truncated)" in markdown


def test_orphan_result_is_not_an_invented_success(tmp_path):
    run_dir = make_package(
        tmp_path,
        manifest={"present": False, "health": "ok", "violations": [], "files": []},
        events=[event(1, "tool_execution_end", {
            "toolCallId": "missing", "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "ok"}]}, "isError": False,
        })],
    )
    report = run_eval(run_dir, execute_replay=False)
    assert report["data_health"]["orphan_results"] == [1]
    assert report["chronology"][0]["kind"] == "tool_result_orphan"


# -- flag check ----------------------------------------------------------

@pytest.mark.parametrize("submitted, verdict, reason", [
    (EXPECTED.encode(), "confirmed", "flag_matches"),
    (EXPECTED.encode() + b"\n", "confirmed", "flag_matches"),
    (EXPECTED.encode() + b"\r\n", "confirmed", "flag_matches"),
    (EXPECTED.encode() + b" ", "not_confirmed", "flag_mismatch"),
    (b" " + EXPECTED.encode(), "not_confirmed", "flag_mismatch"),
    (b"SEKAI{other}", "not_confirmed", "flag_mismatch"),
])
def test_flag_comparison_is_exact_except_one_trailing_newline(tmp_path, submitted, verdict, reason):
    run_dir = make_package(tmp_path, flag=submitted)
    report = run_eval(run_dir, execute_replay=False)
    assert report["flag_check"]["verdict"] == verdict
    assert report["flag_check"]["reason"] == reason


def test_valid_flag_is_checked_even_when_other_entries_are_invalid(tmp_path):
    run_dir = make_package(
        tmp_path,
        flag=EXPECTED.encode(),
        extra_files={"notes.md": "x"},
        manifest={
            "present": True, "health": "ok",
            "violations": [{"kind": "not_a_regular_file", "path": "link", "type": "symlink"}],
            "files": [],
        },
    )
    report = run_eval(run_dir, execute_replay=False)
    assert report["flag_check"]["verdict"] == "confirmed"
    assert report["script_check"]["reason"] == "invalid_submission"


def test_missing_flag_with_healthy_collection_is_not_an_infrastructure_error(tmp_path):
    run_dir = make_package(tmp_path, extra_files={"notes.md": "nothing"})
    report = run_eval(run_dir, execute_replay=False)
    assert report["flag_check"]["verdict"] == "not_confirmed"
    assert report["flag_check"]["reason"] == "flag_missing"
    assert report["pipeline"]["status"] == "ok"


def test_failed_collection_makes_the_check_inconclusive(tmp_path):
    run_dir = make_package(
        tmp_path,
        manifest={"present": False, "health": "error",
                  "violations": [{"kind": "copy_failed", "detail": "daemon gone"}], "files": []},
    )
    report = run_eval(run_dir, execute_replay=False)
    assert report["flag_check"]["verdict"] == "check_error"
    assert report["flag_check"]["reason"] == "submission_collection_error"
    assert report["script_check"]["verdict"] == "check_error"


def test_missing_expected_flag_is_a_check_error_not_a_mismatch(tmp_path):
    run_dir = make_package(tmp_path, expected=None, flag=b"anything")
    report = run_eval(run_dir, execute_replay=False)
    assert report["flag_check"]["verdict"] == "check_error"
    assert report["flag_check"]["reason"] == "expected_flag_unavailable"


# -- replay check --------------------------------------------------------

def test_replay_success_requires_exactly_the_new_flag(tmp_path):
    run_dir = make_package(
        tmp_path,
        flag=EXPECTED.encode(),
        script="print('whatever')\n",
    )
    env = FakeEnv(result=ReplayResult(0, NEW_FLAG + "\n", "", 10,
                                      ["python", "solution.py", "http://fake-service/"]))
    report = run_eval(run_dir, env)
    assert env.prepared_with == NEW_FLAG
    assert env.closed
    assert report["script_check"]["verdict"] == "confirmed"
    assert report["script_check"]["reason"] == "script_reproduced"
    assert report["solution"]["solved"] is True
    assert report["solution"]["status"] == "confirmed"
    assert "stdout.txt" in Path(report["script_check"]["evidence"][0]["path"]).name
    assert NEW_FLAG not in json.dumps(report["script_check"], ensure_ascii=False)


def test_replay_local_script_receives_url_as_positional_argument(tmp_path):
    run_dir = make_package(
        tmp_path,
        flag=EXPECTED.encode(),
        script=(
            "import sys\n"
            "assert sys.argv[1] == 'http://fake-service/'\n"
            f"print('{NEW_FLAG}')\n"
        ),
    )
    report = run_eval(run_dir, FakeEnv(mode="local"))
    assert report["script_check"]["verdict"] == "confirmed"
    assert report["script_check"]["exit_code"] == 0


@pytest.mark.parametrize("result, verdict, reason", [
    (ReplayResult(0, "SEKAI{wrong}\n", "", 5, ["python"]), "not_confirmed", "script_wrong_output"),
    (ReplayResult(0, NEW_FLAG + "\nextra\n", "", 5, ["python"]), "not_confirmed", "script_wrong_output"),
    (ReplayResult(2, NEW_FLAG + "\n", "", 5, ["python"]), "not_confirmed", "script_exit_nonzero"),
    (ReplayResult(None, "", "", 5, ["python"], timed_out=True), "not_confirmed", "script_timeout"),
])
def test_replay_negative_outcomes(tmp_path, result, verdict, reason):
    run_dir = make_package(tmp_path, flag=EXPECTED.encode(), script="print('x')\n")
    report = run_eval(run_dir, FakeEnv(result=result))
    assert report["script_check"]["verdict"] == verdict
    assert report["script_check"]["reason"] == reason
    assert report["solution"]["status"] == "not_confirmed"


def test_missing_script_does_not_start_a_replay(tmp_path):
    run_dir = make_package(tmp_path, flag=EXPECTED.encode(), extra_files={"notes.md": "x"})
    env = FakeEnv(result=ReplayResult(0, NEW_FLAG, "", 1, ["python"]))
    report = run_eval(run_dir, env)
    assert report["script_check"]["verdict"] == "not_confirmed"
    assert report["script_check"]["reason"] == "script_missing"
    assert env.prepared_with is None
    assert env.ran is False


def test_replay_environment_failure_is_a_check_error(tmp_path):
    run_dir = make_package(tmp_path, flag=EXPECTED.encode(), script="print('x')\n")
    env = FakeEnv(prepare_error=RuntimeError("docker daemon unavailable"))
    report = run_eval(run_dir, env)
    assert report["script_check"]["verdict"] == "check_error"
    assert report["script_check"]["reason"] == "environment_error"
    assert report["solution"]["status"] == "check_error"
    assert "docker daemon unavailable" in report["script_check"]["explanation"]


def test_cleanup_failure_is_a_check_error(tmp_path):
    run_dir = make_package(tmp_path, flag=EXPECTED.encode(), script="print('x')\n")
    env = FakeEnv(result=ReplayResult(0, NEW_FLAG + "\n", "", 1, ["python"]),
                  cleanup_errors=["network: in use"])
    report = run_eval(run_dir, env)
    assert report["script_check"]["verdict"] == "check_error"
    assert report["script_check"]["reason"] == "cleanup_error"


def test_invalid_submission_composition_blocks_replay(tmp_path):
    run_dir = make_package(
        tmp_path,
        flag=EXPECTED.encode(),
        script="print('x')\n",
        manifest={
            "present": True, "health": "ok",
            "violations": [{"kind": "not_a_regular_file", "path": "link", "type": "symlink"}],
            "files": [],
        },
    )
    env = FakeEnv(result=ReplayResult(0, NEW_FLAG, "", 1, ["python"]))
    report = run_eval(run_dir, env)
    assert report["script_check"]["verdict"] == "not_confirmed"
    assert report["script_check"]["reason"] == "invalid_submission"
    assert env.ran is False


def test_solution_requires_both_checks(tmp_path):
    run_dir = make_package(tmp_path, flag=b"SEKAI{wrong}", script="print('x')\n")
    report = run_eval(
        run_dir,
        FakeEnv(result=ReplayResult(0, NEW_FLAG + "\n", "", 1, ["python"])),
    )
    assert report["flag_check"]["verdict"] == "not_confirmed"
    assert report["script_check"]["verdict"] == "confirmed"
    assert report["solution"]["solved"] is False


def test_expected_flag_never_appears_in_the_report(tmp_path):
    run_dir = make_package(tmp_path, flag=EXPECTED.encode(), script="print('x')\n")
    env = FakeEnv(result=ReplayResult(0, NEW_FLAG + "\n", "", 1, ["python"]))
    evaluate(run_dir, replay_env_factory=lambda: env, replay_timeout=2, new_flag=NEW_FLAG)
    assert EXPECTED not in (run_dir / "report.json").read_text()
    assert EXPECTED not in (run_dir / "report.md").read_text()
    assert NEW_FLAG not in (run_dir / "report.json").read_text()


# -- recording -----------------------------------------------------------

def test_event_log_keeps_raw_line_links_and_marks_truncation(tmp_path):
    run_dir = tmp_path / "rec"
    log = EventLog(run_dir, "r1", event_limit_bytes=1000)
    raw_line = log.raw_record({"type": "tool_execution_end", "long": "y" * 4000})
    published = log.publish("pi", "tool_execution_end", {"long": "y" * 4000}, raw_line=raw_line)
    log.close()
    assert published["seq"] == 1
    assert published["raw_line"] == 1
    assert published["truncated"] is True
    lines = (run_dir / "raw" / "rpc.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["long"] == "y" * 4000
