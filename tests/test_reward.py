"""Reward arithmetic and conservative handling of unavailable evidence."""
from copy import deepcopy

import pytest

from pipeline.reward import build_reward


def report(flag="not_confirmed", script="not_confirmed"):
    return {
        "chronology": [], "data_health": {"complete": True},
        "flag_check": {"verdict": flag},
        "script_check": {"verdict": script, "reason": "script_missing"},
        "pipeline": {"status": "ok"},
    }


def milestones(points=0, status="evaluated"):
    return {"status": status, "confirmed_points": points,
            "total_points": None if status == "check_error" else points, "checks": []}


def call(start, tool="read", args=None):
    return {"kind": "tool_call", "tool": tool, "args": args or {"path": "a"},
            "start_event": start, "end_event": start + 2}


@pytest.mark.parametrize("flag,script,points,wanted", [
    ("not_confirmed", "not_confirmed", 0, 0),
    ("not_confirmed", "not_confirmed", .3, .3),
    ("confirmed", "not_confirmed", 0, .5),
    ("confirmed", "not_confirmed", .3, .8),
    ("confirmed", "confirmed", 0, 1),
    ("confirmed", "confirmed", .3, 1),
    ("not_confirmed", "confirmed", 0, 0),
])
def test_scores(flag, script, points, wanted):
    result = build_reward(report(flag, script), [], milestones(points))
    assert result["value"] == wanted
    assert result["outcome"] == int(flag == script == "confirmed")


def test_optional_milestones_are_not_a_failed_check():
    assert build_reward(report(), [], milestones(status="not_evaluated"))["value"] == 0


def test_cap_and_success_priority():
    r = report("confirmed", "confirmed")
    r["chronology"] = [call(i * 3 + 1) for i in range(30)]
    result = build_reward(r, [], milestones(.3))
    assert result["penalty_raw"] == .29
    assert result["penalty_applied"] == .1
    assert result["value"] == .9
    assert result["outcome"] == 1


def test_zero_floor():
    r = report()
    r["chronology"] = [call(1), call(4)]
    result = build_reward(r, [], milestones())
    assert result["penalty_raw"] == .01
    assert result["value"] == 0


@pytest.mark.parametrize("data,kind,penalty", [
    ({"timed_out": True, "syntax_check": {"evidence_version": 3}}, "tool_exec", 0),
    ({"timed_out": True}, "tool_exec", .01),
    ({"error": "DockerError: unavailable", "infrastructure_error": True}, "tool_error", 0),
    ({"error": "permission denied", "infrastructure_error": False}, "tool_error", .01),
    ({"error": "not_found"}, "tool_error", .01),
    ({"output": "timed_out=true"}, "tool_exec", .01),
])
def test_retry_exclusions_use_metadata_not_text(data, kind, penalty):
    r = report()
    r["chronology"] = [call(1), call(4)]
    result = build_reward(r, [{"seq": 2, "source": "sandbox", "type": kind,
                              "data": data}], milestones())
    assert result["penalty_raw"] == penalty


def test_only_consecutive_exact_calls():
    r = report()
    r["chronology"] = [call(1), call(4, args={"path": "b"}), call(7)]
    assert build_reward(r, [], milestones())["penalties"] == []


@pytest.mark.parametrize("verdict,penalty", [("invalid", .01), ("valid", 0)])
def test_bash_syntax_from_trusted_event(verdict, penalty):
    r = report()
    r["chronology"] = [call(1, "bash", {"command": "if"})]
    events = [{"seq": 2, "source": "sandbox", "type": "tool_exec", "data": {
        "op": "bash", "command": "if", "syntax_check": {"verdict": verdict, "evidence_version": 3}}}]
    result = build_reward(r, events, milestones())
    assert result["penalty_raw"] == penalty
    assert result["value"] == 0


def test_forged_shell_output_is_not_syntax_evidence():
    r = report()
    r["chronology"] = [call(1, "bash", {"command": "echo syntax error"})]
    events = [{"seq": 2, "source": "sandbox", "type": "tool_exec", "data": {
        "op": "bash", "command": "echo syntax error", "output": "SyntaxError"}}]
    result = build_reward(r, events, milestones())
    assert result["penalties"] == []
    assert result["value"] is None


def test_final_python_syntax_penalty():
    r = report("confirmed")
    r["script_check"] = {"verdict": "not_confirmed", "syntax_check": {"verdict": "invalid"},
                         "evidence": [{"kind": "file", "path": "submission/solution.py"}]}
    result = build_reward(r, [], milestones())
    assert result["value"] == .45
    assert result["penalties"][0]["kind"] == "python_syntax"


@pytest.mark.parametrize("change", ["flag", "script", "infra", "recording", "milestones", "remote"])
def test_unavailable_is_not_zero(change):
    r = report("confirmed", "confirmed")
    m = milestones()
    if change in {"flag", "script"}:
        r[f"{change}_check"]["verdict"] = "check_error"
    elif change == "infra":
        r["pipeline"]["status"] = "error"
    elif change == "recording":
        r["data_health"]["complete"] = False
    elif change == "remote":
        r["script_check"]["verdict"] = "not_evaluated"
    else:
        m["status"] = "check_error"
    original = deepcopy(r)
    result = build_reward(r, [], m)
    assert result["value"] is None
    assert result["unavailable"]
    assert r == original
    if change not in {"recording", "milestones"}:
        assert result["outcome"] is None
