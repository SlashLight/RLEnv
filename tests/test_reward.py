"""Independent local check states; legacy scoring remains remote-only."""
import pytest

from pipeline.evaluate import _solution_status
from pipeline.reward import build_reward


@pytest.mark.parametrize("flag,script,solved,status", [
    ("confirmed", "confirmed", True, "confirmed"),
    ("confirmed", "not_confirmed", False, "not_confirmed"),
    ("not_confirmed", "confirmed", False, "not_confirmed"),
    ("not_confirmed", "not_confirmed", False, "not_confirmed"),
    ("confirmed", "check_error", None, "check_error"),
    ("check_error", "not_confirmed", None, "check_error"),
    ("confirmed", "not_evaluated", None, "not_evaluated"),
    ("not_evaluated", "confirmed", None, "not_evaluated"),
])
def test_solution_uses_both_independent_checks(flag, script, solved, status):
    result = _solution_status({"verdict": flag}, {"verdict": script})
    assert result["solved"] is solved
    assert result["status"] == status
    assert "value" not in result and "penalty" not in result


def test_remote_legacy_reward_remains_unavailable_without_replay():
    report = {
        "chronology": [], "data_health": {"complete": True}, "pipeline": {"status": "ok"},
        "flag_check": {"verdict": "confirmed"},
        "script_check": {"verdict": "not_evaluated", "reason": "remote_no_replay"},
    }
    result = build_reward(report, [], {"status": "not_evaluated", "confirmed_points": 0})
    assert result["spec"] == "web-ctf/reward/v1"
    assert result["value"] is None and result["outcome"] is None
