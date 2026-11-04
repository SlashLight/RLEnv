"""Versioned terminal rewards; observations are not judgements of reasoning."""

from __future__ import annotations

from typing import Any

SPEC = "web-ctf/reward/v1"


def build_reward(report: dict[str, Any], events: list[dict[str, Any]],
                 milestones: dict[str, Any]) -> dict[str, Any]:
    penalties: list[dict[str, Any]] = []
    unavailable: list[str] = []
    calls = [item for item in report["chronology"] if item["kind"] == "tool_call"]
    audits = [e for e in events if e.get("source") == "sandbox"]
    syntax_checks = []

    def penalize(kind: str, amount: float, evidence: list[dict[str, Any]]) -> None:
        penalties.append({"kind": kind, "amount": amount, "evidence": evidence})

    def within(call: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = call.get("start_event"), call.get("end_event")
        if not isinstance(start, int) or not isinstance(end, int):
            return []
        return [e for e in audits if start < e.get("seq", -1) < end]

    previous = None
    repeats = 0
    for call in calls:
        if previous is not None and (call.get("tool"), call.get("args")) == (
                previous.get("tool"), previous.get("args")):
            repeats += 1
            prior_audits = within(previous)
            retry = any(
                ((e.get("data") or {}).get("timed_out") is True
                 and ((e.get("data") or {}).get("syntax_check") or {}).get("evidence_version") == 3)
                or (e.get("type") == "tool_error"
                    and (e.get("data") or {}).get("infrastructure_error") is True)
                for e in prior_audits)
            # shortcut: exact consecutive calls only; semantic usefulness needs task-specific evidence.
            if (not retry and previous.get("end_event") is not None
                    and previous["end_event"] < call["start_event"]):
                penalize("exact_repeat", .01, [
                    {"kind": "event", "seq": previous["start_event"]},
                    {"kind": "event", "seq": call["start_event"]}])
        previous = call
        if call.get("tool") != "bash":
            continue
        matches = [e for e in within(call) if e.get("type") == "tool_exec"
                   and (e.get("data") or {}).get("op") == "bash"
                   and (e.get("data") or {}).get("command") == (call.get("args") or {}).get("command")]
        if len(matches) != 1:
            unavailable.append(f"bash_syntax:{call.get('start_event')}")
            continue
        event = matches[0]
        check = event["data"].get("syntax_check") or {}
        syntax_checks.append({"event": event["seq"], **check})
        if check.get("evidence_version") != 3:
            unavailable.append(f"bash_syntax:{call.get('start_event')}")
        elif check.get("verdict") == "invalid":
            penalize("bash_syntax", .01, [{"kind": "event", "seq": event["seq"]}])
        elif check.get("verdict") != "valid":
            unavailable.append(f"bash_syntax:{call.get('start_event')}")

    script = report["script_check"]
    python_syntax = script.get("syntax_check") or {}
    if python_syntax.get("verdict") == "invalid":
        penalize("python_syntax", .05, script.get("evidence", []))
    elif (python_syntax.get("verdict") != "valid"
          and script.get("reason") not in {"script_missing", "invalid_submission"}):
        unavailable.append("python_syntax")

    flag = report["flag_check"]
    outcome_valid = all(c.get("verdict") in {"confirmed", "not_confirmed"}
                        for c in (flag, script))
    if report.get("pipeline", {}).get("status") == "error" or report.get(
            "stop_reason", {}).get("kind") == "infra_error":
        outcome_valid = False
    solved = all(c.get("verdict") == "confirmed" for c in (flag, script))
    outcome = int(solved) if outcome_valid else None
    if not outcome_valid:
        unavailable.append("outcome_checks")
    if not report["data_health"].get("complete"):
        unavailable.append("incomplete_recording")
    if milestones.get("status") == "check_error":
        unavailable.append("milestone_checks")
    flag_points = .5 if flag.get("verdict") == "confirmed" else 0
    milestone_points = milestones.get("confirmed_points", 0)
    base = 1.0 if solved else round(flag_points + milestone_points, 6)
    raw_penalty = round(sum(p["amount"] for p in penalties), 6)
    penalty = min(.1, raw_penalty)
    return {
        "spec": SPEC, "experimental": True,
        "status": "not_evaluated" if unavailable else "evaluated",
        "unavailable": list(dict.fromkeys(unavailable)),
        "outcome": outcome,
        "value": None if unavailable else round(max(0, base - penalty), 6),
        "base": base if outcome_valid and milestones.get("status") != "check_error" else None,
        "components": {"flag": flag_points, "milestones": milestone_points,
                       "milestone_coverage": milestones.get("status"),
                       "full_solution": solved if outcome_valid else None},
        "penalties": penalties, "penalty_raw": raw_penalty,
        "penalty_applied": penalty, "penalty_cap": .1,
        "diagnostics": {"tool_calls": len(calls), "exact_repeats": repeats,
                        "bash_syntax": syntax_checks, "python_syntax": python_syntax},
    }
