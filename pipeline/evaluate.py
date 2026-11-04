"""Evaluator: reads a saved run package and produces a linked, verifiable report.

The report boundary is the package directory; nothing here talks to a live pi
session. Replay may start a fresh task instance, never a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .milestones import evaluate_milestones
from .runlog import read_events, read_records
from .reward import build_reward
from .util import read_json, sha256_file, tree_manifest, truncate_text, write_json

CONFIRMED = "confirmed"
NOT_CONFIRMED = "not_confirmed"
CHECK_ERROR = "check_error"
SCRIPT_NAME = "solution.py"
FLAG_NAME = "flag.txt"


@dataclass
class ReplayResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    command: list[str]
    timed_out: bool = False
    environment: dict[str, Any] | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    syntax_check: dict[str, Any] | None = None


class ReplayEnvironment(Protocol):
    def prepare(self, flag: str) -> str: ...

    def run(self, submission_dir: Path, url: str, timeout: float) -> ReplayResult: ...

    def close(self) -> list[str]: ...


def _evidence_file(path: Path, run_dir: Path) -> dict[str, Any]:
    rel = str(path.relative_to(run_dir))
    item: dict[str, Any] = {"kind": "file", "path": rel}
    if path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
    return item


def _submission_integrity(run_dir: Path, manifest: dict[str, Any] | None) -> str | None:
    if manifest is None or manifest.get("health") == "error":
        return None
    if not isinstance(manifest.get("files"), list):
        return "Submission manifest has no file inventory."
    root = run_dir / "submission"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        return "Saved submission is not a regular directory."
    try:
        expected = {item["path"]: (item["bytes"], item["sha256"])
                    for item in manifest["files"] if item.get("type") == "file"}
        actual = tree_manifest(root)
        if any(item["type"] != "file" for item in actual):
            return "Saved submission contains links or special files."
        found = {item["path"]: (item["bytes"], item["sha256"]) for item in actual}
        if found != expected:
            return "Saved submission differs from the collected file inventory or hashes."
    except (OSError, KeyError, TypeError) as exc:
        return f"Cannot verify saved submission: {exc}"
    return None


def load_package(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    passport_path = run_dir / "passport.json"
    status_path = run_dir / "status.json"
    expected_path = run_dir / "private" / "expected_flag.txt"
    submission_manifest_path = run_dir / "submission_manifest.json"

    passport = read_json(passport_path) if passport_path.exists() else {}
    status = read_json(status_path) if status_path.exists() else {}
    expected = expected_path.read_text(encoding="utf-8") if expected_path.exists() else None
    if expected is not None and expected.endswith("\n"):
        expected = expected[:-1]
    submission_manifest = (
        read_json(submission_manifest_path) if submission_manifest_path.exists() else None
    )
    return {
        "run_dir": run_dir,
        "passport": passport,
        "status": status,
        "expected_flag": expected,
        "submission_manifest": submission_manifest,
        "submission_integrity_error": _submission_integrity(run_dir, submission_manifest),
        "events": read_events(run_dir),
    }


# -- chronology ----------------------------------------------------------

def _text_of_result(result: Any) -> str:
    if not isinstance(result, dict):
        return "" if result is None else str(result)
    blocks = result.get("content")
    if isinstance(blocks, list):
        parts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False)[:2000]


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(block.get("text", "")) for block in content
                 if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts)
    return ""


def build_chronology(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    health: dict[str, Any] = {
        "events": len(events),
        "unfinished_calls": [],
        "orphan_results": [],
        "truncated_events": [],
        "protocol_errors": [],
    }
    for event in events:
        kind = event.get("type")
        data = event.get("data") or {}
        if event.get("truncated"):
            health["truncated_events"].append(event.get("seq"))
        if kind == "protocol_error":
            health["protocol_errors"].append(event.get("seq"))
        if kind == "tool_execution_start":
            call_id = str(data.get("toolCallId"))
            entry = {
                "kind": "tool_call",
                "call_id": call_id,
                "tool": data.get("toolName"),
                "args": data.get("args"),
                "start_event": event.get("seq"),
                "status": "incomplete",
                "end_event": None,
                "result_text": "",
                "is_error": None,
                "duration_ms": None,
                "truncated": bool(event.get("truncated")),
            }
            calls[call_id] = entry
            order.append(entry)
        elif kind == "tool_execution_update":
            call_id = str(data.get("toolCallId"))
            entry = calls.get(call_id)
            if entry is not None:
                partial = _text_of_result(data.get("partialResult"))
                if partial:
                    entry["partial_result"] = partial
        elif kind == "tool_execution_end":
            call_id = str(data.get("toolCallId"))
            entry = calls.get(call_id)
            if entry is None:
                health["orphan_results"].append(event.get("seq"))
                order.append({
                    "kind": "tool_result_orphan",
                    "call_id": call_id,
                    "tool": data.get("toolName"),
                    "event": event.get("seq"),
                    "result_text": _text_of_result(data.get("result")),
                    "truncated": bool(event.get("truncated")),
                })
                continue
            entry["status"] = "error" if data.get("isError") else "completed"
            entry["end_event"] = event.get("seq")
            entry["is_error"] = bool(data.get("isError"))
            entry["duration_ms"] = data.get("durationMs")
            entry["result_text"] = _text_of_result(data.get("result"))
            entry["truncated"] = bool(entry["truncated"] or event.get("truncated"))
        elif kind == "message_end":
            message = data.get("message") or {}
            if message.get("role") == "assistant":
                text = _message_text(message)
                if text:
                    order.append({
                        "kind": "assistant_message",
                        "event": event.get("seq"),
                        "text": text,
                        "truncated": bool(event.get("truncated")),
                    })
    for call_id, entry in calls.items():
        if entry["status"] == "incomplete":
            health["unfinished_calls"].append({"call_id": call_id, "event": entry["start_event"]})
    health["tool_calls"] = len(calls)
    return order, health


def build_data_health(run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    _, health = build_chronology(events)
    recording_path = run_dir / "raw" / "recording_health.json"
    errors = []
    if recording_path.exists():
        errors = read_json(recording_path).get("errors", [])
    records, event_errors = read_records(run_dir / "events.jsonl")
    raw_lines: list[int] = []
    raw_records, raw_errors = read_records(run_dir / "raw" / "rpc.jsonl", raw_lines)
    raw_by_line = dict(zip(raw_lines, raw_records))
    errors = list(errors) + event_errors + raw_errors
    previous = 0
    for event in records:
        seq = event.get("seq")
        if not isinstance(seq, int) or seq != previous + 1:
            errors.append({"stream": "events.jsonl", "error": "invalid or missing sequence",
                           "previous": previous, "seq": seq})
        if isinstance(seq, int):
            previous = seq
        raw_line = event.get("raw_line")
        if raw_line is not None or event.get("source") == "pi":
            raw = raw_by_line.get(raw_line) if type(raw_line) is int else None
            if raw is None or (not event.get("truncated") and raw != event.get("data")):
                errors.append({"stream": "rpc.jsonl", "seq": seq, "raw_line": raw_line,
                               "error": "raw reference missing or does not match event"})
        if (event.get("source") == "sandbox" and isinstance(event.get("data"), dict)
                and event["data"].get("truncated")):
            health["truncated_events"].append(seq)
    health.update({
        "raw_records": len(raw_records),
        "recording_errors": errors,
        "complete": not (errors or health["protocol_errors"] or health["unfinished_calls"]
                         or health["orphan_results"] or health["truncated_events"]),
    })
    return health


# -- checks --------------------------------------------------------------

def check_flag(package: dict[str, Any]) -> dict[str, Any]:
    run_dir: Path = package["run_dir"]
    expected: str | None = package["expected_flag"]
    manifest = package["submission_manifest"]
    evidence: list[dict[str, Any]] = []
    is_remote = bool(package.get("passport", {}).get("remote_url"))
    if is_remote and expected is not None and not expected.strip():
        return {"check": "flag", "verdict": "not_evaluated",
                "reason": "reference_not_supplied",
                "explanation": "No reference flag supplied; result saved without grading.",
                "evidence": []}
    if expected is None:
        return {
            "check": "flag",
            "verdict": CHECK_ERROR,
            "reason": "expected_flag_unavailable",
            "explanation": "The run package does not contain the expected flag, so the check cannot run.",
            "evidence": [],
        }
    if manifest is None:
        return {
            "check": "flag",
            "verdict": CHECK_ERROR,
            "reason": "submission_manifest_unavailable",
            "explanation": "Submission collection result is missing from the package.",
            "evidence": [],
        }
    if manifest.get("health") == "error":
        return {
            "check": "flag",
            "verdict": CHECK_ERROR,
            "reason": "submission_collection_error",
            "explanation": "The submission could not be collected, so an absent flag is not a reliable negative.",
            "evidence": [],
        }
    if package.get("submission_integrity_error"):
        return {"check": "flag", "verdict": CHECK_ERROR,
                "reason": "submission_integrity_error",
                "explanation": package["submission_integrity_error"],
                "evidence": [_evidence_file(run_dir / "submission_manifest.json", run_dir)]}
    flag_path = run_dir / "submission" / FLAG_NAME
    if flag_path.is_symlink() or not flag_path.is_file():
        if flag_path.exists() and not flag_path.is_file():
            evidence.append(_evidence_file(flag_path, run_dir))
        return {
            "check": "flag",
            "verdict": NOT_CONFIRMED,
            "reason": "flag_missing",
            "explanation": "No regular submission/flag.txt was collected; collection itself is healthy.",
            "evidence": evidence,
        }
    submitted = flag_path.read_bytes()
    wanted = expected.encode("utf-8")
    evidence.append(_evidence_file(flag_path, run_dir))
    if is_remote:
        matched = submitted.decode("utf-8", "replace").strip() == expected.strip()
    else:
        matched = submitted in (wanted, wanted + b"\r\n", wanted + b"\n")
    return {
        "check": "flag",
        "verdict": CONFIRMED if matched else NOT_CONFIRMED,
        "reason": "flag_matches" if matched else "flag_mismatch",
        "explanation": (
            "Exact flag comparison with leading/trailing whitespace ignored."
            if is_remote else (
                "Submission/flag.txt equals the expected flag (trailing newline ignored)."
                if matched else
                "Submission/flag.txt differs from the expected flag; no whitespace was normalized "
                "beyond one trailing line ending."
            )
        ),
        "evidence": evidence,
    }


def _submission_invalid(manifest: dict[str, Any]) -> bool:
    return any(item.get("kind") in {"not_a_regular_file", "submission_too_large"}
               for item in manifest.get("violations", []))


def check_script(package: dict[str, Any], env: ReplayEnvironment | None,
                 replay_timeout: float, new_flag: str,
                 evidence_dir: Path) -> dict[str, Any]:
    run_dir: Path = package["run_dir"]
    manifest = package["submission_manifest"]
    submission = run_dir / "submission"
    base = {"check": "script", "evidence": []}

    if manifest is None:
        return {**base, "verdict": CHECK_ERROR, "reason": "submission_manifest_unavailable",
                "explanation": "Submission collection result is missing from the package."}
    if manifest.get("health") == "error" or any(
        item.get("kind") == "copy_failed" for item in manifest.get("violations", [])
    ):
        return {**base, "verdict": CHECK_ERROR, "reason": "submission_collection_error",
                "explanation": "The submission could not be collected, so replay cannot be judged."}
    if package.get("submission_integrity_error"):
        return {**base, "verdict": CHECK_ERROR, "reason": "submission_integrity_error",
                "explanation": package["submission_integrity_error"],
                "evidence": [_evidence_file(run_dir / "submission_manifest.json", run_dir)]}
    if _submission_invalid(manifest):
        return {**base, "verdict": NOT_CONFIRMED, "reason": "invalid_submission",
                "explanation": "The submission contains symlinks, non-regular entries or exceeds the size limit.",
                "evidence": [_evidence_file(run_dir / "submission_manifest.json", run_dir)]}
    script = submission / SCRIPT_NAME
    if not script.is_file() or script.is_symlink():
        return {**base, "verdict": NOT_CONFIRMED, "reason": "script_missing",
                "explanation": f"No regular submission/{SCRIPT_NAME} was submitted."}
    if env is None:
        return {**base, "verdict": CHECK_ERROR, "reason": "replay_environment_unavailable",
                "explanation": "No replay environment is configured for this evaluation."}

    evidence_dir.mkdir(parents=True, exist_ok=True)
    result: ReplayResult | None = None
    prepare_error: Exception | None = None
    run_error: Exception | None = None
    cleanup_errors: list[str] = []
    try:
        url = env.prepare(new_flag)
        try:
            result = env.run(submission, url, replay_timeout)
        except Exception as exc:  # noqa: BLE001 - environment failures must not look negative
            run_error = exc
    except Exception as exc:  # noqa: BLE001
        prepare_error = exc
    finally:
        try:
            cleanup_errors = env.close()
        except Exception as exc:  # cleanup must not erase the check result
            cleanup_errors = [f"{type(exc).__name__}: {exc}"]
    if prepare_error is not None:
        return {**base, "verdict": CHECK_ERROR, "reason": "environment_error",
                "explanation": f"A clean task instance could not be prepared: {prepare_error}",
                "cleanup_errors": cleanup_errors}
    if run_error is not None:
        return {**base, "verdict": CHECK_ERROR, "reason": "environment_error",
                "explanation": f"The replay environment failed: {run_error}",
                "cleanup_errors": cleanup_errors}
    assert result is not None

    stdout_path = evidence_dir / "stdout.txt"
    stderr_path = evidence_dir / "stderr.txt"
    stdout_text, stdout_cut = truncate_text(result.stdout, 262144)
    stderr_text, stderr_cut = truncate_text(result.stderr, 262144)
    stdout_cut |= result.stdout_truncated
    stderr_cut |= result.stderr_truncated
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    evidence = [
        _evidence_file(stdout_path, run_dir),
        _evidence_file(stderr_path, run_dir),
        _evidence_file(script, run_dir),
    ]

    def verdict(value: str, reason: str, explanation: str) -> dict[str, Any]:
        return {
            **base,
            "verdict": value,
            "reason": reason,
            "explanation": explanation,
            "evidence": evidence,
            "command": result.command,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "fresh_instance": True,
            "stdout_truncated": stdout_cut,
            "stderr_truncated": stderr_cut,
            "environment": result.environment or {},
            "syntax_check": result.syntax_check,
            "cleanup_errors": cleanup_errors,
        }

    if cleanup_errors:
        return verdict(CHECK_ERROR, "cleanup_error",
                       "Replay finished but its resources could not be fully cleaned up.")
    if result.timed_out:
        return verdict(NOT_CONFIRMED, "script_timeout",
                       f"The script did not finish within {replay_timeout:g} seconds on a healthy instance.")
    if result.exit_code != 0:
        return verdict(NOT_CONFIRMED, "script_exit_nonzero",
                       f"The script exited with code {result.exit_code}.")
    wanted = new_flag.encode("utf-8")
    produced = result.stdout.encode("utf-8")
    if not stdout_cut and produced in (wanted, wanted + b"\r\n", wanted + b"\n"):
        return verdict(CONFIRMED, "script_reproduced",
                       "A fresh instance with a new flag was solved: exit code 0 and exactly the new flag on stdout.")
    return verdict(NOT_CONFIRMED, "script_wrong_output",
                   "stdout did not contain exactly the new flag (or contained more than it).")


# -- report --------------------------------------------------------------

def _solution_status(flag_check: dict[str, Any], script_check: dict[str, Any]) -> dict[str, Any]:
    checks = (flag_check, script_check)
    if any(item["verdict"] == CHECK_ERROR for item in checks):
        return {"solved": False, "status": "check_error",
                "note": "A check could not run; this is not a negative result."}
    solved = all(item["verdict"] == CONFIRMED for item in checks)
    return {
        "solved": solved,
        "status": "confirmed" if solved else "not_confirmed",
        "note": "Solution success requires both the submitted flag and the independent replay.",
    }


def evaluate(run_dir: Path, *, replay_env_factory: Callable[[], ReplayEnvironment] | None = None,
             replay_timeout: float = 120, new_flag: str | None = None,
             execute_replay: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    package = load_package(run_dir)
    events = package["events"]
    chronology, health = build_chronology(events)
    data_health = build_data_health(run_dir, events)
    data_health.update({key: value for key, value in health.items() if key not in data_health})

    flag_check = check_flag(package)

    is_remote = bool(package["passport"].get("remote_url"))
    env: ReplayEnvironment | None = None
    if not is_remote and execute_replay and replay_env_factory is not None and new_flag:
        env = replay_env_factory()
    script_check = ({
        "check": "script", "verdict": "not_evaluated", "reason": "remote_no_replay",
        "explanation": "Remote task: no replay, reset or flag rotation.", "evidence": [],
    } if is_remote else check_script(
        package, env, replay_timeout, new_flag or "",
        run_dir / "checks" / "replay",
    ))

    status = package["status"]
    report = {
        "schema": "evaluator-pipeline/report/v1",
        "run_id": package["passport"].get("run_id", status.get("run_id")),
        "package": str(run_dir),
        "publication_ready": False,
        "publication_note": "Local research data. It is not prepared for publication.",
        "passport": package["passport"],
        "pipeline": {
            "status": "ok" if status.get("status") != "failed" else "error",
            "stage": status.get("stage"),
            "error": status.get("error"),
        },
        "stop_reason": status.get("stop_reason", {"kind": "unknown"}),
        "data_health": data_health,
        "chronology": chronology,
        "submission": {
            "present": bool(package["submission_manifest"] and package["submission_manifest"].get("present")),
            "manifest": package["submission_manifest"],
        },
        "flag_check": flag_check,
        "script_check": script_check,
        "solution": ({"status": flag_check["verdict"],
                      "solved": (None if flag_check["verdict"] == "not_evaluated"
                                 else flag_check["verdict"] == CONFIRMED),
                      "note": "Remote task: flag-only evaluation; script not verified."}
                     if is_remote else _solution_status(flag_check, script_check)),
    }
    report["milestones"] = evaluate_milestones(package)
    report["reward"] = build_reward(report, events, report["milestones"])
    write_json(run_dir / "checks" / "reward.json", report["reward"])
    write_json(run_dir / "checks" / "milestones.json", report["milestones"])
    write_json(run_dir / "checks" / "flag.json", flag_check)
    write_json(run_dir / "checks" / "replay.json", script_check)
    write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _fmt_check(item: dict[str, Any]) -> str:
    refs = ", ".join(
        f"event {entry['seq']}" if entry.get("kind") == "event" else str(entry.get("path"))
        for entry in item.get("evidence", [])
    )
    return (
        f"**{item['check']}**: `{item['verdict']}` (`{item['reason']}`) — {item['explanation']}"
        + (f"\nEvidence: {refs}" if refs else "")
    )


def render_markdown(report: dict[str, Any]) -> str:
    passport = report.get("passport", {})
    lines = [
        f"# Run report — {report.get('run_id', 'unknown')}",
        "",
        f"- Task: `{passport.get('task', {}).get('id', '?')}` @ `{passport.get('task', {}).get('version', '?')}`",
        f"- Pipeline: **{report['pipeline']['status']}** (stage `{report['pipeline'].get('stage')}`)",
        f"- Stop reason: `{report['stop_reason'].get('kind')}`"
        + (f" — {report['stop_reason'].get('detail')}" if report['stop_reason'].get('detail') else ""),
        f"- Package: `{report['package']}`",
        "- Local research data, not prepared for publication.",
        "",
        "## Checks",
        "",
        _fmt_check(report["flag_check"]),
        "",
        _fmt_check(report["script_check"]),
        "",
        f"Solution: **{report['solution']['status']}** (solved={report['solution']['solved']}). "
        f"{report['solution']['note']}",
        "",
        "## Data health",
        "",
    ]
    reward = report.get("reward")
    if reward is not None:
        lines[-2:] = ["## Reward", ""]
        lines += [
            f"- Specification: `{reward['spec']}` (experimental)",
            f"- Composite reward: **{reward['value']}**; outcome-only: **{reward['outcome']}**",
            f"- Base: {reward['base']}; penalties: {reward['penalty_raw']} "
            f"(applied {reward['penalty_applied']}, cap {reward['penalty_cap']})",
            f"- Status: `{reward['status']}`; unavailable: {', '.join(reward['unavailable']) or 'none'}",
        ]
        for item in reward["penalties"]:
            lines.append(f"- Penalty `{item['kind']}`: −{item['amount']}; evidence: {item['evidence']}")
        for item in report.get("milestones", {}).get("checks", []):
            lines.append(f"- Milestone `{item.get('id')}`: `{item.get('verdict')}`")
        lines += ["", "## Data health", ""]
    health = report["data_health"]
    lines += [
        f"- Events: {health.get('events')}, raw pi records: {health.get('raw_records')}",
        f"- Unfinished calls: {len(health.get('unfinished_calls', []))}; "
        f"orphan results: {len(health.get('orphan_results', []))}",
        f"- Truncated events: {len(health.get('truncated_events', []))}; "
        f"recording errors: {len(health.get('recording_errors', []))}",
    ]
    for item in health.get("unfinished_calls", []):
        lines.append(f"  - unfinished call `{item['call_id']}` started at event {item['event']}")
    for item in health.get("recording_errors", []):
        lines.append(f"  - recording error: {item}")
    lines += ["", "## Chronology", ""]
    for entry in report["chronology"]:
        if entry["kind"] == "tool_call":
            status = entry["status"]
            lines.append(
                f"- [{entry['start_event']}→{entry['end_event']}] `{entry['tool']}` "
                f"`{json.dumps(entry.get('args'), ensure_ascii=False)}` — **{status}**"
                + (" (truncated)" if entry.get("truncated") else "")
                + (f", {entry['duration_ms']} ms" if entry.get("duration_ms") is not None else "")
            )
            result = (entry.get("result_text") or "").strip()
            if result:
                preview = result[:600] + ("…" if len(result) > 600 else "")
                lines.append(f"  - result [{entry['end_event']}]: `{preview}`")
        elif entry["kind"] == "tool_result_orphan":
            lines.append(f"- [{entry['event']}] orphan result for `{entry.get('tool')}`")
        elif entry["kind"] == "assistant_message":
            preview = entry["text"][:400] + ("…" if len(entry["text"]) > 400 else "")
            lines.append(f"- [{entry['event']}] assistant: {preview}"
                         + (" (truncated)" if entry.get("truncated") else ""))
    return "\n".join(lines) + "\n"
