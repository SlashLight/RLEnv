"""No inference: real pinned pi loads the extension against an in-process SSE stub."""
import json
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.config import Config, ConfigError, Limits
from pipeline.pi_rpc import PiError, PiSession
from pipeline.runlog import EventLog


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "200000"])
def test_token_budget_requires_positive_integer(value):
    with pytest.raises(ConfigError, match="token_budget"):
        Limits(token_budget=value)


def test_session_stop_precedence_and_remote_boundary(tmp_path):
    log = EventLog(tmp_path, "budget-test")
    try:
        session = PiSession(Config("p", "m"), log, tmp_path, Path("bridge.ts"),
                            "url", "token", "p", "m", "off")
        session._inspect({"type": "extension_ui_request", "method": "notify", "message":
                          "PIPELINE_BUDGET:" + json.dumps({
                              "event": "request_complete", "usage": {"stop_reason": "token_limit"}})})
        session._inspect({"type": "message_end", "message": {
            "role": "assistant", "stopReason": "error", "errorMessage": "blocked"}})
        assert session.wait_settled(time.monotonic() + 1) == "token_limit"
        session.submitted.set()
        assert session.wait_settled(time.monotonic() + 1) == "submitted"
        session.local = False
        session.api_error = None
        session._settled.set()
        assert session.wait_settled(time.monotonic() + 1) == "settled"
        assert "PIPELINE_LOCAL" not in session._env()
    finally:
        log.close()


def test_local_settled_continues_until_submission(tmp_path):
    log = EventLog(tmp_path, "continue-test")
    try:
        session = PiSession(Config("p", "m"), log, tmp_path, Path("bridge.ts"),
                            "url", "token", "p", "m", "off")
        session._settled.set()
        def submit(_text):
            session.submitted.set()
            return {"success": True}
        session.prompt = Mock(side_effect=submit)
        assert session.wait_settled(time.monotonic() + 1) == "submitted"
        session.prompt.assert_called_once()
    finally:
        log.close()


def test_version_pin_rejects_other_builds(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.pi_rpc.shutil.which", lambda _: "/pi")
    monkeypatch.setattr("pipeline.pi_rpc.subprocess.run", lambda *a, **k: Mock(stdout="1.5.3\n"))
    log = EventLog(tmp_path, "pin-test")
    try:
        session = PiSession(Config("p", "m"), log, tmp_path, Path("bridge.ts"),
                            "url", "token", "p", "m", "off")
        with pytest.raises(PiError, match="requires pi 1.5.2"):
            session._argv()
    finally:
        log.close()


STUB = r'''
import { writeFileSync } from "node:fs";
const plans = PLANS;
const requests = [];
export default function () {
    globalThis.fetch = async (input, init) => {
        if (!String(input).includes("/chat/completions")) throw new Error("unexpected network request");
        const payload = JSON.parse(init.body);
        requests.push(payload);
        writeFileSync(CALLS, JSON.stringify(requests));
        const plan = plans[requests.length - 1];
        if (!plan) throw new Error("unexpected extra upstream request");
        if (plan.http_error) return new Response("unavailable", { status: 503 });
        const base = { id: "stub", object: "chat.completion.chunk", model: payload.model, created: 1 };
        const chunks = [
            { ...base, choices: [{ index: 0, delta: { role: "assistant", content: plan.text ?? "ok" }, finish_reason: null }] },
            { ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
        ];
        // Usage deliberately comes after the finish chunk, in a separate body chunk.
        if ("usage" in plan) chunks.push({ ...base, choices: [], usage: plan.usage });
        const frames = chunks.map(chunk => "data: " + JSON.stringify(chunk) + "\n\n");
        if (!plan.truncated) frames.push("data: [DONE]\n\n");
        const encoder = new TextEncoder();
        let index = 0;
        return new Response(new ReadableStream({
            pull(controller) {
                if (index === frames.length) controller.close();
                else controller.enqueue(encoder.encode(frames[index++]));
            },
        }), { headers: { "Content-Type": "text/event-stream" } });
    };
}
'''


@pytest.fixture
def pinned_session(tmp_path):
    if shutil.which("pi") is None:
        pytest.skip("requires installed pi 1.5.2 (no model or credentials required)")
    sessions = []
    logs = []
    def make(plans, limit=200000):
        probe = tmp_path / "stub.ts"
        calls = tmp_path / "calls.json"
        probe.write_text(STUB.replace("PLANS", json.dumps(plans)).replace("CALLS", json.dumps(str(calls))))
        log = EventLog(tmp_path, "pinned-test")
        logs.append(log)
        cfg = Config("openrouter", "openai/gpt-4o-mini", api_key="stub-not-a-secret",
                     limits=Limits(token_budget=limit))
        bridge = Path(__file__).resolve().parents[1] / "pipeline/extensions/sandbox_bridge.ts"
        session = PiSession(cfg, log, tmp_path, bridge, "http://unused.invalid", "unused",
                            cfg.provider, cfg.model, cfg.thinking)
        argv = session._argv
        session._argv = lambda: [*argv(), "-e", str(probe)]
        sessions.append(session)
        return session, calls
    yield make
    for session in sessions:
        session.close()
    for log in logs:
        log.close()


def prompt_settled(session, text="research"):
    session._settled.clear()
    assert session.prompt(text)["success"]
    assert session._settled.wait(15), "stub pi did not settle"


def usage(session):
    return json.loads(session.usage_path.read_text())


def test_pinned_loading_late_usage_zero_output_and_shared_limit(pinned_session):
    session, calls = pinned_session([
        {"usage": {"prompt_tokens": 5, "completion_tokens": 0, "cost": 0,
                   "prompt_tokens_details": {"cached_tokens": 3},
                   "completion_tokens_details": {"reasoning_tokens": 10}}},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 3, "cost": 0.25}},
    ], limit=10)
    session.start()  # Actual 1.5.2 executable and actual TypeScript extension loader.
    assert usage(session)["ready"]
    assert session.command("set_auto_compaction", enabled=False)["success"]
    prompt_settled(session)
    first = usage(session)
    assert first["total_tokens"] == 5 and first["usage_reliable"]
    assert first["cost"] == {"kind": "provider_total", "value": 0, "currency": "USD"}
    prompt_settled(session, "tool follow-up")
    assert session.wait_settled(time.monotonic() + 1) == "token_limit"
    assert usage(session)["total_tokens"] == 11  # Last response may cross the soft limit.
    assert usage(session)["cost"]["value"] == 0.25
    prompt_settled(session, "forbidden next request")
    assert len(json.loads(calls.read_text())) == 2
    assert usage(session)["usage_reliable"]


@pytest.mark.parametrize("plan", [
    {}, {"usage": {}}, {"usage": {"prompt_tokens": 1}},
    {"usage": {"prompt_tokens": 1, "completion_tokens": -1}},
    {"usage": {"prompt_tokens": "1", "completion_tokens": 0}},
    {"usage": {"prompt_tokens": 1.5, "completion_tokens": 0}},
    {"usage": {"prompt_tokens": 1, "completion_tokens": 0}, "truncated": True},
    {"http_error": True},
])
def test_pinned_unknown_blocks_next_request_without_transport_retries(pinned_session, plan):
    session, calls = pinned_session([plan])
    session.start()
    prompt_settled(session)
    assert session.wait_settled(time.monotonic() + 1) == "usage_unknown"
    assert session.api_error
    prompt_settled(session, "must not reach transport")
    assert len(json.loads(calls.read_text())) == 1
    state = usage(session)
    assert not state["usage_reliable"]
    assert state["cost"]["kind"] == "unknown" and state["cost"]["value"] is None


@pytest.mark.parametrize("limit", [40, 100])
def test_pinned_native_compaction_uses_same_budget(pinned_session, limit):
    text = "context " * 16000
    session, calls = pinned_session([
        {"text": text, "usage": {"prompt_tokens": 10, "completion_tokens": 10}},
        {"text": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 10}},
        {"text": "summary", "usage": {"prompt_tokens": 59, "completion_tokens": 1}},
    ], limit=limit)
    session.start()
    assert session.command("set_auto_compaction", enabled=False)["success"]
    prompt_settled(session, "first research turn")
    # Keep the second user turn, so native compaction needs exactly one history summary.
    prompt_settled(session, text)
    compacted = session.command("compact", timeout=30)
    state = usage(session)
    assert compacted["success"] is (limit == 100), compacted
    assert state["requests"] == (3 if limit == 100 else 2)
    assert state["total_tokens"] == limit and state["stop_reason"] == "token_limit"
    # Another compaction must be blocked at the same boundary, not reset the ledger.
    session.command("compact", timeout=30)
    prompt_settled(session, "tool-follow-up after compaction")
    assert len(json.loads(calls.read_text())) == (3 if limit == 100 else 2)


def test_missing_extension_readiness_never_starts_research(pinned_session, monkeypatch):
    session, calls = pinned_session([])
    original = session._argv
    def without_budget():
        argv = original()
        extension = str(Path(__file__).resolve().parents[1] / "pipeline/extensions/token_budget.ts")
        index = argv.index(extension)
        del argv[index - 1:index + 1]
        return argv
    monkeypatch.setattr(session, "_argv", without_budget)
    monkeypatch.setattr(session._budget_ready, "wait", lambda _timeout: False)
    with pytest.raises(PiError, match="did not become ready"):
        session.start()
    assert not calls.exists()
    assert not usage(session)["ready"]
