"""Compact, text-only observation for terminal notebook clients."""

from __future__ import annotations

import json
import re
import time
from html import escape
from pathlib import Path

from .runlog import read_records


def summary(state, report=None):
    """Describe saved facts, keeping missing checks and verdicts unknown."""
    lines = [f"Конвейер: {short(state.get('status', 'нет данных'))}"
             f" · стадия: {short(state.get('stage', 'нет данных'))}"]
    reason = (report or {}).get("stop_reason", state.get("stop_reason")) or {}
    lines.append("Причина остановки: " + short(reason or "нет данных"))
    solution = (report or {}).get("solution")
    if solution is None:
        lines.append("Окончательный вердикт: отсутствует — итоговый отчёт недоступен")
    else:
        verdict = {True: "true", False: "false", None: "null (неопределённо)"}
        lines.append("Полное решение: " + verdict[solution.get("solved")])
    facts = report if report is not None else state
    saved_checks = facts.get("available_checks") or {}
    if saved_checks:
        facts = {**facts, "flag_check": saved_checks.get("flag"),
                 "script_check": saved_checks.get("replay"),
                 "milestones": saved_checks.get("milestones")}
    checks = [(name, facts.get(name)) for name in ("flag_check", "script_check")]
    milestones = facts.get("milestones") or {}
    if milestones.get("reason") == "no_milestones":
        lines.append("Промежуточные проверки: не предусмотрены задачей")
    checks += [(item.get("id", "bonus"), item) for item in milestones.get("checks", [])]
    for name, check in checks:
        if not check:
            lines.append(f"Проверка {name}: нет сохранённого результата (не отрицательный)")
            continue
        lines.append(f"Проверка {name}: {short(check.get('verdict', 'not_evaluated'))}"
                     f" · {short(check.get('reason') or check.get('explanation') or '')}")
        if check.get("predicate"):
            lines.append("  Предикат: " + short(check["predicate"]))
        if check.get("evidence"):
            lines.append("  Свидетельства: " + short(check["evidence"]))
        for attempt in check.get("attempts", []):
            lines.append(f"  Попытка {short(attempt.get('attempt_id', attempt.get('id', '?')))}:"
                         f" {short(attempt.get('verdict', 'not_evaluated'))}"
                         f" · {short(attempt.get('reason', ''))}")
            if attempt.get("cleanup_errors"):
                lines.append("    Очистка (отдельно от вердикта): " + short(attempt["cleanup_errors"]))
        if check.get("basis_attempt_id"):
            lines.append("  Итог основан на попытке: " + short(check["basis_attempt_id"]))
    budgets = facts.get("budgets", state.get("budgets"))
    if budgets is not None:
        lines.append("Бюджеты: " + short(budgets, 500))
    usage = facts.get("usage", state.get("usage"))
    if usage:
        reliable = "подтверждён" if usage.get("usage_reliable") else "неподтверждён / нижняя граница"
        lines.append(f"Токены: {usage.get('total_tokens', 'нет данных')} / "
                     f"{usage.get('token_budget', 'нет данных')} · расход {reliable}")
        cost = usage.get("cost") or {}
        value = cost.get("value")
        lines.append("Стоимость: " + ("нет данных" if value is None else
                     f"{value} {cost.get('currency', '')} ({cost.get('kind', 'unknown')})"))
    else:
        lines.append("Расход токенов и стоимость: нет данных")
    cleanup = facts.get("cleanup_errors") or (facts.get("pipeline") or {}).get("cleanup_errors")
    if cleanup:
        lines.append("Очистка (отдельно от вердикта): " + short(cleanup))
    if state.get("events_incomplete"):
        lines.append("⚠ Журнал событий неполон; последняя запись ещё не подтверждена")
    for error in state.get("recording_errors", []):
        lines.append("⚠ Неполные свидетельства: " + short(error))
    return lines


def _pre(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return "<pre>" + escape(text) + "</pre>"


def short(value, limit=180):
    """Keep agent-controlled text from filling or controlling the terminal."""
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(value))
    value = " ".join(value.split())
    return value[:limit] + ("…" if len(value) > limit else "")


def content(message):
    parts = message.get("content") or []
    if isinstance(parts, str):
        return parts
    return "\n".join(part.get("text", "") for part in parts
                     if isinstance(part, dict) and part.get("type") == "text")


class Observer:
    """Keep a cursor across bounded watch calls; never start or stop a run."""

    def __init__(self, api, run_id, run_dir=None):
        self.api, self.run_id = api, run_id
        self.after = 0
        self.message = ""
        self.calls = {}
        self.current = ""
        self.last_shown = None
        self.initial = True
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.events = []
        self.state = {}
        self.report = None
        self.last_activity = None
        self.last_summary = None
        self.warnings = []

    def update(self, events):
        steps = []
        for event in events:
            if (not isinstance(event, dict) or type(event.get("seq")) is not int
                    or not isinstance(event.get("type"), str)):
                self.warnings.append("Неполные свидетельства: повреждённое событие")
                continue
            seq = event["seq"]
            if seq <= self.after:
                continue
            if event.get("run_id", self.run_id) != self.run_id:
                steps.append("⚠ Событие другого прогона не принято")
                continue
            if seq != self.after + 1:
                steps.append(f"⚠ Пропуск событий после #{self.after}")
            self.after = seq
            self.events.append(event)
            self.last_activity = {"seq": seq, "ts": event.get("ts", "время неизвестно"),
                                  "type": event.get("type"), "source": event.get("source")}
            kind, data = event["type"], event.get("data") or {}
            if not isinstance(data, dict):
                steps.append(f"⚠ Неполные свидетельства: повреждённые данные события #{seq}")
                continue
            if event.get("truncated"):
                steps.append("⚠ Запись обрезана")
            if event.get("incomplete"):
                steps.append(f"⚠ Неполные свидетельства в событии #{seq}")
            if kind == "message_start" and (data.get("message") or {}).get("role") == "assistant":
                self.message = ""
                self.current = "Агент готовит ответ"
            elif kind == "message_update":
                update = data.get("assistantMessageEvent") or {}
                if update.get("type") == "text_delta":
                    self.message = (self.message + update.get("delta", ""))[-1000:]
                    self.current = "Агент: " + short(self.message[-180:])
                elif update.get("type") == "toolcall_start":
                    self.current = "Агент готовит вызов " + short(update.get("toolName", "инструмента"))
            elif kind == "message_end" and (data.get("message") or {}).get("role") == "assistant":
                message = data["message"]
                final = content(message) or self.message
                if final.strip():
                    steps.append("Агент: " + short(final))
                if message.get("errorMessage"):
                    steps.append("⚠ Ошибка модели: " + short(message["errorMessage"]))
                self.message = ""
                self.current = "Ожидаем следующий шаг агента"
            elif kind == "tool_execution_start":
                key = data.get("toolCallId", seq)
                args = data.get("args") or {}
                detail = args.get("command") or args.get("path") or args.get("file_path") or ""
                label = f"{data.get('toolName', 'инструмент')}: {short(detail)}"
                self.calls[key] = label
                self.current = "Выполняется " + label
            elif kind == "tool_execution_end":
                key = data.get("toolCallId", seq)
                label = self.calls.pop(key, data.get("toolName", "инструмент"))
                result = data.get("result") or {}
                structured = result.get("structuredContent") if isinstance(result, dict) else None
                structured = structured if isinstance(structured, dict) else {}
                failed = data.get("isError") or structured.get("exit_code", 0) != 0
                timed_out = structured.get("timed_out")
                outcome = "тайм-аут" if timed_out else "ошибка" if failed else "готово"
                if not result and not failed and not timed_out:
                    outcome = "завершён; результат не записан"
                output = content(result) if isinstance(result, dict) else str(result)
                preview = f" → {short(output, 120)}" if output.strip() else ""
                note = " · вывод обрезан" if structured.get("truncated") else ""
                steps.append(f"{'⚠' if failed or timed_out or not result else '✓'} {label} — {outcome}{note}{preview}")
                self.current = ("Выполняется " + list(self.calls.values())[-1] if self.calls
                                else "Ожидаем следующий шаг агента")
            elif kind.startswith("stage."):
                self.current = "Стадия: " + short(kind.removeprefix("stage."))
                steps.append(self.current)
            elif kind == "submission_collected":
                steps.append("Итоговая сдача сохранена")
            elif kind == "agent_finished":
                self.current = "Исследование завершено; проверки могут ещё выполняться"
                steps.append("Исследование остановлено: " + short(data.get("stop_reason", "нет данных")))
            elif kind == "cleanup_errors":
                steps.append("⚠ Ошибка очистки (не отменяет вердикт): " + short(data))
            elif kind in {"run_failed", "run_stopped", "protocol_error"}:
                steps.append("⚠ " + short(kind + ": " + str(data)))
        return steps

    def _saved(self):
        """Read a local package without requiring report.json or a running server."""
        assert self.run_dir is not None
        def load(name, default=None):
            path = self.run_dir / name
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.warnings.append(f"Неполные свидетельства {name}: {exc}")
                return default

        state = load("status.json", dict(self.state)) or {}
        state["live_owned"] = False
        passport = load("passport.json", {}) or {}
        state["viewer"] = {"task": passport.get("task", {}).get("id"),
                           "model": passport.get("agent", {}).get("model")}
        state["budgets"] = passport.get("budgets", passport.get("limits"))
        state["usage"] = load("usage.json")
        for key, path in (("flag_check", "checks/flag.json"), ("script_check", "checks/replay.json"),
                          ("milestones", "checks/milestones.json")):
            state[key] = load(path)
        self.report = load("report.json", self.report)
        records, errors = read_records(self.run_dir / "events.jsonl")
        self.warnings.extend("Неполные свидетельства: " + short(error) for error in errors)
        return state, records

    def details_html(self):
        """Disclose full local evidence on demand, escaping all agent-controlled text."""
        blocks = ["<p>Локальные исследовательские данные; не публикуйте без проверки. "
                  "Сообщения агента — заявления, не независимые доказательства.</p>"]
        raw, raw_errors, raw_lines = [], [], []
        if self.run_dir is not None:
            raw, raw_errors = read_records(self.run_dir / "raw/rpc.jsonl", raw_lines)
            _, event_errors = read_records(self.run_dir / "events.jsonl")
            raw_errors += event_errors
        raw_by_line = dict(zip(raw_lines, raw))
        for event in self.events:
            title = (f"#{event['seq']} · {event.get('ts', 'время неизвестно')} · "
                     f"{event.get('source', '?')} · {event.get('type', '?')}")
            blocks.append("<details><summary>" + escape(title) + "</summary>")
            if event.get("incomplete") or event.get("truncated"):
                blocks.append("<p>Неполные / обрезанные свидетельства; доступная запись ниже.</p>")
            blocks.append(_pre(event))
            refs = event.get("raw_refs") or (
                [{"path": "raw/rpc.jsonl", "line": event["raw_line"]}] if event.get("raw_line") else [])
            for ref in refs:
                blocks.append("<p>Исходное свидетельство: " + escape(str(ref)) + "</p>")
                if self.run_dir is not None and ref.get("path") == "raw/rpc.jsonl":
                    record = raw_by_line.get(ref.get("line"))
                    blocks.append(_pre(record) if record is not None else
                                  "<p>Исходное свидетельство отсутствует / недоступно.</p>")
                elif self.run_dir is None:
                    blocks.append("<p>Полная исходная запись доступна в локальном пакете RUN_DIR.</p>")
            blocks.append("</details>")
        if not self.events:
            blocks.append("<p>Сохранённые события отсутствуют; это не результат проверки.</p>")
        for label, value in (("Состояние, проверки, бюджеты и usage", self.state),
                             ("Итоговый отчёт", self.report)):
            blocks.append("<details><summary>" + label + "</summary>" + _pre(value) + "</details>")
        warnings = self.warnings + (["Неполные исходные свидетельства: " + str(raw_errors)]
                                    if raw_errors else [])
        if warnings:
            blocks.append(_pre(warnings))
        if self.run_dir is not None:
            observations = self.run_dir / "observations.md"
            blocks.append("<p>Журнал наблюдений (не доказательство): " +
                          escape(str(observations)) + "</p>")
            if observations.exists():
                try:
                    blocks.append("<details><summary>Журнал наблюдений; при аварии может быть неполным"
                                  "</summary>" + _pre(observations.read_text(encoding="utf-8"))
                                  + "</details>")
                except OSError as exc:
                    blocks.append(_pre(f"Журнал недоступен: {exc}"))
        return "\n".join(blocks)

    def watch(self, seconds=20):
        """Watch boundedly; retain readable evidence when connection or report is absent."""
        deadline = time.monotonic() + seconds
        self.last_shown = self.last_summary = None
        while True:
            connected, events = self.api is not None, []
            if self.api is not None:
                try:
                    state = self.api(f"/runs/{self.run_id}")
                    if state.get("error") and "status" not in state:
                        raise RuntimeError(state["error"])
                    self.state = state
                    events = self.api(f"/runs/{self.run_id}/events?after={self.after}")
                except Exception as exc:
                    connected = False
                    print("Связь потеряна: " + short(exc))
            if self.run_dir is not None and (self.api is None or not connected):
                self.state, events = self._saved()
            steps = self.update(events)
            if self.initial:
                info = self.state.get("viewer") or {}
                print(f"{short(info.get('task') or self.run_id)} · {short(info.get('model') or '')}")
                print("Последние шаги:")
                steps = steps[-10:]
                self.initial = False
            for step in steps:
                print(step)
            finished = self.state.get("status") in {"done", "failed"}
            stale = not connected or self.state.get("live_owned") is False
            if finished:
                current = f"Завершено: {self.state['status']}"
            elif stale:
                current = ("Неподтверждённое сохранённое состояние: "
                           + short(self.state.get("status", "нет данных")))
            else:
                current = self.current or ("Подготовка" if self.state.get("stage", "").startswith("prepare")
                                           else "Ожидаем действия агента")
            if current != self.last_shown:
                print("Сейчас: " + current)
                self.last_shown = current
                if not stale:
                    activity = self.last_activity or self.state.get("last_activity")
                    print("Последняя активность: " + short(activity or "нет данных", 500))
            if stale:
                activity = self.last_activity or self.state.get("last_activity")
                print("Последняя известная активность: " + short(activity or "нет данных", 500))
                if self.current:
                    print("Последнее действие (не подтверждено сейчас): " + self.current)
            if finished and connected and self.api is not None:
                try:
                    candidate = self.api(f"/runs/{self.run_id}/report")
                    if candidate.get("error"):
                        raise RuntimeError(candidate["error"])
                    self.report = candidate
                except Exception as exc:
                    print("Отчёт недоступен: " + short(exc))
            lines = summary(self.state, self.report)
            if lines != self.last_summary:
                print("\n".join(lines))
                self.last_summary = lines
            for warning in dict.fromkeys(self.warnings):
                print("⚠ " + short(warning, 500))
            if finished or stale:
                return
            if time.monotonic() >= deadline:
                print("Наблюдение приостановлено; повторите ячейку для обновления состояния.")
                return
            time.sleep(min(3, max(0, deadline - time.monotonic())))
