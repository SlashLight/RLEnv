"""Compact, text-only observation for terminal notebook clients."""

from __future__ import annotations

import re
import time


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

    def __init__(self, api, run_id):
        self.api, self.run_id = api, run_id
        self.after = 0
        self.message = ""
        self.calls = {}
        self.current = ""
        self.last_shown = None
        self.initial = True

    def update(self, events):
        steps = []
        for event in events:
            seq = event["seq"]
            if seq <= self.after:
                continue
            if seq != self.after + 1:
                steps.append(f"⚠ Пропуск событий после #{self.after}")
            self.after = seq
            kind, data = event["type"], event.get("data") or {}
            if event.get("truncated"):
                steps.append("⚠ Запись обрезана")
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
                output = content(result) if isinstance(result, dict) else str(result)
                preview = f" → {short(output, 120)}" if output.strip() else ""
                note = " · вывод обрезан" if structured.get("truncated") else ""
                steps.append(f"{'⚠' if failed or timed_out else '✓'} {label} — {outcome}{note}{preview}")
                self.current = ("Выполняется " + list(self.calls.values())[-1] if self.calls
                                else "Ожидаем следующий шаг агента")
            elif kind in {"run_failed", "run_stopped", "protocol_error"}:
                steps.append("⚠ " + short(kind + ": " + str(data)))
        return steps

    def watch(self, seconds=20):
        """Watch for a bounded interval so the kernel becomes available for Stop."""
        deadline = time.monotonic() + seconds
        while True:
            state = self.api(f"/runs/{self.run_id}")
            if state.get("error") and "status" not in state:
                raise RuntimeError(state["error"])
            events = self.api(f"/runs/{self.run_id}/events?after={self.after}")
            steps = self.update(events)
            if self.initial:
                info = state.get("viewer") or {}
                print(f"{short(info.get('task') or self.run_id)} · {short(info.get('model') or '')}")
                print("Последние шаги:")
                steps = steps[-10:]
                self.initial = False
            for step in steps:
                print(step)
            finished = state["status"] in {"done", "failed"}
            current = (f"Завершено: {state['status']}" if finished else
                       self.current or ("Подготовка" if state.get("stage", "").startswith("prepare")
                                        else "Ожидаем действия агента"))
            if current != self.last_shown:
                print("Сейчас: " + current)
                self.last_shown = current
            if finished:
                try:
                    report = self.api(f"/runs/{self.run_id}/report")
                    print("Итог: " + short((report.get("solution") or {}).get("status", "нет отчёта")))
                except Exception as exc:
                    print("Отчёт недоступен: " + short(exc))
                return
            if time.monotonic() >= deadline:
                print("Наблюдение приостановлено; прогон продолжается. Повторите ячейку для обновления.")
                return
            time.sleep(min(3, max(0, deadline - time.monotonic())))
