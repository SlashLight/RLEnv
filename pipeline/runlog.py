"""Run evidence log: raw pi records, normalized events, recording failures."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .util import utcnow, write_json

EVENT_SCHEMA = "evaluator-pipeline/event/v2"
PER_STRING_LIMIT = 131072


def _truncate(value: Any, flags: dict[str, int]) -> Any:
    """Cap long strings without losing the structure needed to follow tool calls."""
    if isinstance(value, str):
        if len(value) > PER_STRING_LIMIT:
            flags["truncated_strings"] = flags.get("truncated_strings", 0) + 1
            flags["dropped_characters"] = flags.get("dropped_characters", 0) + (
                len(value) - PER_STRING_LIMIT
            )
            return value[:PER_STRING_LIMIT] + "\n[pipeline: field truncated]"
        return value
    if isinstance(value, list):
        return [_truncate(item, flags) for item in value]
    if isinstance(value, dict):
        return {key: _truncate(item, flags) for key, item in value.items()}
    return value


class EventLog:
    """Single-writer append-only evidence log.

    `events.jsonl` is the normalized chronology; `raw/rpc.jsonl` keeps the
    original pi records with line numbers that normalized events reference.
    """

    def __init__(self, run_dir: Path, run_id: str, event_limit_bytes: int = 1024 * 1024,
                 *, local_contract: bool = True):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.event_limit_bytes = event_limit_bytes
        self.local_contract = local_contract
        self.seq = 0
        self.raw_line = 0
        self._lock = threading.Lock()
        (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
        self._events = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")
        self._raw = (self.run_dir / "raw" / "rpc.jsonl").open("a", encoding="utf-8")
        self.recording_errors: list[dict[str, Any]] = []

    def publish(self, source: str, kind: str, data: Any = None, **extra: Any) -> dict[str, Any]:
        with self._lock:
            self.seq += 1
            flags: dict[str, int] = {}
            payload = _truncate(data, flags) if data is not None else None
            event: dict[str, Any] = {
                "run_id": self.run_id,
                "seq": self.seq,
                "ts": utcnow(),
                "source": source,
                "type": kind,
                "data": payload,
            }
            if self.local_contract:
                event.update(schema=EVENT_SCHEMA, raw_refs=[], truncated=False,
                             incomplete=bool(self.recording_errors))
            if "raw_line" in extra:
                raw_line = extra.pop("raw_line")
                event["raw_line"] = raw_line
                if self.local_contract:
                    event["raw_refs"].append({"path": "raw/rpc.jsonl", "line": raw_line})
            event.update(extra)
            if flags:
                event["truncated"] = True
                event["truncation"] = flags
            if self.local_contract:
                event["incomplete"] = bool(event["incomplete"] or event["truncated"])
            line = json.dumps(event, ensure_ascii=False)
            if len(line.encode("utf-8")) > self.event_limit_bytes:
                event["data"] = {"_truncated": True, "bytes": len(line.encode("utf-8"))}
                event["truncated"] = True
                event["truncation"] = {**flags, "event_bytes": len(line.encode("utf-8"))}
                if self.local_contract:
                    event["incomplete"] = True
                line = json.dumps(event, ensure_ascii=False)
            try:
                self._events.write(line + "\n")
                self._events.flush()
            except OSError as exc:
                self.recording_error("events", exc)
            return event

    def raw_record(self, record: Any) -> int:
        with self._lock:
            self.raw_line += 1
            line = json.dumps(record, ensure_ascii=False)
            try:
                self._raw.write(line + "\n")
                self._raw.flush()
            except OSError as exc:
                self.recording_error("raw", exc)
            return self.raw_line

    def recording_error(self, stream: str, exc: BaseException) -> None:
        error = {"ts": utcnow(), "stream": stream, "error": f"{type(exc).__name__}: {exc}"}
        self.recording_errors.append(error)
        try:
            (self.run_dir / "raw" / "recorder_errors.log").open("a", encoding="utf-8").write(
                json.dumps(error, ensure_ascii=False) + "\n"
            )
        except OSError:
            pass
        try:
            write_json(self.run_dir / "raw" / "recording_health.json",
                       {"errors": self.recording_errors})
        except OSError:
            pass

    def close(self) -> None:
        for fh in (self._events, self._raw):
            try:
                fh.flush()
                fh.close()
            except OSError:
                pass


def read_records(path: Path, line_numbers: list[int] | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read available JSONL records without hiding damaged or missing evidence."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        with path.open("rb") as fh:
            for number, line in enumerate(fh, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not an object")
                    records.append(record)
                    if line_numbers is not None:
                        line_numbers.append(number)
                except (ValueError, UnicodeDecodeError) as exc:
                    errors.append({"stream": path.name, "line": number,
                                   "error": str(exc)})
    except OSError as exc:
        errors.append({"stream": path.name, "error": str(exc)})
    return records, errors


class EventCursor:
    """One polling cursor; rewinds for older requests or a replaced/truncated journal."""

    def __init__(self):
        self._lock = threading.Lock()
        self._identity = None
        self._offset = 0
        self._last_seq = 0
        self._anchor = b""
        self.recording_errors: list[dict[str, Any]] = []
        self.incomplete_tail = False

    def read(self, run_dir: Path, after: int) -> list[dict[str, Any]]:
        with self._lock:
            try:
                with (run_dir / "events.jsonl").open("rb") as fh:
                    self.incomplete_tail = False
                    stat = os.fstat(fh.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    fh.seek(max(0, self._offset - len(self._anchor)))
                    anchor = fh.read(len(self._anchor))
                    if (identity != self._identity or stat.st_size < self._offset
                            or after < self._last_seq or anchor != self._anchor):
                        self._offset = self._last_seq = 0
                        self.recording_errors = []
                    self._identity = identity
                    fh.seek(self._offset)
                    events = []
                    while line := fh.readline():
                        # The writer may still be appending this record.
                        if not line.endswith(b"\n"):
                            self.incomplete_tail = True
                            break
                        self._offset = fh.tell()
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError) as exc:
                            self.recording_errors.append({
                                "stream": "events.jsonl", "offset": self._offset - len(line),
                                "error": str(exc),
                            })
                            continue
                        if isinstance(event, dict) and type(event.get("seq")) is int:
                            self._last_seq = max(self._last_seq, event["seq"])
                            if event["seq"] > after:
                                events.append(event)
                        else:
                            self.recording_errors.append({
                                "stream": "events.jsonl", "offset": self._offset - len(line),
                                "error": "event is not an object with an integer sequence",
                            })
                    fh.seek(max(0, self._offset - 256))
                    self._anchor = fh.read(min(256, self._offset))
                    return events
            except OSError as exc:
                self._identity = None
                self.recording_errors = [{"stream": "events.jsonl", "error": str(exc)}]
                self.incomplete_tail = True
                return []


def read_events(run_dir: Path, after: int = 0) -> list[dict[str, Any]]:
    records, _ = read_records(Path(run_dir) / "events.jsonl")
    return [event for event in records
            if isinstance(event.get("seq"), int) and event["seq"] > after]
