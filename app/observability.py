"""Structured observability and audit trail.

Kept deliberately separate from the business logic. Nothing in `app/agents/`,
`app/guardrail.py`, `app/retrieval.py` or `app/contracts.py` imports this module;
instrumentation is attached at the composition points - the graph builder and an
HTTP middleware - so the decision code stays readable and testable on its own.

One line of JSON per event, appended to a file. That is enough to reconstruct any
request end to end, and it needs no database.

What is deliberately **not** recorded:

* The staff question text, the drafted answer, and evidence excerpts. A question
  may contain customer detail a staff member pasted in. The log keeps a length and
  a short digest instead, which is enough to correlate and to spot duplicates
  without holding the content.
* Anything key-shaped. Every string value is passed through `redact()` on the way
  out, so a provider error quoting a credential cannot land in the log file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

# Anything shaped like a provider credential never reaches the log or the wire.
_KEY_SHAPED = re.compile(r"\b(?:gsk|sk|xai|ghp|github_pat)[-_][A-Za-z0-9_\-]{16,}\b")

_current_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def redact(text: str, api_key: str | None = None) -> str:
    """Strip a known key and anything key-shaped from a string."""
    if api_key and api_key in text:
        text = text.replace(api_key, "[redacted]")
    return _KEY_SHAPED.sub("[redacted]", text)


def new_trace_id() -> str:
    """A request identifier. Short enough to read in a terminal."""
    return uuid4().hex


def current_trace_id() -> str | None:
    return _current_trace_id.get()


@contextmanager
def trace(trace_id: str) -> Iterator[str]:
    """Bind a trace id for the duration of a request, including nested calls."""
    token = _current_trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _current_trace_id.reset(token)


def digest(text: str) -> str:
    """A short, stable fingerprint. Correlates without storing the content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class EventLog:
    """Append-only JSONL event sink.

    With no path it keeps records in memory, which is what the tests use.
    """

    def __init__(self, path: Path | None = None, *, enabled: bool = True, api_key: str = "") -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled
        self.api_key = api_key
        self.records: list[dict[str, Any]] = []
        self.max_records = 5000  # the in-memory tail is bounded; the file is the record
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        component: str,
        event: str,
        status: str = "ok",
        *,
        trace_id: str | None = None,
        latency_ms: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Record one event. Never raises: observability must not break a request."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": trace_id or current_trace_id() or "unbound",
            "component": component,
            "event": event,
            "status": status,
        }
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 2)
        record.update({k: v for k, v in fields.items() if v is not None})

        record = self._scrub(record)
        if not self.enabled:
            return record

        with self._lock:
            self.records.append(record)
            if len(self.records) > self.max_records:
                del self.records[: len(self.records) - self.max_records]
            if self.path is not None:
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, default=str) + "\n")
                except OSError:  # a full disk must not take the assistant down
                    pass
        return record

    def _scrub(self, record: dict[str, Any]) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return redact(value, self.api_key)
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return value

        return {k: clean(v) for k, v in record.items()}

    def events_for(self, trace_id: str) -> list[dict[str, Any]]:
        """Every event recorded for one request, oldest first.

        Prefers the file, which is the durable record; falls back to the
        in-memory tail when the log is running without a path.
        """
        if self.path is not None and self.path.exists():
            return read_events(self.path, trace_id)
        with self._lock:
            return [r for r in self.records if r.get("trace_id") == trace_id]

    @contextmanager
    def span(self, component: str, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Emit `started`, then `completed` or `failed` with the elapsed time."""
        self.emit(component, event, status="started", **fields)
        started = perf_counter()
        extra: dict[str, Any] = {}
        try:
            yield extra
        except Exception as exc:
            self.emit(
                component,
                event,
                status="failed",
                latency_ms=(perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                **extra,
            )
            raise
        else:
            self.emit(
                component,
                event,
                status="completed",
                latency_ms=(perf_counter() - started) * 1000,
                **extra,
            )


_log: EventLog | None = None


def configure(path: Path | None = None, *, enabled: bool = True, api_key: str = "") -> EventLog:
    """Install the process-wide log. Called once at startup."""
    global _log
    _log = EventLog(path, enabled=enabled, api_key=api_key)
    return _log


def get_event_log() -> EventLog:
    """The process-wide log, built from settings on first use."""
    global _log
    if _log is None:
        from app.config import get_settings

        settings = get_settings()
        _log = EventLog(
            settings.observability_path if settings.observability_enabled else None,
            enabled=settings.observability_enabled,
            api_key=settings.groq_api_key,
        )
    return _log


def reset() -> None:
    """Drop the configured log. For tests."""
    global _log
    _log = None


class ObservedRetriever:
    """Wraps any `PolicyRetriever` and records what it returned.

    Implements the same protocol, so it substitutes anywhere the real retriever
    goes and the Policy Agent is unaware of it.
    """

    def __init__(self, inner: Any, log: EventLog | None = None) -> None:
        self._inner = inner
        self._log = log or get_event_log()

    def search(self, query: str, *, limit: int | None = None) -> Any:
        started = perf_counter()
        result = self._inner.search(query, limit=limit)
        latency = (perf_counter() - started) * 1000

        self._log.emit(
            "retrieval",
            "policy_retrieval",
            status="sufficient" if result.sufficient else "insufficient",
            latency_ms=latency,
            query_chars=len(query),
            query_digest=digest(query),
            best_score=result.best_score,
            threshold=result.threshold,
            evidence_count=len(result.evidence),
            abstain_reason=result.abstain_reason.value if result.abstain_reason else None,
        )
        for evidence in result.evidence:
            # Ids and sections only. The excerpt itself is not logged.
            self._log.emit(
                "retrieval",
                "evidence_selected",
                status="ok",
                policy_id=evidence.policy_id,
                section=evidence.section,
                score=evidence.score,
            )
        return result


# --------------------------------------------------------------------------- #
# Reading the trail back
# --------------------------------------------------------------------------- #


def read_events(path: Path, trace_id: str | None = None) -> list[dict[str, Any]]:
    """Load events from a JSONL file, optionally for one trace."""
    if not Path(path).exists():
        return []
    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if trace_id is None or record.get("trace_id") == trace_id:
                events.append(record)
    return events


def render_trace(events: list[dict[str, Any]]) -> str:
    """Render one request's events as a readable tree."""
    if not events:
        return "(no events)"

    lines = [events[0].get("trace_id", "unknown")]
    labelled = [_label(e) for e in events]
    labelled = [line for line in labelled if line]
    for i, line in enumerate(labelled):
        connector = "└──" if i == len(labelled) - 1 else "├──"
        lines.append(f"{connector} {line}")
    return "\n".join(lines)


def _label(event: dict[str, Any]) -> str | None:
    """One line per meaningful event.

    The JSONL file keeps everything, including the `started` and `completed`
    lifecycle events and their latencies. The tree keeps only what a person
    reading a trace needs: what was decided, and by whom.
    """
    component, name, state = event["component"], event["event"], event["status"]

    if name == "request_received":
        return "request received"
    if name == "response_returned":
        return "response returned"
    if name == "evidence_selected":
        section = event.get("section", "")
        number = section.split(".", 1)[0].strip()
        return f"retrieval \u2192 {event['policy_id']} \u00a7{number}" if number else f"retrieval \u2192 {event['policy_id']}"
    if name == "policy_retrieval":
        return None if state == "sufficient" else f"retrieval \u2192 {state}"
    if name == "decision":
        return f"{component} \u2192 {state}"
    if state == "failed":
        return f"{component} \u2192 failed: {event.get('error', 'unknown error')}"
    return None  # lifecycle start/end lines stay in the file, not in the tree
