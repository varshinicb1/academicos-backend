"""Structured telemetry for outbound provider calls.

The gap this closes
-------------------
Before this, a failed LLM call left one `log.warning` with a string and nothing
else. There was no record of how long the call took, how many attempts it
consumed, how large the prompt was, or how often it failed -- which means there
was no way to answer "did quality drop after Tuesday's prompt change", or "is
one endpoint burning the budget", or "is the provider's p95 degrading". Those
are the questions agent observability exists to answer, and none of them are
answerable from a log line that is discarded at the end of the deploy.

The three levels of agent evaluation that the field has converged on are
outcome, trajectory and component. This module is the **component** layer: one
record per provider call, which is the unit every deeper question is built on.
It deliberately does not attempt trajectory or outcome evaluation -- those
belong with a versioned golden dataset, and inventing a half-version of them
here would be worse than leaving the seam clean.

Design choices
--------------
* **No vendor, no dependency.** Observability that requires a paid endpoint and
  a network round-trip does not get switched on in a project like this one. A
  bounded in-process ring buffer plus an optional JSONL file is enough to debug
  a regression and to feed a future eval harness. Swapping in OpenTelemetry
  later is a change to `record()` alone.
* **Prompts are not recorded by default.** This system handles student data
  under DPDP, so capturing raw prompt text by default would create a new store
  of personal data purely as a side effect of observability. We record *shape*
  (character counts, attempt counts, timings, outcomes). Callers that need raw
  text for a specific investigation can opt in explicitly.
* **Bounded.** A ring buffer that grows without limit is an outage that takes
  hours to arrive.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

DEFAULT_RING_SIZE = 500


@dataclass(frozen=True)
class LLMCallRecord:
    """One provider call. Shape and outcome only -- never prompt content."""

    ts: float
    operation: str
    model: str
    attempt: int
    ok: bool
    latency_ms: float
    status_code: int | None = None
    prompt_chars: int = 0
    completion_chars: int = 0
    error: str | None = None
    endpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def retried(self) -> bool:
        """True when this call needed more than the first attempt."""
        return self.attempt > 0


class LLMTelemetry:
    """Bounded, thread-safe record of provider calls.

    Not a metrics system. It answers "what just happened" and "what has been
    happening lately" for one process, which is the question that actually gets
    asked during an incident.
    """

    def __init__(self, *, ring_size: int = DEFAULT_RING_SIZE, sink: Path | None = None) -> None:
        self._records: deque[LLMCallRecord] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._sink = sink
        self._totals = {"calls": 0, "failures": 0, "retries": 0}

    def record(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._totals["calls"] += 1
            if not record.ok:
                self._totals["failures"] += 1
            if record.retried:
                self._totals["retries"] += 1
        if self._sink is not None:
            self._write_sink(record)

    def _write_sink(self, record: LLMCallRecord) -> None:
        """Append one JSON line. Never raises: telemetry must not break a request.

        A full disk or a bad path is a reason to lose telemetry, not a reason to
        fail a student's paper generation.
        """
        try:
            self._sink.parent.mkdir(parents=True, exist_ok=True)
            with self._sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
        except OSError as exc:  # pragma: no cover - environment dependent
            log.warning("LLM telemetry sink write failed (%s)", exc)

    def recent(self, n: int = 20, *, operation: str | None = None) -> list[LLMCallRecord]:
        with self._lock:
            items: Iterable[LLMCallRecord] = list(self._records)
        if operation is not None:
            items = [r for r in items if r.operation == operation]
        return list(items)[-n:]

    def summary(self) -> dict[str, Any]:
        """Aggregate over the retained window, plus lifetime counters.

        Percentiles come from the retained window only -- the ring buffer is
        bounded precisely so this stays O(ring_size) and cannot become the
        expensive part of a health check.
        """
        with self._lock:
            records = list(self._records)
            totals = dict(self._totals)

        if not records:
            return {**totals, "window": 0, "p50_ms": None, "p95_ms": None, "failure_rate": None}

        latencies = sorted(r.latency_ms for r in records)

        def pct(p: float) -> float:
            idx = min(len(latencies) - 1, int(round((p / 100.0) * (len(latencies) - 1))))
            return round(latencies[idx], 1)

        failures = sum(1 for r in records if not r.ok)
        by_operation: dict[str, int] = {}
        for r in records:
            by_operation[r.operation] = by_operation.get(r.operation, 0) + 1

        return {
            **totals,
            "window": len(records),
            "window_failures": failures,
            "failure_rate": round(failures / len(records), 4),
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "by_operation": by_operation,
        }

    def reset(self) -> None:
        """Clear the retained window and counters (test isolation)."""
        with self._lock:
            self._records.clear()
            self._totals = {"calls": 0, "failures": 0, "retries": 0}


def default_sink() -> Path | None:
    """Where to persist call records, if anywhere.

    Off unless `ACOS_LLM_TELEMETRY_PATH` is set. Persisting by default would
    turn every deploy into a growing file nobody rotates.
    """
    raw = os.environ.get("ACOS_LLM_TELEMETRY_PATH", "").strip()
    return Path(raw) if raw else None


# Process-wide instance. One ring buffer is the right granularity: the question
# being asked is always about this process's recent behaviour.
TELEMETRY = LLMTelemetry(sink=default_sink())


def now_ms() -> float:
    """Monotonic milliseconds, for measuring a call's duration."""
    return time.monotonic() * 1000.0
