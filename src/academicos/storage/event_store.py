"""Append-only learner-event store (event sourcing).

Every observed learning interaction is persisted as one immutable event;
the LearnerModel is a projection rebuilt by replay. This gives:
  - durability: interactions survive restarts (the model was in-memory only),
  - auditability: the raw event log is the source of truth,
  - rebuildability: any model state (mastery, forgetting, affect) can be
    recomputed deterministically from the log.

Events are keyed by (learner_id, seq) with a monotonically increasing
per-learner sequence; payloads keep a JSON escape hatch for future event
fields without schema churn. Local SQLite writes go through the same WAL
used by the graph store.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- same
fallback pattern as assessment/store.py (see supabase_kv.py's module
docstring: Render's disk is ephemeral, so anything meant to survive a
redeploy needs to live in Postgres instead). This was previously the
biggest gap in that migration -- every other durability-critical store
had it, this one didn't, so a redeploy silently reset every learner's
mastery/FSRS history to zero even though nothing about the data was wrong.

Concurrency, two separate locks for two separate problems:
  - `_lock_for(learner_id)`: `_next_seq` reads MAX(seq) then inserts: without
    serializing that read-then-write, two concurrent appends for the same
    learner (two requests racing in the same process) can compute the same
    seq and violate UNIQUE(learner_id, seq). Scoped per-learner so different
    learners don't block each other. This only serializes within one
    process -- it does not protect against two separate processes/
    containers writing for the same learner at once, which is exactly the
    case Supabase-backed writes are meant to make safe instead (Postgres
    enforces the unique constraint server-side; a losing writer gets a real
    conflict rather than silently overwriting).
  - `_conn_lock`: a plain `sqlite3.Connection` opened with
    `check_same_thread=False` disables Python's *same-thread* check, but the
    underlying driver still isn't safe for two threads to call `execute()`
    on the *same* Connection object at the same time (confirmed: without
    this lock, concurrent appends for two *different* learners -- so past
    the per-learner lock above -- intermittently raised
    `sqlite3.InterfaceError: bad parameter or other API misuse`). This lock
    guards every local SQLite access regardless of which learner it's for;
    it does not apply to the Supabase branch, where each thread makes its
    own independent HTTP request.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import requests

from ..algorithms.learner_model import Interaction, LearnerModel
from ..assessment.supabase_kv import SupabaseTable

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS learner_events (
  learner_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  event_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  outcome REAL,
  bloom TEXT,
  difficulty REAL,
  affect TEXT,
  duration_sec REAL,
  ts TEXT NOT NULL,
  payload TEXT,
  UNIQUE(learner_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_learner ON learner_events(learner_id, seq);
"""


class EventStore:
    """Append-only event log for one or many learners."""

    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("learner_events")
        self._locks_guard = threading.Lock()
        self._learner_locks: dict[str, threading.Lock] = {}
        self._conn_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _lock_for(self, learner_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._learner_locks.get(learner_id)
            if lock is None:
                lock = threading.Lock()
                self._learner_locks[learner_id] = lock
            return lock

    # ---- writes ----
    def append(self, learner_id: str, i: Interaction,
               payload: Optional[dict[str, Any]] = None) -> tuple[str, int]:
        """Persist one event; returns (event_id, seq)."""
        with self._lock_for(learner_id):
            return self._append_locked(learner_id, i, payload)

    def _append_locked(self, learner_id: str, i: Interaction,
                        payload: Optional[dict[str, Any]]) -> tuple[str, int]:
        seq = self._next_seq(learner_id)
        event_id = f"ev:{learner_id}:{seq}"
        if self._remote.enabled:
            try:
                self._remote.upsert({
                    "learner_id": learner_id, "seq": seq, "event_id": event_id,
                    "kind": i.kind, "concept_id": i.concept_id, "outcome": i.outcome,
                    "bloom": i.bloom, "difficulty": i.difficulty, "affect": i.affect,
                    "duration_sec": i.duration_sec, "ts": i.ts,
                    "payload": payload or None,
                }, on_conflict="event_id")
                return event_id, seq
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for event %s",
                    event_id, exc_info=True,
                )
        with self._conn_lock:
            self.conn.execute(
                """INSERT INTO learner_events (learner_id, seq, event_id, kind,
                     concept_id, outcome, bloom, difficulty, affect, duration_sec,
                     ts, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (learner_id, seq, event_id, i.kind, i.concept_id, i.outcome,
                 i.bloom, i.difficulty, i.affect, i.duration_sec, i.ts,
                 json.dumps(payload) if payload else None),
            )
            self.conn.commit()
        return event_id, seq

    def append_many(self, learner_id: str,
                    events: list[Interaction]) -> int:
        """Batch append; returns number appended.

        Holds the per-learner lock for the whole batch so seq assignment
        stays correct even if another append() for the same learner is
        racing this call -- not a single SQL transaction when Supabase is
        enabled (each row is its own REST call), but atomic with respect to
        seq numbering either way.
        """
        with self._lock_for(learner_id):
            for i in events:
                self._append_locked(learner_id, i, None)
        return len(events)

    # ---- reads ----
    def events(self, learner_id: str, since_seq: int = 0,
               limit: int = 1000) -> list[Interaction]:
        if self._remote.enabled:
            try:
                rows = self._remote.select(
                    learner_id=learner_id, order="seq.asc",
                    gt={"seq": since_seq}, limit=limit,
                )
                return [_interaction(r) for r in rows]
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for events(%s)",
                    learner_id, exc_info=True,
                )
        with self._conn_lock:
            rows = self.conn.execute(
                """SELECT * FROM learner_events WHERE learner_id=?
                   AND seq > ? ORDER BY seq LIMIT ?""",
                (learner_id, since_seq, limit)).fetchall()
        return [_interaction(r) for r in rows]

    def last_seq(self, learner_id: str) -> int:
        if self._remote.enabled:
            try:
                rows = self._remote.select(learner_id=learner_id, order="seq.desc", limit=1)
                return rows[0]["seq"] if rows else 0
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for last_seq(%s)",
                    learner_id, exc_info=True,
                )
        with self._conn_lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) s FROM learner_events WHERE learner_id=?",
                (learner_id,)).fetchone()
        return row["s"]

    def learner_ids(self) -> list[str]:
        if self._remote.enabled:
            try:
                rows = self._remote.select(order="learner_id.asc")
                return sorted({r["learner_id"] for r in rows})
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for learner_ids()",
                    exc_info=True,
                )
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT DISTINCT learner_id FROM learner_events ORDER BY learner_id"
            ).fetchall()
        return [r["learner_id"] for r in rows]

    def stats(self, learner_id: str) -> dict[str, Any]:
        """Per-learner aggregate: counts by kind, correct answers, span."""
        if self._remote.enabled:
            try:
                return _stats_from_events(self.events(learner_id, limit=1_000_000))
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for stats(%s)",
                    learner_id, exc_info=True,
                )
        with self._conn_lock:
            row = self.conn.execute(
                """SELECT COUNT(*) n, SUM(kind='answer') answers,
                          SUM(kind='answer' AND outcome >= 0.5) correct,
                          MIN(ts) first_ts, MAX(ts) last_ts
                   FROM learner_events WHERE learner_id=?""",
                (learner_id,)).fetchone()
            kinds = dict(self.conn.execute(
                "SELECT kind, COUNT(*) FROM learner_events WHERE learner_id=?"
                " GROUP BY kind", (learner_id,)).fetchall())
        return {
            "events": row["n"],
            "answers": row["answers"] or 0,
            "correct": row["correct"] or 0,
            "kinds": kinds,
            "first_ts": row["first_ts"],
            "last_ts": row["last_ts"],
        }

    # ---- event sourcing ----
    def replay(self, learner_id: str,
               model: Optional[LearnerModel] = None) -> LearnerModel:
        """Rebuild a LearnerModel by replaying the log in seq order.

        Pass an existing model to continue where a previous replay left off
        (e.g. after incremental appends); otherwise a fresh model is built
        and the full log is replayed deterministically.
        """
        m = model or LearnerModel(learner_id=learner_id)
        since = 0 if model is None else sum(
            len(cs.history) for cs in m.concepts.values())
        for i in self.events(learner_id, since_seq=since):
            m.observe(i)
        return m

    def close(self) -> None:
        self.conn.close()

    def _next_seq(self, learner_id: str) -> int:
        return self.last_seq(learner_id) + 1


def _interaction(r: Any) -> Interaction:
    """Accepts a sqlite3.Row or a plain dict (Supabase REST response) --
    both support `r["column"]`."""
    return Interaction(
        concept_id=r["concept_id"], kind=r["kind"], outcome=r["outcome"],
        bloom=r["bloom"], difficulty=r["difficulty"], affect=r["affect"],
        duration_sec=r["duration_sec"], ts=r["ts"],
    )


def _stats_from_events(events: list[Interaction]) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    answers = correct = 0
    first_ts = last_ts = None
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
        if e.kind == "answer":
            answers += 1
            if e.outcome is not None and e.outcome >= 0.5:
                correct += 1
        if first_ts is None or e.ts < first_ts:
            first_ts = e.ts
        if last_ts is None or e.ts > last_ts:
            last_ts = e.ts
    return {
        "events": len(events), "answers": answers, "correct": correct,
        "kinds": kinds, "first_ts": first_ts, "last_ts": last_ts,
    }
