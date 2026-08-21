"""Append-only learner-event store (event sourcing).

Every observed learning interaction is persisted as one immutable event;
the LearnerModel is a projection rebuilt by replay. This gives:
  - durability: interactions survive restarts (the model was in-memory only),
  - auditability: the raw event log is the source of truth,
  - rebuildability: any model state (mastery, forgetting, affect) can be
    recomputed deterministically from the log.

Events are keyed by (learner_id, seq) with a monotonically increasing
per-learner sequence; payloads keep a JSON escape hatch for future event
fields without schema churn. Writes go through the same SQLite WAL used by
the graph store.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..algorithms.learner_model import Interaction, LearnerModel

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
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- writes ----
    def append(self, learner_id: str, i: Interaction,
               payload: Optional[dict[str, Any]] = None) -> tuple[str, int]:
        """Persist one event; returns (event_id, seq)."""
        seq = self._next_seq(learner_id)
        event_id = f"ev:{learner_id}:{seq}"
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
        """Batch append (single transaction); returns number appended."""
        for i in events:
            seq = self._next_seq(learner_id)
            self.conn.execute(
                """INSERT INTO learner_events (learner_id, seq, event_id,
                     kind, concept_id, outcome, bloom, difficulty, affect,
                     duration_sec, ts, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (learner_id, seq, f"ev:{learner_id}:{seq}", i.kind,
                 i.concept_id, i.outcome, i.bloom, i.difficulty, i.affect,
                 i.duration_sec, i.ts, None),
            )
        self.conn.commit()
        return len(events)

    # ---- reads ----
    def events(self, learner_id: str, since_seq: int = 0,
               limit: int = 1000) -> list[Interaction]:
        rows = self.conn.execute(
            """SELECT * FROM learner_events WHERE learner_id=?
               AND seq > ? ORDER BY seq LIMIT ?""",
            (learner_id, since_seq, limit)).fetchall()
        return [_interaction(r) for r in rows]

    def last_seq(self, learner_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) s FROM learner_events WHERE learner_id=?",
            (learner_id,)).fetchone()
        return row["s"]

    def learner_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT learner_id FROM learner_events ORDER BY learner_id"
        ).fetchall()
        return [r["learner_id"] for r in rows]

    def stats(self, learner_id: str) -> dict[str, Any]:
        """Per-learner aggregate: counts by kind, correct answers, span."""
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


def _interaction(r: sqlite3.Row) -> Interaction:
    return Interaction(
        concept_id=r["concept_id"], kind=r["kind"], outcome=r["outcome"],
        bloom=r["bloom"], difficulty=r["difficulty"], affect=r["affect"],
        duration_sec=r["duration_sec"], ts=r["ts"],
    )
