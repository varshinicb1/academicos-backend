"""PracticeStore: persistence for generated practice sets.

Same class of bug as PaperStore/GradedStore: `/practice/generate` used to
stash the PracticeSet only in the process-memory `_practice` dict, keyed by
its id. A student can easily spend more than Render's ~15-minute idle window
working through a practice set before calling `/practice/submit` with that
id -- on a restart in between (which wipes local disk too, not just memory),
the id vanishes and the student's completed work 404s ("practice set not
found") instead of updating their mastery.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py.

PracticeSet/PracticeItem are plain dataclasses (remediation.py); QuestionSchema
nested inside PracticeItem is Pydantic.

`school_id` (2026-09-17): real column, store-level metadata only -- PracticeSet
itself has no assessment_id (unlike GeneratedPaper/ScanSession, it's keyed
only by student_id), so unlike those two stores there's no join available to
recover school ownership after the fact. This column IS the only source of
school scoping for this store; callers must supply it at save() time (routes
resolve it from the authenticated caller, per authz.py's
require_school_owns_student).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .remediation import PracticeItem, PracticeSet
from .schemas import QuestionSchema
from .postgres_kv import durable_table

SCHEMA = """
CREATE TABLE IF NOT EXISTS practice_sets (
  id            TEXT PRIMARY KEY,
  school_id     TEXT,
  practice_json TEXT NOT NULL
);
"""


class PracticeStore:
    def __init__(self, db_path: Path):
        self._remote = durable_table("practice_sets")
        self._conn_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """The index lives here, not in SCHEMA -- see PaperStore._migrate()'s
        docstring for why an index in SCHEMA on a column this method might
        still need to add would crash against a real pre-existing db."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(practice_sets)").fetchall()}
        if "school_id" not in cols:
            self.conn.execute("ALTER TABLE practice_sets ADD COLUMN school_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_practice_sets_school ON practice_sets(school_id)")

    def _encode(self, pset: PracticeSet) -> dict:
        return {
            "id": pset.id,
            "student_id": pset.student_id,
            "concept_ids": pset.concept_ids,
            "items": [{"question": i.question.model_dump(mode="json"), "concept_id": i.concept_id}
                      for i in pset.items],
            "answer_key": pset.answer_key,
            "warnings": pset.warnings,
            "created_at": pset.created_at,
        }

    def save(self, pset: PracticeSet, school_id: Optional[str] = None) -> None:
        d = self._encode(pset)
        if self._remote.enabled:
            self._remote.upsert({"id": pset.id, "school_id": school_id, "payload": d}, on_conflict="id")
            return
        with self._conn_lock:
            self.conn.execute(
                """INSERT INTO practice_sets (id, school_id, practice_json) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     school_id=excluded.school_id, practice_json=excluded.practice_json""",
                (pset.id, school_id, json.dumps(d)),
            )
            self.conn.commit()

    def list_by_school(self, school_id: str) -> list[PracticeSet]:
        """For the school-data export route -- see curriculum/routes.py.
        Practice sets saved before this column existed are invisible here;
        there is no join-based fallback (unlike papers/scan sessions) since
        PracticeSet carries no assessment_id."""
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id)
            return [self._decode(r["payload"]) for r in rows]
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT practice_json FROM practice_sets WHERE school_id=?", (school_id,)
            ).fetchall()
        return [self._decode(json.loads(r["practice_json"])) for r in rows]

    def _decode(self, d: dict) -> PracticeSet:
        items = [PracticeItem(question=QuestionSchema.model_validate(i["question"]),
                              concept_id=i["concept_id"])
                for i in d["items"]]
        return PracticeSet(
            id=d["id"], student_id=d["student_id"], concept_ids=d["concept_ids"],
            items=items, answer_key=d["answer_key"], warnings=d["warnings"],
            created_at=d["created_at"],
        )

    def get(self, set_id: str) -> Optional[PracticeSet]:
        if self._remote.enabled:
            rows = self._remote.select(id=set_id)
            return self._decode(rows[0]["payload"]) if rows else None
        with self._conn_lock:
            row = self.conn.execute(
                "SELECT practice_json FROM practice_sets WHERE id=?", (set_id,)
            ).fetchone()
        if row is None:
            return None
        return self._decode(json.loads(row["practice_json"]))
