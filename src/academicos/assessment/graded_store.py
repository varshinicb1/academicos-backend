"""GradedStore: persistence for evaluated answer sheets.

Same problem as PaperStore (see paper_store.py's docstring): graded answer
sheets used to live only in the process-memory `_graded` dict in
pillar_routes.py, keyed by assessment_id -> student_id -> [(question,
evaluation), ...]. On Render's free tier the container's local disk (SQLite
files included) is wiped on every restart/spin-down/redeploy, not just full
redeploys, so every graded sheet vanished -- Teacher Insights
(`/insights/class/{id}`) and Principal Insights (`/insights/school/{id}`)
would silently go blank/404 after any restart, even though the underlying
mastery data (KnowledgeStore, which *is* durable) was untouched.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py.

QuestionSchema is Pydantic (model_dump/model_validate). Evaluation and its
nested MarkingPointOutcome are plain dataclasses, so they're serialized with
dataclasses.asdict() and rebuilt field-by-field on load.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .evaluate import Evaluation, MarkingPointOutcome
from .schemas import QuestionSchema
from .supabase_kv import SupabaseTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS graded (
  assessment_id TEXT NOT NULL,
  student_id    TEXT NOT NULL,
  graded_json   TEXT NOT NULL,
  PRIMARY KEY (assessment_id, student_id)
);
"""

Graded = list[tuple[QuestionSchema, Evaluation]]


def _evaluation_from_dict(d: dict) -> Evaluation:
    d = dict(d)
    d["marking_points"] = [MarkingPointOutcome(**mp) for mp in d.get("marking_points", [])]
    return Evaluation(**d)


class GradedStore:
    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("graded_evaluations")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _encode(self, graded: Graded) -> list:
        return [[q.model_dump(mode="json"), asdict(e)] for q, e in graded]

    def save(self, assessment_id: str, student_id: str, graded: Graded) -> None:
        if self._remote.enabled:
            self._remote.upsert({
                "assessment_id": assessment_id, "student_id": student_id,
                "payload": self._encode(graded),
            }, on_conflict="assessment_id,student_id")
            return
        blob = json.dumps(self._encode(graded))
        self.conn.execute(
            """INSERT INTO graded (assessment_id, student_id, graded_json) VALUES (?, ?, ?)
               ON CONFLICT(assessment_id, student_id) DO UPDATE SET
                 graded_json=excluded.graded_json""",
            (assessment_id, student_id, blob),
        )
        self.conn.commit()

    def _decode(self, blob) -> Graded:
        pairs = blob if isinstance(blob, list) else json.loads(blob)
        return [(QuestionSchema.model_validate(q), _evaluation_from_dict(e)) for q, e in pairs]

    def get(self, assessment_id: str, student_id: str) -> Optional[Graded]:
        if self._remote.enabled:
            rows = self._remote.select(assessment_id=assessment_id, student_id=student_id)
            return self._decode(rows[0]["payload"]) if rows else None
        row = self.conn.execute(
            "SELECT graded_json FROM graded WHERE assessment_id=? AND student_id=?",
            (assessment_id, student_id),
        ).fetchone()
        return self._decode(row["graded_json"]) if row else None

    def for_assessment(self, assessment_id: str) -> dict[str, Graded]:
        if self._remote.enabled:
            rows = self._remote.select(assessment_id=assessment_id)
            return {r["student_id"]: self._decode(r["payload"]) for r in rows}
        rows = self.conn.execute(
            "SELECT student_id, graded_json FROM graded WHERE assessment_id=?",
            (assessment_id,),
        ).fetchall()
        return {row["student_id"]: self._decode(row["graded_json"]) for row in rows}

    def all_student_ids(self) -> list[str]:
        if self._remote.enabled:
            rows = self._remote.select()
            return sorted({r["student_id"] for r in rows})
        rows = self.conn.execute("SELECT DISTINCT student_id FROM graded").fetchall()
        return [row["student_id"] for row in rows]

    def assessment_count(self) -> int:
        if self._remote.enabled:
            rows = self._remote.select()
            return len({r["assessment_id"] for r in rows})
        row = self.conn.execute("SELECT COUNT(DISTINCT assessment_id) AS n FROM graded").fetchone()
        return row["n"]
