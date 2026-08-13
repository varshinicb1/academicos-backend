"""GradedStore: SQLite-backed persistence for evaluated answer sheets.

Same problem as PaperStore (see paper_store.py's docstring): graded answer
sheets used to live only in the process-memory `_graded` dict in
pillar_routes.py, keyed by assessment_id -> student_id -> [(question,
evaluation), ...]. On Render's free tier the container restarts after ~15
minutes idle, so every graded sheet vanished -- Teacher Insights
(`/insights/class/{id}`) and Principal Insights (`/insights/school/{id}`)
would silently go blank/404 after any restart, even though the underlying
mastery data (KnowledgeStore, which *is* durable) was untouched.

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
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, assessment_id: str, student_id: str, graded: Graded) -> None:
        blob = json.dumps([[q.model_dump(mode="json"), asdict(e)] for q, e in graded])
        self.conn.execute(
            """INSERT INTO graded (assessment_id, student_id, graded_json) VALUES (?, ?, ?)
               ON CONFLICT(assessment_id, student_id) DO UPDATE SET
                 graded_json=excluded.graded_json""",
            (assessment_id, student_id, blob),
        )
        self.conn.commit()

    def _decode(self, blob: str) -> Graded:
        return [(QuestionSchema.model_validate(q), _evaluation_from_dict(e))
                for q, e in json.loads(blob)]

    def get(self, assessment_id: str, student_id: str) -> Optional[Graded]:
        row = self.conn.execute(
            "SELECT graded_json FROM graded WHERE assessment_id=? AND student_id=?",
            (assessment_id, student_id),
        ).fetchone()
        return self._decode(row["graded_json"]) if row else None

    def for_assessment(self, assessment_id: str) -> dict[str, Graded]:
        rows = self.conn.execute(
            "SELECT student_id, graded_json FROM graded WHERE assessment_id=?",
            (assessment_id,),
        ).fetchall()
        return {row["student_id"]: self._decode(row["graded_json"]) for row in rows}

    def all_student_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT student_id FROM graded").fetchall()
        return [row["student_id"] for row in rows]

    def assessment_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(DISTINCT assessment_id) AS n FROM graded").fetchone()
        return row["n"]
