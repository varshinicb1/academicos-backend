"""AssessmentStore: persistence for the central Assessment object.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py. This is the most
critical of the app's Render-ephemeral-disk stores: without it surviving a
restart, `/assessments` comes back empty and every assessment a teacher
created simply disappears, even though nothing about their data was
actually wrong -- the same class of bug as graded_store.py/paper_store.py
but for the object that ties everything else together.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .schemas import Assessment
from .supabase_kv import SupabaseTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
  id                    TEXT PRIMARY KEY,
  school_id             TEXT NOT NULL,
  teacher_id            TEXT NOT NULL,
  title                 TEXT NOT NULL,
  subject               TEXT NOT NULL,
  grade                 INTEGER NOT NULL,
  chapter_ids           TEXT,   -- json list
  blueprint             TEXT,   -- json Blueprint
  status                TEXT NOT NULL,
  created_at            TEXT NOT NULL,
  updated_at            TEXT NOT NULL,
  scheduled_at          TEXT,
  completed_at          TEXT,
  template_id           TEXT,
  metadata              TEXT,   -- json
  selected_question_ids TEXT,   -- json list
  generated_paper_id    TEXT,
  total_students        INTEGER,
  evaluated_count       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_assessments_teacher ON assessments(teacher_id);
CREATE INDEX IF NOT EXISTS idx_assessments_school ON assessments(school_id);
"""


class AssessmentStore:
    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("assessments")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, a: Assessment) -> None:
        if self._remote.enabled:
            self._remote.upsert({
                "id": a.id, "school_id": a.school_id, "teacher_id": a.teacher_id,
                "created_at": a.created_at.isoformat(),
                "payload": json.loads(a.model_dump_json()),
            }, on_conflict="id")
            return
        self.conn.execute(
            """INSERT INTO assessments (id, school_id, teacher_id, title, subject, grade,
                 chapter_ids, blueprint, status, created_at, updated_at, scheduled_at,
                 completed_at, template_id, metadata, selected_question_ids,
                 generated_paper_id, total_students, evaluated_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, subject=excluded.subject, grade=excluded.grade,
                 chapter_ids=excluded.chapter_ids, blueprint=excluded.blueprint,
                 status=excluded.status, updated_at=excluded.updated_at,
                 scheduled_at=excluded.scheduled_at, completed_at=excluded.completed_at,
                 template_id=excluded.template_id, metadata=excluded.metadata,
                 selected_question_ids=excluded.selected_question_ids,
                 generated_paper_id=excluded.generated_paper_id,
                 total_students=excluded.total_students, evaluated_count=excluded.evaluated_count
            """,
            (
                a.id, a.school_id, a.teacher_id, a.title, a.subject, a.grade,
                json.dumps(a.chapter_ids), a.blueprint.model_dump_json(), a.status,
                a.created_at.isoformat(), a.updated_at.isoformat(),
                a.scheduled_at.isoformat() if a.scheduled_at else None,
                a.completed_at.isoformat() if a.completed_at else None,
                a.template_id, json.dumps(a.metadata), json.dumps(a.selected_question_ids),
                a.generated_paper_id, a.total_students, a.evaluated_count,
            ),
        )
        self.conn.commit()

    def get(self, assessment_id: str) -> Optional[Assessment]:
        if self._remote.enabled:
            rows = self._remote.select(id=assessment_id)
            return Assessment.model_validate(rows[0]["payload"]) if rows else None
        row = self.conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
        return _row_to_assessment(row) if row else None

    def list_by_teacher(self, teacher_id: str) -> list[Assessment]:
        if self._remote.enabled:
            rows = self._remote.select(teacher_id=teacher_id, order="created_at.desc")
            return [Assessment.model_validate(r["payload"]) for r in rows]
        rows = self.conn.execute(
            "SELECT * FROM assessments WHERE teacher_id=? ORDER BY created_at DESC", (teacher_id,)
        ).fetchall()
        return [_row_to_assessment(r) for r in rows]

    def list_by_school(self, school_id: str) -> list[Assessment]:
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id, order="created_at.desc")
            return [Assessment.model_validate(r["payload"]) for r in rows]
        rows = self.conn.execute(
            "SELECT * FROM assessments WHERE school_id=? ORDER BY created_at DESC", (school_id,)
        ).fetchall()
        return [_row_to_assessment(r) for r in rows]

    def delete(self, assessment_id: str) -> None:
        if self._remote.enabled:
            self._remote.delete(id=assessment_id)
            return
        self.conn.execute("DELETE FROM assessments WHERE id=?", (assessment_id,))
        self.conn.commit()


def _row_to_assessment(row: sqlite3.Row) -> Assessment:
    d: dict[str, Any] = dict(row)
    return Assessment(
        id=d["id"], school_id=d["school_id"], teacher_id=d["teacher_id"], title=d["title"],
        subject=d["subject"], grade=d["grade"],
        chapter_ids=json.loads(d["chapter_ids"] or "[]"),
        blueprint=json.loads(d["blueprint"]),
        status=d["status"], created_at=d["created_at"], updated_at=d["updated_at"],
        scheduled_at=d["scheduled_at"], completed_at=d["completed_at"],
        template_id=d["template_id"], metadata=json.loads(d["metadata"] or "{}"),
        selected_question_ids=json.loads(d["selected_question_ids"] or "[]"),
        generated_paper_id=d["generated_paper_id"], total_students=d["total_students"],
        evaluated_count=d["evaluated_count"],
    )
