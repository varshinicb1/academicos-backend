"""PracticeStore: SQLite-backed persistence for generated practice sets.

Same class of bug as PaperStore/GradedStore: `/practice/generate` used to
stash the PracticeSet only in the process-memory `_practice` dict, keyed by
its id. A student can easily spend more than Render's ~15-minute idle window
working through a practice set before calling `/practice/submit` with that
id -- on a restart in between, the id vanishes and the student's completed
work 404s ("practice set not found") instead of updating their mastery.

PracticeSet/PracticeItem are plain dataclasses (remediation.py); QuestionSchema
nested inside PracticeItem is Pydantic.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .remediation import PracticeItem, PracticeSet
from .schemas import QuestionSchema

SCHEMA = """
CREATE TABLE IF NOT EXISTS practice_sets (
  id            TEXT PRIMARY KEY,
  practice_json TEXT NOT NULL
);
"""


class PracticeStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, pset: PracticeSet) -> None:
        d = {
            "id": pset.id,
            "student_id": pset.student_id,
            "concept_ids": pset.concept_ids,
            "items": [{"question": i.question.model_dump(mode="json"), "concept_id": i.concept_id}
                      for i in pset.items],
            "answer_key": pset.answer_key,
            "warnings": pset.warnings,
            "created_at": pset.created_at,
        }
        blob = json.dumps(d)
        self.conn.execute(
            """INSERT INTO practice_sets (id, practice_json) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET practice_json=excluded.practice_json""",
            (pset.id, blob),
        )
        self.conn.commit()

    def get(self, set_id: str) -> Optional[PracticeSet]:
        row = self.conn.execute(
            "SELECT practice_json FROM practice_sets WHERE id=?", (set_id,)
        ).fetchone()
        if row is None:
            return None
        d = json.loads(row["practice_json"])
        items = [PracticeItem(question=QuestionSchema.model_validate(i["question"]),
                              concept_id=i["concept_id"])
                for i in d["items"]]
        return PracticeSet(
            id=d["id"], student_id=d["student_id"], concept_ids=d["concept_ids"],
            items=items, answer_key=d["answer_key"], warnings=d["warnings"],
            created_at=d["created_at"],
        )
