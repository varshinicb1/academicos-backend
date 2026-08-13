"""PaperStore: SQLite-backed persistence for generated papers.

Generated papers used to live only in a process-memory dict (`_papers` in
routes.py). That's fine on a machine that never restarts, but this app is
deployed on Render's free tier, which stops the container after ~15 minutes
idle -- every previously generated paper vanished on the next cold start,
which is exactly the "reopening a previously generated paper isn't wired up"
bug a teacher hit in practice. Mirrors store.py's pattern (plain sqlite3, one
JSON blob column for the whole nested object) rather than a new ORM table
per field, since GeneratedPaper/SchoolTemplate are already Pydantic models
with model_dump_json()/model_validate_json().
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .schemas import GeneratedPaper, SchoolTemplate

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id           TEXT PRIMARY KEY,
  paper_json   TEXT NOT NULL,
  template_json TEXT
);
"""


class PaperStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, paper: GeneratedPaper, template: Optional[SchoolTemplate] = None) -> None:
        self.conn.execute(
            """INSERT INTO papers (id, paper_json, template_json) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 paper_json=excluded.paper_json, template_json=excluded.template_json""",
            (paper.id, paper.model_dump_json(), template.model_dump_json() if template else None),
        )
        self.conn.commit()

    def get(self, paper_id: str) -> Optional[GeneratedPaper]:
        row = self.conn.execute("SELECT paper_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        return GeneratedPaper.model_validate_json(row["paper_json"]) if row else None

    def get_template(self, paper_id: str) -> Optional[SchoolTemplate]:
        row = self.conn.execute("SELECT template_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        if row is None or row["template_json"] is None:
            return None
        return SchoolTemplate.model_validate_json(row["template_json"])
