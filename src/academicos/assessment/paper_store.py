"""PaperStore: persistence for generated papers.

Generated papers used to live only in a process-memory dict (`_papers` in
routes.py). That's fine on a machine that never restarts, but this app is
deployed on Render's free tier, which wipes local disk (SQLite included) on
every restart/spin-down/redeploy -- every previously generated paper
vanished on the next cold start, which is exactly the "reopening a
previously generated paper isn't wired up" bug a teacher hit in practice.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py. The exported PDF file
itself is still cached to local disk (see routes.py's export_paper/
get_paper_file) since it's cheaply regenerable from paper_json via
pdf_export.export_pdf whenever it's missing -- only the structural data
here needs to survive a restart.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .schemas import GeneratedPaper, SchoolTemplate
from .supabase_kv import SupabaseTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id           TEXT PRIMARY KEY,
  paper_json   TEXT NOT NULL,
  template_json TEXT
);
"""


class PaperStore:
    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("papers")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, paper: GeneratedPaper, template: Optional[SchoolTemplate] = None) -> None:
        if self._remote.enabled:
            self._remote.upsert({
                "id": paper.id,
                "payload": {"paper": paper.model_dump(mode="json"),
                           "template": template.model_dump(mode="json") if template else None},
            }, on_conflict="id")
            return
        self.conn.execute(
            """INSERT INTO papers (id, paper_json, template_json) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 paper_json=excluded.paper_json, template_json=excluded.template_json""",
            (paper.id, paper.model_dump_json(), template.model_dump_json() if template else None),
        )
        self.conn.commit()

    def get(self, paper_id: str) -> Optional[GeneratedPaper]:
        if self._remote.enabled:
            rows = self._remote.select(id=paper_id)
            return GeneratedPaper.model_validate(rows[0]["payload"]["paper"]) if rows else None
        row = self.conn.execute("SELECT paper_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        return GeneratedPaper.model_validate_json(row["paper_json"]) if row else None

    def get_template(self, paper_id: str) -> Optional[SchoolTemplate]:
        if self._remote.enabled:
            rows = self._remote.select(id=paper_id)
            if not rows or rows[0]["payload"].get("template") is None:
                return None
            return SchoolTemplate.model_validate(rows[0]["payload"]["template"])
        row = self.conn.execute("SELECT template_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        if row is None or row["template_json"] is None:
            return None
        return SchoolTemplate.model_validate_json(row["template_json"])
