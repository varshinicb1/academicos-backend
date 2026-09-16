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

`school_id` (2026-09-17): real column, not derived from `GeneratedPaper`
itself (the domain object is unchanged -- this is store-level metadata, same
choice already made for ScanSessionStore's `assessment_id` column). Fixes
the gap docs/system-review.html called "the largest outstanding correctness
gap": this store previously had no way to scope or list papers by school at
all, so the school-data export route (curriculum/routes.py) couldn't cover
it and cross-school authorization had to be inferred through a join on
whatever assessment a paper happened to be generated from (still true and
still done for GET/export/mail, since a paper's *authoritative* owner is
still its assessment -- this column is for listing/export, not a second
source of truth for authorization).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .schemas import GeneratedPaper, SchoolTemplate
from .postgres_kv import durable_table

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id           TEXT PRIMARY KEY,
  school_id    TEXT,
  paper_json   TEXT NOT NULL,
  template_json TEXT
);
"""


class PaperStore:
    def __init__(self, db_path: Path):
        self._remote = durable_table("papers")
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
        """CREATE TABLE IF NOT EXISTS never adds a column to a table that
        already exists on disk -- a real pre-2026-09-17 papers.sqlite has
        rows with no school_id, same class of fix as curriculum/store.py's
        own _migrate(). The index is created here, not in SCHEMA, because
        SCHEMA's CREATE TABLE IF NOT EXISTS is a no-op against an existing
        table -- an index on school_id there would run before this method
        ever adds the column, crashing on exactly the pre-existing-db case
        this method exists to handle."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()}
        if "school_id" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN school_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_school ON papers(school_id)")

    def save(self, paper: GeneratedPaper, template: Optional[SchoolTemplate] = None,
              school_id: Optional[str] = None) -> None:
        if self._remote.enabled:
            self._remote.upsert({
                "id": paper.id,
                "school_id": school_id,
                "payload": {"paper": paper.model_dump(mode="json"),
                           "template": template.model_dump(mode="json") if template else None},
            }, on_conflict="id")
            return
        with self._conn_lock:
            self.conn.execute(
                """INSERT INTO papers (id, school_id, paper_json, template_json) VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     school_id=excluded.school_id,
                     paper_json=excluded.paper_json, template_json=excluded.template_json""",
                (paper.id, school_id, paper.model_dump_json(), template.model_dump_json() if template else None),
            )
            self.conn.commit()

    def get(self, paper_id: str) -> Optional[GeneratedPaper]:
        if self._remote.enabled:
            rows = self._remote.select(id=paper_id)
            return GeneratedPaper.model_validate(rows[0]["payload"]["paper"]) if rows else None
        with self._conn_lock:
            row = self.conn.execute("SELECT paper_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        return GeneratedPaper.model_validate_json(row["paper_json"]) if row else None

    def list_by_school(self, school_id: str) -> list[GeneratedPaper]:
        """For the school-data export route -- see curriculum/routes.py.
        Papers saved before this column existed (school_id NULL) are
        invisible here; they remain reachable via their assessment's
        generated_paper_id, same as the export route's documented fallback."""
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id)
            return [GeneratedPaper.model_validate(r["payload"]["paper"]) for r in rows]
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT paper_json FROM papers WHERE school_id=?", (school_id,)
            ).fetchall()
        return [GeneratedPaper.model_validate_json(r["paper_json"]) for r in rows]

    def get_template(self, paper_id: str) -> Optional[SchoolTemplate]:
        if self._remote.enabled:
            rows = self._remote.select(id=paper_id)
            if not rows or rows[0]["payload"].get("template") is None:
                return None
            return SchoolTemplate.model_validate(rows[0]["payload"]["template"])
        with self._conn_lock:
            row = self.conn.execute("SELECT template_json FROM papers WHERE id=?", (paper_id,)).fetchone()
        if row is None or row["template_json"] is None:
            return None
        return SchoolTemplate.model_validate_json(row["template_json"])
