"""Pillar 1 — school paper templates (the "template maker" backend).

A school's paper format is institutional identity: header, logo, fonts,
margins, and — most importantly — its own section layout (how many sections,
marks per question, internal choice). This store lets a school define that
once and have every generated paper conform exactly.

Templates carry their own `SectionBlueprint` list, so "the school's format" is
data rather than the hard-coded CBSE default in `templates.py`.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py for why. A school's
branding is exactly the kind of data that must survive a Render redeploy:
losing it silently reverts every generated paper to the default CBSE look.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from .schemas import SchoolTemplate, SectionBlueprint
from .supabase_kv import SupabaseTable
from .templates import default_sections

SCHEMA = """
CREATE TABLE IF NOT EXISTS school_templates (
  id            TEXT PRIMARY KEY,
  school_id     TEXT NOT NULL,
  name          TEXT NOT NULL,
  payload       TEXT NOT NULL,   -- json SchoolTemplate
  sections      TEXT NOT NULL,   -- json list[SectionBlueprint]
  is_default    INTEGER NOT NULL DEFAULT 0,
  updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_templates_school ON school_templates(school_id);
"""


class TemplateStore:
    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("school_templates")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, template: SchoolTemplate, sections: list[SectionBlueprint]) -> SchoolTemplate:
        if not template.id or template.id == "new":
            template = template.model_copy(update={"id": f"tpl_{uuid.uuid4().hex[:10]}"})
        if self._remote.enabled:
            if template.is_default:
                self._remote.update({"is_default": False}, school_id=template.school_id)
            self._remote.upsert({
                "id": template.id, "school_id": template.school_id,
                "is_default": template.is_default,
                "payload": {"template": template.model_dump(mode="json", by_alias=True),
                           "sections": [s.model_dump(mode="json", by_alias=True) for s in sections]},
            }, on_conflict="id")
            return template
        if template.is_default:
            self.conn.execute("UPDATE school_templates SET is_default=0 WHERE school_id=?",
                              (template.school_id,))
        self.conn.execute(
            """INSERT INTO school_templates (id, school_id, name, payload, sections, is_default, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, payload=excluded.payload, sections=excluded.sections,
                 is_default=excluded.is_default, updated_at=excluded.updated_at""",
            (template.id, template.school_id, template.name, template.model_dump_json(),
             json.dumps([s.model_dump(by_alias=True) for s in sections]),
             1 if template.is_default else 0),
        )
        self.conn.commit()
        return template

    def get(self, template_id: str) -> Optional[tuple[SchoolTemplate, list[SectionBlueprint]]]:
        if self._remote.enabled:
            rows = self._remote.select(id=template_id)
            return _remote_row(rows[0]) if rows else None
        row = self.conn.execute("SELECT * FROM school_templates WHERE id=?",
                                (template_id,)).fetchone()
        return _row(row) if row else None

    def list_for_school(self, school_id: str) -> list[SchoolTemplate]:
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id, order="is_default.desc")
            return [_remote_row(r)[0] for r in rows]
        rows = self.conn.execute(
            "SELECT * FROM school_templates WHERE school_id=? ORDER BY is_default DESC, name",
            (school_id,)).fetchall()
        return [_row(r)[0] for r in rows]

    def sections_for(self, school_id: str, template_id: str | None,
                     total_marks: int) -> list[SectionBlueprint]:
        """The section layout a paper should use: explicit template, the school
        default, else the built-in CBSE pattern scaled to the mark total."""
        if template_id:
            found = self.get(template_id)
            if found and found[1]:
                return found[1]
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id, **{"is_default": "true"})
            if rows:
                _, sections = _remote_row(rows[0])
                if sections:
                    return sections
            return default_sections(total_marks)
        row = self.conn.execute(
            "SELECT * FROM school_templates WHERE school_id=? AND is_default=1", (school_id,)
        ).fetchone()
        if row:
            _, sections = _row(row)
            if sections:
                return sections
        return default_sections(total_marks)

    def default_for(self, school_id: str) -> SchoolTemplate:
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id, **{"is_default": "true"})
            if rows:
                return _remote_row(rows[0])[0]
            return SchoolTemplate(id="default", school_id=school_id, name="Default CBSE Template")
        row = self.conn.execute(
            "SELECT * FROM school_templates WHERE school_id=? AND is_default=1", (school_id,)
        ).fetchone()
        if row:
            return _row(row)[0]
        return SchoolTemplate(id="default", school_id=school_id, name="Default CBSE Template")

    def delete(self, template_id: str) -> None:
        if self._remote.enabled:
            self._remote.delete(id=template_id)
            return
        self.conn.execute("DELETE FROM school_templates WHERE id=?", (template_id,))
        self.conn.commit()


def _row(row: sqlite3.Row) -> tuple[SchoolTemplate, list[SectionBlueprint]]:
    template = SchoolTemplate.model_validate_json(row["payload"])
    sections = [SectionBlueprint.model_validate(s) for s in json.loads(row["sections"] or "[]")]
    return template, sections


def _remote_row(row: dict) -> tuple[SchoolTemplate, list[SectionBlueprint]]:
    payload = row["payload"]
    template = SchoolTemplate.model_validate(payload["template"])
    sections = [SectionBlueprint.model_validate(s) for s in payload.get("sections", [])]
    return template, sections
