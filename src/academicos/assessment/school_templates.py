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
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from .schemas import PaperTemplate, SchoolTemplate, SectionBlueprint
from .postgres_kv import durable_table
from .templates import default_sections, template_section_blueprints

log = logging.getLogger(__name__)

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
        self._remote = durable_table("school_templates")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
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

    def row_info(self, template_id: str) -> Optional[tuple[str, str]]:
        """(kind, school_id) of a stored row -- "paper" or "school" -- or None.
        What the older `/school-templates` routes check before touching an id,
        since the table holds both kinds and the URL's school proves nothing
        about which school the row belongs to."""
        if self._remote.enabled:
            rows = self._remote.select(id=template_id)
        else:
            rows = self.conn.execute("SELECT * FROM school_templates WHERE id=?",
                                     (template_id,)).fetchall()
        if not rows:
            return None
        return _payload_kind(rows[0]), rows[0]["school_id"]

    def get_branding(self, template_id: str, school_id: str,
                     ) -> Optional[tuple[SchoolTemplate, list[SectionBlueprint]]]:
        """`get`, limited to one school's branding rows: the lookup for any
        caller that resolves a client-supplied `templateId` without an owner
        check (quick and custom generation, `sections_for`)."""
        info = self.row_info(template_id)
        if info is None or info != ("school", school_id):
            return None
        return self.get(template_id)

    def list_for_school(self, school_id: str) -> list[SchoolTemplate]:
        """Empty here means "school has never saved a template" -- and this
        is the list the AI Assessment Designer checks before it will
        generate a paper at all (CreateAssessmentUseCase on the Flutter
        side). A brand-new school got a hard failure on the very first
        thing a teacher tries to do, with no path forward except finding
        the Template Maker screen first. Auto-seeding + persisting a
        default here means "day one, generate your first paper" just
        works, on every platform that hits this store (Render web, the
        desktop app's embedded backend, and any future client) -- the
        Flutter offline build already got the equivalent fix locally;
        this is the same fix at the one place all networked clients share.
        """
        existing = self._list_raw(school_id)
        if existing:
            return existing
        default_template = SchoolTemplate(
            id="new", school_id=school_id, name="Default CBSE Template", is_default=True,
        )
        self.save(default_template, default_sections(100))
        return self._list_raw(school_id)

    def _list_raw(self, school_id: str) -> list[SchoolTemplate]:
        """The school's BRANDING templates. Teacher paper templates share this
        table but are left out: they can be private to one teacher, and this
        list is served to the whole school by `/school-templates`."""
        return [_row_template(r) for r in self._rows_for_school(school_id)
                if _payload_kind(r) != "paper"]

    def _rows_for_school(self, school_id: str) -> list:
        if self._remote.enabled:
            return self._remote.select(school_id=school_id, order="is_default.desc")
        return self.conn.execute(
            "SELECT * FROM school_templates WHERE school_id=? ORDER BY is_default DESC, name",
            (school_id,)).fetchall()

    # ---- teacher paper templates (Task 901) ----
    #
    # Same table, same durable backend, no schema migration: the kind lives
    # in the JSON payload, so the Supabase table (whose columns this code
    # cannot migrate) needs no change either. Rows written before this
    # existed have no kind and read as branding, which is what they are.

    def save_paper_template(self, template: PaperTemplate) -> PaperTemplate:
        """Stores the template with its SectionBlueprint form in the `sections`
        column, which the table requires. That column is NOT served for a
        paper row: `sections_for` and `get_branding` skip paper templates,
        because they have no owner check and a template can be private."""
        sections = [s if s.id else s.model_copy(update={"id": f"s{i + 1}"})
                    for i, s in enumerate(template.sections)]
        template = template.model_copy(update={
            "kind": "paper", "is_default": False, "is_preset": False, "sections": sections,
            "scope_notes": []})
        saved = self.save(template, template_section_blueprints(template.sections))
        return template.model_copy(update={"id": saved.id})

    def get_paper_template(self, template_id: str) -> Optional[PaperTemplate]:
        if self._remote.enabled:
            rows = self._remote.select(id=template_id)
        else:
            rows = self.conn.execute("SELECT * FROM school_templates WHERE id=?",
                                     (template_id,)).fetchall()
        if not rows or _payload_kind(rows[0]) != "paper":
            return None
        return PaperTemplate.model_validate(_payload_template(rows[0]))

    def list_paper_templates(self, school_id: str) -> list[PaperTemplate]:
        """Every readable paper template of the school. A row that no longer
        validates (saved before a validator tightened, or written through the
        remote path) is skipped and logged: failing the whole list would take
        down the builder's first screen for every teacher in the school over
        one template."""
        out: list[PaperTemplate] = []
        for r in self._rows_for_school(school_id):
            if _payload_kind(r) != "paper":
                continue
            try:
                out.append(PaperTemplate.model_validate(_payload_template(r)))
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                log.warning("skipping unreadable paper template %s of school %s: %s",
                            r["id"], school_id, exc)
        return sorted(out, key=lambda t: t.name.lower())

    def sections_for(self, school_id: str, template_id: str | None,
                     total_marks: int) -> list[SectionBlueprint]:
        """The section layout a paper should use: explicit template, the school
        default, else the built-in CBSE pattern scaled to the mark total.

        An explicit id counts only when it is one of THIS school's branding
        rows. A teacher's paper template in the same table is private to its
        owner, and this method has no caller to check that against; its
        sections are served by `/teacher-templates`, which does."""
        if template_id:
            found = self.get_branding(template_id, school_id)
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


def _payload_template(row) -> dict:
    """The stored template dict, from either backend's row shape."""
    if isinstance(row, dict):
        return row["payload"]["template"]
    return json.loads(row["payload"])


def _payload_kind(row) -> str:
    return _payload_template(row).get("kind") or "school"


def _row_template(row) -> SchoolTemplate:
    return (_remote_row(row) if isinstance(row, dict) else _row(row))[0]


def _row(row: sqlite3.Row) -> tuple[SchoolTemplate, list[SectionBlueprint]]:
    template = SchoolTemplate.model_validate_json(row["payload"])
    sections = [SectionBlueprint.model_validate(s) for s in json.loads(row["sections"] or "[]")]
    return template, sections


def _remote_row(row: dict) -> tuple[SchoolTemplate, list[SectionBlueprint]]:
    payload = row["payload"]
    template = SchoolTemplate.model_validate(payload["template"])
    sections = [SectionBlueprint.model_validate(s) for s in payload.get("sections", [])]
    return template, sections
