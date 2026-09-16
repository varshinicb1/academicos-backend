"""ScanSessionStore: persistence for in-progress mobile scan sessions.

Same class of bug as PaperStore/GradedStore/PracticeStore: a scan session
(booklet photos captured -> OCR'd -> AI-scored -> teacher swipe-reviews each
answer -> finalize) is a multi-step, multi-minute flow that easily outlasts
Render's ~15-minute idle window between steps. Losing `_sessions` mid-review
means a teacher who has already approved half a booklet's answers loses all
of it and has to re-scan from page one.

Postgres-backed (via Supabase) when SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY are set, local SQLite otherwise -- see
knowledge.py's module docstring and supabase_kv.py. This closes the session
*state* half of the durability gap; the *photos themselves* are a separate
problem solved in mobile_scan.py via SupabaseStorage -- CapturedPage's
raw_path/processed_path stay local (mobile_scan.py still needs real files on
disk for the crop/OCR pipeline within one request), but each page also
carries a raw_storage_key/processed_storage_key pointing at the durable copy
in Supabase Storage, uploaded right after processing. Same idea for the
raw/corrected PDFs. A restart loses the local files; the storage keys are
what mobile_routes.py's serving endpoints fall back to.

ScanSession/CapturedPage/ReviewItem are plain dataclasses (mobile_scan.py).
Path and datetime fields aren't JSON-native, so they're converted explicitly
rather than via dataclasses.asdict(), which would silently deepcopy them
into non-serializable objects instead of erroring.

`school_id` (2026-09-17): real column, store-level metadata only, same
reasoning as PaperStore's -- ScanSession's authoritative owner is still the
assessment it's marking against (assessment_id, already a real column here
on the remote side), this column exists purely so list_by_school() and the
school-data export route can work without joining through AssessmentStore
for every row.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .postgres_kv import durable_table

SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_sessions (
  id           TEXT PRIMARY KEY,
  school_id    TEXT,
  assessment_id TEXT,
  session_json TEXT NOT NULL
);
"""


def _page_to_dict(p) -> dict:
    return {
        "page_no": p.page_no, "raw_path": str(p.raw_path),
        "processed_path": str(p.processed_path), "cropped": p.cropped,
        "ocr_text": p.ocr_text, "warnings": p.warnings,
        "raw_storage_key": p.raw_storage_key, "processed_storage_key": p.processed_storage_key,
    }


def _item_to_dict(i) -> dict:
    return {
        "question_id": i.question_id, "display_number": i.display_number, "stem": i.stem,
        "max_marks": i.max_marks, "student_answer": i.student_answer,
        "awarded_marks": i.awarded_marks, "verdict": i.verdict, "confidence": i.confidence,
        "reasoning": i.reasoning, "marking_points": i.marking_points,
        "ocr_warnings": i.ocr_warnings, "needs_review": i.needs_review, "status": i.status,
        "teacher_marks": i.teacher_marks, "teacher_comment": i.teacher_comment,
        "page_numbers": i.page_numbers,
    }


class ScanSessionStore:
    def __init__(self, db_path: Path):
        self._remote = durable_table("scan_sessions")
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
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(scan_sessions)").fetchall()}
        if "school_id" not in cols:
            self.conn.execute("ALTER TABLE scan_sessions ADD COLUMN school_id TEXT")
        if "assessment_id" not in cols:
            self.conn.execute("ALTER TABLE scan_sessions ADD COLUMN assessment_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_sessions_school ON scan_sessions(school_id)")

    def _encode(self, session) -> dict:
        return {
            "id": session.id, "assessment_id": session.assessment_id,
            "student_id": session.student_id, "student_name": session.student_name,
            "subject": session.subject, "grade": session.grade,
            "pages": [_page_to_dict(p) for p in session.pages],
            "review": [_item_to_dict(i) for i in session.review],
            "status": session.status, "created_at": session.created_at.isoformat(),
            "raw_pdf_path": str(session.raw_pdf_path) if session.raw_pdf_path else None,
            "corrected_pdf_path": str(session.corrected_pdf_path) if session.corrected_pdf_path else None,
            "raw_pdf_storage_key": session.raw_pdf_storage_key,
            "corrected_pdf_storage_key": session.corrected_pdf_storage_key,
        }

    def save(self, session, school_id: Optional[str] = None) -> None:
        # save_session() in mobile_scan.py is called many times across one
        # session's lifecycle (page upload, review, finalize, ...), but only
        # knows school_id from an in-process cache populated at
        # create_session() time -- which a Render idle-restart wipes mid-
        # session. Without this guard, resuming a session after exactly that
        # restart would silently overwrite a real school_id with NULL on the
        # next save, breaking the school-scoping this column exists for.
        # COALESCE (local) / a pre-write read (remote) both mean "only ever
        # move from unknown to known, never known back to unknown."
        d = self._encode(session)
        if self._remote.enabled:
            if school_id is None:
                existing = self._remote.select(id=session.id)
                if existing:
                    school_id = existing[0].get("school_id")
            self._remote.upsert({"id": session.id, "school_id": school_id,
                                 "assessment_id": session.assessment_id,
                                 "payload": d}, on_conflict="id")
            return
        with self._conn_lock:
            self.conn.execute(
                """INSERT INTO scan_sessions (id, school_id, assessment_id, session_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     school_id=COALESCE(excluded.school_id, scan_sessions.school_id),
                     assessment_id=excluded.assessment_id,
                     session_json=excluded.session_json""",
                (session.id, school_id, session.assessment_id, json.dumps(d)),
            )
            self.conn.commit()

    def list_by_school(self, school_id: str) -> list:
        """For the school-data export route -- see curriculum/routes.py.
        Sessions saved before this column existed are invisible here; they
        remain reachable via a join on assessment_id if ever needed."""
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id)
            return [self._decode(r["payload"]) for r in rows]
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT session_json FROM scan_sessions WHERE school_id=?", (school_id,)
            ).fetchall()
        return [self._decode(json.loads(r["session_json"])) for r in rows]

    def _decode(self, d: dict):
        from .mobile_scan import CapturedPage, ReviewItem, ScanSession  # avoid import cycle

        return ScanSession(
            id=d["id"], assessment_id=d["assessment_id"], student_id=d["student_id"],
            student_name=d["student_name"], subject=d["subject"], grade=d["grade"],
            pages=[CapturedPage(
                page_no=p["page_no"], raw_path=Path(p["raw_path"]),
                processed_path=Path(p["processed_path"]), cropped=p["cropped"],
                ocr_text=p["ocr_text"], warnings=p["warnings"],
                raw_storage_key=p.get("raw_storage_key", ""),
                processed_storage_key=p.get("processed_storage_key", ""),
            ) for p in d["pages"]],
            review=[ReviewItem(
                question_id=i["question_id"], display_number=i["display_number"], stem=i["stem"],
                max_marks=i["max_marks"], student_answer=i["student_answer"],
                awarded_marks=i["awarded_marks"], verdict=i["verdict"], confidence=i["confidence"],
                reasoning=i["reasoning"], marking_points=i["marking_points"],
                ocr_warnings=i["ocr_warnings"], needs_review=i["needs_review"],
                status=i["status"], teacher_marks=i["teacher_marks"],
                teacher_comment=i["teacher_comment"], page_numbers=i["page_numbers"],
            ) for i in d["review"]],
            status=d["status"], created_at=datetime.fromisoformat(d["created_at"]),
            raw_pdf_path=Path(d["raw_pdf_path"]) if d["raw_pdf_path"] else None,
            corrected_pdf_path=Path(d["corrected_pdf_path"]) if d["corrected_pdf_path"] else None,
            raw_pdf_storage_key=d.get("raw_pdf_storage_key", ""),
            corrected_pdf_storage_key=d.get("corrected_pdf_storage_key", ""),
        )

    def get(self, session_id: str):
        if self._remote.enabled:
            rows = self._remote.select(id=session_id)
            return self._decode(rows[0]["payload"]) if rows else None
        with self._conn_lock:
            row = self.conn.execute(
                "SELECT session_json FROM scan_sessions WHERE id=?", (session_id,)
            ).fetchone()
        return self._decode(json.loads(row["session_json"])) if row else None
