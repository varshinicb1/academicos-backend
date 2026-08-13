"""ScanSessionStore: SQLite-backed persistence for in-progress mobile scan
sessions.

Same class of bug as PaperStore/GradedStore/PracticeStore: a scan session
(booklet photos captured -> OCR'd -> AI-scored -> teacher swipe-reviews each
answer -> finalize) is a multi-step, multi-minute flow that easily outlasts
Render's ~15-minute idle window between steps. Losing `_sessions` mid-review
means a teacher who has already approved half a booklet's answers loses all
of it and has to re-scan from page one.

ScanSession/CapturedPage/ReviewItem are plain dataclasses (mobile_scan.py).
Path and datetime fields aren't JSON-native, so they're converted explicitly
rather than via dataclasses.asdict(), which would silently deepcopy them
into non-serializable objects instead of erroring.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_sessions (
  id           TEXT PRIMARY KEY,
  session_json TEXT NOT NULL
);
"""


def _page_to_dict(p) -> dict:
    return {
        "page_no": p.page_no, "raw_path": str(p.raw_path),
        "processed_path": str(p.processed_path), "cropped": p.cropped,
        "ocr_text": p.ocr_text, "warnings": p.warnings,
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
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, session) -> None:
        d = {
            "id": session.id, "assessment_id": session.assessment_id,
            "student_id": session.student_id, "student_name": session.student_name,
            "subject": session.subject, "grade": session.grade,
            "pages": [_page_to_dict(p) for p in session.pages],
            "review": [_item_to_dict(i) for i in session.review],
            "status": session.status, "created_at": session.created_at.isoformat(),
            "raw_pdf_path": str(session.raw_pdf_path) if session.raw_pdf_path else None,
            "corrected_pdf_path": str(session.corrected_pdf_path) if session.corrected_pdf_path else None,
        }
        self.conn.execute(
            """INSERT INTO scan_sessions (id, session_json) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET session_json=excluded.session_json""",
            (session.id, json.dumps(d)),
        )
        self.conn.commit()

    def get(self, session_id: str):
        from .mobile_scan import CapturedPage, ReviewItem, ScanSession  # avoid import cycle

        row = self.conn.execute(
            "SELECT session_json FROM scan_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        d = json.loads(row["session_json"])
        return ScanSession(
            id=d["id"], assessment_id=d["assessment_id"], student_id=d["student_id"],
            student_name=d["student_name"], subject=d["subject"], grade=d["grade"],
            pages=[CapturedPage(page_no=p["page_no"], raw_path=Path(p["raw_path"]),
                                processed_path=Path(p["processed_path"]), cropped=p["cropped"],
                                ocr_text=p["ocr_text"], warnings=p["warnings"])
                  for p in d["pages"]],
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
        )
