"""Storage for the paper review workflow (EX-6; assessment/paper_review_routes.py)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

REVIEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_reviews (
    id TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    assessment_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    state TEXT NOT NULL,
    request_note TEXT,
    comment TEXT,
    requested_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_reviews_assessment ON paper_reviews(assessment_id);
CREATE INDEX IF NOT EXISTS idx_paper_reviews_reviewer ON paper_reviews(reviewer_id, state);
"""


class ReviewsMixin:
    """Added to OperationsStore."""

    def request_review(self, *, school_id: str, assessment_id: str, author_id: str, reviewer_id: str,
                       note: Optional[str]) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rid = f"rev_{uuid.uuid4().hex[:12]}"
        with self._conn_lock:
            self.conn.execute("UPDATE paper_reviews SET state='withdrawn', decided_at=? WHERE assessment_id=?"
                              " AND state='pending'", (now, assessment_id))
            self.conn.execute("INSERT INTO paper_reviews (id, school_id, assessment_id, author_id, reviewer_id, state,"
                              " request_note, requested_at) VALUES (?,?,?,?,?,?,?,?)",
                              (rid, school_id, assessment_id, author_id, reviewer_id, "pending", note, now))
            self._commit()
        return self.get_review(rid)

    def get_review(self, review_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM paper_reviews WHERE id=?", (review_id,))

    def pending_review_for(self, assessment_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM paper_reviews WHERE assessment_id=? AND state='pending'", (assessment_id,))

    def reviews_for(self, assessment_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM paper_reviews WHERE assessment_id=? ORDER BY requested_at", (assessment_id,))

    def reviews_where(self, school_id: str, *, reviewer_id: Optional[str], state: Optional[str]) -> list[dict]:
        sql, params = "SELECT * FROM paper_reviews WHERE school_id=?", [school_id]
        if reviewer_id:
            sql += " AND reviewer_id=?"
            params.append(reviewer_id)
        if state:
            sql += " AND state=?"
            params.append(state)
        return self._fetchall(sql + " ORDER BY requested_at DESC", tuple(params))

    def decide_review(self, review_id: str, *, state: str, comment: Optional[str]) -> dict:
        with self._conn_lock:
            self.conn.execute("UPDATE paper_reviews SET state=?, comment=?, decided_at=? WHERE id=?",
                              (state, comment, datetime.now(timezone.utc).isoformat(timespec="seconds"), review_id))
            self._commit()
        return self.get_review(review_id)
