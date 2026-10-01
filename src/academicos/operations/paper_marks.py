"""Storage for marks entry (EX-8; assessment/marks_routes.py)."""
from __future__ import annotations

from datetime import datetime, timezone

MARKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_marks (
    paper_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    marks REAL NOT NULL,
    entered_by TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    PRIMARY KEY (paper_id, student_id, question_id)
);
CREATE TABLE IF NOT EXISTS paper_absent (
    paper_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    PRIMARY KEY (paper_id, student_id)
);
"""


class MarksMixin:
    """Added to OperationsStore."""

    def save_marks(self, paper_id: str, entries: list[tuple[str, str, float]], *, absent: list[str],
                   present: list[str], entered_by: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn_lock:
            self.conn.executemany(
                "INSERT INTO paper_marks (paper_id, student_id, question_id, marks, entered_by, entered_at)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(paper_id, student_id, question_id) DO UPDATE SET"
                " marks=excluded.marks, entered_by=excluded.entered_by, entered_at=excluded.entered_at",
                [(paper_id, s, q, m, entered_by, now) for s, q, m in entries])
            self.conn.executemany("INSERT OR IGNORE INTO paper_absent (paper_id, student_id) VALUES (?,?)",
                                  [(paper_id, s) for s in absent])
            self.conn.executemany("DELETE FROM paper_absent WHERE paper_id=? AND student_id=?",
                                  [(paper_id, s) for s in present])
            self.conn.executemany("DELETE FROM paper_marks WHERE paper_id=? AND student_id=?",
                                  [(paper_id, s) for s in absent])
            self._commit()

    def marks_for(self, paper_id: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self._fetchall("SELECT student_id, question_id, marks FROM paper_marks WHERE paper_id=?", (paper_id,)):
            out.setdefault(r["student_id"], {})[r["question_id"]] = r["marks"]
        return out

    def papers_with_marks(self) -> list[str]:
        """Every paper id with at least one typed mark (school insights start
        from these rather than reading every paper a school has)."""
        return [r["paper_id"] for r in self._fetchall("SELECT DISTINCT paper_id FROM paper_marks")]

    def absent_for(self, paper_id: str) -> set[str]:
        return {r["student_id"] for r in self._fetchall("SELECT student_id FROM paper_absent WHERE paper_id=?",
                                                        (paper_id,))}
