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
-- A question's measured difficulty (EX-3): its facility, the marks scored
-- over the marks available, from the marks teachers enter. Aggregates only:
-- no student is named here (API-5). One row per question and paper, so
-- entering a paper's marks again replaces that paper's part instead of
-- adding to it; `question_facility` is their sum, with its sample size.
CREATE TABLE IF NOT EXISTS question_facility_parts (
    question_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    responses INTEGER NOT NULL,
    score REAL NOT NULL,
    max_score REAL NOT NULL,
    PRIMARY KEY (question_id, paper_id)
);
CREATE TABLE IF NOT EXISTS question_facility (
    question_id TEXT PRIMARY KEY,
    responses INTEGER NOT NULL,
    facility REAL NOT NULL,
    updated_at TEXT NOT NULL
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

    def record_facility(self, paper_id: str, max_marks: dict[str, int]) -> None:
        """Recompute this paper's part of each question's facility from the
        marks now entered on it (`max_marks`: question id -> its marks on the
        paper), and each affected question's total. A cell is one response;
        a question with no cell left on the paper drops this paper's part."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT question_id, COUNT(*) AS n, SUM(marks) AS score FROM paper_marks "
                "WHERE paper_id=? GROUP BY question_id", (paper_id,)).fetchall()
            parts = [(r["question_id"], paper_id, r["n"], float(r["score"] or 0.0),
                      float(r["n"] * max_marks[r["question_id"]]))
                     for r in rows if max_marks.get(r["question_id"])]
            touched = set(max_marks) | {r["question_id"] for r in self.conn.execute(
                "SELECT question_id FROM question_facility_parts WHERE paper_id=?", (paper_id,)).fetchall()}
            self.conn.execute("DELETE FROM question_facility_parts WHERE paper_id=?", (paper_id,))
            self.conn.executemany(
                "INSERT INTO question_facility_parts (question_id, paper_id, responses, score, max_score)"
                " VALUES (?,?,?,?,?)", parts)
            for qid in touched:
                total = self.conn.execute(
                    "SELECT SUM(responses) AS n, SUM(score) AS score, SUM(max_score) AS top "
                    "FROM question_facility_parts WHERE question_id=?", (qid,)).fetchone()
                if not total["n"] or not total["top"]:
                    self.conn.execute("DELETE FROM question_facility WHERE question_id=?", (qid,))
                    continue
                self.conn.execute(
                    "INSERT INTO question_facility (question_id, responses, facility, updated_at)"
                    " VALUES (?,?,?,?) ON CONFLICT(question_id) DO UPDATE SET"
                    " responses=excluded.responses, facility=excluded.facility, updated_at=excluded.updated_at",
                    (qid, total["n"], total["score"] / total["top"], now))
            self._commit()

    def facility_for(self, question_ids: list[str], *, min_responses: int) -> dict[str, tuple[float, int]]:
        """Question id -> (facility, responses) for each of `question_ids`
        answered at least `min_responses` times; the rest are left out."""
        out: dict[str, tuple[float, int]] = {}
        ids = list(dict.fromkeys(question_ids))
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self._fetchall(
                    f"SELECT question_id, facility, responses FROM question_facility "
                    f"WHERE responses >= ? AND question_id IN ({marks})", (min_responses, *chunk)):
                out[r["question_id"]] = (r["facility"], r["responses"])
        return out
