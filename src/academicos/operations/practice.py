"""Self-practice (SA-4): a student practises the chapters they are weakest
in, from the bank, with spaced revision.

A set is drawn from the chapters the student's learning progress puts first
under "revise next" (or the chapters they ask for), objective questions
first so it marks itself, and never a question the student answered in the
last REVISIT_AFTER_DAYS days. It is marked by the same evaluator as homework
and answer sheets, and the result goes into the knowledge store untagged,
so it counts as practice in learning progress. A student practises for
themselves: there is no teacher step, and nothing here is visible to
classmates.
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

PRACTICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS practice_sets (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    grade INTEGER NOT NULL,
    chapter_ids_json TEXT NOT NULL,
    questions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    answers_json TEXT,
    evaluations_json TEXT,
    marks REAL,
    max_marks INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_student ON practice_sets(student_id, created_at);
"""

REVISIT_AFTER_DAYS = 14
DEFAULT_COUNT = 8
_TYPE_ORDER = {"mcq": 0, "very_short_answer": 1, "short_answer": 2, "long_answer": 3}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pick(pool: list[dict], *, recent: set[str], count: int, seed: str) -> list[dict]:
    """`count` questions from `pool`: none answered recently, objective ones
    first, spread across the pool's chapters, in a stable order for `seed`."""
    fresh = [q for q in pool if q["id"] not in recent]
    rng = random.Random(seed)
    fresh.sort(key=lambda q: q["id"])
    rng.shuffle(fresh)
    fresh.sort(key=lambda q: _TYPE_ORDER.get(q.get("type", ""), 9))
    by_chapter: dict[str, list[dict]] = {}
    for q in fresh:
        key = q.get("taxonomyChapterId") or next(iter(q.get("chapterIds") or []), "-")
        by_chapter.setdefault(key, []).append(q)
    queues = [list(reversed(v)) for v in by_chapter.values()]
    out: list[dict] = []
    while len(out) < count and any(queues):
        for qu in queues:
            if qu and len(out) < count:
                out.append(qu.pop())
    return out


class PracticeMixin:
    """Added to OperationsStore."""

    def _practice(self, r: Optional[dict]) -> Optional[dict]:
        if r is None:
            return None
        r = dict(r)
        r["chapter_ids"] = json.loads(r.pop("chapter_ids_json"))
        r["questions"] = json.loads(r.pop("questions_json"))
        r["answers"] = json.loads(r.pop("answers_json") or "{}")
        r["evaluations"] = json.loads(r.pop("evaluations_json") or "[]")
        return r

    def create_practice(self, *, student_id: str, subject: str, grade: int, chapter_ids: list[str],
                        questions: list[dict]) -> dict:
        pid = f"prac_{uuid.uuid4().hex[:12]}"
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO practice_sets (id, student_id, subject, grade, chapter_ids_json, questions_json, created_at,"
                " max_marks) VALUES (?,?,?,?,?,?,?,?)",
                (pid, student_id, subject, grade, json.dumps(chapter_ids), json.dumps(questions, ensure_ascii=False),
                 _now(), sum(int(q.get("marks") or 0) for q in questions)))
            self._commit()
        return self.get_practice(pid)

    def get_practice(self, practice_id: str) -> Optional[dict]:
        return self._practice(self._fetchone("SELECT * FROM practice_sets WHERE id=?", (practice_id,)))

    def practice_for(self, student_id: str, limit: int = 20) -> list[dict]:
        return [self._practice(r) for r in self._fetchall(
            "SELECT * FROM practice_sets WHERE student_id=? ORDER BY created_at DESC LIMIT ?", (student_id, limit))]

    def finish_practice(self, practice_id: str, *, answers: dict[str, str], evaluations: list[dict],
                        marks: float) -> dict:
        with self._conn_lock:
            self.conn.execute("UPDATE practice_sets SET submitted_at=?, answers_json=?, evaluations_json=?, marks=?"
                              " WHERE id=?", (_now(), json.dumps(answers, ensure_ascii=False),
                                              json.dumps(evaluations, ensure_ascii=False), marks, practice_id))
            self._commit()
        return self.get_practice(practice_id)

    def recently_answered(self, student_id: str, days: int = REVISIT_AFTER_DAYS) -> set[str]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return {r["question_id"] for r in self._fetchall(
            "SELECT DISTINCT question_id FROM learning_evidence WHERE student_id=? AND occurred_at>=?",
            (student_id, since))}
