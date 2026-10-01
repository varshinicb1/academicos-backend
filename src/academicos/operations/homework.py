"""Homework (TA-4, SA-2): a teacher sets questions from the bank for one or
more sections, students answer them, the objective ones mark themselves and
the teacher marks the rest.

The questions are copied into the homework when it is created. The bank is
rebuilt and re-tagged often; a homework must show a student next month the
same question the teacher chose today, and mark it by the same key.

A submission is marked at once by the same evaluator the answer-sheet and
practice flows use (assessment/evaluate.py). When every answer was marked
with confidence -- an all-MCQ homework, typically -- it is final at once and
the student is told. Otherwise it waits for the teacher, who may override any
question's marks.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

HOMEWORK_SCHEMA = """
CREATE TABLE IF NOT EXISTS homework (
    id TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    academic_year_id TEXT NOT NULL,
    teacher_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    grade INTEGER NOT NULL,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL DEFAULT '',
    chapter_ids_json TEXT NOT NULL DEFAULT '[]',
    questions_json TEXT NOT NULL,
    total_marks INTEGER NOT NULL,
    due_date TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_homework_school ON homework(school_id, status);
CREATE TABLE IF NOT EXISTS homework_sections (
    homework_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    PRIMARY KEY (homework_id, section_id)
);
CREATE INDEX IF NOT EXISTS idx_homework_sections_section ON homework_sections(section_id);
CREATE TABLE IF NOT EXISTS homework_submissions (
    id TEXT PRIMARY KEY,
    homework_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    evaluations_json TEXT NOT NULL,
    auto_marks REAL NOT NULL,
    marks REAL,
    max_marks INTEGER NOT NULL,
    status TEXT NOT NULL,
    late INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 1,
    submitted_at TEXT NOT NULL,
    graded_at TEXT,
    graded_by TEXT,
    feedback TEXT,
    UNIQUE (homework_id, student_id)
);
"""

STATUSES = ("draft", "published", "closed")
MAX_QUESTIONS = 30


class HomeworkError(ValueError):
    """A request the homework cannot honour; the message is for the user."""


@dataclass
class Homework:
    id: str
    school_id: str
    academic_year_id: str
    teacher_id: str
    subject_id: str
    subject_name: str
    grade: int
    title: str
    instructions: str
    chapter_ids: list[str]
    questions: list[dict]          # QuestionSchema dumps, camelCase, as the bank served them
    total_marks: int
    due_date: str                  # YYYY-MM-DD, the school's own calendar
    status: str
    created_by: str
    created_at: str
    published_at: Optional[str]
    section_ids: list[str] = field(default_factory=list)


@dataclass
class Submission:
    id: str
    homework_id: str
    student_id: str
    section_id: str
    answers: dict[str, str]
    evaluations: list[dict]
    auto_marks: float
    marks: Optional[float]
    max_marks: int
    status: str                    # submitted | graded
    late: bool
    attempts: int
    submitted_at: str
    graded_at: Optional[str]
    graded_by: Optional[str]
    feedback: Optional[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def grade_answers(questions: list[dict], answers: dict[str, str]) -> tuple[list[dict], float, bool]:
    """(one evaluation per question, marks awarded, whether any needs a
    teacher). A question left unanswered is evaluated as blank, which the
    evaluator marks 0 with confidence -- so skipping never holds the rest of
    the homework back from being final."""
    from ..assessment.evaluate import evaluate_answer
    from ..assessment.schemas import QuestionSchema

    out: list[dict] = []
    total = 0.0
    review = False
    for raw in questions:
        q = QuestionSchema.model_validate(raw)
        ev = evaluate_answer(q, q.answer_scheme, answers.get(q.id, ""))
        out.append({"questionId": q.id, "awardedMarks": float(ev.awarded_marks), "maxMarks": int(ev.max_marks),
                    "verdict": ev.verdict, "confidence": float(ev.confidence), "reasoning": ev.reasoning,
                    "needsReview": bool(ev.needs_review)})
        total += float(ev.awarded_marks)
        review = review or bool(ev.needs_review)
    return out, total, review


class HomeworkMixin:
    """Added to OperationsStore; uses its _exec/_fetchone/_fetchall/_commit."""

    def _homework(self, row: Optional[dict]) -> Optional[Homework]:
        if row is None:
            return None
        sections = [r["section_id"] for r in self._fetchall(
            "SELECT section_id FROM homework_sections WHERE homework_id=? ORDER BY section_id", (row["id"],))]
        return Homework(
            id=row["id"], school_id=row["school_id"], academic_year_id=row["academic_year_id"],
            teacher_id=row["teacher_id"], subject_id=row["subject_id"], subject_name=row["subject_name"],
            grade=row["grade"], title=row["title"], instructions=row["instructions"],
            chapter_ids=json.loads(row["chapter_ids_json"]), questions=json.loads(row["questions_json"]),
            total_marks=row["total_marks"], due_date=row["due_date"], status=row["status"],
            created_by=row["created_by"], created_at=row["created_at"], published_at=row["published_at"],
            section_ids=sections)

    def create_homework(self, *, school_id: str, academic_year_id: str, teacher_id: str, subject_id: str,
                        subject_name: str, grade: int, section_ids: list[str], title: str, instructions: str,
                        chapter_ids: list[str], questions: list[dict], due_date: str, created_by: str,
                        publish: bool) -> Homework:
        if not questions:
            raise HomeworkError("a homework needs at least one question")
        if len(questions) > MAX_QUESTIONS:
            raise HomeworkError(f"a homework takes at most {MAX_QUESTIONS} questions")
        if not section_ids:
            raise HomeworkError("choose at least one section")
        hid = f"hw_{uuid.uuid4().hex[:12]}"
        now = _now()
        total = sum(int(q.get("marks") or 0) for q in questions)
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO homework (id, school_id, academic_year_id, teacher_id, subject_id, subject_name, grade,"
                " title, instructions, chapter_ids_json, questions_json, total_marks, due_date, status, created_by,"
                " created_at, published_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (hid, school_id, academic_year_id, teacher_id, subject_id, subject_name, grade, title,
                 instructions, json.dumps(chapter_ids), json.dumps(questions, ensure_ascii=False), total,
                 due_date, "published" if publish else "draft", created_by, now, now if publish else None))
            self.conn.executemany("INSERT INTO homework_sections (homework_id, section_id) VALUES (?,?)",
                                  [(hid, s) for s in dict.fromkeys(section_ids)])
            self._commit()
        return self.get_homework(hid)

    def get_homework(self, homework_id: str) -> Optional[Homework]:
        return self._homework(self._fetchone("SELECT * FROM homework WHERE id=?", (homework_id,)))

    def homework_for_school(self, school_id: str, *, status: Optional[str] = None) -> list[Homework]:
        sql, params = "SELECT * FROM homework WHERE school_id=?", [school_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        return [self._homework(r) for r in self._fetchall(sql + " ORDER BY due_date DESC, created_at DESC",
                                                          tuple(params))]

    def homework_for_section(self, section_id: str, *, published_only: bool = True) -> list[Homework]:
        sql = ("SELECT h.* FROM homework h JOIN homework_sections s ON s.homework_id=h.id WHERE s.section_id=?"
               + (" AND h.status != 'draft'" if published_only else "") + " ORDER BY h.due_date DESC, h.created_at DESC")
        return [self._homework(r) for r in self._fetchall(sql, (section_id,))]

    def set_homework_status(self, homework_id: str, status: str) -> Homework:
        if status not in STATUSES:
            raise HomeworkError(f"unknown status {status!r}")
        hw = self.get_homework(homework_id)
        if hw is None:
            raise HomeworkError("homework not found")
        order = {s: i for i, s in enumerate(STATUSES)}
        if order[status] < order[hw.status]:
            raise HomeworkError(f"a {hw.status} homework cannot go back to {status}")
        with self._conn_lock:
            self.conn.execute("UPDATE homework SET status=?, published_at=COALESCE(published_at, ?) WHERE id=?",
                              (status, _now() if status == "published" else None, homework_id))
            self._commit()
        return self.get_homework(homework_id)

    # ---------------- submissions ----------------

    @staticmethod
    def _submission(row: Optional[dict]) -> Optional[Submission]:
        if row is None:
            return None
        return Submission(
            id=row["id"], homework_id=row["homework_id"], student_id=row["student_id"],
            section_id=row["section_id"], answers=json.loads(row["answers_json"]),
            evaluations=json.loads(row["evaluations_json"]), auto_marks=row["auto_marks"], marks=row["marks"],
            max_marks=row["max_marks"], status=row["status"], late=bool(row["late"]), attempts=row["attempts"],
            submitted_at=row["submitted_at"], graded_at=row["graded_at"], graded_by=row["graded_by"],
            feedback=row["feedback"])

    def get_submission(self, homework_id: str, student_id: str) -> Optional[Submission]:
        return self._submission(self._fetchone(
            "SELECT * FROM homework_submissions WHERE homework_id=? AND student_id=?", (homework_id, student_id)))

    def submissions_for(self, homework_id: str) -> list[Submission]:
        return [self._submission(r) for r in self._fetchall(
            "SELECT * FROM homework_submissions WHERE homework_id=? ORDER BY submitted_at", (homework_id,))]

    def submission_counts(self, homework_id: str) -> dict[str, int]:
        rows = self._fetchall("SELECT status, COUNT(*) AS n FROM homework_submissions WHERE homework_id=?"
                              " GROUP BY status", (homework_id,))
        return {r["status"]: r["n"] for r in rows}

    def submit_homework(self, hw: Homework, *, student_id: str, section_id: str, answers: dict[str, str],
                        late: bool, needs_teacher: bool = False) -> Submission:
        """Mark and store a student's answers. A student may send again until
        the homework is marked or closed; each send is marked afresh."""
        if hw.status != "published":
            raise HomeworkError("this homework is closed" if hw.status == "closed"
                                else "this homework has not been set yet")
        known = {q["id"] for q in hw.questions}
        stray = sorted(set(answers) - known)
        if stray:
            raise HomeworkError(f"these are not questions of this homework: {', '.join(stray[:5])}")
        previous = self.get_submission(hw.id, student_id)
        if previous is not None and previous.status == "graded":
            raise HomeworkError("your homework has been marked; it cannot be changed now")
        evaluations, auto, review = grade_answers(hw.questions, answers)
        review = review or needs_teacher      # written work in photos: the teacher marks it
        now = _now()
        status = "submitted" if review else "graded"
        with self._conn_lock:
            if previous is None:
                self.conn.execute(
                    "INSERT INTO homework_submissions (id, homework_id, student_id, section_id, answers_json,"
                    " evaluations_json, auto_marks, marks, max_marks, status, late, attempts, submitted_at,"
                    " graded_at, graded_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)",
                    (f"hws_{uuid.uuid4().hex[:12]}", hw.id, student_id, section_id,
                     json.dumps(answers, ensure_ascii=False), json.dumps(evaluations, ensure_ascii=False), auto,
                     None if review else auto, hw.total_marks, status, int(late), now,
                     None if review else now, None if review else "auto"))
            else:
                self.conn.execute(
                    "UPDATE homework_submissions SET answers_json=?, evaluations_json=?, auto_marks=?, marks=?,"
                    " status=?, late=?, attempts=attempts+1, submitted_at=?, graded_at=?, graded_by=?"
                    " WHERE id=?",
                    (json.dumps(answers, ensure_ascii=False), json.dumps(evaluations, ensure_ascii=False), auto,
                     None if review else auto, status, int(late), now, None if review else now,
                     None if review else "auto", previous.id))
            self._commit()
        return self.get_submission(hw.id, student_id)

    def grade_submission(self, hw: Homework, student_id: str, *, marks: dict[str, float],
                         feedback: Optional[str], graded_by: str) -> Submission:
        """The teacher's marks. Any question not named keeps the evaluator's
        award; a named one takes the teacher's, which must lie between 0 and
        that question's marks."""
        sub = self.get_submission(hw.id, student_id)
        if sub is None:
            raise HomeworkError("this student has not submitted")
        by_id = {e["questionId"]: dict(e) for e in sub.evaluations}
        for qid, m in marks.items():
            if qid not in by_id:
                raise HomeworkError(f"{qid} is not a question of this homework")
            if not 0 <= m <= by_id[qid]["maxMarks"]:
                raise HomeworkError(f"{qid} is out of {by_id[qid]['maxMarks']}; {m} is not a mark it can get")
            by_id[qid].update(awardedMarks=float(m), needsReview=False, verdict="teacher")
        evaluations = [by_id[e["questionId"]] for e in sub.evaluations]
        total = sum(e["awardedMarks"] for e in evaluations)
        with self._conn_lock:
            self.conn.execute(
                "UPDATE homework_submissions SET evaluations_json=?, marks=?, status='graded', graded_at=?,"
                " graded_by=?, feedback=? WHERE id=?",
                (json.dumps(evaluations, ensure_ascii=False), total, _now(), graded_by, feedback, sub.id))
            self._commit()
        return self.get_submission(hw.id, student_id)

    def homework_summary(self, hw: Homework) -> dict[str, Any]:
        counts = self.submission_counts(hw.id)
        return {"submitted": counts.get("submitted", 0), "graded": counts.get("graded", 0)}
