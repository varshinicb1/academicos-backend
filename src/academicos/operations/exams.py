"""Exam planning (EX-7): an exam's datesheet and its invigilation roster.

An exam (a half-yearly, a unit test) spans dates and classes. Its datesheet
places each class's subject papers on working days, never two papers for
one class on one day. The roster gives each section's room an invigilator
for each paper: never a teacher on leave that day, never one who teaches
that subject to that class (CBSE practice), and duties spread evenly -- the
teacher with the fewest duties so far first. Publishing tells students
their datesheet and teachers their duties.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

EXAMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    academic_year_id TEXT NOT NULL,
    name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    grades_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_exams_school ON exams(school_id, academic_year_id);
CREATE TABLE IF NOT EXISTS exam_papers (
    id TEXT PRIMARY KEY,
    exam_id TEXT NOT NULL,
    grade INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exam_papers_exam ON exam_papers(exam_id, date);
CREATE TABLE IF NOT EXISTS invigilation (
    exam_paper_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    teacher_id TEXT,
    assigned_by TEXT NOT NULL,
    PRIMARY KEY (exam_paper_id, section_id)
);
"""


class ExamError(ValueError):
    """A plan the school cannot run; the message is for the user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExamsMixin:
    """Added to OperationsStore."""

    def create_exam(self, *, school_id: str, academic_year_id: str, name: str, start_date: str, end_date: str,
                    grades: list[int], created_by: str) -> dict:
        if end_date < start_date:
            raise ExamError("the exam ends before it starts")
        if not grades:
            raise ExamError("choose at least one class")
        eid = f"exam_{uuid.uuid4().hex[:12]}"
        with self._conn_lock:
            self.conn.execute("INSERT INTO exams (id, school_id, academic_year_id, name, start_date, end_date,"
                              " grades_json, status, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (eid, school_id, academic_year_id, name, start_date, end_date,
                               json.dumps(sorted(set(grades))), "draft", created_by, _now()))
            self._commit()
        return self.get_exam(eid)

    def get_exam(self, exam_id: str) -> Optional[dict]:
        r = self._fetchone("SELECT * FROM exams WHERE id=?", (exam_id,))
        if r:
            r["grades"] = json.loads(r.pop("grades_json"))
        return r

    def exams_for_school(self, school_id: str) -> list[dict]:
        out = []
        for r in self._fetchall("SELECT id FROM exams WHERE school_id=? ORDER BY start_date DESC", (school_id,)):
            out.append(self.get_exam(r["id"]))
        return out

    def exam_papers(self, exam_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM exam_papers WHERE exam_id=? ORDER BY date, start_time, grade", (exam_id,))

    def replace_datesheet(self, exam: dict, papers: list[dict], *, working_dates: set[str]) -> list[dict]:
        """Replace the whole datesheet. Every problem is reported at once.

        A paper sent again unchanged (same class, subject, day and times)
        keeps its id and its invigilators. A paper removed or re-timed loses
        its duties, and an added or re-timed paper gets a new id. The web adds
        a paper by re-sending the whole sheet, which used to delete every
        duty, even after publishing (N-2-4)."""
        problems = []
        seen: dict[tuple[int, str], str] = {}
        for p in papers:
            label = f"class {p['grade']} {p['subject_name']}"
            if p["grade"] not in exam["grades"]:
                problems.append(f"{label}: class {p['grade']} is not in this exam")
            if not exam["start_date"] <= p["date"] <= exam["end_date"]:
                problems.append(f"{label}: {p['date']} is outside the exam ({exam['start_date']} to {exam['end_date']})")
            elif p["date"] not in working_dates:
                problems.append(f"{label}: {p['date']} is not a school day")
            if p["end_time"] <= p["start_time"]:
                problems.append(f"{label}: it ends before it starts")
            key = (p["grade"], p["date"])
            if key in seen:
                problems.append(f"class {p['grade']} has two papers on {p['date']}: {seen[key]} and {p['subject_name']}")
            seen[key] = p["subject_name"]
        if problems:
            raise ExamError("; ".join(problems))
        def key(p: dict) -> tuple:
            return p["grade"], p["subject_id"], p["date"], p["start_time"], p["end_time"]

        with self._conn_lock:
            old = {key(p): p["id"] for p in self.exam_papers(exam["id"])}
            kept: dict[str, dict] = {}
            added = []
            for p in papers:
                pid = old.get(key(p))
                if pid and pid not in kept:
                    kept[pid] = p
                else:
                    added.append(p)
            gone = [(pid,) for pid in old.values() if pid not in kept]
            self.conn.executemany("DELETE FROM invigilation WHERE exam_paper_id=?", gone)
            self.conn.executemany("DELETE FROM exam_papers WHERE id=?", gone)
            self.conn.executemany("UPDATE exam_papers SET subject_name=? WHERE id=?",
                                  [(p["subject_name"], pid) for pid, p in kept.items()])
            self.conn.executemany(
                "INSERT INTO exam_papers (id, exam_id, grade, subject_id, subject_name, date, start_time, end_time)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(f"ep_{uuid.uuid4().hex[:12]}", exam["id"], p["grade"], p["subject_id"], p["subject_name"], p["date"],
                  p["start_time"], p["end_time"]) for p in added])
            self._commit()
        return self.exam_papers(exam["id"])

    def last_notice(self, user_id: str, prefix: str) -> tuple[Optional[str], int]:
        """The newest notification key `prefix` (or `prefix:...`) this user
        was sent, and how many there are: what an invigilator was last told
        about an exam (exam_routes._tell_invigilators)."""
        rows = self._fetchall("SELECT dedupe_key FROM notifications WHERE user_id=? AND (dedupe_key=? OR "
                              "substr(dedupe_key, 1, ?)=?) ORDER BY rowid",
                              (user_id, prefix, len(prefix) + 1, prefix + ":"))
        return (rows[-1]["dedupe_key"] if rows else None), len(rows)

    def roster(self, exam_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT i.*, p.date, p.start_time, p.end_time, p.grade, p.subject_id, p.subject_name FROM invigilation i"
            " JOIN exam_papers p ON p.id=i.exam_paper_id WHERE p.exam_id=? ORDER BY p.date, p.start_time, i.section_id",
            (exam_id,))

    def set_invigilators(self, exam_id: str, rows: list[tuple[str, str, Optional[str]]], *, assigned_by: str) -> None:
        with self._conn_lock:
            self.conn.executemany(
                "INSERT INTO invigilation (exam_paper_id, section_id, teacher_id, assigned_by) VALUES (?,?,?,?)"
                " ON CONFLICT(exam_paper_id, section_id) DO UPDATE SET teacher_id=excluded.teacher_id,"
                " assigned_by=excluded.assigned_by", [(p, s, t, assigned_by) for p, s, t in rows])
            self._commit()

    def published_papers(self, school_id: str, academic_year_id: str) -> list[dict]:
        """Every paper of the year's published exams, each with its exam's
        name and `duties`: the (section, invigilator) pairs that have one.
        The cover engine plans around these (curriculum/cover.py): a class
        sitting a paper is not taught then, and an invigilator is not free
        (v3 audit N-3-12). A draft exam holds nothing until it is published."""
        papers = self._fetchall(
            "SELECT p.*, e.name AS exam_name FROM exam_papers p JOIN exams e ON e.id=p.exam_id"
            " WHERE e.school_id=? AND e.academic_year_id=? AND e.status='published' ORDER BY p.date, p.start_time",
            (school_id, academic_year_id))
        duties: dict[str, list[tuple[str, str]]] = {}
        for r in self._fetchall(
                "SELECT i.exam_paper_id, i.section_id, i.teacher_id FROM invigilation i"
                " JOIN exam_papers p ON p.id=i.exam_paper_id JOIN exams e ON e.id=p.exam_id"
                " WHERE e.school_id=? AND e.academic_year_id=? AND e.status='published' AND i.teacher_id IS NOT NULL",
                (school_id, academic_year_id)):
            duties.setdefault(r["exam_paper_id"], []).append((r["section_id"], r["teacher_id"]))
        return [{**p, "duties": duties.get(p["id"], [])} for p in papers]

    def publish_exam(self, exam_id: str) -> dict:
        with self._conn_lock:
            self.conn.execute("UPDATE exams SET status='published', published_at=COALESCE(published_at, ?) WHERE id=?",
                              (_now(), exam_id))
            self._commit()
        return self.get_exam(exam_id)


def auto_roster(papers: list[dict], sections_by_grade: dict[int, list[str]], teachers: list[str],
                teaches: set[tuple[str, str, str]], busy: dict[str, set[str]],
                existing: dict[tuple[str, str], Optional[str]]) -> tuple[list[tuple[str, str, Optional[str]]], list[str]]:
    """Give every (paper, section) an invigilator.

    `teaches` holds (teacher, section, subject) for the regular allocations:
    nobody invigilates a paper in a subject they teach that section.
    `busy` maps a date to the teachers on leave that day. Papers whose times
    overlap are assigned together as a matching, so one teacher is never in
    two rooms at once and every room that can be covered is covered (a
    greedy pick can strand the one teacher a later room needed); among the
    teachers who fit, those with fewer duties so far are tried first.
    Existing assignments are kept. Returns (rows, the rooms left without
    anyone)."""
    load = {t: 0 for t in teachers}
    for t in existing.values():
        if t in load:
            load[t] += 1
    ordered = sorted(papers, key=lambda x: (x["date"], x["start_time"], x["end_time"]))
    groups: list[list[dict]] = []
    for p in ordered:
        g = groups[-1] if groups else None
        if g and g[0]["date"] == p["date"] and p["start_time"] < max(x["end_time"] for x in g):
            g.append(p)
        else:
            groups.append([p])
    rows, unfilled = [], []
    for group in groups:
        held = {t for (pid, _), t in existing.items() if t and pid in {p["id"] for p in group}}
        slots = [(p, sid) for p in sorted(group, key=lambda x: x["grade"])
                 for sid in sections_by_grade.get(p["grade"], []) if not existing.get((p["id"], sid))]
        cands = {i: sorted((t for t in teachers
                            if t not in busy.get(p["date"], set()) and t not in held
                            and (t, sid, p["subject_id"]) not in teaches), key=lambda t: (load[t], t))
                 for i, (p, sid) in enumerate(slots)}
        owner: dict[str, int] = {}

        def place(i: int, seen: set[str]) -> bool:
            for t in cands[i]:
                if t in seen:
                    continue
                seen.add(t)
                if t not in owner or place(owner[t], seen):
                    owner[t] = i
                    return True
            return False

        for i in sorted(cands, key=lambda i: len(cands[i])):
            place(i, set())
        got = {i: t for t, i in owner.items()}
        for i, (p, sid) in enumerate(slots):
            t = got.get(i)
            if t is None:
                unfilled.append(f"{p['date']} class {p['grade']} {p['subject_name']}, section {sid}")
            else:
                load[t] += 1
            rows.append((p["id"], sid, t))
    return rows, unfilled
