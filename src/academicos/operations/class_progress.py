"""Class progress (TA-6): a section's mastery per chapter and topic, from
homework, tests and practice, and the students who need help.

For the teachers who teach the section and the principal (or a reports
admin) only -- never shown to students, and no teacher is compared with
another (ADM-5). Every read is logged as a read of the section's student
data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..assessment.auth_routes import holds, require_staff
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .learning import _chapter_name, _taxonomy, score
from .routes import store

router = APIRouter(prefix="/api/v1")
STATUSES = ("needs_work", "developing", "mastered", "early")


class UnitProgress(Camel):
    key: str
    name: str
    students_with_evidence: int
    mean_mastery: Optional[float] = None
    counts: dict[str, int]


class ChapterProgressRow(UnitProgress):
    subject: str
    grade: int
    topics: list[UnitProgress]


class StudentHelp(Camel):
    student_id: str
    name: str
    needs_work: list[str]          # chapter names
    answers: int


class ClassProgressResponse(Camel):
    section_id: str
    section_name: str
    students: int
    chapters: list[ChapterProgressRow]
    needing_help: list[StudentHelp]


def _may_see(section, current: User) -> bool:
    cs = cr._require()
    grade = cs.get_grade(section.grade_id)
    if holds(current, "reports", grade=grade.number if grade else None, section_id=section.id):
        return True
    return any(a.teacher_id == current.id for a in cs.allocations_for_year(section.academic_year_id)
               if a.section_id == section.id)


def _unit(key: str, name: str, per_student: dict[str, list[dict]], now) -> UnitProgress:
    counts = {s: 0 for s in STATUSES}
    masteries = []
    for rows in per_student.values():
        sc = score(rows, now)
        counts[sc["status"]] = counts.get(sc["status"], 0) + 1
        if sc["mastery"] is not None:
            masteries.append(sc["mastery"])
    return UnitProgress(key=key, name=name, students_with_evidence=len(per_student),
                        mean_mastery=round(sum(masteries) / len(masteries), 3) if masteries else None, counts=counts)


@router.get("/sections/{section_id}/progress", response_model=ClassProgressResponse)
def class_progress(section_id: str, subject: Optional[str] = Query(default=None),
                   current: User = Depends(require_staff)) -> ClassProgressResponse:
    """Per chapter (and topic, where the bank tags one): how many of the
    section's students are at each status and their mean mastery; then the
    students with chapters at "needs work", most first. `subject` narrows to
    one bank subject (e.g. "Science")."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    cs = cr._require()
    section = cr._require_school_owns_section(section_id, current)
    if not _may_see(section, current):
        raise HTTPException(403, "only the section's teachers or the principal see its progress")
    users = cr._require_users()
    students = [e.student_id for e in cs.enrollments_for_section(section.id)]
    ops = store()
    rows: list[dict] = []
    for i in range(0, len(students), 500):
        chunk = students[i:i + 500]
        rows += ops._fetchall(f"SELECT * FROM learning_evidence WHERE student_id IN ({','.join('?' * len(chunk))})",
                              tuple(chunk))
    if subject:
        rows = [r for r in rows if r["subject"].lower() == subject.lower()]
    now = datetime.now(timezone.utc)
    by_chapter: dict[tuple[str, int, str], dict[str, list[dict]]] = {}
    by_topic: dict[tuple[str, int, str, str], dict[str, list[dict]]] = {}
    for r in rows:
        if r["chapter_id"] == "unmapped":
            continue
        ck = (r["subject"], r["grade"], r["chapter_id"])
        by_chapter.setdefault(ck, {}).setdefault(r["student_id"], []).append(r)
        if r["topic_id"]:
            by_topic.setdefault(ck + (r["topic_id"],), {}).setdefault(r["student_id"], []).append(r)
    chapters: list[ChapterProgressRow] = []
    help_by: dict[str, list[str]] = {}
    for (subj, grade, cid), per_student in by_chapter.items():
        names, order, topic_names = _taxonomy(subj, grade)
        cname = _chapter_name(cid, names)
        base = _unit(cid, cname, per_student, now)
        topics = [_unit(tid, topic_names.get(tid) or tid.rsplit("/", 1)[-1].replace("-", " ").capitalize(), ps, now)
                  for (s2, g2, c2, tid), ps in by_topic.items() if (s2, g2, c2) == (subj, grade, cid)]
        chapters.append(ChapterProgressRow(**base.model_dump(), subject=subj, grade=grade,
                                           topics=sorted(topics, key=lambda t: (t.mean_mastery or 0))))
        for sid, srows in per_student.items():
            if score(srows, now)["status"] == "needs_work":
                help_by.setdefault(sid, []).append(cname)
    chapters.sort(key=lambda c: (c.subject, c.mean_mastery if c.mean_mastery is not None else 1))
    answers = {}
    for r in rows:
        answers[r["student_id"]] = answers.get(r["student_id"], 0) + 1
    needing = []
    for sid, chs in help_by.items():
        u = users.get(sid)
        needing.append(StudentHelp(student_id=sid, name=u.name if u else sid, needs_work=sorted(chs),
                                   answers=answers.get(sid, 0)))
    needing.sort(key=lambda s: (-len(s.needs_work), s.name))
    record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what="class_progress",
                    student_ids=students, section_id=section.id)
    return ClassProgressResponse(section_id=section.id, section_name=cs._section_label(section),
                                 students=len(students), chapters=chapters, needing_help=needing)
