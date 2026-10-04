"""The principal's one-screen summary (ADM-2): today's staff and cover, this
week's homework, exams coming up, and the students who need help -- each a
count with the list behind it one tap away in the workspace."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends

from ..assessment.auth_routes import require_admin
from ..assessment.users import User
from ..curriculum import calendar as calendar_mod
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .routes import store

router = APIRouter(prefix="/api/v1")


class CoverToday(Camel):
    staff: int
    absent: int
    on_leave: int
    unmarked: int
    periods_to_cover: int
    covered: int
    uncovered: int
    # False on a weekly off, a holiday or a closure: no register is expected,
    # so nothing is "not marked" (found 2026-10-04: a Sunday showed 8).
    school_day: bool = True


class HomeworkWeek(Camel):
    set: int
    submissions_expected: int
    submitted: int
    marked: int
    waiting_to_mark: int
    overdue_homework: int


class ExamSoon(Camel):
    id: str
    name: str
    start_date: str
    status: str


class StudentConcern(Camel):
    student_id: str
    name: str
    section_name: Optional[str] = None
    needs_work: int              # chapters at "needs work"
    overdue: int                 # homework past due, not submitted


class DashboardSummary(Camel):
    date: str
    cover: CoverToday
    homework: HomeworkWeek
    exams: list[ExamSoon]
    students_needing_help: list[StudentConcern]


def _cover(today: str, current: User) -> CoverToday:
    cs = cr._require()
    staff = [u for u in cr._require_users().users_for_school(current.school_id) if u.role in ("teacher", "principal")]
    marks = cs.attendance_for_date(current.school_id, today)
    on_leave = {u.id for u in staff for l in cs.leave_for_teacher(u.id)
                if l.status == "approved" and l.start_date <= today <= l.end_date}
    absent = sum(1 for m in marks.values() if m["status"] in ("absent", "first_half_absent", "second_half_absent"))
    year = next((y for y in cs.academic_years_for_school(current.school_id) if y.start_date <= today <= y.end_date), None)
    subs = [s for s in cs.substitutions_between(year.id, today, today) if s.status != "cancelled"] if year else []
    covered = sum(1 for s in subs if s.status in ("accepted", "resolved"))
    school_day = True
    if year is not None:
        try:
            school_day = today in set(calendar_mod.school_days(cs, year.id).dates)
        except ValueError:      # no calendar yet: a register may still be kept
            school_day = True
    return CoverToday(staff=len(staff), absent=absent, on_leave=len(on_leave - set(marks)),
                      unmarked=sum(1 for u in staff if u.id not in marks and u.id not in on_leave) if school_day else 0,
                      periods_to_cover=len(subs), covered=covered, uncovered=len(subs) - covered,
                      school_day=school_day)


def _homework(today: date, current: User) -> HomeworkWeek:
    ops = store()
    cs = cr._require()
    monday = today - timedelta(days=today.weekday())
    items = ops.homework_for_school(current.school_id)
    week = [h for h in items if h.published_at and h.published_at[:10] >= monday.isoformat()]
    expected = submitted = marked = 0
    for h in week:
        roster = sum(len(cs.enrollments_for_section(s)) for s in h.section_ids)
        counts = ops.submission_counts(h.id)
        expected += roster
        submitted += counts.get("submitted", 0) + counts.get("graded", 0)
        marked += counts.get("graded", 0)
    waiting = sum(ops.submission_counts(h.id).get("submitted", 0) for h in items if h.status != "draft")
    overdue = 0
    for h in items:
        if h.status != "published" or h.due_date >= today.isoformat():
            continue
        roster = sum(len(cs.enrollments_for_section(s)) for s in h.section_ids)
        c = ops.submission_counts(h.id)
        if roster > c.get("submitted", 0) + c.get("graded", 0):
            overdue += 1
    return HomeworkWeek(set=len(week), submissions_expected=expected, submitted=submitted, marked=marked,
                        waiting_to_mark=waiting, overdue_homework=overdue)


def _concerns(today: str, current: User, limit: int = 10) -> list[StudentConcern]:
    """Students with chapters at "needs work" or two or more overdue
    homeworks, worst first. One pass over the evidence and one per
    section's homework, so a large school stays quick."""
    from datetime import datetime, timezone
    from .learning import score
    ops = store()
    cs = cr._require()
    students = {u.id: u for u in cr._require_users().users_for_school(current.school_id, role="student")}
    if not students:
        return []
    ids = list(students)
    rows_by: dict[tuple[str, str], list[dict]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in ops._fetchall(f"SELECT * FROM learning_evidence WHERE student_id IN ({marks})", tuple(chunk)):
            if r["chapter_id"] != "unmapped":
                rows_by.setdefault((r["student_id"], f"{r['subject']}|{r['grade']}|{r['chapter_id']}"), []).append(r)
    now = datetime.now(timezone.utc)
    needs: dict[str, int] = {}
    for (sid, _), rows in rows_by.items():
        if score(rows, now)["status"] == "needs_work":
            needs[sid] = needs.get(sid, 0) + 1
    overdue: dict[str, int] = {}
    section_of: dict[str, object] = {}
    for sid in ids:
        e = cs.enrollment_for_student(sid)
        if e and e.section_id:
            section_of[sid] = e.section_id
    for section_id in set(section_of.values()):
        members = [sid for sid, sec in section_of.items() if sec == section_id]
        for h in ops.homework_for_section(section_id):
            if h.status != "published" or h.due_date >= today:
                continue
            done = {s.student_id for s in ops.submissions_for(h.id)}
            for sid in members:
                if sid not in done:
                    overdue[sid] = overdue.get(sid, 0) + 1
    out = []
    for sid, u in students.items():
        n, o = needs.get(sid, 0), overdue.get(sid, 0)
        if n or o >= 2:
            section = cs.get_section(section_of[sid]) if sid in section_of else None
            out.append(StudentConcern(student_id=sid, name=u.name,
                                      section_name=cs._section_label(section) if section else None,
                                      needs_work=n, overdue=o))
    out.sort(key=lambda c: (-(c.needs_work + 2 * c.overdue), c.name))
    return out[:limit]


@router.get("/dashboard/summary", response_model=DashboardSummary)
def summary(principal: User = Depends(require_admin("reports"))) -> DashboardSummary:
    """Today at a glance, for the principal (or a reports admin)."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    today = cr._school_today()
    exams = [ExamSoon(id=e["id"], name=e["name"], start_date=e["start_date"], status=e["status"])
             for e in store().exams_for_school(principal.school_id)
             if e["end_date"] >= today.isoformat() and e["start_date"] <= (today + timedelta(days=30)).isoformat()]
    concerns = _concerns(today.isoformat(), principal)
    if concerns:
        record_pii_read(get_audit_log(cr._cfg.data_root), actor=principal.id, what="dashboard_concerns",
                        student_ids=[c.student_id for c in concerns])
    return DashboardSummary(date=today.isoformat(), cover=_cover(today.isoformat(), principal),
                            homework=_homework(today, principal), exams=sorted(exams, key=lambda x: x.start_date),
                            students_needing_help=concerns)
