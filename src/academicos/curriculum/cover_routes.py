"""HTTP routes for leave, substitution and compensation (SCH-5..7,
curriculum/cover.py), and the day as it will actually run.

A teacher applies for their own leave, sees their substitution duties and
accepts or declines them. The principal applies on a teacher's behalf for a
same-day absence, decides leave, assigns or overrides substitutes, declares
closures, and schedules make-up periods. Every write is audited with before
and after (ROLE-3). Students read only their own section's day.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import STAFF_ROLES, get_current_user, holds, require_admin, require_principal, require_staff
from ..assessment.users import User
from . import routes as cr
from .cover import CoverError
from .schemas import Camel


def _notify(**kw: Any) -> None:
    from ..operations.routes import notify_safely
    notify_safely(**kw)

router = APIRouter(prefix="/api/v1/curriculum")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------- shapes ----------------

class LeaveApplyRequest(_Req):
    teacher_id: Optional[str] = None      # the principal only: applying on a teacher's behalf
    start_date: str
    end_date: str
    kind: str = "full_day"                # full_day | first_half | second_half | periods
    periods: Optional[list[int]] = None
    reason: str = Field(min_length=1, max_length=500)
    handover_note: Optional[str] = Field(default=None, max_length=2000)


class LeaveDecisionRequest(_Req):
    approve: bool


class LeaveResponse(Camel):
    id: str
    academic_year_id: str
    teacher_id: str
    start_date: str
    end_date: str
    kind: str
    periods: list[int]
    reason: str
    handover_note: Optional[str] = None
    status: str
    created_by: str
    decided_by: Optional[str] = None


class SubstitutionResponse(Camel):
    id: str
    leave_id: Optional[str] = None
    date: str
    section_id: str
    period: int
    subject_id: str
    absent_teacher_id: str
    substitute_id: Optional[str] = None
    status: str
    mode: str
    note: Optional[str] = None


class LeaveDecisionResponse(Camel):
    leave: LeaveResponse
    substitutions: list[SubstitutionResponse]


class CandidateResponse(Camel):
    teacher_id: str
    score: int
    qualified: bool
    reasons: list[str]


class AssignRequest(_Req):
    substitute_id: Optional[str] = None
    mode: str = "substitute"          # substitute | supervised | combined | lost
    note: Optional[str] = Field(default=None, max_length=500)


class RespondRequest(_Req):
    accept: bool


class LostPeriodResponse(Camel):
    id: str
    section_id: str
    subject_id: str
    date: str
    period: int
    reason: str
    status: str
    compensation_date: Optional[str] = None
    compensation_period: Optional[int] = None


class AssignResponse(Camel):
    substitution: SubstitutionResponse
    lost_period: Optional[LostPeriodResponse] = None


class SlotResponse(Camel):
    date: str
    period: int


class CompensateRequest(_Req):
    date: str
    period: int = Field(ge=1, le=20)


class WaiveRequest(_Req):
    # Why the school will not make the period up; kept in the audit log.
    reason: str = Field(min_length=1, max_length=500)


class ClosureRequest(_Req):
    date: str
    reason: str = Field(min_length=1, max_length=200)
    add_holiday: bool = True
    reflow_plans: bool = True
    # Some classes only (a trip, an exam hall): these sections' periods are
    # lost and their plans move; the school's calendar keeps the day.
    # Omitted: the whole school.
    section_ids: Optional[list[str]] = Field(default=None, min_length=1)


class ClosureResponse(Camel):
    lost_periods: int
    holiday_added: bool
    plans_reflowed: int


class DayRowResponse(Camel):
    section_id: str
    period: int
    subject_id: str
    teacher_id: Optional[str] = None
    # SCH-8: the other teacher in the room for a co-taught period, when both
    # are there. `co_teacher` as the kind: the co-teacher takes it alone.
    co_teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    kind: str        # regular | substitute | supervised | combined | lost | uncovered | away | makeup | co_teacher
    substitution_id: Optional[str] = None
    lost_period_id: Optional[str] = None
    note: Optional[str] = None
    # Names for the app and the web, so a student or parent (who cannot read
    # the curriculum) and a teacher see "10-B Science" without a second call.
    subject_name: Optional[str] = None
    section_name: Optional[str] = None
    teacher_name: Optional[str] = None
    co_teacher_name: Optional[str] = None


class DebtRowResponse(Camel):
    section_id: str
    subject_id: str
    owed: int
    compensated: int
    waived: int


class CoverSummaryResponse(Camel):
    substitutions_requested: int
    filled: int
    supervised: int
    unfilled: int
    lost: int
    compensated: int
    owed: int
    debt: list[DebtRowResponse]


# ---------------- helpers ----------------

def _audit(action: str, user: User, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=user.id,
                                            details={"schoolId": user.school_id, **details})


def _today() -> str:
    return cr._school_today().isoformat()


def _year_for(store, school_id: str, on: str):
    for y in store.academic_years_for_school(school_id):
        if y.start_date <= on <= y.end_date:
            return y
    raise HTTPException(422, "no academic year of this school contains that date")


def _current_year(store, school_id: str):
    years = store.academic_years_for_school(school_id)
    if not years:
        return None
    today = _today()
    for y in years:
        if y.start_date <= today <= y.end_date:
            return y
    return sorted(years, key=lambda y: y.start_date)[-1]


def _leave(l) -> LeaveResponse:
    return LeaveResponse(id=l.id, academic_year_id=l.academic_year_id, teacher_id=l.teacher_id,
                         start_date=l.start_date, end_date=l.end_date, kind=l.kind, periods=l.periods,
                         reason=l.reason, handover_note=l.handover_note, status=l.status,
                         created_by=l.created_by, decided_by=l.decided_by)


def _sub(s) -> SubstitutionResponse:
    return SubstitutionResponse(id=s.id, leave_id=s.leave_id, date=s.date, section_id=s.section_id,
                                period=s.period, subject_id=s.subject_id, absent_teacher_id=s.absent_teacher_id,
                                substitute_id=s.substitute_id, status=s.status, mode=s.mode, note=s.note)


def _lost(l) -> LostPeriodResponse:
    return LostPeriodResponse(id=l.id, section_id=l.section_id, subject_id=l.subject_id, date=l.date,
                              period=l.period, reason=l.reason, status=l.status,
                              compensation_date=l.compensation_date, compensation_period=l.compensation_period)


def _owned_leave(leave_id: str, user: User):
    l = cr._require().get_leave(leave_id)
    if l is None:
        raise HTTPException(404, "leave request not found")
    if l.school_id != user.school_id:
        raise HTTPException(403, "this leave request belongs to a different school")
    return l


def _owned_sub(sub_id: str, user: User):
    s = cr._require().get_substitution(sub_id)
    if s is None:
        raise HTTPException(404, "substitution not found")
    if s.school_id != user.school_id:
        raise HTTPException(403, "this substitution belongs to a different school")
    return s


def _owned_lost(lost_id: str, user: User):
    l = cr._require().get_lost_period(lost_id)
    if l is None:
        raise HTTPException(404, "lost period not found")
    if l.school_id != user.school_id:
        raise HTTPException(403, "this lost period belongs to a different school")
    return l


def _staff_of_school(user_id: str, principal: User) -> None:
    from ..assessment.auth_routes import STAFF_ROLES
    target = cr._require_users().get(user_id)
    if target is None:
        raise HTTPException(404, "teacher not found")
    if target.school_id != principal.school_id:
        raise HTTPException(403, "that user belongs to a different school")
    if target.role not in STAFF_ROLES:
        raise HTTPException(422, "only a teacher or the principal takes leave or substitutes")


# ---------------- leave (SCH-5) ----------------

@router.post("/leave-requests", response_model=LeaveResponse)
def apply_for_leave(req: LeaveApplyRequest, current: User = Depends(require_staff)) -> LeaveResponse:
    """A teacher applies for their own leave. The principal may apply on a
    teacher's behalf (a same-day absence reported by phone)."""
    store = cr._require()
    teacher_id = current.id
    if req.teacher_id and req.teacher_id != current.id:
        if current.role != "principal":
            raise HTTPException(403, "only the principal applies for someone else's leave")
        _staff_of_school(req.teacher_id, current)
        teacher_id = req.teacher_id
    year = _year_for(store, current.school_id, req.start_date)
    try:
        leave = store.apply_for_leave(school_id=current.school_id, academic_year_id=year.id,
                                      teacher_id=teacher_id, start_date=req.start_date, end_date=req.end_date,
                                      kind=req.kind, reason=req.reason, created_by=current.id,
                                      periods=req.periods, handover_note=req.handover_note)
    except CoverError as e:
        raise HTTPException(422, str(e))
    _audit("leave_applied", current, {"leaveId": leave.id, "teacherId": teacher_id,
                                      "from": leave.start_date, "to": leave.end_date, "kind": leave.kind})
    return _leave(leave)


@router.get("/leave-requests", response_model=list[LeaveResponse])
def list_leave(status: Optional[str] = None, current: User = Depends(require_staff)) -> list[LeaveResponse]:
    """The principal sees the current year's requests; a teacher their own."""
    store = cr._require()
    if holds(current, "leave"):
        year = _current_year(store, current.school_id)
        rows = store.leave_for_year(year.id, status) if year else []
    else:
        rows = [l for l in store.leave_for_teacher(current.id) if status is None or l.status == status]
    return [_leave(l) for l in rows]


@router.post("/leave-requests/{leave_id}/decision", response_model=LeaveDecisionResponse)
def decide_leave(leave_id: str, req: LeaveDecisionRequest,
                 principal: User = Depends(require_admin("leave"))) -> LeaveDecisionResponse:
    """Approve (every missed period gets a substitution with a proposed
    substitute) or reject."""
    _owned_leave(leave_id, principal)
    try:
        leave, subs = cr._require().decide_leave(leave_id, approve=req.approve, decided_by=principal.id)
    except CoverError as e:
        raise HTTPException(409, str(e))
    _audit("leave_decided", principal, {"leaveId": leave_id, "before": "pending", "after": leave.status,
                                        "substitutions": len(subs)})
    _notify(school_id=principal.school_id, user_ids=[leave.teacher_id], kind="leave_decided",
            params={"decision": leave.status, "start": leave.start_date, "end": leave.end_date,
                    "decision_hi": "स्वीकृत" if leave.status == "approved" else "अस्वीकृत"},
            link="/leave", dedupe_key=f"leave:{leave.id}:{leave.status}")
    for s in subs:
        _notify_proposed(s)
    return LeaveDecisionResponse(leave=_leave(leave), substitutions=[_sub(s) for s in subs])


@router.post("/leave-requests/{leave_id}/cancel", response_model=LeaveResponse)
def cancel_leave(leave_id: str, current: User = Depends(require_staff)) -> LeaveResponse:
    leave = _owned_leave(leave_id, current)
    if current.role != "principal" and leave.teacher_id != current.id:
        raise HTTPException(403, "only the teacher or the principal cancels this leave")
    try:
        after = cr._require().cancel_leave(leave_id, today=_today())
    except CoverError as e:
        raise HTTPException(409, str(e))
    _audit("leave_cancelled", current, {"leaveId": leave_id, "before": leave.status, "after": after.status})
    return _leave(after)


# ---------------- substitution (SCH-6) ----------------

def _notify_proposed(s) -> None:
    """Tell a proposed substitute: it is their next action (accept or decline)."""
    if s.status != "proposed" or not s.substitute_id:
        return
    store = cr._require()
    section = store.get_section(s.section_id)
    subject = store.get_subject(s.subject_id)
    _notify(school_id=s.school_id, user_ids=[s.substitute_id], kind="substitution_proposed",
            params={"section": store._section_label(section) if section else "", "period": s.period,
                    "date": s.date, "subject": subject.name if subject else ""},
            link=f"/my-day?date={s.date}", dedupe_key=f"sub:{s.id}:{s.substitute_id}")


@router.get("/substitutions", response_model=list[SubstitutionResponse])
def list_substitutions(from_date: str = Query(alias="from"), to_date: str = Query(alias="to"),
                       current: User = Depends(require_staff)) -> list[SubstitutionResponse]:
    """The principal: every substitution in the range. A teacher: their duties."""
    store = cr._require()
    if not holds(current, "leave"):
        return [_sub(s) for s in store.duties_for(current.id, from_date, to_date)]
    year = _current_year(store, current.school_id)
    if year is None:
        return []
    store.settle_past_uncovered(year.id, _today())
    return [_sub(s) for s in store.substitutions_between(year.id, from_date, to_date)]


@router.get("/substitutions/{sub_id}/candidates", response_model=list[CandidateResponse])
def substitution_candidates(sub_id: str, principal: User = Depends(require_admin("leave"))) -> list[CandidateResponse]:
    """Who is free for this period, best first, with why."""
    _owned_sub(sub_id, principal)
    return [CandidateResponse(teacher_id=c.teacher_id, score=c.score, qualified=c.qualified, reasons=c.reasons)
            for c in cr._require().substitute_candidates(sub_id)]


@router.post("/substitutions/{sub_id}/assign", response_model=AssignResponse)
def assign_substitute(sub_id: str, req: AssignRequest,
                      principal: User = Depends(require_admin("leave"))) -> AssignResponse:
    """Name the substitute, or run the period without one (supervised,
    combined, lost). A period no teacher of the subject takes is a lost
    period for the section and subject."""
    before = _owned_sub(sub_id, principal)
    if req.substitute_id:
        _staff_of_school(req.substitute_id, principal)
    try:
        after, lost = cr._require().assign_substitute(sub_id, substitute_id=req.substitute_id, mode=req.mode,
                                                      note=req.note)
    except CoverError as e:
        raise HTTPException(422, str(e))
    _audit("substitution_assigned", principal,
           {"substitutionId": sub_id, "before": {"substituteId": before.substitute_id, "mode": before.mode},
            "after": {"substituteId": after.substitute_id, "mode": after.mode},
            "lostPeriodId": lost.id if lost else None})
    _notify_proposed(after)
    return AssignResponse(substitution=_sub(after), lost_period=_lost(lost) if lost else None)


@router.post("/substitutions/{sub_id}/respond", response_model=SubstitutionResponse)
def respond_to_substitution(sub_id: str, req: RespondRequest,
                            current: User = Depends(require_staff)) -> SubstitutionResponse:
    """The proposed substitute accepts or declines; a decline passes the
    period to the next candidate."""
    _owned_sub(sub_id, current)
    try:
        after = cr._require().respond_to_substitution(sub_id, teacher_id=current.id, accept=req.accept)
    except CoverError as e:
        raise HTTPException(409, str(e))
    _audit("substitution_answered", current, {"substitutionId": sub_id, "accepted": req.accept})
    if not req.accept:
        _notify_proposed(after)          # the next candidate
    return _sub(after)


# ---------------- lost periods and compensation (SCH-7) ----------------

@router.get("/academic-years/{academic_year_id}/lost-periods", response_model=list[LostPeriodResponse])
def list_lost_periods(academic_year_id: str, status: Optional[str] = None,
                      current: User = Depends(require_staff)) -> list[LostPeriodResponse]:
    cr._require_school_owns_academic_year(academic_year_id, current)
    store = cr._require()
    store.settle_past_uncovered(academic_year_id, _today())
    return [_lost(l) for l in store.lost_periods_for_year(academic_year_id, status)]


@router.get("/lost-periods/{lost_id}/compensation-options", response_model=list[SlotResponse])
def compensation_options(lost_id: str, principal: User = Depends(require_admin("leave"))) -> list[SlotResponse]:
    """The section's free periods over the next three weeks when its subject
    teacher is free and under the day's maximum."""
    _owned_lost(lost_id, principal)
    return [SlotResponse(date=d, period=p)
            for d, p in cr._require().compensation_options(lost_id, today=_today())]


@router.post("/lost-periods/{lost_id}/compensate", response_model=LostPeriodResponse)
def compensate(lost_id: str, req: CompensateRequest,
               principal: User = Depends(require_admin("leave"))) -> LostPeriodResponse:
    before = _owned_lost(lost_id, principal)
    try:
        after = cr._require().compensate(lost_id, on=req.date, period=req.period, today=_today())
    except CoverError as e:
        raise HTTPException(409, str(e))
    _audit("lost_period_compensated", principal, {"lostPeriodId": lost_id, "before": before.status,
                                                  "after": {"date": req.date, "period": req.period}})
    return _lost(after)


@router.post("/lost-periods/{lost_id}/waive", response_model=LostPeriodResponse)
def waive(lost_id: str, req: Optional[WaiveRequest] = None,
          principal: User = Depends(require_admin("leave"))) -> LostPeriodResponse:
    _owned_lost(lost_id, principal)
    try:
        after = cr._require().waive_lost(lost_id)
    except CoverError as e:
        raise HTTPException(409, str(e))
    _audit("lost_period_waived", principal,
           {"lostPeriodId": lost_id, "reason": req.reason if req is not None else None})
    return _lost(after)


@router.post("/academic-years/{academic_year_id}/closures", response_model=ClosureResponse)
def declare_closure(academic_year_id: str, req: ClosureRequest,
                    principal: User = Depends(require_admin("calendar"))) -> ClosureResponse:
    """A day lost at short notice, or after the fact (rain, an event, an exam
    day): every section's periods that day become lost periods, the day's
    cover is cancelled, the holiday goes on the calendar and the section
    plans move past it. With `sectionIds`, only those sections close: the
    calendar keeps the day (the rest of the school teaches) and only their
    plans move."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    store = cr._require()
    only: Optional[set[str]] = None
    if req.section_ids is not None:
        for sid in req.section_ids:
            section = cr._require_school_owns_section(sid, principal)
            if section.academic_year_id != academic_year_id:
                raise HTTPException(422, "that section is not in this academic year")
        only = set(req.section_ids)
    try:
        date.fromisoformat(req.date)
        lost = store.declare_closure(academic_year_id, req.date, reason=req.reason, section_ids=only)
    except (CoverError, ValueError) as e:
        raise HTTPException(422, str(e))
    holiday_added = False
    if req.add_holiday and only is None:
        cal = store.get_calendar_for_year(academic_year_id)
        if cal is not None and not any(h.date == req.date for h in store.holidays_for_calendar(cal.id)):
            store.add_holiday(calendar_id=cal.id, date=req.date, label=req.reason, kind="school")
            holiday_added = True
    reflowed = 0
    if req.reflow_plans:
        from ..assessment.audit_log import get_audit_log
        from . import scheduling as scheduling_mod
        plans = {(r["book_id"], r["section_id"]) for r in store._fetchall(
            "SELECT DISTINCT book_id, section_id FROM scheduled_lessons WHERE academic_year_id=? AND date=? "
            "AND status='scheduled'", (academic_year_id, req.date))}
        if only is not None:
            # The school-wide plan keeps the day: the other classes still teach.
            plans = {(b, s) for b, s in plans if s in only}
        for book_id, section_id in plans:
            try:
                scheduling_mod.push_lessons_after(
                    store, get_audit_log(cr._cfg.data_root), academic_year_id=academic_year_id, book_id=book_id,
                    from_date=req.date, reason=f"closure: {req.reason}", changed_by=principal.id,
                    section_id=section_id)
                reflowed += 1
            except ValueError:
                continue
    _audit("closure_declared", principal, {"academicYearId": academic_year_id, "date": req.date,
                                           "reason": req.reason, "lostPeriods": len(lost),
                                           "plansReflowed": reflowed, "sectionIds": req.section_ids})
    return ClosureResponse(lost_periods=len(lost), holiday_added=holiday_added, plans_reflowed=reflowed)


# ---------------- the day ----------------

def _row(r: dict) -> DayRowResponse:
    store = cr._require()
    subject = store.get_subject(r["subjectId"])
    section = store.get_section(r["sectionId"])
    users = cr._require_users()
    teacher = users.get(r["teacherId"]) if r.get("teacherId") else None
    co = users.get(r["coTeacherId"]) if r.get("coTeacherId") else None
    return DayRowResponse(section_id=r["sectionId"], period=r["period"], subject_id=r["subjectId"],
                          teacher_id=r["teacherId"], co_teacher_id=r.get("coTeacherId"), room_id=r["roomId"],
                          kind=r["kind"], substitution_id=r["substitutionId"], lost_period_id=r["lostPeriodId"],
                          note=r["note"], subject_name=subject.name if subject else None,
                          section_name=store._section_label(section) if section else None,
                          teacher_name=teacher.name if teacher else None,
                          co_teacher_name=co.name if co else None)


@router.get("/day", response_model=list[DayRowResponse])
def day(on: str = Query(alias="date"), section_id: Optional[str] = Query(default=None, alias="sectionId"),
        current: User = Depends(get_current_user)) -> list[DayRowResponse]:
    """A day as it will run, with substitutions, lost and make-up periods.
    Staff: the school or one section. A student: their own section."""
    store = cr._require()
    try:
        year = _year_for(store, current.school_id, on)
    except HTTPException:
        return []
    if current.role == "student":
        enrollment = store.enrollment_for_student(current.id)
        if enrollment is None or not enrollment.section_id:
            return []
        if section_id and section_id != enrollment.section_id:
            raise HTTPException(403, "a student sees their own section's day")
        section_id = enrollment.section_id
    elif section_id:
        cr._require_school_owns_section(section_id, current)
    return [_row(r) for r in store.day_view(year.id, on, section_id=section_id)]


@router.get("/my-day", response_model=list[DayRowResponse])
def my_day(on: str = Query(alias="date"), current: User = Depends(get_current_user)) -> list[DayRowResponse]:
    """The caller's own day: a teacher's periods, duties and make-ups (their
    periods on leave marked away); a student's section's day."""
    store = cr._require()
    try:
        year = _year_for(store, current.school_id, on)
    except HTTPException:
        return []
    if current.role == "student":
        enrollment = store.enrollment_for_student(current.id)
        if enrollment is None or not enrollment.section_id:
            return []
        return [_row(r) for r in store.day_view(year.id, on, section_id=enrollment.section_id)]
    return [_row(r) for r in store.day_view(year.id, on, teacher_id=current.id)]


# ---------------- teacher attendance (SCH-9) ----------------

class AttendanceMark(_Req):
    teacher_id: str
    status: str
    note: Optional[str] = Field(default=None, max_length=300)


class AttendanceRequest(_Req):
    marks: list[AttendanceMark] = Field(min_length=1, max_length=500)


class AttendanceRow(Camel):
    teacher_id: str
    name: str
    status: str                    # present | late | absent | first_half_absent | second_half_absent | on_leave | unmarked
    note: Optional[str] = None
    leave_id: Optional[str] = None
    marked_at: Optional[str] = None


class AttendanceResponse(Camel):
    date: str
    rows: list[AttendanceRow]
    substitutions: list[SubstitutionResponse] = []


def _attendance(on: str, current: User, opened: Optional[list] = None) -> AttendanceResponse:
    store = cr._require()
    marks = store.attendance_for_date(current.school_id, on)
    rows = []
    for u in sorted(cr._require_users().users_for_school(current.school_id), key=lambda u: u.name.lower()):
        if u.role not in STAFF_ROLES:
            continue
        m = marks.get(u.id)
        on_leave = any(l.status == "approved" and l.start_date <= on <= l.end_date
                       for l in store.leave_for_teacher(u.id))
        status = m["status"] if m else ("on_leave" if on_leave else "unmarked")
        rows.append(AttendanceRow(teacher_id=u.id, name=u.name, status=status, note=m["note"] if m else None,
                                  leave_id=m["leave_id"] if m else None, marked_at=m["marked_at"] if m else None))
    return AttendanceResponse(date=on, rows=rows, substitutions=[_sub(s) for s in (opened or [])])


@router.get("/teacher-attendance", response_model=AttendanceResponse)
def teacher_attendance(on: Optional[str] = Query(default=None, alias="date"),
                       principal: User = Depends(require_admin("leave"))) -> AttendanceResponse:
    """Every staff member's attendance for a day (today by default): as
    marked, on leave, or not marked yet."""
    return _attendance(on or _today(), principal)


@router.put("/teacher-attendance", response_model=AttendanceResponse)
def mark_teacher_attendance(req: AttendanceRequest, on: Optional[str] = Query(default=None, alias="date"),
                            principal: User = Depends(require_admin("leave"))) -> AttendanceResponse:
    """Mark attendance. An absence with no leave opens same-day cover at
    once -- every missed period gets a substitution and a proposed
    substitute, who is told -- and the result lists them."""
    day = on or _today()
    store = cr._require()
    for m in req.marks:
        _staff_of_school(m.teacher_id, principal)
    year = _year_for(store, principal.school_id, day)
    opened = []
    for m in req.marks:
        before = store.attendance_for_date(principal.school_id, day).get(m.teacher_id)
        try:
            _, subs = store.mark_attendance(school_id=principal.school_id, academic_year_id=year.id, on=day,
                                            teacher_id=m.teacher_id, status=m.status, marked_by=principal.id,
                                            note=m.note)
        except CoverError as e:
            raise HTTPException(422, str(e))
        _audit("teacher_attendance_marked", principal, {"teacherId": m.teacher_id, "date": day,
                                                         "before": before["status"] if before else None,
                                                         "after": m.status, "substitutions": len(subs)})
        for s in subs:
            _notify_proposed(s)
        opened.extend(subs)
    return _attendance(day, principal, opened)


@router.post("/my-attendance", response_model=AttendanceRow)
def check_in(current: User = Depends(require_staff)) -> AttendanceRow:
    """A teacher marks themselves present today (arrival). It never
    overrides an absence the principal has marked."""
    day = _today()
    store = cr._require()
    existing = store.attendance_for_date(current.school_id, day).get(current.id)
    if existing is None or existing["status"] in ("present", "late"):
        year = _year_for(store, current.school_id, day)
        store.mark_attendance(school_id=current.school_id, academic_year_id=year.id, on=day, teacher_id=current.id,
                              status="present", marked_by=current.id)
    return next(r for r in _attendance(day, current).rows if r.teacher_id == current.id)


@router.get("/academic-years/{academic_year_id}/cover-summary", response_model=CoverSummaryResponse)
def cover_summary(academic_year_id: str, from_date: str = Query(alias="from"), to_date: str = Query(alias="to"),
                  principal: User = Depends(require_admin("reports"))) -> CoverSummaryResponse:
    """Substitutions requested, filled, supervised and unfilled; periods lost,
    owed and made up; the debt per section and subject (ADM-2)."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    store = cr._require()
    store.settle_past_uncovered(academic_year_id, _today())
    s = store.cover_summary(academic_year_id, from_date, to_date)
    return CoverSummaryResponse(
        substitutions_requested=s["substitutionsRequested"], filled=s["filled"], supervised=s["supervised"],
        unfilled=s["unfilled"], lost=s["lost"], compensated=s["compensated"], owed=s["owed"],
        debt=[DebtRowResponse(section_id=d["sectionId"], subject_id=d["subjectId"], owed=d["owed"],
                              compensated=d["compensated"], waived=d["waived"]) for d in s["debt"]])
