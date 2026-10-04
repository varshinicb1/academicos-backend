"""HTTP routes for parents and guardians (M1.5, SA-6; operations/guardians.py).

The principal invites a guardian for one student, links an existing parent
account to another child, and ends a link. Staff see a student's guardians.
A parent sees only the children they are linked to: the child's week, day,
homework and learning progress, and gives or withdraws the parental consent
(DPDP) that lets the school process the child's work. Every other route
refuses a parent account (auth_routes.get_current_user, PARENT_PATHS).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user, require_admin, require_principal, require_staff
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from ..curriculum.cover_routes import DayRowResponse
from ..curriculum.school_model_routes import MyTimetableResponse
from .guardians import RELATIONS, Guardianship
from .calendar_feed import CalendarItem
from .homework_routes import MyHomeworkDetail, MyHomeworkItem, SubmitRequest
from .learning_routes import LearningProgressResponse
from .routes import store

router = APIRouter(prefix="/api/v1")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class GuardianInviteRequest(_Req):
    relation: str = "guardian"
    email: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=90)


class GuardianInviteResponse(Camel):
    code: str
    student_id: str
    relation: str
    email: Optional[str] = None
    expires_at: str


class LinkRequest(_Req):
    parent_id: str
    student_id: str
    relation: str = "guardian"


class GuardianResponse(Camel):
    parent_id: str
    student_id: str
    name: str
    email: str
    relation: str
    active: bool
    created_at: str
    revoked_at: Optional[str] = None


class ChildResponse(Camel):
    student_id: str
    name: str
    relation: str
    section_id: Optional[str] = None
    section_name: Optional[str] = None
    consent: str                   # granted | revoked | none
    # The child's class; 1-5 is a young learner whose parent does the app
    # with them (SA-7).
    grade: Optional[int] = None


class ChildTimetableResponse(MyTimetableResponse):
    subject_names: dict[str, str] = Field(default_factory=dict)


class ChildDayResponse(Camel):
    rows: list[DayRowResponse]
    subject_names: dict[str, str] = Field(default_factory=dict)


def _subject_names(ids) -> dict[str, str]:
    cs = cr._require()
    out = {}
    for sid in set(ids):
        s = cs.get_subject(sid)
        if s is not None:
            out[sid] = s.name
    return out


class ConsentResponse(Camel):
    student_id: str
    status: str                    # granted | revoked | none
    guardian_name: Optional[str] = None
    method: Optional[str] = None
    recorded_at: Optional[str] = None
    revoked_at: Optional[str] = None


# ---------------- helpers ----------------

def _audit(action: str, user: User, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=user.id, details={"schoolId": user.school_id, **details})


def _log_read(current: User, what: str, student_id: str) -> None:
    from ..assessment.audit_log import get_audit_log, record_pii_read
    record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what=what, student_id=student_id)


def _consents():
    from ..assessment.consent import get_consent_store
    return get_consent_store(cr._cfg.data_root)


def _student_of_school(student_id: str, current: User) -> User:
    from ..assessment.authz import require_school_owns_student
    student = require_school_owns_student(cr._require_users(), student_id, current)
    if student.role != "student":
        raise HTTPException(422, "that account is not a student")
    return student


def _parent_of_school(parent_id: str, current: User) -> User:
    parent = cr._require_users().get(parent_id)
    if parent is None:
        raise HTTPException(404, "parent account not found")
    if parent.school_id != current.school_id:
        raise HTTPException(403, "this account belongs to a different school")
    if parent.role != "parent":
        raise HTTPException(422, "that account is not a parent account")
    return parent


def _my_child(student_id: str, current: User) -> User:
    """The child, when the caller is a parent actively linked to them. 404
    otherwise -- an unlinked parent learns nothing about who exists."""
    if current.role != "parent":
        raise HTTPException(403, "this endpoint is for parent accounts")
    if not store().is_guardian(current.id, student_id):
        raise HTTPException(404, "no linked child with that id")
    child = cr._require_users().get(student_id)
    if child is None:
        raise HTTPException(404, "no linked child with that id")
    return child


def _section_of(student_id: str):
    cs = cr._require()
    e = cs.enrollment_for_student(student_id)
    return cs.get_section(e.section_id) if e and e.section_id else None


def _consent_status(school_id: str, student_id: str) -> tuple[str, Any]:
    rec = _consents().get_consent(school_id, student_id)
    return (rec.status if rec else "none"), rec


def _guardian(g: Guardianship) -> GuardianResponse:
    u = cr._require_users().get(g.parent_id)
    return GuardianResponse(parent_id=g.parent_id, student_id=g.student_id, name=u.name if u else "",
                            email=u.email if u else "", relation=g.relation, active=g.active,
                            created_at=g.created_at, revoked_at=g.revoked_at)


def on_register(user: User, invite_code: Optional[str]) -> None:
    """Called after any registration (auth_routes register hooks).

    - A student who joined with an imported invite is enrolled in its section.
    - A parent who joined with a guardian invite is linked to that invite's
      child, or, when the child has not joined yet (bulk import), waits for
      them by email.
    - A student who joins is linked to every parent already waiting for them.
    """
    s = store()
    if user.role == "student":
        if invite_code:
            pending = s.pending_enrollment(invite_code)
            if pending and pending["school_id"] == user.school_id:
                cr._require().enroll_student(school_id=user.school_id, student_id=user.id,
                                             section_id=pending["section_id"])
        for p in s.take_pending_links(user.school_id, user.email):
            s.link_guardian(parent_id=p["parent_id"], student_id=user.id, school_id=user.school_id,
                            relation=p["relation"], created_by=p["created_by"])
        return
    if user.role != "parent" or not invite_code:
        return
    inv = s.guardian_invite(invite_code)
    if inv is None or inv["school_id"] != user.school_id:
        return
    student_id = inv["student_id"]
    if student_id is None and inv["student_email"]:
        child = cr._require_users().get_by_email(inv["student_email"])
        if child is not None and child.school_id == user.school_id and child.role == "student":
            student_id = child.id
        else:
            s.remember_pending_link(parent_id=user.id, school_id=user.school_id, student_email=inv["student_email"],
                                    relation=inv["relation"], created_by=inv["created_by"])
            return
    s.link_guardian(parent_id=user.id, student_id=student_id, school_id=user.school_id,
                    relation=inv["relation"], created_by=inv["created_by"])


# ---------------- school side ----------------

@router.post("/students/{student_id}/guardian-invites", response_model=GuardianInviteResponse)
def invite_guardian(student_id: str, req: GuardianInviteRequest,
                    principal: User = Depends(require_admin("users"))) -> GuardianInviteResponse:
    """A single-use invite for one of this student's guardians. Whoever
    registers with it gets a parent account linked to this student."""
    _student_of_school(student_id, principal)
    if req.relation.strip().lower() not in RELATIONS:
        raise HTTPException(422, f"relation must be one of {', '.join(RELATIONS)}")
    try:
        invite = cr._require_users().create_invite(school_id=principal.school_id, role="parent",
                                                   created_by=principal.id, email=req.email,
                                                   expires_in_days=req.expires_in_days)
    except ValueError as e:
        raise HTTPException(400, str(e))
    store().remember_guardian_invite(code=invite.code, school_id=principal.school_id, student_id=student_id,
                                     relation=req.relation, created_by=principal.id)
    _audit("guardian_invited", principal, {"studentId": student_id, "relation": req.relation.strip().lower(),
                                          "email": req.email or ""})
    return GuardianInviteResponse(code=invite.code, student_id=student_id, relation=req.relation.strip().lower(),
                                  email=invite.email or None, expires_at=invite.expires_at)


@router.get("/students/{student_id}/guardians", response_model=list[GuardianResponse])
def list_guardians(student_id: str, include_ended: bool = Query(default=False, alias="includeEnded"),
                   current: User = Depends(require_staff)) -> list[GuardianResponse]:
    _student_of_school(student_id, current)
    return [_guardian(g) for g in store().guardians_of(student_id, include_ended=include_ended)]


@router.post("/guardianships", response_model=GuardianResponse)
def link(req: LinkRequest, principal: User = Depends(require_admin("users"))) -> GuardianResponse:
    """Link an existing parent account to a student, e.g. a second child."""
    _student_of_school(req.student_id, principal)
    _parent_of_school(req.parent_id, principal)
    before = store().guardianship(req.parent_id, req.student_id)
    try:
        g = store().link_guardian(parent_id=req.parent_id, student_id=req.student_id,
                                  school_id=principal.school_id, relation=req.relation, created_by=principal.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("guardian_linked", principal, {"parentId": req.parent_id, "studentId": req.student_id,
                                         "before": before.relation if before and before.active else None,
                                         "after": g.relation})
    return _guardian(g)


@router.delete("/guardianships/{parent_id}/{student_id}", response_model=GuardianResponse)
def unlink(parent_id: str, student_id: str, principal: User = Depends(require_admin("users"))) -> GuardianResponse:
    _student_of_school(student_id, principal)
    _parent_of_school(parent_id, principal)
    g = store().end_guardianship(parent_id, student_id, revoked_by=principal.id)
    if g is None:
        raise HTTPException(404, "no active link between this parent and student")
    _audit("guardian_unlinked", principal, {"parentId": parent_id, "studentId": student_id,
                                           "before": g.relation, "after": None})
    return _guardian(g)


# ---------------- parent side ----------------

@router.get("/my-children", response_model=list[ChildResponse])
def my_children(current: User = Depends(get_current_user)) -> list[ChildResponse]:
    if current.role != "parent":
        raise HTTPException(403, "this endpoint is for parent accounts")
    cs = cr._require()
    users = cr._require_users()
    out = []
    for g in store().children_of(current.id):
        child = users.get(g.student_id)
        if child is None:
            continue
        section = _section_of(g.student_id)
        out.append(ChildResponse(student_id=child.id, name=child.name, relation=g.relation,
                                 section_id=section.id if section else None,
                                 section_name=cs._section_label(section) if section else None,
                                 consent=_consent_status(child.school_id, child.id)[0],
                                 grade=_grade_of(section)))
    return out


YOUNG_UP_TO = 5     # classes 1-5: the parent does the app with the child (SA-7)


def _grade_of(section) -> Optional[int]:
    if section is None:
        return None
    g = cr._require().get_grade(section.grade_id)
    return g.number if g else None


@router.get("/children/{student_id}/timetable", response_model=ChildTimetableResponse)
def child_timetable(student_id: str, current: User = Depends(get_current_user)):
    """The child's section's week and bell, as the student app shows it."""
    from ..curriculum import school_model_routes as smr
    _my_child(student_id, current)
    cs = cr._require()
    section = _section_of(student_id)
    if section is None:
        return ChildTimetableResponse(role="student")
    bell = cs.bell_for_section(section)
    entries = cs.timetable_for_section(section.id)
    return ChildTimetableResponse(role="student", academic_year_id=section.academic_year_id, section_id=section.id,
                                  bell_schedule=smr._bell(bell) if bell else None,
                                  entries=[smr._entry(e) for e in entries],
                                  subject_names=_subject_names(e.subject_id for e in entries))


@router.get("/children/{student_id}/calendar", response_model=list[CalendarItem])
def child_calendar(student_id: str, days: int = Query(42, ge=1, le=60),
                   current: User = Depends(get_current_user)) -> list[CalendarItem]:
    """The child's school calendar from today, as the child sees it on
    /my-calendar: holidays, half days, exam windows and papers, homework due
    (SA-6). A parent had no way to see a holiday: /my-calendar refused them
    and the day page only said "a holiday, a weekly off, or no timetable"
    (2026-10-04)."""
    from .calendar_feed import calendar_items
    return calendar_items(_my_child(student_id, current), days)


@router.get("/children/{student_id}/day", response_model=ChildDayResponse)
def child_day(student_id: str, on: str = Query(alias="date"), current: User = Depends(get_current_user)):
    """The child's day as it will run: substitutions, lost and make-up periods."""
    from ..curriculum import cover_routes
    child = _my_child(student_id, current)
    section = _section_of(student_id)
    cs = cr._require()
    try:
        year = cover_routes._year_for(cs, child.school_id, on)
    except HTTPException:
        return ChildDayResponse(rows=[])
    if section is None:
        return ChildDayResponse(rows=[])
    rows = [cover_routes._row(r) for r in cs.day_view(year.id, on, section_id=section.id)]
    return ChildDayResponse(rows=rows, subject_names=_subject_names(r.subject_id for r in rows))


@router.get("/children/{student_id}/homework", response_model=list[MyHomeworkItem])
def child_homework(student_id: str, current: User = Depends(get_current_user)):
    """The child's homework and where they stand on each (not the answers)."""
    from . import homework_routes as hr
    _my_child(student_id, current)
    section = _section_of(student_id)
    _log_read(current, "child_homework", student_id)
    if section is None:
        return []
    s = store()
    return [hr._my_item(hw, s.get_submission(hw.id, student_id)) for hw in s.homework_for_section(section.id)]


@router.get("/children/{student_id}/homework/{homework_id}", response_model=MyHomeworkDetail)
def child_homework_detail(student_id: str, homework_id: str, current: User = Depends(get_current_user)):
    """One homework as the child sees it: the questions without answers;
    after marking, the marks and the model answers."""
    from . import homework_routes as hr
    child = _my_child(student_id, current)
    _log_read(current, "child_homework", student_id)
    return hr.my_homework_detail(homework_id, child)


@router.post("/children/{student_id}/homework/{homework_id}/submit", response_model=MyHomeworkDetail)
def child_homework_submit(student_id: str, homework_id: str, req: SubmitRequest,
                          current: User = Depends(get_current_user)):
    """A parent hands in a young child's answers (SA-7: in classes 1-5 the
    parent does the app with the child). From class 6 the child hands in
    their own work. Recorded in the audit log as the parent's action."""
    from ..assessment.audit_log import get_audit_log
    from . import homework_routes as hr
    child = _my_child(student_id, current)
    grade = _grade_of(_section_of(student_id))
    if grade is None or grade > YOUNG_UP_TO:
        raise HTTPException(403, f"from class {YOUNG_UP_TO + 1}, children hand in their own homework")
    out = hr.submit(homework_id, req, child)
    get_audit_log(cr._cfg.data_root).append("homework_submitted_by_guardian", actor=current.id, details={
        "schoolId": child.school_id, "studentId": child.id, "homeworkId": homework_id})
    return out


@router.get("/children/{student_id}/learning", response_model=LearningProgressResponse)
def child_learning(student_id: str, current: User = Depends(get_current_user)):
    from .learning_routes import progress_for
    _my_child(student_id, current)
    _log_read(current, "child_learning_progress", student_id)
    return progress_for(student_id)


@router.get("/children/{student_id}/consent", response_model=ConsentResponse)
def child_consent(student_id: str, current: User = Depends(get_current_user)) -> ConsentResponse:
    child = _my_child(student_id, current)
    status, rec = _consent_status(child.school_id, student_id)
    return ConsentResponse(student_id=student_id, status=status, guardian_name=rec.guardian_name if rec else None,
                           method=rec.method if rec else None, recorded_at=rec.recorded_at if rec else None,
                           revoked_at=rec.revoked_at if rec else None)


@router.post("/children/{student_id}/consent", response_model=ConsentResponse)
def give_consent(student_id: str, current: User = Depends(get_current_user)) -> ConsentResponse:
    """The parent's own verifiable consent (DPDP s.9): given from the account
    the school linked to this child, so it is recorded as theirs."""
    child = _my_child(student_id, current)
    g = store().guardianship(current.id, student_id)
    _consents().record_consent(school_id=child.school_id, student_id=student_id, guardian_name=current.name,
                               guardian_relationship=g.relation, method="Parent account (in-app confirmation)",
                               recorded_by=current.id)
    _audit("consent_given_by_parent", current, {"studentId": student_id})
    return child_consent(student_id, current)


@router.delete("/children/{student_id}/consent", response_model=ConsentResponse)
def withdraw_consent(student_id: str, current: User = Depends(get_current_user)) -> ConsentResponse:
    """Withdraw consent. The school then cannot mark or process the child's
    work until it is given again."""
    child = _my_child(student_id, current)
    if _consents().revoke_consent(child.school_id, student_id, current.id) is None:
        raise HTTPException(404, "there is no consent on record to withdraw")
    _audit("consent_withdrawn_by_parent", current, {"studentId": student_id})
    return child_consent(student_id, current)
