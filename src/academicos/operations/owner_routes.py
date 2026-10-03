"""HTTP routes for owners over several schools (ROLE-4). See owners.py for
the link model.

Two sides:

* a school's own principal issues, lists and revokes owner links
  (`/owner-links`). An owner acting as that school's principal is refused
  here: it must not be able to mint or withdraw owner access, or an owner the
  principal revoked could have kept a second account linked;
* the owner registers with a code, redeems further codes, lists its schools,
  chooses the one its session acts in, and reads the overview
  (`/owner/...`). Choosing a school makes the session that school's
  principal (auth_routes.get_current_user); every other route keeps its
  single-school scoping, so no route returns two schools' records in one
  response. The overview is the one exception, by design: per-school
  aggregates, with no student, teacher or record named.

Every issue, redemption, revoke and switch is in the audit log.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..api.rate_limit import rate_limit_register
from ..assessment import auth_routes
from ..assessment.auth_routes import (
    AuthResponse, UserResponse, _bearer_token, _to_response, acting_view, require_owner, require_principal,
)
from ..assessment.users import EmailAlreadyRegistered, User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .owners import AlreadyLinked, OwnerLinkRefused, link_status
from .routes import store

router = APIRouter(prefix="/api/v1")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


def _audit(action: str, actor: str, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=actor, details=details)


def _school_name(school_id: str) -> Optional[str]:
    try:
        p = cr._require().get_school_profile(school_id)
    except HTTPException:
        return None
    return p.name if p is not None else None


# ---------------------------------------------------------------- the principal's side


class OwnerLinkCreateRequest(_Req):
    # Who the code is for, as the principal will recognise it in the list.
    label: str = Field(default="", max_length=100)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=90)


class OwnerLinkResponse(Camel):
    id: str
    label: str
    code_hint: str
    status: str                      # "open" | "linked" | "revoked" | "expired"
    created_at: str
    created_by: str
    expires_at: str
    owner_id: Optional[str] = None
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    redeemed_at: Optional[str] = None
    revoked_at: Optional[str] = None


class OwnerLinkCreatedResponse(OwnerLinkResponse):
    code: str                        # shown this once


def _link_out(row: dict, cls=OwnerLinkResponse, **extra: Any):
    owner = cr._require_users().get(row["owner_id"]) if row.get("owner_id") else None
    return cls(id=row["id"], label=row["label"], code_hint=row["code_hint"], status=link_status(row),
               created_at=row["created_at"], created_by=row["created_by"], expires_at=row["expires_at"],
               owner_id=row.get("owner_id"), owner_name=owner.name if owner else None,
               owner_email=owner.email if owner else None, redeemed_at=row.get("redeemed_at"),
               revoked_at=row.get("revoked_at"), **extra)


def _own_principal(principal: User) -> User:
    if principal.acting_owner:
        raise HTTPException(403, "owner links are issued and withdrawn by the school's own principal, "
                                 "not by an owner acting in the school")
    return principal


@router.post("/owner-links", response_model=OwnerLinkCreatedResponse)
def create_owner_link(req: OwnerLinkCreateRequest,
                      principal: User = Depends(require_principal)) -> OwnerLinkCreatedResponse:
    """A single-use code that links one owner account to THIS school. Copy
    `code` now: only its hash is kept. Whoever redeems it acts in this school
    with a principal's rights until you revoke the link."""
    _own_principal(principal)
    try:
        code, row = store().create_owner_link(school_id=principal.school_id, created_by=principal.id,
                                              label=req.label, expires_in_days=req.expires_in_days)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("owner_link_created", principal.id, {"schoolId": principal.school_id, "linkId": row["id"],
                                                "label": row["label"], "expiresAt": row["expires_at"]})
    return _link_out(row, OwnerLinkCreatedResponse, code=code)


@router.get("/owner-links", response_model=list[OwnerLinkResponse])
def list_owner_links(principal: User = Depends(require_principal)) -> list[OwnerLinkResponse]:
    """This school's owner links, newest first: open codes, linked owners
    (who they are), and withdrawn or expired ones."""
    return [_link_out(r) for r in store().owner_links_for_school(principal.school_id)]


@router.delete("/owner-links/{link_id}", response_model=OwnerLinkResponse)
def revoke_owner_link(link_id: str, principal: User = Depends(require_principal)) -> OwnerLinkResponse:
    """Withdraw a code, or end a linked owner's access to this school. It
    takes effect on the owner's next request. Another school's link is a 404."""
    _own_principal(principal)
    row = store().get_owner_link(link_id)
    if row is None or row["school_id"] != principal.school_id:
        raise HTTPException(404, "no such owner link at your school")
    if row.get("revoked_at"):
        raise HTTPException(409, "this owner link is already revoked")
    row = store().revoke_owner_link(link_id, revoked_by=principal.id)
    _audit("owner_link_revoked", principal.id, {"schoolId": principal.school_id, "linkId": link_id,
                                                "ownerId": row.get("owner_id"),
                                                "wasLinked": bool(row.get("redeemed_at"))})
    return _link_out(row)


# ---------------------------------------------------------------- the owner's side


class OwnerRegisterRequest(_Req):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=200)
    password: str
    code: str = Field(min_length=1, max_length=200)


class RedeemRequest(_Req):
    code: str = Field(min_length=1, max_length=200)


class ActiveSchoolRequest(_Req):
    school_id: str = Field(min_length=1, max_length=200)


class OwnerSchool(Camel):
    school_id: str
    name: Optional[str] = None
    linked_at: str
    active: bool


def _by_name(rows: list) -> list:
    """Schools in the order a person looks for them: by name, then id."""
    return sorted(rows, key=lambda s: ((s.name or s.school_id).casefold(), s.school_id))


def _redeem(code: str, owner_id: str) -> dict:
    try:
        return store().claim_owner_link(code, owner_id)
    except OwnerLinkRefused as e:
        raise HTTPException(403, str(e))
    except AlreadyLinked:
        raise HTTPException(409, "your account is already linked to this school")


def _school_out(row: dict, active: Optional[str]) -> OwnerSchool:
    return OwnerSchool(school_id=row["school_id"], name=_school_name(row["school_id"]),
                       linked_at=row["redeemed_at"], active=row["school_id"] == active)


@router.post("/owner/register", response_model=AuthResponse, dependencies=[Depends(rate_limit_register)])
def register_owner(req: OwnerRegisterRequest) -> AuthResponse:
    """Create an owner account with the first school's owner link code. More
    schools are linked later with POST /owner/links/redeem. 403 for a code
    that is unknown, used, withdrawn or expired; 409 for a taken email."""
    users = auth_routes._require()
    ops = store()
    if not req.code.strip().startswith("own_"):
        raise HTTPException(403, "that is not an owner link code; a teacher or student invite "
                                 "is used under Create account instead")
    if len(req.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if users.get_by_email(req.email) is not None:
        raise HTTPException(409, "an account with this email already exists")
    user_id = users.new_user_id()
    link = _redeem(req.code, user_id)
    try:
        user = users.register_owner(user_id=user_id, name=req.name.strip(), email=req.email, password=req.password)
    except EmailAlreadyRegistered:
        ops.release_owner_link(link["id"], user_id)
        raise HTTPException(409, "an account with this email already exists")
    except BaseException:
        ops.release_owner_link(link["id"], user_id)
        raise
    _audit("owner_registered", user.id, {"schoolId": link["school_id"], "linkId": link["id"]})
    _audit("owner_link_redeemed", user.id, {"schoolId": link["school_id"], "linkId": link["id"]})
    token = users.create_session(user.id)
    return AuthResponse(user=_to_response(user), token=token)


@router.post("/owner/links/redeem", response_model=OwnerSchool)
def redeem_owner_link(req: RedeemRequest, request: Request, owner: User = Depends(require_owner)) -> OwnerSchool:
    """Link one more school to this owner account with its principal's code."""
    link = _redeem(req.code, owner.id)
    _audit("owner_link_redeemed", owner.id, {"schoolId": link["school_id"], "linkId": link["id"]})
    return _school_out(link, _active(request, owner))


def _token(request: Request) -> str:
    token = _bearer_token(request.headers.get("authorization", ""))
    if not token:  # require_owner has already refused this; kept for the type
        raise HTTPException(401, "missing bearer token")
    return token


def _active(request: Request, owner: User) -> Optional[str]:
    return store().owner_session_school(_token(request), owner.id)


@router.get("/owner/schools", response_model=list[OwnerSchool])
def owner_schools(request: Request, owner: User = Depends(require_owner)) -> list[OwnerSchool]:
    """Exactly the schools linked to this account, and which one this
    session acts in. A revoked link is gone from the list at once."""
    active = _active(request, owner)
    return _by_name([_school_out(r, active) for r in store().owner_memberships(owner.id)])


@router.post("/owner/active-school", response_model=UserResponse)
def choose_school(req: ActiveSchoolRequest, request: Request, owner: User = Depends(require_owner)) -> UserResponse:
    """Act in one linked school: from now on this session is that school's
    principal on every route, and sees that school's data only. Answers the
    session's new identity (as /auth/me will)."""
    if not store().is_linked(owner.id, req.school_id):
        raise HTTPException(403, "this school is not linked to your account")
    before = _active(request, owner)
    store().set_owner_session_school(_token(request), owner.id, req.school_id)
    _audit("owner_school_switched", owner.id, {"from": before, "to": req.school_id})
    return _to_response(acting_view(owner, req.school_id))


@router.delete("/owner/active-school", response_model=UserResponse)
def leave_school(request: Request, owner: User = Depends(require_owner)) -> UserResponse:
    """Stop acting in a school: back to the owner's own pages."""
    before = _active(request, owner)
    store().set_owner_session_school(_token(request), owner.id, None)
    if before:
        _audit("owner_school_switched", owner.id, {"from": before, "to": None})
    return _to_response(owner)


# ---------------------------------------------------------------- the overview


class SchoolSummary(Camel):
    """One school's numbers. Counts only: nothing here names a student, a
    teacher, a paper or a section."""
    school_id: str
    name: Optional[str] = None
    period: Optional[str] = None             # the term (or the year so far) the numbers cover
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    papers_this_term: Optional[int] = None
    syllabus_planned: Optional[int] = None   # lessons planned to date
    syllabus_taught: Optional[int] = None    # lessons taught to date
    syllabus_pace_pct: Optional[float] = None
    rows_behind: Optional[int] = None        # class-subject plans behind
    cover_requested: Optional[int] = None
    cover_filled: Optional[int] = None
    cover_filled_pct: Optional[float] = None
    note: Optional[str] = None               # why a number is missing


class OwnerOverview(Camel):
    schools: list[SchoolSummary]


def _pct(part: int, whole: int) -> Optional[float]:
    return round(100.0 * part / whole, 1) if whole else None


def _summary(owner: User, school_id: str) -> SchoolSummary:
    """The same counting as the school's own term report (term_report.py),
    run as that school's principal, one school at a time."""
    from . import term_report as tr
    view = acting_view(owner, school_id)
    out = SchoolSummary(school_id=school_id, name=_school_name(school_id))
    try:
        year_id, scope = tr._resolve(view, None)
    except HTTPException as e:
        out.note = str(e.detail)
        return out
    school = tr._School(cr._require(), year_id)
    f = tr._Filter(None, None, None)
    syllabus = tr._syllabus(view, year_id, scope, f)
    exams, _ = tr._exams_and_time(view, year_id, scope, school, f)
    cover = tr._cover(year_id, scope, school, f)
    out.period, out.period_start, out.period_end = scope.name, scope.start_date, scope.end_date
    out.papers_this_term = exams.papers_set
    out.syllabus_planned, out.syllabus_taught = syllabus.planned_to_date, syllabus.taught_to_date
    out.syllabus_pace_pct = _pct(syllabus.taught_to_date, syllabus.planned_to_date)
    out.rows_behind = syllabus.behind
    out.cover_requested, out.cover_filled = cover.requested, cover.filled
    out.cover_filled_pct = _pct(cover.filled, cover.requested)
    if scope.note:
        out.note = scope.note
    return out


@router.get("/owner/overview", response_model=OwnerOverview)
def owner_overview(owner: User = Depends(require_owner)) -> OwnerOverview:
    """Each linked school's term at a glance: papers set this term, syllabus
    taught against plan, and cover filled. Aggregates per school, side by
    side; open a school to see anything behind a number."""
    return OwnerOverview(schools=_by_name([_summary(owner, sid) for sid in store().linked_school_ids(owner.id)]))
