"""HTTP routes for delegated admin (M1.4): the principal grants and revokes
admin capabilities on teacher accounts; anyone reads their own."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import ADMIN_CAPABILITIES, get_current_user, require_principal
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .routes import store

router = APIRouter(prefix="/api/v1")

CAPABILITY_LABELS = {
    "users": "People: invites, enrolment, parents, bulk import",
    "calendar": "Calendar: years, terms, holidays",
    "timetable": "Timetable: allocations, rooms, bells, timetables",
    "leave": "Leave, attendance and substitutions",
    "exams": "Exams: datesheets, invigilation, paper approval",
    "qbank_review": "Question bank review",
    "reports": "Reports and analytics",
}


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class GrantScope(_Req):
    grades: list[int] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)


class GrantRequest(_Req):
    user_id: str
    capability: str
    scope: Optional[GrantScope] = None


class GrantResponse(Camel):
    id: str
    user_id: str
    user_name: str = ""
    capability: str
    label: str
    scope: dict[str, Any]
    granted_by: str
    granted_at: str
    revoked_at: Optional[str] = None
    active: bool


class CapabilityInfo(Camel):
    capability: str
    label: str


class MyGrantsResponse(Camel):
    capabilities: list[str]
    grants: list[GrantResponse]
    available: list[CapabilityInfo]


def _out(g: dict) -> GrantResponse:
    u = cr._require_users().get(g["user_id"])
    return GrantResponse(id=g["id"], user_id=g["user_id"], user_name=u.name if u else "", capability=g["capability"],
                         label=CAPABILITY_LABELS.get(g["capability"], g["capability"]), scope=g["scope"],
                         granted_by=g["granted_by"], granted_at=g["granted_at"], revoked_at=g["revoked_at"],
                         active=g["active"])


def _audit(action: str, user: User, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=user.id, details={"schoolId": user.school_id, **details})


@router.get("/admin/grants", response_model=list[GrantResponse])
def list_grants(include_revoked: bool = False, principal: User = Depends(require_principal)) -> list[GrantResponse]:
    return [_out(g) for g in store().grants_for_school(principal.school_id, include_revoked=include_revoked)]


@router.post("/admin/grants", response_model=GrantResponse)
def grant(req: GrantRequest, principal: User = Depends(require_principal)) -> GrantResponse:
    """Give a teacher one admin capability, for the whole school or limited
    to classes, sections or subjects. Granting again replaces the scope."""
    if req.capability not in ADMIN_CAPABILITIES:
        raise HTTPException(422, f"capability must be one of {', '.join(ADMIN_CAPABILITIES)}")
    u = cr._require_users().get(req.user_id)
    if u is None:
        raise HTTPException(404, "user not found")
    if u.school_id != principal.school_id:
        raise HTTPException(403, "that user belongs to a different school")
    if u.role != "teacher":
        raise HTTPException(422, "admin capabilities are granted to teacher accounts")
    scope = req.scope.model_dump(by_alias=True) if req.scope else {}
    for sid in scope.get("sectionIds") or []:
        cr._require_school_owns_section(sid, principal)
    before = [g["scope"] for g in store().active_grants(u.id, req.capability)]
    try:
        g = store().grant(school_id=principal.school_id, user_id=u.id, capability=req.capability, scope=scope,
                          granted_by=principal.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("admin_granted", principal, {"userId": u.id, "capability": req.capability,
                                         "before": before[0] if before else None, "after": g["scope"]})
    return _out(g)


@router.delete("/admin/grants/{grant_id}", response_model=GrantResponse)
def revoke(grant_id: str, principal: User = Depends(require_principal)) -> GrantResponse:
    g = store().get_grant(grant_id)
    if g is None or g["school_id"] != principal.school_id:
        raise HTTPException(404, "no such grant at your school")
    if not g["active"]:
        raise HTTPException(409, "this grant is already revoked")
    g = store().revoke_grant(grant_id, revoked_by=principal.id)
    _audit("admin_revoked", principal, {"userId": g["user_id"], "capability": g["capability"],
                                         "before": g["scope"], "after": None})
    return _out(g)


@router.get("/my-grants", response_model=MyGrantsResponse)
def my_grants(current: User = Depends(get_current_user)) -> MyGrantsResponse:
    """What the caller may administer: every capability for the principal,
    their grants for a teacher, nothing for anyone else. The web shows the
    matching admin tools from this."""
    grants = store().active_grants(current.id) if current.role == "teacher" else []
    caps = list(ADMIN_CAPABILITIES) if current.role == "principal" else sorted({g["capability"] for g in grants})
    return MyGrantsResponse(capabilities=caps, grants=[_out(g) for g in grants],
                            available=[CapabilityInfo(capability=c, label=CAPABILITY_LABELS[c]) for c in ADMIN_CAPABILITIES])
