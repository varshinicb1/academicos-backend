"""HTTP routes for the Phase 1 pilot agreement (pilot_agreement.py)."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from ..api.rate_limit import get_client_ip
from ..assessment.auth_routes import require_principal, require_staff
from ..assessment.schemas import Camel
from ..assessment.users import User
from .pilot_agreement import TITLE, VERSION, text, text_sha256

router = APIRouter(prefix="/api/v1/pilot-agreement")


class Acceptance(Camel):
    school_legal_name: str
    signatory_name: str
    designation: str
    accepted_email: str
    accepted_at: str
    version: str


class PilotAgreementResponse(Camel):
    version: str
    title: str
    text: str
    sha256: str
    # The school's name from its profile, to fill in the legal name.
    school_name: Optional[str] = None
    accepted: Optional[Acceptance] = None


class AcceptRequest(Camel):
    version: str
    sha256: str = Field(min_length=64, max_length=64)
    school_legal_name: str = Field(min_length=2, max_length=160)
    signatory_name: str = Field(min_length=2, max_length=120)
    designation: str = Field(min_length=2, max_length=80)
    agree: Literal[True]


def _store():
    from .routes import _cfg, store
    if _cfg is None:
        raise HTTPException(503, "the pilot agreement is not initialised")
    return store()


def _school_name(school_id: str) -> Optional[str]:
    try:
        from ..curriculum import routes as cr
        profile = cr._require().get_school_profile(school_id)
    except Exception:  # noqa: BLE001 - only a default for a field the principal fills in
        return None
    return profile.name if profile else None


def _response(school_id: str) -> PilotAgreementResponse:
    row = _store().pilot_acceptance(school_id)
    accepted = None
    if row is not None:
        accepted = Acceptance(school_legal_name=row["school_legal_name"], signatory_name=row["signatory_name"],
                              designation=row["designation"], accepted_email=row["accepted_email"],
                              accepted_at=row["accepted_at"], version=row["version"])
    return PilotAgreementResponse(version=VERSION, title=TITLE, text=text(), sha256=text_sha256(),
                                  school_name=_school_name(school_id), accepted=accepted)


def _clean(value: str, what: str) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) < 2:
        raise HTTPException(422, f"type the {what}")
    return cleaned


@router.get("", response_model=PilotAgreementResponse)
def get_pilot_agreement(current: User = Depends(require_staff)) -> PilotAgreementResponse:
    """The agreement and whether this school's principal has accepted it."""
    return _response(current.school_id)


@router.post("/accept", response_model=PilotAgreementResponse)
def accept_pilot_agreement(req: AcceptRequest, request: Request,
                           principal: User = Depends(require_principal)) -> PilotAgreementResponse:
    """The principal accepts the agreement for the school. 409 when the text
    shown is not the current one (reload and read it again)."""
    if req.version != VERSION or req.sha256 != text_sha256():
        raise HTTPException(409, "the agreement has changed since this page opened; reload it and read it again")
    row, made = _store().accept_pilot(
        school_id=principal.school_id, version=VERSION, sha256=req.sha256,
        school_legal_name=_clean(req.school_legal_name, "school's legal name"),
        signatory_name=_clean(req.signatory_name, "name of the person accepting"),
        designation=_clean(req.designation, "designation"), user=principal,
        client_ip=get_client_ip(request), user_agent=request.headers.get("user-agent"))
    if made:
        from .routes import _cfg
        from ..assessment.audit_log import get_audit_log
        get_audit_log(_cfg.data_root).append(
            "pilot_agreement_accepted", actor=principal.id,
            details={"schoolId": principal.school_id, "version": VERSION, "sha256": req.sha256,
                     "schoolLegalName": row["school_legal_name"], "signatory": row["signatory_name"],
                     "designation": row["designation"]})
    return _response(principal.school_id)
