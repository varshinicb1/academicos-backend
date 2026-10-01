"""HTTP routes for SMS and WhatsApp consent (operations/messaging.py)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user
from ..assessment.users import User
from ..curriculum.schemas import Camel
from .messaging import MESSAGE_KINDS, MESSAGE_ROLES, _configured

router = APIRouter(prefix="/api/v1")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ContactRequest(_Req):
    """phone null removes the number and withdraws both."""
    phone: Optional[str] = Field(default=None, max_length=20)
    sms_ok: bool = False
    whatsapp_ok: bool = False


class ContactResponse(Camel):
    phone: Optional[str] = None
    sms_ok: bool = False
    whatsapp_ok: bool = False
    # Whether the school's messaging is switched on at all (the owner's
    # provider accounts): the app says so rather than promising messages.
    sms_available: bool = False
    whatsapp_available: bool = False
    kinds: list[str] = []


def _store():
    from .routes import store
    return store()


def _response(contact: Optional[dict]) -> ContactResponse:
    return ContactResponse(
        phone=contact["phone"] if contact else None,
        sms_ok=bool(contact and contact["sms_ok"]), whatsapp_ok=bool(contact and contact["whatsapp_ok"]),
        sms_available=_configured("sms") is None, whatsapp_available=_configured("whatsapp") is None,
        kinds=sorted(MESSAGE_KINDS))


def _may_message(current: User) -> None:
    if current.role not in MESSAGE_ROLES:
        raise HTTPException(403, "messages about a student go to their parent's number; a student does not add one")


@router.get("/me/contact", response_model=ContactResponse)
def my_contact(current: User = Depends(get_current_user)) -> ContactResponse:
    """The caller's own number and consent. Nobody else can read it."""
    _may_message(current)
    return _response(_store().contact_for(current.id))


@router.put("/me/contact", response_model=ContactResponse)
def set_my_contact(req: ContactRequest, current: User = Depends(get_current_user)) -> ContactResponse:
    """Give, change or remove the caller's own number, and say per channel
    whether they want SMS and WhatsApp. The consent is audited (not the
    number: its last two digits only)."""
    _may_message(current)
    if req.phone is None and (req.sms_ok or req.whatsapp_ok):
        raise HTTPException(422, "give a number to receive SMS or WhatsApp")
    try:
        before, after = _store().set_contact(current.id, phone=req.phone, sms_ok=req.sms_ok,
                                             whatsapp_ok=req.whatsapp_ok)
    except ValueError as e:
        raise HTTPException(422, str(e))

    def shape(c):
        return None if c is None else {"phoneEnding": c["phone"][-2:], "sms": bool(c["sms_ok"]),
                                       "whatsapp": bool(c["whatsapp_ok"])}
    if shape(before) != shape(after):
        from ..assessment.audit_log import get_audit_log
        from ..curriculum import routes as cr
        get_audit_log(cr._cfg.data_root).append(
            "message_consent_set", actor=current.id,
            details={"schoolId": current.school_id, "before": shape(before), "after": shape(after)})
    return _response(after)
