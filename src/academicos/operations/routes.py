"""HTTP routes for the notification service (NTF-1): the caller's own inbox,
their preferences and quiet hours, and their devices for push. Nothing here
reads another user's notifications -- every route is scoped to the caller."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user
from ..assessment.users import User
from ..config import Config
from ..curriculum.schemas import Camel
from .notifications import CATALOGUE, CHANNELS
from .store import OperationsStore, get_operations_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
_cfg: Optional[Config] = None


def init(cfg: Config) -> None:
    global _cfg
    _cfg = cfg

    def _email(user_id: str) -> Optional[str]:
        from ..curriculum import routes as cr
        try:
            u = cr._require_users().get(user_id)
        except Exception:  # noqa: BLE001
            return None
        return getattr(u, "email", None) if u else None
    OperationsStore.email_for = staticmethod(_email)


def store() -> OperationsStore:
    if _cfg is None:
        raise HTTPException(503, "the notification service is not initialised")
    return get_operations_store(_cfg.data_root)


def notify_safely(**kw: Any) -> None:
    """For other modules: a notification must never break the action that
    caused it (a leave decision, a substitution), so failures are logged."""
    if _cfg is None:
        return
    try:
        store().notify(**kw)
    except Exception:  # noqa: BLE001
        logger.warning("could not notify %s about %s", kw.get("user_ids"), kw.get("kind"), exc_info=True)


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class NotificationResponse(Camel):
    id: str
    kind: str
    title: str
    body: str
    link: Optional[str] = None
    created_at: str
    read: bool
    push_status: str
    email_status: str


class InboxResponse(Camel):
    unread_count: int
    items: list[NotificationResponse]


class KindPreference(Camel):
    kind: str
    label: str
    push: bool
    email: bool


class PreferencesResponse(Camel):
    quiet_start: str
    quiet_end: str
    language: str
    kinds: list[KindPreference]


class KindPreferenceBody(_Req):
    kind: str
    push: Optional[bool] = None
    email: Optional[bool] = None


class PreferencesRequest(_Req):
    quiet_start: Optional[str] = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    quiet_end: Optional[str] = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    language: Optional[str] = None
    kinds: list[KindPreferenceBody] = []


class DeviceRequest(_Req):
    token: str = Field(min_length=10, max_length=4096)
    platform: str = "android"


def _item(n) -> NotificationResponse:
    return NotificationResponse(id=n.id, kind=n.kind, title=n.title, body=n.body, link=n.link,
                                created_at=n.created_at, read=n.read_at is not None,
                                push_status=n.push_status, email_status=n.email_status)


def _prefs(user_id: str) -> PreferencesResponse:
    s = store()
    settings = s.settings_for(user_id)
    return PreferencesResponse(
        quiet_start=settings["quiet_start"], quiet_end=settings["quiet_end"], language=settings["language"],
        kinds=[KindPreference(kind=k.key, label=k.label, push=s.channel_enabled(user_id, k.key, "push"),
                              email=s.channel_enabled(user_id, k.key, "email")) for k in CATALOGUE.values()])


@router.get("/notifications", response_model=InboxResponse)
def inbox(unread_only: bool = Query(default=False, alias="unreadOnly"), limit: int = Query(default=50, ge=1, le=200),
          current: User = Depends(get_current_user)) -> InboxResponse:
    """The caller's inbox, newest first. Also sends anything whose quiet
    hours have ended."""
    s = store()
    try:
        s.deliver_due()
    except Exception:  # noqa: BLE001
        logger.warning("deferred delivery failed", exc_info=True)
    return InboxResponse(unread_count=s.unread_count(current.id),
                         items=[_item(n) for n in s.inbox(current.id, unread_only=unread_only, limit=limit)])


@router.post("/notifications/{notification_id}/read")
def mark_read(notification_id: str, current: User = Depends(get_current_user)) -> dict:
    n = store().get_notification(notification_id)
    if n is None:
        raise HTTPException(404, "notification not found")
    if n.user_id != current.id:
        raise HTTPException(403, "this is not your notification")
    store().mark_read(current.id, notification_id)
    return {"ok": True}


@router.post("/notifications/read-all")
def mark_all_read(current: User = Depends(get_current_user)) -> dict:
    return {"marked": store().mark_read(current.id)}


@router.get("/notification-preferences", response_model=PreferencesResponse)
def get_preferences(current: User = Depends(get_current_user)) -> PreferencesResponse:
    return _prefs(current.id)


@router.put("/notification-preferences", response_model=PreferencesResponse)
def set_preferences(req: PreferencesRequest, current: User = Depends(get_current_user)) -> PreferencesResponse:
    """Quiet hours, language (en or hi), and push/email per kind. The inbox
    always receives every notification."""
    s = store()
    try:
        s.set_settings(current.id, quiet_start=req.quiet_start, quiet_end=req.quiet_end, language=req.language)
        for k in req.kinds:
            for ch in CHANNELS:
                v = getattr(k, ch)
                if v is not None:
                    s.set_channel(current.id, k.kind, ch, v)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _prefs(current.id)


@router.post("/devices")
def register_device(req: DeviceRequest, current: User = Depends(get_current_user)) -> dict:
    """Register this device's push token for the caller."""
    try:
        store().register_device(current.id, req.token, req.platform)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}


@router.delete("/devices/{token}")
def remove_device(token: str, current: User = Depends(get_current_user)) -> dict:
    if not store().remove_device(current.id, token):
        raise HTTPException(404, "no such device for you")
    return {"ok": True}
