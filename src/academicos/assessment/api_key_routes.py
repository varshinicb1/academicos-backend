"""Issuing keys for the question-bank API (audit D38): the principal mints,
lists, inspects and revokes their own school's keys. The plaintext key is
returned once, at creation, and is never stored (api_keys.py keeps its
SHA-256). Every mint and revoke is audited. The store itself is snapshotted
to durable storage (ApiKeyStore(durable=True)), so a key survives a restart.

A key's webhooks (API-6) live here too: the principal registers a partner's
HTTPS URL for a key (its secret shown once), lists them, revokes one, and
reads each one's delivery log. webhooks.py has the signing, the address
checks and the delivery; a webhook only ever hears of questions its key may
read.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from . import webhooks
from .api_keys import DEFAULT_QUOTA_PER_MINUTE, SCOPES, ApiKey, ScopeError
from .auth_routes import require_principal
from .schemas import Camel
from .users import User

router = APIRouter(prefix="/api/v1/api-keys")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ApiKeyCreateRequest(_Req):
    label: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(min_length=1)
    quota_per_minute: int = Field(default=DEFAULT_QUOTA_PER_MINUTE, ge=1, le=6000)
    # API-3: "scoped (read-only, which grades/subjects)". Empty is every class
    # and subject. Every /v1 read then serves only records inside them, and a
    # request that names a class or subject outside them is a 403.
    grades: list[Annotated[int, Field(ge=1, le=12)]] = Field(default_factory=list, max_length=12)
    subjects: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        default_factory=list, max_length=40)


class ApiKeyResponse(Camel):
    id: str
    prefix: str
    label: str
    scopes: list[str]
    grades: list[int] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    quota_per_minute: int
    created_at: str
    created_by: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None
    active: bool


class ApiKeyCreatedResponse(ApiKeyResponse):
    key: str                        # the plaintext, shown this once
    scopes_available: list[str]


class UsageWindow(Camel):
    window_start: str
    count: int


def _store():
    from . import qbank_routes
    if qbank_routes._store is None:
        raise HTTPException(503, "the question-bank API is not initialised")
    return qbank_routes._store


def _out(k: ApiKey, cls=ApiKeyResponse, **extra: Any):
    return cls(id=k.id, prefix=k.prefix, label=k.label, scopes=sorted(k.scopes), quota_per_minute=k.quota_per_minute,
               grades=sorted(k.grades), subjects=sorted(k.subjects),
               created_at=k.created_at, created_by=k.created_by, last_used_at=k.last_used_at,
               revoked_at=k.revoked_at, active=k.active, **extra)


def _bank_subjects(asked: list[str]) -> list[str]:
    """The subjects a key is limited to, spelled as the bank spells them.

    A limit compares case-insensitively, so "science" works; but a subject the
    bank does not carry at all ("Sceince") would mint a key that reads nothing
    and says nothing about why, which is the silent empty page this API keeps
    refusing elsewhere. So it is a 422 naming what the bank does carry. With
    no bank loaded there is nothing to check against, and the names are kept
    as typed.
    """
    from . import qbank_routes
    bank = qbank_routes._bank
    if bank is None or not asked:
        return asked
    known = {str(r.get("subject") or "").lower(): str(r.get("subject") or "") for r in bank.records}
    missing = sorted(s for s in asked if s.strip().lower() not in known)
    if missing:
        raise HTTPException(422, f"the question bank has no subject {missing}; it has: "
                                 + ", ".join(sorted(v for v in known.values() if v)))
    return [known[s.strip().lower()] for s in asked]


def _own(key_id: str, principal: User) -> ApiKey:
    k = _store().get(key_id)
    if k is None or k.school_id != principal.school_id:
        raise HTTPException(404, "no such key at your school")
    return k


def _audit(action: str, principal: User, details: dict[str, Any]) -> None:
    from ..curriculum import routes as cr
    from .audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=principal.id,
                                            details={"schoolId": principal.school_id, **details})


@router.post("", response_model=ApiKeyCreatedResponse)
def create_key(req: ApiKeyCreateRequest, principal: User = Depends(require_principal)) -> ApiKeyCreatedResponse:
    """Mint a key for this school. Copy `key` now: it is not shown again.
    Scopes are the question-bank API's; any that would reach student data
    are refused outright, and any no route checks yet are refused as "not
    available yet". `grades` and `subjects` limit what the key can read."""
    subjects = _bank_subjects(req.subjects)
    try:
        plaintext, k = _store().create(school_id=principal.school_id, scopes=req.scopes, label=req.label.strip(),
                                       created_by=principal.id, quota_per_minute=req.quota_per_minute,
                                       grades=req.grades, subjects=subjects)
    except ScopeError as e:
        raise HTTPException(422, str(e))
    _audit("api_key_created", principal, {"keyId": k.id, "prefix": k.prefix, "scopes": sorted(k.scopes),
                                         "label": k.label, "grades": sorted(k.grades),
                                         "subjects": sorted(k.subjects)})
    return _out(k, ApiKeyCreatedResponse, key=plaintext, scopes_available=sorted(SCOPES))


@router.get("", response_model=list[ApiKeyResponse])
def list_keys(principal: User = Depends(require_principal)) -> list[ApiKeyResponse]:
    return [_out(k) for k in _store().list_for_school(principal.school_id)]


@router.delete("/{key_id}", response_model=ApiKeyResponse)
def revoke_key(key_id: str, principal: User = Depends(require_principal)) -> ApiKeyResponse:
    """Revoke at once. The record is kept for incident review."""
    before = _own(key_id, principal)
    k = _store().revoke(key_id)
    if before.active:
        _audit("api_key_revoked", principal, {"keyId": key_id, "prefix": before.prefix})
        # Its webhooks stop with it, and what they had not delivered is
        # cancelled (each send checks the key as well).
        svc = webhooks.service()
        if svc is not None:
            for hook in svc.store.for_key(key_id):
                if hook["revoked_at"] is None:
                    svc.store.revoke(hook["id"], revoked_by=principal.id)
    return _out(k)


@router.get("/{key_id}/usage", response_model=list[UsageWindow])
def key_usage(key_id: str, principal: User = Depends(require_principal)) -> list[UsageWindow]:
    """Requests per minute over the last few minutes (the quota's own count)."""
    _own(key_id, principal)
    return [UsageWindow(window_start=r["window_start"], count=r["count"]) for r in _store().usage(key_id)]


# ---------------------------------------------------------------- webhooks (API-6)


class WebhookCreateRequest(_Req):
    url: str = Field(min_length=9, max_length=2000)
    events: list[str] = Field(default_factory=lambda: ["questions.added"], min_length=1, max_length=10)


class WebhookResponse(Camel):
    id: str
    key_id: str
    url: str
    events: list[str]
    secret_hint: str
    created_at: str
    created_by: str
    revoked_at: Optional[str] = None
    active: bool


class WebhookCreatedResponse(WebhookResponse):
    secret: str                     # shown this once: verify X-AcademicOS-Signature with it


class WebhookDelivery(Camel):
    id: str
    event_id: str
    event_type: str
    status: str                     # "pending" | "delivered" | "failed" | "cancelled"
    attempts: int
    last_status_code: Optional[int] = None
    last_error: str = ""
    next_attempt_at: Optional[str] = None
    created_at: str
    updated_at: str
    delivered_at: Optional[str] = None
    # What the event announced, as numbers: never question text, never a person.
    subject: Optional[str] = None
    grade: Optional[int] = None
    chapter_id: Optional[str] = None
    question_count: int = 0


def _hooks() -> "webhooks.WebhookService":
    svc = webhooks.service()
    if svc is None:
        raise HTTPException(503, "the question-bank API is not initialised")
    return svc


def _hook_out(row: dict, cls=WebhookResponse, **extra: Any):
    return cls(id=row["id"], key_id=row["key_id"], url=row["url"], events=sorted(row["events"].split(",")),
               secret_hint=row["secret_hint"], created_at=row["created_at"], created_by=row["created_by"],
               revoked_at=row["revoked_at"], active=row["revoked_at"] is None, **extra)


def _own_hook(key_id: str, webhook_id: str, principal: User) -> dict:
    _own(key_id, principal)
    row = _hooks().store.get(webhook_id)
    if row is None or row["key_id"] != key_id:
        raise HTTPException(404, "no such webhook on this key")
    return row


def _delivery_out(d: dict) -> WebhookDelivery:
    data = (json.loads(d["payload"]).get("data") or {})
    grade = data.get("grade")
    return WebhookDelivery(
        id=d["id"], event_id=d["event_id"], event_type=d["event_type"], status=d["status"],
        attempts=d["attempts"], last_status_code=d["last_status_code"], last_error=d["last_error"] or "",
        next_attempt_at=d["next_attempt_at"] if d["status"] == "pending" else None,
        created_at=d["created_at"], updated_at=d["updated_at"], delivered_at=d["delivered_at"],
        subject=data.get("subject"), grade=grade if isinstance(grade, int) else None,
        chapter_id=data.get("chapterId"), question_count=int(data.get("count") or 0))


@router.post("/{key_id}/webhooks", response_model=WebhookCreatedResponse)
def create_webhook(key_id: str, req: WebhookCreateRequest,
                   principal: User = Depends(require_principal)) -> WebhookCreatedResponse:
    """Register a partner's HTTPS URL for this key's events. Copy `secret`
    now: it signs every delivery and is not shown again. The URL must be
    public HTTPS (a private, loopback, link-local or metadata address is
    refused, now and at every send), and only questions the key may read are
    ever announced to it."""
    key = _own(key_id, principal)
    try:
        secret, row = _hooks().register(key, url=req.url, events=req.events, created_by=principal.id)
    except (webhooks.UnsafeWebhookUrl, webhooks.WebhookError) as e:
        raise HTTPException(422, str(e))
    _audit("webhook_created", principal, {"keyId": key_id, "webhookId": row["id"], "url": row["url"],
                                          "events": sorted(row["events"].split(","))})
    return _hook_out(row, WebhookCreatedResponse, secret=secret)


@router.get("/{key_id}/webhooks", response_model=list[WebhookResponse])
def list_webhooks(key_id: str, principal: User = Depends(require_principal)) -> list[WebhookResponse]:
    """This key's webhooks, newest first. The secret is never listed."""
    _own(key_id, principal)
    return [_hook_out(r) for r in _hooks().store.for_key(key_id)]


@router.delete("/{key_id}/webhooks/{webhook_id}", response_model=WebhookResponse)
def revoke_webhook(key_id: str, webhook_id: str, principal: User = Depends(require_principal)) -> WebhookResponse:
    """Stop the webhook at once; what it had not delivered is cancelled."""
    row = _own_hook(key_id, webhook_id, principal)
    if row["revoked_at"] is not None:
        return _hook_out(row)
    row = _hooks().store.revoke(webhook_id, revoked_by=principal.id)
    _audit("webhook_revoked", principal, {"keyId": key_id, "webhookId": webhook_id})
    return _hook_out(row)


@router.get("/{key_id}/webhooks/{webhook_id}/deliveries", response_model=list[WebhookDelivery])
def webhook_deliveries(key_id: str, webhook_id: str,
                       principal: User = Depends(require_principal)) -> list[WebhookDelivery]:
    """The webhook's last 50 deliveries, newest first: status, attempts, and
    the receiver's last answer or the error."""
    _own_hook(key_id, webhook_id, principal)
    return [_delivery_out(d) for d in _hooks().store.deliveries(webhook_id)]
