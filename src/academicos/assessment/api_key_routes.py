"""Issuing keys for the question-bank API (audit D38): the principal mints,
lists, inspects and revokes their own school's keys. The plaintext key is
returned once, at creation, and is never stored (api_keys.py keeps its
SHA-256). Every mint and revoke is audited. The store itself is snapshotted
to durable storage (ApiKeyStore(durable=True)), so a key survives a restart.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

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
    return _out(k)


@router.get("/{key_id}/usage", response_model=list[UsageWindow])
def key_usage(key_id: str, principal: User = Depends(require_principal)) -> list[UsageWindow]:
    """Requests per minute over the last few minutes (the quota's own count)."""
    _own(key_id, principal)
    return [UsageWindow(window_start=r["window_start"], count=r["count"]) for r in _store().usage(key_id)]
