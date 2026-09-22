"""Parental consent API endpoints — DPDP Act 2023 compliance.

Endpoints for recording, querying, and revoking verifiable parental consent
for CBSE minors. All routes require a teacher or principal (require_staff)
and are strictly scoped to the caller's school_id. They used to accept any
logged-in account, so a student could list every guardian's name at the
school, or record "consent" for themselves.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .audit_log import AuditLog, get_audit_log, record_pii_read
from .auth_routes import require_staff
from .authz import students_without_consent
from .consent import ConsentStore, ParentalConsentRecord, get_consent_store
from .schemas import Camel
from .users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consent", tags=["Parental Consent"])

_store: Optional[ConsentStore] = None
_audit: Optional[AuditLog] = None


def init(data_root: Any) -> None:
    global _store, _audit
    _store = get_consent_store(data_root)
    _audit = get_audit_log(data_root)


def _require_store() -> ConsentStore:
    global _store
    if _store is None:
        _store = get_consent_store()
    return _store


def _require_audit() -> Optional[AuditLog]:
    global _audit
    if _audit is None:
        try:
            _audit = get_audit_log()
        except Exception:
            # Was `pass`. A failed audit-store construction left `_audit = None`
            # and every caller then skipped its audit write in silence, so
            # consent could be recorded with no trail and no signal anywhere.
            # The audit store is local SQLite; failing to build it means
            # something is genuinely wrong and must be visible.
            logger.exception(
                "could not construct the audit log; consent changes will be "
                "refused rather than recorded without a trail"
            )
    return _audit


def _audit_or_refuse(action: str, *, actor_id: str, consent_id: str,
                     student_id: str, details: dict) -> None:
    """Write an audit record, or fail the request.

    Two defects lived here, and the second was hiding the first.

    1. Both call sites wrapped the write in `except Exception: pass`, so the
       consent write returned 200 even when the trail was incomplete. CERT-In
       requires the trail, and a consent record it cannot describe is not a
       compliant record.

    2. The calls were `audit.record(action=..., actor_id=..., resource_type=...,
       resource_id=...)`. `AuditLog` has no `record` method -- it has `append`,
       with a different signature, and every other caller in the repo uses it.
       So this raised `AttributeError` on EVERY consent write, from the day it
       was written, and defect 1 swallowed it. The consent audit trail was never
       written at all, and nothing anywhere said so.

    Removing the swallow is what exposed it: the tests failed with
    `AttributeError: 'AuditLog' object has no attribute 'record'`. That is the
    argument for failing loudly in one line.
    """
    audit = _require_audit()
    if audit is None:
        raise HTTPException(
            503, "audit trail unavailable; refusing to record this consent "
                 "change without one")
    try:
        audit.append(
            action,
            student_id=student_id,
            actor=actor_id,
            details={**details, "consent_id": consent_id},
        )
    except Exception:
        logger.exception("audit write failed for action %s", action)
        raise HTTPException(
            503, "audit trail write failed; refusing to report this consent "
                 "change as complete")


def _log_read(user: User, what: str, **kw: Any) -> None:
    """Log that `user` read consent records (audit_log.record_pii_read;
    docs/compliance.md box 2). A consent record names the guardian and says
    whether the school may process this child's work, so reading it is an
    access to student PII like any other. With no audit log the read is
    refused, the same rule _audit_or_refuse applies to a consent change."""
    audit = _require_audit()
    if audit is None:
        raise HTTPException(
            503, "audit trail unavailable; refusing a consent read that cannot be logged")
    record_pii_read(audit, actor=user.id, what=what, **kw)


class RecordConsentRequest(Camel):
    student_id: str
    guardian_name: str
    guardian_relationship: str = "parent"
    method: str  # e.g. "Signed admission form", "Physical consent slip", "Parent portal confirmation"
    notes: Optional[str] = None
    purpose: str = "assessment_and_grading"


class ConsentStatusResponse(Camel):
    has_consent: bool
    record: Optional[ParentalConsentRecord] = None


@router.get("", response_model=list[ParentalConsentRecord])
def list_consents(
    user: User = Depends(require_staff),
) -> list[ParentalConsentRecord]:
    """List all parental consent records for the caller's school."""
    records = _require_store().list_for_school(user.school_id)
    _log_read(user, "consent_records", student_ids=[r.student_id for r in records])
    return records


# A class is ~40 students; 500 leaves room for a whole grade and stops one
# request from turning into thousands of consent lookups.
MAX_ROSTER = 500


class MissingConsentResponse(Camel):
    checked: int
    missing: list[str]


@router.get("/missing", response_model=MissingConsentResponse)
def missing_consents(
    student_ids: list[str] = Query(default_factory=list, alias="studentIds"),
    user: User = Depends(require_staff),
) -> MissingConsentResponse:
    """Which of these students have no active parental consent at the
    caller's school -- so a school can see, before a test, whose sheets
    grading will refuse (authz.require_consent). Withdrawn consent is listed
    as missing.

    Takes the roster as `studentIds` (repeat the parameter) rather than a
    class or assessment id: the server holds no roster for either. An
    assessment records how many students sit it, not which, and the web
    client's Classes page is where the class list lives."""
    ids = [sid.strip() for sid in student_ids if sid.strip()]
    if len(ids) > MAX_ROSTER:
        raise HTTPException(400, f"at most {MAX_ROSTER} studentIds per request")
    missing = students_without_consent(_require_store(), user.school_id, ids)
    _log_read(user, "consent_status", student_ids=ids)
    return MissingConsentResponse(checked=len(dict.fromkeys(ids)), missing=missing)


@router.get("/students/{student_id}", response_model=ConsentStatusResponse)
def get_student_consent(
    student_id: str,
    user: User = Depends(require_staff),
) -> ConsentStatusResponse:
    """Retrieve parental consent status for a specific student."""
    store = _require_store()
    record = store.get_consent(user.school_id, student_id)
    _log_read(user, "consent_record", student_id=student_id)
    return ConsentStatusResponse(
        has_consent=record is not None and record.status == "granted",
        record=record,
    )


@router.post("", response_model=ParentalConsentRecord)
def record_consent(
    req: RecordConsentRequest,
    user: User = Depends(require_staff),
) -> ParentalConsentRecord:
    """Record a verified parental consent event for a student."""
    if not req.student_id.strip():
        raise HTTPException(400, "studentId is required")
    if not req.guardian_name.strip():
        raise HTTPException(400, "guardianName is required")
    if not req.method.strip():
        raise HTTPException(400, "method is required")

    store = _require_store()
    record = store.record_consent(
        school_id=user.school_id,
        student_id=req.student_id.strip(),
        guardian_name=req.guardian_name.strip(),
        guardian_relationship=req.guardian_relationship.strip() or "parent",
        method=req.method.strip(),
        recorded_by=user.id,
        notes=req.notes,
        purpose=req.purpose,
    )

    _audit_or_refuse(
        "parental_consent_recorded",
        actor_id=user.id,
        consent_id=record.id,
        student_id=record.student_id,
        details={
            "school_id": user.school_id,
            "method": record.method,
            "guardian_name": record.guardian_name,
        },
    )

    return record


@router.delete("/students/{student_id}", response_model=ConsentStatusResponse)
def revoke_consent(
    student_id: str,
    user: User = Depends(require_staff),
) -> ConsentStatusResponse:
    """Revoke parental consent for a student."""
    store = _require_store()
    updated = store.revoke_consent(user.school_id, student_id, revoked_by=user.id)
    if not updated:
        raise HTTPException(404, f"no consent record found for student '{student_id}'")

    _audit_or_refuse(
        "parental_consent_revoked",
        actor_id=user.id,
        consent_id=updated.id,
        student_id=student_id,
        details={"school_id": user.school_id},
    )

    return ConsentStatusResponse(has_consent=False, record=updated)
