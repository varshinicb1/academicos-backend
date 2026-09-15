"""Parental consent API endpoints — DPDP Act 2023 compliance.

Endpoints for recording, querying, and revoking verifiable parental consent
for CBSE minors. All routes require an authenticated user and are strictly
scoped to the caller's school_id.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .audit_log import AuditLog, get_audit_log
from .auth_routes import get_current_user
from .consent import ConsentStore, ParentalConsentRecord, get_consent_store
from .schemas import Camel
from .users import User

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
            pass
    return _audit


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
    user: User = Depends(get_current_user),
) -> list[ParentalConsentRecord]:
    """List all parental consent records for the caller's school."""
    store = _require_store()
    return store.list_for_school(user.school_id)


@router.get("/students/{student_id}", response_model=ConsentStatusResponse)
def get_student_consent(
    student_id: str,
    user: User = Depends(get_current_user),
) -> ConsentStatusResponse:
    """Retrieve parental consent status for a specific student."""
    store = _require_store()
    record = store.get_consent(user.school_id, student_id)
    return ConsentStatusResponse(
        has_consent=record is not None and record.status == "granted",
        record=record,
    )


@router.post("", response_model=ParentalConsentRecord)
def record_consent(
    req: RecordConsentRequest,
    user: User = Depends(get_current_user),
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

    audit = _require_audit()
    if audit:
        try:
            audit.record(
                action="parental_consent_recorded",
                actor_id=user.id,
                resource_type="parental_consent",
                resource_id=record.id,
                details={
                    "school_id": user.school_id,
                    "student_id": record.student_id,
                    "method": record.method,
                    "guardian_name": record.guardian_name,
                },
            )
        except Exception:
            pass

    return record


@router.delete("/students/{student_id}", response_model=ConsentStatusResponse)
def revoke_consent(
    student_id: str,
    user: User = Depends(get_current_user),
) -> ConsentStatusResponse:
    """Revoke parental consent for a student."""
    store = _require_store()
    updated = store.revoke_consent(user.school_id, student_id, revoked_by=user.id)
    if not updated:
        raise HTTPException(404, f"no consent record found for student '{student_id}'")

    audit = _require_audit()
    if audit:
        try:
            audit.record(
                action="parental_consent_revoked",
                actor_id=user.id,
                resource_type="parental_consent",
                resource_id=updated.id,
                details={
                    "school_id": user.school_id,
                    "student_id": student_id,
                },
            )
        except Exception:
            pass

    return ConsentStatusResponse(has_consent=False, record=updated)
