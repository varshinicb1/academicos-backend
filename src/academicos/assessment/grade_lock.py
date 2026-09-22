"""A finalized grade can change only deliberately and visibly.

Audit item 8.5 (reproduced 2026-09-21): a sheet was finalized, then its marks
changed 0->2 through /award and back again, and the audit row count stayed at 2
before and after. Neither the sheet review nor the grade-by-question award
checked whether the sheet was finalized, and neither wrote an audit entry.

The rule, shared by every route that writes marks to the graded store:

* **Finalized means locked.** A sheet is finalized once its finalize entry
  (`FINALIZED_ACTION`) is in the audit log. Because the log is append-only, a
  sheet cannot be un-finalized. A mark change to a finalized sheet without a
  non-empty reason is refused with **409 Conflict**: the request is well-formed,
  but it conflicts with the sheet's state. (400 would suggest the body alone was
  wrong.) With a reason, the change goes through.
* **Every grading write is audited** with the authenticated actor, the marks
  before and after, the reason (if any) and whether the sheet was finalized.
  Before finalization a reason is optional, but the change is still logged.

The lock lives here and not in GradedStore because the fact that decides it is
in the audit log, and because the actor and the reason are request-level facts
the store never sees. mobile_scan.review_decision's regrade path had already
applied this rule to scan sessions (a reason and a reviewer before an existing
decision changes). This module applies it to the graded store.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException

from .audit_log import AuditLog

# The action finalize_sheet_review has written since before this rule existed.
# The Flutter offline engine (local_pillar_api.dart) writes the same name, so
# it is kept rather than renamed.
FINALIZED_ACTION = "sheet_reviewed"


def is_finalized(audit: AuditLog, assessment_id: str, student_id: str) -> bool:
    return audit.has_entry(FINALIZED_ACTION, assessment_id=assessment_id,
                           student_id=student_id)


def clean_reason(reason: Optional[str]) -> str:
    return (reason or "").strip()


def require_reason_if_finalized(audit: AuditLog, assessment_id: str,
                                student_id: str, reason: Optional[str]) -> bool:
    """Raise 409 if the sheet is finalized and no reason was given. Otherwise
    return whether the sheet is finalized, for the audit entry."""
    finalized = is_finalized(audit, assessment_id, student_id)
    if finalized and not clean_reason(reason):
        raise HTTPException(
            409,
            f"the sheet for student {student_id} is finalized; changing its marks "
            "requires a non-empty reason, which is recorded in the audit log",
        )
    return finalized


def record_grade_change(audit: AuditLog, action: str, *, assessment_id: str,
                        student_id: str, actor: str, finalized: bool,
                        reason: Optional[str], question_id: Optional[str] = None,
                        before: Any = None, after: Any = None,
                        extra: Optional[dict[str, Any]] = None) -> str:
    """Append one grading entry. An empty actor is refused instead of written
    as None, which is how `sheet_evaluated` used to be logged."""
    if not actor:
        raise HTTPException(403, "a grading write needs an authenticated actor")
    details: dict[str, Any] = {"before": before, "after": after,
                               "finalized": finalized}
    if question_id is not None:
        details["questionId"] = question_id
    if clean_reason(reason):
        details["reason"] = clean_reason(reason)
    details.update(extra or {})
    return audit.append(action, assessment_id=assessment_id, student_id=student_id,
                        actor=actor, details=details)
