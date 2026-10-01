"""The paper review workflow (EX-6): draft -> reviewer (the HOD, or any
teacher the author names) -> principal approval -> locked. When the reviewer
is the principal, approving the review is that approval and locks the paper.

The author, or the principal, asks one staff member to review; the paper is
then "underReview". The reviewer approves it for the principal or sends it
back with a comment, which returns it to editable ("paperGenerated"). The
principal's approval (PATCH /assessments/{id}/approve) locks it as before;
the principal may approve without a review, and the audit shows which. Each
step notifies the next person and is audited. Reviews are kept in the
operations store; the assessment's own status is the AssessmentStore's.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from ..operations.routes import notify_safely, store as ops
from .auth_routes import STAFF_ROLES, require_staff
from .authz import require_school_owns_assessment
from .users import User

router = APIRouter(prefix="/api/v1")

EDITABLE_FOR_REVIEW = {"paperGenerated", "questionOptimized", "questionsSelected", "underReview"}


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ReviewRequest(_Req):
    reviewer_id: str
    note: Optional[str] = Field(default=None, max_length=1000)


class ReviewDecision(_Req):
    decision: str                    # approve | changes
    comment: Optional[str] = Field(default=None, max_length=2000)


class ReviewResponse(Camel):
    id: str
    assessment_id: str
    title: str = ""
    author_id: str
    reviewer_id: str
    reviewer_name: str = ""
    state: str                       # pending | approved | changes_requested | withdrawn
    request_note: Optional[str] = None
    comment: Optional[str] = None
    requested_at: str
    decided_at: Optional[str] = None


def _assessments():
    from . import routes as ar
    return ar._require()[1]


def _audit(action: str, user: User, assessment_id: str, details: dict[str, Any]) -> None:
    from .audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, assessment_id=assessment_id, actor=user.id,
                                            details={"schoolId": user.school_id, **details})


def _out(r: dict, title: str = "") -> ReviewResponse:
    u = cr._require_users().get(r["reviewer_id"])
    return ReviewResponse(id=r["id"], assessment_id=r["assessment_id"], title=title, author_id=r["author_id"],
                          reviewer_id=r["reviewer_id"], reviewer_name=u.name if u else "", state=r["state"],
                          request_note=r["request_note"], comment=r["comment"], requested_at=r["requested_at"],
                          decided_at=r["decided_at"])


def _set_status(assessment, status: str) -> None:
    assessment.status = status
    assessment.updated_at = datetime.now(timezone.utc)
    _assessments().save(assessment)


@router.post("/assessments/{assessment_id}/review-request", response_model=ReviewResponse)
def request_review(assessment_id: str, req: ReviewRequest, current: User = Depends(require_staff)) -> ReviewResponse:
    """The author (or the principal) sends the paper to one reviewer. A new
    request replaces a pending one."""
    a = require_school_owns_assessment(_assessments(), assessment_id, current)
    if current.role != "principal" and a.teacher_id != current.id:
        raise HTTPException(403, "only the paper's author or the principal asks for a review")
    if a.status not in EDITABLE_FOR_REVIEW:
        raise HTTPException(409, f"a {a.status} paper cannot be sent for review")
    reviewer = cr._require_users().get(req.reviewer_id)
    if reviewer is None or reviewer.school_id != current.school_id or reviewer.role not in STAFF_ROLES:
        raise HTTPException(422, "the reviewer must be a teacher or the principal of this school")
    if reviewer.id == a.teacher_id:
        raise HTTPException(422, "the author cannot review their own paper")
    r = ops().request_review(school_id=current.school_id, assessment_id=a.id, author_id=a.teacher_id,
                             reviewer_id=reviewer.id, note=req.note)
    before = a.status
    _set_status(a, "underReview")
    _audit("paper_review_requested", current, a.id, {"reviewerId": reviewer.id, "before": before,
                                                      "after": "underReview"})
    notify_safely(school_id=current.school_id, user_ids=[reviewer.id], kind="paper_review_requested",
                  params={"title": a.title, "author": current.name}, link=f"/assessments/{a.id}",
                  dedupe_key=f"review:{r['id']}")
    return _out(r, a.title)


@router.post("/assessments/{assessment_id}/review-decision", response_model=ReviewResponse)
def decide_review(assessment_id: str, req: ReviewDecision, current: User = Depends(require_staff)) -> ReviewResponse:
    """The reviewer (or the principal) approves the paper for the
    principal, or sends it back to its author with a comment."""
    a = require_school_owns_assessment(_assessments(), assessment_id, current)
    r = ops().pending_review_for(a.id)
    if r is None:
        raise HTTPException(409, "this paper has no review waiting")
    if current.id != r["reviewer_id"] and current.role != "principal":
        raise HTTPException(403, "only the named reviewer or the principal decides this review")
    if req.decision not in ("approve", "changes"):
        raise HTTPException(422, "decision is approve or changes")
    if req.decision == "changes" and not (req.comment or "").strip():
        raise HTTPException(422, "say what to change")
    state = "approved" if req.decision == "approve" else "changes_requested"
    r = ops().decide_review(r["id"], state=state, comment=(req.comment or "").strip() or None)
    before = a.status
    if state == "changes_requested":
        _set_status(a, "paperGenerated")
    # The principal's approval is the final one: approving a review they were
    # asked for left the paper "under review", waiting for the principal to
    # approve it a second time from the paper (E2E run, 2026-10-01).
    locked = state == "approved" and current.role == "principal"
    if locked:
        from .routes import approve_assessment
        a = approve_assessment(a.id, principal=current)
    _audit("paper_review_decided", current, a.id, {"reviewId": r["id"], "decision": state, "before": before,
                                                    "after": a.status})
    notify_safely(school_id=current.school_id, user_ids=[a.teacher_id], kind="paper_review_decided",
                  params={"title": a.title, "reviewer": current.name,
                          "decision": ("approved and locked it" if locked
                                       else "approved it for the principal" if state == "approved"
                                       else "asked for changes"),
                          "decision_hi": ("स्वीकृत कर लॉक किया" if locked
                                          else "प्रधानाचार्य के लिए स्वीकृत किया" if state == "approved"
                                          else "बदलाव माँगे"),
                          "comment": r["comment"] or ""},
                  link=f"/assessments/{a.id}", dedupe_key=f"reviewed:{r['id']}")
    if state == "approved" and not locked:
        principals = [u.id for u in cr._require_users().users_for_school(current.school_id, role="principal")]
        notify_safely(school_id=current.school_id, user_ids=principals, kind="principal_alert",
                      params={"title": f"Paper ready to approve: {a.title}",
                              "body": f"{current.name} reviewed it and approved it for you.",
                              "title_hi": f"स्वीकृति हेतु प्रश्नपत्र तैयार: {a.title}",
                              "body_hi": f"{current.name} ने इसकी समीक्षा कर आपके लिए स्वीकृत किया है।"},
                      link=f"/assessments/{a.id}", dedupe_key=f"ready:{r['id']}")
    return _out(r, a.title)


@router.get("/assessments/{assessment_id}/reviews", response_model=list[ReviewResponse])
def list_reviews(assessment_id: str, current: User = Depends(require_staff)) -> list[ReviewResponse]:
    a = require_school_owns_assessment(_assessments(), assessment_id, current)
    return [_out(r, a.title) for r in ops().reviews_for(a.id)]


@router.get("/paper-reviews", response_model=list[ReviewResponse])
def my_reviews(state: Optional[str] = Query(default="pending"), current: User = Depends(require_staff)
               ) -> list[ReviewResponse]:
    """The papers waiting for the caller's review; the principal sees the
    school's."""
    rows = ops().reviews_where(current.school_id, reviewer_id=None if current.role == "principal" else current.id,
                               state=state or None)
    store = _assessments()
    out = []
    for r in rows:
        a = store.get(r["assessment_id"])
        out.append(_out(r, a.title if a else ""))
    return out
