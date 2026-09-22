"""Grade by question, not by student (C3).

`docs/research/case-studies.md` section 5 surveyed the assessment category and
singled this out:

> **Grade by question, not by student** (Gradescope). A teacher marking 40
> scripts is dramatically faster grading question 1 across all 40 than working
> student by student. Our evaluation flow is per-script. **This is the single
> most transferable UI idea in this entire document.**

The existing flow is `POST /evaluations/sheet/{assessment}/{student}/review/{q}`
-- one decision per student per question, reached by walking students. Marking
40 scripts that way means 40 context switches per question, and holding the
marking scheme for question 1 in your head while you also hold question 2's.

This module adds the transposed view: **one question, every student's answer,
one marking scheme, and one action that awards the same decision to many
students at once.** The teacher reads question 1's value points once and applies
them 40 times, which is how marking is actually done on paper.

Why this is safe to bulk-award
------------------------------
`awarded_marks` is clamped to `[0, max_marks]` per student inside the store's own
bounds, and every write goes through the same `GradedStore.save` the per-student
path uses -- so a bulk award cannot produce a state the single-student path could
not. It also cannot touch a different assessment or school: the assessment
ownership check is the same one the existing endpoint uses.

No student data leaves the school: these routes are behind the normal user auth,
not the question-bank API key, because a student's answer is exactly the thing
`docs/question-bank-api.md:119` says must never be reachable through a content
key.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .audit_log import record_pii_read
from .auth_routes import require_staff
from .authz import require_consent_for_all
from .grade_lock import record_grade_change, require_reason_if_finalized
from .schemas import Camel
from .users import User

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["evaluations"])

# Injected by `init`, same pattern as the sibling route modules, so this file
# does not import the store singleton directly and stays testable.
_deps: dict[str, Any] = {}


def init(*, graded, require, require_school_owns_assessment, questions_by_id,
         audit=None, consent=None) -> None:
    """Wire the module to the app's stores and guards.

    Passed in rather than imported so a test can substitute a fake store without
    monkeypatching module globals -- and so this module cannot accidentally
    reach a store the rest of the app has not initialised.

    `audit` is a zero-argument callable that returns the AuditLog. Without it
    the award route answers 503 and writes nothing, because a grading write
    that cannot be audited is refused (grade_lock.py).

    `consent` is a zero-argument callable that returns the ConsentStore. The
    same rule: without it the award route answers 503, because a grade change
    whose student's parental consent cannot be checked is refused
    (authz.require_consent).
    """
    _deps.update(
        audit=audit,
        consent=consent,
        graded=graded,
        require=require,
        require_school_owns_assessment=require_school_owns_assessment,
        questions_by_id=questions_by_id,
    )


def _graded():
    store = _deps.get("graded")
    if store is None:
        raise HTTPException(503, "evaluation store not initialised")
    return store


def _consents():
    get_consents = _deps.get("consent")
    if get_consents is None:
        raise HTTPException(
            503, "consent store not initialised; refusing a grade change that "
                 "cannot be checked against parental consent")
    return get_consents()


def _audit():
    get_audit = _deps.get("audit")
    if get_audit is None:
        raise HTTPException(503, "audit log not initialised; refusing an unaudited grade change")
    return get_audit()


# --------------------------------------------------------------------------- #
# read: one question across every student
# --------------------------------------------------------------------------- #

class QuestionAnswerRow(Camel):
    student_id: str
    answer_text: str = ""
    awarded_marks: int = 0
    max_marks: int = 0
    verdict: str = ""
    confidence: float = 0.0
    needs_review: bool = True
    reasoning: str = ""


class QuestionAcrossStudents(Camel):
    assessment_id: str
    question_id: str
    stem: str = ""
    marks: int = 0
    marking_points: list[dict[str, Any]] = Field(default_factory=list)
    scheme_provenance: str = "none"
    answers: list[QuestionAnswerRow] = Field(default_factory=list)
    total: int = 0
    already_graded: int = 0
    needs_review: int = 0


@router.get("/evaluations/assessment/{assessment_id}/question/{question_id}",
            response_model=QuestionAcrossStudents)
def question_across_students(
    assessment_id: str,
    question_id: str,
    current: User = Depends(require_staff),
) -> QuestionAcrossStudents:
    """Every student's answer to one question, with the marking scheme.

    The marking scheme is fetched once and returned with the batch, because the
    whole point of the transposed view is that the teacher reads the value
    points once rather than once per student.
    """
    require = _deps.get("require")
    owns = _deps.get("require_school_owns_assessment")
    if require:
        require()
    if owns:
        owns(assessment_id, current)

    store = _graded()
    sheets = store.for_assessment(assessment_id)
    if not sheets:
        raise HTTPException(404, "no evaluated sheets for this assessment")

    rows: list[QuestionAnswerRow] = []
    stem = ""
    marks = 0
    for student_id, graded in sheets.items():
        for question, evaluation in graded:
            if question.id != question_id:
                continue
            stem = stem or getattr(question, "stem", "") or getattr(question, "source_text", "")
            marks = marks or int(getattr(question, "marks", 0) or 0)
            rows.append(QuestionAnswerRow(
                student_id=student_id,
                answer_text=str(getattr(evaluation, "student_answer", "") or ""),
                awarded_marks=int(evaluation.awarded_marks),
                max_marks=int(evaluation.max_marks),
                verdict=str(getattr(evaluation, "verdict", "")),
                confidence=float(getattr(evaluation, "confidence", 0.0)),
                needs_review=bool(getattr(evaluation, "needs_review", True)),
                reasoning=str(getattr(evaluation, "reasoning", ""))[:400],
            ))
            break

    if not rows:
        raise HTTPException(404, f"question {question_id} is not in any sheet")

    rows.sort(key=lambda r: r.student_id)
    # Every student's answer is read here, so one audit entry lists them all
    # (docs/compliance.md box 2), written before any of it is returned.
    record_pii_read(_audit(), actor=current.id, what="answers_to_question",
                    assessment_id=assessment_id, questionId=question_id,
                    student_ids=[r.student_id for r in rows])
    scheme = _scheme_for(question_id)
    return QuestionAcrossStudents(
        assessment_id=assessment_id,
        question_id=question_id,
        stem=stem,
        marks=marks,
        marking_points=scheme.get("markingPoints") or [],
        scheme_provenance=scheme.get("provenance") or "none",
        answers=rows,
        total=len(rows),
        already_graded=sum(1 for r in rows if not r.needs_review),
        needs_review=sum(1 for r in rows if r.needs_review),
    )


def _scheme_for(question_id: str) -> dict[str, Any]:
    """The official scheme, if the question bank can supply one.

    Optional on purpose: grading must work for a teacher-authored question the
    bank has never seen, so a miss returns an empty scheme rather than failing
    the whole batch.
    """
    lookup = _deps.get("questions_by_id")
    if not lookup:
        return {}
    try:
        rec = lookup(question_id)
    except Exception:                                          # noqa: BLE001
        log.warning("scheme lookup failed for %s", question_id, exc_info=True)
        return {}
    return (rec or {}).get("answerScheme") or {}


# --------------------------------------------------------------------------- #
# write: one decision applied across many students
# --------------------------------------------------------------------------- #

class BulkAwardRequest(Camel):
    marks: int = Field(ge=0)
    student_ids: Optional[list[str]] = None
    # Explicitly opt in to "everyone", rather than inferring it from an empty
    # student_ids. A caller that forgot to send the list must not silently
    # award marks to the whole class.
    all_students: bool = False
    note: str = ""
    # Needed only when a target sheet is finalized (grade_lock.py). `note`
    # is the teacher's feedback on the answer; `reason` explains the change
    # and goes to the audit log.
    reason: str = ""


class BulkAwardResponse(Camel):
    assessment_id: str
    question_id: str
    marks: int
    updated: int = 0
    skipped: list[str] = Field(default_factory=list)
    unchanged: int = 0


@router.post("/evaluations/assessment/{assessment_id}/question/{question_id}/award",
             response_model=BulkAwardResponse)
def award_across_students(
    assessment_id: str,
    question_id: str,
    req: BulkAwardRequest,
    current: User = Depends(require_staff),
) -> BulkAwardResponse:
    """Award the same mark for one question to many students.

    This is the write half of grade-by-question and the reason it saves time: a
    teacher who has just read the value points decides once and applies it
    forty times.

    Deliberately server-side and per-student rather than a single bulk UPDATE:
    each student's sheet is loaded, clamped and saved through the same path the
    single-student endpoint uses, so a bulk award can never produce a state the
    per-student path could not, and the audit trail stays uniform.

    Every changed student gets its own audit entry, with the caller as the
    actor. If any target sheet is finalized and no reason is given, or any
    target student has no active parental consent, the whole request is
    refused with 409 before anything is written, so a bulk award never lands
    half-applied.
    """
    require = _deps.get("require")
    owns = _deps.get("require_school_owns_assessment")
    if require:
        require()
    if owns:
        owns(assessment_id, current)

    if not req.all_students and not req.student_ids:
        raise HTTPException(
            400, "send student_ids, or all_students=true to mean the whole class")
    if req.all_students and req.student_ids:
        raise HTTPException(400, "send student_ids or all_students=true, not both")

    store = _graded()
    sheets = store.for_assessment(assessment_id)
    if not sheets:
        raise HTTPException(404, "no evaluated sheets for this assessment")

    targets = list(sheets) if req.all_students else list(req.student_ids or [])
    updated = unchanged = 0
    skipped: list[str] = []
    audit = _audit()
    changes: list[tuple[str, list, int, Any, Any, int]] = []

    for student_id in targets:
        graded = sheets.get(student_id)
        if graded is None:
            skipped.append(student_id)
            continue
        idx = next((i for i, (q, _e) in enumerate(graded) if q.id == question_id), None)
        if idx is None:
            skipped.append(student_id)
            continue

        question, evaluation = graded[idx]
        # Clamp to this student's own maximum, never the request's idea of it.
        clamped = max(0, min(int(evaluation.max_marks), int(req.marks)))
        if clamped == int(evaluation.awarded_marks):
            unchanged += 1
            continue

        changes.append((student_id, graded, idx, question, evaluation, clamped))

    # Consent, like the lock, is checked for every target before the first
    # write, so one student without it refuses the whole request (409 naming
    # each such student) rather than awarding the rest. Every student whose
    # sheet holds this question is checked, changed or not: fail closed.
    require_consent_for_all(
        _consents(), current.school_id,
        [sid for sid in targets
         if sid in sheets and any(q.id == question_id for q, _e in sheets[sid])])

    # Every lock is checked before the first write (see the docstring).
    finalized = {sid: require_reason_if_finalized(audit, assessment_id, sid, req.reason)
                 for sid, *_rest in changes}

    from dataclasses import replace
    for student_id, graded, idx, question, evaluation, clamped in changes:
        updated_eval = replace(
            evaluation, awarded_marks=clamped, needs_review=False,
            reasoning=(req.note or evaluation.reasoning),
        )
        graded[idx] = (question, updated_eval)
        record_grade_change(audit, "grade_awarded", assessment_id=assessment_id,
                            student_id=student_id, actor=current.id,
                            finalized=finalized[student_id], reason=req.reason,
                            question_id=question_id,
                            before=int(evaluation.awarded_marks), after=clamped)
        store.save(assessment_id, student_id, graded)
        updated += 1

    return BulkAwardResponse(
        assessment_id=assessment_id, question_id=question_id, marks=req.marks,
        updated=updated, skipped=skipped, unchanged=unchanged,
    )
