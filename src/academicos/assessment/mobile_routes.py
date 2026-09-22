"""Mobile scan-and-grade API: capture pages, process against a real
assessment's questions, swipe-review the AI's scores, export.

Mounted alongside routes.py/pillar_routes.py under the same /api/v1 prefix.
Reaches into `routes._papers`/`routes._store` for the generated paper a scan
session is being marked against — the same in-memory pattern those modules
already use, not a new architecture.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import Field

from . import mobile_scan
from . import routes as assessment_routes
from .authz import require_consent, require_school_owns_assessment, require_school_owns_scan_session
from .auth_routes import require_staff
from .evaluate import Evaluation, MarkingPointOutcome
from .mapping import to_question_schema
from .pool import get_pool
from .schemas import Camel
from .users import User


def _get_session_owned(session_id: str, current: User) -> mobile_scan.ScanSession:
    """Fetch a scan session and verify it belongs to the caller's school --
    every route below needs exactly this pair of checks (404 missing / 403
    wrong school) before touching session state, scanned photos, or grades.
    Fixed 2026-09-15: every one of these routes used to skip both checks
    entirely (or, for submit_review_decision, use optional auth that never
    actually rejected an unauthenticated caller)."""
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    if assessment_routes._store is None:
        raise HTTPException(503, "assessment module not initialized")
    return require_school_owns_scan_session(session, assessment_routes._store, current)


def _get_session_to_process(session_id: str, current: User) -> mobile_scan.ScanSession:
    """_get_session_owned, plus the student's parental consent (DPDP Act
    2023 s.9; authz.require_consent). For every route that adds to or
    changes a session: a page photo, OCR and scoring, a review decision, the
    finalize into mastery. Consent is checked on each one, not only at
    creation, so a withdrawal mid-session stops the rest of it. The read
    routes (review queue, page image, PDFs) use _get_session_owned alone:
    looking at what was captured while consent held is not new processing."""
    session = _get_session_owned(session_id, current)
    require_consent(_consents(), current.school_id, session.student_id)
    return session


def _consents():
    from . import pillar_routes
    return pillar_routes._consents()


def _log_read(session: mobile_scan.ScanSession, current: User, what: str) -> None:
    """Log that `current` read this session's student's work: the review
    queue (every transcribed answer and its marks), a photo of the
    handwriting itself, or an exported PDF (docs/compliance.md box 2; see
    pillar_routes._log_read). Called once the data is found and before it
    is returned."""
    from . import pillar_routes
    pillar_routes._log_read(current, what, student_id=session.student_id,
                            assessment_id=session.assessment_id,
                            scanSessionId=session.id)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _questions_by_id(subject: str, grade_roman: str) -> dict:
    cfg, _, _ = _pillar_require()
    pool = get_pool(cfg, subject=subject, grade=grade_roman)
    return {q.question.canonical_id: to_question_schema(q) for q in pool.questions}


def _pillar_require():
    from . import pillar_routes
    return pillar_routes._require()


class CreateScanSessionRequest(Camel):
    assessment_id: str
    student_id: str
    student_name: str


class ScanSessionResponse(Camel):
    id: str
    assessment_id: str
    student_id: str
    student_name: str
    status: str
    pages_captured: int


def _session_response(session: mobile_scan.ScanSession) -> ScanSessionResponse:
    return ScanSessionResponse(
        id=session.id, assessment_id=session.assessment_id, student_id=session.student_id,
        student_name=session.student_name, status=session.status, pages_captured=len(session.pages),
    )


@router.post("/scan/sessions", response_model=ScanSessionResponse)
def create_scan_session(
    req: CreateScanSessionRequest, current: User = Depends(require_staff),
) -> ScanSessionResponse:
    if assessment_routes._store is None:
        raise HTTPException(503, "assessment module not initialized")
    # An unknown assessment used to become a Science class 10 session. Every later
    # route on it 404s through require_school_owns_scan_session, so it could never
    # be processed; refusing here reports the mistake where it is made.
    assessment = require_school_owns_assessment(assessment_routes._store, req.assessment_id, current)
    # DPDP: no scanning of a named student's work without recorded parental consent.
    require_consent(_consents(), current.school_id, req.student_id)
    session = mobile_scan.create_session(req.assessment_id, req.student_id, req.student_name,
                                         subject=assessment.subject, grade=assessment.grade,
                                         school_id=current.school_id)
    return _session_response(session)


class CapturedPageResponse(Camel):
    page_no: int
    cropped: bool
    ocr_preview: str
    warnings: list[str] = Field(default_factory=list)


@router.post("/scan/sessions/{session_id}/pages", response_model=CapturedPageResponse)
async def upload_scan_page(
    session_id: str, file: UploadFile = File(...), current: User = Depends(require_staff),
) -> CapturedPageResponse:
    session = _get_session_to_process(session_id, current)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "empty file upload")
    page = mobile_scan.add_page(session, image_bytes)
    return CapturedPageResponse(
        page_no=page.page_no, cropped=page.cropped,
        ocr_preview=page.ocr_text[:300], warnings=page.warnings,
    )


class ReviewItemResponse(Camel):
    question_id: str
    display_number: int
    stem: str
    max_marks: int
    student_answer: str
    awarded_marks: int
    verdict: str
    confidence: float
    reasoning: str
    marking_points: list[dict] = Field(default_factory=list)
    ocr_warnings: list[str] = Field(default_factory=list)
    needs_review: bool
    status: str
    teacher_marks: int | None = None
    teacher_comment: str = ""
    final_marks: int
    page_image_urls: list[str] = Field(default_factory=list)


def _item_response(session_id: str, item: mobile_scan.ReviewItem) -> ReviewItemResponse:
    return ReviewItemResponse(
        question_id=item.question_id, display_number=item.display_number, stem=item.stem,
        max_marks=item.max_marks, student_answer=item.student_answer,
        awarded_marks=item.awarded_marks, verdict=item.verdict, confidence=item.confidence,
        reasoning=item.reasoning, marking_points=item.marking_points,
        ocr_warnings=item.ocr_warnings, needs_review=item.needs_review, status=item.status,
        teacher_marks=item.teacher_marks, teacher_comment=item.teacher_comment,
        final_marks=item.final_marks,
        page_image_urls=[f"/api/v1/scan/sessions/{session_id}/pages/{p}/image"
                         for p in item.page_numbers],
    )


class ProcessSessionResponse(Camel):
    items: list[ReviewItemResponse]
    warnings: list[str] = Field(default_factory=list)


@router.post("/scan/sessions/{session_id}/process", response_model=ProcessSessionResponse)
def process_scan_session(session_id: str, current: User = Depends(require_staff)) -> ProcessSessionResponse:
    session = _get_session_to_process(session_id, current)

    papers = assessment_routes._require_papers()
    paper = papers.get(session.assessment_id)
    if paper is None:
        assessment = assessment_routes._store.get(session.assessment_id) if assessment_routes._store else None
        paper_id = assessment.generated_paper_id if assessment else None
        paper = papers.get(paper_id) if paper_id else None
    if paper is None:
        raise HTTPException(409, "this assessment has no generated paper to mark against")

    ordered_questions = [gq for section in paper.sections for gq in section.questions]
    grade_roman = assessment_routes._int_grade_to_roman(session.grade)
    q_by_id = _questions_by_id(session.subject, grade_roman)

    try:
        items = mobile_scan.process_session(session, q_by_id, ordered_questions)
    except mobile_scan.ScanError as e:
        raise HTTPException(400, str(e))

    warnings = []
    if len(items) < len(ordered_questions):
        warnings.append(f"only {len(items)}/{len(ordered_questions)} questions could be matched "
                        "to the question bank")
    return ProcessSessionResponse(items=[_item_response(session_id, i) for i in items], warnings=warnings)


@router.get("/scan/sessions/{session_id}/review", response_model=ProcessSessionResponse)
def get_scan_review(session_id: str, current: User = Depends(require_staff)) -> ProcessSessionResponse:
    session = _get_session_owned(session_id, current)
    _log_read(session, current, "scan_review")
    return ProcessSessionResponse(items=[_item_response(session_id, i) for i in session.review])


class ReviewDecisionRequest(Camel):
    action: str          # "approve" | "edit" | "regrade"
    marks: int | None = None
    comment: str = ""
    reason: str = ""     # required for regrades / edits that change an existing grade
    reviewer_id: str = ""  # required for regrades / edits that change an existing grade


@router.post("/scan/sessions/{session_id}/review/{question_id}", response_model=ReviewItemResponse)
def submit_review_decision(session_id: str, question_id: str,
                           req: ReviewDecisionRequest,
                           current: User = Depends(require_staff),
                           ) -> ReviewItemResponse:
    """Requires a real logged-in caller from the session's own school (fixed
    2026-09-15: auth used to be optional here -- get_current_user_optional
    never rejects an unauthenticated request, so this endpoint mutated real
    grades for any session/question with a client-supplied free-text
    reviewerId and no verification at all, the weakest of the handful of
    "gated" examples in this whole module). reviewer_id is always the real
    authenticated identity now; the request field stays on the wire schema
    only so an older client that still sends it doesn't 422."""
    session = _get_session_to_process(session_id, current)
    cfg, _ = assessment_routes._require()
    reviewer_id = current.id
    try:
        item = mobile_scan.review_decision(
            session, question_id, req.action,
            marks=req.marks, comment=req.comment,
            reason=req.reason, reviewer_id=reviewer_id,
            data_root=cfg.data_root,
        )
    except mobile_scan.ScanError as e:
        raise HTTPException(400, str(e))
    return _item_response(session_id, item)


class FinalizeResponse(Camel):
    total_awarded: int
    total_max: int
    raw_pdf_url: str
    corrected_pdf_url: str


@router.post("/scan/sessions/{session_id}/finalize", response_model=FinalizeResponse)
def finalize_scan_session(session_id: str, current: User = Depends(require_staff)) -> FinalizeResponse:
    session = _get_session_to_process(session_id, current)
    if not session.review:
        raise HTTPException(409, "nothing to finalize — process the session first")

    from . import pillar_routes
    from .grade_lock import FINALIZED_ACTION, is_finalized
    from .knowledge import sheet_source
    cfg, knowledge, templates = pillar_routes._require()
    # This route overwrites the graded sheet for the same assessment and
    # student, and it takes no reason. A sheet that is already finalized is
    # refused before any export or write. Corrections go through the review
    # or award routes, which take a reason (grade_lock.py, audit item 8.5).
    audit = pillar_routes._audit()
    if is_finalized(audit, session.assessment_id, session.student_id):
        raise HTTPException(
            409, f"the sheet for student {session.student_id} is already finalized; "
                 "correct individual marks through review or award with a reason")
    template = templates.default_for("school_1")

    mobile_scan.export_raw_booklet_pdf(session, cfg.data_root / "exports")
    mobile_scan.export_corrected_pdf(session, cfg.data_root / "exports", template=template)
    awarded, maximum = mobile_scan.finalize_totals(session)
    session.status = "finalized"
    mobile_scan.save_session(session)

    # Confirmed by testing: a finalized scan session never reached Teacher
    # Insights or Mastery — both read from pillar_routes._graded /
    # knowledge.record_evaluations, which only the older
    # POST /evaluations/sheet demo-evaluation path wrote to. The scan flow's
    # results lived only in mobile_scan's own session state, so a school
    # actually using it would see nothing on the teacher/principal
    # dashboards. Feed the teacher's final (not the AI's raw) marks into both.
    grade_roman = assessment_routes._int_grade_to_roman(session.grade)
    q_by_id = _questions_by_id(session.subject, grade_roman)
    graded: list[tuple] = []
    for item in session.review:
        question = q_by_id.get(item.question_id)
        if question is None:
            continue
        evaluation = Evaluation(
            question_id=item.question_id,
            awarded_marks=item.final_marks,
            max_marks=item.max_marks,
            verdict=item.verdict,
            confidence=item.confidence,
            reasoning=item.reasoning,
            marking_points=[MarkingPointOutcome(
                marking_point_id=mp.get("id", ""), description=mp.get("description", ""),
                awarded=mp.get("awarded", False), marks=mp.get("marks", 0),
                reason=mp.get("reason", ""), similarity=1.0 if mp.get("awarded") else 0.0,
            ) for mp in item.marking_points],
            ocr_warnings=item.ocr_warnings,
            needs_review=item.needs_review,
        )
        graded.append((question, evaluation))
    if graded:
        # The same finalize entry and mastery tag the sheet route writes, so
        # this sheet is locked from now on too, and a correction re-finalized
        # through the sheet route replaces these answers in mastery instead of
        # adding a second copy (knowledge.KnowledgeStore.record_sheet).
        mastery_source = sheet_source(session.assessment_id, session.student_id)
        audit.append(FINALIZED_ACTION, assessment_id=session.assessment_id,
                     student_id=session.student_id, actor=current.id,
                     details={"totalAwarded": awarded, "totalMax": maximum,
                              "questionCount": len(graded), "source": "scan",
                              "scanSessionId": session.id,
                              "masterySource": mastery_source, "refinalized": False})
        pillar_routes._require_graded().save(session.assessment_id, session.student_id, graded)
        knowledge.record_sheet(session.student_id, graded, mastery_source)

    return FinalizeResponse(
        total_awarded=awarded, total_max=maximum,
        raw_pdf_url=f"/api/v1/scan/sessions/{session_id}/raw-pdf",
        corrected_pdf_url=f"/api/v1/scan/sessions/{session_id}/corrected-pdf",
    )


@router.get("/scan/sessions/{session_id}/pages/{page_no}/image")
def get_page_image(session_id: str, page_no: int, current: User = Depends(require_staff)):
    """Serves the processed (cropped/lit) photo of one captured page — lets the
    review UI show a teacher the actual handwriting an answer was read from,
    instead of asking them to trust the transcription blind.

    Local disk first (fast path, same-instance), Supabase Storage second
    (survives a Render restart between capture and review).

    Requires the caller's school to own the session (fixed 2026-09-15 --
    previously served to anyone who knew/guessed a session_id, no auth at
    all, for what is literally a real student's handwritten answer sheet)."""
    session = _get_session_owned(session_id, current)
    page = next((p for p in session.pages if p.page_no == page_no), None)
    if page is None:
        raise HTTPException(404, f"no page {page_no} in this session")
    path = page.processed_path if page.processed_path.exists() else page.raw_path
    if path.exists():
        _log_read(session, current, "scan_page_image")
        return FileResponse(str(path), media_type="image/jpeg")
    data = mobile_scan.fetch_storage_bytes(page.processed_storage_key) \
        or mobile_scan.fetch_storage_bytes(page.raw_storage_key)
    if data is None:
        raise HTTPException(404, "page image not found on disk or in storage")
    _log_read(session, current, "scan_page_image")
    return Response(content=data, media_type="image/jpeg")


@router.get("/scan/sessions/{session_id}/raw-pdf")
def get_raw_pdf(session_id: str, current: User = Depends(require_staff)):
    session = _get_session_owned(session_id, current)
    if session.raw_pdf_path is not None and session.raw_pdf_path.exists():
        _log_read(session, current, "scan_raw_pdf")
        return FileResponse(str(session.raw_pdf_path), media_type="application/pdf",
                            filename=session.raw_pdf_path.name)
    data = mobile_scan.fetch_storage_bytes(session.raw_pdf_storage_key)
    if data is None:
        raise HTTPException(404, "raw booklet PDF not generated yet — finalize the session first")
    _log_read(session, current, "scan_raw_pdf")
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{session_id}_raw.pdf"'})


@router.get("/scan/sessions/{session_id}/corrected-pdf")
def get_corrected_pdf(session_id: str, current: User = Depends(require_staff)):
    session = _get_session_owned(session_id, current)
    if session.corrected_pdf_path is not None and session.corrected_pdf_path.exists():
        _log_read(session, current, "scan_corrected_pdf")
        return FileResponse(str(session.corrected_pdf_path), media_type="application/pdf",
                            filename=session.corrected_pdf_path.name)
    data = mobile_scan.fetch_storage_bytes(session.corrected_pdf_storage_key)
    if data is None:
        raise HTTPException(404, "corrected PDF not generated yet — finalize the session first")
    _log_read(session, current, "scan_corrected_pdf")
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{session_id}_corrected.pdf"'})
