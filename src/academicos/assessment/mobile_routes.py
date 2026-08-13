"""Mobile scan-and-grade API: capture pages, process against a real
assessment's questions, swipe-review the AI's scores, export.

Mounted alongside routes.py/pillar_routes.py under the same /api/v1 prefix.
Reaches into `routes._papers`/`routes._store` for the generated paper a scan
session is being marked against — the same in-memory pattern those modules
already use, not a new architecture.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field

from . import mobile_scan
from . import routes as assessment_routes
from .evaluate import Evaluation, MarkingPointOutcome
from .mapping import to_question_schema
from .pool import get_pool
from .schemas import Camel

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
def create_scan_session(req: CreateScanSessionRequest) -> ScanSessionResponse:
    assessment = assessment_routes._store.get(req.assessment_id) if assessment_routes._store else None
    subject = assessment.subject if assessment else "Science"
    grade = assessment.grade if assessment else 10
    session = mobile_scan.create_session(req.assessment_id, req.student_id, req.student_name,
                                         subject=subject, grade=grade)
    return _session_response(session)


class CapturedPageResponse(Camel):
    page_no: int
    cropped: bool
    ocr_preview: str
    warnings: list[str] = Field(default_factory=list)


@router.post("/scan/sessions/{session_id}/pages", response_model=CapturedPageResponse)
async def upload_scan_page(session_id: str, file: UploadFile = File(...)) -> CapturedPageResponse:
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
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
def process_scan_session(session_id: str) -> ProcessSessionResponse:
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))

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
def get_scan_review(session_id: str) -> ProcessSessionResponse:
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    return ProcessSessionResponse(items=[_item_response(session_id, i) for i in session.review])


class ReviewDecisionRequest(Camel):
    action: str          # "approve" | "edit"
    marks: int | None = None
    comment: str = ""


@router.post("/scan/sessions/{session_id}/review/{question_id}", response_model=ReviewItemResponse)
def submit_review_decision(session_id: str, question_id: str,
                           req: ReviewDecisionRequest) -> ReviewItemResponse:
    try:
        session = mobile_scan.get_session(session_id)
        item = mobile_scan.review_decision(session, question_id, req.action,
                                           marks=req.marks, comment=req.comment)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    return _item_response(session_id, item)


class FinalizeResponse(Camel):
    total_awarded: int
    total_max: int
    raw_pdf_url: str
    corrected_pdf_url: str


@router.post("/scan/sessions/{session_id}/finalize", response_model=FinalizeResponse)
def finalize_scan_session(session_id: str) -> FinalizeResponse:
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    if not session.review:
        raise HTTPException(409, "nothing to finalize — process the session first")

    from . import pillar_routes
    cfg, knowledge, templates = pillar_routes._require()
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
        pillar_routes._require_graded().save(session.assessment_id, session.student_id, graded)
        knowledge.record_evaluations(session.student_id, graded)

    return FinalizeResponse(
        total_awarded=awarded, total_max=maximum,
        raw_pdf_url=f"/api/v1/scan/sessions/{session_id}/raw-pdf",
        corrected_pdf_url=f"/api/v1/scan/sessions/{session_id}/corrected-pdf",
    )


@router.get("/scan/sessions/{session_id}/pages/{page_no}/image")
def get_page_image(session_id: str, page_no: int):
    """Serves the processed (cropped/lit) photo of one captured page — lets the
    review UI show a teacher the actual handwriting an answer was read from,
    instead of asking them to trust the transcription blind."""
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    page = next((p for p in session.pages if p.page_no == page_no), None)
    if page is None:
        raise HTTPException(404, f"no page {page_no} in this session")
    path = page.processed_path if page.processed_path.exists() else page.raw_path
    if not path.exists():
        raise HTTPException(404, "page image file not found on disk")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/scan/sessions/{session_id}/raw-pdf")
def get_raw_pdf(session_id: str):
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    if session.raw_pdf_path is None or not session.raw_pdf_path.exists():
        raise HTTPException(404, "raw booklet PDF not generated yet — finalize the session first")
    return FileResponse(str(session.raw_pdf_path), media_type="application/pdf",
                        filename=session.raw_pdf_path.name)


@router.get("/scan/sessions/{session_id}/corrected-pdf")
def get_corrected_pdf(session_id: str):
    try:
        session = mobile_scan.get_session(session_id)
    except mobile_scan.ScanError as e:
        raise HTTPException(404, str(e))
    if session.corrected_pdf_path is None or not session.corrected_pdf_path.exists():
        raise HTTPException(404, "corrected PDF not generated yet — finalize the session first")
    return FileResponse(str(session.corrected_pdf_path), media_type="application/pdf",
                        filename=session.corrected_pdf_path.name)
