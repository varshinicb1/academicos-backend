"""Mobile scan-and-grade pipeline: capture -> crop/enhance -> OCR ->
Sarvam-105b evaluation -> teacher swipe review -> export.

This is the orchestration layer over pieces that were each already built and
verified independently:
  * `imaging.py`       — document-edge crop + CLAHE lighting correction
  * `vision.py`         — Sarvam Vision OCR per page
  * `scan.py`           — splits OCR'd pages into per-question answers on the
                          question-number markers a student writes by hand
  * `llm_evaluate.py`   — Sarvam 105b scores one answer at a time

Sessions are in-memory (`_sessions`) backed by `ScanSessionStore` (Postgres
via Supabase when configured, local SQLite otherwise -- see that module's
docstring), same durability story as the other stores.

The captured photos and exported PDFs are a separate durability problem:
`crop_to_document`/`ocr_page` need real files on local disk to do their
work, so raw/processed images are still written to local scratch space
during a request. What survives a restart is the *upload* to Supabase
Storage right after processing (`_upload_page`/`_upload_pdf` below) -- each
CapturedPage/ScanSession carries a storage key alongside its local Path, and
mobile_routes.py's serving endpoints fall back to fetching from Storage
when the local file is gone.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import imaging, llm_evaluate, scan, vision
from .scan_session_store import ScanSessionStore
from .schemas import GeneratedQuestionSchema, QuestionSchema
from .supabase_kv import SupabaseStorage

log = logging.getLogger(__name__)
_storage = SupabaseStorage("scan-media")


@dataclass
class CapturedPage:
    page_no: int
    raw_path: Path
    processed_path: Path
    cropped: bool
    ocr_text: str
    warnings: list[str] = field(default_factory=list)
    raw_storage_key: str = ""
    processed_storage_key: str = ""


@dataclass
class ReviewItem:
    question_id: str
    display_number: int
    stem: str
    max_marks: int
    student_answer: str
    awarded_marks: int
    verdict: str
    confidence: float
    reasoning: str
    marking_points: list[dict[str, Any]]
    ocr_warnings: list[str]
    needs_review: bool
    status: str = "pending"          # pending | approved | edited
    teacher_marks: Optional[int] = None
    teacher_comment: str = ""
    # Captured page(s) this answer's text was read from — lets the review UI
    # show the actual handwritten photo next to the AI's transcription, not
    # just trust it blind.
    page_numbers: list[int] = field(default_factory=list)

    @property
    def final_marks(self) -> int:
        return self.teacher_marks if self.teacher_marks is not None else self.awarded_marks


@dataclass
class ScanSession:
    id: str
    assessment_id: str
    student_id: str
    student_name: str
    subject: str = ""
    grade: int = 10
    pages: list[CapturedPage] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)
    status: str = "capturing"        # capturing | processing | reviewing | finalized
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_pdf_path: Optional[Path] = None
    corrected_pdf_path: Optional[Path] = None
    raw_pdf_storage_key: str = ""
    corrected_pdf_storage_key: str = ""

    @property
    def workdir(self) -> Path:
        return _WORKDIR_ROOT / self.id

    @property
    def review_progress(self) -> tuple[int, int]:
        done = sum(1 for i in self.review if i.status != "pending")
        return done, len(self.review)


# In-memory cache for fast access within a single process lifetime; _store is
# the durable copy that survives a Render idle-restart (see module docstring).
_sessions: dict[str, ScanSession] = {}
_WORKDIR_ROOT = Path("academicos-data") / "scan-sessions"
_store: Optional[ScanSessionStore] = None


class ScanError(RuntimeError):
    pass


def _upload(session_id: str, key: str, path: Path, content_type: str) -> str:
    """Best-effort upload to Supabase Storage; returns the storage key on
    success, "" on failure or when Storage isn't configured. Never raises --
    a failed durability upload shouldn't sink a scan a teacher is mid-way
    through, it just means that one artifact stays local-only."""
    if not _storage.enabled:
        return ""
    full_key = f"{session_id}/{key}"
    try:
        _storage.upload(full_key, path.read_bytes(), content_type)
        return full_key
    except Exception as exc:
        log.warning("Supabase Storage upload failed for %s: %s", full_key, exc)
        return ""


def fetch_storage_bytes(key: str) -> Optional[bytes]:
    """Fetches an artifact from Supabase Storage by key; None if unavailable
    (Storage not configured, or the object doesn't exist there either --
    e.g. it was captured before this durability fix shipped)."""
    if not key or not _storage.enabled:
        return None
    try:
        return _storage.download(key)
    except Exception as exc:
        log.warning("Supabase Storage download failed for %s: %s", key, exc)
        return None


def configure_workdir(root: Path) -> None:
    global _WORKDIR_ROOT, _store
    _WORKDIR_ROOT = root
    _store = ScanSessionStore(root / "sessions.sqlite")


def save_session(session: ScanSession) -> None:
    _sessions[session.id] = session
    if _store is not None:
        _store.save(session)


def create_session(assessment_id: str, student_id: str, student_name: str,
                   subject: str = "", grade: int = 10) -> ScanSession:
    sid = f"scan_{uuid.uuid4().hex[:12]}"
    session = ScanSession(id=sid, assessment_id=assessment_id, student_id=student_id,
                          student_name=student_name, subject=subject, grade=grade)
    save_session(session)
    return session


def get_session(session_id: str) -> ScanSession:
    session = _sessions.get(session_id)
    if session is None and _store is not None:
        session = _store.get(session_id)
        if session is not None:
            _sessions[session_id] = session
    if session is None:
        raise ScanError(f"no scan session {session_id}")
    return session


def add_page(session: ScanSession, image_bytes: bytes, *, language: str = "en-IN") -> CapturedPage:
    """Ingests one captured page photo: saves the raw bytes, runs the crop +
    lighting pipeline, then OCRs the processed image. A failure in any one
    step degrades to "keep the raw/less-processed image and flag a warning"
    rather than losing the page — one bad photo should not sink a 12-page
    booklet a teacher already captured.
    """
    page_no = len(session.pages) + 1
    raw_dir = session.workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"page_{page_no:03d}.jpg"
    raw_path.write_bytes(image_bytes)

    warnings: list[str] = []
    processed_path = raw_path
    cropped = False
    try:
        result = imaging.preprocess_page(raw_path, session.workdir / "processed", page_no)
        processed_path = result.path
        cropped = result.cropped
        warnings.extend(result.warnings)
    except Exception as exc:
        log.warning("preprocessing failed for page %s of %s: %s", page_no, session.id, exc)
        warnings.append(f"image preprocessing failed ({exc}); used the original photo")

    ocr_text = ""
    try:
        ocr = vision.ocr_page(processed_path, page_no, workdir=session.workdir / "ocr",
                              language=language, crop_furniture=False)
        ocr_text = ocr.text
        warnings.extend(ocr.warnings)
    except Exception as exc:
        log.warning("OCR failed for page %s of %s: %s", page_no, session.id, exc)
        warnings.append(f"OCR failed for this page ({exc}); it will be treated as blank")

    page = CapturedPage(page_no=page_no, raw_path=raw_path, processed_path=processed_path,
                        cropped=cropped, ocr_text=ocr_text, warnings=warnings)
    page.raw_storage_key = _upload(session.id, f"raw/page_{page_no:03d}.jpg", raw_path, "image/jpeg")
    page.processed_storage_key = (
        _upload(session.id, f"processed/page_{page_no:03d}.jpg", processed_path, "image/jpeg")
        if processed_path != raw_path else page.raw_storage_key
    )
    session.pages.append(page)
    save_session(session)
    return page


def process_session(session: ScanSession,
                    questions_by_id: dict[str, QuestionSchema],
                    ordered_questions: list[GeneratedQuestionSchema]) -> list[ReviewItem]:
    """Segments the captured pages into per-question answers (by the
    question-number markers the student wrote) and scores each one against
    the assessment's real questions, in printed order.
    """
    if not session.pages:
        raise ScanError("no pages captured for this session")
    session.status = "processing"

    page_ocrs = [vision.PageOCR(page_no=p.page_no, text=p.ocr_text, warnings=p.warnings)
                for p in session.pages]
    segments = scan.segment_answers(page_ocrs)
    by_number = {s.question_no: s for s in segments}

    items: list[ReviewItem] = []
    for gq in ordered_questions:
        q = questions_by_id.get(gq.question_id)
        if q is None:
            log.warning("question %s in paper not found in question bank; skipped", gq.question_id)
            continue
        seg = by_number.get(gq.display_number)
        answer_text = seg.text if seg else ""
        seg_warnings = seg.warnings if seg else (["no answer found under this question number"])

        ev = llm_evaluate.evaluate_answer_llm(q, q.answer_scheme, answer_text)
        items.append(ReviewItem(
            question_id=q.id, display_number=gq.display_number, stem=q.stem,
            max_marks=ev.max_marks, student_answer=answer_text,
            awarded_marks=ev.awarded_marks, verdict=ev.verdict, confidence=ev.confidence,
            reasoning=ev.reasoning,
            marking_points=[{"id": m.marking_point_id, "description": m.description,
                             "awarded": m.awarded, "marks": m.marks, "reason": m.reason}
                            for m in ev.marking_points],
            ocr_warnings=list(dict.fromkeys(ev.ocr_warnings + seg_warnings)),
            needs_review=ev.needs_review,
            page_numbers=list(seg.page_numbers) if seg else [],
        ))
    items.sort(key=lambda i: i.display_number)
    session.review = items
    session.status = "reviewing"
    save_session(session)
    return items


def review_decision(session: ScanSession, question_id: str, action: str, *,
                    marks: Optional[int] = None, comment: str = "") -> ReviewItem:
    item = next((i for i in session.review if i.question_id == question_id), None)
    if item is None:
        raise ScanError(f"question {question_id} not in this session's review queue")
    if action == "approve":
        item.status = "approved"
    elif action == "edit":
        item.status = "edited"
        if marks is not None:
            item.teacher_marks = max(0, min(item.max_marks, marks))
        item.teacher_comment = comment
    else:
        raise ScanError(f"unknown review action {action!r}")
    save_session(session)
    return item


def finalize_totals(session: ScanSession) -> tuple[int, int]:
    awarded = sum(i.final_marks for i in session.review)
    maximum = sum(i.max_marks for i in session.review)
    return awarded, maximum


def export_raw_booklet_pdf(session: ScanSession, output_dir: Path) -> Path:
    """Assembles the captured page photos into one PDF — the permanent record
    of what the student actually wrote, independent of anything OCR or the
    AI evaluator did with it. Uses the *processed* (cropped/lit) image when
    available so the archived copy is the readable one, not the raw photo.
    """
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{session.id}_raw.pdf"
    doc = fitz.open()
    for page in session.pages:
        img_path = page.processed_path if page.processed_path.exists() else page.raw_path
        img = fitz.open(img_path)
        rect = img[0].rect
        pdf_page = doc.new_page(width=rect.width, height=rect.height)
        pdf_page.insert_image(rect, filename=str(img_path))
        img.close()
    doc.save(str(out_path))
    doc.close()
    session.raw_pdf_path = out_path
    session.raw_pdf_storage_key = _upload(session.id, "raw.pdf", out_path, "application/pdf")
    save_session(session)
    return out_path


def export_corrected_pdf(session: ScanSession, output_dir: Path, *,
                         template=None) -> Path:
    """The reviewed, marked-up result: per-question awarded marks, which
    marking points were met, and the teacher's own comment where they edited
    the AI's score. This is what a parent or the school office actually wants
    to see — not the raw OCR transcript.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from . import pdf as pdf_export

    pdf_export._register_unicode_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{session.id}_corrected.pdf"

    brand = colors.HexColor(template.brand_color) if template and template.brand_color else colors.black
    school_name = template.name if template and template.name else "AcademicOS School"
    ss = getSampleStyleSheet()
    school_style = ParagraphStyle("CSchool", parent=ss["Title"], fontName=pdf_export._BODY_FONT_BOLD,
                                  fontSize=15, textColor=brand, alignment=TA_CENTER)
    title_style = ParagraphStyle("CTitle", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD,
                                 fontSize=11, alignment=TA_CENTER, spaceBefore=2)
    meta_style = ParagraphStyle("CMeta", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                fontSize=9.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
    q_style = ParagraphStyle("CQ", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD, fontSize=9.5, leading=13)
    body_style = ParagraphStyle("CBody", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                fontSize=9, leading=12.5, alignment=TA_LEFT)
    point_style = ParagraphStyle("CPoint", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                 fontSize=8.5, leading=11.5, leftIndent=10,
                                 textColor=colors.HexColor("#333333"))
    comment_style = ParagraphStyle("CComment", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                   fontSize=8.5, leading=11.5, leftIndent=10,
                                   textColor=colors.HexColor("#8a5a00"))

    awarded, maximum = finalize_totals(session)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=16 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title=f"Corrected sheet — {session.student_name}", author="AssessmentOS")
    content_width = A4[0] - 36 * mm

    story: list = [
        Paragraph(pdf_export.escape(school_name), school_style),
        Paragraph("Corrected Answer Sheet", title_style),
        Paragraph(f"{pdf_export.escape(session.student_name)} &nbsp;|&nbsp; "
                  f"Roll/ID: {pdf_export.escape(session.student_id)} &nbsp;|&nbsp; "
                  f"Subject: {pdf_export.escape(session.subject)}", meta_style),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.2, color=brand, spaceAfter=6),
        Paragraph(f"<b>Total: {awarded} / {maximum}</b>",
                 ParagraphStyle("CTotal", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD,
                                fontSize=13, alignment=TA_CENTER, spaceAfter=10)),
    ]

    for item in sorted(session.review, key=lambda i: i.display_number):
        header = Table(
            [[Paragraph(f"Q{item.display_number}. {pdf_export.escape(item.stem[:140])}"
                       f"{'…' if len(item.stem) > 140 else ''}", q_style),
              Paragraph(f"<b>{item.final_marks} / {item.max_marks}</b>", q_style)]],
            colWidths=[content_width * 0.82, content_width * 0.18],
        )
        header.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#dddddd")),
        ]))
        story.append(header)
        for mp in item.marking_points:
            # Plain text, not a check/cross glyph: confirmed by testing that
            # neither Segoe UI nor Arial has U+2713/U+2717, so those printed
            # as tofu boxes on the actual exported PDF.
            mark = "[Awarded]" if mp["awarded"] else "[Not awarded]"
            story.append(Paragraph(f"{mark} {pdf_export.escape(mp['description'])} — "
                                   f"{pdf_export.escape(mp['reason'])} ({mp['marks']} mark)", point_style))
        if item.status == "edited" and item.teacher_comment:
            story.append(Paragraph(f"Teacher's note: {pdf_export.escape(item.teacher_comment)}",
                                   comment_style))
        elif item.status == "edited":
            story.append(Paragraph(
                f"Teacher adjusted the AI's score of {item.awarded_marks} to {item.final_marks}.",
                comment_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    session.corrected_pdf_path = out_path
    session.corrected_pdf_storage_key = _upload(session.id, "corrected.pdf", out_path, "application/pdf")
    save_session(session)
    return out_path
