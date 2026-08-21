"""FastAPI surface for Pillars 1(templates/marking) through 6.

Kept separate from `routes.py` (the Assessment Designer) so each pillar's
endpoints stay readable. Mounted under the same /api/v1 prefix.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import Field

from ..config import Config
from . import insights as insights_mod
from . import mailer as mailer_mod
from . import remediation as remediation_mod
from .evaluate import Evaluation, evaluate_answer
from .graded_store import GradedStore
from .knowledge import KnowledgeStore
from .mapping import to_question_schema
from .marking import build_answer_key, build_answer_scheme
from .pool import get_pool
from .practice_store import PracticeStore
from .schemas import (
    AnswerSchemeSchema,
    Camel,
    QuestionSchema,
    SchoolTemplate,
    SectionBlueprint,
)
from .school_templates import TemplateStore
from ..syllabus.cbse_syllabus import load_syllabus
from ..syllabus.timetable import generate_timetable

router = APIRouter(prefix="/api/v1")

_cfg: Optional[Config] = None
_knowledge: Optional[KnowledgeStore] = None
_templates: Optional[TemplateStore] = None
_graded: Optional[GradedStore] = None
_practice: Optional[PracticeStore] = None
# Write-only scratch state (never read back) -- fine to lose on restart.
_answer_keys: dict[str, dict[str, AnswerSchemeSchema]] = {}


def init(config: Config) -> None:
    global _cfg, _knowledge, _templates, _graded, _practice
    _cfg = config
    _knowledge = KnowledgeStore(config.data_root / "knowledge")
    _templates = TemplateStore(config.data_root / "templates" / "templates.sqlite")
    _graded = GradedStore(config.data_root / "assessments" / "graded.sqlite")
    _practice = PracticeStore(config.data_root / "assessments" / "practice.sqlite")


def _require() -> tuple[Config, KnowledgeStore, TemplateStore]:
    if _cfg is None or _knowledge is None or _templates is None:
        raise HTTPException(503, "pillar module not initialized")
    return _cfg, _knowledge, _templates


def _require_graded() -> GradedStore:
    if _graded is None:
        raise HTTPException(503, "pillar module not initialized")
    return _graded


def _require_practice() -> PracticeStore:
    if _practice is None:
        raise HTTPException(503, "pillar module not initialized")
    return _practice


def _pool_questions(subject: str = "Science", grade: str = "X") -> list[QuestionSchema]:
    cfg, _, _ = _require()
    return [to_question_schema(q) for q in get_pool(cfg, subject=subject, grade=grade).questions]


# ---------------- Pillar 1: templates + marking ----------------

class TemplateSaveRequest(Camel):
    template: SchoolTemplate
    sections: list[SectionBlueprint] = Field(default_factory=list)


@router.get("/schools/{school_id}/school-templates", response_model=list[SchoolTemplate])
def list_templates(school_id: str) -> list[SchoolTemplate]:
    _, _, store = _require()
    existing = store.list_for_school(school_id)
    return existing or [store.default_for(school_id)]


@router.post("/schools/{school_id}/school-templates", response_model=SchoolTemplate)
def save_template(school_id: str, req: TemplateSaveRequest) -> SchoolTemplate:
    _, _, store = _require()
    template = req.template.model_copy(update={"school_id": school_id})
    sections = req.sections or store.sections_for(school_id, None, 80)
    return store.save(template, sections)


@router.get("/schools/{school_id}/school-templates/{template_id}/sections",
            response_model=list[SectionBlueprint])
def template_sections(school_id: str, template_id: str) -> list[SectionBlueprint]:
    _, _, store = _require()
    return store.sections_for(school_id, template_id, 80)


@router.delete("/schools/{school_id}/school-templates/{template_id}")
def delete_template(school_id: str, template_id: str) -> dict:
    _, _, store = _require()
    store.delete(template_id)
    return {"ok": True}


class AnswerKeyRequest(Camel):
    assessment_id: str
    questions: list[QuestionSchema]
    correct_options: dict[str, str] = Field(default_factory=dict)


class AnswerKeyResponse(Camel):
    assessment_id: str
    schemes: dict[str, AnswerSchemeSchema]
    total_marks: int
    objective_count: int


@router.post("/assessments/{assessment_id}/answer-key", response_model=AnswerKeyResponse)
def make_answer_key(assessment_id: str, req: AnswerKeyRequest) -> AnswerKeyResponse:
    schemes = build_answer_key(req.questions, req.correct_options)
    _answer_keys[assessment_id] = schemes
    return AnswerKeyResponse(
        assessment_id=assessment_id, schemes=schemes,
        total_marks=sum(s.total_marks for s in schemes.values()),
        objective_count=sum(1 for s in schemes.values() if s.metadata.get("objective")),
    )


# ---------------- Catalog: what is actually in the bank ----------------

class CatalogEntry(Camel):
    subject: str
    grade: int
    question_count: int
    chapters: int
    marks_available: list[int] = Field(default_factory=list)


class ChapterEntry(Camel):
    chapter_id: str
    chapter_name: str
    question_count: int
    marks_available: list[int] = Field(default_factory=list)
    unit_name: str | None = None
    syllabus_marks: int | None = None  # official CBSE marks-weightage for this chapter's unit


class CatalogResponse(Camel):
    entries: list[CatalogEntry]
    total_questions: int


_GRADE_ROMAN = {8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}
# Subjects the bulk board ingest stages. Kept explicit so the catalog never
# probes hundreds of empty (subject, grade) pairs on every request.
_CATALOG_SCOPE: list[tuple[str, int]] = [
    ("Science", 10), ("Mathematics", 10), ("Social Science", 10), ("English", 10),
    ("Physics", 12), ("Chemistry", 12), ("Biology", 12), ("Mathematics", 12),
    ("Accountancy", 12), ("Business Studies", 12), ("Economics", 12),
    ("History", 12), ("Geography", 12), ("Political Science", 12), ("English", 12),
]


@router.get("/catalog", response_model=CatalogResponse)
def catalog(response: Response) -> CatalogResponse:
    """Subjects/grades that actually have questions, with counts.

    Drives the question-bank and syllabus screens so they show what exists
    rather than a hard-coded subject list.
    """
    cfg, _, _ = _require()
    entries: list[CatalogEntry] = []
    total = 0
    for subject, grade in _CATALOG_SCOPE:
        pool = get_pool(cfg, subject=subject, grade=_GRADE_ROMAN.get(grade, "X"))
        if not pool.questions:
            continue
        schemas = [to_question_schema(q) for q in pool.questions]
        chapters = {c for q in schemas for c in q.chapter_ids}
        entries.append(CatalogEntry(
            subject=subject, grade=grade, question_count=len(schemas),
            chapters=len(chapters),
            marks_available=sorted({q.marks for q in schemas}),
        ))
        total += len(schemas)
    entries.sort(key=lambda e: (e.grade, -e.question_count))
    response.headers["Cache-Control"] = "public, max-age=300"
    return CatalogResponse(entries=entries, total_questions=total)


@router.get("/catalog/{subject}/{grade}/chapters", response_model=list[ChapterEntry])
def catalog_chapters(subject: str, grade: int, response: Response) -> list[ChapterEntry]:
    """Chapter breakdown for one subject — the real syllabus view."""
    cfg, _, _ = _require()
    pool = get_pool(cfg, subject=subject, grade=_GRADE_ROMAN.get(grade, "X"))
    schemas = [to_question_schema(q) for q in pool.questions]

    by_chapter: dict[str, list[QuestionSchema]] = {}
    for q in schemas:
        for cid in (q.chapter_ids or ["unmapped"]):
            by_chapter.setdefault(cid, []).append(q)

    # Real official CBSE units/chapters, when we have them (currently Class
    # X only) — shown even with zero questions yet, a gap the teacher should
    # see rather than one silently hidden behind an "unmapped" bucket.
    syllabus = load_syllabus(subject, grade)
    out: list[ChapterEntry] = []
    seen: set[str] = set()
    if syllabus is not None:
        for unit, chapter in syllabus.all_chapters():
            qs = by_chapter.get(chapter.id, [])
            out.append(ChapterEntry(
                chapter_id=chapter.id, chapter_name=chapter.name,
                question_count=len(qs), marks_available=sorted({q.marks for q in qs}),
                unit_name=unit.name, syllabus_marks=unit.marks,
            ))
            seen.add(chapter.id)

    for cid in sorted(by_chapter):
        if cid in seen or cid == "unmapped":
            continue
        qs = by_chapter[cid]
        out.append(ChapterEntry(
            chapter_id=cid, chapter_name=cid.replace("-", " ").title(),
            question_count=len(qs), marks_available=sorted({q.marks for q in qs}),
        ))
    if "unmapped" in by_chapter and not syllabus:
        qs = by_chapter["unmapped"]
        out.append(ChapterEntry(
            chapter_id="unmapped", chapter_name="Unmapped",
            question_count=len(qs), marks_available=sorted({q.marks for q in qs}),
        ))
    response.headers["Cache-Control"] = "public, max-age=300"
    return out


# ---------------- Syllabus pacing (AI-suggested weekly timetable) ----------------

class SyllabusUnitResponse(Camel):
    unit_no: str
    name: str
    marks: int
    chapter_names: list[str] = Field(default_factory=list)


class SyllabusResponse(Camel):
    subject: str
    grade: int
    total_marks: int
    source: str
    units: list[SyllabusUnitResponse]


@router.get("/syllabus/{subject}/{grade}", response_model=SyllabusResponse)
def get_syllabus(subject: str, grade: int, response: Response) -> SyllabusResponse:
    doc = load_syllabus(subject, grade)
    if doc is None:
        raise HTTPException(404, f"no CBSE syllabus data for {subject} grade {grade} yet")
    response.headers["Cache-Control"] = "public, max-age=300"
    return SyllabusResponse(
        subject=doc.subject, grade=doc.grade, total_marks=doc.total_marks, source=doc.source,
        units=[
            SyllabusUnitResponse(
                unit_no=u.unit_no, name=u.name, marks=u.marks,
                chapter_names=[c.name for c in u.chapters],
            )
            for u in doc.units
        ],
    )


class UnitAllocationResponse(Camel):
    unit_name: str
    marks: int
    suggested_periods: int


class WeekSlotResponse(Camel):
    week: int
    unit_name: str
    periods: int


class TimetableResponse(Camel):
    subject: str
    grade: int
    periods_per_week: int
    weeks: int
    allocations: list[UnitAllocationResponse]
    schedule: list[WeekSlotResponse]


@router.get("/syllabus/{subject}/{grade}/timetable", response_model=TimetableResponse)
def get_timetable(subject: str, grade: int, response: Response, periods_per_week: int = 6, weeks: int = 20) -> TimetableResponse:
    """AI-suggested pacing: the school's weekly periods allocated proportionally
    to each unit's official CBSE marks-weightage (see timetable.py's docstring
    for why marks-weightage, not hours, is the source signal)."""
    tt = generate_timetable(subject, grade, periods_per_week=periods_per_week, weeks=weeks)
    if tt is None:
        raise HTTPException(404, f"no CBSE syllabus data for {subject} grade {grade} yet")
    response.headers["Cache-Control"] = "public, max-age=300"
    return TimetableResponse(
        subject=tt.subject, grade=tt.grade, periods_per_week=tt.periods_per_week, weeks=tt.weeks,
        allocations=[UnitAllocationResponse(unit_name=a.unit_name, marks=a.marks, suggested_periods=a.suggested_periods) for a in tt.allocations],
        schedule=[WeekSlotResponse(week=s.week, unit_name=s.unit_name, periods=s.periods) for s in tt.schedule],
    )


# ---------------- Email delivery ----------------

class MailBackendStatus(Camel):
    backends: dict[str, dict[str, Any]]
    any_configured: bool


@router.get("/mail/status", response_model=MailBackendStatus)
def mail_status() -> MailBackendStatus:
    backends = mailer_mod.available_backends()
    return MailBackendStatus(
        backends=backends,
        any_configured=any(b["configured"] for b in backends.values()),
    )


class SendPaperRequest(Camel):
    paper_id: str
    recipients: list[str]
    subject_name: str = "Science"
    grade: int = 10
    paper_title: str = "Question Paper"
    school_name: str = "AcademicOS School"
    note: str = ""
    include_answer_key: bool = True
    prefer: str = "auto"          # auto | composio | smtp


class SendPaperResponse(Camel):
    ok: bool
    backend: str
    detail: str
    recipients: list[str]


@router.post("/mail/send-paper", response_model=SendPaperResponse)
def send_paper(req: SendPaperRequest) -> SendPaperResponse:
    """Email an exported paper (and its answer key) to the given recipients.

    The PDF must already have been exported — this endpoint never regenerates
    or silently creates content it then sends out.
    """
    cfg, _, _ = _require()
    papers_dir = cfg.artifacts_dir / "papers"
    pdf = papers_dir / f"{req.paper_id}.pdf"
    if not pdf.exists():
        raise HTTPException(404, f"paper {req.paper_id} has not been exported to PDF yet")
    key_pdf = papers_dir / f"{req.paper_id}_answer_key.pdf"

    msg = mailer_mod.paper_email(
        paper_title=req.paper_title, school=req.school_name,
        subject_name=req.subject_name, grade=req.grade, pdf=pdf,
        answer_key=key_pdf if req.include_answer_key else None,
        recipients=req.recipients, note=req.note,
    )
    try:
        result = mailer_mod.send(msg, prefer=req.prefer)
    except mailer_mod.MailError as e:
        raise HTTPException(400, str(e))
    return SendPaperResponse(ok=result.ok, backend=result.backend,
                             detail=result.detail, recipients=result.recipients)


# ---------------- Pillar 2: evaluation ----------------

class EvaluateAnswerRequest(Camel):
    assessment_id: str
    student_id: str
    question: QuestionSchema
    student_answer: str
    answer_scheme: Optional[AnswerSchemeSchema] = None


class EvaluationResponse(Camel):
    question_id: str
    stem: str = ""
    student_answer: str = ""
    awarded_marks: int
    max_marks: int
    percentage: float
    verdict: str
    confidence: float
    reasoning: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    ocr_warnings: list[str] = Field(default_factory=list)
    needs_review: bool = True
    marking_points: list[dict[str, Any]] = Field(default_factory=list)


def _to_response(ev: Evaluation, question: QuestionSchema, student_answer: str) -> EvaluationResponse:
    return EvaluationResponse(
        question_id=ev.question_id, stem=question.stem, student_answer=student_answer,
        awarded_marks=ev.awarded_marks, max_marks=ev.max_marks,
        percentage=ev.percentage, verdict=ev.verdict, confidence=ev.confidence,
        reasoning=ev.reasoning, strengths=ev.strengths, gaps=ev.gaps,
        misconceptions=ev.misconceptions, ocr_warnings=ev.ocr_warnings,
        needs_review=ev.needs_review,
        marking_points=[{
            "id": m.marking_point_id, "description": m.description,
            "awarded": m.awarded, "marks": m.marks, "reason": m.reason,
        } for m in ev.marking_points],
    )


@router.post("/evaluations/answer", response_model=EvaluationResponse)
def evaluate_one(req: EvaluateAnswerRequest) -> EvaluationResponse:
    _require()
    scheme = req.answer_scheme or build_answer_scheme(req.question)
    concept = req.question.chapter_ids[0] if req.question.chapter_ids else None
    ev = evaluate_answer(req.question, scheme, req.student_answer, concept_label=concept)
    store = _require_graded()
    bucket = store.get(req.assessment_id, req.student_id) or []
    bucket = [(q, e) for q, e in bucket if q.id != req.question.id]
    bucket.append((req.question, ev))
    store.save(req.assessment_id, req.student_id, bucket)
    return _to_response(ev, req.question, req.student_answer)


class EvaluateSheetRequest(Camel):
    assessment_id: str
    student_id: str
    student_name: str = ""
    questions: list[QuestionSchema]
    answers: dict[str, str]
    correct_options: dict[str, str] = Field(default_factory=dict)


class EvaluateSheetResponse(Camel):
    assessment_id: str
    student_id: str
    total_awarded: int
    total_max: int
    percentage: float
    needs_review_count: int
    evaluations: list[EvaluationResponse]


@router.post("/evaluations/sheet", response_model=EvaluateSheetResponse)
def evaluate_sheet(req: EvaluateSheetRequest) -> EvaluateSheetResponse:
    cfg, knowledge, _ = _require()
    schemes = build_answer_key(req.questions, req.correct_options)
    graded: list[tuple[QuestionSchema, Evaluation]] = []
    for q in req.questions:
        concept = q.chapter_ids[0] if q.chapter_ids else None
        ev = evaluate_answer(q, schemes[q.id], req.answers.get(q.id, ""), concept_label=concept)
        graded.append((q, ev))

    _require_graded().save(req.assessment_id, req.student_id, graded)
    knowledge.record_evaluations(req.student_id, graded)   # Pillar 3 update

    awarded = sum(e.awarded_marks for _, e in graded)
    maximum = sum(e.max_marks for _, e in graded)
    from .audit_log import get_audit_log
    get_audit_log(cfg.data_root).append(
        "sheet_evaluated", assessment_id=req.assessment_id, student_id=req.student_id,
        details={"totalAwarded": awarded, "totalMax": maximum, "questionCount": len(graded)},
    )
    return EvaluateSheetResponse(
        assessment_id=req.assessment_id, student_id=req.student_id,
        total_awarded=awarded, total_max=maximum,
        percentage=round(100.0 * awarded / maximum, 2) if maximum else 0.0,
        needs_review_count=sum(1 for _, e in graded if e.needs_review),
        evaluations=[_to_response(e, q, req.answers.get(q.id, "")) for q, e in graded],
    )


# ---------------- Pillar 3: knowledge ----------------

class ConceptMasteryResponse(Camel):
    concept_id: str
    concept_name: str
    mastery: float
    accuracy: float
    retention: float
    confidence: float
    evidence_count: int
    confident: bool
    status: str
    is_weak: bool


class StudentMasteryResponse(Camel):
    student_id: str
    overall_mastery: float
    concepts: list[ConceptMasteryResponse]
    weak_concepts: list[str]


@router.get("/knowledge/{student_id}", response_model=StudentMasteryResponse)
def student_knowledge(student_id: str) -> StudentMasteryResponse:
    _, knowledge, _ = _require()
    views = knowledge.mastery(student_id)
    return StudentMasteryResponse(
        student_id=student_id,
        overall_mastery=knowledge.overall(student_id),
        concepts=[ConceptMasteryResponse(
            concept_id=v.concept_id, concept_name=v.concept_name, mastery=v.mastery,
            accuracy=v.accuracy, retention=v.retention, confidence=v.confidence,
            evidence_count=v.evidence_count, confident=v.confident, status=v.status,
            is_weak=v.is_weak) for v in views],
        weak_concepts=[v.concept_name for v in knowledge.weak_concepts(student_id)],
    )


@router.get("/knowledge/{student_id}/report")
def student_progress_report(student_id: str, student_name: str = "Student",
                            subject: str = "Science"):
    """Branded PDF progress report: concept mastery table + concrete next
    steps per weak concept. See report_pdf.py for what "branded" means here —
    the school's own template (name, logo, brand color), not a generic export.
    """
    from . import report_pdf
    from .school_templates import TemplateStore

    cfg, knowledge, _ = _require()
    views = knowledge.mastery(student_id)
    if not views:
        raise HTTPException(404, f"no graded answers recorded for {student_id}")
    template = TemplateStore(cfg.data_root / "templates" / "templates.sqlite").default_for("school_1")
    path = report_pdf.export_progress_report_pdf(
        student_name, student_id, subject, views, cfg.data_root / "exports", template=template)
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


# ---------------- Pillar 4: remediation ----------------

class PracticeRequestBody(Camel):
    student_id: str
    per_concept: int = 2
    subject: str = "Science"
    grade: str = "X"
    correct_options: dict[str, str] = Field(default_factory=dict)


class PracticeItemResponse(Camel):
    question_id: str
    concept_id: str
    stem: str
    marks: int
    difficulty: str


class PracticeSetResponse(Camel):
    id: str
    student_id: str
    concept_ids: list[str]
    items: list[PracticeItemResponse]
    total_marks: int
    warnings: list[str] = Field(default_factory=list)


@router.post("/practice/generate", response_model=PracticeSetResponse)
def generate_practice(req: PracticeRequestBody) -> PracticeSetResponse:
    _, knowledge, _ = _require()
    weak = knowledge.weak_concepts(req.student_id, limit=3)
    if not weak:
        raise HTTPException(400, "No weak concepts for this student — nothing to remediate yet.")
    pool = _pool_questions(req.subject, req.grade)
    pset = remediation_mod.build_practice_set(
        req.student_id, weak, pool, per_concept=req.per_concept,
        answer_key=req.correct_options)
    _require_practice().save(pset)
    return PracticeSetResponse(
        id=pset.id, student_id=pset.student_id, concept_ids=pset.concept_ids,
        items=[PracticeItemResponse(
            question_id=i.question.id, concept_id=i.concept_id, stem=i.question.stem,
            marks=i.question.marks, difficulty=i.question.difficulty) for i in pset.items],
        total_marks=pset.total_marks, warnings=pset.warnings,
    )


class PracticeSubmitBody(Camel):
    set_id: str
    answers: dict[str, str]


class PracticeOutcomeResponse(Camel):
    concept_id: str
    concept_name: str
    mastery_before: float
    mastery_after: float
    delta: float
    mastered: bool


class PracticeResultResponse(Camel):
    set_id: str
    student_id: str
    score: int
    max_score: int
    percentage: float
    outcomes: list[PracticeOutcomeResponse]
    next_action: str


@router.post("/practice/submit", response_model=PracticeResultResponse)
def submit_practice(body: PracticeSubmitBody) -> PracticeResultResponse:
    _, knowledge, _ = _require()
    pset = _require_practice().get(body.set_id)
    if pset is None:
        raise HTTPException(404, "practice set not found")
    res = remediation_mod.submit_practice(pset, body.answers, knowledge)
    return PracticeResultResponse(
        set_id=res.set_id, student_id=res.student_id, score=res.score, max_score=res.max_score,
        percentage=res.percentage,
        outcomes=[PracticeOutcomeResponse(
            concept_id=o.concept_id, concept_name=o.concept_name,
            mastery_before=o.mastery_before, mastery_after=o.mastery_after,
            delta=o.delta, mastered=o.mastered) for o in res.outcomes],
        next_action=res.next_action,
    )


# ---------------- Pillars 5 & 6: teacher + principal ----------------

class ConceptInsightResponse(Camel):
    concept_id: str
    concept_name: str
    class_accuracy: float
    students_rated: int
    at_risk_students: list[str]
    recommendation: str
    estimated_minutes: int


class SharedMistakeResponse(Camel):
    marking_point: str
    concept_name: str
    students_affected: int
    percentage: float


class ClassInsightsResponse(Camel):
    assessment_id: str
    class_id: str
    students: int
    average_percentage: float
    hardest_concept: Optional[str] = None
    headline: str
    concepts: list[ConceptInsightResponse]
    shared_mistakes: list[SharedMistakeResponse]


@router.get("/insights/class/{assessment_id}", response_model=ClassInsightsResponse)
def class_report(assessment_id: str, class_id: str = "10A") -> ClassInsightsResponse:
    _require()
    per_student = _require_graded().for_assessment(assessment_id)
    if not per_student:
        raise HTTPException(404, "No evaluated answer sheets for this assessment yet.")
    ci = insights_mod.class_insights(assessment_id, class_id, per_student)
    return ClassInsightsResponse(
        assessment_id=ci.assessment_id, class_id=ci.class_id, students=ci.students,
        average_percentage=ci.average_percentage, hardest_concept=ci.hardest_concept,
        headline=ci.headline,
        concepts=[ConceptInsightResponse(
            concept_id=c.concept_id, concept_name=c.concept_name,
            class_accuracy=c.class_accuracy, students_rated=c.students_rated,
            at_risk_students=c.at_risk_students, recommendation=c.recommendation,
            estimated_minutes=c.estimated_minutes) for c in ci.concepts],
        shared_mistakes=[SharedMistakeResponse(
            marking_point=m.marking_point, concept_name=m.concept_name,
            students_affected=m.students_affected, percentage=m.percentage)
            for m in ci.shared_mistakes],
    )


class SubjectRollupResponse(Camel):
    subject: str
    grade: int
    average_mastery: float
    students: int
    weakest_concept: Optional[str] = None
    curriculum_coverage: float


class SchoolInsightsResponse(Camel):
    school_id: str
    students: int
    assessments: int
    average_mastery: float
    subjects: list[SubjectRollupResponse]
    interventions: list[str]


@router.get("/insights/school/{school_id}", response_model=SchoolInsightsResponse)
def school_report(school_id: str) -> SchoolInsightsResponse:
    cfg, knowledge, _ = _require()
    graded = _require_graded()
    student_ids = sorted(graded.all_student_ids())
    if not student_ids:
        student_ids = [p.stem for p in (cfg.data_root / "knowledge").glob("*.json")]
    si = insights_mod.school_insights(school_id, knowledge, student_ids,
                                      assessments=graded.assessment_count())
    return SchoolInsightsResponse(
        school_id=si.school_id, students=si.students, assessments=si.assessments,
        average_mastery=si.average_mastery,
        subjects=[SubjectRollupResponse(
            subject=s.subject, grade=s.grade, average_mastery=s.average_mastery,
            students=s.students, weakest_concept=s.weakest_concept,
            curriculum_coverage=s.curriculum_coverage) for s in si.subjects],
        interventions=si.interventions,
    )
