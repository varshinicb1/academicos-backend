"""FastAPI routes for Assessment Designer, matching the Flutter ApiClient
contract exactly (paths, camelCase JSON). Mounted at /api/v1 in api/main.py.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import Config
from ..integrations.composio_calendar import sync_to_google_calendar
from . import pdf as pdf_export
from . import selection
from .mapping import grade_to_int, to_question_schema
from .paper import generate_paper as build_generated_paper
from .pool import get_pool
from .schemas import (
    Assessment,
    Blueprint,
    BlueprintRequest,
    CreateAssessmentRequest,
    GeneratedPaper,
    PaperGenerationRequest,
    QuestionOptimizationRequest,
    QuestionOptimizationResult,
    QuestionSchema,
    QuestionSearchParams,
    SchoolTemplate,
    SectionBlueprint,
)
from .paper_store import PaperStore
from .store import AssessmentStore
from .templates import default_sections

router = APIRouter(prefix="/api/v1")

_cfg: Optional[Config] = None
_store: Optional[AssessmentStore] = None
# SQLite-backed (paper_store.py), not a process dict: this app deploys on
# Render's free tier, which stops the container after ~15min idle. A dict
# here loses every previously generated paper on the next cold start --
# confirmed live as the "reopening a previously generated paper isn't wired
# up" bug a teacher hit, which was actually about persistence, not routing.
_papers: Optional[PaperStore] = None


def init(config: Config) -> None:
    global _cfg, _store, _papers
    _cfg = config
    _store = AssessmentStore(config.data_root / "assessments" / "assessments.sqlite")
    _papers = PaperStore(config.data_root / "assessments" / "papers.sqlite")


def _require() -> tuple[Config, AssessmentStore]:
    if _cfg is None or _store is None:
        raise HTTPException(503, "assessment module not initialized")
    return _cfg, _store


def _require_papers() -> PaperStore:
    if _papers is None:
        raise HTTPException(503, "assessment module not initialized")
    return _papers


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---- Blueprint ----

@router.post("/blueprints/generate", response_model=Blueprint)
def generate_blueprint(request: BlueprintRequest) -> Blueprint:
    sections = request.sections or default_sections(request.total_marks)
    return Blueprint(
        total_marks=request.total_marks,
        duration_minutes=request.duration_minutes,
        difficulty=request.difficulty,
        bloom=request.bloom,
        chapter_weights=request.chapter_weights,
        competency_weights=request.competency_weights,
        sections=sections,
        metadata=request.school_template or {},
    )


@router.get("/schools/{school_id}/templates", response_model=list[SectionBlueprint])
def get_section_templates(school_id: str) -> list[SectionBlueprint]:
    return list(default_sections(80))


@router.get("/schools/{school_id}/paper-templates", response_model=list[SchoolTemplate])
def get_paper_templates(school_id: str) -> list[SchoolTemplate]:
    """Kept for the older Dart `ApiClient` (pillar_api.dart's `/school-templates`
    is the canonical one template_maker_page.dart saves to). This used to be a
    stub that always returned a hardcoded "Default CBSE Template" regardless of
    what a school configured — every generated paper silently ignored the
    school's real branding. Delegates to the same store now so both clients
    see the same templates.
    """
    from .school_templates import TemplateStore

    cfg, _ = _require()
    store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
    existing = store.list_for_school(school_id)
    return existing or [store.default_for(school_id)]


# ---- Questions ----

@router.post("/questions/search", response_model=list[QuestionSchema])
def search_questions(params: QuestionSearchParams) -> list[QuestionSchema]:
    cfg, _ = _require()
    grade_roman = _int_grade_to_roman(params.grade)
    pool = get_pool(cfg, subject=params.subject, grade=grade_roman)
    candidates = pool.filter(subject=params.subject, grade=grade_roman, chapter_ids=params.chapter_ids)
    schemas = [to_question_schema(c) for c in candidates]

    if params.bloom_levels:
        schemas = [q for q in schemas if q.bloom_level in params.bloom_levels]
    if params.difficulties:
        schemas = [q for q in schemas if q.difficulty in params.difficulties]
    if params.types:
        schemas = [q for q in schemas if q.type in params.types]
    if params.min_marks is not None:
        schemas = [q for q in schemas if q.marks >= params.min_marks]
    if params.max_marks is not None:
        schemas = [q for q in schemas if q.marks <= params.max_marks]
    if params.keyword:
        kw = params.keyword.lower()
        schemas = [q for q in schemas if kw in q.stem.lower()]
    if params.offset:
        schemas = schemas[params.offset:]
    if params.limit:
        schemas = schemas[: params.limit]
    return schemas


def _int_grade_to_roman(grade: int) -> str:
    romans = {8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}
    return romans.get(grade, "X")


@router.post("/questions/optimize", response_model=QuestionOptimizationResult)
def optimize_questions(request: QuestionOptimizationRequest) -> QuestionOptimizationResult:
    cfg, _ = _require()
    fallback: list[QuestionSchema] = []
    if request.candidates:
        subject = request.candidates[0].subject
        grade_roman = _int_grade_to_roman(request.candidates[0].grade)
        pool = get_pool(cfg, subject=subject, grade=grade_roman)
        fallback = [to_question_schema(c) for c in pool.filter(subject=subject, grade=grade_roman)]
    return selection.optimize(request.candidates, request.blueprint, fallback_candidates=fallback)


# ---- Papers ----

@router.post("/papers/generate", response_model=GeneratedPaper)
def generate_paper_endpoint(request: PaperGenerationRequest) -> GeneratedPaper:
    cfg, store = _require()
    assessment = store.get(request.assessment_id)
    title = assessment.title if assessment else "Assessment"
    subject = assessment.subject if assessment else (request.selected_questions[0].subject if request.selected_questions else "Science")
    grade = assessment.grade if assessment else (request.selected_questions[0].grade if request.selected_questions else 10)

    paper = build_generated_paper(
        paper_id=f"paper_{uuid.uuid4().hex[:12]}",
        assessment_id=request.assessment_id,
        assessment_title=title,
        subject=subject,
        grade=grade,
        blueprint=request.blueprint,
        selected_questions=request.selected_questions,
    )
    _require_papers().save(paper, request.template)

    if assessment:
        assessment.generated_paper_id = paper.id
        assessment.selected_question_ids = [q.id for q in request.selected_questions]
        assessment.status = "paperGenerated"
        assessment.updated_at = _now()
        store.save(assessment)
    return paper


@router.get("/papers/{paper_id}", response_model=GeneratedPaper)
def get_paper(paper_id: str) -> GeneratedPaper:
    paper = _require_papers().get(paper_id)
    if paper is None:
        raise HTTPException(404, "paper not found")
    return paper


@router.post("/papers/{paper_id}/export/{fmt}")
def export_paper(paper_id: str, fmt: str) -> dict:
    cfg, _ = _require()
    paper = _require_papers().get(paper_id)
    if paper is None:
        raise HTTPException(404, "paper not found")
    if fmt != "pdf":
        raise HTTPException(400, "only pdf export is supported currently")
    out_dir = cfg.artifacts_dir / "papers"
    template = _require_papers().get_template(paper_id)
    if template is None:
        from .school_templates import TemplateStore
        store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        template = store.default_for("school_1")
    path = pdf_export.export_pdf(paper, out_dir, template=template)
    return {"url": f"/api/v1/papers/{paper_id}/file"}


@router.get("/papers/{paper_id}/file")
def get_paper_file(paper_id: str):
    cfg, _ = _require()
    path = cfg.artifacts_dir / "papers" / f"{paper_id}.pdf"
    if not path.exists():
        raise HTTPException(404, "pdf not generated yet")
    return FileResponse(str(path), media_type="application/pdf", filename=f"{paper_id}.pdf")


# ---- Assessments ----

@router.post("/assessments", response_model=Assessment)
def create_assessment(request: CreateAssessmentRequest) -> Assessment:
    _, store = _require()
    sections = request.blueprint.sections or default_sections(request.blueprint.total_marks)
    blueprint = Blueprint(
        total_marks=request.blueprint.total_marks,
        duration_minutes=request.blueprint.duration_minutes,
        difficulty=request.blueprint.difficulty,
        bloom=request.blueprint.bloom,
        chapter_weights=request.blueprint.chapter_weights,
        competency_weights=request.blueprint.competency_weights,
        sections=sections,
        metadata=request.blueprint.school_template or {},
    )
    now = _now()
    assessment = Assessment(
        id=f"assess_{uuid.uuid4().hex[:12]}",
        school_id=request.school_id,
        teacher_id=request.teacher_id,
        title=request.title,
        subject=request.subject,
        grade=request.grade,
        chapter_ids=request.chapter_ids,
        blueprint=blueprint,
        status="blueprintReady",
        created_at=now,
        updated_at=now,
        template_id=request.template_id,
    )
    store.save(assessment)
    return assessment


@router.get("/assessments/{assessment_id}", response_model=Assessment)
def get_assessment(assessment_id: str) -> Assessment:
    _, store = _require()
    a = store.get(assessment_id)
    if a is None:
        raise HTTPException(404, "assessment not found")
    return a


@router.get("/assessments", response_model=list[Assessment])
def list_assessments(teacher_id: Optional[str] = None, school_id: Optional[str] = None) -> list[Assessment]:
    _, store = _require()
    if teacher_id:
        return store.list_by_teacher(teacher_id)
    if school_id:
        return store.list_by_school(school_id)
    return []


@router.put("/assessments/{assessment_id}", response_model=Assessment)
def update_assessment(assessment_id: str, assessment: Assessment) -> Assessment:
    _, store = _require()
    assessment.id = assessment_id
    assessment.updated_at = _now()
    sync_to_google_calendar(assessment)  # best-effort; never blocks the save below
    store.save(assessment)
    return assessment


@router.delete("/assessments/{assessment_id}")
def delete_assessment(assessment_id: str) -> dict:
    _, store = _require()
    store.delete(assessment_id)
    return {"ok": True}


@router.patch("/assessments/{assessment_id}/status", response_model=Assessment)
def update_status(assessment_id: str, body: dict) -> Assessment:
    _, store = _require()
    a = store.get(assessment_id)
    if a is None:
        raise HTTPException(404, "assessment not found")
    a.status = body.get("status", a.status)
    a.updated_at = _now()
    store.save(a)
    return a
