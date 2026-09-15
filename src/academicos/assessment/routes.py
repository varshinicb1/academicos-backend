"""FastAPI routes for Assessment Designer, matching the Flutter ApiClient
contract exactly (paths, camelCase JSON). Mounted at /api/v1 in api/main.py.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from typing import get_args

from ..config import Config
from ..integrations.composio_calendar import sync_to_google_calendar
from . import pdf as pdf_export
from . import selection
from .audit_log import get_audit_log
from .authz import require_own_school, require_school_owns_assessment, require_school_owns_paper
from .auth_routes import get_current_user, require_principal
from .users import User, get_user_store
from .mapping import grade_to_int, to_question_schema
from .paper import generate_paper as build_generated_paper, generate_paper_sets
from .pool import get_pool
from .schemas import (
    Assessment,
    AssessmentStatus,
    BloomDistribution,
    Blueprint,
    BlueprintRequest,
    ChapterWeights,
    CompetencyWeights,
    CreateAssessmentRequest,
    DifficultyDistribution,
    GenerateFromIdsRequest,
    GeneratedPaper,
    PaperGenerationRequest,
    QuestionOptimizationRequest,
    QuestionOptimizationResult,
    QuestionSchema,
    QuestionSearchParams,
    QuickPaperRequest,
    SchoolTemplate,
    SectionBlueprint,
)
from .paper_store import PaperStore
from .store import AssessmentStore
from .templates import (
    TIER_BLOOM,
    TIER_DIFFICULTY,
    default_sections,
    get_sections_for_exam_type,
)

router = APIRouter(prefix="/api/v1")

_cfg: Optional[Config] = None
_store: Optional[AssessmentStore] = None
# SQLite-backed (paper_store.py), not a process dict: this app deploys on
# Render's free tier, which stops the container after ~15min idle. A dict
# here loses every previously generated paper on the next cold start --
# confirmed live as the "reopening a previously generated paper isn't wired
# up" bug a teacher hit, which was actually about persistence, not routing.
_papers: Optional[PaperStore] = None

# The assessment lifecycle statuses in which the paper is still being authored
# and question selection may be re-run. Once an assessment passes principal
# approval (schema's `principalApproved`, set via PATCH /status) it is locked:
# regenerating the paper or PUTting a changed assessment (a question swap)
# would quietly alter a finalized paper, which is exactly the hard edit-lock
# docs/compliance.md calls for. Statuses after approval (printed, conducted,
# scanning, ... archived) are at least as locked.
_EDITABLE_STATUSES = frozenset({
    "draft", "blueprintReady", "questionsSelected", "questionOptimized",
    "paperGenerated", "underReview",
})
# Every status the schema's AssessmentStatus Literal allows. PATCH /status must
# be validated against this set: accepting arbitrary strings corrupts the
# lifecycle and bricks the assessment (editable checks, evaluation, reports all
# branch on these exact values).
_KNOWN_STATUSES = frozenset(get_args(AssessmentStatus))


def _require_editable(assessment: Assessment) -> None:
    if assessment.status not in _EDITABLE_STATUSES:
        raise HTTPException(
            409,
            f"assessment {assessment.id} is {assessment.status} — the paper is locked; "
            "question selection and paper content cannot be changed after principal approval",
        )


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


def _users():
    """Same singleton auth_routes.py's own init() populates (curriculum/
    routes.py's _require_users() does the identical thing) -- reused here
    only to verify a principal-supplied teacher_id actually belongs to their
    own school, never to duplicate registration/auth logic."""
    from . import auth_routes
    if auth_routes._users is not None:
        return auth_routes._users
    if _cfg is None:
        raise HTTPException(503, "assessment module not initialized")
    return get_user_store(_cfg.data_root)


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
def get_section_templates(school_id: str, current: User = Depends(get_current_user)) -> list[SectionBlueprint]:
    require_own_school(school_id, current)
    return list(default_sections(80))


@router.get("/schools/{school_id}/paper-templates", response_model=list[SchoolTemplate])
def get_paper_templates(school_id: str, current: User = Depends(get_current_user)) -> list[SchoolTemplate]:
    """Kept for the older Dart `ApiClient` (pillar_api.dart's `/school-templates`
    is the canonical one template_maker_page.dart saves to). This used to be a
    stub that always returned a hardcoded "Default CBSE Template" regardless of
    what a school configured — every generated paper silently ignored the
    school's real branding. Delegates to the same store now so both clients
    see the same templates.
    """
    from .school_templates import TemplateStore

    require_own_school(school_id, current)
    cfg, _ = _require()
    store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
    existing = store.list_for_school(school_id)
    return existing or [store.default_for(school_id)]


# ---- Questions ----
#
# Deliberately unauthenticated, same reasoning as curriculum/routes.py's own
# public read endpoints and pillar_routes.py's /catalog and /syllabus: these
# browse the shared CBSE previous-year-paper question pool (public, official,
# not any one school's confidential exam), not a specific school's generated
# assessment. Search/optimize never touch AssessmentStore/PaperStore -- see
# ---- Papers ---- and ---- Assessments ---- below for where a real school's
# confidential content actually starts, and where auth is required.

@router.post("/questions/search", response_model=list[QuestionSchema])
def search_questions(params: QuestionSearchParams) -> list[QuestionSchema]:
    cfg, _ = _require()
    grade_roman = _int_grade_to_roman(params.grade)
    pool = get_pool(cfg, subject=params.subject, grade=grade_roman)
    candidates = pool.filter(subject=params.subject, grade=grade_roman, chapter_ids=params.chapter_ids)
    schemas = [to_question_schema(c) for c in candidates]

    if params.subtopic_ids:
        # The §21-25 bridge: "Generate Question Paper" from a set of
        # selected, approved Subtopics. Real question ids only -- a
        # subtopic with nothing tagged to it correctly excludes every
        # question rather than falling back to the whole chapter.
        from ..curriculum.store import get_curriculum_store
        eligible = set(get_curriculum_store(cfg.data_root).question_ids_for_subtopics(params.subtopic_ids))
        schemas = [q for q in schemas if q.id in eligible]
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
def generate_paper_endpoint(
    request: PaperGenerationRequest, current: User = Depends(get_current_user),
) -> GeneratedPaper:
    cfg, store = _require()
    # Found live in production data (2026-08-19): a real assessment whose
    # target chapters didn't match anything in the corpus (search/optimize
    # legitimately returned zero candidates) still ended up with
    # status="paperGenerated", a real paper id, and every section showing
    # 0 questions / 0 marks -- a signed-off blank exam paper with no error
    # anywhere in the chain. Refuse outright rather than silently
    # "succeeding" with nothing to hand a student: an empty paper is not a
    # valid generated paper, it's the search step failing quietly.
    if not request.selected_questions:
        raise HTTPException(
            400,
            "cannot generate a paper with zero selected questions — the "
            "chapters/filters matched no real questions in the corpus; "
            "widen the chapter selection or check the subject/grade",
        )
    assessment = store.get(request.assessment_id)
    if assessment is not None:
        if assessment.school_id != current.school_id:
            raise HTTPException(403, "this assessment belongs to a different school")
        _require_editable(assessment)
    title = assessment.title if assessment else "Assessment"
    subject = assessment.subject if assessment else (request.selected_questions[0].subject if request.selected_questions else "Science")
    grade = assessment.grade if assessment else (request.selected_questions[0].grade if request.selected_questions else 10)

    set_count = getattr(request, "set_count", 1) or 1
    paper = generate_paper_sets(
        paper_id=f"paper_{uuid.uuid4().hex[:12]}",
        assessment_id=request.assessment_id,
        assessment_title=title,
        subject=subject,
        grade=grade,
        blueprint=request.blueprint,
        selected_questions=request.selected_questions,
        set_count=set_count,
    )
    _require_papers().save(paper, request.template)
    if paper.sets:
        for s in paper.sets:
            _require_papers().save(s, request.template)

    if assessment:
        assessment.generated_paper_id = paper.id
        assessment.selected_question_ids = [q.id for q in request.selected_questions]
        assessment.status = "paperGenerated"
        assessment.updated_at = _now()
        store.save(assessment)
    return paper


@router.post("/papers/quick-generate", response_model=GeneratedPaper)
def quick_generate_paper(request: QuickPaperRequest, current: User = Depends(get_current_user)) -> GeneratedPaper:
    """Rapid generation of complete, sectioned papers (Examzo-style 1-click generation)."""
    cfg, store = _require()
    subject = request.subject
    grade = request.grade
    grade_roman = _int_grade_to_roman(grade)
    pool = get_pool(cfg, subject=subject, grade=grade_roman)
    total_marks = request.total_marks
    tier = request.tier or "standard"

    duration = request.duration_minutes
    if not duration:
        duration = 60 if total_marks <= 25 else (120 if total_marks <= 50 else 180)

    diff_preset = TIER_DIFFICULTY.get(tier, TIER_DIFFICULTY["standard"])
    bloom_preset = TIER_BLOOM.get(tier, TIER_BLOOM["standard"])

    if request.exam_type:
        sections = get_sections_for_exam_type(request.exam_type, total_marks)
    else:
        sections = default_sections(total_marks)

    bp = Blueprint(
        total_marks=total_marks,
        duration_minutes=duration,
        difficulty=diff_preset,
        bloom=bloom_preset,
        chapter_weights=ChapterWeights(),
        competency_weights=CompetencyWeights(),
        sections=sections,
        tier=tier,
        competency_percentage=0.50,
        exam_type=request.exam_type,
    )

    target_chapters = request.chapter_ids or []
    chapter_candidates = []
    if target_chapters:
        p_cands = pool.filter(subject=subject, grade=grade_roman, chapter_ids=target_chapters)
        chapter_candidates = [to_question_schema(c) for c in p_cands]

    all_pool_questions = [to_question_schema(c) for c in pool.questions]
    candidates = chapter_candidates if chapter_candidates else all_pool_questions
    fallback = all_pool_questions if chapter_candidates else None

    opt_result = selection.optimize(candidates, bp, fallback_candidates=fallback)
    if not opt_result.selected_questions:
        raise HTTPException(
            400,
            f"Unable to find sufficient questions for {subject} Grade {grade} in the question bank. "
            "Please broaden chapters or check subject/grade.",
        )

    asm_id = f"asm_quick_{uuid.uuid4().hex[:8]}"
    title = request.title or f"{subject} Class {grade} {request.exam_type or 'Assessment'}"
    paper_id = f"paper_{uuid.uuid4().hex[:12]}"
    paper = generate_paper_sets(
        paper_id=paper_id,
        assessment_id=asm_id,
        assessment_title=title,
        subject=subject,
        grade=grade,
        blueprint=bp,
        selected_questions=opt_result.selected_questions,
        set_count=max(1, request.set_count),
    )

    template = None
    if request.template_id:
        from .school_templates import TemplateStore
        t_store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        template = t_store.get(request.template_id)

    _require_papers().save(paper, template)
    if paper.sets:
        for s in paper.sets:
            _require_papers().save(s, template)

    assessment = Assessment(
        id=asm_id,
        school_id=current.school_id,
        teacher_id=current.id,
        title=title,
        subject=subject,
        grade=grade,
        chapter_ids=target_chapters,
        blueprint=bp,
        status="paperGenerated",
        created_at=_now(),
        updated_at=_now(),
        generated_paper_id=paper.id,
        selected_question_ids=[q.id for q in opt_result.selected_questions],
    )
    store.save(assessment)

    get_audit_log(cfg.data_root).append(
        "quick_paper_generated", assessment_id=asm_id,
        details={
            "paperId": paper.id,
            "setCount": request.set_count,
            "tier": tier,
            "userId": current.id,
            "schoolId": current.school_id,
        },
    )

    return paper


@router.post("/papers/generate-from-ids", response_model=GeneratedPaper)
def generate_from_ids(request: GenerateFromIdsRequest, current: User = Depends(get_current_user)) -> GeneratedPaper:
    """Instantly compile selected question IDs into a structured sectioned paper (Examzo-style ID compilation)."""
    cfg, store = _require()
    if not request.question_ids:
        raise HTTPException(400, "question_ids list cannot be empty")

    grade_roman = _int_grade_to_roman(request.grade)
    pool = get_pool(cfg, subject=request.subject, grade=grade_roman)
    all_pool_questions = [to_question_schema(c) for c in pool.questions]
    by_id = {q.id: q for q in all_pool_questions}
    found_questions: list[QuestionSchema] = [by_id[qid] for qid in request.question_ids if qid in by_id]

    if not found_questions:
        raise HTTPException(404, "None of the specified question_ids were found in the question bank")

    by_marks: dict[int, list[QuestionSchema]] = {}
    for q in found_questions:
        by_marks.setdefault(q.marks, []).append(q)

    sections: list[SectionBlueprint] = []
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    for idx, (marks, q_list) in enumerate(sorted(by_marks.items())):
        lbl = labels[idx] if idx < len(labels) else f"S{idx+1}"
        sections.append(SectionBlueprint(
            id=f"sec_{lbl.lower()}",
            label=lbl,
            name=f"{marks}-Mark Questions",
            marks_per_question=marks,
            question_count=len(q_list),
            total_marks=marks * len(q_list),
            has_internal_choice=False,
            internal_choice_count=0,
        ))

    total_marks = sum(q.marks for q in found_questions)
    duration = max(30, int(total_marks * 1.8))
    diff_preset = TIER_DIFFICULTY["standard"]
    bloom_preset = TIER_BLOOM["standard"]

    bp = Blueprint(
        total_marks=total_marks,
        duration_minutes=duration,
        difficulty=diff_preset,
        bloom=bloom_preset,
        chapter_weights=ChapterWeights(),
        competency_weights=CompetencyWeights(),
        sections=sections,
    )

    asm_id = request.assessment_id or f"asm_curated_{uuid.uuid4().hex[:8]}"
    title = request.title or "Curated Question Paper"
    paper_id = f"paper_{uuid.uuid4().hex[:12]}"

    paper = build_generated_paper(
        paper_id=paper_id,
        assessment_id=asm_id,
        assessment_title=title,
        subject=request.subject,
        grade=request.grade,
        blueprint=bp,
        selected_questions=found_questions,
    )

    template = None
    if request.template_id:
        from .school_templates import TemplateStore
        t_store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        template = t_store.get(request.template_id)

    _require_papers().save(paper, template)

    existing_asm = store.get(asm_id)
    if existing_asm:
        existing_asm.generated_paper_id = paper.id
        existing_asm.selected_question_ids = [q.id for q in found_questions]
        existing_asm.status = "paperGenerated"
        existing_asm.updated_at = _now()
        store.save(existing_asm)
    else:
        new_asm = Assessment(
            id=asm_id,
            school_id=current.school_id,
            teacher_id=current.id,
            title=title,
            subject=request.subject,
            grade=request.grade,
            chapter_ids=list({cid for q in found_questions for cid in q.chapter_ids}),
            blueprint=bp,
            status="paperGenerated",
            created_at=_now(),
            updated_at=_now(),
            generated_paper_id=paper.id,
            selected_question_ids=[q.id for q in found_questions],
        )
        store.save(new_asm)

    get_audit_log(cfg.data_root).append(
        "id_curated_paper_generated", assessment_id=asm_id,
        details={
            "paperId": paper.id,
            "questionCount": len(found_questions),
            "userId": current.id,
            "schoolId": current.school_id,
        },
    )
    return paper


@router.get("/papers/{paper_id}", response_model=GeneratedPaper)
def get_paper(paper_id: str, current: User = Depends(get_current_user)) -> GeneratedPaper:
    _, store = _require()
    return require_school_owns_paper(_require_papers(), store, paper_id, current)


@router.post("/papers/{paper_id}/export/{fmt}")
def export_paper(paper_id: str, fmt: str, current: User = Depends(get_current_user)) -> dict:
    cfg, store = _require()
    paper = require_school_owns_paper(_require_papers(), store, paper_id, current)
    if fmt not in ("pdf", "answer-key", "answer_key", "answerKey"):
        raise HTTPException(400, "only pdf and answer-key exports are supported currently")
    out_dir = cfg.artifacts_dir / "papers"
    template = _require_papers().get_template(paper_id)
    if template is None:
        from .school_templates import TemplateStore
        store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        template = store.default_for("school_1")
    # Every export gets a unique, audit-logged watermark ID stamped into the
    # footer -- if a printed/exported copy of an unreleased paper leaks, it's
    # traceable to exactly which export request produced it, not just "the
    # paper leaked" with no way to narrow down how. See pdf.py's
    # _page_furniture docstring and docs/compliance.md's paper-release-
    # locking checklist item.
    watermark_id = f"exp_{uuid.uuid4().hex[:10]}"
    if fmt in ("answer-key", "answer_key", "answerKey"):
        path = pdf_export.export_answer_key_pdf(paper, out_dir, template=template)
        get_audit_log(cfg.data_root).append(
            "answer_key_exported", assessment_id=paper.assessment_id,
            details={"paperId": paper_id, "format": fmt, "file": path.name, "watermarkId": watermark_id},
        )
        return {"url": f"/api/v1/papers/{paper_id}/file?format=answer-key", "watermarkId": watermark_id}
    path = pdf_export.export_pdf(paper, out_dir, template=template, watermark_id=watermark_id)
    get_audit_log(cfg.data_root).append(
        "paper_exported", assessment_id=paper.assessment_id,
        details={"paperId": paper_id, "format": fmt, "watermarkId": watermark_id},
    )
    return {"url": f"/api/v1/papers/{paper_id}/file", "watermarkId": watermark_id}


@router.get("/papers/{paper_id}/file")
def get_paper_file(paper_id: str, format: str = "pdf", current: User = Depends(get_current_user)):
    cfg, store = _require()
    require_school_owns_paper(_require_papers(), store, paper_id, current)
    suffix = "_answer_key.pdf" if format in ("answer-key", "answer_key") else ".pdf"
    path = cfg.artifacts_dir / "papers" / f"{paper_id}{suffix}"
    if not path.exists():
        raise HTTPException(404, "pdf not generated yet")
    return FileResponse(str(path), media_type="application/pdf", filename=f"{paper_id}{suffix}")


# ---- Assessments ----

@router.post("/assessments", response_model=Assessment)
def create_assessment(
    request: CreateAssessmentRequest, current: User = Depends(get_current_user),
) -> Assessment:
    _, store = _require()
    # Real bug fixed 2026-09-15: school_id/teacher_id used to come straight
    # from the request body with zero verification -- full impersonation, not
    # just a read leak (anyone could POST an assessment claiming to belong to
    # any school). school_id is always the caller's own now, same principle
    # curriculum/routes.py's seed_cbse10 already applies. teacher_id: a
    # principal may legitimately create on behalf of a teacher in their own
    # school (the request body's value, verified against that school); anyone
    # else can only create as themselves.
    school_id = current.school_id
    teacher_id = current.id
    if current.role == "principal" and request.teacher_id:
        teacher = _users().get(request.teacher_id)
        if teacher is None or teacher.school_id != school_id:
            raise HTTPException(403, "teacher_id must belong to your own school")
        teacher_id = request.teacher_id
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
        school_id=school_id,
        teacher_id=teacher_id,
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
def get_assessment(assessment_id: str, current: User = Depends(get_current_user)) -> Assessment:
    _, store = _require()
    return require_school_owns_assessment(store, assessment_id, current)


@router.get("/assessments", response_model=list[Assessment])
def list_assessments(
    teacher_id: Optional[str] = None, school_id: Optional[str] = None,
    current: User = Depends(get_current_user),
) -> list[Assessment]:
    # Real bug fixed 2026-09-15: teacher_id/school_id used to be trusted
    # straight from the query string with no check that the caller actually
    # was that teacher or belonged to that school -- any authenticated caller
    # could list any other school's or teacher's assessments by guessing an
    # id. school_id is always the caller's own now; teacher_id is only
    # honored if it's the caller's own id or the caller is a principal
    # (who may legitimately look up any teacher in their own school).
    _, store = _require()
    if teacher_id:
        if teacher_id != current.id and current.role != "principal":
            raise HTTPException(403, "cannot list another teacher's assessments")
        return [a for a in store.list_by_teacher(teacher_id) if a.school_id == current.school_id]
    return store.list_by_school(current.school_id)


@router.put("/assessments/{assessment_id}", response_model=Assessment)
def update_assessment(
    assessment_id: str, assessment: Assessment, current: User = Depends(get_current_user),
) -> Assessment:
    _, store = _require()
    existing = store.get(assessment_id)
    if existing is not None:
        if existing.school_id != current.school_id:
            raise HTTPException(403, "this assessment belongs to a different school")
        _require_editable(existing)
    elif assessment.school_id != current.school_id:
        # No pre-existing row to check ownership against (first PUT acting as
        # create) -- the body must still claim the caller's own school, same
        # rule as POST /assessments, not an arbitrary one.
        raise HTTPException(403, "cannot create an assessment for a different school")
    assessment.id = assessment_id
    assessment.school_id = current.school_id
    assessment.updated_at = _now()
    sync_to_google_calendar(assessment)  # best-effort; never blocks the save below
    store.save(assessment)
    return assessment


@router.delete("/assessments/{assessment_id}")
def delete_assessment(assessment_id: str, current: User = Depends(get_current_user)) -> dict:
    _, store = _require()
    require_school_owns_assessment(store, assessment_id, current)
    store.delete(assessment_id)
    return {"ok": True}


@router.patch("/assessments/{assessment_id}/status", response_model=Assessment)
def update_status(assessment_id: str, body: dict, current: User = Depends(get_current_user)) -> Assessment:
    _, store = _require()
    a = require_school_owns_assessment(store, assessment_id, current)
    new_status = body.get("status", a.status)
    if new_status not in _KNOWN_STATUSES:
        raise HTTPException(
            422,
            f"unknown status {new_status!r}; expected one of {sorted(_KNOWN_STATUSES)}",
        )
    if a.status not in _EDITABLE_STATUSES and new_status in _EDITABLE_STATUSES:
        raise HTTPException(
            409,
            f"assessment {assessment_id} is {a.status} — its status cannot move back to "
            "an editable state after principal approval",
        )
    a.status = new_status
    a.updated_at = _now()
    store.save(a)
    return a


# docs/compliance.md's recommended minimal principal-approval workflow: the
# *lock* (principalApproved being terminal, enforced by _require_editable
# above) already existed with no real way to reach it, since nothing could
# authenticate as a principal. This is that missing action -- gated on a
# real principal identity (auth_routes.require_principal, itself gated on
# users.py's "first registrant per school" bootstrap rule) rather than a
# bare status PATCH anyone could call.
@router.patch("/assessments/{assessment_id}/approve", response_model=Assessment)
def approve_assessment(assessment_id: str, principal: User = Depends(require_principal)) -> Assessment:
    cfg, store = _require()
    a = store.get(assessment_id)
    if a is None:
        raise HTTPException(404, "assessment not found")
    if a.school_id != principal.school_id:
        raise HTTPException(403, "this assessment belongs to a different school")
    a.status = "principalApproved"
    a.updated_at = _now()
    store.save(a)
    get_audit_log(cfg.data_root).append(
        "assessment_approved", assessment_id=assessment_id, actor=principal.id,
        details={"principalName": principal.name, "principalEmail": principal.email},
    )
    return a
