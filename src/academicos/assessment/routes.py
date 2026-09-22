"""FastAPI routes for Assessment Designer, matching the Flutter ApiClient
contract exactly (paths, camelCase JSON). Mounted at /api/v1 in api/main.py.
"""
from __future__ import annotations

import json
import logging
import time
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
from . import grades, paper_timing, selection
from .audit_log import get_audit_log
from .authz import (
    require_own_school, require_own_subtopics, require_school_owns_assessment,
    require_school_owns_paper,
)
from .auth_routes import get_current_user, require_principal, require_staff
from .users import User, get_user_store
from .mapping import grade_to_int, to_question_schema
from .paper import generate_paper as build_generated_paper, generate_paper_sets
from .pool import get_pool, near_duplicate
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

log = logging.getLogger(__name__)

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
def generate_blueprint(request: BlueprintRequest,
                       current: User = Depends(require_staff)) -> Blueprint:
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
def get_section_templates(school_id: str, current: User = Depends(require_staff)) -> list[SectionBlueprint]:
    require_own_school(school_id, current)
    return list(default_sections(80))


@router.get("/schools/{school_id}/paper-templates", response_model=list[SchoolTemplate])
def get_paper_templates(school_id: str, current: User = Depends(require_staff)) -> list[SchoolTemplate]:
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
# Signed-in, but not school-scoped as a whole: these browse the shared CBSE
# previous-year-paper question pool (public, official, not any one school's
# confidential exam), not a specific school's generated assessment.
# Search/optimize never touch AssessmentStore/PaperStore -- see ---- Papers
# ---- and ---- Assessments ---- below for where a real school's confidential
# content starts. The one school-owned input is search's subtopic_ids: those
# are a school's own approved curriculum rows, and the question ids tagged to
# them are that school's tagging, so they get the same ownership check as
# curriculum/routes.py's questions/by-subtopics.

@router.post("/questions/search", response_model=list[QuestionSchema])
def search_questions(params: QuestionSearchParams,
                     current: User = Depends(require_staff)) -> list[QuestionSchema]:
    cfg, _ = _require()
    curriculum = None
    if params.subtopic_ids:
        # Checked before the pool is built, so a refused request costs one
        # query. Until 2026-09-21 this path resolved any school's subtopic:
        # school_2 holding a school_1 subtopic id got school_1's tagged
        # question ids back. by-subtopics had the check; this route -- the
        # one assessment_create_page.dart actually calls -- did not.
        from ..curriculum.store import get_curriculum_store
        curriculum = get_curriculum_store(cfg.data_root)
        require_own_subtopics(curriculum, params.subtopic_ids, current)
    grade_roman = _int_grade_to_roman(params.grade)
    pool = get_pool(cfg, subject=params.subject, grade=grade_roman)
    candidates = pool.filter(subject=params.subject, grade=grade_roman, chapter_ids=params.chapter_ids)
    schemas = [to_question_schema(c) for c in candidates]

    if curriculum is not None:
        # The §21-25 bridge: "Generate Question Paper" from a set of
        # selected, approved Subtopics. Real question ids only -- a
        # subtopic with nothing tagged to it correctly excludes every
        # question rather than falling back to the whole chapter.
        eligible = set(curriculum.question_ids_for_subtopics(params.subtopic_ids))
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
    """Pool key for a request's grade. An unreadable grade is a 422, never class 10.

    This used to be `{8: "VIII", ..., 12: "XII"}.get(grade, "X")`, which turned a
    class 6 or 7 request into class 10 without saying so. See grades.py.
    """
    try:
        return grades.to_roman(grade)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/questions/optimize", response_model=QuestionOptimizationResult)
def optimize_questions(request: QuestionOptimizationRequest,
                       current: User = Depends(require_staff)) -> QuestionOptimizationResult:
    cfg, _ = _require()
    fallback: list[QuestionSchema] = []
    if request.candidates:
        subject = request.candidates[0].subject
        grade_roman = _int_grade_to_roman(request.candidates[0].grade)
        pool = get_pool(cfg, subject=subject, grade=grade_roman)
        fallback = [to_question_schema(c) for c in pool.filter(subject=subject, grade=grade_roman)]
    return selection.optimize(request.candidates, request.blueprint, fallback_candidates=fallback)


# ---- Papers ----

def _similar_question_warnings(selected: list[QuestionSchema], *,
                               reject_similar: bool) -> tuple[list[str], list[list[str]]]:
    """One warning per pair of questions this paper would print that restate
    each other, and the same pairs as [earlier id, later id] (the client's
    "swap one" needs the ids, not a sentence to parse them back out of); 422
    for the same question id twice, and for a similar pair when the client
    set `rejectSimilar`.

    POST /papers/generate and /papers/generate-from-ids print what the
    teacher chose; they do not run selection.optimize, so the near-duplicate
    guard there (and on the template and swap paths) never sees a hand-built
    selection. OR partners are printed too, so they count. Warned, not
    refused (Task 906): Task 905 refused, but `pool.near_duplicate` is a
    measure and a teacher who means to set two alike questions must not be
    blocked -- the builder shows the warning with "keep both" / "swap one".
    Never silently dropped: a paper one question shorter than they built is a
    wrong answer they would not notice. The same id twice is no judgement
    call, so it stays refused."""
    printed: list[tuple[str, str]] = []
    for q in selected:
        printed.append((q.id, q.stem))
        meta = q.metadata or {}
        partner = meta.get("internal_choice_question") or {}
        pid = meta.get("internal_choice_id") or partner.get("id")
        pstem = meta.get("internal_choice_stem") or partner.get("stem")
        if pid and pstem:
            printed.append((pid, pstem))
    warnings: list[str] = []
    pairs: list[list[str]] = []
    for i, (aid, astem) in enumerate(printed):
        for bid, bstem in printed[:i]:
            if aid == bid:
                raise HTTPException(
                    422, f"question {aid} is selected twice; a paper prints each "
                         f"question once -- remove one and generate again")
            if not near_duplicate(astem, bstem):
                continue
            if reject_similar:
                raise HTTPException(
                    422,
                    f"questions {bid} and {aid} are the same question; a paper "
                    f"may carry only one of them -- remove one and generate again",
                )
            warnings.append(
                f"questions {bid} and {aid} look like the same question -- "
                f"keep both, or swap one")
            pairs.append([bid, aid])
    return warnings, pairs


def _borrowing_warnings(opt_result) -> list[str]:
    """The optimizer's gap lines that say a section was filled from outside
    the chapters the request chose. The other gap lines are about a section
    printing short, which `_shortfall_warnings` says from the paper itself."""
    return [g for g in opt_result.gaps if "outside the selected chapters" in g]


def _shortfall_warnings(paper: GeneratedPaper, blueprint: Blueprint) -> list[str]:
    """What a paper holding fewer marks than its blueprint asked for says
    about it: the total first, then each section that prints short, in
    selection.optimize's gap wording.

    The live release returned these papers with `warnings` empty (re-audit,
    2026-09-22): English 10 asked for 80 marks and held 20, Hindi 10 held 2,
    Computer Applications 27, Biology 12 48. A teacher looking at a paper
    with a full-looking header would not notice until the exam. Empty when
    the paper is full."""
    held = paper.metadata.total_marks
    if held >= blueprint.total_marks:
        return []
    out = [f"This paper holds {held} of the {blueprint.total_marks} marks asked for; "
           f"the sections below print short. Its header prints Maximum Marks: {held}."]
    printed = {s.section_id: len(s.questions) for s in paper.sections}
    for section in blueprint.sections or []:
        got = printed.get(section.id, 0)
        if got < section.question_count:
            out.append(
                f"Section {section.label} ({section.name}): only {got} of "
                f"{section.question_count} questions; the section prints short.")
    return out


def _report_competency(paper: GeneratedPaper, target: float | None) -> list[str]:
    """The CBQ share of the printed questions and whether it meets `target`
    (CBSE: at least 50%). Missed in 17/25 subject/grade pairs at the
    2026-09-21 audit, with nothing on the paper to say so.

    Set B/C print other questions than set A, so each set gets its own share:
    reporting set A's for all three said 0.45 where B and C held 0.25 and 0.15.
    Returns a warning per set that misses the target.
    """
    target = 0.50 if target is None else target
    warnings: list[str] = []
    for p in [paper, *paper.sets]:
        flags = [q.is_competency for s in p.sections for q in s.questions]
        p.competency_share = round(sum(flags) / len(flags), 3) if flags else 0.0
        p.competency_target_met = p.competency_share >= target
        if p is not paper and not p.competency_target_met:
            warnings.append(
                f"Set {p.set_label} has {round(p.competency_share * 100, 1)}% "
                f"competency-based questions (CBSE target: {int(target * 100)}%).")
    return warnings


def _overlap_warnings(paper: GeneratedPaper) -> list[str]:
    return [f"Set {label} repeats {n} question(s) from an earlier set: the question bank "
            f"has no unused question of the same marks and type left in the chapters "
            f"this paper draws from."
            for label, n in paper.set_overlap.items() if n]


@router.post("/papers/generate", response_model=GeneratedPaper)
def generate_paper_endpoint(
    request: PaperGenerationRequest, current: User = Depends(require_staff),
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
    warnings, pairs = _similar_question_warnings(request.selected_questions,
                                                 reject_similar=request.reject_similar)
    assessment = store.get(request.assessment_id)
    if assessment is not None:
        if assessment.school_id != current.school_id:
            raise HTTPException(403, "this assessment belongs to a different school")
        _require_editable(assessment)
    title = assessment.title if assessment else "Assessment"
    subject = assessment.subject if assessment else (request.selected_questions[0].subject if request.selected_questions else "Science")
    grade = assessment.grade if assessment else request.selected_questions[0].grade

    set_count = getattr(request, "set_count", 1) or 1
    alternatives = None
    if set_count > 1:
        # Sets B, C... draw other questions of the same marks and type.
        # Limited to the assessment's chapters: a unit test on chapters 1-3
        # must not print, in set C, a chapter nobody has taught. Running out
        # there shows up as setOverlap instead.
        grade_roman = _int_grade_to_roman(grade)
        chapters = assessment.chapter_ids if assessment else []
        alternatives = [to_question_schema(c) for c in
                        get_pool(cfg, subject=subject, grade=grade_roman).filter(
                            chapter_ids=chapters or None)]
    paper = generate_paper_sets(
        paper_id=f"paper_{uuid.uuid4().hex[:12]}",
        assessment_id=request.assessment_id,
        assessment_title=title,
        subject=subject,
        grade=grade,
        blueprint=request.blueprint,
        selected_questions=request.selected_questions,
        set_count=set_count,
        alternatives_pool=alternatives,
    )
    paper.warnings = _overlap_warnings(paper)
    paper.warnings += _report_competency(paper, request.blueprint.competency_percentage)
    _require_papers().save(paper, request.template, school_id=current.school_id)
    if paper.sets:
        for s in paper.sets:
            _require_papers().save(s, request.template, school_id=current.school_id)

    if assessment:
        assessment.generated_paper_id = paper.id
        assessment.selected_question_ids = [q.id for q in request.selected_questions]
        assessment.status = "paperGenerated"
        assessment.updated_at = _now()
        store.save(assessment)
    # Added after saving: these are about this request, not the paper. The
    # overlap/competency warnings above are about the paper and stay stored.
    paper.warnings = [*paper.warnings,
                      *_shortfall_warnings(paper, request.blueprint), *warnings]
    paper.similar_pairs = pairs
    return paper


@router.post("/papers/quick-generate", response_model=GeneratedPaper)
def quick_generate_paper(request: QuickPaperRequest, current: User = Depends(require_staff)) -> GeneratedPaper:
    """Rapid generation of complete, sectioned papers (Examzo-style 1-click generation)."""
    # Monotonic, so a clock adjustment during generation cannot change a
    # reported duration. See assessment/paper_timing.py for what this is for.
    _started = time.perf_counter()
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
        # The chosen chapters only. Given the whole subject, preferring the
        # replaced question's chapter was not enough: on Science 10, 40 marks,
        # three chapters, set B printed 4 and set C 10 questions from other
        # chapters with setOverlap 0. Running out now repeats a set A question
        # and setOverlap counts it.
        alternatives_pool=chapter_candidates or all_pool_questions,
    )
    # The optimizer's warnings (the CBQ one, "tiers unavailable") were dropped
    # here, and so were its `gaps`: a paper that filled a short section from
    # OUTSIDE the chapters the teacher chose said nothing about it (3 of 20
    # questions on Science 10, 40 marks, three chapters -- merge of
    # 2026-09-23). A unit test carrying untaught chapters is exactly the wrong
    # answer a teacher would not catch until the exam.
    paper.warnings = [*opt_result.warnings, *_borrowing_warnings(opt_result),
                      *_overlap_warnings(paper),
                      *_report_competency(paper, bp.competency_percentage)]
    paper.tiers_available = opt_result.optimization_metrics["tierSignals"]["available"]
    if paper.tiers_available and not selection.tier_changed_selection(
            candidates, bp, fallback, opt_result.selected_questions):
        paper.tiers_available = False
        paper.warnings.append(
            f"Tiers unavailable for this subject: the {tier} tier chose the same questions "
            f"as the standard tier -- too few questions of the marks this paper asks for "
            f"differ in difficulty or Bloom level.")

    template = None
    if request.template_id:
        from .school_templates import TemplateStore
        t_store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        # get_branding returns (template, sections); PaperStore.save wants the
        # template alone -- passing the tuple raised on every templateId.
        branding = t_store.get_branding(request.template_id, current.school_id)
        template = branding[0] if branding else None

    _require_papers().save(paper, template, school_id=current.school_id)
    if paper.sets:
        for s in paper.sets:
            _require_papers().save(s, template, school_id=current.school_id)

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
            # MEASURED: how long the machine took. Decision 11 makes this the
            # renewal criterion, and it was not recorded anywhere before.
            "generationSeconds": round(time.perf_counter() - _started, 3),
            "questionCount": len(paper.questions) if hasattr(paper, "questions") else 0,
        },
    )
    # After saving, like the other generate routes: about this request. Kept
    # alongside the optimizer/overlap/competency warnings set before the save.
    paper.warnings = [*paper.warnings, *_shortfall_warnings(paper, bp)]
    return paper


@router.post("/papers/generate-from-ids", response_model=GeneratedPaper)
def generate_from_ids(request: GenerateFromIdsRequest, current: User = Depends(require_staff)) -> GeneratedPaper:
    _started_ids = time.perf_counter()
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
    # The Flutter client's curated path builds the paper straight from these
    # ids and never runs optimize, so the near-duplicate guard has to run here
    # too -- warned (or refused on rejectSimilar), exactly like /papers/generate.
    warnings, pairs = _similar_question_warnings(found_questions,
                                                 reject_similar=request.reject_similar)

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
    _report_competency(paper, bp.competency_percentage)

    template = None
    if request.template_id:
        from .school_templates import TemplateStore
        t_store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        # get_branding returns (template, sections); PaperStore.save wants the
        # template alone -- passing the tuple raised on every templateId.
        branding = t_store.get_branding(request.template_id, current.school_id)
        template = branding[0] if branding else None

    _require_papers().save(paper, template, school_id=current.school_id)

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
            # MEASURED, same as quick generation. This route is the curated
            # path, so a school using only it would otherwise report zero
            # papers and conclude nothing was saved.
            "generationSeconds": round(time.perf_counter() - _started_ids, 3),
        },
    )
    # After saving: about this request, not the paper.
    paper.warnings, paper.similar_pairs = warnings, pairs
    return paper


@router.get("/paper-timing")
def get_paper_timing(current: User = Depends(require_principal)) -> dict:
    """Time saved per paper — the renewal criterion, made answerable.

    Decision 11: *"Renewal is measured as time saved per paper."* Before this
    route the number existed nowhere, so the criterion could not be checked.

    Scoped to the caller's school. The response separates what was MEASURED
    (generation time, from a monotonic clock) from what was DECLARED (the manual
    baseline), and says so in the payload rather than only in the docs --
    because a saving built on a declared baseline will be quoted at somebody,
    and the word "estimated" has to travel with it.
    """
    cfg, _store = _require()
    return paper_timing.report(cfg.data_root, school_id=current.school_id).as_dict()


@router.get("/papers/{paper_id}", response_model=GeneratedPaper)
def get_paper(paper_id: str, current: User = Depends(require_staff)) -> GeneratedPaper:
    _, store = _require()
    return require_school_owns_paper(_require_papers(), store, paper_id, current)


@router.post("/papers/{paper_id}/export/{fmt}")
def export_paper(paper_id: str, fmt: str, current: User = Depends(require_staff)) -> dict:
    cfg, store = _require()
    paper = require_school_owns_paper(_require_papers(), store, paper_id, current)
    if fmt not in ("pdf", "answer-key", "answer_key", "answerKey"):
        raise HTTPException(400, "only pdf and answer-key exports are supported currently")
    out_dir = cfg.artifacts_dir / "papers"
    template = _require_papers().get_template(paper_id)
    if template is None:
        # The paper's own school's default branding, found through its
        # assessment -- never another school's. This used to read
        # default_for("school_1"), so school_2's PDF printed school_1's name,
        # logo and tagline (re-audit, 2026-09-22). A school with no branding
        # configured gets default_for's placeholder (id "default"), which is
        # not branding: it gets the neutral header instead.
        from .school_templates import TemplateStore
        owner = store.get(paper.assessment_id)
        t_store = TemplateStore(cfg.data_root / "templates" / "templates.sqlite")
        configured = t_store.default_for(owner.school_id) if owner is not None else None
        template = configured if configured is not None and configured.id != "default" else None
    # Every export gets a unique, audit-logged watermark ID stamped into the
    # footer -- if a printed/exported copy of an unreleased paper leaks, it's
    # traceable to exactly which export request produced it, not just "the
    # paper leaked" with no way to narrow down how. See pdf.py's
    # _page_furniture docstring and docs/compliance.md's paper-release-
    # locking checklist item.
    watermark_id = f"exp_{uuid.uuid4().hex[:10]}"
    # A render failure is the paper's content, not the server: say so with a
    # 422 naming the cause instead of a bare 500 (ReportLab raises
    # LayoutError/IndexError/ValueError on content it cannot lay out).
    if fmt in ("answer-key", "answer_key", "answerKey"):
        try:
            path = pdf_export.export_answer_key_pdf(paper, out_dir, template=template)
        except Exception as exc:
            log.exception("answer key for paper %s failed to render", paper_id)
            raise HTTPException(422, f"this answer key could not be rendered: {exc}") from exc
        get_audit_log(cfg.data_root).append(
            "answer_key_exported", assessment_id=paper.assessment_id,
            details={"paperId": paper_id, "format": fmt, "file": path.name, "watermarkId": watermark_id},
        )
        return {"url": f"/api/v1/papers/{paper_id}/file?format=answer-key", "watermarkId": watermark_id}
    try:
        path = pdf_export.export_pdf(paper, out_dir, template=template, watermark_id=watermark_id)
    except Exception as exc:
        log.exception("paper %s failed to render", paper_id)
        raise HTTPException(422, f"this paper could not be rendered: {exc}") from exc
    get_audit_log(cfg.data_root).append(
        "paper_exported", assessment_id=paper.assessment_id,
        details={"paperId": paper_id, "format": fmt, "watermarkId": watermark_id},
    )
    return {"url": f"/api/v1/papers/{paper_id}/file", "watermarkId": watermark_id}


@router.get("/papers/{paper_id}/file")
def get_paper_file(paper_id: str, format: str = "pdf", current: User = Depends(require_staff)):
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
    request: CreateAssessmentRequest, current: User = Depends(require_staff),
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
def get_assessment(assessment_id: str, current: User = Depends(require_staff)) -> Assessment:
    _, store = _require()
    return require_school_owns_assessment(store, assessment_id, current)


@router.get("/assessments", response_model=list[Assessment])
def list_assessments(
    teacher_id: Optional[str] = None, school_id: Optional[str] = None,
    current: User = Depends(require_staff),
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
    assessment_id: str, assessment: Assessment, current: User = Depends(require_staff),
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
def delete_assessment(assessment_id: str, current: User = Depends(require_staff)) -> dict:
    _, store = _require()
    require_school_owns_assessment(store, assessment_id, current)
    store.delete(assessment_id)
    return {"ok": True}


@router.patch("/assessments/{assessment_id}/status", response_model=Assessment)
def update_status(assessment_id: str, body: dict, current: User = Depends(require_staff)) -> Assessment:
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
