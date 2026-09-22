"""API for a teacher's paper templates: presets, CRUD, sharing, availability,
and generating a paper from a template (Task 901; the builder UI is Task 902).

A new module rather than more routes in routes.py / pillar_routes.py: other
work streams edit those files concurrently, and this surface is self-contained.

Who may do what (staff = teacher or principal; students get 403 everywhere,
another school gets 403, a missing id 404 -- the authz.py conventions):

  * read:   presets, anything in your school that is yours or shared; a
            principal may read every template in the school.
  * edit:   the owner only. A shared template is still its owner's; a
            colleague who wants changes takes a copy (duplicate).
  * share / unshare, delete: the owner or the school's principal.
  * presets are read-only for everyone; "use as starting point" is duplicate.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from ..config import Config
from . import paper_edit
from .audit_log import get_audit_log
from .auth_routes import require_staff
from .authz import require_own_school, require_school_owns_paper
from .paper import generate_paper_sets
from .paper_edit import SwapCounts, bank_for
from .paper_store import PaperStore, paper_question_ids
from .paper_templates import AvailabilityReport, SectionAvailability, SectionPlan, \
    build_scope_filter, check_scope_ids, plan
from .school_templates import TemplateStore
from .schemas import (
    Assessment,
    Blueprint,
    Camel,
    ChapterWeights,
    CompetencyWeights,
    GeneratedPaper,
    PaperTemplate,
    PaperTemplateDraft,
    QuestionSchema,
    SchoolTemplate,
    SectionBlueprint,
    TemplateScope,
)
from .store import AssessmentStore
from .template_presets import get_preset, instructions_for, is_preset_id, presets
from .templates import TIER_BLOOM, TIER_DIFFICULTY
from .users import User

router = APIRouter(prefix="/api/v1")

_cfg: Optional[Config] = None
_templates: Optional[TemplateStore] = None
_assessments: Optional[AssessmentStore] = None
_papers: Optional[PaperStore] = None


def init(config: Config) -> None:
    """Same files routes.py and pillar_routes.py open (each module owns its
    own store instances pointed at the shared file; see authz.py)."""
    global _cfg, _templates, _assessments, _papers
    _cfg = config
    _templates = TemplateStore(config.data_root / "templates" / "templates.sqlite")
    _assessments = AssessmentStore(config.data_root / "assessments" / "assessments.sqlite")
    _papers = PaperStore(config.data_root / "assessments" / "papers.sqlite")


def _require() -> tuple[Config, TemplateStore]:
    if _cfg is None or _templates is None:
        raise HTTPException(503, "paper template module not initialized")
    return _cfg, _templates


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---- request / response shapes ----

class TemplateListResponse(Camel):
    presets: list[PaperTemplate]
    mine: list[PaperTemplate]
    shared: list[PaperTemplate]


class ShareRequest(Camel):
    shared: bool


class DuplicateRequest(Camel):
    name: Optional[str] = None


class AvailabilityRequest(Camel):
    # This paper's chapters, without saving them into the template: a preset
    # has no scope, and a teacher reusing their own template for next month's
    # test should not have to edit it first.
    scope: Optional[TemplateScope] = None
    # The same switch generation takes, so the check can show what borrowing
    # would fill -- availability is generation run dry, not an estimate.
    fill_from_outside_scope: bool = False


class GenerateFromTemplateRequest(Camel):
    template_id: str
    scope: Optional[TemplateScope] = None
    title: Optional[str] = None
    # A-E: `generate_paper_sets` labels five sets, and every set is saved as
    # its own papers row, so an unbounded count is unbounded work and writes
    # in one request. Out of range is a 422, not a silent clamp.
    set_count: int = Field(1, ge=1, le=5)
    # A short paper is never produced silently: without this, any section
    # the bank cannot fill is a 409 carrying the gaps.
    allow_gaps: bool = False
    fill_from_outside_scope: bool = False


class TemplatePaperResponse(Camel):
    paper: GeneratedPaper
    assessment_id: str
    complete: bool
    gaps: list[SectionAvailability]
    notes: list[str]
    availability: AvailabilityReport


# ---- permission helpers ----

def _require_staff(current: User) -> None:
    if current.role not in ("teacher", "principal"):
        raise HTTPException(403, "paper templates are for teachers and principals")


def _load(template_id: str, current: User) -> PaperTemplate:
    """A template the caller may READ, or the right 403/404."""
    if is_preset_id(template_id):
        preset = get_preset(template_id)
        if preset is None:
            raise HTTPException(404, "preset not found")
        return preset
    _, store = _require()
    t = store.get_paper_template(template_id)
    if t is None:
        raise HTTPException(404, "template not found")
    if t.school_id != current.school_id:
        raise HTTPException(403, "this template belongs to a different school")
    if t.owner_id != current.id and not t.shared and current.role != "principal":
        raise HTTPException(403, "this template is private to the teacher who made it")
    return t


def _refuse_preset(t: PaperTemplate) -> None:
    if t.is_preset:
        raise HTTPException(403, "presets are read-only; duplicate one to make it your own")


def _require_owner_or_principal(t: PaperTemplate, current: User, action: str) -> None:
    if t.owner_id != current.id and current.role != "principal":
        raise HTTPException(403, f"only the template's owner or the principal can {action} it")


def _checked_scope(scope: TemplateScope) -> list[str]:
    """422 naming any topic/subtopic id the taxonomy does not define;
    otherwise the notes on what could not be checked (no taxonomy yet)."""
    unknown, notes = check_scope_ids(scope)
    if unknown:
        raise HTTPException(422, "not in the topic taxonomy: " + ", ".join(unknown))
    return notes


def _scoped(school_id: str, current: User) -> None:
    _require_staff(current)
    require_own_school(school_id, current)


# ---- CRUD + sharing ----

@router.get("/schools/{school_id}/teacher-templates", response_model=TemplateListResponse)
def list_teacher_templates(
    school_id: str, grade: Optional[int] = None, subject: Optional[str] = None,
    current: User = Depends(require_staff),
) -> TemplateListResponse:
    """Presets + mine + shared with the school, in one call: the builder's
    first screen needs all three, and a phone on a slow link should not make
    three round trips to draw it."""
    _scoped(school_id, current)
    _, store = _require()

    def keep(t: PaperTemplate) -> bool:
        return ((grade is None or t.grade == grade)
                and (subject is None or t.subject.lower() == subject.lower()))

    saved = [t for t in store.list_paper_templates(school_id) if keep(t)]
    return TemplateListResponse(
        presets=presets(grade=grade, subject=subject),
        mine=[t for t in saved if t.owner_id == current.id],
        shared=[t for t in saved if t.owner_id != current.id
                and (t.shared or current.role == "principal")],
    )


@router.post("/schools/{school_id}/teacher-templates", response_model=PaperTemplate)
def create_teacher_template(
    school_id: str, draft: PaperTemplateDraft, current: User = Depends(require_staff),
) -> PaperTemplate:
    _scoped(school_id, current)
    _, store = _require()
    notes = _checked_scope(draft.scope)
    now = _now()
    t = PaperTemplate(**draft.model_dump(), id="new", school_id=school_id,
                      owner_id=current.id, created_at=now, updated_at=now)
    return store.save_paper_template(t).model_copy(update={"scope_notes": notes})


@router.get("/schools/{school_id}/teacher-templates/{template_id}", response_model=PaperTemplate)
def get_teacher_template(
    school_id: str, template_id: str, current: User = Depends(require_staff),
) -> PaperTemplate:
    _scoped(school_id, current)
    return _load(template_id, current)


@router.put("/schools/{school_id}/teacher-templates/{template_id}", response_model=PaperTemplate)
def update_teacher_template(
    school_id: str, template_id: str, draft: PaperTemplateDraft,
    current: User = Depends(require_staff),
) -> PaperTemplate:
    _scoped(school_id, current)
    _, store = _require()
    t = _load(template_id, current)
    _refuse_preset(t)
    if t.owner_id != current.id:
        raise HTTPException(403, "only the template's owner can edit it; duplicate it to change a copy")
    notes = _checked_scope(draft.scope)
    # Generated instructions stay generated through a section edit (the
    # client sends them back unchanged); the teacher's first change to the
    # text makes it theirs, printed as written from then on.
    generated = (t.instructions_generated
                 and draft.instructions.strip() == t.instructions.strip())
    updated = t.model_copy(update={**dict(draft), "pattern_status": "teacher",
                                   "instructions_generated": generated,
                                   "updated_at": _now()})
    saved = store.save_paper_template(PaperTemplate.model_validate(updated.model_dump()))
    return saved.model_copy(update={"scope_notes": notes})


@router.delete("/schools/{school_id}/teacher-templates/{template_id}")
def delete_teacher_template(
    school_id: str, template_id: str, current: User = Depends(require_staff),
) -> dict:
    _scoped(school_id, current)
    _, store = _require()
    t = _load(template_id, current)
    _refuse_preset(t)
    _require_owner_or_principal(t, current, "delete")
    store.delete(t.id)
    return {"ok": True}


@router.post("/schools/{school_id}/teacher-templates/{template_id}/share",
             response_model=PaperTemplate)
def share_teacher_template(
    school_id: str, template_id: str, req: ShareRequest,
    current: User = Depends(require_staff),
) -> PaperTemplate:
    _scoped(school_id, current)
    _, store = _require()
    t = _load(template_id, current)
    _refuse_preset(t)
    _require_owner_or_principal(t, current, "share or unshare")
    return store.save_paper_template(t.model_copy(update={"shared": req.shared,
                                                          "updated_at": _now()}))


@router.post("/schools/{school_id}/teacher-templates/{template_id}/duplicate",
             response_model=PaperTemplate)
def duplicate_teacher_template(
    school_id: str, template_id: str, req: DuplicateRequest,
    current: User = Depends(require_staff),
) -> PaperTemplate:
    """"Use as starting point": any template the caller can read, including a
    preset, becomes a private template they own.

    The copy is labelled the teacher's own pattern from the start, not the
    preset's `cbse_board` / `suggested`: it is theirs to change, and a label
    that stays "CBSE board pattern" until the first edit would vouch for
    whatever they change it to. Where it came from stays in `based_on` and
    in the source line."""
    _scoped(school_id, current)
    _, store = _require()
    src = _load(template_id, current)
    now = _now()
    copy = src.model_copy(update={
        "id": "new", "school_id": current.school_id, "owner_id": current.id,
        "shared": False, "is_preset": False, "based_on": src.id,
        "pattern_status": "teacher",
        "source": f"Copied from {src.name}." + (f" {src.source}" if src.source else ""),
        "name": req.name or (src.name if src.is_preset else f"{src.name} (copy)"),
        "created_at": now, "updated_at": now,
    })
    return store.save_paper_template(copy)


# ---- availability + generation ----

def _candidates(cfg: Config, template: PaperTemplateDraft) -> list[QuestionSchema]:
    """The class+subject bank, mapped once per pool (`paper_edit.bank_for`)
    rather than per request: 22 ms of every generation on the class 10
    Science bank. The questions are shared -- never mutate one here."""
    try:
        return list(bank_for(cfg, template.subject, template.grade).questions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _plan(template: PaperTemplate, scope: Optional[TemplateScope], *,
          fill_from_outside_scope: bool = False) -> tuple[AvailabilityReport, list[SectionPlan]]:
    cfg, _ = _require()
    if scope is not None:
        template = template.model_copy(update={"scope": scope})
    candidates = _candidates(cfg, template)

    def subtopic_questions(ids: list[str]) -> list[str]:
        from ..curriculum.store import get_curriculum_store
        return get_curriculum_store(cfg.data_root).question_ids_for_subtopics(ids)

    scope_filter = build_scope_filter(template.scope, candidates, subtopic_questions)
    return plan(template, template.id, candidates, scope_filter,
                fill_from_outside_scope=fill_from_outside_scope)


@router.post("/schools/{school_id}/teacher-templates/{template_id}/availability",
             response_model=AvailabilityReport)
def template_availability(
    school_id: str, template_id: str, req: AvailabilityRequest,
    current: User = Depends(require_staff),
) -> AvailabilityReport:
    """Per section: needed vs. eligible (verified key, right marks, in scope),
    what the paper will hold, every note generation would add, and why any
    shortfall exists with the changes that would close it -- shown BEFORE
    generating, so a teacher never discovers a gap on the printed paper.
    It is the generator's own selection run without saving anything."""
    _scoped(school_id, current)
    if req.scope is not None:
        _checked_scope(req.scope)
    report, _ = _plan(_load(template_id, current), req.scope,
                      fill_from_outside_scope=req.fill_from_outside_scope)
    return report


@router.post("/papers/generate-from-template", response_model=TemplatePaperResponse)
def generate_from_template(
    req: GenerateFromTemplateRequest, current: User = Depends(require_staff),
) -> TemplatePaperResponse:
    started = time.perf_counter()
    _require_staff(current)
    cfg, _ = _require()
    if _assessments is None or _papers is None:
        raise HTTPException(503, "paper template module not initialized")
    template = _load(req.template_id, current)
    if req.scope is not None:
        _checked_scope(req.scope)
        template = template.model_copy(update={"scope": req.scope})
    report, plans = _plan(template, None, fill_from_outside_scope=req.fill_from_outside_scope)

    # The gaps are the availability report's own: one plan run decides both.
    gaps: list[SectionAvailability] = [p.availability for p in plans if p.availability.shortfall]
    if sum(p.availability.filled for p in plans) == 0:
        raise HTTPException(400, {
            "message": "no question in the bank fits this template",
            "availability": report.model_dump(by_alias=True, mode="json")})
    if gaps and not req.allow_gaps:
        raise HTTPException(409, {
            "message": f"{len(gaps)} section(s) cannot be filled from the bank; "
                       "see availability, or generate anyway with allowGaps",
            "availability": report.model_dump(by_alias=True, mode="json"),
            "gaps": [g.model_dump(by_alias=True, mode="json") for g in gaps]})

    _attach_choices(plans)
    selected, blueprint = _blueprint(template, plans)
    asm_id = f"asm_tpl_{uuid.uuid4().hex[:8]}"
    title = req.title or template.header.exam_name or template.name
    paper = generate_paper_sets(
        paper_id=f"paper_{uuid.uuid4().hex[:12]}", assessment_id=asm_id,
        assessment_title=title, subject=template.subject, grade=template.grade,
        blueprint=blueprint, selected_questions=selected, set_count=req.set_count,
        # Sets rotate within each section, never across two same-mark
        # sections (an assertion-reason question under the MCQ heading).
        rotation_groups=[p.picked + p.borrowed for p in plans if p.picked or p.borrowed],
    )
    instructions = _printed_instructions(template, plans)
    paper = _with_header(paper, template, instructions)
    branding = _branding(template, current.school_id)
    _papers.save(paper, branding, school_id=current.school_id)
    for s in paper.sets:
        _papers.save(s, branding, school_id=current.school_id)

    notes = [n for p in plans for n in p.notes] + list(report.scope_notes)
    now = _now()
    _assessments.save(Assessment(
        id=asm_id, school_id=current.school_id, teacher_id=current.id, title=title,
        subject=template.subject, grade=template.grade,
        chapter_ids=list(template.scope.chapter_ids), blueprint=blueprint,
        status="paperGenerated", created_at=now, updated_at=now,
        template_id=None if template.is_preset else template.id,
        generated_paper_id=paper.id, selected_question_ids=[q.id for q in selected],
        # Where the paper came from and what it was told; the header and
        # instructions it prints are on the paper itself (`_with_header`).
        metadata={"paperTemplate": {
            "id": template.id, "name": template.name,
            "header": template.header.model_dump(by_alias=True),
            "instructions": instructions,
            "gaps": [g.model_dump(by_alias=True) for g in gaps], "notes": notes,
            # The rules each section was filled under, keyed by the section
            # id the paper prints, so a swap or pick later applies the same
            # ones even after the template itself is edited or deleted.
            "sections": [{**s.model_dump(by_alias=True, mode="json"),
                          "id": s.id or f"s{i + 1}"}
                         for i, s in enumerate(template.sections)],
            "scope": template.scope.model_dump(by_alias=True, mode="json"),
            "fillFromOutsideScope": req.fill_from_outside_scope}},
    ))
    get_audit_log(cfg.data_root).append(
        "template_paper_generated", assessment_id=asm_id,
        details={"paperId": paper.id, "templateId": template.id, "userId": current.id,
                 "schoolId": current.school_id, "gaps": len(gaps),
                 "generationSeconds": round(time.perf_counter() - started, 3)},
    )
    return TemplatePaperResponse(paper=paper, assessment_id=asm_id, complete=not gaps,
                                 gaps=gaps, notes=notes, availability=report)


def _attach_choices(plans: list[SectionPlan]) -> None:
    """Each planned OR alternative onto its question, under the metadata keys
    `paper.generate_paper` prints as "OR" and keys as "<n>_OR" -- the same
    keys `selection.optimize` uses, so the renderer, the PDF exporter and the
    set rotation (which swaps primary and alternative) need nothing new. The
    bank's questions are shared across requests (`_candidates`), so a
    question carrying an OR is a copy, put back in the plan in its place."""
    def with_choice(q: QuestionSchema, p: SectionPlan) -> QuestionSchema:
        alt = p.alternatives.get(q.id)
        if alt is None:
            return q
        q = q.model_copy(deep=True)
        q.metadata["internal_choice_id"] = alt.id
        q.metadata["internal_choice_stem"] = alt.stem
        q.metadata["internal_choice_scheme"] = alt.answer_scheme.model_answer
        q.metadata["internal_choice_question"] = alt.model_dump(by_alias=False)
        return q

    for p in plans:
        p.picked = [with_choice(q, p) for q in p.picked]
        p.borrowed = [with_choice(q, p) for q in p.borrowed]


def _printed_instructions(template: PaperTemplate, plans: list[SectionPlan]) -> str:
    """The instructions to print: the teacher's own words as written, but
    instructions a preset generated (still unedited, on the preset or on a
    copy of it, whatever sections the teacher has changed since) rebuilt
    from what the paper holds.

    Those lines are the preset describing its own sections, and they replace
    the PDF's default list, so stamping them unchanged printed promises the
    paper did not keep: on the real class 10 bank the Social Science paper
    said "Section D ... 4 of them offer an internal choice (OR)" with not one
    OR in the section. Text the teacher wrote is theirs to word; the
    availability notes already name the ORs and questions the bank lacks.

    Generated text is known by `instructions_generated`; text equal to what
    the sections generate also counts, for a template created from a
    preset's fields rather than duplicated, and for copies saved before the
    flag existed."""
    generated = (template.instructions_generated
                 or template.instructions.strip() == instructions_for(template.sections).strip())
    if not generated:
        return template.instructions
    return instructions_for(template.sections, [
        (p.availability.filled, p.availability.choices_filled) for p in plans])


def _with_header(paper: GeneratedPaper, template: PaperTemplate,
                 instructions: str) -> GeneratedPaper:
    """The template's header and duration, and the instructions to print
    (`_printed_instructions`), on the paper and on every set, where the PDF
    exporter reads them. Blank header fields stay None, so the exporter
    falls back to the school's branding name."""
    header = template.header
    update = {
        "school_name": header.school_name or None,
        "exam_name": header.exam_name or None,
        "date_line": header.date_line or None,
        "instructions": instructions or None,
        "duration_minutes": template.duration_minutes,
    }

    def stamp(p: GeneratedPaper) -> GeneratedPaper:
        return p.model_copy(update={
            "metadata": p.metadata.model_copy(update=update),
            "sets": [stamp(s) for s in p.sets]})

    return stamp(paper)


def _branding(template: PaperTemplate, school_id: str) -> SchoolTemplate:
    """What the PDF exporter prints around the header: the school's own
    branding (logo, colour, fonts, tagline), named as the teacher's header
    names the school.

    The exporter prints `name` as the school's name, so storing the paper
    template itself there printed "Unit test - Chemical reactions" where the
    school goes and dropped the logo. The date line is not folded into the
    tagline any more: it travels on the paper (`_with_header`), and folding
    it in too printed it twice."""
    _, store = _require()
    school = store.default_for(school_id)
    return SchoolTemplate.model_validate({
        **school.model_dump(),
        "name": template.header.school_name or school.name,
    })


def _blueprint(template: PaperTemplate,
               plans: list[SectionPlan]) -> tuple[list[QuestionSchema], Blueprint]:
    """The picked questions in section order, and a blueprint whose sections
    say exactly how many each got. `paper.generate_paper` buckets questions
    by mark value and takes each section's count in order, so recording the
    real count per section is what keeps two same-mark sections apart.

    The paper's total is the attempted marks it actually carries: a short
    paper the teacher accepted must not print the template's full total as
    its maximum marks."""
    selected: list[QuestionSchema] = []
    sections: list[SectionBlueprint] = []
    total = 0
    for i, p in enumerate(plans):
        got = p.picked + p.borrowed
        if not got:
            continue
        attempts = min(p.section.attempts, len(got))
        label = chr(ord("A") + len(sections))
        name = p.section.title
        if attempts < len(got):
            name = f"{name} (attempt any {attempts} of {len(got)})"
        sections.append(SectionBlueprint(
            id=p.section.id or f"s{i + 1}", label=label, name=name,
            marks_per_question=p.section.marks_each, question_count=len(got),
            total_marks=attempts * p.section.marks_each,
            has_internal_choice=attempts < len(got),
            internal_choice_count=len(got) - attempts,
        ))
        total += attempts * p.section.marks_each
        selected.extend(got)
    blueprint = Blueprint(
        total_marks=total, duration_minutes=template.duration_minutes,
        difficulty=TIER_DIFFICULTY["standard"], bloom=TIER_BLOOM["standard"],
        chapter_weights=ChapterWeights(), competency_weights=CompetencyWeights(),
        sections=sections, exam_type=template.exam_type,
        metadata={"templateId": template.id},
    )
    return selected, blueprint


# ---- swap / pick one question (Task 903; the rules are paper_edit.py's)

class PickRequest(Camel):
    question_id: str = Field(min_length=1)


class PaperEditResponse(Camel):
    paper: GeneratedPaper
    slot: str
    replaced_question_id: str
    question_id: str
    # What the teacher should know about the new question (from a recent
    # paper, outside the chapters, not the section's kind).
    notes: list[str] = []
    # A swap's alternatives, constraint by constraint (none for a pick).
    counts: Optional[SwapCounts] = None


@router.post("/papers/{paper_id}/questions/{slot}/swap", response_model=PaperEditResponse)
def swap_paper_question(paper_id: str, slot: str,
                        current: User = Depends(require_staff)) -> PaperEditResponse:
    """The next best question for the slot ("7", or "7_OR" for its OR
    alternative), saved onto the paper and every set, answer key included.
    409 "no alternative in this scope" with the counts when there is none."""
    return _edit_question(paper_id, slot, current, pick=None)


@router.post("/papers/{paper_id}/questions/{slot}", response_model=PaperEditResponse)
def pick_paper_question(paper_id: str, slot: str, req: PickRequest,
                        current: User = Depends(require_staff)) -> PaperEditResponse:
    """Put the teacher's chosen question in the slot. 422 for a question with
    no verified answer key (rule Q1) or marks that do not fit the slot, 409
    for one already on the paper or a near-duplicate of one."""
    return _edit_question(paper_id, slot, current, pick=req.question_id)


def _edit_question(paper_id: str, slot_key: str, current: User, *,
                   pick: Optional[str]) -> PaperEditResponse:
    from .routes import _require_editable

    started = time.perf_counter()
    _require_staff(current)
    cfg, _ = _require()
    if _assessments is None or _papers is None:
        raise HTTPException(503, "paper template module not initialized")
    paper = require_school_owns_paper(_papers, _assessments, paper_id, current)
    asm = _assessments.get(paper.assessment_id)
    _require_editable(asm)
    try:
        bank = bank_for(cfg, paper.metadata.subject, paper.metadata.grade)
        slot = paper_edit.find_slot(paper, slot_key)

        def subtopic_questions(ids: list[str]) -> list[str]:
            from ..curriculum.store import get_curriculum_store
            return get_curriculum_store(cfg.data_root).question_ids_for_subtopics(ids)

        rule = paper_edit.rule_for(slot, asm.metadata.get("paperTemplate") or {},
                                   asm.chapter_ids, bank, subtopic_questions)
        printed = paper_edit.printed_questions(paper)
        edits = dict(asm.metadata.get("paperEdits") or {})
        removed = set(edits.get("removed") or [])
        recent = _recent_question_ids(asm)
        if pick is None:
            choice = paper_edit.choose_swap(slot, rule, bank, printed, removed, recent)
        else:
            choice = paper_edit.check_pick(slot, rule, bank, pick, printed, recent,
                                           grade=paper.metadata.grade,
                                           subject=paper.metadata.subject)
    except paper_edit.EditError as err:
        raise HTTPException(err.status, err.detail) from err
    except ValueError as err:                      # an unreadable grade on the paper
        raise HTTPException(422, str(err)) from err

    old_id, new = slot.question_id, choice.question
    family = _papers.family(paper.id)
    if not any(m.id == paper.id for m in family):  # a row outside the set naming
        family.append(paper)
    edited = paper
    for member in family:
        updated = paper_edit.replace_question(member, old_id, new)
        _papers.save(updated, _papers.get_template(member.id), school_id=asm.school_id)
        _forget_pdfs(cfg, member.id)
        if member.id == paper.id:
            edited = updated

    removed.add(old_id)
    removed.discard(new.id)
    kind = "swap" if pick is None else "pick"
    edits["removed"] = sorted(removed)
    edits["log"] = [*(edits.get("log") or []), {
        "kind": kind, "slot": slot.key, "from": old_id, "to": new.id,
        "by": current.id, "at": _now().isoformat()}]
    _assessments.save(asm.model_copy(update={
        "selected_question_ids": [new.id if i == old_id else i
                                  for i in asm.selected_question_ids],
        "metadata": {**asm.metadata, "paperEdits": edits},
        "updated_at": _now()}))
    get_audit_log(cfg.data_root).append(
        f"paper_question_{kind}", assessment_id=asm.id,
        details={"paperId": paper.id, "slot": slot.key, "from": old_id, "to": new.id,
                 "userId": current.id, "schoolId": current.school_id,
                 "editSeconds": round(time.perf_counter() - started, 3)})
    return PaperEditResponse(paper=edited, slot=slot.key, replaced_question_id=old_id,
                             question_id=new.id, notes=choice.notes, counts=choice.counts)


def _recent_question_ids(asm: Assessment) -> set[str]:
    """Every question on the teacher's `RECENT_PAPERS` most recent other
    papers for this class and subject (see paper_edit's module doc)."""
    assert _assessments is not None and _papers is not None
    others = sorted(
        (a for a in _assessments.list_by_teacher(asm.teacher_id)
         if a.id != asm.id and a.generated_paper_id and a.grade == asm.grade
         and a.subject.lower() == asm.subject.lower()),
        key=lambda a: a.created_at, reverse=True)
    ids: set[str] = set()
    for a in others[:paper_edit.RECENT_PAPERS]:
        p = _papers.get(a.generated_paper_id)
        if p is not None:
            ids |= paper_question_ids(p)
    return ids


def _forget_pdfs(cfg: Config, paper_id: str) -> None:
    """The export route renders from the stored paper, but a PDF rendered
    before this edit would still be served by GET /papers/{id}/file until the
    next export, printing the question the teacher removed. Dropping it makes
    the file route answer "not generated yet", and the client exports again."""
    for suffix in (".pdf", "_answer_key.pdf"):
        (cfg.artifacts_dir / "papers" / f"{paper_id}{suffix}").unlink(missing_ok=True)
