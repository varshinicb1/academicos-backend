"""Marks entry and result analysis (EX-8) for a paper sat on paper.

The teacher types each student's marks per question into a grid -- no scan
needed. The grid is the section's roster by the paper's questions; a student
can be marked absent. The analysis reads the same grid: per question (mean
and facility), per chapter and topic (the bank's tags on each question),
and per class (mean, median, highest, lowest, and the CBSE 8-point grade
bands). A row is graded only once every question has a mark (0 is a mark):
a partly entered row is "incomplete", has no total, percent or band, and is
left out of the class figures (v3 audit N-2-10: 5 of 38 cells read 5.0, 6.2%,
band E, and were averaged in). Saving marks feeds each student's learning
progress, replacing that paper's earlier marks (source
"marks:{paper}:{student}"). Entering marks is
processing a student's work, so it needs the parent's consent.
"""
from __future__ import annotations

import statistics
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from ..operations.routes import store as ops
from .auth_routes import require_staff
from .authz import require_consent_for_all, require_school_owns_paper, student_names, students_without_consent
from .users import User

router = APIRouter(prefix="/api/v1")

# CBSE's 8-point scale for classes 9-10 (and the common scale below): the
# lower bound of each band, in percent.
GRADE_BANDS = (("A1", 91), ("A2", 81), ("B1", 71), ("B2", 61), ("C1", 51), ("C2", 41), ("D", 33), ("E", 0))


def band(percent: float) -> str:
    return next(label for label, floor in GRADE_BANDS if percent >= floor)


def is_complete(marks: dict[str, float], question_ids) -> bool:
    """Whether a student's marks cover every question of the paper -- the
    only row that has a total. Marks for a question no longer on the paper
    neither complete a row nor count."""
    qids = list(question_ids)
    return bool(qids) and all(q in marks for q in qids)


def paper_total(paper) -> int:
    """What the paper is out of: its sections' marks as printed, so an
    "attempt any 10 of 12" section counts 10, not 12 (an 80-mark English or
    Hindi paper was graded out of 94; review 2026-10-04)."""
    from .paper_store import printed_marks
    return printed_marks(paper)


def paper_score(paper, marks: Optional[dict[str, float]]) -> Optional[float]:
    """The student's total on `paper`, or None while a mark is missing. A
    section that prints more questions than are answered needs only that
    many marked, and counts the best that many: a student who answered 11 of
    "any 10" is not scored above the section's marks."""
    from .template_presets import section_attempts
    if not marks:
        return None
    got, scored = 0.0, False
    for sec in paper.sections:
        qids = [q.question_id for q in sec.questions]
        if not qids:
            continue
        need = section_attempts(sec)
        have = sorted((marks[q] for q in qids if q in marks), reverse=True)
        if len(have) < need:
            return None
        got += sum(have[:need])
        scored = True
    return got if scored else None


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class MarkEntry(_Req):
    student_id: str
    question_id: str
    marks: float = Field(ge=0)


class MarksRequest(_Req):
    entries: list[MarkEntry] = Field(default_factory=list, max_length=20000)
    absent: list[str] = Field(default_factory=list)       # students absent for the paper
    present: list[str] = Field(default_factory=list)      # clear an earlier absent


class GridQuestion(Camel):
    question_id: str
    number: str
    max_marks: int


class GridRow(Camel):
    student_id: str
    name: str
    absent: bool
    marks: dict[str, float]
    # True once every question has a mark. A row with only some cells
    # entered is not graded: total, percent and band stay None.
    complete: bool = False
    total: Optional[float] = None
    percent: Optional[float] = None
    band: Optional[str] = None


class MarksGrid(Camel):
    paper_id: str
    section_id: Optional[str] = None
    total_marks: int
    questions: list[GridQuestion]
    rows: list[GridRow]
    # Students whose marks this save left out because no parental consent is
    # on file, by name. One such student refused the whole section's marks,
    # naming an internal id (v3 audit N-4-5).
    without_consent: list[str] = []


class QuestionAnalysis(Camel):
    question_id: str
    number: str
    max_marks: int
    attempted: int
    mean: Optional[float] = None
    facility: Optional[float] = None       # mean / max: 0 hard, 1 easy
    chapter: Optional[str] = None
    topic: Optional[str] = None


class GroupAnalysis(Camel):
    key: str
    name: str
    max_marks: int
    facility: Optional[float] = None
    questions: int


class ClassAnalysis(Camel):
    students: int
    absent: int
    # Rows with only some marks entered: counted here, left out of the
    # averages and the bands.
    incomplete: int = 0
    mean_percent: Optional[float] = None
    median_percent: Optional[float] = None
    highest_percent: Optional[float] = None
    lowest_percent: Optional[float] = None
    bands: dict[str, int]


class PaperAnalysis(Camel):
    paper_id: str
    section_id: Optional[str] = None
    questions: list[QuestionAnalysis]
    chapters: list[GroupAnalysis]
    topics: list[GroupAnalysis]
    summary: ClassAnalysis
    weakest: list[str]                      # question numbers with the lowest facility, for re-teaching


# ---------------- helpers ----------------

def _papers():
    from . import routes as ar
    cfg, store = ar._require()
    return ar._require_papers(), store


def _paper(paper_id: str, current: User):
    papers, assessments = _papers()
    paper = require_school_owns_paper(papers, assessments, paper_id, current)
    return paper, assessments.get(paper.assessment_id)


def _questions(paper) -> list[GridQuestion]:
    out = []
    for sec in paper.sections:
        for q in sec.questions:
            out.append(GridQuestion(question_id=q.question_id, number=f"{sec.label}{q.display_number}"
                                    if sec.label and not str(q.display_number).startswith(sec.label)
                                    else str(q.display_number), max_marks=q.marks))
    return out


def _roster(section_id: Optional[str], current: User, marked: set[str]) -> list[tuple[str, str]]:
    """(student id, name): the section's students, or everyone marked."""
    users = cr._require_users()
    if section_id:
        cr._require_school_owns_section(section_id, current)
        ids = [e.student_id for e in cr._require().enrollments_for_section(section_id)]
    else:
        ids = sorted(marked)
    out = []
    for sid in ids:
        u = users.get(sid)
        out.append((sid, u.name if u else sid))
    return sorted(out, key=lambda t: t[1].lower())


def _grid(paper, section_id: Optional[str], current: User) -> MarksGrid:
    qs = _questions(paper)
    total = paper_total(paper)
    marks = ops().marks_for(paper.id)
    absent = ops().absent_for(paper.id)
    rows = []
    for sid, name in _roster(section_id, current, set(marks) | absent):
        m = marks.get(sid, {})
        got = None if sid in absent else paper_score(paper, m)
        complete = got is not None
        pct = round(100 * got / total, 1) if got is not None and total else None
        rows.append(GridRow(student_id=sid, name=name, absent=sid in absent, marks=m, complete=complete, total=got,
                            percent=pct, band=band(pct) if pct is not None else None))
    return MarksGrid(paper_id=paper.id, section_id=section_id, total_marks=total, questions=qs, rows=rows)


def _bank_schemas(subject: str, grade: int, qids: set[str]) -> dict[str, Any]:
    """The bank's QuestionSchema for each of `qids` it has. Tests replace this."""
    from .mapping import to_question_schema
    from .pool import get_pool
    pool = get_pool(cr._cfg, subject=subject, grade=str(grade))
    out = {}
    for pq in pool.questions:
        q = to_question_schema(pq)
        if q.id in qids:
            out[q.id] = q
    return out


def _schemas(assessment, qids) -> dict[str, Any]:
    if assessment is None:
        return {}
    try:
        return _bank_schemas(assessment.subject, int(assessment.grade), set(qids))
    except Exception:  # noqa: BLE001
        return {}


def _tags(assessment, qids: list[str]) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """question id -> (chapter, topic) names, from the bank's tags."""
    from ..operations.learning import _taxonomy
    if assessment is None:
        return {}
    chapters, _, topics = _taxonomy(assessment.subject, int(assessment.grade))
    out = {}
    for qid, q in _schemas(assessment, qids).items():
        ch = q.taxonomy_chapter_id or (q.chapter_ids[0] if q.chapter_ids else None)
        tp = next(iter(q.topic_ids), None)
        out[qid] = (chapters.get(ch, ch) if ch else None, topics.get(tp, tp) if tp else None)
    return out


def _feed_learning(paper, assessment, per_student: dict[str, dict[str, float]]) -> None:
    """Each student's marks on this paper, as learning evidence, replacing
    the paper's earlier copy. Only bank questions carry the tags progress
    needs; others are left out."""
    import logging
    from . import pillar_routes
    from .evaluate import Evaluation
    try:
        maxes = {q.question_id: q.marks for s in paper.sections for q in s.questions}
        schemas = _schemas(assessment, list(maxes))
        _, knowledge, _ = pillar_routes._require()
        for sid, m in per_student.items():
            graded = [(schemas[qid], Evaluation(question_id=qid, awarded_marks=v, max_marks=maxes[qid],
                                                verdict="teacher", confidence=1.0, reasoning="marks entry",
                                                needs_review=False))
                      for qid, v in m.items() if qid in schemas]
            if graded:
                knowledge.record_sheet(sid, graded, f"marks:{paper.id}:{sid}")
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("paper %s: learning evidence not recorded", paper.id, exc_info=True)


def may_enter_marks(current: User, assessment) -> bool:
    """Who may write marks on a paper: the principal, the paper's author, a
    teacher the principal made an exams admin for its class, or a teacher of
    its subject in its class (either teacher of a co-taught cell). Any staff
    member of the school could overwrite any paper's marks (E2E run on
    production's commit, 2026-10-01); reading the grid stays open to staff."""
    from .auth_routes import holds
    if current.role == "principal":
        return True
    if assessment is None:
        return False
    if assessment.teacher_id == current.id:
        return True
    try:
        grade = int(assessment.grade)
    except (TypeError, ValueError):
        return False
    if holds(current, "exams", grade=grade):
        return True
    return (grade, str(assessment.subject).casefold()) in teaching_cells(current)


def teaching_cells(current: User) -> set[tuple[int, str]]:
    """(class number, subject name casefolded) for every subject the teacher
    teaches in some section, co-taught cells included."""
    store = cr._require()
    out: set[tuple[int, str]] = set()
    for cell in store.allocations_for_teacher(current.id) + store.co_taught_allocations(current.id):
        subject = store.get_subject(cell.subject_id)
        g = store.get_grade(subject.grade_id) if subject else None
        if subject and g:
            out.add((g.number, subject.name.casefold()))
    return out


def chapters_by_student_from_marks(school_id: str) -> dict[str, dict[str, set[str]]]:
    """{assessment id: {student id: the chapter ids of the questions they have
    marks on}} for the school's papers with marks typed into the grid -- the
    same shape `GradedStore.chapters_by_student` gives for scanned sheets.
    School insights read only scanned sheets and showed 0 students and 0
    papers for a school that entered its marks by hand (E2E run, 2026-10-01)."""
    from . import routes as ar
    if ar._papers is None:
        return {}
    papers, assessments = ar._papers, ar._require()[1]
    out: dict[str, dict[str, set[str]]] = {}
    # From the papers that have marks, not every paper (and set) the school
    # has: a school's hundreds of papers were each read and queried.
    for paper_id in ops().papers_with_marks():
        paper = papers.get(paper_id)
        assessment = assessments.get(paper.assessment_id) if paper is not None else None
        if assessment is None or assessment.school_id != school_id:
            continue
        marks = ops().marks_for(paper.id)
        schemas = _schemas(assessment, {q for m in marks.values() for q in m})
        per = out.setdefault(paper.assessment_id, {})
        for sid, m in marks.items():
            per.setdefault(sid, set()).update(c for qid in m for c in (schemas[qid].chapter_ids if qid in schemas else []))
    return out


# ---------------- routes ----------------

@router.put("/papers/{paper_id}/marks", response_model=MarksGrid)
def enter_marks(paper_id: str, req: MarksRequest, section_id: Optional[str] = Query(default=None, alias="sectionId"),
                current: User = Depends(require_staff)) -> MarksGrid:
    """Save marks (any number of cells) and absences. Each mark lies between
    0 and its question's marks; a student must be of this school."""
    from ..assessment.audit_log import get_audit_log
    from ..assessment.consent import get_consent_store
    paper, assessment = _paper(paper_id, current)
    if not may_enter_marks(current, assessment):
        raise HTTPException(403, "only the paper's author, a teacher of its subject in its class, "
                                 "an exams admin or the principal enters marks on it")
    grid = _questions(paper)
    maxes = {q.question_id: q.max_marks for q in grid}
    numbers = {q.question_id: q.number for q in grid}
    users = cr._require_users()
    students = {e.student_id for e in req.entries} | set(req.absent) | set(req.present)
    for sid in students:
        u = users.get(sid)
        if u is None or u.school_id != current.school_id or u.role != "student":
            raise HTTPException(422, f"{sid} is not a student of this school")
    for e in req.entries:
        if e.question_id not in maxes:
            raise HTTPException(422, f"{e.question_id} is not a question of this paper")
        if e.marks > maxes[e.question_id]:
            # The question as the grid numbers it and the student by name:
            # "cbe:q:Maths10MM1 is out of 1" named neither (QA P-19).
            u = users.get(e.student_id)
            number = numbers[e.question_id]
            raise HTTPException(422, f"{'Q' + number if number.isdigit() else number} for "
                                     f"{u.name if u else e.student_id}: "
                                     f"{e.marks:g} is more than its {maxes[e.question_id]:g} "
                                     f"mark{'s' if maxes[e.question_id] != 1 else ''}")
    if set(req.absent) & {e.student_id for e in req.entries}:
        raise HTTPException(422, "a student cannot be absent and have marks")
    consents = get_consent_store(cr._cfg.data_root)
    marked = sorted({e.student_id for e in req.entries})
    missing = students_without_consent(consents, current.school_id, marked)
    if missing and len(missing) == len(marked):
        require_consent_for_all(consents, current.school_id, marked)      # 409 naming them
    entries = [e for e in req.entries if e.student_id not in set(missing)]
    ops().save_marks(paper.id, [(e.student_id, e.question_id, e.marks) for e in entries],
                     absent=req.absent, present=req.present, entered_by=current.id)
    # Each question's measured difficulty (EX-3): its facility across every
    # paper, student and school that sat it, stored as an aggregate only.
    ops().record_facility(paper.id, maxes)
    get_audit_log(cr._cfg.data_root).append(
        "marks_entered", assessment_id=paper.assessment_id, actor=current.id,
        details={"schoolId": current.school_id, "paperId": paper.id, "cells": len(entries),
                 "students": len({e.student_id for e in entries}), "absent": len(req.absent),
                 "withoutConsent": len(missing)})
    if assessment is not None and entries:
        all_marks = ops().marks_for(paper.id)
        _feed_learning(paper, assessment, {sid: all_marks.get(sid, {}) for sid in {e.student_id for e in entries}})
        _tell_results(paper, assessment, {e.student_id for e in entries}, all_marks)
    return _grid(paper, section_id, current).model_copy(update={"without_consent": student_names(missing)})


def _tell_results(paper, assessment, students: set[str], all_marks: dict[str, dict[str, float]]) -> None:
    """Each student whose marks were entered, and their parents, hear their
    total once per paper (SA-5: only homework results were ever sent), once
    every question has a mark. Later corrections are seen in the app; they
    are not sent again -- so a partly entered total, sent, was the one they
    kept (v3 audit N-2-10)."""
    from ..operations.routes import notify_parents_safely, notify_safely
    total = paper_total(paper)
    for sid in sorted(students):
        got = paper_score(paper, all_marks.get(sid, {}))
        if got is None:
            continue
        params = {"title": paper.metadata.assessment_title, "subject": assessment.subject,
                  "marks": f"{got:g} out of {total}"}
        notify_safely(school_id=assessment.school_id, user_ids=[sid], kind="test_marks", params=params,
                      link="/my-learning", dedupe_key=f"testmarks:{paper.id}:{sid}")
        notify_parents_safely(school_id=assessment.school_id, student_ids=[sid], kind="test_marks",
                              params=params, dedupe_key=f"testmarks-parent:{paper.id}:{sid}")


@router.get("/papers/{paper_id}/marks", response_model=MarksGrid)
def marks_grid(paper_id: str, section_id: Optional[str] = Query(default=None, alias="sectionId"),
               current: User = Depends(require_staff)) -> MarksGrid:
    """The grid: the section's students (or everyone marked) by the paper's
    questions, with totals, percentages and grade bands for each complete
    row."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    paper, _ = _paper(paper_id, current)
    grid = _grid(paper, section_id, current)
    record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what="paper_marks",
                    student_ids=[r.student_id for r in grid.rows], assessment_id=paper.assessment_id)
    return grid


@router.get("/papers/{paper_id}/analysis", response_model=PaperAnalysis)
def analysis(paper_id: str, section_id: Optional[str] = Query(default=None, alias="sectionId"),
             current: User = Depends(require_staff)) -> PaperAnalysis:
    """Results by question, chapter, topic and class. Absent students, and
    students whose marks are only partly entered, are counted, not averaged
    in. A question's mean and facility read the marks entered for it."""
    paper, assessment = _paper(paper_id, current)
    grid = _grid(paper, section_id, current)
    sat = [r for r in grid.rows if not r.absent]
    graded = [r for r in sat if r.complete]
    tags = _tags(assessment, [q.question_id for q in grid.questions])
    qa = []
    for q in grid.questions:
        vals = [r.marks[q.question_id] for r in sat if q.question_id in r.marks]
        mean = statistics.fmean(vals) if vals else None
        ch, tp = tags.get(q.question_id, (None, None))
        qa.append(QuestionAnalysis(question_id=q.question_id, number=q.number, max_marks=q.max_marks,
                                   attempted=len(vals), mean=round(mean, 2) if mean is not None else None,
                                   facility=round(mean / q.max_marks, 3) if mean is not None and q.max_marks else None,
                                   chapter=ch, topic=tp))

    def groups(attr: str) -> list[GroupAnalysis]:
        acc: dict[str, list[QuestionAnalysis]] = {}
        for x in qa:
            name = getattr(x, attr)
            if name:
                acc.setdefault(name, []).append(x)
        out = []
        for name, xs in acc.items():
            maxm = sum(x.max_marks for x in xs)
            got = [x.mean for x in xs if x.mean is not None]
            fac = round(sum(got) / sum(x.max_marks for x in xs if x.mean is not None), 3) if got else None
            out.append(GroupAnalysis(key=name, name=name, max_marks=maxm, facility=fac, questions=len(xs)))
        return sorted(out, key=lambda g: (g.facility is None, g.facility or 0))

    pcts = [r.percent for r in graded if r.percent is not None]
    bands = {label: 0 for label, _ in GRADE_BANDS}
    for p in pcts:
        bands[band(p)] += 1
    summary = ClassAnalysis(students=len(grid.rows), absent=sum(1 for r in grid.rows if r.absent),
                            incomplete=sum(1 for r in sat if r.marks and not r.complete),
                            mean_percent=round(statistics.fmean(pcts), 1) if pcts else None,
                            median_percent=round(statistics.median(pcts), 1) if pcts else None,
                            highest_percent=max(pcts) if pcts else None, lowest_percent=min(pcts) if pcts else None,
                            bands=bands)
    weakest = [x.number for x in sorted((x for x in qa if x.facility is not None), key=lambda x: x.facility)[:3]]
    return PaperAnalysis(paper_id=paper.id, section_id=section_id, questions=qa, chapters=groups("chapter"),
                         topics=groups("topic"), summary=summary, weakest=weakest)
