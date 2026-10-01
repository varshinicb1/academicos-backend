"""HTTP routes for self-practice (SA-4; operations/practice.py). Students
only, and only their own sets."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .homework import grade_answers
from . import homework_routes as hr
from .homework_routes import EvaluationItem, StudentQuestion, _consent, _student_question
from .practice import DEFAULT_COUNT, pick
from .routes import store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class PracticeRequest(_Req):
    subject: Optional[str] = None          # a bank subject name; default: the one "revise next" puts first
    chapter_ids: list[str] = Field(default_factory=list, max_length=10)
    count: int = Field(default=DEFAULT_COUNT, ge=3, le=20)
    # SA-4 exam preparation: a published, upcoming paper of the student's class.
    exam_paper_id: Optional[str] = Field(default=None, max_length=80)


class PracticeSubmit(_Req):
    answers: dict[str, str]


class PracticeView(Camel):
    id: str
    subject: str
    grade: int
    chapter_ids: list[str]
    questions: list[StudentQuestion]
    created_at: str
    submitted_at: Optional[str] = None
    marks: Optional[float] = None
    max_marks: int
    answers: dict[str, str] = Field(default_factory=dict)
    evaluations: list[EvaluationItem] = Field(default_factory=list)


def _student(current: User):
    if current.role != "student":
        raise HTTPException(403, "practice is for students")
    cs = cr._require()
    e = cs.enrollment_for_student(current.id)
    section = cs.get_section(e.section_id) if e and e.section_id else None
    if section is None:
        raise HTTPException(409, "you are not placed in a class yet -- ask your school")
    return cs.get_grade(section.grade_id).number


def _view(p: dict) -> PracticeView:
    done = p["submitted_at"] is not None
    models = {q["id"]: (q.get("answerScheme") or {}).get("modelAnswer") or None for q in p["questions"]}
    evals = [EvaluationItem(question_id=e["questionId"], awarded_marks=e["awardedMarks"], max_marks=e["maxMarks"],
                            verdict=e["verdict"], reasoning=e["reasoning"], needs_review=e["needsReview"],
                            model_answer=models.get(e["questionId"])) for e in p["evaluations"]] if done else []
    return PracticeView(id=p["id"], subject=p["subject"], grade=p["grade"], chapter_ids=p["chapter_ids"],
                        questions=[_student_question(q) for q in p["questions"]], created_at=p["created_at"],
                        submitted_at=p["submitted_at"], marks=p["marks"], max_marks=p["max_marks"],
                        answers=p["answers"] if done else {}, evaluations=evals)


def _own(practice_id: str, current: User) -> dict:
    p = store().get_practice(practice_id)
    if p is None or p["student_id"] != current.id:
        raise HTTPException(404, "practice set not found")
    return p


def _view_chapters(subject: str, grade: int, chapter_ids: list[str]) -> list[str]:
    """The chapter view's ids for chapters named any way the app may name
    them (N-5-2). "Revise next" names a chapter by the tagger's id
    ("science-10/acids-bases-and-salts"); the bank files questions by the
    view's ("acids-bases-salts"), so practising a revise row found nothing for
    every class whose two ids differ. An id the view cannot place is kept as
    sent: the filter still matches records that carry it."""
    from ..assessment.chapter_filing import UNMAPPED, ChapterFiling
    filing = ChapterFiling.for_class(hr.bank_subject(subject), grade)
    out = []
    for cid in chapter_ids:
        view = filing.chapter_of(cid, [cid])
        out.append(cid if view == UNMAPPED else view)
    return list(dict.fromkeys(out))


def _exam_paper(current: User, grade: int, paper_id: str) -> dict:
    """The student's own upcoming paper, from a published datesheet."""
    ops = store()
    today = cr._school_today().isoformat()
    for exam in ops.exams_for_school(current.school_id):
        if exam["status"] != "published":
            continue
        for p in ops.exam_papers(exam["id"]):
            if p["id"] == paper_id:
                if p["grade"] != grade:
                    break
                if p["date"] < today:
                    raise HTTPException(409, "that paper has already been written")
                return {**p, "exam_name": exam["name"]}
    raise HTTPException(404, "no such paper on your class's datesheet")


def _taught_chapters(current: User, subject: str, grade: int, until: str) -> list[str]:
    """The chapters of `subject` the student's section has been taught by
    `until`, as the bank's chapter view names them: what an exam on that
    date can fairly ask. Matched by the chapter's printed name, so a plan
    built from either book edition finds the bank's chapter."""
    from ..assessment.chapter_filing import ChapterFiling, name_key
    cs = cr._require()
    e = cs.enrollment_for_student(current.id)
    section = e.section_id if e else None
    years = [y for y in cs.academic_years_for_school(current.school_id) if y.start_date <= until <= y.end_date]
    if not years or section is None:
        return []
    filing = ChapterFiling.for_class(hr.bank_subject(subject), grade)
    out: list[str] = []
    seen: dict[str, str] = {}
    for lesson in cs.scheduled_lessons_for_date_range(current.school_id, years[0].start_date, until):
        if lesson.status != "completed" or lesson.section_id not in (None, section):
            continue
        if lesson.subtopic_id not in seen:
            sub = cs.get_subtopic(lesson.subtopic_id)
            topic = cs.get_topic(sub.topic_id) if sub else None
            chapter = cs.get_chapter(topic.chapter_id) if topic else None
            seen[lesson.subtopic_id] = filing.by_name.get(name_key(chapter.name), "") if chapter else ""
        if seen[lesson.subtopic_id] and seen[lesson.subtopic_id] not in out:
            out.append(seen[lesson.subtopic_id])
    return out


@router.post("/my-practice", response_model=PracticeView)
def start_practice(req: PracticeRequest, current: User = Depends(get_current_user)) -> PracticeView:
    """A practice set from the caller's weakest chapters (or the ones named),
    skipping questions answered in the last two weeks."""
    grade = _student(current)
    s = store()
    subject, chapters = req.subject, list(req.chapter_ids)
    count = req.count
    if req.exam_paper_id:
        # SA-4 exam preparation: the paper's subject, from what has been taught
        # by its date, the student's weak chapters first; a longer set.
        paper = _exam_paper(current, grade, req.exam_paper_id)
        subject = paper["subject_name"]
        weak = [r["chapter_id"] for r in s.learning_progress(current.id)["revise_next"]
                if r["grade"] == grade and r["subject"].lower() == subject.lower()]
        chapters = list(dict.fromkeys(_view_chapters(subject, grade, weak) if weak else []))
        chapters += [c for c in _taught_chapters(current, subject, grade, paper["date"]) if c not in chapters]
        chapters = chapters[:10]
        count = max(count, 10)
    if not chapters and not req.exam_paper_id:
        revise = [r for r in s.learning_progress(current.id)["revise_next"]
                  if r["grade"] == grade and (subject is None or r["subject"].lower() == subject.lower())]
        if revise:
            subject = subject or revise[0]["subject"]
            chapters = list(dict.fromkeys(r["chapter_id"] for r in revise if r["subject"] == subject))[:3]
    if subject is None:
        raise HTTPException(409, "choose a subject: nothing is marked yet to say which chapters to revise")
    chapters = _view_chapters(subject, grade, chapters) if chapters else chapters
    pool = hr._usable(hr._bank_questions(subject, grade, chapters or None), current)
    if not pool:
        raise HTTPException(422, f"the question bank has no {subject} questions for class {grade} there yet")
    questions = pick(pool, recent=s.recently_answered(current.id), count=count,
                     seed=f"{current.id}:{subject}:{','.join(chapters)}:{len(s.practice_for(current.id, 1000))}")
    if not questions:
        raise HTTPException(409, "you have practised every question here in the last two weeks -- well done; "
                                 "try another chapter")
    p = s.create_practice(student_id=current.id, subject=subject, grade=grade, chapter_ids=chapters,
                          questions=questions)
    return _view(p)


@router.get("/my-practice", response_model=list[PracticeView])
def list_practice(current: User = Depends(get_current_user)) -> list[PracticeView]:
    _student(current)
    return [_view(p) for p in store().practice_for(current.id)]


@router.get("/my-practice/{practice_id}", response_model=PracticeView)
def get_practice(practice_id: str, current: User = Depends(get_current_user)) -> PracticeView:
    _student(current)
    return _view(_own(practice_id, current))


@router.post("/my-practice/{practice_id}/submit", response_model=PracticeView)
def submit_practice(practice_id: str, req: PracticeSubmit, current: User = Depends(get_current_user)) -> PracticeView:
    """Marked at once; the marks feed the student's learning progress as
    practice. A set is submitted once -- start another to practise again."""
    _student(current)
    p = _own(practice_id, current)
    if p["submitted_at"] is not None:
        raise HTTPException(409, "this set is already marked -- start a new one")
    known = {q["id"] for q in p["questions"]}
    if set(req.answers) - known:
        raise HTTPException(422, "those are not questions of this set")
    _consent(current.school_id, current.id, own=True)
    answers = {k: (v or "")[:5000] for k, v in req.answers.items()}
    evaluations, marks, _ = grade_answers(p["questions"], answers)
    p = store().finish_practice(p["id"], answers=answers, evaluations=evaluations, marks=marks)
    try:
        from ..assessment import pillar_routes
        from ..assessment.evaluate import Evaluation
        from ..assessment.schemas import QuestionSchema
        by_id = {q["id"]: q for q in p["questions"]}
        graded = [(QuestionSchema.model_validate(by_id[e["questionId"]]),
                   Evaluation(question_id=e["questionId"], awarded_marks=e["awardedMarks"], max_marks=e["maxMarks"],
                              verdict=e["verdict"], confidence=e["confidence"], reasoning=e["reasoning"],
                              needs_review=e["needsReview"]))
                  for e in evaluations]
        _, knowledge, _ = pillar_routes._require()
        knowledge.record_evaluations(current.id, graded)
    except Exception:  # noqa: BLE001
        log.warning("practice %s: progress not recorded", p["id"], exc_info=True)
    return _view(p)
