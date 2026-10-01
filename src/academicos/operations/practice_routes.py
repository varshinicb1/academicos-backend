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


@router.post("/my-practice", response_model=PracticeView)
def start_practice(req: PracticeRequest, current: User = Depends(get_current_user)) -> PracticeView:
    """A practice set from the caller's weakest chapters (or the ones named),
    skipping questions answered in the last two weeks."""
    grade = _student(current)
    s = store()
    subject, chapters = req.subject, list(req.chapter_ids)
    if not chapters:
        revise = [r for r in s.learning_progress(current.id)["revise_next"]
                  if r["grade"] == grade and (subject is None or r["subject"].lower() == subject.lower())]
        if revise:
            subject = subject or revise[0]["subject"]
            chapters = list(dict.fromkeys(r["chapter_id"] for r in revise if r["subject"] == subject))[:3]
    if subject is None:
        raise HTTPException(409, "choose a subject: nothing is marked yet to say which chapters to revise")
    pool = hr._usable(hr._bank_questions(subject, grade, chapters or None), current)
    if not pool:
        raise HTTPException(422, f"the question bank has no {subject} questions for class {grade} there yet")
    questions = pick(pool, recent=s.recently_answered(current.id), count=req.count,
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
    _consent(current.school_id, current.id)
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
