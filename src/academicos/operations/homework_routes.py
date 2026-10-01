"""HTTP routes for homework (TA-4, SA-2; operations/homework.py).

A teacher sets homework for the sections they teach a subject in -- the
principal for any section -- from bank questions, chosen one by one or picked
from chapters. The suggestion route starts from the lesson the section's plan
has on that day. Students see their own section's homework without answers,
submit, and see their marks once they are final. The teacher sees who has
and has not submitted, marks what the evaluator could not, and reminds the
rest. Every write is audited; submitting and marking a student's work need
the parent's consent (authz.require_consent).
"""
from __future__ import annotations

import logging
import random
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user, require_staff
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .homework import MAX_QUESTIONS, Homework, HomeworkError, Submission
from .routes import notify_safely, store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# Curriculum subject names as schools type them, against the bank's.
_SUBJECT_ALIASES = {"maths": "Mathematics", "math": "Mathematics", "sst": "Social Science",
                    "social studies": "Social Science", "evs": "EVS"}


def bank_subject(name: str) -> str:
    return _SUBJECT_ALIASES.get(name.strip().lower(), name.strip())


def _usable(questions: list[dict], current) -> list[dict]:
    """Bank questions less those the caller's school rejected in its review
    queue (operations/question_reviews.py). Applied here, at the draw, because
    `_bank_questions` serves every school and tests replace it."""
    from .question_reviews import rejected_for
    rejected = rejected_for(current.school_id)
    return [q for q in questions if q["id"] not in rejected] if rejected else questions


def _bank_questions(subject: str, grade: int, chapter_ids: Optional[list[str]] = None) -> list[dict]:
    """Bank questions for a class, as the wire shape the bank serves
    (camelCase QuestionSchema dumps), optionally only those the chapter view
    files under `chapter_ids`. Tests replace this."""
    from ..assessment.mapping import to_question_schema
    from ..assessment.pool import get_pool
    pool = get_pool(cr._cfg, subject=bank_subject(subject), grade=str(grade))
    qs = pool.filter(chapter_ids=chapter_ids) if chapter_ids else pool.questions
    return [to_question_schema(q).model_dump(by_alias=True, mode="json") for q in qs]


def _bank_chapters(subject: str, grade: int) -> list[dict]:
    """The bank's chapters for a class with their question counts, the same
    list the paper builder's chapter picker shows. Tests replace this."""
    from ..assessment.pillar_routes import catalog_chapters
    try:
        entries = catalog_chapters(bank_subject(subject), grade, Response())
    except HTTPException:
        return []
    return [{"chapter_id": e.chapter_id, "chapter_name": e.chapter_name, "question_count": e.question_count}
            for e in entries if e.chapter_id != "unmapped"]


def _name_key(name: str) -> str:
    from ..assessment.pillar_routes import _name_key as key
    return key(name)


def _audit(action: str, user: User, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(action, actor=user.id,
                                            details={"schoolId": user.school_id, **details})


def _log_read(current: User, what: str, **kw: Any) -> None:
    """A staff read of named students' work is logged (docs/compliance.md box 2)."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what=what, **kw)


def _consent(school_id: str, student_id: str, *, own: bool = False) -> None:
    """`own`: the student (or their parent) is the caller, and reads a
    refusal worded for them, not the staff wording (v3 audit N-5-1)."""
    from ..assessment.authz import require_consent, require_own_consent
    from ..assessment.consent import get_consent_store
    (require_own_consent if own else require_consent)(get_consent_store(cr._cfg.data_root), school_id, student_id)


def _today() -> str:
    return cr._school_today().isoformat()


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------- shapes ----------------

class HomeworkCreateRequest(_Req):
    subject_id: str
    section_ids: list[str] = Field(min_length=1, max_length=20)
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(default="", max_length=2000)
    due_date: str
    question_ids: list[str] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    chapter_ids: list[str] = Field(default_factory=list, max_length=20)
    count: int = Field(default=5, ge=1, le=MAX_QUESTIONS)   # with chapter_ids and no question_ids
    # With chapter_ids: only these question types (mcq, very_short_answer,
    # short_answer, long_answer) and difficulties (easy, medium, hard).
    types: list[str] = Field(default_factory=list, max_length=4)
    difficulties: list[str] = Field(default_factory=list, max_length=3)
    publish: bool = True


class GradeRequest(_Req):
    marks: dict[str, float] = Field(default_factory=dict)
    feedback: Optional[str] = Field(default=None, max_length=2000)


class SubmitRequest(_Req):
    answers: dict[str, str]


class QuestionPreview(Camel):
    id: str
    type: str
    stem: str
    marks: int


class BankChapter(Camel):
    chapter_id: str
    chapter_name: str
    question_count: int


class SuggestionResponse(Camel):
    date: str
    lesson_found: bool
    chapter_name: Optional[str] = None
    topic_name: Optional[str] = None
    subtopic_name: Optional[str] = None
    bank_chapters: list[BankChapter]
    questions: list[QuestionPreview]
    note: Optional[str] = None


class HomeworkSummary(Camel):
    id: str
    title: str
    subject_id: str
    subject_name: str
    grade: int
    section_ids: list[str]
    section_names: list[str]
    teacher_id: str
    due_date: str
    status: str
    question_count: int
    total_marks: int
    submitted: int = 0
    graded: int = 0
    students: int = 0


class HomeworkDetail(HomeworkSummary):
    instructions: str
    chapter_ids: list[str]
    questions: list[dict]


class EvaluationItem(Camel):
    question_id: str
    awarded_marks: float
    max_marks: int
    verdict: str
    reasoning: str
    needs_review: bool
    model_answer: Optional[str] = None


class SubmissionRow(Camel):
    photo_ids: list[str] = Field(default_factory=list)
    student_id: str
    student_name: str
    section_id: str
    status: str                    # not_submitted | submitted | graded
    late: bool = False
    attempts: int = 0
    submitted_at: Optional[str] = None
    auto_marks: Optional[float] = None
    marks: Optional[float] = None
    max_marks: int
    needs_review: int = 0
    answers: dict[str, str] = Field(default_factory=dict)
    evaluations: list[EvaluationItem] = Field(default_factory=list)
    feedback: Optional[str] = None


class StudentQuestion(Camel):
    id: str
    type: str
    stem: str
    marks: int
    parts: list[dict] = Field(default_factory=list)


class MyHomeworkItem(Camel):
    id: str
    title: str
    subject_name: str
    due_date: str
    status: str                    # the homework's: published | closed
    my_status: str                 # not_submitted | submitted | graded
    overdue: bool
    marks: Optional[float] = None
    total_marks: int


class MyHomeworkDetail(MyHomeworkItem):
    instructions: str
    questions: list[StudentQuestion]
    answers: dict[str, str] = Field(default_factory=dict)
    evaluations: list[EvaluationItem] = Field(default_factory=list)
    feedback: Optional[str] = None


# ---------------- helpers ----------------

def _manages(hw: Homework, user: User) -> bool:
    if user.school_id != hw.school_id or user.role not in ("teacher", "principal"):
        return False
    if user.role == "principal" or hw.teacher_id == user.id:
        return True
    cs = cr._require()
    for sid in hw.section_ids:
        a = cs.allocation_for(sid, hw.subject_id)
        if a is not None and a.teacher_id == user.id:
            return True
    return False


def _require_managed(homework_id: str, user: User) -> Homework:
    hw = store().get_homework(homework_id)
    if hw is None:
        raise HTTPException(404, "homework not found")
    if hw.school_id != user.school_id:
        raise HTTPException(403, "this homework belongs to a different school")
    if not _manages(hw, user):
        raise HTTPException(403, "only the teacher of this subject in these sections, or the principal")
    return hw


def _students(hw: Homework) -> list[tuple[str, str]]:
    """(student id, section id) for every student enrolled in its sections."""
    cs = cr._require()
    return [(e.student_id, sid) for sid in hw.section_ids for e in cs.enrollments_for_section(sid)]


def _summary(hw: Homework, cls=HomeworkSummary, **extra) -> Any:
    cs = cr._require()
    names = []
    for sid in hw.section_ids:
        s = cs.get_section(sid)
        names.append(cs._section_label(s) if s else sid)
    counts = store().homework_summary(hw)
    return cls(id=hw.id, title=hw.title, subject_id=hw.subject_id, subject_name=hw.subject_name, grade=hw.grade,
               section_ids=hw.section_ids, section_names=names, teacher_id=hw.teacher_id, due_date=hw.due_date,
               status=hw.status, question_count=len(hw.questions), total_marks=hw.total_marks,
               submitted=counts["submitted"] + counts["graded"], graded=counts["graded"],
               students=len(_students(hw)), **extra)


def _preview(q: dict) -> QuestionPreview:
    return QuestionPreview(id=q["id"], type=q.get("type", ""), stem=q.get("stem", ""), marks=int(q.get("marks") or 0))


def _evaluations(hw: Homework, sub: Submission, *, with_answers: bool) -> list[EvaluationItem]:
    models = {q["id"]: (q.get("answerScheme") or {}).get("modelAnswer") or None for q in hw.questions}
    return [EvaluationItem(question_id=e["questionId"], awarded_marks=e["awardedMarks"], max_marks=e["maxMarks"],
                           verdict=e["verdict"], reasoning=e["reasoning"], needs_review=e["needsReview"],
                           model_answer=models.get(e["questionId"]) if with_answers else None)
            for e in sub.evaluations]


def _student_question(q: dict) -> StudentQuestion:
    """What a student may see before marking: never the scheme, the correct
    option, or an expected answer."""
    parts = [{"id": p.get("id"), "partNumber": p.get("partNumber"), "text": p.get("text", ""),
              "marks": p.get("marks"), "options": p.get("options")} for p in q.get("parts") or []]
    return StudentQuestion(id=q["id"], type=q.get("type", ""), stem=q.get("stem", ""),
                           marks=int(q.get("marks") or 0), parts=parts)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"{value!r} is not a date (YYYY-MM-DD)")


def _record_mastery(hw: Homework, sub: Submission) -> None:
    """Final marks feed the student's mastery and learning progress, as a
    finalized answer sheet does; a re-mark replaces the homework's earlier
    copy (KnowledgeStore.record_sheet). Marks are already saved, so a failure
    here is logged, not raised."""
    from ..assessment import pillar_routes
    from ..assessment.evaluate import Evaluation
    from ..assessment.schemas import QuestionSchema
    by_id = {q["id"]: q for q in hw.questions}
    graded = [(QuestionSchema.model_validate(by_id[e["questionId"]]),
               Evaluation(question_id=e["questionId"], awarded_marks=e["awardedMarks"], max_marks=e["maxMarks"],
                          verdict=e["verdict"], confidence=e["confidence"], reasoning=e["reasoning"],
                          needs_review=False))
              for e in sub.evaluations if e["questionId"] in by_id]
    try:
        _, knowledge, _ = pillar_routes._require()
        knowledge.record_sheet(sub.student_id, graded, f"homework:{hw.id}")
    except Exception:  # noqa: BLE001
        logger.warning("homework %s: mastery not recorded for %s", hw.id, sub.student_id, exc_info=True)


def _notify_graded(hw: Homework, sub: Submission) -> None:
    _record_mastery(hw, sub)
    marks = sub.marks if sub.marks is not None else sub.auto_marks
    parents = [g.parent_id for g in store().guardians_of(sub.student_id)]
    notify_safely(school_id=hw.school_id, user_ids=[sub.student_id, *parents], kind="homework_graded",
                  params={"subject": hw.subject_name, "title": hw.title,
                          "marks": f"{marks:g} out of {sub.max_marks}"},
                  link=f"/my-homework/{hw.id}", dedupe_key=f"hwgraded:{hw.id}:{sub.student_id}:{sub.graded_at}")


def _notify_submitted(hw: Homework, student: User) -> None:
    """The teacher hears that work is coming in: once a day per homework,
    with the count so far, not once per student (TA-7)."""
    if not hw.teacher_id:
        return
    handed_in = len(store().submissions_for(hw.id))
    notify_safely(school_id=hw.school_id, user_ids=[hw.teacher_id], kind="homework_submitted",
                  params={"title": hw.title, "student": student.name, "count": handed_in,
                          "total": len(_students(hw))},
                  link=f"/homework/{hw.id}", dedupe_key=f"hwsubmitted:{hw.id}:{_today()}")


def _notify_assigned(hw: Homework) -> None:
    students = [sid for sid, _ in _students(hw)]
    if students:
        notify_safely(school_id=hw.school_id, user_ids=students, kind="homework_assigned",
                      params={"subject": hw.subject_name, "title": hw.title, "due": hw.due_date},
                      link=f"/my-homework/{hw.id}", dedupe_key=f"hwassigned:{hw.id}")


# ---------------- teacher / principal ----------------

@router.get("/homework/suggestion", response_model=SuggestionResponse)
def suggest(section_id: str = Query(alias="sectionId"), subject_id: str = Query(alias="subjectId"),
            on: Optional[str] = Query(default=None, alias="date"),
            current: User = Depends(require_staff)) -> SuggestionResponse:
    """What to set after the lesson the section's plan has on `date` (today
    by default, else the latest lesson before it): the bank chapters that
    match that lesson's chapter by name, and up to ten of their questions.
    The school's book and the bank's chapters are not always one edition;
    when no name matches, the note says so and the teacher picks chapters."""
    cs = cr._require()
    section = cr._require_school_owns_section(section_id, current)
    cr._require_school_owns_subject(subject_id, current)
    subject = cs.get_subject(subject_id)
    if subject.grade_id != section.grade_id:
        raise HTTPException(422, "that subject is not taught in that section's class")
    day = (_parse_date(on) if on else cr._school_today()).isoformat()
    grade = cs.get_grade(subject.grade_id)
    chapters = [BankChapter(**c) for c in _bank_chapters(subject.name, grade.number)]

    lesson = None
    book = cs.selected_book_for_subject(subject_id)
    if book is not None:
        lessons = (cs.scheduled_lessons_for_book(section.academic_year_id, book.id, section_id)
                   or cs.scheduled_lessons_for_book(section.academic_year_id, book.id, None))
        past = [l for l in lessons if l.date <= day and l.status != "unscheduled"]
        lesson = past[-1] if past else None
    if lesson is None:
        return SuggestionResponse(date=day, lesson_found=False, bank_chapters=chapters, questions=[],
                                  note="No lesson is planned for this section on or before this day; "
                                       "pick chapters from the list.")
    subtopic = cs.get_subtopic(lesson.subtopic_id)
    topic = cs.get_topic(subtopic.topic_id) if subtopic else None
    chapter = cs.get_chapter(topic.chapter_id) if topic else None
    matched = [c for c in chapters if chapter and _name_key(c.chapter_name) == _name_key(chapter.name)]
    questions = (_usable(_bank_questions(subject.name, grade.number, [c.chapter_id for c in matched]), current)
                 if matched else [])
    note = None if matched else (
        f"The question bank has no chapter named '{chapter.name if chapter else ''}' for this class -- the "
        "school's book and the bank may be different editions. Pick chapters from the list.")
    return SuggestionResponse(date=lesson.date, lesson_found=True, chapter_name=chapter.name if chapter else None,
                              topic_name=topic.name if topic else None,
                              subtopic_name=subtopic.name if subtopic else None,
                              bank_chapters=matched or chapters, questions=[_preview(q) for q in questions[:10]],
                              note=note)


@router.post("/homework", response_model=HomeworkDetail)
def create_homework(req: HomeworkCreateRequest, current: User = Depends(require_staff)) -> HomeworkDetail:
    """Set homework. A teacher only for sections they teach this subject in;
    the principal for any section of the school. Questions are the ones named,
    or `count` picked from the chapters named. Published at once unless
    `publish` is false, and then the students are told."""
    cs = cr._require()
    cr._require_school_owns_subject(req.subject_id, current)
    subject = cs.get_subject(req.subject_id)
    grade = cs.get_grade(subject.grade_id)
    for sid in req.section_ids:
        section = cr._require_school_owns_section(sid, current)
        if section.grade_id != subject.grade_id:
            raise HTTPException(422, f"{cs._section_label(section)} is not a class that studies this subject")
        if current.role == "teacher":
            a = cs.allocation_for(sid, req.subject_id)
            if a is None or a.teacher_id != current.id:
                raise HTTPException(403, f"you do not teach {subject.name} in {cs._section_label(section)}")
    due = _parse_date(req.due_date)
    if due.isoformat() < _today():
        raise HTTPException(422, "the due date has already passed")

    if req.question_ids:
        whole = {q["id"]: q for q in _bank_questions(subject.name, grade.number)}
        bank = {q["id"]: q for q in _usable(list(whole.values()), current)}
        refused = [qid for qid in req.question_ids if qid in whole and qid not in bank]
        if refused:
            raise HTTPException(422, "your school's reviewers rejected the answer to "
                                     + ", ".join(refused[:5]) + "; choose another question")
        missing = [qid for qid in req.question_ids if qid not in bank]
        if missing:
            raise HTTPException(422, f"not in the question bank for class {grade.number} {subject.name}: "
                                     + ", ".join(missing[:5]))
        questions = [bank[qid] for qid in dict.fromkeys(req.question_ids)]
    elif req.chapter_ids:
        pool = _usable(_bank_questions(subject.name, grade.number, req.chapter_ids), current)
        if req.types:
            pool = [q for q in pool if q.get("type") in set(req.types)]
        if req.difficulties:
            pool = [q for q in pool if q.get("difficulty") in set(req.difficulties)]
        if not pool:
            raise HTTPException(422, "the question bank has no questions of that kind in those chapters")
        # Questions these sections have already had come last (API-2, TA-4):
        # a repeat is used only when the chapters have nothing new left.
        had = {q["id"] for hw in store().homework_for_school(current.school_id)
               if set(hw.section_ids) & set(req.section_ids) for q in hw.questions}
        rng = random.Random(f"{req.subject_id}:{req.due_date}:{req.title}")
        fresh = [q for q in pool if q["id"] not in had]
        old = [q for q in pool if q["id"] in had]
        rng.shuffle(fresh)
        rng.shuffle(old)
        questions = sorted((fresh + old)[:req.count], key=lambda q: (q.get("marks") or 0, q["id"]))
    else:
        raise HTTPException(422, "name the questions, or the chapters to pick them from")

    hw = store().create_homework(
        school_id=current.school_id, academic_year_id=grade.academic_year_id, teacher_id=current.id,
        subject_id=subject.id, subject_name=subject.name, grade=grade.number, section_ids=req.section_ids,
        title=req.title.strip(), instructions=req.instructions.strip(), chapter_ids=req.chapter_ids,
        questions=questions, due_date=due.isoformat(), created_by=current.id, publish=req.publish)
    _audit("homework_created", current, {"homeworkId": hw.id, "sectionIds": hw.section_ids,
                                         "questionIds": [q["id"] for q in hw.questions], "status": hw.status})
    if hw.status == "published":
        _notify_assigned(hw)
    return _summary(hw, HomeworkDetail, instructions=hw.instructions, chapter_ids=hw.chapter_ids,
                    questions=hw.questions)


@router.get("/homework", response_model=list[HomeworkSummary])
def list_homework(status: Optional[str] = None, section_id: Optional[str] = Query(default=None, alias="sectionId"),
                  current: User = Depends(require_staff)) -> list[HomeworkSummary]:
    """The principal sees the school's homework; a teacher what they set or
    what is set in a subject and section they teach."""
    items = store().homework_for_school(current.school_id, status=status)
    return [_summary(h) for h in items
            if _manages(h, current) and (section_id is None or section_id in h.section_ids)]


@router.get("/homework/{homework_id}", response_model=HomeworkDetail)
def get_homework(homework_id: str, current: User = Depends(require_staff)) -> HomeworkDetail:
    hw = _require_managed(homework_id, current)
    return _summary(hw, HomeworkDetail, instructions=hw.instructions, chapter_ids=hw.chapter_ids,
                    questions=hw.questions)


def _move(homework_id: str, status: str, current: User) -> Homework:
    hw = _require_managed(homework_id, current)
    try:
        after = store().set_homework_status(hw.id, status)
    except HomeworkError as e:
        raise HTTPException(409, str(e))
    _audit(f"homework_{status}", current, {"homeworkId": hw.id, "before": hw.status, "after": after.status})
    return after


@router.post("/homework/{homework_id}/publish", response_model=HomeworkSummary)
def publish_homework(homework_id: str, current: User = Depends(require_staff)) -> HomeworkSummary:
    before = _require_managed(homework_id, current).status
    hw = _move(homework_id, "published", current)
    if before == "draft":
        _notify_assigned(hw)
    return _summary(hw)


@router.post("/homework/{homework_id}/close", response_model=HomeworkSummary)
def close_homework(homework_id: str, current: User = Depends(require_staff)) -> HomeworkSummary:
    """No more submissions. Marking what has come in goes on."""
    return _summary(_move(homework_id, "closed", current))


@router.get("/homework/{homework_id}/submissions", response_model=list[SubmissionRow])
def list_submissions(homework_id: str, current: User = Depends(require_staff)) -> list[SubmissionRow]:
    """Every student of the sections, submitted or not, with their answers
    and the evaluator's marks per question."""
    hw = _require_managed(homework_id, current)
    subs = {s.student_id: s for s in store().submissions_for(hw.id)}
    _log_read(current, "homework_submissions", student_ids=[sid for sid, _ in _students(hw)], homework_id=hw.id)
    users = cr._require_users()
    rows = []
    for sid, section_id in _students(hw):
        u = users.get(sid)
        s = subs.get(sid)
        if s is None:
            rows.append(SubmissionRow(student_id=sid, student_name=getattr(u, "name", "") or sid,
                                      section_id=section_id, status="not_submitted", max_marks=hw.total_marks))
            continue
        rows.append(SubmissionRow(
            student_id=sid, student_name=getattr(u, "name", "") or sid, section_id=s.section_id, status=s.status,
            late=s.late, attempts=s.attempts, submitted_at=s.submitted_at, auto_marks=s.auto_marks, marks=s.marks,
            max_marks=s.max_marks, needs_review=sum(1 for e in s.evaluations if e["needsReview"]),
            answers=s.answers, evaluations=_evaluations(hw, s, with_answers=True), feedback=s.feedback,
            photo_ids=[p["id"] for p in store().photos_for(hw.id, sid)]))
    return rows


@router.post("/homework/{homework_id}/submissions/{student_id}/grade", response_model=SubmissionRow)
def grade_submission(homework_id: str, student_id: str, req: GradeRequest,
                     current: User = Depends(require_staff)) -> SubmissionRow:
    """Final marks for one student: the evaluator's award for every question
    not named, the teacher's for each one named. The student is told."""
    hw = _require_managed(homework_id, current)
    _consent(hw.school_id, student_id)
    before = store().get_submission(hw.id, student_id)
    try:
        sub = store().grade_submission(hw, student_id, marks=req.marks, feedback=req.feedback, graded_by=current.id)
    except HomeworkError as e:
        raise HTTPException(404 if before is None else 422, str(e))
    _audit("homework_graded", current, {"homeworkId": hw.id, "studentId": student_id,
                                        "before": before.marks if before else None, "after": sub.marks,
                                        "overrides": req.marks})
    _notify_graded(hw, sub)
    u = cr._require_users().get(student_id)
    return SubmissionRow(student_id=student_id, student_name=getattr(u, "name", "") or student_id,
                         section_id=sub.section_id, status=sub.status, late=sub.late, attempts=sub.attempts,
                         submitted_at=sub.submitted_at, auto_marks=sub.auto_marks, marks=sub.marks,
                         max_marks=sub.max_marks, answers=sub.answers,
                         evaluations=_evaluations(hw, sub, with_answers=True), feedback=sub.feedback)


@router.post("/homework/{homework_id}/remind")
def remind(homework_id: str, current: User = Depends(require_staff)) -> dict:
    """Tell every student who has not submitted. Once a day at most per
    student, however often it is pressed."""
    hw = _require_managed(homework_id, current)
    if hw.status != "published":
        raise HTTPException(409, "only homework that is open for submission can be reminded about")
    done = {s.student_id for s in store().submissions_for(hw.id)}
    waiting = [sid for sid, _ in _students(hw) if sid not in done]
    today = _today()
    for sid in waiting:
        notify_safely(school_id=hw.school_id, user_ids=[sid], kind="homework_due",
                      params={"title": hw.title, "due": hw.due_date}, link=f"/my-homework/{hw.id}",
                      dedupe_key=f"hwdue:{hw.id}:{sid}:{today}")
    return {"reminded": len(waiting)}


# ---------------- students ----------------

def _enrollment(current: User):
    return cr._require_student_enrollment(cr._require(), current)


def _my_homework(homework_id: str, current: User) -> tuple[Homework, Any]:
    enrollment = _enrollment(current)
    hw = store().get_homework(homework_id)
    if hw is None or hw.status == "draft" or enrollment.section_id not in hw.section_ids:
        raise HTTPException(404, "homework not found")
    return hw, enrollment


def _my_item(hw: Homework, sub: Optional[Submission], cls=MyHomeworkItem, **extra) -> Any:
    graded = sub is not None and sub.status == "graded"
    return cls(id=hw.id, title=hw.title, subject_name=hw.subject_name, due_date=hw.due_date, status=hw.status,
               my_status=sub.status if sub else "not_submitted",
               overdue=sub is None and hw.status == "published" and hw.due_date < _today(),
               marks=sub.marks if graded else None, total_marks=hw.total_marks, **extra)


@router.get("/my-homework", response_model=list[MyHomeworkItem])
def my_homework(current: User = Depends(get_current_user)) -> list[MyHomeworkItem]:
    """The caller's section's homework, newest due date first, with where
    they stand on each."""
    enrollment = _enrollment(current)
    s = store()
    return [_my_item(hw, s.get_submission(hw.id, current.id)) for hw in s.homework_for_section(enrollment.section_id)]


@router.get("/my-homework/{homework_id}", response_model=MyHomeworkDetail)
def my_homework_detail(homework_id: str, current: User = Depends(get_current_user)) -> MyHomeworkDetail:
    """The questions without answers; after marking, the marks per question
    and the model answers."""
    hw, _ = _my_homework(homework_id, current)
    sub = store().get_submission(hw.id, current.id)
    graded = sub is not None and sub.status == "graded"
    return _my_item(hw, sub, MyHomeworkDetail, instructions=hw.instructions,
                    questions=[_student_question(q) for q in hw.questions],
                    answers=sub.answers if sub else {},
                    evaluations=_evaluations(hw, sub, with_answers=True) if graded else [],
                    feedback=sub.feedback if graded else None)


@router.post("/my-homework/{homework_id}/submit", response_model=MyHomeworkDetail)
def submit(homework_id: str, req: SubmitRequest, current: User = Depends(get_current_user)) -> MyHomeworkDetail:
    """Send answers ({questionId: answer}; an MCQ answer is its option
    letter). May be sent again until it is marked or the homework closes.
    After the due date it is accepted and marked late."""
    hw, enrollment = _my_homework(homework_id, current)
    stray = sorted(set(req.answers) - {q["id"] for q in hw.questions})
    if stray:
        raise HTTPException(422, "these are not questions of this homework: " + ", ".join(stray[:5]))
    _consent(hw.school_id, current.id, own=True)
    try:
        sub = store().submit_homework(hw, student_id=current.id, section_id=enrollment.section_id,
                                      answers={k: (v or "")[:5000] for k, v in req.answers.items()},
                                      late=_today() > hw.due_date,
                                      needs_teacher=bool(store().photos_for(hw.id, current.id)))
    except HomeworkError as e:
        raise HTTPException(409, str(e))
    if sub.status == "graded":
        _notify_graded(hw, sub)
    _notify_submitted(hw, current)
    return my_homework_detail(homework_id, current)
