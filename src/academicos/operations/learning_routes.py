"""HTTP routes for learning progress (SA-3; operations/learning.py): a
student's own, and a named student's for their school's staff. Parents read
their child's through the same shape once guardians are linked (SA-6)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..assessment.auth_routes import get_current_user
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .routes import store

router = APIRouter(prefix="/api/v1")


class TopicProgress(Camel):
    topic_id: str
    topic_name: str
    mastery: Optional[float] = None
    accuracy: Optional[float] = None
    evidence_count: int
    last_seen: Optional[str] = None
    status: str


class ChapterProgress(Camel):
    chapter_id: str
    chapter_name: str
    mastery: Optional[float] = None
    accuracy: Optional[float] = None
    evidence_count: int
    last_seen: Optional[str] = None
    status: str                    # not_started | early | needs_work | developing | mastered
    topics: list[TopicProgress]


class SubjectProgress(Camel):
    subject: str
    grade: int
    mastery: Optional[float] = None
    accuracy: Optional[float] = None
    evidence_count: int
    last_seen: Optional[str] = None
    status: str
    chapters: list[ChapterProgress]


class ReviseItem(Camel):
    subject: str
    grade: int
    chapter_id: str
    chapter_name: str
    topic_id: Optional[str] = None
    topic_name: Optional[str] = None
    mastery: float
    last_seen: str
    reason: str


class WeekProgress(Camel):
    week_start: str
    answered: int
    average: Optional[float] = None


class LearningProgressResponse(Camel):
    student_id: str
    as_of: str
    subjects: list[SubjectProgress]
    revise_next: list[ReviseItem]
    weekly: list[WeekProgress]
    sources: dict[str, int]


def _term_start(student_id: str) -> Optional[date]:
    """The start of the student's academic year, when they are enrolled."""
    cs = cr._require()
    e = cs.enrollment_for_student(student_id)
    section = cs.get_section(e.section_id) if e and e.section_id else None
    year = cs.get_academic_year(section.academic_year_id) if section else None
    try:
        return date.fromisoformat(year.start_date) if year else None
    except ValueError:
        return None


def progress_for(student_id: str) -> LearningProgressResponse:
    return LearningProgressResponse(**store().learning_progress(student_id, term_start=_term_start(student_id)))


@router.get("/my-learning", response_model=LearningProgressResponse)
def my_learning(current: User = Depends(get_current_user)) -> LearningProgressResponse:
    """The caller's own progress: per subject, chapter and topic, what to
    revise next, and the term week by week. Students only."""
    if current.role != "student":
        raise HTTPException(403, "this endpoint is for students only")
    return progress_for(current.id)


@router.get("/students/{student_id}/learning", response_model=LearningProgressResponse)
def student_learning(student_id: str, current: User = Depends(get_current_user)) -> LearningProgressResponse:
    """A student of the caller's school (a student: only themselves). A
    staff read is logged as a read of student data."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    from ..assessment.authz import require_school_owns_student
    require_school_owns_student(cr._require_users(), student_id, current)
    if current.id != student_id:
        record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what="learning_progress",
                        student_id=student_id)
    return progress_for(student_id)
