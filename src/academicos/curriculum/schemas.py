"""Wire schemas for curriculum/routes.py -- camelCase over the wire,
matching assessment/schemas.py's Camel convention.
"""
from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict, BaseModel, Field
from pydantic.alias_generators import to_camel


class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BoardResponse(Camel):
    id: str
    name: str
    code: str


class AcademicYearResponse(Camel):
    id: str
    school_id: str
    label: str
    start_date: str
    end_date: str
    status: str


class GradeResponse(Camel):
    id: str
    academic_year_id: str
    number: int
    section: Optional[str] = None


class SubjectResponse(Camel):
    id: str
    grade_id: str
    name: str
    code: Optional[str] = None


class BookResponse(Camel):
    id: str
    subject_id: str
    board_id: str
    title: str
    publisher: Optional[str] = None
    status: str


class UnitResponse(Camel):
    id: str
    canonical_id: str
    book_id: str
    unit_no: str
    name: str
    marks: Optional[int] = None
    seq: int


class ChapterResponse(Camel):
    id: str
    canonical_id: str
    unit_id: str
    name: str
    seq: int


class TopicResponse(Camel):
    id: str
    canonical_id: str
    chapter_id: str
    name: str
    seq: int
    description: Optional[str] = None
    source_type: str
    source_reference: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    model_used: Optional[str] = None
    generation_version: Optional[str] = None


class SubtopicResponse(Camel):
    id: str
    canonical_id: str
    topic_id: str
    name: str
    seq: int
    description: Optional[str] = None
    source_type: str
    source_reference: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    model_used: Optional[str] = None
    generation_version: Optional[str] = None


class TopicWithSubtopicsResponse(TopicResponse):
    subtopics: list[SubtopicResponse] = Field(default_factory=list)


class ExtractRequest(Camel):
    grounding_text: Optional[str] = None


class ExtractionRunResponse(Camel):
    id: str
    school_id: str
    book_id: str
    chapter_id: str
    source_hash: Optional[str] = None
    model: Optional[str] = None
    prompt_version: str
    created_at: str
    status: str


class ProposalResponse(Camel):
    id: str
    run_id: str
    entity_type: str
    proposed_name: str
    proposed_description: Optional[str] = None
    proposed_parent: Optional[str] = None
    sequence: int
    confidence: float
    status: str
    edited_name: Optional[str] = None
    materialized_id: Optional[str] = None


class ExtractionRunWithProposalsResponse(Camel):
    run: ExtractionRunResponse
    proposals: list[ProposalResponse]


class ApproveRunRequest(Camel):
    edits: dict[str, str] = Field(default_factory=dict)
    rejected_proposal_ids: list[str] = Field(default_factory=list)


class ApproveRunResponse(Camel):
    run_id: str
    topics_created: int
    subtopics_created: int
    topic_ids: list[str]
    subtopic_ids: list[str]


class AddTopicRequest(Camel):
    name: str = Field(min_length=1)
    description: Optional[str] = None
    seq: int = 0


class AddSubtopicRequest(Camel):
    name: str = Field(min_length=1)
    description: Optional[str] = None
    seq: int = 0


class RenameRequest(Camel):
    name: str = Field(min_length=1)


class SetSequenceRequest(Camel):
    seq: int


class TagQuestionRequest(Camel):
    question_id: str
    question_text: str = Field(min_length=1)


class TaggedSubtopic(Camel):
    subtopic_id: str
    subtopic_name: str
    method: str
    confidence: float


class TagQuestionResponse(Camel):
    question_id: str
    tagged: list[TaggedSubtopic]


class QuestionsForSubtopicsRequest(Camel):
    subtopic_ids: list[str] = Field(min_length=1)


class QuestionsForSubtopicsResponse(Camel):
    question_ids: list[str]


class SeedCbse10Request(Camel):
    academic_year_label: str = Field(min_length=1)
    start_date: str
    end_date: str


class SeedCbse10Response(Camel):
    board_id: str
    academic_year_id: str
    grade_id: str
    subjects_seeded: int
    units_seeded: int
    chapters_seeded: int
    subjects_skipped: list[str]


# ---------------- academic calendar (§10, §27-29) ----------------

class CreateCalendarRequest(Camel):
    """Body of both POST (create) and PUT (correct) .../calendar. The routes
    validate the values (calendar.normalize_weekly_off_days /
    normalize_alternate_saturday_rule) and answer 422 naming what is
    accepted -- plain `str` here so that message, not pydantic's, is what
    the web admin shows."""
    # Full weekday names, any case ("sunday"). The web client sent ints
    # ([7]) until 2026-09-22, which 422'd every create.
    weekly_off_days: list[str] = Field(default_factory=lambda: ["sunday"])
    # "none" | "all" | "second_fourth" | "first_third" | ordinals ("2nd,4th").
    alternate_saturday_rule: str = "none"


class CalendarResponse(Camel):
    id: str
    academic_year_id: str
    weekly_off_days: list[str]
    alternate_saturday_rule: str


class AddHolidayRequest(Camel):
    date: str
    label: str = Field(min_length=1)
    # calendar.HOLIDAY_KINDS -- every kind but "event" removes teaching days.
    kind: str = "holiday"
    # Set for a real multi-day block (a 30-45 day summer break) instead of
    # entering one row per date; omitted/None means a single-day holiday.
    end_date: Optional[str] = None


class HolidayResponse(Camel):
    id: str
    calendar_id: str
    date: str
    label: str
    kind: str
    end_date: Optional[str] = None


class SetPeriodConfigurationRequest(Camel):
    period_minutes: int = Field(gt=0)


class PeriodConfigurationResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    period_minutes: int


class WorkingDaysResponse(Camel):
    academic_year_id: str
    total_days: int
    working_days: int
    weekly_off_count: int
    alternate_saturday_off_count: int
    holiday_count: int
    dates: list[str]


class SetSubjectPeriodAllocationRequest(Camel):
    subject: str = Field(min_length=1)
    periods_per_week: int = Field(gt=0)


class SubjectPeriodAllocationResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    subject: str
    periods_per_week: int


class AddTimetableSlotRequest(Camel):
    subject: str = Field(min_length=1)
    day_of_week: int = Field(ge=0, le=6)
    period_number: int = Field(gt=0)


class TimetableSlotResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    subject: str
    day_of_week: int
    period_number: int


class ComputeTeachingTimeRequest(Camel):
    # Optional: when omitted, the route resolves this book's subject's
    # persisted SubjectPeriodAllocation for the year instead -- an explicit
    # value here still always wins (a one-off override), see
    # compute_teaching_time_estimates route's docstring.
    periods_per_week: Optional[int] = Field(default=None, gt=0)


class TeachingTimeEstimateResponse(Camel):
    id: str
    subtopic_id: str
    academic_year_id: str
    estimated_minutes: int
    estimated_periods: Optional[int] = None
    method: str
    approved_by: Optional[str] = None


class ComputeTeachingTimeResponse(Camel):
    academic_year_id: str
    book_id: str
    periods_per_week: int
    period_minutes: int
    calendar_weeks: int
    total_subject_periods: int
    total_instructional_minutes: int
    units_skipped_no_subtopics: list[str]
    estimates_created: int


# ---------------- micro scheduling (§11-14, §38-41) ----------------

class ScheduleBookRequest(Camel):
    periods_per_week: int = Field(gt=0)
    force: bool = False


class ScheduleBookResponse(Camel):
    academic_year_id: str
    book_id: str
    periods_per_week: int
    teaching_days_available: int
    lessons_created: int
    subtopics_scheduled: int
    subtopics_partially_scheduled: list[str]
    subtopics_unscheduled: list[str]
    subtopics_without_estimate: list[str]
    first_scheduled_date: Optional[str] = None
    last_scheduled_date: Optional[str] = None


class ScheduledLessonResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    book_id: str
    subtopic_id: str
    date: str
    status: str
    note: Optional[str] = None
    completed_by: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str


# ---------------- completion tracking (§15) ----------------

class MarkLessonRequest(Camel):
    status: str = Field(pattern="^(scheduled|completed|skipped)$")
    note: Optional[str] = None


# ---------------- rescheduling on disruption (§14) ----------------

class AdjustLessonRequest(Camel):
    new_date: str
    reason: str = Field(min_length=1)


class PushScheduleRequest(Camel):
    from_date: str
    # Optional: omitted, PUSH uses the subject's stored
    # SubjectPeriodAllocation (scheduling.push_lessons_after).
    periods_per_week: Optional[int] = Field(default=None, gt=0)
    reason: str = Field(min_length=1)


class RescheduleResultResponse(Camel):
    lesson_id: str
    subtopic_id: str
    old_date: str
    new_date: str
    reason: str
    mode: str


class PushScheduleResponse(Camel):
    academic_year_id: str
    book_id: str
    from_date: str
    periods_per_week: int
    lessons_pushed: int
    lessons_dropped: list[str]
    reschedules: list[RescheduleResultResponse]


class RescheduleHistoryEntryResponse(Camel):
    timestamp: str
    actor: Optional[str] = None
    mode: str
    old_date: str
    new_date: str
    reason: str


# ---------------- student visibility (§18) ----------------
# Deliberately a reduced, read-only subset -- no note, no completed_by (a
# teacher's private note isn't necessarily meant for a student to read),
# no write endpoints of any kind. "No management controls" per the
# transcript.

class EnrollStudentRequest(Camel):
    student_id: str = Field(min_length=1)
    grade_id: str = Field(min_length=1)


class StudentEnrollmentResponse(Camel):
    id: str
    school_id: str
    student_id: str
    grade_id: str
    created_at: str


class MyClassScheduleEntryResponse(Camel):
    date: str
    status: str
    subject_name: str
    chapter_name: str
    topic_name: str
    subtopic_name: str


class SubjectProgressResponse(Camel):
    subject_name: str
    scheduled_count: int
    completed_count: int
    skipped_count: int
    total_count: int


class MyProgressResponse(Camel):
    academic_year_id: str
    as_of_date: str
    subjects: list[SubjectProgressResponse]


# ---------------- teacher assignments + "what do I teach today" (§15) ----------------

class AssignTeacherRequest(Camel):
    teacher_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)


class TeacherAssignmentResponse(Camel):
    id: str
    school_id: str
    teacher_id: str
    book_id: str
    created_at: str


class MyScheduleEntryResponse(Camel):
    lesson_id: str
    date: str
    status: str
    note: Optional[str] = None
    book_id: str
    book_title: str
    subject_name: str
    chapter_name: str
    topic_name: str
    subtopic_id: str
    subtopic_name: str


# ---------------- management reporting & variance (§17, §32) ----------------

class ChapterCoverageResponse(Camel):
    chapter_id: str
    chapter_name: str
    total_lessons: int
    completed_lessons: int
    skipped_lessons: int
    coverage_pct: float


class SubjectCoverageResponse(Camel):
    subject_id: str
    subject_name: str
    grade_number: int
    book_id: str
    book_title: str
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    total_lessons: int
    completed_lessons: int
    skipped_lessons: int
    planned_to_date: int
    completed_to_date: int
    coverage_pct: float
    pace_pct: float
    variance: int
    chapters: list[ChapterCoverageResponse] = Field(default_factory=list)


class CoverageReportResponse(Camel):
    school_id: str
    academic_year_id: str
    as_of_date: str
    total_lessons: int
    completed_lessons: int
    skipped_lessons: int
    planned_to_date: int
    completed_to_date: int
    overall_coverage_pct: float
    overall_pace_pct: float
    overall_variance: int
    subjects: list[SubjectCoverageResponse]


class DelayedLessonResponse(Camel):
    lesson_id: str
    scheduled_date: str
    days_overdue: int
    grade_number: int
    subject_name: str
    chapter_name: str
    topic_name: str
    subtopic_name: str
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None


class DelayedTopicsReportResponse(Camel):
    school_id: str
    academic_year_id: str
    as_of_date: str
    delayed_count: int
    delayed_lessons: list[DelayedLessonResponse]


class IngestTocRequest(Camel):
    toc_text: str = Field(..., min_length=1, description="Raw text of the textbook Table of Contents")


class IngestTocResponse(Camel):
    book_id: str
    units_created: int
    chapters_created: int
    unit_ids: list[str]
    chapter_ids: list[str]
