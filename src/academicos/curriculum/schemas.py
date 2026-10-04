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


class CamelRequest(Camel):
    """A request body that refuses a field it does not know. The older
    curriculum requests ignore one (audit D7: a misspelled field is dropped
    and the call succeeds); new ones start strict."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SectionResponse(Camel):
    """One class of a grade ("10-B", M1.1). studentCount is how many
    students are enrolled in it now."""
    id: str
    academic_year_id: str
    grade_id: str
    grade_number: int
    name: str
    class_teacher_id: Optional[str] = None
    student_count: int = 0


class CreateSectionRequest(CamelRequest):
    name: str = Field(min_length=1, max_length=20)
    class_teacher_id: Optional[str] = None


class UpdateSectionRequest(CamelRequest):
    """A field left out is unchanged; classTeacherId null clears it."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=20)
    class_teacher_id: Optional[str] = None


class SectionStudentResponse(Camel):
    """A student enrolled in a section. Ids only: the console already
    holds the school's user list, so no name or email travels twice."""
    student_id: str
    enrollment_id: str
    enrolled_at: str


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
    # Whether this is the subject's edition for its year (PRD 12.7). `status`
    # is the book's content state and says nothing about the choice.
    selected: bool = False


class AddBookRequest(Camel):
    title: str = Field(min_length=1)
    # Publisher and/or edition, free text ("NCERT, 2026 reprint").
    publisher: Optional[str] = None
    board_id: str = Field(min_length=1)


class SelectBookRequest(Camel):
    book_id: str = Field(min_length=1)


class OtherEditionResponse(Camel):
    """A book of the subject that is not its edition, with what still
    points at it: this year's scheduled lessons and teacher assignments."""
    book_id: str
    title: str
    scheduled_lessons: int = 0
    teacher_assignments: int = 0


class SelectBookResponse(Camel):
    book: BookResponse
    previous_book_id: Optional[str] = None
    # Lessons of this year scheduled from the edition chosen just before.
    previous_book_scheduled_lessons: int = 0
    # Every other book of the subject, not only the previous one: after
    # A -> B -> C, A's lessons are still there. Nothing listed here is moved
    # to the new book; `warning` says so whenever either total is above 0.
    other_editions: list[OtherEditionResponse] = []
    other_editions_scheduled_lessons: int = 0
    other_editions_teacher_assignments: int = 0
    warning: Optional[str] = None


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
    # How many questions are tagged to this subtopic in question_subtopic_links.
    # POST /questions/search intersects on exactly that table, so 0 here means
    # "picking this subtopic yields an empty paper" -- the picker has to be able
    # to say so before a teacher generates one. None where it was not counted
    # (the routes that return a subtopic on its own do not pay for the query).
    tagged_question_count: Optional[int] = None


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


class SeedGradeRequest(SeedCbse10Request):
    """Requirements v3: grades 1-12 (PRD section 0 decision 4 said 6-12). The bound is pydantic's so
    the answer to grade 0 or 13 is a 422 naming the range, before any row is
    created -- there is no syllabus data outside it (seed_cbse10.MIN_GRADE /
    MAX_GRADE, which raises the same bound for the CLI and library callers)."""
    grade: int = Field(ge=1, le=12)


class SubjectTemplateReportResponse(Camel):
    """How far below chapter level the seed got for one subject, and why it
    stopped where it did. `note` is the honest half: "Class 7 Science is
    chapter-only because the only textbook tree is the 2024-25 NEP edition and
    the syllabus JSON still lists the previous edition's chapter names"."""
    subject: str
    chapters: int
    chapters_with_topics: int
    chapters_without_topics: int
    topics_proposed: int
    subtopics_proposed: int
    provenance: list[str]
    note: Optional[str] = None


class SeedCbse10Response(Camel):
    """Also the response of the any-grade route. The topic-template fields were
    added 2026-09-22 (new fields only -- an older client keeps working): a seed
    that stops at chapters and a seed that reached Topic/Subtopic used to be
    indistinguishable in this body, and on a fresh school it was always the
    former."""
    board_id: str
    academic_year_id: str
    grade_id: str
    grade: int
    subjects_seeded: int
    units_seeded: int
    chapters_seeded: int
    subjects_skipped: list[str]
    # Proposed, pending the principal's approval -- never approved rows.
    chapters_with_topics: int = 0
    chapters_without_topics: int = 0
    topics_proposed: int = 0
    subtopics_proposed: int = 0
    topic_templates: list[SubjectTemplateReportResponse] = Field(default_factory=list)


class ApproveChapterTopicsResponse(Camel):
    """Result of "Approve all for this chapter" -- possibly several runs, since
    a chapter can carry one template run per provenance."""
    chapter_id: str
    runs_approved: int
    topics_created: int
    subtopics_created: int
    topic_ids: list[str]
    subtopic_ids: list[str]


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
    # NTF-3: lessons still to teach on these days move to the next teaching
    # periods, and the teachers and students affected are told. False only
    # records the day (e.g. entering last year's list before any plan exists).
    move_lessons: bool = True
    # A day that runs differently (calendar.DAY_KINDS, v3 audit N-3-19) and
    # its own field: half_day keeps periods 1..lastPeriod; exam_window stops
    # teaching for these classes (grade numbers; none is every class);
    # working_day (an off day made a working day) runs the timetable of
    # timetableWeekday (0 = Monday).
    last_period: Optional[int] = None
    grades: Optional[list[int]] = None
    timetable_weekday: Optional[int] = None


class HolidayResponse(Camel):
    id: str
    calendar_id: str
    date: str
    label: str
    kind: str
    end_date: Optional[str] = None
    last_period: Optional[int] = None
    grades: Optional[list[int]] = None
    timetable_weekday: Optional[int] = None
    # On the add only: lessons moved off the holiday, and the plans that
    # could not be moved (no cadence to reflow them on), named.
    lessons_moved: Optional[int] = None
    not_moved: Optional[list[str]] = None
    # Lessons the change pushed past the year's last teaching period: their
    # subtopics now have no day (the term report counts them). Said, not
    # dropped without a word; removing the day gives them back.
    lessons_dropped: Optional[int] = None


class TermRequest(Camel):
    """Create and update take the whole term: a term is three fields, and a
    full replace keeps the overlap check about one complete range."""
    name: str = Field(min_length=1)
    start_date: str
    end_date: str


class TermResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    name: str
    start_date: str
    end_date: str
    manual_baseline_minutes: Optional[float] = None


class TermBaselineRequest(Camel):
    """Minutes a teacher takes to set a full paper by hand, for this term;
    null clears it. Bounded to one sitting (10 hours): anything outside it is
    a typo, and it would move the saving a principal quotes."""
    minutes: Optional[float] = Field(..., gt=0, le=600)


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
    # Weekly offs and alternate Saturdays made working days (N-3-19); they
    # are among working_days.
    compensatory_count: int = 0


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
    # True replaces this computation's own earlier estimates (not an admin's)
    # -- how estimates made before a calendar change, or by the
    # pre-2026-09-22 periods x weeks formula, are brought back inside the year.
    recompute: bool = False
    # Size the budget from this section's own week (its periods of the
    # subject on the year's working days); periodsPerWeek then defaults to
    # its allocation. Omitted, the class's sections' weeks size it (N-3-8).
    section_id: Optional[str] = None


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
    # The budget: real teaching periods this subject has this year (working
    # days x the subject's periods on them), not periods_per_week x weeks.
    total_subject_periods: int
    total_instructional_minutes: int
    units_skipped_no_subtopics: list[str]
    estimates_created: int
    # Periods every estimate of this book now holds. Above the budget only
    # when the syllabus has more subtopics than the year has periods.
    periods_allocated: int = 0
    periods_short: int = 0
    fits_in_year: bool = True
    # The section whose own week sized the budget; null: the school-wide slots.
    section_id: Optional[str] = None


# ---------------- micro scheduling (§11-14, §38-41) ----------------

class ScheduleBookRequest(Camel):
    # Required for the school-wide plan. For a section's plan (sectionId) it
    # defaults to the section's allocation, and the section's own timetable
    # decides the days when it has one (SCH-4).
    periods_per_week: Optional[int] = Field(default=None, gt=0)
    force: bool = False
    section_id: Optional[str] = None
    # D35: when what is left to teach needs more periods than are left, give
    # each subtopic proportionally fewer (at least one) so all of it is dated.
    fit_to_periods_left: bool = False


class BuildPlansRequest(Camel):
    # The principal's own approval of the topics proposed when the school was
    # seeded (decomposition templates). Never assumed: false plans only what
    # already has approved subtopics.
    approve_proposed_topics: bool = False
    # D35: fit each plan to the periods left (see ScheduleBookRequest).
    fit_to_periods_left: bool = False


class PlanBuiltResponse(Camel):
    section_id: str
    section_name: str
    subject_name: str
    lessons_created: int
    subtopics_without_estimate: int = 0
    last_scheduled_date: Optional[str] = None
    warning: Optional[str] = None
    # Chapters with no approved subtopic: not in the plan at all (D119).
    chapters_without_subtopics: list[str] = []
    # D35: subtopics given fewer periods to fit the periods left, and why.
    subtopics_compressed: int = 0
    fit_note: Optional[str] = None


class BuildPlansResponse(Camel):
    runs_approved: int = 0
    topics_created: int = 0
    subtopics_created: int = 0
    plans: list[PlanBuiltResponse] = []
    # One sentence per section and subject that got no plan, and why.
    not_planned: list[str] = []


class CompressedSubtopicResponse(Camel):
    subtopic_id: str
    periods_needed: int
    periods_planned: int


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
    teaching_periods_available: int = 0
    lessons_kept: int = 0
    # Unmarked lessons dated before today that a force regenerate kept (a
    # past day is taught or overdue, never replanned); they count toward
    # their subtopic's estimate like completed ones.
    past_lessons_kept: int = 0
    # The shortfall flag (rule Q4: a gap is reported, not implied): false,
    # with `warning` saying how many subtopics have no date and why, whenever
    # a subtopic is unscheduled, partially scheduled or has no estimate. Still
    # a 200 -- the lessons that fit were created -- but no longer a silent one.
    all_subtopics_scheduled: bool = True
    warning: Optional[str] = None
    section_id: Optional[str] = None
    # Chapters with no approved subtopic, in delivery order: the plan leaves
    # them out, and allSubtopicsScheduled is false while there are any (D119).
    chapters_without_subtopics: list[str] = []
    # fitToPeriodsLeft (D35): each subtopic given fewer periods than its
    # estimate so the syllabus fits the periods left, and the sentence saying
    # so. Empty when the plan was not fitted or already fitted.
    subtopics_compressed: list[CompressedSubtopicResponse] = []
    fit_note: Optional[str] = None


class ScheduledLessonResponse(Camel):
    id: str
    school_id: str
    academic_year_id: str
    book_id: str
    subtopic_id: str
    date: str
    # scheduled | completed | skipped | unscheduled ('unscheduled': a PUSH
    # could not fit it before the year ends; `date` is the last day it was
    # planned for and it holds no day).
    status: str
    note: Optional[str] = None
    completed_by: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
    section_id: Optional[str] = None   # SCH-4; null: the school-wide plan


# ---------------- completion tracking (§15) ----------------

class MarkLessonRequest(Camel):
    # "partly": taught, but not all of it -- recorded as completed with the
    # note "Partly taught", and the rest is planned for the next period.
    status: str = Field(pattern="^(scheduled|completed|skipped|partly)$")
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
    # SCH-4: push one section's plan (null: the school-wide plan).
    section_id: Optional[str] = None


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
    section_id: Optional[str] = None


class RescheduleHistoryEntryResponse(Camel):
    timestamp: str
    actor: Optional[str] = None
    mode: str               # adjust | push | unscheduled
    old_date: str
    new_date: Optional[str] = None   # null when a PUSH left the lesson with no day
    reason: str


# ---------------- student visibility (§18) ----------------
# Deliberately a reduced, read-only subset -- no note, no completed_by (a
# teacher's private note isn't necessarily meant for a student to read),
# no write endpoints of any kind. "No management controls" per the
# transcript.

class EnrollStudentRequest(Camel):
    """sectionId is the class (M1.1). gradeId alone still works for a
    grade with one section, as it did before sections existed; for a grade
    with several it is a 422 naming them."""
    student_id: str = Field(min_length=1)
    grade_id: Optional[str] = Field(default=None, min_length=1)
    section_id: Optional[str] = Field(default=None, min_length=1)


class StudentEnrollmentResponse(Camel):
    id: str
    school_id: str
    student_id: str
    grade_id: str
    section_id: Optional[str] = None
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
    # Lessons a PUSH left with no day (status 'unscheduled'), counted
    # whatever their stale date; totalCount includes them, so
    # scheduled + completed + skipped + unscheduled == total.
    unscheduled_count: int = 0
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
    # SCH-4: the section this plan is for (null: the school-wide plan).
    section_id: Optional[str] = None
    section_name: Optional[str] = None


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
    # SCH-4: the section this plan is for (null: the school-wide plan).
    section_id: Optional[str] = None
    section_name: Optional[str] = None
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
    # Which book this lesson belongs to, and whether the school still teaches
    # that edition: an abandoned edition's lessons stay overdue for ever.
    book_id: str = ""
    book_title: str = ""
    is_chosen_edition: bool = True
    # SCH-4: the section this plan is for (null: the school-wide plan).
    section_id: Optional[str] = None
    section_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None


class UnscheduledLessonResponse(Camel):
    """A lesson a PUSH found no period for before the year ends. It has no
    day, so no days-overdue figure; lastPlannedDate is the day it held
    before it was dropped."""
    lesson_id: str
    last_planned_date: str
    book_id: str = ""
    book_title: str = ""
    is_chosen_edition: bool = True
    grade_number: int
    subject_name: str
    chapter_name: str
    topic_name: str
    subtopic_name: str
    # SCH-4: the section this plan is for (null: the school-wide plan).
    section_id: Optional[str] = None
    section_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None


class DelayedTopicsReportResponse(Camel):
    school_id: str
    academic_year_id: str
    as_of_date: str
    delayed_count: int
    delayed_lessons: list[DelayedLessonResponse]
    unscheduled_count: int = 0
    unscheduled_lessons: list[UnscheduledLessonResponse] = []


class IngestTocRequest(Camel):
    toc_text: str = Field(..., min_length=1, description="Raw text of the textbook Table of Contents")


class IngestTocResponse(Camel):
    book_id: str
    units_created: int
    chapters_created: int
    unit_ids: list[str]
    chapter_ids: list[str]
