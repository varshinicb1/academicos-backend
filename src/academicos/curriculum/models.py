"""Curriculum/schedule entities -- see docs/ACADEMIC_DATA_MODEL.md for the
design rationale (why this is a separate system from the knowledge graph,
bridged by canonical_id).

Plain dataclasses, matching the style of assessment/schemas.py and
syllabus/cbse_syllabus.py rather than the graph's pydantic Node model --
these are administrative records with a single source of truth (an admin
created/approved them), not probabilistic extraction output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Board:
    id: str
    name: str              # "CBSE" | "ICSE" | "State Board: <name>"
    code: str


@dataclass
class AcademicYear:
    id: str
    school_id: str
    label: str              # "2026-27"
    start_date: str         # ISO date
    end_date: str
    status: str = "draft"   # draft | active | closed


@dataclass
class Grade:
    id: str
    academic_year_id: str
    number: int              # 1-12
    section: Optional[str] = None


@dataclass
class Subject:
    id: str
    grade_id: str
    name: str
    code: Optional[str] = None


@dataclass
class Book:
    """The school's actual chosen book for a subject -- §6: not every
    available book, one deliberate selection."""
    id: str
    subject_id: str
    board_id: str
    title: str
    publisher: Optional[str] = None
    source_doc_ids: list[str] = field(default_factory=list)
    status: str = "selected"   # selected | processing | ready


@dataclass
class Unit:
    """CBSE's real marks-weightage lives here, not on Chapter -- the
    official curriculum gives weightage per unit, not per chapter
    (confirmed against the real data: academicos-data/syllabus/*.json)."""
    id: str
    canonical_id: str
    book_id: str
    unit_no: str
    name: str
    marks: Optional[int] = None
    seq: int = 0


@dataclass
class Chapter:
    id: str
    canonical_id: str
    unit_id: str
    name: str
    seq: int = 0            # delivery order -- independently reorderable
                             #   from textbook/unit order (§13)


# Where a Topic/Subtopic's name actually came from -- distinct handling
# matters: an admin trusts "official_source" and "manual" outright,
# "llm_proposed" only after admin review turned it into a real row (see
# extraction.py -- an AI proposal is never written here directly, only an
# *approved* one), "imported" for a future bulk-import path.
SOURCE_TYPES = ("official_source", "llm_proposed", "manual", "imported")


@dataclass
class Topic:
    id: str
    canonical_id: str
    chapter_id: str
    name: str
    seq: int = 0
    description: Optional[str] = None
    source_type: str = "manual"
    source_reference: Optional[str] = None   # doc_id/page or extraction_run_id, when known
    approved_by: Optional[str] = None        # admin user id -- a Topic row only
                                              #   exists once approved, so this is
                                              #   never null in practice, but kept
                                              #   nullable for a future bulk-import path
    approved_at: Optional[str] = None        # ISO datetime
    model_used: Optional[str] = None         # e.g. "sarvam-105b", null for manual/official
    generation_version: Optional[str] = None # prompt_version from the extraction run, if any


@dataclass
class Subtopic:
    """canonical_id here is what qmap.py resolves a question against --
    see docs/ACADEMIC_DATA_MODEL.md section 1."""
    id: str
    canonical_id: str
    topic_id: str
    name: str
    seq: int = 0
    description: Optional[str] = None
    source_type: str = "manual"
    source_reference: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    model_used: Optional[str] = None
    generation_version: Optional[str] = None


@dataclass
class PeriodConfiguration:
    id: str
    school_id: str
    academic_year_id: str
    period_minutes: int     # 30 | 40 | 45 | 50 | admin-set -- never hardcoded (§8)


@dataclass
class SubjectPeriodAllocation:
    """How many periods/week a subject gets -- previously taken as an
    explicit call argument every time (calendar.py's
    compute_teaching_time_estimates(), scheduling.py's schedule_book()),
    documented there as a real per-school decision this module didn't yet
    persist. One real school row per (academic_year, subject); unlike
    PeriodConfiguration's one-time-set-then-409 posture, this is a real
    upsert -- periods-per-week is the kind of thing a school plausibly
    revises term to term, where period duration in minutes is not."""
    id: str
    school_id: str
    academic_year_id: str
    subject: str
    periods_per_week: int


@dataclass
class SubjectTimetableSlot:
    """One real weekday a subject meets this year (e.g. "Science meets
    Monday, Wednesday, Friday") -- what scheduling.py's own docstring
    called the missing "real signal for exactly *which* weekdays a school
    actually assigns this subject", previously always the deterministic
    "first N working days of the week" heuristic regardless of a school's
    real timetable. `period_number` (which slot of the day, e.g. 1st
    period) is real, forward-looking data for a future period-by-period
    timetable display -- scheduling itself only needs day_of_week."""
    id: str
    school_id: str
    academic_year_id: str
    subject: str
    day_of_week: int         # 0=Monday .. 6=Sunday, matching date.weekday()
    period_number: int


@dataclass
class Calendar:
    id: str
    academic_year_id: str
    weekly_off_days: list[str] = field(default_factory=lambda: ["sunday"])
    alternate_saturday_rule: str = "none"   # "2nd,4th" | "all" | "none"


@dataclass
class Holiday:
    id: str
    calendar_id: str
    date: str                # ISO date -- start date for a multi-day block
    label: str
    kind: str = "holiday"    # holiday | event | unexpected_closure
    # None means a single-day holiday (the common case: a national holiday,
    # Annual Day). Set for a real multi-day block -- a 30-45 day summer
    # break -- so it doesn't require one row per date; inclusive of both
    # ends, same convention as `date`.
    end_date: Optional[str] = None


@dataclass
class CurriculumExtractionRun:
    """One attempt at proposing Topic/Subtopic content for a chapter --
    never the source of truth by itself (see extraction.py: a run's
    proposals only become real Topic/Subtopic rows on admin approval).
    Kept so "why did the system split this chapter this way" (a real
    question a principal will ask) has a real, traceable answer: which
    model, which prompt version, against what source text, when."""
    id: str
    school_id: str
    book_id: str
    chapter_id: str
    source_hash: Optional[str]   # hash of whatever grounding text was used,
                                  #   null when there was none (chapter-name-only
                                  #   generation -- see extraction.py)
    model: Optional[str]         # null for the no-LLM-available run
    prompt_version: str
    created_at: str
    status: str = "pending"      # pending | reviewed | approved | superseded


@dataclass
class CurriculumExtractionProposal:
    """One proposed Topic or Subtopic from a run -- admin-reviewable,
    never auto-materialized. proposed_parent is null for a top-level
    Topic proposal, or another proposal's id for a Subtopic (nested under
    a Topic proposal from the *same run*, so a Subtopic can be approved
    alongside the Topic it belongs to before either has a real id)."""
    id: str
    run_id: str
    entity_type: str             # "topic" | "subtopic"
    proposed_name: str
    proposed_description: Optional[str] = None
    proposed_parent: Optional[str] = None   # another proposal's id, or null
    sequence: int = 0
    confidence: float = 0.5
    status: str = "pending"      # pending | approved | edited | rejected
    edited_name: Optional[str] = None       # admin's replacement, if status="edited"
    materialized_id: Optional[str] = None   # the real topics/subtopics.id, once approved


@dataclass
class QuestionSubtopicLink:
    """A question tagged to a real, approved Subtopic -- the bridge
    docs/ACADEMIC_DATA_MODEL.md section 1 describes: qmap.py resolves a
    question's text against real subtopic canonical_ids and records the
    match here, independent of the PDF-pool's own (chapter-level, keyword-
    based) tagging in assessment/chapters.py, which this does not replace."""
    id: str
    question_id: str
    subtopic_id: str
    method: str = "lexical"      # lexical | llm | manual
    confidence: float = 0.5


@dataclass
class TeachingTimeEstimate:
    """A school's estimate for a subtopic in a given academic year -- not
    an intrinsic property of the subtopic (§27 lists it as its own
    entity for exactly this reason: the same subtopic takes different
    real time at different schools/paces)."""
    id: str
    subtopic_id: str
    academic_year_id: str
    estimated_minutes: int
    estimated_periods: Optional[int] = None
    method: str = "admin_override"   # marks_weightage_proportional | admin_override | llm_estimate
    approved_by: Optional[str] = None  # admin user id; null until reviewed (§29)


@dataclass
class TeacherAssignment:
    """Which real book (a school's one chosen book per subject) a real
    teacher is assigned to teach -- the missing link needed to scope a
    teacher's own schedule to just their subjects. `assessment/users.py`'s
    `User` has no subject/class field at all (confirmed: id, school_id,
    name, email, role, created_at only), so without this a "what do I
    teach today" view could only show a whole school's schedule, not one
    teacher's real slice of it."""
    id: str
    school_id: str
    teacher_id: str
    book_id: str
    created_at: str = ""


@dataclass
class StudentEnrollment:
    """Which real Grade (class/section) a real student belongs to -- the
    missing link needed to scope a student's own view (§18) to their real
    class, the same gap TeacherAssignment closes for teachers.
    `assessment/users.py`'s `User` has no class/grade field at all. One
    enrollment per student (unlike a teacher, who can be assigned several
    books): a real student is in exactly one class."""
    id: str
    school_id: str
    student_id: str
    grade_id: str
    created_at: str = ""


STATUS_VALUES = ("scheduled", "completed", "skipped")


@dataclass
class ScheduledLesson:
    """One real, dated period of instruction for one Subtopic -- the §11-14
    "date-wise ScheduledLesson[]" the transcript calls the actual point of
    the project. Produced by curriculum/scheduling.py from a real Calendar
    (working days), real PeriodConfiguration, and real TeachingTimeEstimate
    rows -- never fabricated independent of those three.

    `status`/`note`/`completed_by`/`completed_at` are §15's completion
    tracking -- the scheduler only ever writes status="scheduled"; a
    teacher marking a lesson done/skipped (`mark_lesson` in store.py) is
    the only thing that ever changes the rest, same "who did this and
    when" shape as Topic/Subtopic's approved_by/approved_at."""
    id: str
    school_id: str
    academic_year_id: str
    book_id: str
    subtopic_id: str
    date: str                     # ISO date, always a real working day
    status: str = "scheduled"     # scheduled | completed | skipped
    note: Optional[str] = None
    completed_by: Optional[str] = None   # the teacher user id who marked it, null until they do
    completed_at: Optional[str] = None   # ISO datetime, null until marked
    created_at: str = ""
