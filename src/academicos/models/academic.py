"""Academic-domain models: the canonical CBSE academic objects.

These are the structured-before-generative layer. Extraction pipelines produce
instances of these; the graph builder upserts them as nodes/edges; retrieval and
agents consume them. Every object embeds Provenance.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Evidence, Provenance
from .enums import BloomLevel


class AcademicObject(BaseModel):
    """Base for all canonical academic objects."""
    canonical_id: str            # e.g. "cbse:ch:2026:xii:physics:1"
    board: str = "CBSE"
    grade: Optional[str] = None
    subject: Optional[str] = None
    academic_year: Optional[str] = None
    title: str
    evidence: Evidence = Field(default_factory=Evidence)


class Subject(AcademicObject):
    code: Optional[str] = None
    medium: Optional[str] = None


class Chapter(AcademicObject):
    """Canonical chapter identifiers: board:grade:subject:chapter-number."""
    chapter_no: Optional[str] = None
    overview: Optional[str] = None
    page_range: Optional[list[int]] = None


class Topic(AcademicObject):
    chapter_id: str
    seq: int = 0
    time_estimate_min: Optional[int] = None
    bloom_level: Optional[BloomLevel] = None


class Subtopic(AcademicObject):
    topic_id: str
    seq: int = 0


class LearningOutcome(AcademicObject):
    """What a student can do after instruction. Statement is the title."""
    verb: Optional[str] = None
    bloom_level: Optional[BloomLevel] = None
    aligned_concepts: list[str] = Field(default_factory=list)
    competency_ids: list[str] = Field(default_factory=list)


class Competency(AcademicObject):
    """Board-defined competency / learning competency statement."""
    code: Optional[str] = None
    category: Optional[str] = None       # e.g. conceptual, procedural, applicative
    cross_cutting: bool = False


class Concept(AcademicObject):
    """A domain concept with definitions/examples; the IKG spine."""
    definition: Optional[str] = None
    formulas: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)


class Prerequisite(AcademicObject):
    """Required prior knowledge for a chapter/topic/concept."""
    for_id: str
    requires_id: str


class Question(AcademicObject):
    question_paper_id: Optional[str] = None
    q_no: str = ""
    section: Optional[str] = None
    marks: Optional[float] = None
    question_type: Optional[str] = None       # MCQ, short answer, long answer, case study
    cognitive: Optional[BloomLevel] = None
    difficulty: Optional[str] = None
    parts: list["QuestionPart"] = Field(default_factory=list)
    competency_ids: list[str] = Field(default_factory=list)
    source_text: Optional[str] = None


class QuestionPart(AcademicObject):
    question_id: str
    part: str = "a"
    marks: Optional[float] = None
    text: Optional[str] = None


class AnswerScheme(AcademicObject):
    question_id: Optional[str] = None
    document_id: Optional[str] = None
    marking_points: list["MarkingPoint"] = Field(default_factory=list)
    rubric: Optional["Rubric"] = None
    expected_answer: Optional[str] = None


class MarkingPoint(AcademicObject):
    answer_scheme_id: str
    point_no: int = 1
    marks: float = 1.0
    detail: str = ""


class Rubric(BaseModel):
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    total_marks: Optional[float] = None
    evidence: Evidence = Field(default_factory=Evidence)


class AssessmentBlueprint(AcademicObject):
    """Unit-wise mark/type distribution (blueprint / design of question paper)."""
    subject_code: Optional[str] = None
    exam_type: Optional[str] = None       # term, board, compett
    total_marks: Optional[float] = None
    duration_min: Optional[int] = None
    unit_allocations: list[dict[str, Any]] = Field(default_factory=list)  # [{unit, marks, weight}]
    question_type_allocations: list[dict[str, Any]] = Field(default_factory=list)


class QuestionPaper(AcademicObject):
    exam_year: Optional[str] = None
    subject_code: Optional[str] = None
    series: Optional[str] = None
    paper_type: Optional[str] = None      # regular, compartment, improvement
    duration_min: Optional[int] = None
    total_marks: Optional[float] = None
    general_instructions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)  # canonical ids


class SyllabusItem(AcademicObject):
    """An entry from a curriculum/syllabus document."""
    unit_id: Optional[str] = None
    chapter_id: Optional[str] = None
    detail: Optional[str] = None
    weightage: Optional[float] = None
    periods: Optional[int] = None
    learning_outcomes: list[str] = Field(default_factory=list)


class Circular(AcademicObject):
    circular_no: Optional[str] = None
    date: Optional[str] = None
    applies_to: list[str] = Field(default_factory=list)
    supersedes_circulars: list[str] = Field(default_factory=list)
    effective_from: Optional[str] = None
