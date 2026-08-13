"""API schemas mirroring the Flutter app's freezed entities field-for-field.

json_serializable (no fieldRename configured) uses each Dart field name as-is
for JSON keys, i.e. camelCase. These models use alias_generator=to_camel so
Python stays snake_case internally but the wire format matches exactly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal[
    "mcq", "very_short_answer", "short_answer", "long_answer", "very_long_answer",
    "case_study", "assertion_reason", "map_based", "diagram_based", "graph_based",
    "table_based", "competency_based",
]
QuestionSource = Literal[
    "cbse_board_paper", "cbse_sample_paper", "cbse_question_bank", "ncert_exemplar",
    "ncert_textbook", "school_database", "teacher_created", "ai_generated", "competency_framework",
]
AnswerType = Literal[
    "singleChoice", "multipleChoice", "textShort", "textLong", "numeric",
    "diagram", "map", "graph", "table",
]
AssessmentStatus = Literal[
    "draft", "blueprintReady", "questionsSelected", "questionOptimized", "paperGenerated",
    "underReview", "principalApproved", "printed", "conducted", "scanning", "scanned",
    "evaluating", "evaluated", "teacherReviewed", "reportsGenerated", "remediationSent", "archived",
]


class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---- Blueprint ----

class DifficultyDistribution(Camel):
    easy: float
    medium: float
    hard: float


class BloomDistribution(Camel):
    remember: float
    understand: float
    apply: float
    analyze: float
    evaluate: float
    create: float


class ChapterWeights(Camel):
    weights: dict[str, float] = Field(default_factory=dict)


class CompetencyWeights(Camel):
    weights: dict[str, float] = Field(default_factory=dict)


class SectionBlueprint(Camel):
    id: str
    label: str
    name: str
    marks_per_question: int
    question_count: int
    total_marks: int
    allowed_bloom_levels: list[BloomLevel] = Field(default_factory=list)
    allowed_difficulties: list[Difficulty] = Field(default_factory=list)
    has_internal_choice: bool = False
    internal_choice_count: int = 0


class Blueprint(Camel):
    total_marks: int
    duration_minutes: int
    difficulty: DifficultyDistribution
    bloom: BloomDistribution
    chapter_weights: ChapterWeights
    competency_weights: CompetencyWeights
    sections: list[SectionBlueprint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlueprintRequest(Camel):
    total_marks: int
    duration_minutes: int
    difficulty: DifficultyDistribution
    bloom: BloomDistribution
    chapter_weights: ChapterWeights
    competency_weights: CompetencyWeights
    sections: list[SectionBlueprint] = Field(default_factory=list)
    school_template: Optional[dict[str, Any]] = None


# ---- Question ----

class QuestionPartSchema(Camel):
    id: str
    part_number: int
    text: str
    text_latex: str = ""
    marks: int
    answer_type: AnswerType = "textLong"
    options: Optional[list[str]] = None
    correct_option: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_answer_latex: Optional[str] = None
    keywords: Optional[list[str]] = None
    alternative_answers: list[str] = Field(default_factory=list)


class MarkingPointSchema(Camel):
    id: str
    description: str
    marks: int
    keyword: str = ""
    is_required: bool = False
    synonyms: list[str] = Field(default_factory=list)


class RubricLevelSchema(Camel):
    level: int
    label: str
    min_marks: int
    max_marks: int
    description: str = ""


class AnswerSchemeSchema(Camel):
    total_marks: int
    marking_points: list[MarkingPointSchema] = Field(default_factory=list)
    rubric_levels: list[RubricLevelSchema] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    alternative_answers: list[str] = Field(default_factory=list)
    model_answer: str = ""
    model_answer_latex: str = ""
    has_partial_credit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionSchema(Camel):
    id: str
    question_bank_id: str
    subject: str
    grade: int
    chapter_ids: list[str] = Field(default_factory=list)
    competency_ids: list[str] = Field(default_factory=list)
    bloom_level: BloomLevel
    difficulty: Difficulty
    type: QuestionType
    stem: str
    stem_latex: str = ""
    parts: list[QuestionPartSchema] = Field(default_factory=list)
    answer_scheme: AnswerSchemeSchema
    estimated_time_minutes: int
    marks: int
    language: str = "en"
    source: QuestionSource
    quality_score: float = 0.7
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    diagram_asset_id: Optional[str] = None
    map_asset_id: Optional[str] = None
    graph_asset_id: Optional[str] = None
    table_asset_id: Optional[str] = None


class QuestionSearchParams(Camel):
    subject: str
    grade: int
    chapter_ids: Optional[list[str]] = None
    competency_ids: Optional[list[str]] = None
    bloom_levels: Optional[list[BloomLevel]] = None
    difficulties: Optional[list[Difficulty]] = None
    types: Optional[list[QuestionType]] = None
    min_marks: Optional[int] = None
    max_marks: Optional[int] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    keyword: Optional[str] = None
    min_quality_score: Optional[float] = None
    sources: Optional[list[QuestionSource]] = None


class QuestionOptimizationRequest(Camel):
    candidates: list[QuestionSchema]
    blueprint: Blueprint


class QuestionOptimizationResult(Camel):
    selected_questions: list[QuestionSchema]
    rejected_questions: list[QuestionSchema]
    optimization_metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ---- Paper generation ----

class SchoolTemplate(Camel):
    id: str
    school_id: str
    name: str
    tagline: str = ""
    brand_color: str = "#000000"
    header_html: str = ""
    footer_html: str = ""
    logo_url: str = ""
    margin_top: float = 20
    margin_bottom: float = 20
    margin_left: float = 20
    margin_right: float = 20
    font_family: str = "Helvetica"
    font_size: int = 11
    line_height: float = 1.4
    section_formatting: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = True


class PaperGenerationRequest(Camel):
    assessment_id: str
    blueprint: Blueprint
    selected_questions: list[QuestionSchema]
    template: SchoolTemplate
    formatting_options: Optional[dict[str, Any]] = None


class GeneratedQuestionSchema(Camel):
    question_id: str
    display_number: int
    stem: str
    stem_latex: str = ""
    parts: list[QuestionPartSchema] = Field(default_factory=list)
    marks: int
    bloom_level: str
    difficulty: str
    internal_choice_text: Optional[str] = None


class GeneratedSectionSchema(Camel):
    section_id: str
    label: str
    name: str
    questions: list[GeneratedQuestionSchema] = Field(default_factory=list)
    total_marks: int


class PaperMetadataSchema(Camel):
    assessment_title: str
    subject: str
    grade: int
    total_marks: int
    duration_minutes: int
    generated_at: datetime
    generated_by: str = "AssessmentOS"
    version: str = "1.0"


class GeneratedPaper(Camel):
    id: str
    assessment_id: str
    sections: list[GeneratedSectionSchema]
    formatted_content: str = ""
    formatted_content_latex: str = ""
    answer_key: dict[str, Any] = Field(default_factory=dict)
    metadata: PaperMetadataSchema


# ---- Assessment ----

class CreateAssessmentRequest(Camel):
    teacher_id: str
    school_id: str
    title: str
    subject: str
    grade: int
    chapter_ids: list[str] = Field(default_factory=list)
    blueprint: BlueprintRequest
    template_id: Optional[str] = None


class Assessment(Camel):
    id: str
    school_id: str
    teacher_id: str
    title: str
    subject: str
    grade: int
    chapter_ids: list[str] = Field(default_factory=list)
    blueprint: Blueprint
    status: AssessmentStatus
    created_at: datetime
    updated_at: datetime
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    template_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    selected_question_ids: list[str] = Field(default_factory=list)
    generated_paper_id: Optional[str] = None
    total_students: Optional[int] = None
    evaluated_count: Optional[int] = None
