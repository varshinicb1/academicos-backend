"""API schemas mirroring the Flutter app's freezed entities field-for-field.

json_serializable (no fieldRename configured) uses each Dart field name as-is
for JSON keys, i.e. camelCase. These models use alias_generator=to_camel so
Python stays snake_case internally but the wire format matches exactly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
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


StudentLevelTier = Literal["foundation", "standard", "advanced"]
ExamType = Literal["class_test", "weekly_test", "monthly_test", "mid_term", "pre_board", "board"]


class Blueprint(Camel):
    total_marks: int
    duration_minutes: int
    difficulty: DifficultyDistribution
    bloom: BloomDistribution
    chapter_weights: ChapterWeights
    competency_weights: CompetencyWeights
    sections: list[SectionBlueprint] = Field(default_factory=list)
    tier: str = "standard"
    competency_percentage: float = 0.50
    exam_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlueprintRequest(Camel):
    total_marks: int
    duration_minutes: int
    difficulty: DifficultyDistribution
    bloom: BloomDistribution
    chapter_weights: ChapterWeights
    competency_weights: CompetencyWeights
    sections: list[SectionBlueprint] = Field(default_factory=list)
    tier: str = "standard"
    competency_percentage: float = 0.50
    exam_type: Optional[str] = None
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
    # Which official scheme this came from, if any. Without this a consumer
    # cannot tell an official CBSE value-point scheme from an empty
    # placeholder, and two thirds of the bank is the latter.
    # "ncert_exemplar_answer": NCERT's own answer from an Exemplar book's
    # answers file -- official, but not a CBSE value-point marking scheme.
    provenance: str = "none"          # "none" | "cbse_marking_scheme" | "ncert_exemplar_answer" | "teacher"
    source_paper_code: str = ""        # e.g. "31/1/1"
    source_document_id: str = ""


# ---- Rights, provenance, calibration (question-bank-api.md sections 3) ----

class RightsSchema(Camel):
    """The rights position for a question's *content*.

    Separate from ingestion provenance: this is about who owns the CBSE paper
    and whether this project may redistribute it. Defaults to the most
    restrictive answer, so an unsettled decision fails closed. A compliance
    ruling then becomes a filter over this field rather than a migration.
    """

    origin: str = "CBSE"
    redistribution: Literal["unknown", "private-only", "public"] = "unknown"
    basis: str = ""
    attribution: str = ""


class ProvenanceSchema(Camel):
    """Where in the source document the question text actually sits.

    `questionBankId` identifies the document; this identifies the place in it.
    Without page and box, "here is the page in the 2023 paper this came from"
    is a claim the data cannot support, and that claim is the stated moat.
    """

    source_document_id: Optional[str] = None
    page_number: Optional[int] = None
    bounding_box: Optional[list[float]] = None
    method: str = "pdf_native"
    confidence: Optional[float] = None


class CalibrationSchema(Camel):
    """Difficulty as measured, not as assigned.

    The mapper derives difficulty from mark value and Bloom level, which is a
    heuristic, not a measurement. Saying so in the payload is the difference
    between an honest field and an implied statistic.
    """

    measured: bool = False
    sample_size: int = 0
    facility: Optional[float] = None          # proportion answering correctly
    discrimination: Optional[float] = None    # point-biserial, if computed
    basis: str = "assigned"                   # "assigned" | "responses" | "consensus"


ReviewState = Literal["draft", "in_review", "published", "superseded"]


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

    # -- added for the standalone-asset requirements -------------------------
    rights: RightsSchema = Field(default_factory=RightsSchema)
    provenance: ProvenanceSchema = Field(default_factory=ProvenanceSchema)
    calibration: CalibrationSchema = Field(default_factory=CalibrationSchema)
    # Monotonic per question. Never rewritten in place: a correction is a new
    # version, so a consumer can reproduce exactly what they saw.
    version: int = 1
    review_state: ReviewState = "published"
    # Set instead of deleting when a curriculum revision retires a question.
    superseded_by: Optional[str] = None

    # -- taxonomy tags (academicos.syllabus.tagger) --------------------------
    # Ids from academicos-data/syllabus/taxonomy (the NCERT textbook headings,
    # classes 6-10). Written only where the tagger's confidence clears the
    # gold-set threshold; below it they stay empty and the question is in
    # taxonomy/review_queue.json. Optional: most banks do not carry them yet.
    taxonomy_chapter_id: Optional[str] = None
    topic_ids: list[str] = Field(default_factory=list)
    subtopic_ids: list[str] = Field(default_factory=list)
    tag_confidence: dict[str, Optional[float]] = Field(default_factory=dict)
    tag_method: Optional[str] = None


class QuestionSearchParams(Camel):
    subject: str
    grade: int
    chapter_ids: Optional[list[str]] = None
    # curriculum/ Subtopic canonical row ids (not chapter_ids' keyword-
    # tagger strings) -- questions tagged via
    # POST /api/v1/curriculum/chapters/{chapter_id}/questions/tag. See
    # docs/ACADEMIC_DATA_MODEL.md section 1 and qmap.py::map_subtopics.
    subtopic_ids: Optional[list[str]] = None
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


# ---- Teacher paper templates (Task 901) ----
#
# A teacher's template is the paper's STRUCTURE (sections, marks, how many to
# attempt, difficulty mix, scope), where `SchoolTemplate` above is its LOOK
# (header, logo, fonts). `PaperTemplate` extends `SchoolTemplate` rather than
# being a parallel model so it is stored by the same `TemplateStore`, and a
# paper generated from one carries its branding into the PDF exporter
# unchanged -- `PaperStore.save` and `pdf.export_pdf` already take a
# `SchoolTemplate`.

PaperExamType = Literal["unit_test", "periodic", "half_yearly", "annual", "custom"]


class TemplateSection(Camel):
    """One section of a teacher's paper.

    `attempt_count` is CBSE's "attempt any N": `question_count` questions are
    printed, `attempt_count` are answered, and only the attempted ones count
    towards the paper's total. None means every question is attempted.

    `question_types` empty means any type at this mark value. The bank's type
    labels are mostly derived from marks, not read from the paper (231 class
    10 Mathematics 1-mark questions are typed `very_short_answer` and none
    `mcq`; every 5-mark question is `case_study`), so the engine matches the
    long-form kinds -- long answer, case study, map -- by what the question
    says (paper_templates._content_fits), and the presets set types only on
    those sections.

    `choice_count` is the CBSE internal choice: that many of the section's
    questions print with an OR alternative of the same kind and marks, and
    the student answers one of the two. It adds no marks -- unlike
    `attempt_count`, nothing extra is scored -- so the totals check ignores
    it. The CBSE papers use it, not "attempt any N": the Class X 2024-25
    Science SQP has ORs on 2 of 6 questions in Section B, 1 of 7 in C and all
    3 in D (academicos-data/corpus/cbse-sqp/ClassX_2024_25/Science-SQP.pdf).
    """

    id: str = ""
    title: str = Field(min_length=1)
    question_types: list[QuestionType] = Field(default_factory=list)
    question_count: int = Field(ge=1)
    marks_each: int = Field(ge=1)
    attempt_count: Optional[int] = Field(default=None, ge=1)
    difficulty_mix: Optional[DifficultyDistribution] = None
    competency_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    choice_count: int = Field(default=0, ge=0)

    @property
    def attempts(self) -> int:
        return self.attempt_count if self.attempt_count is not None else self.question_count

    @property
    def marks(self) -> int:
        """The marks this section contributes to the paper's total."""
        return self.attempts * self.marks_each


class TemplateScope(Camel):
    """What the paper may draw on. Empty chapter_ids means the whole syllabus.

    topic_ids / subtopic_ids come from the board-level taxonomy another stream
    is building (`academicos-data/syllabus/taxonomy/`). When that file is
    absent they are stored as given, not validated -- the availability check
    says so in its `scopeNotes` rather than pretending it checked them.
    """

    chapter_ids: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)
    subtopic_ids: list[str] = Field(default_factory=list)


class TemplateHeader(Camel):
    school_name: str = ""
    exam_name: str = ""
    date_line: str = ""


class PaperTemplateDraft(Camel):
    """What a teacher edits. Totals are checked here, on the way in, so a
    template that cannot be printed as described is never saved."""

    name: str = Field(min_length=1)
    grade: int = Field(ge=1, le=12)
    subject: str = Field(min_length=1)
    exam_type: PaperExamType = "custom"
    total_marks: int = Field(ge=1)
    duration_minutes: int = Field(ge=1)
    instructions: str = ""
    header: TemplateHeader = Field(default_factory=TemplateHeader)
    sections: list[TemplateSection] = Field(min_length=1)
    scope: TemplateScope = Field(default_factory=TemplateScope)

    @model_validator(mode="after")
    def _totals_are_exact(self):
        problems: list[str] = []
        for s in self.sections:
            if s.attempt_count is not None and s.attempt_count > s.question_count:
                problems.append(
                    f"{s.title}: attempt {s.attempt_count} is more than its "
                    f"{s.question_count} questions")
            if s.choice_count > s.question_count:
                problems.append(
                    f"{s.title}: {s.choice_count} OR choices is more than its "
                    f"{s.question_count} questions")
            mix = s.difficulty_mix
            if mix is not None and abs(mix.easy + mix.medium + mix.hard - 1.0) > 0.011:
                problems.append(
                    f"{s.title}: difficulty mix adds up to "
                    f"{round((mix.easy + mix.medium + mix.hard) * 100)}%, not 100%")
        if problems:
            raise ValueError("; ".join(problems))
        added = sum(s.marks for s in self.sections)
        if added != self.total_marks:
            parts = ", ".join(
                f"{s.title} {s.attempts}x{s.marks_each}={s.marks}" for s in self.sections)
            raise ValueError(
                f"sections add up to {added} marks but the total is {self.total_marks} "
                f"({parts})")
        return self


PatternStatus = Literal["cbse_board", "suggested", "teacher"]


class PaperTemplate(SchoolTemplate, PaperTemplateDraft):
    """A saved (or preset) teacher template.

    `kind` is how `TemplateStore` tells these apart from a school's branding
    rows in the same table, so the older `/school-templates` list never shows
    a teacher's private paper to the rest of the school. `is_default` is
    always False: a paper template must never displace the school's branding
    default, which every PDF without a template falls back to.
    """

    kind: Literal["paper"] = "paper"
    is_default: bool = False
    owner_id: str = ""
    shared: bool = False
    is_preset: bool = False
    pattern_status: PatternStatus = "teacher"
    # Where a preset's pattern comes from, in words a teacher can check.
    source: str = ""
    # The preset or template this one was copied from, if any.
    based_on: Optional[str] = None
    # True while `instructions` is the text a preset generated from its
    # sections: set on a preset, kept by a copy, cleared once the teacher
    # changes the instructions. Such text is rebuilt from the paper actually
    # printed; comparing it with what the *current* sections would generate
    # was not enough, because a copy whose sections the teacher edited no
    # longer matched and printed the preset's old lines verbatim ("Section E
    # ... of 4 mark(s)" after Section E was set to 5 marks). Server-set: the
    # draft a client sends has no such field.
    instructions_generated: bool = False
    # Response-only, recomputed on create and update, never stored: what the
    # save could not check about the scope (e.g. topic ids accepted without a
    # taxonomy to check them against).
    scope_notes: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PaperGenerationRequest(Camel):
    assessment_id: str
    blueprint: Blueprint
    selected_questions: list[QuestionSchema]
    template: SchoolTemplate
    formatting_options: Optional[dict[str, Any]] = None
    set_count: int = 1
    tier: Optional[str] = None
    # The teacher chose these questions: two that look alike print with a
    # `warnings` entry. True refuses them (422) instead. See routes.py
    # `_similar_question_warnings`.
    reject_similar: bool = False


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
    internal_choice_question_id: Optional[str] = None
    is_competency: bool = False
    # The question's type, so the renderer grids (A)-(D) only for an MCQ. Without
    # it, a descriptive question's sub-parts (a)-(d) were printed as options.
    # Empty for papers generated before this field existed.
    type: str = ""


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
    generated_by: str = "AcademicOS"
    version: str = "1.0"
    set_label: Optional[str] = None
    tier: Optional[str] = None
    exam_type: Optional[str] = None
    # A teacher template's header and instructions, printed by the PDF
    # exporter. They used to live only in the assessment's metadata, which
    # the exporter never reads, so a template paper printed the canned
    # instructions under the school's branding name. None = not from a
    # template: the exporter keeps its defaults.
    school_name: Optional[str] = None
    exam_name: Optional[str] = None
    date_line: Optional[str] = None
    # One instruction per line, printed numbered in place of the defaults.
    instructions: Optional[str] = None


class GeneratedPaper(Camel):
    id: str
    assessment_id: str
    sections: list[GeneratedSectionSchema]
    formatted_content: str = ""
    formatted_content_latex: str = ""
    answer_key: dict[str, Any] = Field(default_factory=dict)
    metadata: PaperMetadataSchema
    set_label: Optional[str] = None
    sets: list[GeneratedPaper] = Field(default_factory=list)
    # About the request that made this paper, not stored with it: the
    # optimizer's warnings (a CBQ share under target, "tiers unavailable for
    # this subject", a short paper), and two hand-picked questions that look
    # like the same question. Empty on a stored paper.
    warnings: list[str] = Field(default_factory=list)
    # The pairs those warnings name, as [earlier id, later id], so a client
    # can offer "swap one" on the later question without parsing ids back out
    # of the sentence. Empty on a stored paper.
    similar_pairs: list[list[str]] = Field(default_factory=list)
    # Per set label, how many printed questions (OR alternatives included) the
    # set shares with an earlier set: 0 unless the pool ran out of alternatives.
    set_overlap: dict[str, int] = Field(default_factory=dict)
    # The competency-based share of the printed questions and whether it meets
    # the blueprint's target (CBSE: at least 50%). None on a paper stored before.
    competency_share: Optional[float] = None
    competency_target_met: Optional[bool] = None
    # False when the subject's questions carry no difficulty or Bloom signal a
    # tier could act on (selection.tier_signals). None where no optimizer ran.
    tiers_available: Optional[bool] = None


class QuickPaperRequest(Camel):
    subject: str
    # Required: a missing grade used to become a class 10 paper.
    grade: int
    chapter_ids: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    total_marks: int = 80
    duration_minutes: Optional[int] = None
    tier: str = "standard"  # "foundation" | "standard" | "advanced"
    exam_type: Optional[str] = None  # "class_test", "weekly_test", "board"
    set_count: int = 1
    template_id: Optional[str] = None


class GenerateFromIdsRequest(Camel):
    assessment_id: Optional[str] = None
    title: str = "Custom Question Paper"
    subject: str
    # Required: a missing grade used to become a class 10 paper.
    grade: int
    question_ids: list[str]
    template_id: Optional[str] = None
    # As PaperGenerationRequest.reject_similar.
    reject_similar: bool = False


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
