"""Canonical enums for the AcademicOS ontology.

These enums are the stable contract of the brain. Downstream apps and agents
reference these names; changing them is a breaking schema change (bump version).
"""
from __future__ import annotations

import enum


class Board(str, enum.Enum):
    CBSE = "CBSE"
    NCERT = "NCERT"
    OTHER = "OTHER"


class DocType(str, enum.Enum):
    QUESTION_PAPER = "question_paper"
    MARKING_SCHEME = "marking_scheme"
    MODEL_ANSWER = "model_answer"
    SAMPLE_PAPER = "sample_paper"
    SYLLABUS = "syllabus"
    CURRICULUM = "curriculum"
    QUESTION_BANK = "question_bank"
    CIRCULAR = "circular"
    BYE_LAW = "bye_law"
    POLICY = "policy"
    NOTIFICATION = "notification"
    PRESS_RELEASE = "press_release"
    GUIDELINE = "guideline"
    DATESHEET = "datesheet"
    REFERENCE_MATERIAL = "reference_material"
    UNKNOWN = "unknown"


class PageKind(str, enum.Enum):
    TEXT = "text"
    SCAN = "scan"
    MIXED = "mixed"
    BLANK = "blank"
    IMAGE_ONLY = "image_only"


class BlockKind(str, enum.Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    DIAGRAM = "diagram"
    EQUATION = "equation"
    LIST = "list"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER_FOOTER = "header_footer"
    QUESTION = "question"
    QUESTION_PART = "question_part"
    ANSWER = "answer"
    MARKING_POINT = "marking_point"
    RUBRIC = "rubric"
    KEYWORD = "keyword"
    NOTE = "note"
    INSTRUCTION = "instruction"
    OTHER = "other"


class NodeType(str, enum.Enum):
    BOARD = "Board"
    BOARD_POLICY = "BoardPolicy"
    ACADEMIC_YEAR = "AcademicYear"
    SCHOOL_TYPE = "SchoolType"
    GRADE = "Grade"
    SUBJECT = "Subject"
    UNIT = "Unit"
    CHAPTER = "Chapter"
    TOPIC = "Topic"
    SUBTOPIC = "Subtopic"
    LEARNING_OUTCOME = "LearningOutcome"
    COMPETENCY = "Competency"
    CONCEPT = "Concept"
    PREREQUISITE = "Prerequisite"
    MISCONCEPTION = "Misconception"
    ASSESSMENT_BLUEPRINT = "AssessmentBlueprint"
    QUESTION_PAPER = "QuestionPaper"
    QUESTION = "Question"
    QUESTION_PART = "QuestionPart"
    ANSWER_SCHEME = "AnswerScheme"
    MARKING_POINT = "MarkingPoint"
    RUBRIC = "Rubric"
    EXAM_PATTERN = "ExamPattern"
    SAMPLE_PAPER = "SamplePaper"
    CIRCULAR = "Circular"
    SYLLABUS_ITEM = "SyllabusItem"
    ACTIVITY = "Activity"
    EXPERIMENT = "Experiment"
    WORKSHEET = "Worksheet"
    RESOURCE = "Resource"
    DIAGRAM = "Diagram"
    TABLE = "Table"
    FIGURE = "Figure"
    PAGE = "Page"
    DOCUMENT = "Document"
    PASSAGE = "Passage"
    FORMULA = "Formula"
    TERM = "Term"
    DEFINITION = "Definition"
    EXAMPLE = "Example"
    STANDARD = "Standard"
    POLICY_CLAUSE = "PolicyClause"
    TEACHER_NOTE = "TeacherNote"
    SCHOOL_PLAN = "SchoolPlan"
    REVISION_PLAN = "RevisionPlan"


class EdgeType(str, enum.Enum):
    CONTAINS = "CONTAINS"
    PRECEDES = "PRECEDES"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    MAPS_TO = "MAPS_TO"
    EVIDENCED_BY = "EVIDENCED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    REPLACES = "REPLACES"
    SUPERSEDES = "SUPERSEDES"
    SUPPORTED_BY = "SUPPORTED_BY"
    ASSESSED_BY = "ASSESSED_BY"
    EXEMPLIFIED_BY = "EXEMPLIFIED_BY"
    RELATED_TO = "RELATED_TO"
    WEAKENS = "WEAKENS"
    STRENGTHENS = "STRENGTHENS"
    ALIGNS_WITH = "ALIGNS_WITH"
    PART_OF = "PART_OF"
    HAS_DIFFICULTY = "HAS_DIFFICULTY"
    HAS_TIME_ESTIMATE = "HAS_TIME_ESTIMATE"
    HAS_BLOOM_LEVEL = "HAS_BLOOM_LEVEL"
    HAS_COMPETENCY = "HAS_COMPETENCY"
    HAS_MARKING_POINT = "HAS_MARKING_POINT"
    HAS_SOURCE = "HAS_SOURCE"


class BloomLevel(str, enum.Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ExtractionMethod(str, enum.Enum):
    PDF_NATIVE = "pdf_native"
    UNLIMITED_OCR = "unlimited_ocr"
    PADDLE_OCR = "paddle_ocr"
    TESSERACT = "tesseract"
    LAYOUT_HEURISTIC = "layout_heuristic"
    LLM = "llm"
    RULE_BASED = "rule_based"
    HUMAN = "human"
    ENSEMBLE = "ensemble"


class GraphStatus(str, enum.Enum):
    DRAFT = "draft"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
