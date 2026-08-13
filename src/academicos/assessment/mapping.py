"""PoolQuestion (internal extraction result) -> QuestionSchema (API/Dart contract)."""
from __future__ import annotations

from datetime import datetime, timezone

from .marking import build_answer_scheme
from .pool import PoolQuestion
from .schemas import AnswerSchemeSchema, QuestionSchema

_ROMAN_GRADE = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
                "IX": 9, "X": 10, "XI": 11, "XII": 12}

_QUESTION_TYPE_MAP = {
    "mcq": "mcq",
    "short_answer": "short_answer",
    "long_answer": "long_answer",
    "case_study": "case_study",
    "unknown": None,  # resolved by marks below
}

_BLOOM_MAP = {"remember", "understand", "apply", "analyze", "evaluate", "create"}


def grade_to_int(grade: str) -> int:
    g = grade.strip().upper()
    return _ROMAN_GRADE.get(g, 10)


# CBSE question marks never exceed this. The extractor's "(N)" regex happily
# matches years and clause numbers, so papers were carrying questions worth
# "2023" or "34" marks — clamp to a real range and fall back to the section
# position instead of trusting the capture.
MAX_QUESTION_MARKS = 10


def _plausible_marks(marks: float | None) -> int | None:
    if marks is None:
        return None
    value = int(marks)
    if 1 <= value <= MAX_QUESTION_MARKS:
        return value
    return None


def _infer_marks_from_position(q_no: str) -> int:
    """CBSE Class X Science's standard 39-question structure doesn't print
    per-question mark annotations (marks are implied by section), so when
    extraction found no explicit "(N marks)" text, fall back to the
    well-known section boundaries: 1-20 MCQ (1m), 21-26 (2m), 27-33 (3m),
    34-36 (5m), 37-39 case study (4m)."""
    if not q_no.isdigit():
        return 1
    n = int(q_no)
    if n <= 20:
        return 1
    if n <= 26:
        return 2
    if n <= 33:
        return 3
    if n <= 36:
        return 5
    return 4


def _resolve_type(raw_type: str | None, marks: int) -> str:
    mapped = _QUESTION_TYPE_MAP.get(raw_type or "unknown")
    if mapped:
        return mapped
    if marks <= 1:
        return "very_short_answer"
    if marks == 2:
        return "short_answer"
    if marks in (3, 4):
        return "long_answer"
    if marks >= 5:
        return "case_study"
    return "short_answer"


def _resolve_difficulty(marks: int, bloom: str) -> str:
    """Marks are the primary signal: CBSE's own mark allocation already
    encodes difficulty — a 1-mark MCQ is recall, a 5-mark question is a
    multi-step derivation or extended explanation. Bloom level nudges by at
    most one band; it must never be able to gate a question out of "hard"
    entirely, because `cognitive` is almost never populated by extraction and
    every question was silently defaulting to "understand" (weight 0),
    capping *everything* at "medium" regardless of marks.
    """
    base = 0 if marks <= 2 else 1 if marks == 3 else 2
    bloom_nudge = {"remember": -1, "understand": 0, "apply": 0,
                   "analyze": 1, "evaluate": 1, "create": 1}
    score = max(0, min(2, base + bloom_nudge.get(bloom, 0)))
    return ("easy", "medium", "hard")[score]


def _resolve_bloom(raw: str | None) -> str:
    if raw and raw in _BLOOM_MAP:
        return raw
    return "understand"


def to_question_schema(pq: PoolQuestion) -> QuestionSchema:
    q = pq.question
    marks = _plausible_marks(pq.marks) or _infer_marks_from_position(q.q_no)
    bloom = _resolve_bloom(q.cognitive.value if q.cognitive else None)
    difficulty = _resolve_difficulty(marks, bloom)
    now = datetime.now(timezone.utc)
    qtype = _resolve_type(q.question_type, marks)
    schema = QuestionSchema(
        id=q.canonical_id,
        question_bank_id=q.question_paper_id or "cbse-pilot",
        subject=pq.subject,
        grade=grade_to_int(pq.grade),
        chapter_ids=[pq.chapter_id] if pq.chapter_id else [],
        competency_ids=[],
        bloom_level=bloom,
        difficulty=difficulty,
        type=qtype,
        stem=(q.source_text or q.title or "").strip(),
        stem_latex="",
        parts=[],
        answer_scheme=AnswerSchemeSchema(total_marks=marks, model_answer=""),
        estimated_time_minutes=max(1, marks * 2),
        marks=marks,
        source="cbse_board_paper",
        quality_score=round(0.6 + 0.3 * pq.chapter_confidence, 2),
        tags=[pq.chapter_name] if pq.chapter_name else [],
        created_at=now,
        updated_at=now,
    )
    if pq.has_answer:
        # The official CBSE marking scheme was found for this paper — build the
        # real scheme instead of leaving the placeholder above, so the teacher
        # review queue can auto-score rather than asking for a key it already has.
        scheme = build_answer_scheme(
            schema, correct_option=pq.correct_option,
            model_answer="" if pq.correct_option else pq.answer_text,
            value_points=pq.value_points,
        )
        schema = schema.model_copy(update={"answer_scheme": scheme})
    return schema
