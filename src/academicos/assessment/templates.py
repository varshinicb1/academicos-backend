"""Default CBSE-style section template.

The Flutter blueprint UI never actually populates Blueprint.sections (the
review step lets a teacher set marks/duration/difficulty/Bloom mix but not a
custom section layout) — so a sensible default has to fill that gap, matching
the standard CBSE pattern: Section A (MCQ) -> B (2-mark) -> C (3-mark) ->
D (long answer) -> E (case study). Scales proportionally to the blueprint's
total_marks.
"""
from __future__ import annotations

from .schemas import BloomDistribution, DifficultyDistribution, SectionBlueprint

# Student-level tier presets (Testmate / PARAKH differentiated assessment paradigm)
TIER_DIFFICULTY: dict[str, DifficultyDistribution] = {
    "foundation": DifficultyDistribution(easy=0.50, medium=0.35, hard=0.15),
    "standard": DifficultyDistribution(easy=0.25, medium=0.55, hard=0.20),
    "advanced": DifficultyDistribution(easy=0.15, medium=0.40, hard=0.45),
}

# Standard Bloom targets by tier
TIER_BLOOM: dict[str, BloomDistribution] = {
    "foundation": BloomDistribution(
        remember=0.35, understand=0.35, apply=0.20, analyze=0.10, evaluate=0.0, create=0.0,
    ),
    "standard": BloomDistribution(
        remember=0.20, understand=0.30, apply=0.25, analyze=0.15, evaluate=0.05, create=0.05,
    ),
    "advanced": BloomDistribution(
        remember=0.10, understand=0.20, apply=0.30, analyze=0.25, evaluate=0.10, create=0.05,
    ),
}

# (label, name, marks_per_question, share_of_total_marks, allowed difficulties)
_LAYOUT = (
    ("A", "MCQ", 1, 0.25, ["easy", "medium"]),
    ("B", "Very Short Answer", 2, 0.20, ["easy", "medium"]),
    ("C", "Short Answer", 3, 0.25, ["medium", "hard"]),
    ("D", "Long Answer", 5, 0.20, ["medium", "hard"]),
    ("E", "Case Study", 4, 0.10, ["medium", "hard"]),
)

EXAM_PRESETS: dict[str, dict] = {
    "class_test": {
        "name": "Class Test",
        "default_marks": 20,
        "default_duration": 40,
        "sections": [
            ("A", "MCQ", 1, 10, ["easy", "medium"], False, 0),
            ("B", "Very Short Answer", 2, 5, ["easy", "medium"], True, 1),
        ],
    },
    "weekly_test": {
        "name": "Weekly Periodic Test",
        "default_marks": 25,
        "default_duration": 45,
        "sections": [
            ("A", "MCQ", 1, 10, ["easy", "medium"], False, 0),
            ("B", "Short Answer", 3, 5, ["medium", "hard"], True, 1),
        ],
    },
    "monthly_test": {
        "name": "Monthly Unit Exam",
        "default_marks": 50,
        "default_duration": 90,
        "sections": [
            ("A", "MCQ", 1, 15, ["easy", "medium"], False, 0),
            ("B", "Very Short Answer", 2, 5, ["easy", "medium"], False, 0),
            ("C", "Short Answer", 3, 5, ["medium", "hard"], True, 1),
            ("D", "Case / Long Answer", 5, 2, ["medium", "hard"], True, 1),
        ],
    },
    "board": {
        "name": "Board Exam (CBSE 80m Pattern)",
        "default_marks": 80,
        "default_duration": 180,
        "sections": [
            ("A", "MCQ", 1, 20, ["easy", "medium"], False, 0),
            ("B", "Very Short Answer", 2, 5, ["easy", "medium"], True, 1),
            ("C", "Short Answer", 3, 6, ["medium", "hard"], True, 2),
            ("D", "Long Answer", 5, 4, ["medium", "hard"], True, 2),
            ("E", "Case Study", 4, 3, ["medium", "hard"], True, 1),
        ],
    },
}


def get_sections_for_exam_type(exam_type: str, total_marks: int = 80) -> list[SectionBlueprint]:
    preset = EXAM_PRESETS.get(exam_type)
    if not preset:
        return default_sections(total_marks)

    sections: list[SectionBlueprint] = []
    for label, name, marks_per_q, count, difficulties, has_choice, choice_count in preset["sections"]:
        actual_marks = count * marks_per_q
        sections.append(SectionBlueprint(
            id=f"section-{label.lower()}",
            label=label,
            name=name,
            marks_per_question=marks_per_q,
            question_count=count,
            total_marks=actual_marks,
            allowed_bloom_levels=[],
            allowed_difficulties=difficulties,
            has_internal_choice=has_choice,
            internal_choice_count=choice_count,
        ))
    return sections


def default_sections(total_marks: int) -> list[SectionBlueprint]:
    if total_marks == 80:
        return get_sections_for_exam_type("board", 80)

    sections: list[SectionBlueprint] = []
    allocated = 0
    for i, (label, name, marks_per_q, share, difficulties) in enumerate(_LAYOUT):
        is_last = i == len(_LAYOUT) - 1
        section_marks = (total_marks - allocated) if is_last else round(total_marks * share)
        section_marks = max(section_marks, 0)
        count = max(1, round(section_marks / marks_per_q)) if section_marks > 0 else 0
        if count == 0:
            continue
        actual_marks = count * marks_per_q
        allocated += actual_marks
        sections.append(SectionBlueprint(
            id=f"section-{label.lower()}",
            label=label,
            name=name,
            marks_per_question=marks_per_q,
            question_count=count,
            total_marks=actual_marks,
            allowed_bloom_levels=[],
            allowed_difficulties=difficulties,
            has_internal_choice=label in ("C", "D", "E"),
            internal_choice_count=1 if label in ("C", "D", "E") else 0,
        ))
    return sections
