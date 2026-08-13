"""Default CBSE-style section template.

The Flutter blueprint UI never actually populates Blueprint.sections (the
review step lets a teacher set marks/duration/difficulty/Bloom mix but not a
custom section layout) — so a sensible default has to fill that gap, matching
the standard CBSE pattern: Section A (MCQ) -> B (2-mark) -> C (3-mark) ->
D (long answer) -> E (case study). Scales proportionally to the blueprint's
total_marks.
"""
from __future__ import annotations

from .schemas import SectionBlueprint

# (label, name, marks_per_question, share_of_total_marks, allowed difficulties)
_LAYOUT = (
    ("A", "MCQ", 1, 0.25, ["easy", "medium"]),
    ("B", "Very Short Answer", 2, 0.20, ["easy", "medium"]),
    ("C", "Short Answer", 3, 0.25, ["medium", "hard"]),
    ("D", "Long Answer", 5, 0.20, ["medium", "hard"]),
    ("E", "Case Study", 4, 0.10, ["medium", "hard"]),
)


def default_sections(total_marks: int) -> list[SectionBlueprint]:
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
            has_internal_choice=label in ("D", "E"),
            internal_choice_count=1 if label in ("D", "E") else 0,
        ))
    return sections
