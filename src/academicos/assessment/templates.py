"""Default CBSE-style section template.

The Flutter blueprint UI never actually populates Blueprint.sections (the
review step lets a teacher set marks/duration/difficulty/Bloom mix but not a
custom section layout) — so a sensible default has to fill that gap, matching
the standard CBSE pattern: Section A (MCQ) -> B (2-mark) -> C (3-mark) ->
D (long answer) -> E (case study). Scales proportionally to the blueprint's
total_marks.
"""
from __future__ import annotations

from .schemas import BloomDistribution, DifficultyDistribution, SectionBlueprint, TemplateSection

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
    """The CBSE A-E layout scaled to `total_marks`, summing to it EXACTLY.

    This used to round each section's share independently, and the rounding
    errors did not cancel: 61 of the 81 totals from 20 to 100 missed (40 came
    out as 41, 50 as 52, 70 as 69, 100 as 101), so a teacher who asked for a
    50-mark paper was handed a 52-mark one. Now Section A (1-mark items, the
    only size that can absorb any remainder) is fixed at its share, and the
    B-E question counts are the combination nearest their shares whose marks
    make up exactly the rest. The search is a few hundred combinations, so it
    is exhaustive rather than clever.
    """
    if total_marks == 80:
        return get_sections_for_exam_type("board", 80)
    if total_marks < 1:
        return []

    shares = [share for _, _, _, share, _ in _LAYOUT]
    marks_per_q = [mpq for _, _, mpq, _, _ in _LAYOUT]
    counts = _nearest_exact_counts(total_marks, marks_per_q, shares)

    sections: list[SectionBlueprint] = []
    for (label, name, mpq, _share, difficulties), count in zip(_LAYOUT, counts):
        if count == 0:
            continue
        sections.append(SectionBlueprint(
            id=f"section-{label.lower()}",
            label=label,
            name=name,
            marks_per_question=mpq,
            question_count=count,
            total_marks=count * mpq,
            allowed_bloom_levels=[],
            allowed_difficulties=difficulties,
            has_internal_choice=label in ("C", "D", "E"),
            internal_choice_count=1 if label in ("C", "D", "E") else 0,
        ))
    return sections


def _nearest_exact_counts(total: int, marks_per_q: list[int], shares: list[float]) -> list[int]:
    """Question counts per section, index 0 being the 1-mark section, whose
    marks sum to `total` exactly and sit as close to `shares` as possible."""
    targets = [total * share / mpq for share, mpq in zip(shares, marks_per_q)]
    first = max(1, round(targets[0]))
    best: tuple[float, list[int]] | None = None
    # Let the 1-mark section drift from its share only when nothing closer fits.
    for drift in range(0, total):
        for a in sorted({first - drift, first + drift}):
            if a < 1 or a > total:
                continue
            rest = total - a
            for combo in _combos(rest, marks_per_q[1:], targets[1:]):
                counts = [a, *combo]
                cost = sum((c - t) ** 2 / max(t, 1.0) for c, t in zip(counts, targets))
                if best is None or cost < best[0]:
                    best = (cost, counts)
        if best is not None:
            return best[1]
    return [total] + [0] * (len(marks_per_q) - 1)


def _combos(rest: int, marks_per_q: list[int], targets: list[float]):
    """Every count vector within 2 of its target whose marks sum to `rest`."""
    if not marks_per_q:
        if rest == 0:
            yield []
        return
    mpq, target = marks_per_q[0], targets[0]
    low = max(0, int(target) - 2)
    for count in range(low, int(target) + 3):
        used = count * mpq
        if used > rest:
            break
        for tail in _combos(rest - used, marks_per_q[1:], targets[1:]):
            yield [count, *tail]


def template_section_blueprints(sections: list[TemplateSection]) -> list[SectionBlueprint]:
    """A teacher template's sections in the shape the selector and paper
    builder already use, so a template id works anywhere a SectionBlueprint
    list does (e.g. `TemplateStore.sections_for` from quick-generate).

    `total_marks` is the section's ATTEMPTED marks -- what it contributes to
    the paper -- not the printed marks, which differ under "attempt any N".
    """
    out: list[SectionBlueprint] = []
    for i, s in enumerate(sections):
        label = chr(ord("A") + i) if i < 26 else f"S{i + 1}"
        allowed = []
        if s.difficulty_mix is not None:
            mix = s.difficulty_mix
            allowed = [d for d, v in (("easy", mix.easy), ("medium", mix.medium),
                                      ("hard", mix.hard)) if v > 0]
        out.append(SectionBlueprint(
            id=s.id or f"section-{label.lower()}",
            label=label,
            name=s.title,
            marks_per_question=s.marks_each,
            question_count=s.question_count,
            total_marks=s.marks,
            allowed_bloom_levels=[],
            allowed_difficulties=allowed,
            has_internal_choice=s.attempts < s.question_count,
            internal_choice_count=s.question_count - s.attempts,
        ))
    return out
