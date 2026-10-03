"""Measured difficulty (EX-3): a question's facility, from the marks teachers
enter.

No source in the served bank judges a question's difficulty: a board record's
is derived from its marks, a CBE item's is the importer's default, an SQP
item's a constant (`selection.difficulty_signal`). So "Foundation" and
"Advanced" could not differ (audit 2026-09-30: identical on 14/14 pairs).

Marks entry (`PUT /papers/{id}/marks`) records, per question, how much of its
marks students scored: the facility index, mean score over maximum, across
every paper, student and school that sat it -- 1 is a question everyone got,
0 one nobody did. With at least `MIN_RESPONSES` answers it is a measurement,
and it becomes the question's difficulty where a tier ranks questions, in
preference to the inferred one.

Only the aggregate is stored and used: a question's facility and its sample
size, never a student's mark (API-5; operations/paper_marks.py).
"""
from __future__ import annotations

from typing import Iterable

from .schemas import QuestionSchema

# Fewer answers than this and a facility is noise, not a measurement.
MIN_RESPONSES = 20

# Facility bands: at or above EASY_AT most students score most of the marks;
# below HARD_BELOW most score little. The classical test-theory cut-offs are
# 0.7 and 0.3; school tests run easy, so hard starts at 0.4.
EASY_AT = 0.7
HARD_BELOW = 0.4

# Where a measured value is kept on a question (its metadata).
FACILITY = "facility"
RESPONSES = "facilityResponses"
MEASURED = "measuredDifficulty"


def difficulty_from(facility: float) -> str:
    """The difficulty a facility measures."""
    if facility >= EASY_AT:
        return "easy"
    if facility < HARD_BELOW:
        return "hard"
    return "medium"


def lookup(question_ids: Iterable[str]) -> dict[str, tuple[float, int]]:
    """Question id -> (facility, responses) for each question answered at
    least MIN_RESPONSES times. Empty when marks storage is not running (a
    paper is then made on inferred difficulty, as before)."""
    from ..operations import routes as ops_routes
    if ops_routes._cfg is None:
        return {}
    return ops_routes.store().facility_for(list(question_ids), min_responses=MIN_RESPONSES)


def annotate(questions: list[QuestionSchema]) -> int:
    """Write each question's measured difficulty into its metadata, where it
    has one; returns how many questions do. The questions must be this
    request's own copies -- never the bank's shared ones."""
    measured = lookup(q.id for q in questions)
    count = 0
    for q in questions:
        found = measured.get(q.id)
        if found is None:
            continue
        facility, responses = found
        q.metadata[FACILITY] = round(facility, 3)
        q.metadata[RESPONSES] = responses
        q.metadata[MEASURED] = difficulty_from(facility)
        count += 1
    return count


def measured_difficulty(q: QuestionSchema) -> str | None:
    """The question's measured difficulty, or None where it has none."""
    return (q.metadata or {}).get(MEASURED)


def sentence(measured: int, printed: int, tier: str) -> str:
    """What a version (tier) of a paper says about its difficulty."""
    level = (tier or "standard").capitalize()
    if not measured:
        return (f"{level} version: none of its {printed} questions has a difficulty measured from "
                f"marks teachers entered (at least {MIN_RESPONSES} answers); every difficulty is "
                f"inferred.")
    return (f"{level} version: {measured} of its {printed} questions have a difficulty measured from "
            f"marks teachers entered (at least {MIN_RESPONSES} answers each); the rest are inferred.")
