"""Pillar 1 — answer key + flexible marking scheme.

A marking scheme is what makes AI evaluation auditable: every mark awarded
traces to a named marking point, so a teacher can see *why* the score is what
it is instead of trusting a black box.

Built deterministically from the question itself:
  * MCQ / assertion-reason -> options parsed out of the stem, one all-or-nothing
    marking point (the correct option, teacher-supplied or from the source key).
  * Descriptive -> marks split into `marks` marking points seeded from the
    salient content terms of the question, each carrying keyword synonyms so
    partial credit can be awarded on meaning rather than exact wording.

Teachers edit these before approval; the structure is the contract that
`evaluate.py` scores against.
"""
from __future__ import annotations

import re

from .chapters import CLASS_X_SCIENCE_CHAPTERS
from .schemas import (
    AnswerSchemeSchema,
    MarkingPointSchema,
    QuestionSchema,
    RubricLevelSchema,
)

_OPTION_RE = re.compile(r"\(([A-D])\)\s*(.+?)(?=\([A-D]\)|$)", re.S)
_STOPWORDS = {
    "the", "and", "for", "with", "what", "how", "why", "when", "which", "that", "this",
    "from", "into", "will", "are", "is", "was", "were", "be", "been", "of", "in", "on",
    "to", "a", "an", "it", "its", "as", "at", "by", "or", "not", "following", "given",
    "state", "explain", "define", "list", "describe", "write", "name", "answer",
    "question", "questions", "marks", "mark", "each", "any", "two", "three", "one",
    "you", "your", "their", "they", "them", "if", "then", "also", "may", "can", "have",
}
# Terms worth a marking point even though they are short.
_KEEP_SHORT = {"ph", "dna", "rna", "atp", "emf", "ncert", "co2", "h2o", "o2"}

_CONCEPT_TERMS = {kw for ch in CLASS_X_SCIENCE_CHAPTERS for kw in ch.keywords}


def parse_options(stem: str) -> dict[str, str]:
    """Pull (A)/(B)/(C)/(D) options out of an MCQ stem."""
    out: dict[str, str] = {}
    for label, body in _OPTION_RE.findall(stem):
        text = " ".join(body.split()).strip(" .;")
        if text:
            out[label] = text
    return out


def _salient_terms(stem: str, limit: int) -> list[str]:
    """Content words most likely to appear in a correct answer, concept terms first."""
    body = _OPTION_RE.sub("", stem)  # score on the question, not the distractors
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", body.lower())

    phrases = [t for t in _CONCEPT_TERMS if t in body.lower()]
    singles = [w for w in words if w not in _STOPWORDS and (len(w) > 3 or w in _KEEP_SHORT)]

    ordered: list[str] = []
    for term in phrases + singles:
        if term not in ordered:
            ordered.append(term)
    return ordered[:limit] if ordered else ["key idea"]


def build_answer_scheme(q: QuestionSchema, *, correct_option: str | None = None,
                        model_answer: str = "",
                        value_points: list[str] | None = None) -> AnswerSchemeSchema:
    """Derive a marking scheme for one question.

    When the official marking scheme's value points are available, they
    replace the keyword-guessed ones below — an examiner's own scoring steps
    ("Plugging of the leak in blood vessels prevents lowering of blood
    pressure") are what a teacher actually expects to see, not terms lifted
    back out of the question stem.
    """
    marks = max(1, q.marks)
    options = parse_options(q.stem)
    objective = bool(options) or q.type in ("mcq", "assertion_reason")

    if objective:
        key = correct_option or ""
        points = [MarkingPointSchema(
            id=f"{q.id}:mp1",
            description=(f"Correct option {key}: {options.get(key, '')}".strip()
                         if key else "Correct option selected"),
            marks=marks,
            keyword=options.get(key, key),
            is_required=True,
            synonyms=[key] if key else [],
        )]
        rubric = [
            RubricLevelSchema(level=1, label="Correct", min_marks=marks, max_marks=marks,
                              description="Correct option identified."),
            RubricLevelSchema(level=0, label="Incorrect", min_marks=0, max_marks=0,
                              description="Wrong or no option selected."),
        ]
        return AnswerSchemeSchema(
            total_marks=marks, marking_points=points, rubric_levels=rubric,
            common_errors=[f"Chose distractor {lbl}" for lbl in options if lbl != key],
            alternative_answers=[], model_answer=model_answer or options.get(key, ""),
            has_partial_credit=False,
            metadata={"objective": True, "options": options, "correctOption": key},
        )

    value_points = [v for v in (value_points or []) if v.strip()]
    if value_points:
        base = marks // len(value_points)
        remainder = marks - base * len(value_points)
        points = [
            MarkingPointSchema(
                id=f"{q.id}:mp{i + 1}",
                description=point,
                marks=base + (1 if i < remainder else 0),  # spread rounding onto the first few
                keyword=point.split()[0] if point.split() else point,
                is_required=(i == 0),
                synonyms=[],
            )
            for i, point in enumerate(value_points)
        ]
    else:
        terms = _salient_terms(q.stem, marks)
        points = [
            MarkingPointSchema(
                id=f"{q.id}:mp{i + 1}",
                description=f"Addresses '{term}'",
                marks=1,
                keyword=term,
                is_required=(i == 0),
                synonyms=_synonyms(term),
            )
            for i, term in enumerate(terms)
        ]
    # Spread any leftover marks onto the first point so the scheme totals correctly.
    if points and sum(p.marks for p in points) < marks:
        points[0] = points[0].model_copy(
            update={"marks": points[0].marks + (marks - sum(p.marks for p in points))})

    rubric = [
        RubricLevelSchema(level=3, label="Complete", min_marks=marks, max_marks=marks,
                          description="All required points present and correct."),
        RubricLevelSchema(level=2, label="Substantial", min_marks=max(1, marks // 2),
                          max_marks=max(1, marks - 1),
                          description="Most points present; minor omissions."),
        RubricLevelSchema(level=1, label="Partial", min_marks=1, max_marks=max(1, marks // 2),
                          description="Some relevant content; key ideas missing."),
        RubricLevelSchema(level=0, label="No credit", min_marks=0, max_marks=0,
                          description="Irrelevant, blank, or conceptually wrong."),
    ]
    return AnswerSchemeSchema(
        total_marks=marks, marking_points=points, rubric_levels=rubric,
        common_errors=[], alternative_answers=[], model_answer=model_answer,
        has_partial_credit=True, metadata={"objective": False},
    )


def _synonyms(term: str) -> list[str]:
    """Cheap morphological variants so keyword matching is not brittle."""
    t = term.lower()
    out = {t}
    if t.endswith("y"):
        out.add(t[:-1] + "ies")
    if not t.endswith("s"):
        out.add(t + "s")
    if t.endswith("s") and len(t) > 3:
        out.add(t[:-1])
    if " " in t:
        out.update(p for p in t.split() if len(p) > 3)
    return sorted(out - {t})


def build_answer_key(questions: list[QuestionSchema],
                     correct_options: dict[str, str] | None = None,
                     ) -> dict[str, AnswerSchemeSchema]:
    """Answer key for a whole paper, keyed by question id.

    Each `QuestionSchema` already carries a real answer scheme when it came
    from the question bank (`to_question_schema` builds it from the official
    marking scheme, when one was found for that paper). Rebuilding from
    scratch here would discard that official key and fall back to a blind
    keyword guess, so a question's own scheme is kept unless the caller
    supplies an explicit override for it.
    """
    correct_options = correct_options or {}
    key: dict[str, AnswerSchemeSchema] = {}
    for q in questions:
        override = correct_options.get(q.id)
        has_official = bool(q.answer_scheme.metadata.get("correctOption")) or bool(
            q.answer_scheme.marking_points)
        if override or not has_official:
            key[q.id] = build_answer_scheme(q, correct_option=override)
        else:
            key[q.id] = q.answer_scheme
    return key
