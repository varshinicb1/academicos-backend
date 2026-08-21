"""Pillar 2 — AI evaluation with teacher-in-the-loop.

Scores one answer at a time (never a whole paper in one shot — per-question is
debuggable and lets the teacher approve incrementally).

Design commitments:
  * Every mark traces to a named marking point. No opaque scores.
  * Confidence is reported and *acted on*: anything below `REVIEW_THRESHOLD`
    is queued for the teacher rather than auto-accepted.
  * OCR damage is detectable. Real evaluated CBSE scripts showed struck-through
    working merging into tokens like `44220` (a cancelled 220 with 44 above it);
    `ocr_risk_flags` spots that shape and drops confidence so a human looks.
  * The teacher never starts from zero — they approve, edit, or comment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..algorithms import misconceptions
from .schemas import AnswerSchemeSchema, QuestionSchema

REVIEW_THRESHOLD = 0.75   # below this, a human must look
# Length below which a *descriptive* answer cannot be meaningfully assessed.
# Never applied to objective questions: "(A)" is a complete MCQ answer, and
# treating it as blank silently zeroes correct work.
DESCRIPTIVE_MIN_CHARS = 8

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
# A digit run long enough that it is probably two numbers fused by a strikethrough.
_FUSED_NUMBER = re.compile(r"\b\d{5,}\b")
_OPTION_ANSWER = re.compile(r"\b(?:option\s*)?\(?([A-D])\)?\b")
# Phrases a scan pipeline or demo answer legitimately uses in place of an empty
# string for a skipped question. These contain alphanumeric characters, so the
# "no answer written" check below (which looks for *any* alnum char) missed
# them — an MCQ answer of "Not attempted." fell through to option-matching,
# found no option letter, and returned a confusing "could not identify which
# option" verdict instead of a clean blank/0-mark one.
_BLANK_PHRASES = {"not attempted", "no answer", "blank", "skipped", "n/a", "na", "-", "none"}


@dataclass
class MarkingPointOutcome:
    marking_point_id: str
    description: str
    awarded: bool
    marks: int
    reason: str
    similarity: float


@dataclass
class Evaluation:
    question_id: str
    awarded_marks: int
    max_marks: int
    verdict: str                 # fullCredit | partialCredit | noCredit | blank
    confidence: float
    reasoning: str
    marking_points: list[MarkingPointOutcome] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    ocr_warnings: list[str] = field(default_factory=list)
    needs_review: bool = True

    @property
    def percentage(self) -> float:
        return round(100.0 * self.awarded_marks / self.max_marks, 2) if self.max_marks else 0.0


def ocr_risk_flags(answer: str) -> list[str]:
    """Signals that the transcription — not the student — may be wrong."""
    flags: list[str] = []
    if _FUSED_NUMBER.search(answer):
        flags.append("possible struck-through working merged into one number "
                     "(cancellations are not reliably transcribed)")
    if answer.count("?") > 2:
        flags.append("multiple unresolved characters in transcription")
    letters = [c for c in answer if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) > 0x2FF) / len(letters) > 0.3:
        flags.append("mixed-script transcription; may be mis-segmented")
    return flags


def _normalise(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def _match_point(point, answer_words: set[str], answer_low: str) -> MarkingPointOutcome:
    candidates = [point.keyword, *point.synonyms]
    best = 0.0
    hit = ""
    for cand in candidates:
        c = (cand or "").strip().lower()
        if not c:
            continue
        if " " in c:
            score = 1.0 if c in answer_low else 0.0
        else:
            score = 1.0 if c in answer_words else 0.0
        if score > best:
            best, hit = score, c
    awarded = best >= 1.0
    return MarkingPointOutcome(
        marking_point_id=point.id,
        description=point.description,
        awarded=awarded,
        marks=point.marks if awarded else 0,
        reason=(f"found '{hit}' in the answer" if awarded
                else f"no mention of '{point.keyword}' or its variants"),
        similarity=best,
    )


def evaluate_answer(question: QuestionSchema, scheme: AnswerSchemeSchema,
                    student_answer: str, *, concept_label: str | None = None) -> Evaluation:
    answer = (student_answer or "").strip()
    max_marks = scheme.total_marks or question.marks or 1
    ocr_flags = ocr_risk_flags(answer)
    objective = bool(scheme.metadata.get("objective"))

    # Truly blank: nothing a student could have written, or a stand-in phrase
    # for "skipped" (see _BLANK_PHRASES above).
    if not any(c.isalnum() for c in answer) or answer.strip(" .").lower() in _BLANK_PHRASES:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks, verdict="blank",
            confidence=0.95, reasoning="No answer written for this question.",
            ocr_warnings=ocr_flags, needs_review=False,
        )

    if objective:
        return _evaluate_objective(question, scheme, answer, max_marks, ocr_flags)

    if len(answer) < DESCRIPTIVE_MIN_CHARS:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks, verdict="noCredit",
            confidence=0.4,
            reasoning="Answer is too short to assess against the marking scheme.",
            ocr_warnings=ocr_flags, needs_review=True,
        )

    if not scheme.marking_points:
        # C4: a question whose answer_scheme has zero marking points has no key
        # at all. The descriptive scorer used to fall through anyway and compute
        # a confident-sounding verdict ("none of the expected marking points were
        # found", ~0.9 confidence, needs_review=False) on the strength of an
        # empty rubric -- an authoritative-looking score with nothing to score
        # against. Not reachable through the shipped client today (it always
        # sends full bank questions with answerScheme attached, and
        # build_answer_key synthesizes a key otherwise), but a hand-imported
        # keyless question must fail safe: low confidence, forced review, and a
        # reason that names the missing scheme -- distinct from a genuine
        # zero-credit answer, which a real key still scores confidently.
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks,
            verdict="partialCredit", confidence=0.0,
            reasoning=("This question has no marking scheme (no marking points) to "
                       "score against — it cannot be graded automatically; a teacher "
                       "must mark it manually."),
            ocr_warnings=ocr_flags, needs_review=True,
        )

    return _evaluate_descriptive(question, scheme, answer, max_marks, ocr_flags, concept_label)


def _evaluate_objective(question: QuestionSchema, scheme: AnswerSchemeSchema, answer: str,
                        max_marks: int, ocr_flags: list[str]) -> Evaluation:
    key = str(scheme.metadata.get("correctOption") or "").upper()
    chosen = ""
    ambiguous = False
    letters = [g.upper() for g in _OPTION_ANSWER.findall(answer.upper())]
    distinct = list(dict.fromkeys(letters))  # first-seen order, de-duplicated
    if distinct:
        # Keep the first-mentioned letter: verified against real CBSE scan
        # data (today's harness run), the genuine selection consistently
        # comes first and contamination is trailing -- boilerplate assertion/
        # reason phrasing ("...Reason(R)...Assertion(A)..."), stray OCR noise
        # from a neighboring question's segment, or a garbled marker like
        # "CD)". A "prefer the last letter" heuristic was tried and measured
        # WORSE against that same real data (5/18 previously-correct real
        # answers flipped to wrong), so first-match stays the pick.
        #
        # What was genuinely missing: when MULTIPLE DISTINCT letters appear,
        # the pick is inherently uncertain, and the old code trusted it with
        # full confidence regardless -- a student whose own reasoning named
        # an earlier option before their real answer ("I first considered
        # (A) but eliminated it, then chose (D)") got marked wrong at 93%
        # confidence with no review flag. That is fixed below: ambiguous
        # cases still use the same first-match best guess, but confidence is
        # capped low enough to force review instead of a silent, confident
        # wrong mark.
        chosen = distinct[0]
        ambiguous = len(distinct) > 1

    if not key:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks,
            verdict="partialCredit", confidence=0.0,
            reasoning="No correct option recorded in the marking scheme — teacher must set the key.",
            ocr_warnings=ocr_flags, needs_review=True,
        )
    if not chosen:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks,
            verdict="noCredit", confidence=0.45,
            reasoning="Could not identify which option the student selected.",
            ocr_warnings=ocr_flags, needs_review=True,
        )

    correct = chosen == key
    confidence = 0.93 if not ocr_flags else 0.6
    if ambiguous:
        # Below REVIEW_THRESHOLD (0.75) so needs_review below picks it up
        # through the same mechanism as every other uncertain read, rather
        # than a second ad-hoc flag.
        confidence = min(confidence, 0.5)
    point = scheme.marking_points[0] if scheme.marking_points else None
    outcomes = []
    if point:
        outcomes.append(MarkingPointOutcome(
            marking_point_id=point.id, description=point.description, awarded=correct,
            marks=max_marks if correct else 0,
            reason=f"student selected {chosen}; key is {key}", similarity=1.0 if correct else 0.0,
        ))
    reasoning = (f"Selected option {chosen}, which matches the key."
                 if correct else
                 f"Selected option {chosen}; the correct option is {key}.")
    if ambiguous:
        reasoning += (f" Multiple option letters appeared in the transcribed answer "
                      f"({', '.join(distinct)}) — {chosen} was taken as the final "
                      f"selection, but this is uncertain and needs a human check.")
    return Evaluation(
        question_id=question.id, awarded_marks=max_marks if correct else 0, max_marks=max_marks,
        verdict="fullCredit" if correct else "noCredit", confidence=confidence,
        reasoning=reasoning,
        marking_points=outcomes,
        strengths=([point.description] if correct and point else []),
        gaps=([] if correct else [f"Correct option was {key}"]),
        ocr_warnings=ocr_flags,
        needs_review=confidence < REVIEW_THRESHOLD,
    )


def _evaluate_descriptive(question: QuestionSchema, scheme: AnswerSchemeSchema, answer: str,
                          max_marks: int, ocr_flags: list[str],
                          concept_label: str | None) -> Evaluation:
    words = _normalise(answer)
    low = answer.lower()
    outcomes = [_match_point(p, words, low) for p in scheme.marking_points]
    awarded = sum(o.marks for o in outcomes)
    awarded = min(awarded, max_marks)

    if awarded >= max_marks:
        verdict = "fullCredit"
    elif awarded > 0:
        verdict = "partialCredit"
    else:
        verdict = "noCredit"

    matched = sum(1 for o in outcomes if o.awarded)
    total_points = len(outcomes) or 1
    # Confidence is highest when the evidence is decisive either way; a middling
    # keyword hit-rate is exactly where a keyword matcher is least trustworthy.
    ratio = matched / total_points
    confidence = 0.9 - 0.5 * (1 - abs(2 * ratio - 1))
    if len(answer) < 25:
        confidence -= 0.1
    if ocr_flags:
        confidence -= 0.25
    confidence = round(max(0.05, min(0.95, confidence)), 2)

    mis: list[str] = []
    if concept_label:
        try:
            mis = [m.name for m, _ in misconceptions.match_answer(concept_label, answer)]
        except Exception:  # catalogue is best-effort, never block scoring
            mis = []

    return Evaluation(
        question_id=question.id, awarded_marks=awarded, max_marks=max_marks, verdict=verdict,
        confidence=confidence,
        reasoning=_reason(outcomes, awarded, max_marks),
        marking_points=outcomes,
        strengths=[o.description for o in outcomes if o.awarded],
        gaps=[o.description for o in outcomes if not o.awarded],
        misconceptions=mis,
        ocr_warnings=ocr_flags,
        needs_review=confidence < REVIEW_THRESHOLD or bool(ocr_flags),
    )


def _reason(outcomes: list[MarkingPointOutcome], awarded: int, max_marks: int) -> str:
    missing = [o.description for o in outcomes if not o.awarded]
    if not missing:
        return f"All marking points addressed ({awarded}/{max_marks})."
    if awarded == 0:
        return ("None of the expected marking points were found: "
                + "; ".join(missing[:3]) + ".")
    return (f"{awarded}/{max_marks}. Missing: " + "; ".join(missing[:3]) + ".")
