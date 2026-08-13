"""Pillar 4 — Personalized remediation.

    Weak concept -> practice set -> submit -> AI evaluation -> mastery update
                 -> repeat until mastered

Practice is drawn from the same real question pool as the exam, filtered to the
student's weak concepts and biased toward easier items first so a struggling
student gets a foothold rather than another failure. Each submission runs
through the same marking-scheme evaluator as the exam, so mastery updates are
consistent with how the paper was scored.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .evaluate import Evaluation, evaluate_answer
from .knowledge import MASTERY_TARGET, ConceptMasteryView, KnowledgeStore
from .marking import build_answer_scheme, parse_options
from .schemas import QuestionSchema

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


@dataclass
class PracticeItem:
    question: QuestionSchema
    concept_id: str


@dataclass
class PracticeSet:
    id: str
    student_id: str
    concept_ids: list[str]
    items: list[PracticeItem] = field(default_factory=list)
    answer_key: dict[str, str] = field(default_factory=dict)   # question id -> correct option
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_marks(self) -> int:
        return sum(i.question.marks for i in self.items)


@dataclass
class PracticeOutcome:
    concept_id: str
    concept_name: str
    mastery_before: float
    mastery_after: float
    delta: float
    mastered: bool

    @property
    def improved(self) -> bool:
        return self.delta > 0


@dataclass
class PracticeResult:
    set_id: str
    student_id: str
    score: int
    max_score: int
    evaluations: list[Evaluation] = field(default_factory=list)
    outcomes: list[PracticeOutcome] = field(default_factory=list)
    next_action: str = ""

    @property
    def percentage(self) -> float:
        return round(100.0 * self.score / self.max_score, 2) if self.max_score else 0.0


def is_scorable(q: QuestionSchema, answer_key: dict[str, str]) -> bool:
    """Can this question be auto-scored?

    An objective question is only scorable if we know the correct option.
    Without an ingested marking scheme an MCQ would silently score 0 for every
    student, so mastery could never improve — practice must exclude it rather
    than quietly poison the loop.
    """
    if q.id in answer_key:
        return True
    if q.answer_scheme.metadata.get("correctOption"):
        return True
    return not parse_options(q.stem) and q.type not in ("mcq", "assertion_reason")


def build_practice_set(student_id: str, weak: list[ConceptMasteryView],
                       pool: list[QuestionSchema], *, per_concept: int = 3,
                       exclude_question_ids: set[str] | None = None,
                       answer_key: dict[str, str] | None = None) -> PracticeSet:
    """Targeted practice for a student's weakest concepts.

    Only includes questions the loop can actually score, and reports any weak
    concept it could not build practice for instead of dropping it silently.
    """
    exclude = exclude_question_ids or set()
    key = dict(answer_key or {})
    items: list[PracticeItem] = []
    warnings: list[str] = []

    for view in weak:
        candidates = [q for q in pool
                      if view.concept_id in q.chapter_ids and q.id not in exclude]
        scorable = [q for q in candidates if is_scorable(q, key)]
        if not scorable:
            warnings.append(
                f"No auto-scorable practice available for {view.concept_name}: "
                f"{len(candidates)} question(s) exist but need an answer key "
                f"(ingest the marking scheme or set the key manually).")
            continue
        # Easiest first: rebuild confidence before pushing difficulty.
        scorable.sort(key=lambda q: (_DIFFICULTY_ORDER.get(q.difficulty, 1), -q.quality_score))
        for q in scorable[:per_concept]:
            items.append(PracticeItem(question=q, concept_id=view.concept_id))

    return PracticeSet(
        id=f"practice_{uuid.uuid4().hex[:10]}",
        student_id=student_id,
        concept_ids=[v.concept_id for v in weak],
        items=items,
        answer_key=key,
        warnings=warnings,
    )


def submit_practice(practice: PracticeSet, answers: dict[str, str],
                    store: KnowledgeStore) -> PracticeResult:
    """Evaluate a practice submission and fold it back into mastery."""
    before = {v.concept_id: v.mastery for v in store.mastery(practice.student_id)}

    evaluations: list[Evaluation] = []
    graded: list[tuple[QuestionSchema, Evaluation]] = []
    for item in practice.items:
        scheme = item.question.answer_scheme
        if not scheme.marking_points or not scheme.metadata:
            scheme = build_answer_scheme(
                item.question, correct_option=practice.answer_key.get(item.question.id))
        ev = evaluate_answer(item.question, scheme, answers.get(item.question.id, ""),
                             concept_label=item.concept_id)
        evaluations.append(ev)
        graded.append((item.question, ev))

    store.record_evaluations(practice.student_id, graded)

    after_views = {v.concept_id: v for v in store.mastery(practice.student_id)}
    outcomes: list[PracticeOutcome] = []
    for cid in practice.concept_ids:
        view = after_views.get(cid)
        if view is None:
            continue
        b = before.get(cid, 0.0)
        outcomes.append(PracticeOutcome(
            concept_id=cid, concept_name=view.concept_name,
            mastery_before=round(b, 4), mastery_after=round(view.mastery, 4),
            delta=round(view.mastery - b, 4),
            mastered=view.mastery >= MASTERY_TARGET,
        ))

    score = sum(e.awarded_marks for e in evaluations)
    max_score = sum(e.max_marks for e in evaluations)
    still_weak = [o.concept_name for o in outcomes if not o.mastered]
    next_action = ("All targeted concepts reached mastery — move to new material."
                   if not still_weak else
                   "Continue practice on: " + ", ".join(still_weak[:3]))

    return PracticeResult(
        set_id=practice.id, student_id=practice.student_id,
        score=score, max_score=max_score,
        evaluations=evaluations, outcomes=outcomes, next_action=next_action,
    )
