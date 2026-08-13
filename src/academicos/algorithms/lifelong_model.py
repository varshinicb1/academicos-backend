"""Algorithm 14 — Lifelong / longitudinal learner model (cross-year).

The LearnerModel is a snapshot of *now*: per-concept state replayed from
events, with no notion of academic year, growth, or subject boundaries.
Learning is a story, not a snapshot — a child who does "fractions" in Class 3
and again in Class 5 is a different fact from a child who touched it once.

This algorithm derives, deterministically from the raw interaction log:

  * **Per-academic-year portraits** (CBSE year = Apr 1 → Mar 31): concepts
    touched, events, accuracy, active days, subject breakdown.
  * **Cross-year continuity** — concepts that recurred across successive
    academic years, with per-year accuracy (did longitudinal engagement
    improve them?).
  * **Cross-subject growth** — given an optional `subject_of` attribution,
    which subjects improved between first and latest active year; subject
    accuracies.

Pure + deterministic, no `ConceptState` aggregation needed (the event log is
the source of truth). Academic-year boundaries follow the CBSE calendar; the
label is the *starting* year (2025-04..2026-03 → "2025-26").
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .learner_model import Interaction


def academic_year(iso: str) -> str:
    """CBSE academic-year label 'YYYY-YY' for a UTC ISO timestamp.

    Apr 1 – Mar 31 window; label = starting year (Feb '27 → 2026-27).
    """
    d = date.fromisoformat(iso[:10])
    start = d.year if d.month >= 4 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


@dataclass
class YearPortrait:
    year: str
    concepts: set[str] = field(default_factory=set)
    events: int = 0
    answers: int = 0
    correct: int = 0
    accuracy: float = 0.0
    active_days: set[str] = field(default_factory=set)
    subject_breakdown: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "concepts": len(self.concepts),
            "events": self.events,
            "answers": self.answers,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 3),
            "active_days": len(self.active_days),
            "subjects": dict(sorted(self.subject_breakdown.items(),
                                    key=lambda kv: -kv[1])[:5]),
        }


@dataclass
class Continuity:
    concept: str
    chain: list[tuple[str, float]]   # [(academic_year, accuracy)]
    improves: float = 0.0            # last accuracy - first accuracy


@dataclass
class LifelongProfile:
    learner_id: str
    years: list[str]                          # ascending academic years
    portraits: dict[str, YearPortrait]
    recurring: list[str]                      # concepts seen in >1 year
    continuity: list[Continuity]              # per recurring concept
    growth: float = 0.0                       # last-year acc − first-year acc
    subjects: dict[str, float] = field(default_factory=dict)  # attn subject acc
    confident: bool = False
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "years": self.years,
            "annual": [self.portraits[y].as_dict() for y in self.years],
            "recurring": self.recurring,
            "continuity": [
                {"concept": c, "chain": [(y, round(a, 3)) for y, a in c.chain],
                 "improves": round(c.improves, 3)}
                for c in self.continuity],
            "growth": round(self.growth, 3),
            "subjects": self.subjects,
            "confident": self.confident,
        }


class LifelongLearner:
    """Derive per-year + cross-year structure from the raw event log."""

    version = "1.0"

    def __init__(self, learner_id: str = "default",
                 subject_of: Callable[[str], str | None] | None = None):
        self.learner_id = learner_id
        self.subject_of = subject_of

    def analyze(self, events: list[Interaction]) -> LifelongProfile:
        portraits: dict[str, YearPortrait] = {}
        for i in events:
            year = academic_year(i.ts)
            p = portraits.setdefault(year, YearPortrait(year=year))
            p.events += 1
            p.concepts.add(i.concept_id)
            p.active_days.add(i.ts[:10])
            if i.kind == "answer" and i.outcome is not None:
                p.answers += 1
                if i.outcome >= 0.5:
                    p.correct += 1
            if self.subject_of:
                s = self.subject_of(i.concept_id)
                if s:
                    p.subject_breakdown[s] = p.subject_breakdown.get(s, 0) + 1
        for p in portraits.values():
            p.accuracy = (p.correct / p.answers) if p.answers else 0.0

        years = sorted(portraits)

        # continuity: concepts active in >1 academic year
        concept_years: dict[str, list[str]] = defaultdict(list)
        for y in years:
            for cid in portraits[y].concepts:
                if cid not in concept_years[cid]:
                    concept_years[cid].append(y)
        recurring = [c for c, ys in concept_years.items() if len(ys) > 1]

        continuity: list[Continuity] = []
        for cid in recurring:
            chain = []
            for y in years:
                if cid not in portraits[y].concepts:
                    continue
                yr_ans = [i for i in events
                          if i.concept_id == cid and i.kind == "answer"
                          and i.outcome is not None and academic_year(i.ts) == y]
                acc = (sum(1 for a in yr_ans if a.outcome >= 0.5) / len(yr_ans)
                       if yr_ans else 0.0)
                chain.append((y, acc))
            if len(chain) >= 2:
                continuity.append(Continuity(
                    concept=cid, chain=chain,
                    improves=round(chain[-1][1] - chain[0][1], 4)))

        # growth: last active year accuracy minus first
        growth = 0.0
        if len(years) >= 2:
            growth = portraits[years[-1]].accuracy - portraits[years[0]].accuracy

        # subject aggregate accuracies across all years
        subjects: dict[str, list[float]] = defaultdict(list)
        if self.subject_of:
            for i in events:
                if i.kind != "answer" or i.outcome is None:
                    continue
                s = self.subject_of(i.concept_id)
                if s:
                    subjects[s].append(i.outcome)
        subject_acc = {s: round(sum(v) / len(v), 4) for s, v in subjects.items()}

        confident = len(years) >= 2 and sum(p.events for p in portraits.values()) >= 5
        return LifelongProfile(
            learner_id=self.learner_id,
            years=years, portraits=portraits,
            recurring=sorted(recurring), continuity=continuity,
            growth=round(growth, 4),
            subjects=subject_acc, confident=confident,
            params=self._params(),
        )

    def _params(self) -> dict:
        return {"version": self.version, "annual_calendar": "CBSE Apr-Mar"}