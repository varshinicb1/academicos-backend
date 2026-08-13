"""Algorithm 12 — Study Planner v2: prereq order + forgetting spacing + capacity.

Sits on top of three already-validated pieces:
  * P1.8 `graph.query.prereq_plan` — topological study order from enriched
    PREREQUISITE_OF edges (foundational prerequisites first).
  * P2.4 `ForgettingModel` (FSRS-6 default) — per-concept retention now and
    days-until-forgotten, which decides *when* each concept must next be seen.
  * Session capacity — a daily time budget in minutes; the planner never asks
    for more than fits.

A 15-minute session should not return the full revision queue and a 9-step
prerequisite chain in one shot: it should return the first few, in
prerequisite order, each with a concrete action ("learn" because never seen,
"retrieve" because retention dipped below threshold), and defer the rest to
the earliest day they are actually due.

Why the two signals need each other: prereq plan alone over-schedules
(every concept "now"); retention alone ignores that you must attempt a
prerequisite before the concept it unlocks. The planner enforces this by
walking the topological path left-to-right and filling until capacity runs
out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from .forgetting import ForgettingModel, MINUTES_PER_CONCEPT
from .learner_model import LearnerModel
from ..graph.store import GraphStore


@dataclass
class PlanItem:
    concept_id: str
    label: str
    action: str              # "learn" | "retrieve" | "skip"
    position: int            # prereq-order position (0 = most foundational)
    minutes: int = 0
    retention: float | None = None
    mastery: float | None = None
    difficulty: float | None = None
    scheduled_for: str | None = None   # iso date, only for deferred items
    reason: str = ""


@dataclass
class StudyPlan:
    target: str
    session: list[PlanItem]        # fits today's capacity, prereq-ordered
    upcoming: list[PlanItem]       # deferred; each tagged with scheduled_for
    totals: dict
    params: dict = field(default_factory=dict)


def _today_iso() -> str:
    return date.today().isoformat()


def _today() -> date:
    return date.today()


class StudyPlanner:
    version = "2.0"

    def __init__(self, store: GraphStore, forgetting: ForgettingModel | None = None,
                 minutes_per_concept: int = MINUTES_PER_CONCEPT,
                 default_capacity_minutes: int = 15):
        self.store = store
        self.forgetting = forgetting or ForgettingModel()
        self.minutes_per_concept = minutes_per_concept
        self.default_capacity_minutes = default_capacity_minutes

    def plan(self, target: str, capacity_minutes: int | None = None,
             learner: LearnerModel | None = None, *, lo: bool = False,
             min_confidence: float = 0.0, max_depth: int = 4,
             mastery_gate: float = 0.5) -> StudyPlan:
        cap = int(capacity_minutes if capacity_minutes is not None
                  else self.default_capacity_minutes)
        cap = max(0, cap)

        # 1. Topological study order (P1.8 rule-path + learner mastery skip).
        from ..graph.query import prereq_plan
        from ..models.enums import NodeType
        raw = prereq_plan(
            self.store, target, min_confidence=min_confidence,
            max_depth=max_depth,
            ntype=NodeType.LEARNING_OUTCOME if lo else NodeType.CONCEPT,
            learner=learner, mastery_gate=mastery_gate)
        if not raw["concept"]:
            return StudyPlan(target=target, session=[], upcoming=[], totals={},
                             params=self._params())
        steps = raw["steps"]

        # ---- Per-step action from FSRS retention ----
        session: list[PlanItem] = []
        upcoming: list[PlanItem] = []
        for position, s in enumerate(steps):
            label = s["label"]
            cid = (self._learner_id(learner, label) or label)
            state = learner.concepts.get(cid) if learner else None
            never_seen = not (state and state.attempts > 0)

            retention = None
            if not never_seen:
                r = self.forgetting.predict_forgetting(learner, cid)
                retention = r.retention_now

            if never_seen:
                action = "learn"
            elif retention < self.forgetting.p.threshold:
                action = "retrieve"
            else:
                action = "skip"

            item = PlanItem(
                concept_id=cid, label=label, action=action, position=position,
                minutes=self.minutes_per_concept,
                retention=round(retention, 4) if retention is not None else None,
                mastery=s.get("mastery"), difficulty=s.get("difficulty"),
                reason="target" if s.get("target") else
                       ("transitive" if s.get("transitive") else "prereq"),
            )

            used = sum(i.minutes for i in session)
            if action == "skip":
                item.scheduled_for = self._next_due(retention, position)
                upcoming.append(item)
            elif used + item.minutes <= cap:
                session.append(item)  # still per-prereq-order: we iterate topo
            else:
                item.scheduled_for = self._next_due(retention, position)
                upcoming.append(item)

        used = sum(i.minutes for i in session)
        totals = {
            "steps": len(session) + len(upcoming),
            "in_session": len(session),
            "session_minutes": used,
            "deferred": len(upcoming),
        }
        return StudyPlan(target=raw["concept"], session=session,
                         upcoming=upcoming, totals=totals, params=self._params())

    def _learner_id(self, learner, label: str) -> str | None:
        if not learner:
            return None
        for key in (label, " ".join(label.split()).lower()):
            if key in learner.concepts:
                return key
        return None

    def _next_due(self, retention: float | None, position: int) -> str:
        """Earliest meaningful day: as soon as retention is expected to fall
        below threshold (FSRS days_until_retention) or the concept is re-seen
        again tomorrow if brand new."""
        start = _today()
        if retention is None:
            return (start + timedelta(days=1)).isoformat()
        rec = self.forgetting.p
        if retention <= rec.threshold:
            return start.isoformat()  # already due: earliest possible
        # linear proxy for the days until retention decays to threshold
        drift = max(1, round((retention - rec.threshold)
                             / max(0.05, 1.0 - rec.threshold) * 5))
        return (start + timedelta(days=drift)).isoformat()

    def _params(self) -> dict:
        return {
            "minutes_per_concept": self.minutes_per_concept,
            "default_capacity_minutes": self.default_capacity_minutes,
            "forgetting_model": self.forgetting.p.model,
            "threshold": self.forgetting.p.threshold,
            "version": self.version,
        }