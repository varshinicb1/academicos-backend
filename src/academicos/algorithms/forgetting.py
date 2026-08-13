"""Algorithm 2 — Forgetting Prediction and Revision Scheduling.

Per-concept memory strength S (in days) is rebuilt from the interaction
log. Two models:

* model="fsrs" (default, v2): the FSRS-6 DSR engine
  (`algorithms/fsrs.py` — Open Spaced Repetition, MIT). S and D evolve
  per review with the state-of-the-art transition equations; retention
  follows the FSRS forgetting curve R(t) = (1 + factor*t/S)^decay with
  R(S) = 0.9. A concept is "forgotten" once R falls below the threshold.
* model="exponential": legacy v1 — every correct answer multiplies S by a
  growth factor, every wrong answer shrinks it (Ebbinghaus-style).

The RevisionScheduler turns predictions into a daily plan: concepts below
threshold are due now, the rest get a next-review date.

Parameters are research-shaped defaults; the FSRS model ships with the
published default weights (calibratable per learner via the optimizer in
Phase 2, see docs/research/learning-science.md).

Two cognitive-science extensions (v2.x, both evidence-tracked):

* **Sleep consolidation (P2.5)** — overnight memory consolidation is
  sleep-driven, not time-driven (Walker & Stickgold, Neuron 2004; Annu Rev
  Psychol 2006): a review followed by a night's sleep yields a stability
  boost, strongest on the first night. `sleep_boost` per consolidation
  night, saturating (walks of W, no unbounded growth).
* **Retrieval practice (P2.6)** — once an item is recallable, *testing* it
  slows forgetting whereas repeated restudy gives little extra benefit
  (Karpicke & Roediger 2008, Science 319:966; Rowland 2014 meta-analysis).
  `answer` interactions are retrieval (full gain); `watch` interactions are
  restudy (discounted by `restudy_gain`), and the scheduler emits retrieval
  mode (test, not re-read) by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .fsrs import (FSRS, FSRSCard, outcome_to_grade, seed_difficulty,
                   forgetting_curve as fsrs_curve)
from .learner_model import Interaction, LearnerModel
from .mastery import KnowledgeMastery

PREDICTION_WINDOW_DAYS = (1, 5, 14, 30, 90)
MINUTES_PER_CONCEPT = 5


@dataclass
class ForgettingParams:
    """Calibratable constants for the forgetting model."""
    model: str = "fsrs"              # "fsrs" (v2, default) | "exponential" (v1)
    initial_strength: float = 1.0    # S0 in days: strength after first exposure
    strength_growth_factor: float = 2.2  # correct answer: S *= growth
    threshold: float = 0.6           # below this retention the concept is forgotten
    decay_sensitivity: float = 0.8   # wrong answer: S *= exp(-decay_sensitivity)
    noise_floor: float = 0.05        # minimum reported retention
    min_evidence: int = 3            # interactions before predictions are confident

    # --- v2.1 sleep consolidation (Walker & Stickgold 2004/2006) ---
    sleep_boost: float = 0.04        # per consolidation night (fraction of S)
    sleep_gap_max_days: float = 2.0  # gap beyond this stops counting as "overnight"
    sleep_gain_cap: float = 0.3      # total stability gain cap from sleep (30%)

    # --- v2.2 retrieval practice (Karpicke & Roediger 2008) ---
    restudy_gain: float = 0.4        # fraction of retrieval gain given to restudy
    # Scheduler: emit retrieval mode by default for revision sessions.
    retrieval_priority_boost: float = 0.0  # extra credit for retrieval-mode items


@dataclass
class ForgettingResult:
    concept_id: str
    retention_now: float
    predicted_retention_at: list  # [(days, retention), ...] over PREDICTION_WINDOW_DAYS
    days_until_forgotten: float
    evidence_count: int
    confident: bool
    params: dict = field(default_factory=dict)


@dataclass
class RevisionPlan:
    due_today: list     # [(concept_id, retention, priority), ...] sorted by priority desc
    next_reviews: list  # [(concept_id, due_iso), ...] sorted by due date
    totals: dict
    confident: bool
    params: dict = field(default_factory=dict)


@dataclass
class SessionItem:
    concept_id: str
    retention: float
    priority: float
    minutes: int
    expected_gain: float
    mode: str = "retrieval"  # retrieval (test) | restudy (re-read)


class ForgettingModel:
    version = "2.0"

    def __init__(self, params: ForgettingParams | None = None):
        self.p = params or ForgettingParams()
        self._fsrs = FSRS()
        if self.p.model not in ("fsrs", "exponential"):
            raise ValueError(f"unknown forgetting model: {self.p.model}")

    def _card(self, model: LearnerModel, concept_id: str) -> FSRSCard:
        """Replay the learner's history through the FSRS engine.

        * `answer` events are *retrieval practice* → full FSRS review.
        * `watch` events are *restudy* → simulated successful review, but
          the resulting stability is scaled by `restudy_gain` (Karpicke &
          Roediger 2008: restudy gives ~40% of the benefit of testing).
        After replay, stability is boosted by the sleep-consolidation gain.
        """
        state = model.concepts.get(concept_id)
        card = FSRSCard()
        if state is None:
            return card
        events = sorted(state.history, key=lambda x: x.ts)
        for i in events:
            if i.kind == "answer" and i.outcome is not None:
                d = card.difficulty or seed_difficulty(i.difficulty, i.bloom)
                card.difficulty = d
                self._fsrs.review(card, outcome_to_grade(i.outcome), now=i.ts)
            elif i.kind == "watch":
                d = card.difficulty or seed_difficulty(i.difficulty, i.bloom)
                card.difficulty = d
                if card.is_reviewed():
                    self._fsrs.review(card, 3, now=i.ts)
                else:
                    card.last_review = i.ts  # exposure without a verdict
                card.stability = card.stability * self.p.restudy_gain
        card.stability *= self._sleep_gain(events)
        return card

    def _sleep_gain(self, events: list) -> float:
        """Overnight-consolidation stability multiplier (Walker & Stickgold).

        Each consecutive pair of events separated by ~1-2 days (i.e. the
        learner slept between exposures) counts as one consolidation night;
        gains are multiplicative but capped by `sleep_gain_cap` so the
        boost saturates — matching the strongest-first-night effect.
        """
        nights = 0
        for a, b in zip(events, events[1:]):
            gap_days = (_ts(b.ts) - _ts(a.ts)) / 86400.0
            if 1.0 <= gap_days <= self.p.sleep_gap_max_days:
                nights += 1
        gain = (1.0 + self.p.sleep_boost) ** nights
        return min(gain, 1.0 + self.p.sleep_gain_cap)

    def components(self, model: LearnerModel, concept_id: str) -> dict:
        """Explain the current stability for debugging / product surfaces."""
        from .fsrs import init_difficulty
        card = self._card(model, concept_id)
        if self.p.model == "fsrs":
            return {
                "model": "fsrs",
                "stability_days": round(card.stability, 4),
                "difficulty": round(card.difficulty, 4),
                "reps": card.reps,
                "lapses": card.lapses,
                "sleep_gain": round(self._sleep_gain(
                    sorted(model.concepts.get(concept_id, model.concepts[concept_id]).history,
                           key=lambda x: x.ts)), 4),
                "retrievability_now": self.retention_at(
                    model, concept_id,
                    (datetime.now(timezone.utc)).isoformat()),
            }
        return {"model": "exponential"}

    def strength(self, model: LearnerModel, concept_id: str) -> float:
        if self.p.model == "fsrs":
            return round(self._card(model, concept_id).stability, 4)
        state = model.concepts.get(concept_id)
        s = self.p.initial_strength
        if state is None:
            return s
        for i in state.history:
            if i.kind != "answer" or i.outcome is None:
                continue
            if i.outcome >= 0.5:
                s *= self.p.strength_growth_factor
            else:
                s *= self._decay_factor()
        return round(s, 4)

    def retention_at(self, model: LearnerModel, concept_id: str,
                     t_iso: str) -> float:
        state = model.concepts.get(concept_id)
        if state is None or state.attempts == 0:
            return round(self.p.noise_floor, 4)
        if self.p.model == "fsrs":
            card = self._card(model, concept_id)
            last = card.last_review or state.last_seen
            if not card.is_reviewed():
                return round(self.p.noise_floor, 4)
            t = _ts(t_iso)
            dt_days = max(0.0, (t - _ts(last)) / 86400.0)
            r = fsrs_curve(self._fsrs.w, dt_days, card.stability)
            return round(max(self.p.noise_floor, r), 4)
        t = _ts(t_iso)
        last = _ts(state.last_seen or _now_utc())
        dt_days = max(0.0, (t - last) / 86400.0)
        r = math.exp(-dt_days / self.strength(model, concept_id))
        return round(max(self.p.noise_floor, r), 4)

    def predict_forgetting(self, model: LearnerModel, concept_id: str,
                           now: str | None = None) -> ForgettingResult:
        now_iso = now or _now_utc()
        state = model.concepts.get(concept_id)
        if state is None or state.attempts == 0:
            floor = round(self.p.noise_floor, 4)
            return ForgettingResult(
                concept_id=concept_id, retention_now=floor,
                predicted_retention_at=[(d, floor) for d in PREDICTION_WINDOW_DAYS],
                days_until_forgotten=0.0, evidence_count=0, confident=False,
                params=self._params(),
            )
        s = self.strength(model, concept_id)
        retention = self.retention_at(model, concept_id, now_iso)
        window = [
            (d, self.retention_at(model, concept_id,
                                  (_dt(now_iso) + timedelta(days=d)).isoformat()))
            for d in PREDICTION_WINDOW_DAYS
        ]
        remaining = 0.0
        if retention >= self.p.threshold:
            if self.p.model == "fsrs":
                remaining = self._fsrs.days_until_retention(
                    self._card(model, concept_id), self.p.threshold)
            else:
                remaining = max(0.0, s * math.log(retention / self.p.threshold))
        evidence = [i for i in state.history
                    if i.kind == "answer" and i.outcome is not None]
        return ForgettingResult(
            concept_id=concept_id, retention_now=retention,
            predicted_retention_at=window,
            days_until_forgotten=round(remaining, 4),
            evidence_count=len(evidence),
            confident=len(evidence) >= self.p.min_evidence,
            params=self._params(),
        )

    def _decay_factor(self) -> float:
        return math.exp(-self.p.decay_sensitivity)

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "initial_strength": self.p.initial_strength,
            "strength_growth_factor": self.p.strength_growth_factor,
            "threshold": self.p.threshold,
            "decay_sensitivity": self.p.decay_sensitivity,
            "noise_floor": self.p.noise_floor,
            "min_evidence": self.p.min_evidence,
            "sleep_boost": self.p.sleep_boost,
            "sleep_gap_max_days": self.p.sleep_gap_max_days,
            "sleep_gain_cap": self.p.sleep_gain_cap,
            "restudy_gain": self.p.restudy_gain,
        }


class RevisionScheduler:
    version = "1.0"

    def __init__(self, prediction: ForgettingModel | None = None,
                 minutes_per_concept: int = MINUTES_PER_CONCEPT):
        self.prediction = prediction or ForgettingModel()
        self.minutes_per_concept = minutes_per_concept

    def plan(self, model: LearnerModel, now: str | None = None) -> RevisionPlan:
        now_iso = now or _now_utc()
        due, upcoming, confident = [], [], True
        for cid, state in model.concepts.items():
            r = self.prediction.predict_forgetting(model, cid, now_iso)
            confident = confident and r.confident
            if r.retention_now < self.prediction.p.threshold:
                urgency = max(0.0, min(1.0, (self.prediction.p.threshold - r.retention_now)
                                       / self.prediction.p.threshold))
                importance = 0.5 + 0.5 * state.mastery
                due.append((cid, r.retention_now, round(urgency * importance, 4)))
            else:
                days = max(1, math.ceil(r.days_until_forgotten))
                due_iso = (_dt(now_iso).date() + timedelta(days=days)).isoformat()
                upcoming.append((cid, due_iso))
        due.sort(key=lambda t: t[2], reverse=True)
        upcoming.sort(key=lambda t: t[1])
        totals = {
            "concepts": len(model.concepts),
            "due_today": len(due),
            "next_reviews": len(upcoming),
        }
        return RevisionPlan(due_today=due, next_reviews=upcoming, totals=totals,
                            confident=confident, params=self._params())

    def suggest_revision_session(self, model: LearnerModel, minutes: int = 20,
                                 now: str | None = None,
                                 mode: str = "retrieval") -> list:
        """Suggest a session; items default to *retrieval* practice (testing
        beats restudy — K&R 2008), which is what our answer events are.
        `mode="restudy"` yields re-review items for near-forgotten concepts
        whose retention is so low that recall is expected to fail."""
        capacity = int(minutes // self.minutes_per_concept)
        items = []
        for cid, retention, priority in self.plan(model, now).due_today[:capacity]:
            gain = 1.0 - retention
            if mode == "restudy":
                priority += self.prediction.p.retrieval_priority_boost
            items.append(SessionItem(
                concept_id=cid, retention=retention, priority=priority,
                minutes=self.minutes_per_concept, expected_gain=round(gain, 4),
                mode=mode,
            ))
        return items

    def _params(self) -> dict:
        return {
            "minutes_per_concept": self.minutes_per_concept,
            "initial_strength": self.prediction.p.initial_strength,
            "strength_growth_factor": self.prediction.p.strength_growth_factor,
            "threshold": self.prediction.p.threshold,
            "decay_sensitivity": self.prediction.p.decay_sensitivity,
            "noise_floor": self.prediction.p.noise_floor,
            "min_evidence": self.prediction.p.min_evidence,
            "sleep_boost": self.prediction.p.sleep_boost,
            "sleep_gap_max_days": self.prediction.p.sleep_gap_max_days,
            "sleep_gain_cap": self.prediction.p.sleep_gain_cap,
            "restudy_gain": self.prediction.p.restudy_gain,
        }


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def demo(learner_id: str = "demo-forgetting") -> LearnerModel:
    """A quick synthetic learner with concepts at different memory strengths."""
    from datetime import timedelta

    model = LearnerModel(learner_id=learner_id)
    now = datetime.now(timezone.utc)
    for cid, days, outcome in [
        ("fractions", 0, 1.0), ("fractions", 2, 1.0),
        ("fractions", 5, 1.0), ("fractions", 7, 1.0),
        ("algebra", 1, 1.0), ("algebra", 9, 0.0), ("algebra", 14, 1.0),
        ("trigonometry", 45, 1.0),
    ]:
        model.observe(Interaction(
            concept_id=cid, kind="answer", outcome=outcome,
            ts=(now - timedelta(days=days)).isoformat(),
        ))
    km = KnowledgeMastery()
    for cid, state in model.concepts.items():
        state.mastery = km.score(model, cid, now=now.isoformat()).mastery
    return model
