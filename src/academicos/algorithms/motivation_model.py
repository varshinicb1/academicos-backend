"""Algorithm A3 / P2.7 — "motivation_model": deterministic SDT motivation.

Deterministic self-regulated motivation from engagement + persistence signals,
grounded in Deci & Ryan's Self-Determination Theory (SDT) and Vallerand's
tripartite academic motivation.

SDT (Deci & Ryan 1985; 2000) distinguishes intrinsic motivation (learning
"for its own sake" — genuine interest, curiosity) from extrinsic motivation
(learning for an *instrumental* reason: a grade, a promise, fear). The theory
names three basic needs whose satisfaction sustains intrinsic motivation:

  * competence  — "I can do this" (mastery feedback drives it),
  * autonomy    — "this is my choice" (self-directed practice indexes it),
  * relatedness — "I am not alone" (not yet observable; learner-level param).

Vallerand et al. (1992; 1993) refines the extrinsic pole into
external / introjected / identified regulation and adds **amotivation**
(a helpless "what's the point"). Here amotivation surfaces as the *fragile*
band: near-zero competence, flat affect, and sparse persistence.

Vallerand's measurement is the AMS (Academic Motivation Scale) — a survey.
We instead *infer* the same constructs from the interaction stream (no self
report, per Talsma 2018 — young learners' self-reports lag performance):

  * competence_signal   <- recency-weighted outcome accuracy (7-day half-life;
                           proven to align with the confidence model's mastery
                           proxy, Bandura's strongest source),
  * engagement_signal   <- affect valence blended with persistence ("genuine
                           interest" reads as valence *that keeps showing up"),
  * persistence_signal  <- study-day density x steadiness (autonomy proxy:
                           self-initiated, regular practice rather than
                           one-off bursts).

Signals combine into an SDT-style composite bounded in [0, 1]; the dominant
type band (intrinsic / extrinsic / fragile) and flags come from the composite.

All deterministic, recency-weighted (7-day half-life, like A13 confidence),
no randomness, no LLM, no side effects — the same offline contract as the
other algorithms (`version`, `_params`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .learner_model import Interaction, LearnerModel

try:  # reuse the shared affect->valence map (import-safe, no side effects)
    from .confidence_model import _AFFECT_VALENCE
except Exception:  # pragma: no cover - identical local fallback
    _AFFECT_VALENCE = {
        "excited": 1.0, "confident": 0.95, "engaged": 0.9, "happy": 0.85,
        "curious": 0.8, "neutral": 0.6, "bored": 0.45, "surprised": 0.4,
        "tired": 0.3, "frustrated": 0.25, "anxious": 0.2, "stressed": 0.2,
        "angry": 0.15,
    }

_DEFAULT_AFFECT = 0.6           # neutral default (matches A13/A6)
_PRIOR = 0.5                    # neutral prior for a learner with no evidence
RECENCY_HALF_LIFE_DAYS = 7.0
_STUDY_WINDOW_DAYS = 14         # persistence horizon ("last 14 days")
_TREND_WINDOW_DAYS = 7          # trend compares the last 7d to the prior 7d
_LOW_THRESHOLD = 0.35           # below -> fragile / flagged low (amotivation)
_INTRINSIC_THRESHOLD = 0.6      # above -> autonomous / intrinsic motivation
_EFFORT_AFFECT_THRESHOLDS = {
    "engaged_floor": 0.5,          # valence at/above -> counts as engaged effort
    "disengaged_ceiling": 0.35,   # valence at/below -> counts as disengaged
    "persistence_floor": 0.7,     # persistence above -> intrinsic flag needs
    "affect_floor": 0.6,          # valence above -> intrinsic flag needs
}


@dataclass
class MotivationParams:
    """Calibratable constants for the SDT motivation model (v1)."""
    model: str = "sdt"
    w_competence: float = 0.4     # weight on the competence signal
    w_engagement: float = 0.4     # weight on the engagement signal
    w_persistence: float = 0.2    # weight on the persistence signal
    outcome_weight: float = 0.6   # engagement read: valence share vs outcome
    effort_affect_thresholds: dict = field(
        default_factory=lambda: dict(_EFFORT_AFFECT_THRESHOLDS))
    min_recent: int = 5          # answers needed before competence is trusted
    prior: float = _PRIOR
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    study_window_days: int = _STUDY_WINDOW_DAYS
    trend_window_days: int = _TREND_WINDOW_DAYS
    low_threshold: float = _LOW_THRESHOLD
    intrinsic_threshold: float = _INTRINSIC_THRESHOLD


@dataclass
class MotivationResult:
    learner_id: str
    motivation_score: float       # 0..1 composite (blend, SDT-weighted)
    trend: float                  # -1..1 delta vs the previous window
    competence_signal: float      # 0..1 recency-weighted outcome
    engagement_signal: float      # 0..1 valence x persistence, outcome-blended
    persistence_signal: float     # 0..1 study-day density x steadiness
    type: str                     # "intrinsic" | "extrinsic" | "fragile"
    flagged_low: bool             # "low_motivation": score < low_threshold
    flagged_intrinsic: bool       # "prairied"/autonomous: high persistence + valence
    params: dict = field(default_factory=dict)


def _valence(affect: str | None) -> float:
    if not affect:
        return _DEFAULT_AFFECT
    return _AFFECT_VALENCE.get(affect.lower().strip(), _DEFAULT_AFFECT)


class MotivationModel:
    """Deterministic self-regulated motivation (A3 / P2.7).

    v1 (deterministic):

        competence   = trust.recent_accuracy + (1-trust).prior
        persistence  = study_density x steadiness
                          study_density  = study-days / study_window_days
                          steadiness     = (days-1) / span (1 for a daily run)
        engagement   = outcome_weight * (valence.mean x (0.5 + 0.5*pers))
                       + (1 - outcome_weight) * (recency-weighted accuracy)
        motivation_score = w_competence*competence + w_engagement*engagement
                         + w_persistence*persistence

        type: score >= intrinsic_threshold -> "intrinsic"
              >= low_threshold              -> "extrinsic"
              else                          -> "fragile"
        flagged_low            = score < low_threshold
        flagged_intrinsic      = persistence > floor AND valence > title

    trend: motivation recomputed for the last `trend_window_days` minus the
    same-length earlier window (delta in [-1, 1]).
    """
    version = "1.0"

    def __init__(self, params: MotivationParams | None = None):
        self.p = params or MotivationParams()

    def estimate(self, model: LearnerModel | list[Interaction],
                 concept_id: str | None = None,
                 now: str | None = None) -> MotivationResult:
        events, learner_id = _extract(model, concept_id)
        if not events:
            prior = self.p.prior
            return MotivationResult(
                learner_id=learner_id,
                motivation_score=prior,
                trend=0.0,
                competence_signal=prior,
                engagement_signal=prior,
                persistence_signal=prior,
                type=self._type(prior),
                flagged_low=prior < self.p.low_threshold,
                flagged_intrinsic=False,
                params=self._params(),
            )

        t_now = _ts(now) if now else _ts(self._last_ts(events))
        c, e, pers, score, valence_mean = self._profile(events, t_now)

        m = MotivationResult(
            learner_id=learner_id,
            motivation_score=round(score, 4),
            trend=round(self._trend(events, t_now), 4),
            competence_signal=round(c, 4),
            engagement_signal=round(e, 4),
            persistence_signal=round(pers, 4),
            type=self._type(score),
            flagged_low=score < self.p.low_threshold,
            flagged_intrinsic=self._flag_intrinsic(pers, valence_mean),
            params=self._params(),
        )
        return m

    # ------------------------------------------------------------------ #
    def _profile(self, events: list[Interaction], t_now: float):
        """Compute the three SDT signals, blended score, and mean valence.

        `events` are already sorted by ts. Returns
        (competence, engagement, persistence, score, valence_mean).
        """
        evs = [e for e in events if _dt(e.ts) is not None]
        if not evs:
            return self.p.prior, self.p.prior, self.p.prior, self.p.prior, self.p.prior

        half = self.p.recency_half_life_days * 86400.0
        weights = [math.exp(-(t_now - _ts(e.ts)) / half) for e in evs]
        total_w = sum(weights)

        # -- competence: recency-weighted outcome accuracy, shrunk to prior -- #
        answers = [e for e in evs if e.kind == "answer" and e.outcome is not None]
        if answers:
            aw = [math.exp(-(t_now - _ts(e.ts)) / half) for e in answers]
            acc = sum(wx * e.outcome for wx, e in zip(aw, answers)) / sum(aw)
            trust = min(1.0, len(answers) / max(1, self.p.min_recent))
            competence = trust * acc + (1.0 - trust) * self.p.prior
        else:
            competence = self.p.prior

        # -- persistence: study-day density x steadiness (14-day view) -- #
        window_start = t_now - self.p.study_window_days * 86400.0
        active = [e for e in evs if _ts(e.ts) >= window_start]
        distinct_days = {_dt(e.ts).date() for e in active}
        n_days = len(distinct_days)
        if n_days == 0:
            persistence = 0.0
        elif n_days == 1:
            persistence = 0.0
        else:
            times = sorted(_ts(e.ts) for e in active)
            span = max(1.0, (times[-1] - times[0]) / 86400.0)
            steadiness = (n_days - 1) / span
            density = n_days / self.p.study_window_days
            persistence = density * steadiness
        persistence = max(0.0, min(1.0, persistence))

        # -- engagement: valence blended with persistence + outcome anchor -- #
        valence_mean = (sum(wx * _valence(e.affect) for wx, e in zip(weights, evs))
                        / total_w)
        valence_effort = valence_mean * (0.5 + 0.5 * persistence)
        performance = acc if answers else self.p.prior
        engagement = (self.p.outcome_weight * valence_effort
                      + (1.0 - self.p.outcome_weight) * performance)

        score = (self.p.w_competence * competence
                 + self.p.w_engagement * engagement
                 + self.p.w_persistence * persistence)
        score = max(0.0, min(1.0, score))
        return competence, engagement, persistence, score, valence_mean

    def _trend(self, events: list[Interaction], t_now: float) -> float:
        """Recent (last trend_window) score vs the preceding window's score."""
        t = self.p.trend_window_days * 86400.0
        recent = [e for e in events if t_now - t <= _ts(e.ts) <= t_now]
        earlier = [e for e in events if t_now - 2 * t <= _ts(e.ts) < t_now - t]
        recent_score = self._profile(recent, t_now)[3]
        earlier_score = self._profile(earlier, t_now)[3]
        return max(-1.0, min(1.0, recent_score - earlier_score))

    def _type(self, score: float) -> str:
        if score >= self.p.intrinsic_threshold:
            return "intrinsic"
        if score >= self.p.low_threshold:
            return "extrinsic"
        return "fragile"

    def _flag_intrinsic(self, persistence: float, affect: float) -> bool:
        floors = self.p.effort_affect_thresholds or {}
        floor_p = floors.get("persistence_floor",
                             _EFFORT_AFFECT_THRESHOLDS["persistence_floor"])
        floor_a = floors.get("affect_floor",
                             _EFFORT_AFFECT_THRESHOLDS["affect_floor"])
        return persistence > floor_p and affect > floor_a

    def _last_ts(self, events) -> str:
        cand = [e.ts for e in events if _dt(e.ts) is not None]
        if not cand:
            return _now_utc()
        return max(cand, key=lambda iso: _ts(iso))

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "w_competence": self.p.w_competence,
            "w_engagement": self.p.w_engagement,
            "w_persistence": self.p.w_persistence,
            "outcome_weight": self.p.outcome_weight,
            "effort_affect_thresholds": self.p.effort_affect_thresholds,
            "min_recent": self.p.min_recent,
            "prior": self.p.prior,
            "recency_half_life_days": self.p.recency_half_life_days,
            "study_window_days": self.p.study_window_days,
            "trend_window_days": self.p.trend_window_days,
            "low_threshold": self.p.low_threshold,
            "intrinsic_threshold": self.p.intrinsic_threshold,
        }


def _extract(source, concept_id):
    """Normalize a LearnerModel OR a list[Interaction] into (events, learner_id).

    Type-tolerant by duck-typing: a LearnerModel exposes `.observe`.
    """
    if hasattr(source, "observe"):
        learner_id = source.learner_id
        if concept_id is not None:
            st = source.concepts.get(concept_id)
            events = list(st.history) if st else []
        else:
            events = [i for c in source.concepts.values() for i in c.history]
    else:
        learner_id = "anonymous"
        events = list(source)
        if concept_id is not None:
            events = [i for i in events if i.concept_id == concept_id]
    events = [i for i in events if _dt(i.ts) is not None]
    events.sort(key=lambda i: _ts(i.ts))
    return events, learner_id


def _dt(iso: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _ts(iso) -> float:
    d = _dt(iso)
    if d is None:
        return 0.0
    return d.timestamp()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()