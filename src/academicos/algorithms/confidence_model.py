"""Algorithm 13 — Self-Efficacy / Confidence dynamics per concept.

Bandura (1977; 1997) models self-efficacy as a domain-specific belief in
one's capability, fed by four sources:

  * mastery experiences (the strongest source — authentic prior success),
  * vicarious experiences (observing similar others succeed),
  * social persuasion (feedback from significant others — teacher, parent),
  * physiological and affective states (anxiety, fatigue, excitement).

The calibration literature adds the key diagnostic: what matters is not the
*level* of self-efficacy alone but the gap between perceived and actual
capability. Overconfidence -> skipped practice / careless errors;
under-confidence -> avoidance and premature giving up.

This module maps the four sources to signals already collected:

  * mastery   -> recency-weighted outcome accuracy per concept,
  * affect    -> valence from `affect` tags (engaged > neutral > tired >
                 frustrated > anxious),
  * social    -> learner-level parameter, default neutral (0.5),
  * prior     -> a neutral prior (0.5).

Efficacy is updated with an asymmetry that mirrors recovery dynamics:
success propels it up, failure drags it down, but a learner with high
efficacy recovers fast from failure while a low-efficacy learner
over-generalizes a single failure.

Pure, deterministic, no APIs — same contract as the other algorithms in this
library (version, research-shaped defaults, `_params`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .learner_model import Interaction, LearnerModel

_AFFECT_VALENCE: dict[str, float] = {
    "excited": 1.0, "confident": 0.95, "engaged": 0.9, "happy": 0.85,
    "curious": 0.8, "neutral": 0.6, "bored": 0.45, "surprised": 0.4,
    "tired": 0.3, "frustrated": 0.25, "anxious": 0.2, "stressed": 0.2,
    "angry": 0.15,
}
_DEFAULT_AFFECT = 0.6
_SOCIAL_DEFAULT = 0.5
RECENCY_HALF_LIFE_DAYS = 7.0   # recency decay for the accuracy proxy


@dataclass
class ConfidenceParams:
    """Calibratable constants for the self-efficacy model."""
    model: str = "bandura"
    w_mastery: float = 0.50      # mastery experience (strongest source)
    w_social: float = 0.15       # social persuasion / vicarious (learner-level)
    w_affect: float = 0.20       # physiological & affective state
    w_prior: float = 0.15        # residual prior belief
    prior: float = 0.5           # initial belief when no evidence yet
    alpha: float = 0.35          # how fast success builds efficacy
    beta: float = 0.45           # how fast failure erodes it
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    social: float = 0.5          # learner-level social support (neutral default)
    overconfident_gap: float = 0.15   # calibration above which flagged
    underconfident_gap: float = -0.15  # calibration below which flagged
    min_evidence: int = 3


@dataclass
class ConfidenceResult:
    concept_id: str
    efficacy: float          # 0..1 belief "I can do this"
    accuracy: float          # observed recency-weighted outcome (proxy skill)
    affect: float            # 0..1 mean affective valence
    calibration: float       # efficacy - accuracy (+ over, - under)
    category: str            # overconfident | calibrated | underconfident
    confident: bool
    params: dict = field(default_factory=dict)


def _valence(affect: str | None) -> float:
    if not affect:
        return _DEFAULT_AFFECT
    return _AFFECT_VALENCE.get(affect.lower().strip(), _DEFAULT_AFFECT)


class ConfidenceModel:
    """Per-concept self-efficacy dynamics (task-specific, per Bandura).

    v1 (deterministic):

        efficacy = w_mastery * recent_accuracy
                 + w_affect  * mean_affect_valence
                 + w_social  * social
                 + w_prior   * prior

    then an asymmetry update that replays each answer in time order:

        outcome >= 0.5 -> efficacy += alpha * (1 - efficacy)
        outcome <  0.5 -> efficacy -= beta  * efficacy

    calibration = efficacy - accuracy; thresholds from the params. There is
    no separate temporal drift — mastery is already recency-weighted; efficacy
    is a belief integral (its relative stability is the point).
    """
    version = "1.0"

    def __init__(self, params: ConfidenceParams | None = None):
        self.p = params or ConfidenceParams()

    def estimate(self, model: LearnerModel, concept_id: str,
                 now: str | None = None) -> ConfidenceResult:
        state = model.concepts.get(concept_id)
        if state is None or state.attempts == 0:
            return ConfidenceResult(
                concept_id=concept_id, efficacy=round(self.p.prior, 4),
                accuracy=0.0, affect=round(_DEFAULT_AFFECT, 4),
                calibration=0.0, category="calibrated",
                confident=False, params=self._params(),
            )
        answers = [i for i in state.history
                   if i.kind == "answer" and i.outcome is not None]

        # mastery proxy: recency-weighted accuracy
        t_now = _ts(now or _now_utc())
        half = self.p.recency_half_life_days * 86400.0
        w_sum = c_sum = 0.0
        for i in answers:
            age = max(0.0, t_now - _ts(i.ts))
            w = math.exp(-age / half)
            w_sum += w
            c_sum += w * i.outcome
        accuracy = max(0.0, min(1.0, (c_sum / w_sum) if w_sum > 0 else 0.0))

        # affect: mean valence over all tagged history
        vals = [_valence(i.affect) for i in state.history]
        affect = (sum(vals) / len(vals)) if vals else _DEFAULT_AFFECT

        social = self.p.social if self.p.social >= 0 else _SOCIAL_DEFAULT
        efficacy = (self.p.w_mastery * accuracy + self.p.w_affect * affect
                    + self.p.w_social * social + self.p.w_prior * self.p.prior)

        # asymmetry update across recent evidence, in time order
        for i in answers:
            o = i.outcome if i.outcome is not None else accuracy
            if o >= 0.5:
                efficacy += self.p.alpha * (1.0 - efficacy)
            else:
                efficacy -= self.p.beta * efficacy
        efficacy = max(0.0, min(1.0, efficacy))

        calibration = efficacy - accuracy
        if calibration >= self.p.overconfident_gap:
            cat = "overconfident"
        elif calibration <= self.p.underconfident_gap:
            cat = "underconfident"
        else:
            cat = "calibrated"

        return ConfidenceResult(
            concept_id=concept_id,
            efficacy=round(efficacy, 4),
            accuracy=round(accuracy, 4),
            affect=round(affect, 4),
            calibration=round(calibration, 4),
            category=cat,
            confident=len(answers) >= self.p.min_evidence,
            params=self._params(),
        )

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "w_mastery": self.p.w_mastery, "w_social": self.p.w_social,
            "w_affect": self.p.w_affect, "w_prior": self.p.w_prior,
            "prior": self.p.prior, "alpha": self.p.alpha, "beta": self.p.beta,
            "recency_half_life_days": self.p.recency_half_life_days,
            "social": self.p.social,
            "overconfident_gap": self.p.overconfident_gap,
            "underconfident_gap": self.p.underconfident_gap,
            "min_evidence": self.p.min_evidence,
        }


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def demo(learner_id: str = "demo-confidence") -> LearnerModel:
    """A learner with violent overconfidence: mostly-wrong, high affect."""
    from datetime import timedelta

    model = LearnerModel(learner_id=learner_id)
    now = datetime.now(timezone.utc)
    for days, outcome, affect in [
        (0, 0.0, "confident"), (3, 0.0, "confident"), (7, 1.0, "engaged"),
    ]:
        model.observe(Interaction(
            concept_id="fractions", kind="answer", outcome=outcome, affect=affect,
            ts=(now - timedelta(days=days)).isoformat()))
    return model