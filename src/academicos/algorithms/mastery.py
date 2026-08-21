"""Algorithm 1 — Knowledge Mastery Score.

Replaces raw marks with a mastery % per concept, combining:

  mastery = w_acc * accuracy + w_conf * confidence + w_ret * retention + w_app * application

- accuracy:   long-run proportion correct
- confidence: recency-weighted accuracy (recent evidence dominates)
- retention:  exponential memory decay since last practice
              R = exp(-(now - last_seen) / tau)
- application: whether evidence exists at higher Bloom levels (apply/analyze)

Parameters are research-shaped defaults (v1); calibration against
learning-science findings happens in Phase 2 (docs/research/learning-science.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .learner_model import ConceptState, Interaction, LearnerModel

APPLICATION_LEVELS = {"apply", "analyze", "evaluate", "create"}
MAX_RETENTION = 30.0  # days; beyond this, retention floor


@dataclass
class MasteryParams:
    """Calibratable weights/constants for the mastery formula."""
    w_acc: float = 0.25
    w_conf: float = 0.25
    w_ret: float = 0.30
    w_app: float = 0.20
    recency_half_life_days: float = 7.0
    retention_tau_days: float = 14.0
    retention_floor: float = 0.10
    min_evidence: int = 3  # interactions before mastery is meaningful


@dataclass
class MasteryResult:
    concept_id: str
    mastery: float
    accuracy: float
    confidence: float
    retention: float
    application: float
    evidence_count: int
    confident: bool  # evidence sufficient to trust the score
    params: dict = field(default_factory=dict)


class KnowledgeMastery:
    version = "1.0"

    def __init__(self, params: MasteryParams | None = None):
        self.p = params or MasteryParams()

    def score(self, model: LearnerModel, concept_id: str,
              now: str | None = None) -> MasteryResult:
        state = model.concepts.get(concept_id)
        if state is None or state.attempts == 0:
            return MasteryResult(
                concept_id=concept_id, mastery=0.0, accuracy=0.0,
                confidence=0.0, retention=self.p.retention_floor,
                application=0.0, evidence_count=0, confident=False,
                params=self._params(),
            )
        answers = [i for i in state.history if i.kind == "answer" and i.outcome is not None]
        accuracy = (state.correct / state.attempts) if state.attempts else 0.0

        # confidence: recency-weighted accuracy (exponential decay of old evidence)
        t_now = _ts(now or _now_utc())
        half_life = self.p.recency_half_life_days * 86400.0
        w_sum = c_sum = 0.0
        for i in answers:
            age = max(0.0, t_now - _ts(i.ts))
            w = math.exp(-age / half_life)
            w_sum += w
            c_sum += w * i.outcome
        # outcomes are expected in [0, 1]; clamp anyway so a single out-of-range
        # evaluation can't push a served signal outside the unit interval
        confidence = max(0.0, min(1.0, (c_sum / w_sum) if w_sum > 0 else accuracy))

        # retention: exponential decay since last practice
        last = _ts(state.last_seen or _now_utc())
        elapsed_days = max(0.0, (t_now - last) / 86400.0)
        retention = math.exp(-elapsed_days / self.p.retention_tau_days)
        retention = max(self.p.retention_floor, retention)

        # application: higher-bloom evidence counts toward transfer
        app_levels = [i for i in state.history
                      if i.bloom and i.bloom.lower() in APPLICATION_LEVELS and i.outcome is not None]
        application = max(0.0, min(1.0, (sum(i.outcome for i in app_levels) / len(app_levels))
                                   if app_levels else 0.0))

        mastery = (self.p.w_acc * accuracy + self.p.w_conf * confidence
                   + self.p.w_ret * retention + self.p.w_app * application)
        return MasteryResult(
            concept_id=concept_id,
            mastery=round(max(0.0, min(1.0, mastery)), 4),
            accuracy=round(accuracy, 4),
            confidence=round(confidence, 4),
            retention=round(retention, 4),
            application=round(application, 4),
            evidence_count=len(answers),
            confident=len(answers) >= self.p.min_evidence,
            params=self._params(),
        )

    def _params(self) -> dict:
        return {
            "w_acc": self.p.w_acc, "w_conf": self.p.w_conf,
            "w_ret": self.p.w_ret, "w_app": self.p.w_app,
            "recency_half_life_days": self.p.recency_half_life_days,
            "retention_tau_days": self.p.retention_tau_days,
            "retention_floor": self.p.retention_floor,
            "min_evidence": self.p.min_evidence,
        }


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def demo(learner_id: str = "demo-1") -> LearnerModel:
    """A quick synthetic learner for tests/demos."""
    from datetime import timedelta

    model = LearnerModel(learner_id=learner_id)
    now = datetime.now(timezone.utc)
    for days, outcome, bloom in [
        (0, 1.0, "remember"), (0, 0.0, "remember"),
        (2, 1.0, "understand"), (5, 1.0, "apply"),
        (9, 1.0, "analyze"), (12, 1.0, "apply"),
    ]:
        model.observe(Interaction(
            concept_id="fractions", kind="answer", outcome=outcome, bloom=bloom,
            ts=(now - timedelta(days=days)).isoformat(),
        ))
    return model
