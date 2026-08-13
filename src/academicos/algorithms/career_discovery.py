"""A10 — Career Discovery: learner behavior → Holland's RIASEC dimensions.

Holland (1959; *Making Vocational Choices*, 1997) models vocational interest
as six stable dimensions — **Realistic, Investigative, Artistic, Social,
Enterprising, Conventional** ("RIASEC"). People and occupations cluster on
the same six-code map, and engagement with a domain is a well-replicated
behavioral correlate of the corresponding interest type (e.g. sustained
STEM practice clusters on Investigative/Realistic; creative-writing and
performance work clusters on Artistic/Social).

This algorithm maps observed *behavior* — which subjects a learner actually
spends time on, and for how long — deterministically onto the six dimensions:

  * each interaction contributes its concept-subject's RIASEC weights
    accented by **duration** (a learner who *spends heavily* in a subject is
    stimulated by that subject's dimensions);
  * a concept with an unknown subject (no `subject_of`, or the subject is
    missing from the table) contributes **uniform** weights across all six
    dimensions — it never skews the profile;
  * `sim_coef` shrinks every contribution toward that uniform vector
    (regularization: positive values make the profile trust the table less);
  * scores are normalized (raw + normalized to an optional `sum_to`, default
    1), the top-3 become `holland_code`, and `confidence` is evidence /
    `min_evidence`, capped at 1.

**Stance.** Careers are *secondary and informational*. The output is an
enrichment signal for guidance/recommendation surfaces, never a tracker, and
never a stereotype — no fixed occupational labels are attached to a learner,
and nothing here is used to stream or pigeonhole. Pure, deterministic,
offline — same contract as the rest of this library (`version`, params)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .learner_model import Interaction, LearnerModel

RIASEC: tuple[str, ...] = (
    "Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional",
)

# Default subject → RIASEC stimulation table. Seed the classifier: each row is
# a probability-ish weight distribution; a subject not listed resolves to the
# uniform vector below (neutral, equal weights across RIASEC).
SUBJECT_RIASEC: dict[str, dict[str, float]] = {
    "math":           {"Investigative": 0.5, "Conventional": 0.3, "Realistic": 0.2},
    "science":        {"Investigative": 0.6, "Realistic": 0.4},
    "physics":        {"Investigative": 0.6, "Realistic": 0.4},
    "cs":             {"Investigative": 0.6, "Realistic": 0.3, "Conventional": 0.1},
    "language":       {"Social": 0.4, "Artistic": 0.3, "Conventional": 0.3},
    "arts":           {"Artistic": 0.7, "Social": 0.3},
    "music":          {"Artistic": 0.7, "Social": 0.3},
    "social_science": {"Social": 0.5, "Enterprising": 0.3, "Investigative": 0.2},
    "commerce":       {"Enterprising": 0.5, "Conventional": 0.3, "Social": 0.2},
}

_NEUTRAL: dict[str, float] = {d: 1.0 / len(RIASEC) for d in RIASEC}


@dataclass
class InterestParams:
    """Calibratable constants for RIASEC interest profiling."""
    model: str = "riasec"
    subject_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _subject_riasec().items()})
    min_evidence: float = 4.0      # interaction-weights before confident
    sim_coef: float = 0.0          # shrink subject contribution toward uniform
    sum_to: float = 1.0            # normalization target (<=0 keeps raw scores)


def _subject_riasec() -> dict[str, dict[str, float]]:
    return dict(SUBJECT_RIASEC)


@dataclass
class CareerResult:
    learner_id: str
    riasec_scores: dict[str, float]   # normalized  (raw sum to sum_to)
    raw_scores: dict[str, float]      # un-normalized accumulated
    top3: list[tuple[str, float]]     # [(dimension, normalized score)] desc
    holland_code: str                 # the top-3 three-letter code
    confidence: float                 # 0..1 (evidence / min_evidence capped)
    evidence_count: float             # duration-weighted interaction count
    flags: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _interaction_weight(i: Interaction) -> float:
    """Evidence count for an interaction = duration; sub-minute counts as 1."""
    dur = i.duration_sec or 0.0
    return 1.0 if dur <= 0 else max(1.0, dur / 60.0)


class CareerDiscovery:
    """Map observed behavior → six RIASEC interest scores (deterministic)."""

    version = "1.0"

    def __init__(self, params: InterestParams | None = None):
        self.p = params or InterestParams()

    def estimate(self, evidence: list[Interaction] | LearnerModel,
                 subject_of: Callable[[str], str | None] | None = None,
                 now: str | None = None) -> CareerResult:
        if isinstance(evidence, LearnerModel):
            learner_id = evidence.learner_id
            interactions = [i for s in evidence.concepts.values() for i in s.history]
        else:
            learner_id = "list"
            interactions = list(evidence)
        if now is not None:
            interactions = [i for i in interactions if i.ts <= now]

        raw = {d: 0.0 for d in RIASEC}
        evidence_count = 0.0
        seen_unknown = False
        for i in interactions:
            subj = subject_of(i.concept_id) if subject_of else None
            vec = self._vector(subj)
            if not subj or subj not in self.p.subject_weights:
                seen_unknown = True
            w = _interaction_weight(i)
            for d in RIASEC:
                raw[d] += w * vec[d]
            evidence_count += w

        flags: list[str] = []
        raw_scores = {d: round(v, 4) for d, v in raw.items()}
        if not interactions:
            flags.append("insufficient-data")
            return CareerResult(
                learner_id=learner_id,
                riasec_scores={d: 0.0 for d in RIASEC},
                raw_scores=raw_scores, top3=[],
                holland_code="---", confidence=0.0,
                evidence_count=0.0, flags=flags, params=self._params(),
            )

        total = sum(raw.values())
        if self.p.sum_to and self.p.sum_to > 0 and total > 0:
            scores = {d: round(self.p.sum_to * raw[d] / total, 4) for d in RIASEC}
        else:
            scores = raw_scores
        riasec_scores = scores

        top = sorted(RIASEC, key=lambda d: (-scores[d], d))
        top3 = [(d, scores[d]) for d in top[:3]]
        holland_code = "".join(d[0] for d, _ in top3)

        confidence = round(min(1.0, evidence_count / self.p.min_evidence), 4) if self.p.min_evidence else 0.0
        if confidence < 1.0:
            flags.append("low-confidence")
        if seen_unknown:
            flags.append("subject-unknown")
        if total > 0 and max(scores.values()) - min(scores.values()) < 0.05:
            flags.append("flat-profile")

        return CareerResult(
            learner_id=learner_id,
            riasec_scores=riasec_scores, raw_scores=raw_scores,
            top3=top3, holland_code=holland_code,
            confidence=confidence, evidence_count=evidence_count,
            flags=flags, params=self._params(),
        )

    def _vector(self, subject: str | None) -> dict[str, float]:
        row = self.p.subject_weights.get(subject) if subject else None
        if not row:
            return dict(_NEUTRAL)
        sim = self.p.sim_coef
        if sim and sim > 0:
            return {d: round((1.0 - sim) * row.get(d, 0.0) + sim * _NEUTRAL[d], 4)
                    for d in RIASEC}
        return {d: row.get(d, 0.0) for d in RIASEC}

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "dimensions": list(RIASEC),
            "min_evidence": self.p.min_evidence,
            "sim_coef": self.p.sim_coef,
            "sum_to": self.p.sum_to,
            "subject_weights": {k: dict(v) for k, v in self.p.subject_weights.items()},
        }


def demo(learner_id: str = "demo-career") -> LearnerModel:
    """A science-heavy learner: long STEM watches → Investigative top."""
    from datetime import timedelta

    model = LearnerModel(learner_id=learner_id)
    now = datetime.now(timezone.utc)
    for days, concept, dur in [
        (0, "friction", 3600), (1, "circuits", 3600), (2, "algebra", 1800),
        (3, "geometry", 1800), (5, "essay-writing", 600),
    ]:
        model.observe(Interaction(
            concept_id=concept, kind="watch", duration_sec=dur,
            ts=(now - timedelta(days=days)).isoformat()))
    return model