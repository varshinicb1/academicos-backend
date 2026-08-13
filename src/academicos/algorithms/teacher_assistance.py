"""Algorithm A9 — Teacher Assistance: class-level mastery aggregation.

Turns a set of per-learner concept snapshots (a whole class) into a single
report a teacher can act on:

  (a) class-average mastery per concept, ranked weakest-first,
  (b) which learners are at risk per concept (accuracy strictly below the
      weak threshold, with enough attempts to be trustworthy),
  (c) a suggested whole-class focus plan — the concepts worth reteaching to
      the group rather than remediating one-on-one.

No learner is ever *labelled* (e.g. "slow", "weak student"): the output is
concept × accuracy aggregates plus the list of learners who would benefit from
correctives *for that concept*. That is the Bloom correctives loop, not a
student-labeling engine.

Deterministic only: no randomness, no clock (``now`` is accepted for API
compatibility and unused). Inputs can be ``LearnerSnapshot`` dataclasses or
any object exposing ``learner_id`` and a ``concepts`` mapping (e.g. the
Phase-5 ``LearnerModel``) — concept states are read by duck typing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .learner_model import LearnerModel

MIN_ATTEMPTS = 3
WEAK_THRESHOLD = 0.55
MASTERY = 0.6
MAX_FOCUS = 5


@dataclass
class ClassParams:
    """Calibratable parameters for the class-level aggregation."""
    model: str = "class_aggregate"
    min_attempts: int = MIN_ATTEMPTS   # attempts a learner needs before their
                                       # accuracy counts for a concept
    weak_threshold: float = WEAK_THRESHOLD  # accuracy strictly below => at risk
    mastery: float = MASTERY           # class_accuracy at/above => whole-class mastered
    max_focus: int = MAX_FOCUS         # top-N concepts in the suggested focus plan


@dataclass
class ConceptSnapshot:
    """Per-concept rating inside a :class:`LearnerSnapshot`."""
    accuracy: float = 0.0    # 0..1 accuracy/rating evidence
    attempts: int = 0


@dataclass
class LearnerSnapshot:
    """A per-learner snapshot for class aggregation."""
    learner_id: str
    concepts: dict[str, ConceptSnapshot] = field(default_factory=dict)


@dataclass
class ConceptRank:
    """One concept row of the class report."""
    concept_id: str
    class_accuracy: float          # mean accuracy over learners with >= min_attempts
    n_students_rated: int
    at_risk: list[str]             # learner ids, sorted, accuracy < weak_threshold
    focus_priority: float          # 0..1, blend of weakness + spread + at-risk mass

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "class_accuracy": self.class_accuracy,
            "n_students_rated": self.n_students_rated,
            "at_risk": list(self.at_risk),
            "focus_priority": self.focus_priority,
        }


@dataclass
class ClassReport:
    """Output of :meth:`TeacherAssistance.report`."""
    concepts: list[ConceptRank] = field(default_factory=list)
    suggested_focus: list[str] = field(default_factory=list)
    n_learners: int = 0            # snapshots submitted (roster size)
    n_rated: int = 0               # learners with >= 1 rated concept
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concepts": [c.as_dict() for c in self.concepts],
            "suggested_focus": list(self.suggested_focus),
            "n_learners": self.n_learners,
            "n_rated": self.n_rated,
            "params": dict(self.params),
        }


def _rating(concept: Any) -> tuple[float, int] | None:
    """Duck-type a ConceptState (LearnerModel) or ConceptSnapshot into
    (accuracy, attempts). Returns None when attempts is absent/zero."""
    attempts = getattr(concept, "attempts", None)
    if attempts is None or attempts <= 0:
        return None
    accuracy = getattr(concept, "accuracy", None)
    if accuracy is None:
        correct = getattr(concept, "correct", None)
        if correct is None:
            return None
        accuracy = correct / attempts if attempts else 0.0
    return (float(accuracy), int(attempts))


def _extract(snapshot: Any) -> tuple[str, list[tuple[str, float, int]]]:
    """From a LearnerSnapshot or LearnerModel, pull (concept, accuracy, attempts)."""
    kinds = []
    concepts = getattr(snapshot, "concepts", None) or {}
    for cid, value in concepts.items():
        rating = _rating(value)
        if rating is not None:
            kinds.append((cid, rating[0], rating[1]))
    return str(getattr(snapshot, "learner_id", "")), kinds


class TeacherAssistance:
    version = "1.0"

    def __init__(self, params: ClassParams | None = None):
        self.p = params or ClassParams()

    def report(self, snapshots: list[LearnerSnapshot] | list[LearnerModel],
               now: Any = None) -> ClassReport:
        """Class-level report over per-learner concept snapshots. ``now`` unused."""
        learners: list[tuple[str, list[tuple[str, float, int]]]] = []
        for s in snapshots:
            learner_id, kinds = _extract(s)
            rated = [k for k in kinds if k[2] >= self.p.min_attempts]
            if rated:
                learners.append((learner_id, rated))

        n_learners = len(snapshots)
        n_rated = len(learners)
        class_size = max(1, n_learners)

        agg: dict[str, dict[str, Any]] = {}
        for learner_id, kinds in learners:
            for cid, accuracy, attempts in kinds:
                d = agg.setdefault(cid, {"acc": [], "at_risk": []})
                d["acc"].append(accuracy)
                if accuracy < self.p.weak_threshold:
                    d["at_risk"].append(learner_id)

        ranks: list[ConceptRank] = []
        for cid in agg:
            d = agg[cid]
            n = len(d["acc"])
            class_accuracy = sum(d["acc"]) / n
            at_risk = sorted(set(d["at_risk"]))
            coverage = n / class_size
            at_risk_fraction = len(at_risk) / max(1, n) if n else 0.0
            priority = (1.0 - class_accuracy) * coverage * at_risk_fraction
            ranks.append(ConceptRank(
                concept_id=cid,
                class_accuracy=round(class_accuracy, 4),
                n_students_rated=n,
                at_risk=at_risk,
                focus_priority=round(priority, 4),
            ))

        ranks.sort(key=lambda r: (-r.focus_priority, r.concept_id))
        focus = [r.concept_id for r in ranks if r.focus_priority > 0][: self.p.max_focus]

        return ClassReport(
            concepts=ranks,
            suggested_focus=focus,
            n_learners=n_learners,
            n_rated=n_rated,
            params=self._params(),
        )

    def _params(self) -> dict[str, Any]:
        return {
            "model": self.p.model,
            "min_attempts": self.p.min_attempts,
            "weak_threshold": self.p.weak_threshold,
            "mastery": self.p.mastery,
            "max_focus": self.p.max_focus,
            "version": self.version,
        }