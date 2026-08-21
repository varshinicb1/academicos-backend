"""Shared learner-model schema for the algorithms library.

The learner model is the persistent state every algorithm reads and updates:
per-concept mastery, confidence, retention, session/affect signals, and a
raw interaction log. Kept dependency-light and deterministic so algorithms
stay testable offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Interaction:
    """One observed learning event (an answer attempt, a watch, a session end)."""
    concept_id: str
    kind: str                      # "answer" | "attempt" | "watch" | "session_end" | "affect"
    outcome: float | None = None   # 0.0..1.0 (e.g. correct=1.0, incorrect=0.0, partial)
    bloom: str | None = None       # BloomLevel value if known
    difficulty: float | None = None
    affect: str | None = None      # "frustrated" | "engaged" | "tired" | ...
    duration_sec: float | None = None
    ts: str = field(default_factory=_now)


@dataclass
class ConceptState:
    """Per-concept knowledge state.

    ⚠ ``mastery``/``confidence``/``retention`` are UNPOPULATED CACHES — do
    not read them. ``record()`` below (the only real write path, reached via
    ``LearnerModel.observe`` and ``EventStore.replay``) updates
    ``attempts``/``correct``/``last_seen``/``history`` and nothing else, so on
    every real learner these three keep their defaults (0.0, 0.0, 1.0)
    forever. The sole assigner anywhere is ``forgetting.demo()``, a synthetic
    fixture. A concept answered correctly 22/22 times still reports
    ``mastery == 0.0``.

    Reading them silently degrades a feature to a constant instead of failing
    loudly, which has already caused three real defects (daily_loop's
    underconfident advice gate, RevisionScheduler's importance weighting, and
    ``cli learn replay``'s output — all fixed; see docs/compliance.md).

    Compute the real values at the point of use instead:
        mastery     -> ``mastery.KnowledgeMastery().score(model, cid, now=...)``
        confidence  -> ``confidence_model.ConfidenceModel().estimate(...)``
                       (``.accuracy`` is the recency-weighted skill proxy this
                       field is named for; ``.efficacy`` is self-belief)
        retention   -> ``forgetting.ForgettingModel().predict_forgetting(...)``

    ``retention`` in particular must NEVER be cached: it decays with wall-clock
    time, so a stored value is wrong the moment after it is written. All three
    models take an explicit ``now``, which is why compute-at-use is the design.
    The fields are kept only for backwards compatibility with persisted rows.
    """
    concept_id: str
    attempts: int = 0
    correct: int = 0
    mastery: float = 0.0           # ⚠ cache, never written by record() — see above
    confidence: float = 0.0        # ⚠ cache, never written by record() — see above
    retention: float = 1.0         # ⚠ cache, never written by record() — see above
    last_seen: str | None = None
    history: list[Interaction] = field(default_factory=list)

    def record(self, i: Interaction) -> None:
        self.history.append(i)
        self.attempts += 1
        if i.kind == "answer" and i.outcome is not None:
            self.correct += 1 if i.outcome >= 0.5 else 0
        self.last_seen = i.ts


@dataclass
class LearnerModel:
    """The persistent learner state (Phase 5 engine's core object)."""
    learner_id: str
    concepts: dict[str, ConceptState] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def state(self, concept_id: str) -> ConceptState:
        if concept_id not in self.concepts:
            self.concepts[concept_id] = ConceptState(concept_id=concept_id)
        return self.concepts[concept_id]

    def observe(self, i: Interaction) -> None:
        self.state(i.concept_id).record(i)
        self.updated_at = _now()
