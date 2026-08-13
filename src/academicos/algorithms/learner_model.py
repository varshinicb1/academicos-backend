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
    """Per-concept knowledge state."""
    concept_id: str
    attempts: int = 0
    correct: int = 0
    mastery: float = 0.0           # 0..1 composite
    confidence: float = 0.0        # 0..1 recency-weighted accuracy
    retention: float = 1.0         # 0..1 memory-strength estimate
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
