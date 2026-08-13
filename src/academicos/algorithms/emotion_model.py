"""Algorithm A6 — "emotional_state": deterministic emotion/affect summary.

Summarizes an interaction history into an emotional state, both overall and
per concept. The model is the two-dimensional **valence-arousal** circumplex
of affect (Russell, 1980): every affective experience is a point on a plane
spanned by

  * **valence**  — how pleasant / unpleasant the state feels (0..1),
  * **arousal**  — how energized / activated the state is (0..1).

Each `Interaction.affect` tag collapses to a fixed (valence, arousal) point
through two v1 constant tables (see `params`). The estimator then aggregates
*all* snapshots by plain arithmetic mean, so the result is:

  * **deterministic** — same history, same output, every time,
  * **order-independent** — a shuffled history leaves the means unchanged,
  * **timestamp-blind** — `now` (the injection point for future windowing)
    does not change the aggregate; the mean is a commutative summary.

The overall dominant primary mood is a simple valence band
(`>= positive_threshold` → "positive", `<= negative_threshold` → "negative",
otherwise "neutral"), and the single dominant `affect` tag is picked by
frequency with the "single-item dominance rule": one tag must account for at
least `w_primary` of the tagged snapshots or the state is reported as mixed
(`dominant_affect` is ``None``).

Pure, deterministic, no I/O — the same contract as the other algorithms in
this library (version, params, result dataclass).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .learner_model import Interaction

try:  # reuse the shared affect→valence map (import-safe, no side effects)
    from .confidence_model import _AFFECT_VALENCE
except Exception:  # pragma: no cover - identical local fallback
    _AFFECT_VALENCE = {
        "excited": 1.0, "confident": 0.95, "engaged": 0.9, "happy": 0.85,
        "curious": 0.8, "neutral": 0.6, "bored": 0.45, "surprised": 0.4,
        "tired": 0.3, "frustrated": 0.25, "anxious": 0.2, "stressed": 0.2,
        "angry": 0.15,
    }

# v1 constant arousal table (same spirit as the valence table). Deterministic
# and pinned here, not derived, so the mapping is auditable.
_AFFECT_AROUSAL: dict[str, float] = {
    "excited": 0.9, "angry": 0.9,
    "happy": 0.7, "engaged": 0.7, "curious": 0.7,
    "confident": 0.6,
    "surprised": 0.5, "neutral": 0.5,
    "bored": 0.35,
    "tired": 0.25,
    "stressed": 0.85, "anxious": 0.85,
}

_DEFAULT_VALENCE = 0.6    # neutral (matches the confidence model's default)
_DEFAULT_AROUSAL = 0.5
_POSITIVE_THRESHOLD = 0.55
_NEGATIVE_THRESHOLD = 0.45


@dataclass
class EmotionParams:
    """Calibratable constants for the valence-arousal emotion summary."""
    model: str = "valence_arousal"
    valence: dict[str, float] = field(default_factory=lambda: dict(_AFFECT_VALENCE))
    arousal: dict[str, float] = field(default_factory=lambda: dict(_AFFECT_AROUSAL))
    w_primary: float = 0.6              # single-item dominance rule: a tag is
                                        # "dominant" only if >= this share
    positive_threshold: float = _POSITIVE_THRESHOLD
    negative_threshold: float = _NEGATIVE_THRESHOLD
    default_valence: float = _DEFAULT_VALENCE    # None affect → neutral
    default_arousal: float = _DEFAULT_AROUSAL


@dataclass
class EmotionAccum:
    """Per-concept emotional accumulation."""
    concept_id: str
    n_snapshots: int
    valence: float          # 0..1 mean valence for this concept
    arousal: float          # 0..1 mean arousal for this concept
    dominant_primary: str   # positive | neutral | negative
    dominant_affect: str | None   # most frequent tag, or None if mixed


@dataclass
class EmotionResult:
    learner_id: str
    valence: float                  # 0..1 mean valence of all snapshots
    arousal: float                  # 0..1 mean arousal of all snapshots
    dominant_primary: str           # positive | neutral | negative
    dominant_affect: str | None     # single dominant tag, or None if mixed
    per_concept: dict[str, EmotionAccum]
    positive_ratio: float           # fraction of snapshots with valence >= positive_threshold
    n_snapshots: int
    params: dict = field(default_factory=dict)


class EmotionModel:
    """Summarize an interaction history into an emotion state (Russell, 1980).

    v1 (deterministic):

        each snapshot i (an interaction with an affect tag)  -> (valence_i, arousal_i)
          from the param tables (untagged -> neutral default),
        overall valence  = mean_i valence_i
        overall arousal  = mean_i arousal_i
        dominant_primary = bucket(overall valence):
                               valence >= +th  -> positive
                               valence <= -th  -> negative
                               else            -> neutral
        dominant_affect  = most frequent tag, if its share >= w_primary else None
        positive_ratio   = #(snapshots with valence >= +th) / n_snapshots

    The aggregate is a commutative mean: timestamps and `now` do not change
    any output (time injections are accepted for API parity / future windowing).
    """
    version = "1.0"

    def __init__(self, params: EmotionParams | None = None,
                 learner_id: str = ""):
        self.p = params or EmotionParams()
        self.learner_id = learner_id

    def estimate(self, history_seq: list[Interaction],
                 now: str | None = None) -> EmotionResult:
        _ = now  # aggregate is time-invariant by construction (deterministic)
        snapshots: list[tuple[str, str | None, float, float]] = []
        for it in history_seq:
            if it.affect:
                tag = it.affect.lower().strip()
                val = self.p.valence.get(tag, self.p.default_valence)
                aro = self.p.arousal.get(tag, self.p.default_arousal)
            else:
                tag, val, aro = None, self.p.default_valence, self.p.default_arousal
            snapshots.append((it.concept_id, tag or None, val, aro))

        if not snapshots:
            return EmotionResult(
                learner_id=self.learner_id,
                valence=round(self.p.default_valence, 4),
                arousal=round(self.p.default_arousal, 4),
                dominant_primary="neutral",
                dominant_affect=None,
                per_concept={},
                positive_ratio=0.0,
                n_snapshots=0,
                params=self._params(),
            )

        val_sum = sum(v for _, _, v, _ in snapshots)
        aro_sum = sum(a for _, _, _, a in snapshots)
        n = len(snapshots)
        valence = val_sum / n
        arousal = aro_sum / n
        positive_ratio = sum(
            1.0 for _, _, v, _ in snapshots
            if v >= self.p.positive_threshold) / n

        per_concept = self._accumulate(snapshots)
        tagged = [(t, v, a) for _, t, v, a in snapshots]

        return EmotionResult(
            learner_id=self.learner_id,
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominant_primary=self._bucket(valence),
            dominant_affect=self._dominant_tag(tagged),
            per_concept=per_concept,
            positive_ratio=round(positive_ratio, 4),
            n_snapshots=n,
            params=self._params(),
        )

    def _accumulate(self, snapshots) -> dict[str, EmotionAccum]:
        by_concept: dict[str, list] = {}
        for cid, tag, v, a in snapshots:
            by_concept.setdefault(cid, []).append((tag, v, a))
        out: dict[str, EmotionAccum] = {}
        for cid, rows in by_concept.items():
            vals = [r[1] for r in rows]
            aros = [r[2] for r in rows]
            mean_v = sum(vals) / len(vals)
            out[cid] = EmotionAccum(
                concept_id=cid,
                n_snapshots=len(rows),
                valence=round(mean_v, 4),
                arousal=round(sum(aros) / len(aros), 4),
                dominant_primary=self._bucket(mean_v),
                dominant_affect=self._dominant_tag(rows),
            )
        return out

    def _bucket(self, mean_valence: float) -> str:
        if mean_valence >= self.p.positive_threshold:
            return "positive"
        if mean_valence <= self.p.negative_threshold:
            return "negative"
        return "neutral"

    def _dominant_tag(self, rows: list[tuple]) -> str | None:
        tags = [r[0] for r in rows if r[0]]
        if not tags:
            return None
        counts: dict[str, int] = {}
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        top = max(sorted(counts), key=counts.get)  # ties -> smallest tag
        share = counts[top] / len(tags)
        if share < self.p.w_primary:
            return None
        return top

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "valence": dict(self.p.valence),
            "arousal": dict(self.p.arousal),
            "w_primary": self.p.w_primary,
            "positive_threshold": self.p.positive_threshold,
            "negative_threshold": self.p.negative_threshold,
            "default_valence": self.p.default_valence,
            "default_arousal": self.p.default_arousal,
        }


_DEMO_TS = "2026-01-15T10:00:00+00:00"


def demo(learner_id: str = "demo-emotion") -> list[Interaction]:
    """A morning of mostly-bright affect with one frustration blip."""
    return [
        Interaction(concept_id="fractions", kind="answer", outcome=0.0,
                    affect="frustrated", ts=_DEMO_TS),
        Interaction(concept_id="fractions", kind="answer", outcome=1.0,
                    affect="happy", ts="2024-01-15T10:10:00+00:00"),
        Interaction(concept_id="fractions", kind="answer", outcome=1.0,
                    affect="engaged", ts="2024-01-15T10:20:00+00:00"),
        Interaction(concept_id="decimals", kind="watch", affect="curious",
                    ts="2024-01-15T10:30:00+00:00"),
    ]