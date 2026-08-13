"""A4 — In-session attention / fatigue predictor (time-on-task).

Deterministic model of how sustained attention decays within a single
learning session. Two grounding inputs:

  * **time-on-task.** The sustained-attention literature (Anderson's
    "psychological time"; classic 10-20 minute saturation caps and the
    20-25 minute Pomodoro-type windows) puts effortful, contiguous
    attention well under 30 minutes before it needs a repair. We treat
    attention as a budget that drains with in-session elapsed time.

  * **affect.** Baker, Rodrigo, Xoloch, D'Mello & Graesser (2010),
    *The Relationship Between Online Learning and Learner Affect*
    (Int. J. Human-Computer Studies 68) show that reported
    frustrated/bored states co-occur with disengagement and dropout
    in tutoring systems. Yerkes-Dodson (1908, *"The relation of strength
    of stimulus to rapidity of habit-formation"*) adds the inverted-U:
    moderate arousal (engaged/neutral) sustains best performance;
    under-arousal (bored) and over-arousal (anxious, frustrated,
    stressed) degrade it. Both therefore predict *when* a learner
    should break — the content of this model.

Mechanics. Each affect tag maps to a hazard multiplier (see
`FatigueParams.hazard_rate_others`); the multiplier acts on *decay* so a
multiplier > 1.0 (tired, frustrated, anxious) **shrinks** the attention
budget and a multiplier < 1.0 (engaged, happy) **stretches** it. The
attention budget for a point is::

    H = max_attention_minutes / hazard_multiplier                # minutes
    a(t) = max(0, 1 - (t / H) ** attention_exponent)             # 0..1

with `t` = minutes since `session_start`, plus a monotone clamp so an
in-session curve never rises even if affect improves mid-session. Verdict
bands follow the attention ladder: `focused`, `distracted`, `waning`,
`fatigued`.

Pure, deterministic, offline — same contract as the other algorithms in
this library (version, research-shaped defaults, `_params`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .learner_model import Interaction

_DEFAULT_HAZARD: dict[str, float] = {
    "tired": 1.5, "frustrated": 1.8, "bored": 1.4, "anxious": 1.2,
    "neutral": 1.0, "engaged": 0.8, "happy": 0.7,
}
_DISTRACTED_AT = 0.7    # attention below this leaves "focused"
_WANING_AT = 0.5        # attention below this drops "distracted"
_FATIGUED_AT = 0.2      # attention below this is "fatigued"


@dataclass
class FatigueParams:
    """Calibratable constants for the attention decay model."""
    model: str = "time_on_air"
    max_attention_minutes: float = 25.0       # fresh budget before decay (any pose)
    decay_half_life_minutes: float = 20.0     # reference recovery/recharge half-time
    attention_exponent: float = 1.7             # how sharply a(t) falls at the edge
    hazard_rate_others: dict = field(
        default_factory=lambda: dict(_DEFAULT_HAZARD))
    # hazard multipliers multiply the *decay*: >1 shrinks the budget, <1 stretches it.


@dataclass
class CurvePoint:
    """One sampled attention observation in a session."""
    t_minutes: float          # minutes elapsed since session start
    attention: float          # 0..1 residual attention
    verdict: str              # focused | distracted | waning | fatigued
    affect_factor: float      # average hazard multiplier up to this point


@dataclass
class SessionAttentionParams:
    """Derived, per-session scalars (from FatigueParams + the affect mix)."""
    session_id: str | None = None
    hazard_multiplier: float = 1.0          # average decay-multiplier of the session
    hazard_scaled_max_minutes: float = 25.0  # max_attention / hazard_multiplier
    exposure_minutes: float = 0.0           # elapsed uptime actually logged
    attention_at_endpoint: float = 1.0      # residual attention at the current instant
    rest_needed_minutes: float = 0.0        # recommended break to recover


@dataclass
class AttentionResult:
    """One session's attention trace and the current fatigue assessment."""
    session_id: str | None = None
    computed_uptime_minutes: float = 0.0
    attention_curve: list = field(default_factory=list)   # list[CurvePoint]
    peak_attention_window: tuple[int, str] = (0, "focus")
    fatigue_state: str = "focused"          # focused | waning | resting | fatigued
    params: dict = field(default_factory=dict)


def _hazard(p: FatigueParams, affect: str | None) -> float:
    if not affect:
        return 1.0
    key = affect.lower().strip()
    h = p.hazard_rate_others.get(key, 1.0)
    return h if h is not None and h > 0 else 1.0


def _avg_hazard(p: FatigueParams, affects: list[str | None]) -> float:
    if not affects:
        return 1.0
    vals = [_hazard(p, a) for a in affects]
    return sum(vals) / len(vals)


def _verdict(attention: float) -> str:
    if attention < _FATIGUED_AT:
        return "fatigued"
    if attention < _WANING_AT:
        return "waning"
    if attention < _DISTRACTED_AT:
        return "distracted"
    return "focused"


def _fatigue_state(attention: float) -> str:
    """Four-level state used for the report (verbal set from the contract)."""
    if attention < _FATIGUED_AT:
        return "fatigued"
    if attention < _WANING_AT:
        return "resting"          # needs a repair: fade zone below distracted
    if attention >= _DISTRACTED_AT:
        return "focused"
    return "waning"               # 0.5..0.7: the distracted → waning crossover


def _attention_at(p: FatigueParams, elapsed_minutes: float, hazard: float) -> float:
    """a(t) = max(0, 1 - (t / (max / hazard)) ** exponent), safe for t or hazard 0."""
    if elapsed_minutes <= 0:
        return 1.0
    budget = p.max_attention_minutes / hazard if hazard > 0 else p.max_attention_minutes
    if budget <= 0:
        return 0.0
    ratio = elapsed_minutes / budget
    return max(0.0, 1.0 - ratio ** p.attention_exponent)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


class AttentionModel:
    """In-session attention depletion forecast (deterministic).

    v1 (time_on_air):

        * budget per observation = max_attention_minutes / hazard_multiplier,
          where hazard_multiplier is the cumulative average of the affect
          hazards seen so far in the session (Baker et al. valence coding,
          Yerkes-Dodson cannot be over- or under-aroused without cost).
        * attention(t) = max(0, 1 - (t / budget) ** exponent), elapsed t =
          the timestamp diff vs session_start (in minutes).
        * a monotone clamp keeps the curve non-increasing inside a session.

    The current instant's verdict (fatigue_state) comes from the last point
    of the curve; a "way beyond max" session lands in "fatigued". There is
    no wall-clock dependence beyond the timestamps/`now` you pass in.
    """
    version = "1.0"

    def __init__(self, params: FatigueParams | None = None):
        self.p = params or FatigueParams()

    def estimate(self, session_start: str | None = None,
                 interactions: list | None = None,
                 now: str | None = None) -> AttentionResult:
        interactions = interactions or []
        if session_start is None:
            if not interactions:
                return AttentionResult(
                    computed_uptime_minutes=0.0,
                    attention_curve=[CurvePoint(0.0, 1.0, "focused", 1.0)],
                    peak_attention_window=(0, "focused"),
                    fatigue_state="focused",
                    params=self._params(),
                )
            session_start = interactions[0].ts
        start = _ts(session_start)

        # chronological in-session points, with elapsed in minutes
        ordered = sorted(interactions, key=lambda i: _ts(i.ts))
        points: list[dict] = []
        affects_seen: list[str | None] = []
        monotone = 1.0
        for i in ordered:
            elapsed = max(0.0, (_ts(i.ts) - start) / 60.0)
            affects_seen.append(i.affect)
            fac = _avg_hazard(self.p, affects_seen) if affects_seen else 1.0
            fac = round(fac, 4)
            att = _attention_at(self.p, elapsed, fac)
            att = min(att, monotone)      # never rise inside the curve
            monotone = att
            points.append({
                "t": round(elapsed, 2), "att": round(att, 4), "fac": fac,
                "verdict": _verdict(att),
            })

        # bound the current instant with an optional `now` extrapolation
        if now is not None and _ts(now) >= start:
            elapsed_now = max(0.0, (_ts(now) - start) / 60.0)
            if not points or elapsed_now > points[-1]["t"]:
                fac = points[-1]["fac"] if points else _avg_hazard(self.p, [])
                att = min(_attention_at(self.p, elapsed_now, fac),
                          monotone if points else 1.0)
                points.append({
                    "t": round(elapsed_now, 2), "att": round(att, 4),
                    "fac": round(fac, 4), "verdict": _verdict(att),
                })

        last_t = points[-1]["t"] if points else 0.0
        last_att = points[-1]["att"] if points else 1.0
        last_fac = points[-1]["fac"] if points else 1.0

        # peak = the span where attention is still >= "focused"
        start_hazard = _avg_hazard(self.p, [i.affect for i in ordered]) or 1.0
        budget = (self.p.max_attention_minutes / start_hazard
                  if start_hazard > 0 else 0.0)
        if budget > 0:
            peak_m = budget * (1.0 - _DISTRACTED_AT) ** (
                1.0 / self.p.attention_exponent)
        else:
            peak_m = 0.0
        peak_window = (int(peak_m), "focused")

        rest = max(0.0, self.p.decay_half_life_minutes
                   * max(_DISTRACTED_AT - last_att, 0.0) / _DISTRACTED_AT)
        rest = round(rest, 2)

        local = SessionAttentionParams(
            hazard_multiplier=last_fac,
            hazard_scaled_max_minutes=round(
                (self.p.max_attention_minutes / last_fac) if last_fac
                else self.p.max_attention_minutes, 2),
            exposure_minutes=round(last_t, 2),
            attention_at_endpoint=last_att,
            rest_needed_minutes=rest,
        )

        return AttentionResult(
            computed_uptime_minutes=round(last_t, 2),
            attention_curve=[CurvePoint(t_minutes=pt["t"], attention=pt["att"],
                                        verdict=pt["verdict"], affect_factor=pt["fac"])
                             for pt in points],
            peak_attention_window=peak_window,
            fatigue_state=_fatigue_state(last_att),
            params=dict(self._params(), **{
                "session_hazard": round(local.hazard_multiplier, 4),
                "hazard_scaled_max_minutes": local.hazard_scaled_max_minutes,
                "rest_needed_minutes": local.rest_needed_minutes,
            }),
        )

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "max_attention_minutes": self.p.max_attention_minutes,
            "decay_half_life_minutes": self.p.decay_half_life_minutes,
            "attention_exponent": self.p.attention_exponent,
            "hazard_rate_others": dict(self.p.hazard_rate_others),
        }