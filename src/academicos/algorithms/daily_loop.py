"""P5.1 — The daily loop (phase-5 orchestrator).

For a learner's stored state, produce **the day's session**: what to
remember (what is drifting / forgotten), whether the learner is tired,
unmotivated or miscalibrated — then turn all of that into a capacity-bounded
plan of concrete session blocks.

This module is an *orchestrator*, not a reimplementation. Each signal is
delegated to an already-validated algorithm:

| Signal                       | Module                                     | Contribution                                |
|------------------------------|--------------------------------------------|---------------------------------------------|
| retention / forgotten        | `forgetting.ForgettingModel` (FSRS-6 v2)  | `retention_now` per concept; below `min_retention` → the concept is *forgotten* |
| due / learn ordering        | `study_planner.StudyPlanner` (A12)         | prerequisite order + capacity for brand-new concepts |
| confidence calibration      | `confidence_model.ConfidenceModel` (A13)   | over/under-confidence (efficacy − accuracy)      |
| attention / fatigue         | `attention_model.AttentionModel` (A4)      | a *fatigued* learner starts with a reset (break) |
| habit consistency           | `habit_model.HabitModel` (A5)              | a weak 7-day consistency adds a persistence snippet |
| motivation                  | `motivation_model.MotivationModel` (A3)    | a low motivation score adds a targeted snippet |

Determinism contract: no randomness anywhere; the wall-clock is only used
for the default *date* when no `now` is injected. Provide `now=` for fully
reproducible output. When `now is None` the loop falls back to the learner's
most recent event timestamp — every computation stays deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..graph.store import GraphStore
from .attention_model import AttentionModel
from .confidence_model import ConfidenceModel, ConfidenceParams
from .forgetting import ForgettingModel, ForgettingParams
from .habit_model import HabitModel
from .learner_model import Interaction, LearnerModel
from .motivation_model import MotivationModel
from .study_planner import StudyPlanner

KIND_REVIEW = "review"
KIND_RETRIEVE = "retrieve"
KIND_LEARN = "learn"
KIND_PRACTICE = "practice"
KIND_RECAP = "recap"
KIND_RESET = "reset"
BLOCK_KINDS = (KIND_REVIEW, KIND_RETRIEVE, KIND_LEARN, KIND_PRACTICE,
               KIND_RECAP, KIND_RESET)

# Larger = earlier in the day. A finer urgency offset is added on top (e.g.
# the deeper the forgetting, the closer to the top the retrieve block lands).
_KIND_PRIORITY: dict[str, float] = {
    KIND_RESET: 100.0,
    KIND_RETRIEVE: 90.0,
    KIND_PRACTICE: 85.0,
    KIND_LEARN: 60.0,
    KIND_REVIEW: 40.0,
    KIND_RECAP: 20.0,
}

# Trim weights: when blocks do not fit the day's budget, the *score* of a
# block is `priority * weight[kind]`; the lowest scored blocks are dropped
# first. Reset and retention blocks are protected most.
_DEFAULT_TRIM_WEIGHTS: dict[str, float] = {
    KIND_RESET: 2.0,
    KIND_RETRIEVE: 1.6,
    KIND_PRACTICE: 1.4,
    KIND_LEARN: 1.2,
    KIND_REVIEW: 1.0,
    KIND_RECAP: 0.8,
}

_DEFAULT_CAP_REASON = (
    "Session-capacity enforced: {kept}/{total} blocks scheduled "
    "({capacity} min) — the rest defer to another day."
)

# Only recent "in-session" activity can make a learner look *fatigued*.
# Old history is history, not present fatigue; events outside this window do
# not contribute (so a learner who last studied a month ago is not sent to
# "reset" for that reason).
_ATTENTION_WINDOW_MINUTES = 120.0

# Targeted motivation/persistence snippets (deterministic, no templates).
_SNIPPET_MOTIVATION = (
    "Motivation is low today — aim for one short block, not a perfect "
    "session. One tick is a win."
)
_SNIPPET_HABIT = (
    "Your 7-day study consistency is low — a single session today keeps the "
    "attempt alive and feeds the habit."
)


@dataclass
class DailyLoopParams:
    """Tunable constants for the daily loop (all deterministic).

    * ``cap_reason``     — the trimmed-capacity advice message; accepts
      ``{kept}``, ``{total}`` and ``{capacity}`` placeholders.
    * ``trim_weights``   — per-kind weights used when the day does not fit;
      the amendment keeps *retention* work before *new* work.
    * ``min_retention``  — retention below this marks a concept *forgotten*
      (retrieve / practice block).
    * ``forgetting_threshold`` — retention below this marks a concept *due*
      (review block); mirrors the forgetting model's ``threshold``.
    * ``forgiveness_margin`` — slack band below ``min_retention`` inside
      which a concept is *not* counted as forgotten (avoids flapping).
    * ``confidence_overconfident_gap`` / ``_underconfident_gap`` — hoisted
      from the A13 confidence model so the orchestrator can calibrate
      calibration without touching that module.
    """
    model: str = "daily-loop"
    cap_reason: str = _DEFAULT_CAP_REASON
    trim_weights: dict = field(default_factory=lambda: dict(_DEFAULT_TRIM_WEIGHTS))

    min_retention: float = 0.35
    forgetting_threshold: float = 0.60
    forgiveness_margin: float = 0.0
    minutes_per_block: int = 5
    confidence_gate: float = 0.60          # mastery above = "you know this"
    attention_window_minutes: float = _ATTENTION_WINDOW_MINUTES

    forgetting_model: str = "fsrs"         # "fsrs" | "exponential"

    confidence_overconfident_gap: float = 0.15
    confidence_underconfident_gap: float = -0.15


@dataclass
class SessionBlock:
    """One concrete chunk of the day's session.

    ``kind`` ∈ ``review | retrieve | learn | practice | recap | reset``;
    ``source`` is a ``concept_id`` or ``"focus_area"`` (for learner-level
    blocks like a break); ``priority`` is the ordering (larger first);
    ``action`` is the noun-verb for the UI; ``reason`` is the evidence.
    """
    priority: float
    kind: str
    source: str
    reason: str
    action: str
    suggested_minutes: int
    motivation_note: str | None = None

    def as_dict(self) -> dict:
        return {
            "priority": self.priority,
            "kind": self.kind,
            "source": self.source,
            "reason": self.reason,
            "action": self.action,
            "suggested_minutes": self.suggested_minutes,
            "motivation_note": self.motivation_note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionBlock":
        return cls(
            priority=float(d["priority"]),
            kind=d["kind"],
            source=d["source"],
            reason=d["reason"],
            action=d["action"],
            suggested_minutes=int(d["suggested_minutes"]),
            motivation_note=d.get("motivation_note"),
        )


@dataclass
class DailyPlan:
    """The whole day's session for one learner (P5.1 output)."""
    date_iso: str
    learner_id: str
    session_capacity_minutes: int
    blocks: list[SessionBlock] = field(default_factory=list)
    blocked_because: list[str] = field(default_factory=list)
    motivational_snippets: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    version: str = "1.0"

    @property
    def total_minutes(self) -> int:
        return sum(b.suggested_minutes for b in self.blocks)

    def as_dict(self) -> dict:
        return {
            "date_iso": self.date_iso,
            "learner_id": self.learner_id,
            "session_capacity_minutes": self.session_capacity_minutes,
            "blocks": [b.as_dict() for b in self.blocks],
            "blocked_because": list(self.blocked_because),
            "motivational_snippets": list(self.motivational_snippets),
            "advice": list(self.advice),
            "params": dict(self.params),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailyPlan":
        return cls(
            date_iso=d["date_iso"],
            learner_id=d["learner_id"],
            session_capacity_minutes=int(d["session_capacity_minutes"]),
            blocks=[SessionBlock.from_dict(b) for b in d.get("blocks", [])],
            blocked_because=list(d.get("blocked_because", [])),
            motivational_snippets=list(d.get("motivational_snippets", [])),
            advice=list(d.get("advice", [])),
            params=dict(d.get("params", {})),
            version=d.get("version", "1.0"),
        )


class DailyLoop:
    """P5.1 orchestrator: learner model → ``DailyPlan``.

    Signature (strict)::

        DailyLoop(state_learner, store=None, capacity_minutes=15, now=None,
                  params=None)()  ->  DailyPlan

    ``store`` is optional — when present the A12 ``StudyPlanner`` contributes
    prerequisite ordering to brand-new concepts; when absent the loop falls
    back to the deterministic intra-model ordering so the runner never
    crashes.
    """

    version = "1.0"

    def __init__(self, state_learner: LearnerModel,
                 store: GraphStore | None = None,
                 capacity_minutes: int = 15,
                 now: str | None = None,
                 params: DailyLoopParams | None = None):
        self.learner = state_learner
        self.store = store
        self.capacity_minutes = int(capacity_minutes or 0)
        self.now = now
        self.params = params or DailyLoopParams()
        if self.params.min_retention > self.params.forgetting_threshold:
            raise ValueError(
                "DailyLoopParams.min_retention must be <= forgetting_threshold")

    def __call__(self) -> DailyPlan:
        return self.plan()

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    def plan(self) -> DailyPlan:
        p = self.params
        now_iso = self._effective_now()
        date_iso = self._date_iso(now_iso)

        learner = self.learner
        if not learner.concepts:
            return DailyPlan(
                date_iso=date_iso,
                learner_id=learner.learner_id,
                session_capacity_minutes=self.capacity_minutes,
                blocks=[],
                blocked_because=["insufficient-evidence"],
                motivational_snippets=[],
                advice=["Record a few interactions first so the loop has "
                          "evidence to plan from."],
                params=self._params(now_iso),
                version=self.version,
            )

        events = self._all_events()
        recent = self._recent_events(events, now_iso)
        advice: list[str] = []
        blocked: list[str] = []
        snippets: list[str] = []

        # --- 1) algorithms (deterministic, replay from the learner log) ---
        forgetting = ForgettingModel(ForgettingParams(
            model=p.forgetting_model, threshold=p.forgetting_threshold))
        confidence = ConfidenceModel(ConfidenceParams(
            overconfident_gap=p.confidence_overconfident_gap,
            underconfident_gap=p.confidence_underconfident_gap))

        # --- 2) attention / fatigue (recent in-session activity only) ----
        fatigue = self._fatigue_state(recent, now_iso)

        # --- 3) candidate blocks, one per concept -----------------------
        candidates: list[SessionBlock] = []
        for cid in sorted(learner.concepts):
            st = learner.concepts[cid]
            fr = forgetting.predict_forgetting(learner, cid, now=now_iso)
            cr = confidence.estimate(learner, cid, now=now_iso)
            block = self._block_for(cid, st, fr.retention_now, cr)
            if block.suggested_minutes > 0:
                candidates.append(block)

        # optional A12: prerequisite order for brand-new concepts
        candidates = self._apply_study_order(candidates, now_iso)

        # --- 4) capacity trimming -----------------------------------------
        blocks = self._trim(candidates)

        # --- 5) calibration advice ------------------------------------------
        for cid in sorted(learner.concepts):
            st = learner.concepts[cid]
            cr = confidence.estimate(learner, cid, now=now_iso)
            if (cr.category == "underconfident" and st.attempts > 0
                    and st.mastery >= p.confidence_gate):
                advice.append(
                    f"{cid}: you know this — do one quick win then move on.")
            if (cr.category == "overconfident" and cr.confident
                    and st.attempts >= 3):
                fr = forgetting.predict_forgetting(learner, cid, now=now_iso)
                if fr.retention_now < p.min_retention:
                    advice.append(
                        f"{cid}: overconfident but drifting — practice "
                        "without hints to recalibrate the belief.")

        # --- 6) motivation & habit ------------------------------------------
        motivation = None
        habit = None
        if events:
            motivation = MotivationModel().estimate(learner, now=now_iso)
        if motivation is None or not motivation.flagged_low:
            pass
        if motivation and motivation.flagged_low:
            snippets.append(_SNIPPET_MOTIVATION)
        habit = HabitModel().estimate(events, now_iso=now_iso,
                                      learner_id=learner.learner_id)
        if habit.risk > 0.0 or habit.consistency_7 < 0.4:
            snippets.append(_SNIPPET_HABIT)

        # --- 7) blocked reasons -----------------------------------------------
        if fatigue == "fatigued":
            blocked.append("tired")
        if any(b.kind in (KIND_RETRIEVE, KIND_PRACTICE) for b in blocks):
            blocked.append("forgotten")
        if motivation and motivation.flagged_low:
            blocked.append("low-motivation")

        # --- 8) reset first, then finalize -------------------------------------
        blocks_final = self._blocks_final(blocks)
        if fatigue == "fatigued":
            reset = SessionBlock(
                priority=_KIND_PRIORITY[KIND_RESET],
                kind=KIND_RESET,
                source="focus_area",
                reason="attention model reports 'fatigued'",
                action="reset/break",
                suggested_minutes=min(p.minutes_per_block,
                                      max(1, self.capacity_minutes)),
                motivation_note="Audience-cheer: rest a couple minutes, "
                                  "then come back fresh.",
            )
            if not any(b.kind == KIND_RESET for b in blocks_final):
                blocks_final.insert(0, reset)

        if not blocks_final and fatigue != "fatigued":
            advice.append("You are all caught up — nothing is due today.")

        return DailyPlan(
            date_iso=date_iso,
            learner_id=learner.learner_id,
            session_capacity_minutes=self.capacity_minutes,
            blocks=blocks_final,
            blocked_because=blocked,
            motivational_snippets=_dedupe(snippets),
            advice=_dedupe(advice),
            params=self._params(now_iso),
            version=self.version,
        )

    # ------------------------------------------------------------------ #
    # block construction
    # ------------------------------------------------------------------ #
    def _block_for(self, cid: str, st, retention: float, cr) -> SessionBlock:
        p = self.params
        if st.attempts == 0:
            return SessionBlock(
                priority=_KIND_PRIORITY[KIND_LEARN] + 0.0, kind=KIND_LEARN,
                source=cid, reason="new — no evidence in the event log yet",
                action="learn", suggested_minutes=p.minutes_per_block,
            )

        if retention < p.min_retention - p.forgiveness_margin:
            urgency = min(1.0, (p.min_retention - retention) / p.min_retention)
            if cr.category == "overconfident" and cr.confident:
                return SessionBlock(
                    priority=_KIND_PRIORITY[KIND_PRACTICE] + 5.0 * urgency,
                    kind=KIND_PRACTICE, source=cid,
                    reason=("forgotten but self-efficacy stayed high — "
                            "confidence is out of calibration"),
                    action="practice-without-hints",
                    suggested_minutes=p.minutes_per_block,
                    motivation_note=(
                        "Your confidence runs ahead of the data — prove it "
                        "without hints."),
                )
            return SessionBlock(
                priority=_KIND_PRIORITY[KIND_RETRIEVE] + 5.0 * urgency,
                kind=KIND_RETRIEVE, source=cid,
                reason=("forgotten: retention was below min_retention "
                        f"({retention:.3f})"),
                action="retrieve", suggested_minutes=p.minutes_per_block,
            )

        if retention < p.forgetting_threshold:
            return SessionBlock(
                priority=_KIND_PRIORITY[KIND_REVIEW]
                          + 1.0 * (1.0 - retention),
                kind=KIND_REVIEW, source=cid,
                reason=("due: retention below the forgetting threshold "
                    f"({retention:.3f})"),
                action="review", suggested_minutes=p.minutes_per_block,
            )

        # strong concept: nothing to do, keep the day minimal.
        return SessionBlock(
            priority=_KIND_PRIORITY[KIND_RECAP] + 0.0, kind=KIND_RECAP,
            source=cid, reason="retention above threshold (up to date)",
            action="pass", suggested_minutes=0,
        )

    def _apply_study_order(self, candidates: list[SessionBlock],
                           now_iso: str) -> list[SessionBlock]:
        """Reuse A12's StudyPlanner to order brand-new concepts by prerequisite
        precedence when a graph store is available. Best-effort: any planner
        failure degrades to the deterministic in-loop order."""
        if self.store is None:
            return candidates
        learn_ids = [b.source for b in candidates if b.kind == KIND_LEARN]
        if not learn_ids:
            return candidates
        try:
            focus = self._focus_label(learn_ids)
            if focus is None:
                return candidates
            sp = StudyPlanner(self.store,
                              ForgettingModel(ForgettingParams(
                                  model=self.params.forgetting_model,
                                  threshold=self.params.forgetting_threshold)))
            study = sp.plan(focus, capacity_minutes=self.capacity_minutes,
                            learner=self.learner)
            order: dict[str, int] = {}
            for item in study.session:
                order[item.concept_id] = item.position
                order[item.label] = item.position
            if order:
                def pos(b: SessionBlock) -> int:
                    return order.get(b.source, len(order) + 1)
                candidates.sort(key=lambda b: (b.kind != KIND_LEARN,
                                               pos(b), b.source))
        except Exception:
            pass  # never crash the loop run
        return candidates

    def _focus_label(self, learn_ids: list[str]) -> str | None:
        # Prefer the most recently touched candidate; else the first in a
        # deterministic (sorted) order.
        best, best_ts = None, None
        for cid in learn_ids:
            st = self.learner.concepts.get(cid)
            if st and st.last_seen:
                ts = _st_ts(st.last_seen)
                if best_ts is None or ts > best_ts:
                    best, best_ts = cid, ts
        if best is None and learn_ids:
            best = learn_ids[0]
        return best

    def _trim(self, candidates: list[SessionBlock]) -> list[SessionBlock]:
        p = self.params
        cap = max(0, self.capacity_minutes)
        scored = sorted(
            candidates,
            key=lambda b: (b.priority * p.trim_weights.get(b.kind, 1.0)),
            reverse=True)
        kept: list[SessionBlock] = []
        used = 0
        dropped = 0
        for b in scored:
            if b.suggested_minutes <= 0:
                continue
            if used + b.suggested_minutes <= cap:
                kept.append(b)
                used += b.suggested_minutes
            else:
                dropped += 1
        kept.sort(key=lambda b: b.priority, reverse=True)
        return kept

    # ------------------------------------------------------------------ #
    # signals
    # ------------------------------------------------------------------ #
    def _fatigue_state(self, events, now_iso: str) -> str:
        if not events:
            return "focused"
        result = AttentionModel().estimate(interactions=events, now=now_iso)
        return result.fatigue_state

    def _recent_events(self, events, now_iso: str):
        window_secs = self.params.attention_window_minutes * 60.0
        now_dt = _dt(now_iso)
        return [i for i in events
                if (now_dt - _dt(i.ts)).total_seconds() <= window_secs]

    def _all_events(self):
        events = [i for c in self.learner.concepts.values()
                  for i in (c.history or [])]
        return sorted(events, key=lambda i: _dt(i.ts))

    def _blocks_final(self, blocks: list[SessionBlock]) -> list[SessionBlock]:
        return sorted((b for b in blocks if b.suggested_minutes > 0),
                      key=lambda b: b.priority, reverse=True)

    # ------------------------------------------------------------------ #
    # time helpers (deterministic; wall clock only as a last-resort)
    # ------------------------------------------------------------------ #
    def _last_fs(self) -> str | None:
        tss = [i.ts for c in self.learner.concepts.values()
               for i in (c.history or [])]
        return max(tss, key=lambda iso: _dt(iso)) if tss else None

    def _effective_now(self) -> str:
        if self.now:
            return self.now
        last = self._last_fs()
        if last:
            return last
        return datetime.now(timezone.utc).isoformat()

    def _date_iso(self, now_iso: str | None) -> str:
        try:
            return _dt(now_iso or "").date().isoformat()
        except (TypeError, ValueError):
            return date.today().isoformat()

    def _params(self, now_iso: str) -> dict:
        p = self.params
        return {
            "model": p.model,
            "capacity_minutes": self.capacity_minutes,
            "min_retention": p.min_retention,
            "forgetting_threshold": p.forgetting_threshold,
            "forgiveness_margin": p.forgiveness_margin,
            "minutes_per_block": p.minutes_per_block,
            "confidence_gate": p.confidence_gate,
            "attention_window_minutes": p.attention_window_minutes,
            "forgetting_model": p.forgetting_model,
            "confidence_overconfident_gap": p.confidence_overconfident_gap,
            "confidence_underconfident_gap": p.confidence_underconfident_gap,
            "trim_weights": dict(p.trim_weights),
            "cap_reason": p.cap_reason,
            "version": self.version,
        }


def _block_from_dict(d: dict) -> SessionBlock:
    return SessionBlock.from_dict(d)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _st_ts(iso: str) -> float:
    return _dt(iso).timestamp()