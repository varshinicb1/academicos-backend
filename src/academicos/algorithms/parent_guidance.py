"""Algorithm A8 — Parent / caretaker "say-this-today" guidance.

Deterministic "one breadbasket per day" messages for a parent or caretaker,
generated from behavioral signals. Two evidence strands, both deliberately
bounded:

  * Dweck (mindset) — praise the *process* (effort, strategy, persistence),
    not the person or easy wins. Therefore praise only triggers when mastery
    is real (``accuracy >= praise_threshold``) AND extra effort was signaled
    or the concept is genuinely hard. Never praise an easy win.
  * Scaffolding (Wood, Bruner & Ross 1975; Vygotsky's zone of proximal
    development) — when the learner is clearly below mastery with only a
    couple of attempts, the caretaker's job is to sit beside, not to grade.

Output is all that can fit in a parent's pocket: at most ``max_messages``
messages, ordered by an internal urgency field (safety/health > comfort/
reassure > growth).

Pure, deterministic, no IO. Same contract as the other algorithms
(``model``-tagged params, ``_params`` snapshot, versioned).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .learner_model import LearnerModel

COMPONENTS = ("positive", "praise", "scaffold", "comfort", "attention")

# Urgency tiers used for deterministic ordering (lower fires first).
_PRIORITY_BREAK = 1      # safety / health
_PRIORITY_REASSURE = 2   # comfort / connect
_PRIORITY_ENCOURAGE = 3  # growth: motivation
_PRIORITY_PRAISE = 4     # growth: earned praise
_PRIORITY_FOCUS = 5      # growth: attention environment
_PRIORITY_SCAFFOLD = 6   # growth: study-together proposal

_VALENT = "frustrated"


@dataclass(frozen=True)
class GuidanceTemplate:
    """One catalog row: how to render, how urgent, how much parent time."""
    component: str
    priority: int
    hours: float
    text: str
    description: str = ""


CATALOG: dict[str, GuidanceTemplate] = {
    "break_fatigue": GuidanceTemplate(
        component="comfort", priority=_PRIORITY_BREAK, hours=0.0,
        text="{concept} hit fatigue fast today. A real 10-minute break plus a shorter focus block beats pushing through.",
        description="stop-and-rest on exhaustion (tired/fatigued affect or fatigue flag).",
    ),
    "break_frustration": GuidanceTemplate(
        component="comfort", priority=_PRIORITY_BREAK, hours=0.0,
        text="{concept} produced a burst of frustration back-to-back. Pause the practice and step out for ten minutes.",
        description="very recent high frustration count (>= break_frustration_threshold).",
    ),
    "reassure_single_spike": GuidanceTemplate(
        component="comfort", priority=_PRIORITY_REASSURE, hours=0.05,
        text="A single off moment in {concept} — say 'everyone has a tough five minutes, tomorrow is normal' and leave the lesson for later.",
        description="one-off frustration spike (count 1..2), not a pattern.",
    ),
    "encourage_rising": GuidanceTemplate(
        component="positive", priority=_PRIORITY_ENCOURAGE, hours=0.1,
        text="{concept} is trending up after a rocky patch. Point at the effort: 'I saw you keep trying.'",
        description="frustration followed by a rising trend — process encouragement.",
    ),
    "praise_effort": GuidanceTemplate(
        component="praise", priority=_PRIORITY_PRAISE, hours=0.25,
        text="{concept} took real effort and the accuracy stuck. Say the process, not the person: you kept going when it was hard.",
        description="high accuracy + explicit effort signal (frustration carried through). Dweck process-praise.",
    ),
    "praise_difficulty": GuidanceTemplate(
        component="praise", priority=_PRIORITY_PRAISE, hours=0.25,
        text="{concept} is normally a hard concept and it went well. Notice the strategy that made that happen.",
        description="high accuracy on a concept flagged high-difficulty (no easy-win praise).",
    ),
    "focus_attention": GuidanceTemplate(
        component="attention", priority=_PRIORITY_FOCUS, hours=0.17,
        text="Attention flagged on {concept}. Set a phone-away environment and a visible 10-minute timer.",
        description="attention signal below attention_threshold.",
    ),
    "scaffold_study_together": GuidanceTemplate(
        component="scaffold", priority=_PRIORITY_SCAFFOLD, hours=0.5,
        text="Offer to sit down and do one part of {concept} together next time — a couple of attempts only.",
        description="low accuracy with very few attempts (Vygotsky ZPD: co-acting before solo).",
    ),
}

# Documented deterministic rule table (template_key -> condition).
RULE_TABLE: list[dict[str, str]] = [
    {"template_key": "break_fatigue", "condition": "fatigue flag OR 'tired'/'fatigued' in recent_affect"},
    {"template_key": "break_frustration", "condition": "recent_frustrated_count >= break_frustration_threshold"},
    {"template_key": "reassure_single_spike", "condition": "1 <= recent_frustrated_count < break_frustration_threshold AND NOT rising"},
    {"template_key": "encourage_rising", "condition": "rising trend AND recent_frustrated_count >= 1 AND accuracy >= effort_win_threshold"},
    {"template_key": "praise_effort", "condition": "accuracy >= praise_threshold AND attempts >= praise_min_attempts AND recent_frustrated_count in 1..2 AND (frustration carried through)"},
    {"template_key": "praise_difficulty", "condition": "accuracy >= praise_threshold AND attempts >= praise_min_attempts AND concept_difficulty >= high_difficulty"},
    {"template_key": "(none)", "condition": "accuracy >= praise_threshold but NO effort signal AND difficulty < high_difficulty -> FIRES NOTHING (easy win)"},
    {"template_key": "focus_attention", "condition": "attention < attention_threshold"},
    {"template_key": "scaffold_study_together", "condition": "accuracy < scaffold_accuracy AND 1 <= attempts <= scaffold_max_attempts"},
]


@dataclass
class GuidanceParams:
    """Calibratable constants for the parent-guidance rule set."""
    model: str = "guidance"
    praise_threshold: float = 0.7          # accuracy bar for a REAL praise
    high_difficulty: float = 0.7           # concept_difficulty bar for the "hard win" praise
    effort_win_threshold: float = 0.3      # accuracy floor for an "effort win" encourage
    scaffold_accuracy: float = 0.4         # accuracy below this -> scaffolding zone
    scaffold_max_attempts: int = 2         # at most this many attempts for scaffolding
    praise_min_attempts: int = 3           # praise is meaningless on 1-2 samples
    break_frustration_threshold: int = 3   # frustration burst -> break, not reassurance
    attention_threshold: float = 0.4
    concernable: bool = True               # allow comfort/scaffold messages (False: suppress soft ones)
    max_messages: int = 3                  # one "breadbasket" per day (see docs/research/parent-guidance.md)
    render_text: Callable[..., str] | None = None  # override template renderer


@dataclass
class GuidanceMessage:
    """One parent-directed "say this" line for today."""
    target_concept: str
    component: str          # COMPONENTS
    template_key: str       # CATALOG key
    text: str
    hours: float = 0.0


@dataclass
class GuidanceResult:
    learner_id: str
    day_guidance: list[GuidanceMessage] = field(default_factory=list)
    scope: str = "parent:today"
    params: dict = field(default_factory=dict)

    @property
    def messages(self) -> list[GuidanceMessage]:
        return self.day_guidance


class _SafeMapping(dict):
    """format_map helper: never raises KeyError for absent placeholders."""
    def __missing__(self, key):
        return ""


def render_text(template: str, **kwargs: Any) -> str:
    """Fill template named keys with `template.format`; missing keys -> ''."""
    return template.format_map(_SafeMapping(kwargs))


def snapshot_from_learner(model: LearnerModel, recent: int = 3) -> dict:
    """Build the snapshot dict directly from a persistent LearnerModel."""
    data: dict[str, Any] = {
        "learner_id": model.learner_id,
        "accuracy_per_concept": {},
        "attempts": {},
        "recent_affect": {},
        "trend": {},
        "fatigue": {},
    }
    for cid, cs in model.concepts.items():
        n = cs.attempts
        data["attempts"][cid] = n
        data["accuracy_per_concept"][cid] = round(cs.correct / n, 4) if n else 0.0
        affects = [i.affect for i in cs.history if i.affect]
        data["recent_affect"][cid] = affects[-recent:]
        outcomes = [i.outcome for i in cs.history
                    if i.kind == "answer" and i.outcome is not None]
        data["trend"][cid] = _trend_for(outcomes)
        data["fatigue"][cid] = any(
            (a or "").lower() in ("tired", "fatigued", "exhausted") for a in affects)
    return data


class ParentGuidance:
    """Deterministic 'say this today' generator for a parent/caretaker.

    Usage::

        from academicos.algorithms.parent_guidance import ParentGuidance
        guide = ParentGuidance()                      # or ParentGuidance(params)
        result = guide.guide(snapshot)                # -> GuidanceResult

    Snapshot schema (dict)::

        {
          "learner_id": "riya",                          # optional
          "accuracy_per_concept": {"fractions": 0.85},   # required for rules
          "attempts": {"fractions": 4},                  # per-concept samples
          "recent_affect": {"fractions": ["frustrated", "frustrated", "engaged"]},
          "trend": {"fractions": "rising"},              # "rising" | "flat" | "declining"
          "difficulty": {"fractions": 0.75},             # 0..1 concept hardness
          "fatigue": {"fractions": False},               # optional short-circuit for break
          "attention": {"fractions": 0.35},              # optional 0..1 focus quality
          "render_text": callable,                       # optional template renderer
        }

    A `LearnerModel` is also accepted and converted via
    `snapshot_from_learner`. Rules follow the `RULE_TABLE`; messages are
    ordered by urgency (break > reassure > encourage > praise > focus >
    scaffold), then by concept/template, and truncated to `max_messages`.
    """
    version = "1.0"

    def __init__(self, params: GuidanceParams | None = None):
        self.p = params or GuidanceParams()

    def guide(self, snapshot: dict | LearnerModel | None,
              now: Any = None) -> GuidanceResult:
        snap = _coerce(snapshot)
        p = self.p
        render: Callable[..., str] = (
            (p.render_text if p.render_text is not None
             else render_text))

        ac = snap.get("accuracy_per_concept") or {}
        attempts = snap.get("attempts") or {}
        affects = snap.get("recent_affect") or {}
        trend = snap.get("trend") or {}
        difficulty = snap.get("difficulty") or {}
        fatigue = snap.get("fatigue") or {}
        attention = snap.get("attention") or {}

        cids = set(ac) | set(attempts) | set(affects) | set(difficulty) \
            | set(fatigue) | set(attention)
        candidates: list[tuple[int, str, str, GuidanceMessage]] = []

        for cid in sorted(cids):
            num = _int0(attempts.get(cid))
            acc = _f0(ac.get(cid, 0.0))
            affect_tags = [str(x).strip().lower()
                           for x in (affects.get(cid) or []) if x]
            fr = affect_tags.count("frustrated")
            fad = bool(fatigue.get(cid, False)) or \
                any(t in ("tired", "fatigued", "exhausted") for t in affect_tags)
            hard = _f0(difficulty.get(cid, 0.0)) >= p.high_difficulty
            rising = str(trend.get(cid, "")).strip().lower() == "rising"
            att = _f0(attention.get(cid, 1.0))

            if fad or fr >= p.break_frustration_threshold:
                key = "break_fatigue" if fad else "break_frustration"
                self._push(candidates, key, cid, render)
                continue  # a break concept -> no other instructions for it
            if p.concernable and fr >= 1 and fr < p.break_frustration_threshold and not rising:
                self._push(candidates, "reassure_single_spike", cid, render)
            if rising and fr >= 1 and acc >= p.effort_win_threshold:
                self._push(candidates, "encourage_rising", cid, render)
            if acc >= p.praise_threshold and num >= p.praise_min_attempts and fr < p.break_frustration_threshold:
                if fr >= 1:
                    self._push(candidates, "praise_effort", cid, render)
                elif hard:
                    self._push(candidates, "praise_difficulty", cid, render)
            if acc < p.scaffold_accuracy and 1 <= num <= p.scaffold_max_attempts:
                if p.concernable:
                    self._push(candidates, "scaffold_study_together", cid, render)
            if att < p.attention_threshold:
                if p.concernable:
                    self._push(candidates, "focus_attention", cid, render)

        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        chosen = [m for *_, m in candidates[: max(0, p.max_messages)]]
        return GuidanceResult(
            learner_id=str(snap.get("learner_id") or "learner"),
            day_guidance=chosen,
            scope=self._scope(),
            params=self._params(),
        )

    def _push(self, candidates, key, cid, render) -> None:
        tpl = CATALOG[key]
        text = render(tpl.text, concept=cid)
        candidates.append((
            float(tpl.priority), str(cid), key,
            GuidanceMessage(target_concept=cid, component=tpl.component,
                            template_key=key, text=text, hours=tpl.hours),
        ))

    def _scope(self) -> str:
        return "parent:today"

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "praise_threshold": self.p.praise_threshold,
            "high_difficulty": self.p.high_difficulty,
            "effort_win_threshold": self.p.effort_win_threshold,
            "scaffold_accuracy": self.p.scaffold_accuracy,
            "scaffold_max_attempts": self.p.scaffold_max_attempts,
            "praise_min_attempts": self.p.praise_min_attempts,
            "break_frustration_threshold": self.p.break_frustration_threshold,
            "attention_threshold": self.p.attention_threshold,
            "concernable": self.p.concernable,
            "max_messages": self.p.max_messages,
        }


def _coerce(snapshot: Any) -> dict:
    if snapshot is None:
        return {}
    if isinstance(snapshot, dict):
        return snapshot
    if hasattr(snapshot, "concepts"):
        return snapshot_from_learner(snapshot)
    raise TypeError("snapshot must be a dict or a LearnerModel")


def _int0(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _f0(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _trend_for(outcomes: list[float]) -> str:
    """Cheap trend: compare the last few outcomes against the average."""
    if len(outcomes) < 4:
        return "flat"
    head = outcomes[: len(outcomes) // 2]
    tail = outcomes[len(outcomes) // 2 :]
    h = (sum(head) / len(head)) if head else 0.0
    t = (sum(tail) / len(tail)) if tail else 0.0
    if t > h + 0.05:
        return "rising"
    if t < h - 0.05:
        return "declining"
    return "flat"


if __name__ == "__main__":  # pragma: no cover
    from pprint import pprint

    snap = {
        "learner_id": "riya",
        "accuracy_per_concept": {"fractions": 0.85, "algebra": 0.3},
        "attempts": {"fractions": 4, "algebra": 1},
        "recent_affect": {"fractions": ["frustrated", "engaged"], "algebra": []},
        "difficulty": {"fractions": 0.8},
    }
    r = ParentGuidance().guide(snap)
    pprint([(m.template_key, m.target_concept, m.text) for m in r.messages])