"""Algorithm A11 — Reward / celebration engine (Self-Determination Theory).

Deterministic feedback policy deciding *what* to surface to a learner at a
single moment, from a synthetic, normalized progress snapshot.

Theoretical anchors (see docs/research/reward.md for the full write-up):

  * Self-Determination Theory (Deci & Ryan 1985; Ryan & Deci 2000) — three
    basic needs: autonomy, competence, relatedness. Feedback is designed to
    feed each need rather than substitute for it.
  * Praise-undermines-intrinsic-motivation risk (Lepper et al. 1973; Deci et
    al. 1999 meta-analysis) — contingent, controlling, or over-frequent praise
    shifts perceived locus of causality outward. The engine therefore spends
    "extrinsic-celebration" tokens sparingly and gates them with a cooldown.
  * Positive-reinforcement scheduling — continuous reinforcement extinguishes
    fastest on schedule withdrawal, so high-frequency celebrations must be
    throttled (varied-ratio feel), while pure encouragement (no gift) is never
    throttled.
  * Effort & strategy praise (Mueller & Dweck 1998) supports competence
    without ego-contingency: effort/relatedness acknowledgments are *never*
    cooldown-gated.

Design contract
---------------

Input — `RewardSnapshot` is *synthetic and normalized*: the caller (a
session/engine upper layer) computes each 0..1 float as an average of the raw
signals it observes (per learner), so this engine does no statistics. All
floats are expected in [0, 1]; the engine clamps defensively only.

The `achievement` field is the caller's envelope average (e.g. rolling
correct-rate, mastery-light). `mastery` is the concept/domain mastery the
caller already computed. `streak` / `attempts` are integer counters.
`breakthrough` and `near_miss` mark caller-observed events:
  * breakthrough — first success on a genuinely hard/difficult target,
  * near-miss  — the learner just barely missed a milestone (flag supplied by
    caller, or ignored; the engine does not second-guess mastery deltas).
`autonomy_flag` — learner self-selected the target/route and then achieved
it. `relatedness` — peer-inhalation (e.g. class average). `mode` selects the
feedback channel ("intrinsic" default, or "premi"); it is metadata for the
surface layer and does not change the decision rules.

Decision rules (ordered by priority — first match wins, at most ONE reward):

  1. attempts == 0  -> action rm="none" (no observations, no feedback noise).
  2. frustration or near-miss event -> rm="encourage" (intent=effort).
     NEVER cooldown-gated: reassurance is not an extrinsic gift, so a
     struggling / just-missed learner is always encouraged.
  3. autonomy -> if the learner self-selected (`autonomy` flag) and *kept* at
     it (attempts accumulated) and reached a milestone -> rm="celebrate",
     intent="autonomy". Gated by cooldown like other celebrations.
  4. breakthrough OR mastery >= breakthrough_threshold -> rm="celebrate",
     intent="competence". Gated by cooldown.
  5. achievement+streak+effort blend >= celebration_ceiling ->
     rm="celebrate", intent="competence". Gated by cooldown.
  6. relatedness >= relatedness_threshold -> rm="celebrate", intent=
     "relatedness". NOT gated: peer/relational feedback is intrinsic-support,
     not an object reward, so it is never silenced.
  7. achievement < low_floor (sub-floor stagnation) -> rm="encourage"
     (intent=effort). Never cooldown-gated like all encouragement.
  8. enough attempts/streak to have put in visible effort -> rm="acknowledge",
     intent="effort". NOT gated: process praise is about the work, not a gift.

Cooldown contract (explicit):
  * `celebration_cooldown_days` gates ONLY the competence/autonomy
    celebrations (rules 3/4/5) — the ones that behave like an externally
    administered reward and carry the Deci-style undermining risk.
  * It NEVER gates encourage (rules 2/7) or effort/relatedness acknowledgements
    (6/8): those are informational/intrinsic feedback, so quieting them would
    be counterproductive to the very needs SDT says to satisfy.
  * While gated, the engine returns a single action rm="cooldown" so the
    caller knows a celebration was *warranted but throttled*.

Scheduling & caps:
  - `max_rewards_per_ack` is the budget a caller may spend before it must emit
    an acknowledgement of last reward; the engine itself emits at most ONE
    action per `estimate` call (each event gets at most one reward).

Pure & deterministic: `estimate(snapshot, last_reward_iso, now?)` injects
`now` for cooldown math; no I/O, no clocks otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

ACTION_NONE = "none"
ACTION_COOLDOWN = "cooldown"
ACTION_ACKNOWLEDGE = "acknowledge"
ACTION_ENCOURAGE = "encourage"
ACTION_CELEBRATE = "celebrate"

INTENT_COMPETENCE = "competence"
INTENT_AUTONOMY = "autonomy"
INTENT_RELATEDNESS = "relatedness"
INTENT_EFFORT = "effort"
INTENT_NONE = "none"

# Most celebrations fall within this set for the cooldown gate, while
# effort/relatedness/intrinsic feedback is never silenced.
_GATED_INTENTS = frozenset({INTENT_COMPETENCE, INTENT_AUTONOMY})

_MAX_ACTIONS_PER_CALL = 1  # "reward each event limit 1"


@dataclass
class RewardSnapshot:
    """Synthetic, normalized progress snapshot built by the caller.

    All float fields are expected in [0, 1] (caller-computed averages); the
    engine clamps defensively.
    """
    learner_id: str
    achievement: float = 0.0   # caller's envelope average of progress signals
    mastery: float = 0.0       # existing mastery estimate (0..1)
    attempts: int = 0          # movement-actions / answers / sessions so far
    streak: int = 0            # current consecutive-success (or effort) streak
    breakthrough: bool = False     # first success on a hard/difficult target
    first: bool = False            # "first-ever" milestone flag (informational)
    autonomy_flag: bool = False    # learner self-selected route/mode (SDT autonomy)
    relatedness: float = 0.0       # peer-inhalation / class-avg support
    frustrated: bool = False       # caller-recorded frustration event
    near_miss: bool = False        # caller-recorded near-miss event
    mode: str = "intrinsic"        # "intrinsic" | "premi" — feedback channel

    def __post_init__(self) -> None:
        self.achievement = _clip01(self.achievement)
        self.mastery = _clip01(self.mastery)
        self.relatedness = _clip01(self.relatedness)
        self.attempts = max(0, int(self.attempts))
        self.streak = max(0, int(self.streak))
        if self.mode not in ("intrinsic", "premi"):
            self.mode = "intrinsic"


@dataclass
class RewardParams:
    """Calibratable constants for the celebration/feedback policy."""
    model: str = "sdt-reward"
    favor_effort_ratio: float = 0.6     # weight on effort vs raw achievement
    celebration_ceiling: float = 0.8    # combined score to celebrate competence
    breakthrough_threshold: float = 0.85  # mastery crossing => competence celebrate
    celebration_cooldown_days: float = 2.0  # throttle strong celebrations
    max_rewards_per_ack: int = 3        # budget a caller may spend per ack turn
    low_floor: float = 0.2              # achievement below this -> encourage
    min_streak_celebrate: int = 3
    min_attempts_celebrate: int = 5
    min_streak_acknowledge: int = 2
    min_attempts_acknowledge: int = 2
    relatedness_threshold: float = 0.75


@dataclass
class Reward:
    """One surfacing action for the learner."""
    rm: str                    # celebrate | acknowledge | encourage | cooldown | none
    intent: str                # competence | autonomy | relatedness | effort | none
    message_key: str
    priority: int
    cooldown: bool = False     # True if suppressed because a celebration was warranted but cooling


@dataclass
class RewardResult:
    learner_id: str
    actions: list[Reward] = field(default_factory=list)
    summary: str = ""
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "actions": [_action_dict(a) for a in self.actions],
            "summary": self.summary,
            "params": dict(self.params),
        }

    def __str__(self) -> str:
        return self.summary or "[no action]"


class RewardEngine:
    """Deterministic celebration/feedback policy (A11, model "sdt-reward")."""

    version = "1.0"

    def __init__(self, params: RewardParams | None = None):
        self.p = params or RewardParams()

    def estimate(self, snap: RewardSnapshot, last_reward_iso: str | None = None,
                 now: str | None = None) -> RewardResult:
        """Decide what (if anything) to surface for this snapshot.

        Deterministic: `now` is injected (defaults to UTC now) and
        `last_reward_iso` is an ISO timestamp of the prior celebration.
        """
        t_now = _ts(now or _now_utc())
        in_cooldown = self._in_cooldown(last_reward_iso, t_now)

        actions: list[Reward] = []
        reward = self._decide(snap, in_cooldown)
        if reward is not None:
            actions.append(reward)
        actions = actions[:_MAX_ACTIONS_PER_CALL]

        return RewardResult(
            learner_id=snap.learner_id,
            actions=actions,
            summary=self._summary(actions),
            params=self._params(),
        )

    # ---- decision ---------------------------------------------------------

    def _decide(self, snap: RewardSnapshot, in_cooldown: bool) -> Reward | None:
        # (1) no observations -> keep silence
        if snap.attempts < 1:
            return Reward(rm=ACTION_NONE, intent=INTENT_NONE,
                          message_key="no_evidence", priority=0)

        # (2) frustration / near-miss -> encourage, never gated
        if snap.frustrated or snap.near_miss:
            kind = "near_miss" if snap.near_miss else "frustration"
            return Reward(rm=ACTION_ENCOURAGE, intent=INTENT_EFFORT,
                          message_key=f"encourage_{kind}", priority=90)

        # (3) autonomy: self-selected + kept + milestone reached
        if (snap.autonomy_flag and self._milestone_reached(snap)):
            return self._maybe_gated(
                Reward(rm=ACTION_CELEBRATE, intent=INTENT_AUTONOMY,
                       message_key="celebrate_autonomy", priority=85),
                in_cooldown)

        # (4) breakthrough hard target OR mastery crossing threshold
        if snap.breakthrough or snap.mastery >= self.p.breakthrough_threshold:
            return self._maybe_gated(
                Reward(rm=ACTION_CELEBRATE, intent=INTENT_COMPETENCE,
                       message_key="celebrate_mastery", priority=80),
                in_cooldown)

        # (5) combined achievement + streak + effort ceiling
        effort = self._effort(snap)
        score = self.p.favor_effort_ratio * effort \
            + (1.0 - self.p.favor_effort_ratio) * snap.achievement
        if score >= self.p.celebration_ceiling and snap.streak >= 1:
            return self._maybe_gated(
                Reward(rm=ACTION_CELEBRATE, intent=INTENT_COMPETENCE,
                       message_key="celebrate_achievement", priority=70),
                in_cooldown)

        # (6) relatedness (never gated: relational feedback is intrinsic)
        if snap.relatedness >= self.p.relatedness_threshold:
            return Reward(rm=ACTION_CELEBRATE, intent=INTENT_RELATEDNESS,
                          message_key="celebrate_relatedness", priority=60)

        # (7) sub-floor achievement -> struggling encouragement, never gated
        if snap.achievement < self.p.low_floor:
            return Reward(rm=ACTION_ENCOURAGE, intent=INTENT_EFFORT,
                          message_key="encourage_struggling", priority=55)

        # (8) visible effort -> acknowledge (process praise, never gated)
        if (snap.streak >= self.p.min_streak_acknowledge
                or snap.attempts >= self.p.min_attempts_acknowledge):
            return Reward(rm=ACTION_ACKNOWLEDGE, intent=INTENT_EFFORT,
                          message_key="acknowledge_effort", priority=50)

        return Reward(rm=ACTION_NONE, intent=INTENT_NONE,
                      message_key="no_reward", priority=0)

    def _maybe_gated(self, reward: Reward, in_cooldown: bool) -> Reward:
        """If this celebration is a "warm gift" (competence/autonomy) and the
        learner is inside the celebration cooldown, downgrade it to a "cooldown"
        action instead of rewarding again (avoids de-motivation via
        continuous-reinforcement fatigue)."""
        if in_cooldown and reward.intent in _GATED_INTENTS:
            return Reward(rm=ACTION_COOLDOWN, intent=reward.intent,
                          message_key="cooldown", priority=60, cooldown=True)
        return reward

    # ---- helpers ----------------------------------------------------------

    def _milestone_reached(self, snap: RewardSnapshot) -> bool:
        return (snap.streak >= self.p.min_streak_celebrate
                or snap.mastery >= self.p.breakthrough_threshold
                or snap.breakthrough)

    def _effort(self, snap: RewardSnapshot) -> float:
        eff = max(snap.streak / self.p.min_streak_celebrate,
                  snap.attempts / self.p.min_attempts_celebrate)
        return max(0.0, min(1.0, eff))

    def _in_cooldown(self, last_reward_iso: str | None, t_now: float) -> bool:
        if not last_reward_iso:
            return False
        try:
            last = _ts(last_reward_iso)
        except (ValueError, TypeError):
            return False
        elapsed_days = (t_now - last) / 86400.0
        return elapsed_days < self.p.celebration_cooldown_days

    def _summary(self, actions: list[Reward]) -> str:
        if not actions:
            return "[no action]"
        a = actions[0]
        return f"[{a.rm}/{a.intent}] {a.message_key}"

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "favor_effort_ratio": self.p.favor_effort_ratio,
            "celebration_ceiling": self.p.celebration_ceiling,
            "breakthrough_threshold": self.p.breakthrough_threshold,
            "celebration_cooldown_days": self.p.celebration_cooldown_days,
            "max_rewards_per_ack": self.p.max_rewards_per_ack,
            "low_floor": self.p.low_floor,
            "min_streak_celebrate": self.p.min_streak_celebrate,
            "min_attempts_celebrate": self.p.min_attempts_celebrate,
            "min_streak_acknowledge": self.p.min_streak_acknowledge,
            "min_attempts_acknowledge": self.p.min_attempts_acknowledge,
            "relatedness_threshold": self.p.relatedness_threshold,
        }


def _action_dict(a: Reward) -> dict:
    return {
        "rm": a.rm, "intent": a.intent,
        "message_key": a.message_key, "priority": a.priority,
        "cooldown": a.cooldown,
    }


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def demo(learner_id: str = "demo-reward") -> RewardSnapshot:
    """A perfect, stuttering learner: high achievement + a sustained streak."""
    return RewardSnapshot(
        learner_id=learner_id, achievement=1.0, mastery=1.0,
        attempts=12, streak=8, relatedness=0.85, mode="intrinsic")