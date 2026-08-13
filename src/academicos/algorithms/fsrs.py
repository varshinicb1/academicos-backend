"""FSRS-6 (Free Spaced Repetition Scheduler) — the DSR memory model.

Faithful, dependency-free port of the FSRS-6 algorithm as published by the
Open Spaced Repetition project (MIT License):
  https://github.com/open-spaced-repetition/ts-fsrs (packages/fsrs/src/algorithm.ts)
  https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

FSRS is the state-of-the-art spaced repetition scheduler (Anki's default
since 2023; validated on 1.2B+ review logs; 30-50% fewer reviews than SM-2
at the same retention). It models each memory trace with three variables:

  D — difficulty (1..10, intrinsic to the material)
  S — stability (days until predicted recall falls to the target retention)
  R — retrievability (current probability of recall)

Forgetting curve (decay = -w[20], factor = e^(ln(0.9)/decay) - 1):

  R(t, S) = (1 + factor * t / S) ^ decay        with R(S) = 0.9

State transitions on a rating g (1=Again, 2=Hard, 3=Good, 4=Easy):

  S0(g)         = max(w[g-1], 0.1)
  D0(g)         = w4 - e^((g-1) * w5) + 1
  D'(D, g)      = clamp(w7 * D0(4) + (1-w7) * (D + (-w6*(g-3)) * (10-D)/9), 1, 10)
  S'_r(D,S,R,g) = S * (1 + e^w8 * (11-D) * S^-w9 * (e^((1-R)*w10) - 1)
                       * (g==2 ? w15 : 1) * (g==4 ? w16 : 1))
  S'_f(D,S,R)   = w11 * D^-w12 * ((S+1)^w13 - 1) * e^(w14*(1-R))     (g==1)
  S'_s(S,g)     = S * max(S^-w19 * e^(w17*(g-3+w18)), 1)             (t==0)

This port keeps the engine deterministic (fuzzing is off — randomness is a
product concern) and maps every AcademicOS Interaction outcome to a rating.

Adapted for AcademicOS: concept difficulty (1..5) from the curriculum graph
seeds the initial difficulty prior; see `seed_difficulty`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# FSRS-6 default weights (open-spaced-repetition/ts-fsrs, default.ts).
# w[20] is the decay; learning/relearning step scheduling is handled by the
# caller (AcademicOS schedules at day granularity), so w0..w3 seed stability.
DEFAULT_W: tuple[float, ...] = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722,
    0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425,
    0.0912, 0.0658, 0.1542,
)

RATING_AGAIN, RATING_HARD, RATING_GOOD, RATING_EASY = 1, 2, 3, 4

S_MIN = 0.1
S_MAX = 36500.0


def compute_decay_factor(w: tuple[float, ...]) -> tuple[float, float]:
    """(decay, factor) with decay = -w[20] and factor = e^(ln(0.9)/decay) - 1."""
    decay = -w[20]
    factor = math.exp(math.log(0.9) / decay) - 1.0
    return decay, factor


def forgetting_curve(w: tuple[float, ...], elapsed_days: float,
                     stability: float) -> float:
    """R(t, S) = (1 + factor * t / S) ^ decay — recall probability now."""
    decay, factor = compute_decay_factor(w)
    return math.pow(1.0 + factor * elapsed_days / stability, decay)


def interval_modifier(w: tuple[float, ...], desired_retention: float) -> float:
    """I = (r^(1/decay) - 1) / factor — days multiplier from retention target."""
    decay, factor = compute_decay_factor(w)
    return (math.pow(desired_retention, 1.0 / decay) - 1.0) / factor


def init_stability(w: tuple[float, ...], grade: int) -> float:
    return max(w[grade - 1], 0.1)


def init_difficulty(w: tuple[float, ...], grade: int) -> float:
    d = w[4] - math.exp((grade - 1) * w[5]) + 1
    return round(min(max(d, 1.0), 10.0), 8)


def next_difficulty(w: tuple[float, ...], difficulty: float, grade: int) -> float:
    """D' with linear damping toward D0(Easy), mean-reverted by w7."""
    delta_d = -w[6] * (grade - 3)
    next_d = difficulty + delta_d * (10.0 - difficulty) / 9.0
    d0_easy = init_difficulty(w, RATING_EASY)
    return min(max(w[7] * d0_easy + (1.0 - w[7]) * next_d, 1.0), 10.0)


def next_recall_stability(w: tuple[float, ...], difficulty: float,
                          stability: float, retrievability: float,
                          grade: int) -> float:
    """S' after a successful recall (grades 2-4)."""
    hard_penalty = w[15] if grade == RATING_HARD else 1.0
    easy_bound = w[16] if grade == RATING_EASY else 1.0
    s = stability * (
        1.0
        + math.exp(w[8])
        * (11.0 - difficulty)
        * math.pow(stability, -w[9])
        * (math.exp((1.0 - retrievability) * w[10]) - 1.0)
        * hard_penalty
        * easy_bound
    )
    return min(max(s, S_MIN), S_MAX)


def next_forget_stability(w: tuple[float, ...], difficulty: float,
                          stability: float, retrievability: float) -> float:
    """S' after a failed recall (grade 1); capped at the current stability."""
    s = (
        w[11]
        * math.pow(difficulty, -w[12])
        * (math.pow(stability + 1.0, w[13]) - 1.0)
        * math.exp((1.0 - retrievability) * w[14])
    )
    return min(max(s, S_MIN), stability)


def next_short_term_stability(w: tuple[float, ...], stability: float,
                              grade: int) -> float:
    """S' for same-day repeats (t == 0)."""
    sinc = math.pow(stability, -w[19]) * math.exp(w[17] * (grade - 3 + w[18]))
    if grade >= RATING_HARD:
        sinc = max(sinc, 1.0)
    return min(max(stability * sinc, S_MIN), S_MAX)


def outcome_to_grade(outcome: float | None) -> int:
    """Map an Interaction outcome to an FSRS rating.

    1.0 → Good, 0.5..<1.0 → Good (passing), 0.3..0.5 → Hard, <0.3 → Again.
    None outcomes (watches, session_end) don't produce reviews.
    """
    if outcome is None:
        raise ValueError("outcome-less events are not reviews")
    if outcome >= 0.5:
        return RATING_GOOD
    if outcome >= 0.3:
        return RATING_HARD
    return RATING_AGAIN


def seed_difficulty(graph_difficulty: float | None,
                    bloom: str | None = None) -> float:
    """Concept difficulty (1..5, from the curriculum graph) → FSRS D (1..10).

    Bloom levels shift the prior: remember/understand slightly easier,
    apply/analyze/evaluate/create slightly harder. Unprofiled → D0(Good).
    """
    if graph_difficulty is None:
        d = 5.5
    else:
        d = 1.0 + 1.8 * (graph_difficulty - 1.0)  # 1..5 → 1.0..8.2
    if bloom:
        b = bloom.lower()
        if b in ("remember", "understand"):
            d -= 0.5
        elif b in ("apply", "analyze", "evaluate", "create"):
            d += 0.5
    return min(max(round(d, 2), 1.0), 10.0)


@dataclass
class FSRSCard:
    """Per-concept memory state (the DSR triple + scheduling bookkeeping)."""
    difficulty: float = 0.0          # 1..10; 0 = never reviewed
    stability: float = 0.0           # days; 0 = never reviewed
    due: str = ""                    # ISO datetime of next scheduled review
    reps: int = 0                    # number of rated reviews
    lapses: int = 0                  # failed recalls (grade 1)
    last_review: str | None = None
    state: str = "new"               # new | learning | review | relearning

    def is_reviewed(self) -> bool:
        return self.reps > 0


@dataclass
class FSRSReview:
    """What a review produced."""
    card: FSRSCard
    grade: int
    retrievability: float
    interval_days: float | None = None  # scheduled gap before this review


class FSRS:
    """FSRS-6 scheduler. Deterministic: fuzzing disabled, UTC-only."""

    version = "6.0"

    def __init__(self, w: tuple[float, ...] | None = None,
                 desired_retention: float = 0.9,
                 maximum_interval: float = 36500.0):
        if len(w or DEFAULT_W) != 21:
            raise ValueError("FSRS requires 21 weights")
        self.w = tuple(w or DEFAULT_W)
        if not 0.0 < desired_retention <= 1.0:
            raise ValueError("desired_retention must be in (0, 1]")
        self.desired_retention = desired_retention
        self.maximum_interval = maximum_interval
        self.ivl_modifier = interval_modifier(self.w, desired_retention)

    # ---- recall ----
    def retrievability(self, card: FSRSCard, now: str | None = None) -> float:
        """R now; 0.0 for never-reviewed cards."""
        if not card.is_reviewed():
            return 0.0
        t = self._elapsed_days(card.last_review, now)
        return max(0.0, forgetting_curve(self.w, t, card.stability))

    def days_until_retention(self, card: FSRSCard,
                             retention: float | None = None) -> float:
        """Days until recall probability falls to `retention` (target by default).

        Solves R(t,S) = r for t: t = ((r^(1/decay) - 1) / factor) * S.
        """
        r = retention if retention is not None else self.desired_retention
        return max(0.0, interval_modifier(self.w, r) * card.stability)

    # ---- scheduling ----
    def next_interval(self, stability: float) -> float:
        return min(max(1.0, round(stability * self.ivl_modifier)), self.maximum_interval)

    def review(self, card: FSRSCard, grade: int,
               now: str | None = None, t_days: float | None = None) -> FSRSReview:
        """Apply one rating to the card; returns the updated card + log."""
        now_iso = now or _now_utc()
        if not card.is_reviewed():
            d = init_difficulty(self.w, grade)
            s = init_stability(self.w, grade)
            state = "learning"
            r_obs = None
        else:
            t = self._elapsed_days(card.last_review, now_iso) if t_days is None else t_days
            r_obs = forgetting_curve(self.w, t, card.stability)
            d = next_difficulty(self.w, card.difficulty, grade)
            if grade == RATING_AGAIN:
                s = next_forget_stability(self.w, card.difficulty,
                                          card.stability, r_obs)
                state = "relearning"
            else:
                s = next_recall_stability(self.w, card.difficulty,
                                          card.stability, r_obs, grade)
                state = "review"
        card.difficulty = round(min(max(d, 1.0), 10.0), 8)
        card.stability = round(min(max(s, S_MIN), S_MAX), 8)
        card.reps += 1
        card.lapses += 1 if grade == RATING_AGAIN else 0
        card.last_review = now_iso
        card.due = (_dt(now_iso) + timedelta(days=self.next_interval(card.stability))).isoformat()
        card.state = state
        return FSRSReview(card=card, grade=grade,
                          retrievability=r_obs if r_obs is not None else 1.0,
                          interval_days=self.next_interval(card.stability))

    def replay_history(self, grades: list[tuple[float, int]],
                       now: str | None = None) -> FSRSCard:
        """Replay (days_ago, grade) events through a fresh card."""
        card = FSRSCard()
        if not grades:
            return card
        now_iso = now or _now_utc()
        base = _dt(now_iso)
        for days_ago, grade in sorted(grades, key=lambda x: x[0]):
            t = (base - timedelta(days=days_ago))
            self.review(card, grade, now=t.isoformat())
        return card

    def _elapsed_days(self, last_review: str | None, now: str | None) -> float:
        return max(0.0, (_ts(now or _now_utc()) - _ts(last_review or now or _now_utc())) / 86400.0)

    def params(self) -> dict:
        return {
            "version": self.version,
            "w": list(self.w),
            "desired_retention": self.desired_retention,
            "maximum_interval": self.maximum_interval,
            "interval_modifier": self.ivl_modifier,
        }


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
