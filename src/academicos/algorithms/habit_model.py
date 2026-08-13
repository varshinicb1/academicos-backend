"""Algorithm A5 — "habit_formation": streaks, consistency, formation phase.

Grounded in Lally et al. (2010, *How Habits Are Formed: Modelling Habit
Formation in the Real World*, Eur. J. Soc. Psychol. 40(6):998-1009): the
median time to reach an automaticity asymptote was 66 days (range 18-254),
gains fast early, slow late. A learner with an uninterrupted daily run into
~two months is treated as "automated"; the first ~3 weeks is the bootstrap
"forming" phase where outcomes are still deliberative.

Wood & Neal (2007, *A new look at habits*, Curr. Dir. Psychol. Sci. 16(4):
198-202) supply the relapse caveat: daily repetition in a stable context
builds automaticity, but a slip — days without the cue — decays it. The
relapse heuristic below treats a week with low calendar-day consistency as
the alarm: `risk` grows first with idle days and then with a weak 7-day
consistency.

Deterministic, no external calls, all time injected via `now_iso` (defaults
to UTC now only as a fallback) — same contract as A13 (`version`, `_params`,
dataclass result with `params` round-trip).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .learner_model import Interaction

_FORMING_MAX_DAYS = 21
_RELAPSE_CONSISTENCY = 0.4
_SHORT_WINDOW = 7  # consistency_7 window in calendar days


@dataclass
class HabitParams:
    """Calibratable constants for the habit model (Lally et al. 2010)."""
    formation_days: int = 66          # median days to the automaticity asymptote
    streak_window_days: int = 1       # 1 = strictly-consecutive run; >1 bridges a skipped day
    consistency_window: int = 30      # calendar-day window behind consistency_30
    breakday_limit: int = 3           # idle days that max out relapse risk
    streak_grace_days: int = 1        # idle days after last study before the current streak lapses


@dataclass
class HabitResult:
    learner_id: str
    study_days: list[str]             # ascending ISO dates, one per studied day
    current_streak: int               # consecutive study days ending at now (0 if lapsed)
    longest_streak: int
    last_day: str | None              # ISO date of the most recent study day
    consistency_7: float              # fraction of last 7 calendar days with study
    consistency_30: float             # fraction of last consistency_window days studied
    phase: str                        # forming | consolidating | automated | relapse
    risk: float                       # 0..1 high-risk-of-relapse heuristic
    params: dict = field(default_factory=dict)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_date(iso: str) -> date:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


class HabitModel:
    """Per-learner habit signals from an interaction stream (A5, deterministic v1)."""

    version = "1.0"

    def __init__(self, params: HabitParams | None = None):
        self.p = params or HabitParams()

    def estimate(self, interactions: list[Interaction], now_iso: str | None = None,
                 learner_id: str = "habit") -> HabitResult:
        today = _utc_date(now_iso or _now_utc())
        days = sorted({_utc_date(i.ts) for i in interactions
                       if _utc_date(i.ts) <= today})
        total_days = len(days)

        last_day = days[-1] if days else None
        idle_days = (today - last_day).days if last_day else None

        runs = []
        for day in days:
            if runs and (day - runs[-1][-1]).days <= self.p.streak_window_days:
                runs[-1].append(day)
            else:
                runs.append([day])
        longest_streak = max((len(r) for r in runs), default=0)

        alive = (last_day is not None and idle_days is not None
                 and idle_days <= self.p.streak_grace_days)
        current_streak = len(runs[-1]) if alive else 0

        count_7 = sum(1 for d in days if (today - d).days < _SHORT_WINDOW)
        count_30 = sum(1 for d in days if (today - d).days < self.p.consistency_window)
        consistency_7 = round(count_7 / _SHORT_WINDOW, 4)
        consistency_30 = round(count_30 / self.p.consistency_window, 4)

        if total_days and consistency_7 < _RELAPSE_CONSISTENCY:
            phase = "relapse"
        elif total_days <= _FORMING_MAX_DAYS:
            phase = "forming"
        elif total_days <= self.p.formation_days:
            phase = "consolidating"
        else:
            phase = "automated"

        idle_risk = (min(idle_days / self.p.breakday_limit, 1.0)
                     if self.p.breakday_limit > 0 and idle_days is not None else 0.0)
        risk = round(max(1.0 - consistency_7, idle_risk), 4)

        return HabitResult(
            learner_id=learner_id,
            study_days=[d.isoformat() for d in days],
            current_streak=current_streak,
            longest_streak=longest_streak,
            last_day=last_day.isoformat() if last_day else None,
            consistency_7=consistency_7,
            consistency_30=consistency_30,
            phase=phase,
            risk=risk,
            params=self._params(),
        )

    def _params(self) -> dict:
        return {
            "formation_days": self.p.formation_days,
            "streak_window_days": self.p.streak_window_days,
            "consistency_window": self.p.consistency_window,
            "breakday_limit": self.p.breakday_limit,
            "streak_grace_days": self.p.streak_grace_days,
        }