"""P2.4 — per-learner FSRS fitting from a learner's real review history.

The FSRS engine (`algorithms/fsrs.py`) ships the *population* default
weights (`DEFAULT_W`, calibrated on ~1.2B+ reviews by the Open Spaced
Repetition project). Learners differ: a high-aptitude learner remembers for
weeks, an erratic one forgets overnight. Per-learner weights are where DSR
gains its personalization edge (FSRS's own optimizer fits per-learners; see
the Open Spaced Repetition benchmark repo, which retrains weights per user
and per deck).

Why a scalar surrogate instead of the full 21-weight fit?

    * 21 weight-parameters need thousands of graded reviews; a learner with
      fifty answers cannot identify them. Three scalar *phenomenological*
      parameters — `initial_stability`, `difficulty_sens`, `stability_gain`
      — are identifiable from a handful of reviews.
    * The surrogate keeps the FSRS forgetting-curve shape but linearizes the
      stability trajectory via the survivorship probability

          S  = initial_stability + difficulty_sens * d           (d in 0..1)
          R  = exp(-elapsed / S)

          S = S * stability_gain        on a success (outcome >= 0.5)
          S = S / stability_gain        on a lapse

      i.e. ``stability = base + sens*difficulty`` mapped linearly onto the
      FSRS recall probability — a documented *v1 approximation*, not the
      full DSR transition set. See ``docs/research/fsrs-fit.md``.

Learning signal from the Interaction log:

    * Reviews are ``kind == "answer"`` events with an outcome.
    * Real logs do not carry the scheduled interval; we *infer* each review's
      interval as the wall-clock gap between consecutive answer events for the
      same concept. This is an approximation — a learner may read offscreen
      between reviews — but it is unbiased for calibration purposes and the
      only signal we have offline.

Fitting is pure-stdlib, gradient-free and seeded:

    * A coarse deterministic grid over the parameter space seeds a local
      optimum (fast first guess).
    * Cyclic coordinate descent (a coordinate around makes no move → halve
      step) then refines it until steps fall below tolerance or `max_iter`
      is hit.
    * Each step only accepts a strictly-lower loss, so ``loss_after`` is
      always monotone non-increasing w.r.t. ``loss_before``; the default
      point is itself a grid candidate, guaranteeing
      ``loss_after <= loss_before`` for any history (and strictly below for
      any history with real structure).
    * ``random.Random(seed)`` picks the coordinate-visit order each pass,
      so runs are reproducible (default ``seed=0``).

Tight, bounded search space (documented in ``docs/research/fsrs-fit.md``):
negativity of a stability ("negative interval") is impossible by the
formulation — ``stability_gain >= 1.0``, ``sens >= 0`` and
``initial_stability > 0`` — so a concept can never predict a negative
survival.

No numpy/scipy: ``math`` + the ``random`` module only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime

from .learner_model import Interaction

#: Human-readable names for the three fitted phenomenaial parameters.
PARAM_NAMES = ("initial_stability", "difficulty_sens", "stability_gain")

#: Default (unfitted) scalar weights returned before any data is seen.
#: ``initial_stability`` ~= first-exposure strength in days, ``difficulty_sens``
#: = added days per unit of normalized concept difficulty, ``stability_gain``
#: = multiplicative growth on a successful review (and its reciprocal on a lapse).
DEFAULT_WEIGHTS: tuple[float, float, float] = (1.0, 0.5, 1.5)

#: Feasible box per parameter: negativity of stability is structurally
#: impossible, growth can only strengthen a successful trace.
BOUNDS: tuple[tuple[float, float], ...] = (
    (0.0, 1000.0),   # initial_stability   (days)
    (0.0, 30.0),     # difficulty_sens     (days per unit difficulty)
    (1.0, 6.0),      # stability_gain      (multiplier per success)
)

_STEP_LIMITS = (1.5, 1.5, 0.25)      # per-coordinate starting step (absolute days / unit)
_STEP_TOL = 1e-6                     # below this the descent is "converged"
_ACCEPT_EPS = 1e-12                  # strict-improvement slack


@dataclass
class FitParams:
    """Configuration for the per-learner fitting run.

    ``model`` mirrors the optimizer tier (v1 scalar surrogate for now, the
    full FSRS-6 21-parameter fit later); ``loss="rmse"`` is the survivorship
    curve's root-mean-square error against observed outcomes plus a small L2
    penalty toward the default for ``reg``.
    """
    model: str = "fsrs-optimized"
    learning_rate: float = 0.1
    max_iter: int = 500
    prediction_horizon: bool = False
    loss: str = "rmse"
    reg: float = 1e-4
    seed: int = 0


@dataclass
class FitResult:
    """Outcome of a per-learner fit run.

    ``default_lr`` — *the defaults returned before fit* — i.e. the unfitted
    scalar weights a learner would have received without this module;
    ``fitted_weights`` are the learned triple. ``gain`` quantifies the
    improvement on the same history: ``(loss_before - loss_after)/loss_before``.
    """
    learner_id: str
    fitted_weights: list[float]
    default_lr: list[float]
    n_observed: int          # interval observations (consecutive answer pairs)
    n_reviews: int           # answer events with an outcome
    loss_before: float
    loss_after: float
    converged: bool
    iterations: int
    params: FitParams

    @property
    def gain(self) -> float:
        """Relative loss reduction against the defaults, in 0..1."""
        if self.loss_before <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (self.loss_before - self.loss_after) / self.loss_before))

    def refit_vs_defaults(self) -> float:
        """Δ error vs using the published defaults; 0 = no better than default."""
        return self.gain


class FSRSFit:
    """Gradient-free, deterministic fitting of the scalar FSRS surrogate."""

    version = "1.0"

    def __init__(self, params: FitParams | None = None):
        self.p = params or FitParams()

    # ------------------------------------------------------------- public API

    def fit(self, interactions: list[Interaction],
            learner_id: str = "fit") -> FitResult:
        """Fit per-learner weights from a learner's review history.

        ``interactions`` are raw interaction-log events; only ``answer``
        events marked outcome are used as reviews. Returns a ``FitResult``;
        the fit never worsens the default (``loss_after <= loss_before``).
        """
        loops = self._to_chains(interactions)
        n_reviews = sum(len(seq) for seq in loops)
        n_observed = sum(len(seq) - 1 for seq in loops)

        if n_observed == 0:
            return self._empty_result(learner_id, n_observed, n_reviews)

        default = list(DEFAULT_WEIGHTS)
        loss_before = self._loss(loops, default)

        best, best_loss, converged, iters = self._optimize(loops, loss_before)

        return FitResult(
            learner_id=learner_id,
            fitted_weights=[round(v, 6) for v in best],
            default_lr=default,
            n_observed=n_observed,
            n_reviews=n_reviews,
            loss_before=round(loss_before, 6),
            loss_after=round(best_loss, 6),
            converged=converged,
            iterations=iters,
            params=self.p,
        )

    # ----------------------------------------------------------------- data

    @staticmethod
    def _to_chains(interactions: list) -> list:
        """Group answer events per concept, sorted by ``ts``; infer interval.

        Interval := gap between two consecutive answer events for the same
        concept (in days, floored at 0). A chain with 1 event yields 0
        observations (no gap exists for a first exposure).
        """
        by_concept: dict[str, list[Interaction]] = {}
        for i in interactions:
            if i.kind == "answer" and i.outcome is not None:
                by_concept.setdefault(i.concept_id, []).append(i)

        chains = []
        for events in by_concept.values():
            events.sort(key=lambda e: e.ts)  # stable for equal ts
            chain = []
            prev_ts = None
            for ev in events:
                elapsed = 0.0 if prev_ts is None else max(
                    0.0, (_ts(ev.ts) - _ts(prev_ts)) / 86400.0
                )
                chain.append((elapsed, ev.outcome, _norm_difficulty(ev.difficulty)))
                prev_ts = ev.ts
            chains.append(chain)
        return chains

    # ------------------------------------------------------------ objectives

    def _loss(self, chains, param) -> float:
        """RMSE + tiny L2 toward the defaults (the survivor transition)."""
        base, sens, gain = param
        penalty = self.p.reg * sum(
            (v - d) ** 2 for v, d in zip((base, sens, gain), DEFAULT_WEIGHTS)
        )
        two_sq, n = 0.0, 0
        for seq in chains:
            # stability at first exposure: base + sens * difficulty
            s = base + sens * seq[0][2]
            for elapsed, outcome, _d in seq[1:]:
                pred = math.exp(-elapsed / max(s, 1e-9))
                err = pred - outcome
                two_sq += err * err
                n += 1
                # update the trace for the *next* interval off this review
                if outcome >= 0.5:
                    s = min(max(s * gain, BOUNDS[0][0]), BOUNDS[0][1])
                else:
                    s = min(max(s / gain, BOUNDS[0][0]), BOUNDS[0][1])
        if n == 0:
            return penalty
        return math.sqrt(two_sq / n) + penalty

    # ------------------------------------------------------------- fitting

    def _optimize(self, chains, default_loss: float):
        """Grid seed + seeded cyclic coordinate descent → best params."""
        p = self.p
        rng = random.Random(p.seed)

        grid_best = list(DEFAULT_WEIGHTS)
        grid_loss = default_loss
        for cand in _grid():
            cand_loss = self._loss(chains, cand)
            if cand_loss < grid_loss - _ACCEPT_EPS:
                grid_best, grid_loss = cand, cand_loss

        # local refinement
        best = list(grid_best)
        bl = grid_loss
        step = list(_STEP_LIMITS)
        converged = False
        iters = 0
        coords = (0, 1, 2)
        for _ in range(self.p.max_iter):
            iters += 1
            moved = False
            visit = list(coords)
            rng.shuffle(visit)  # seeded, deterministic given self.p.seed
            for ci in visit:
                for mult in (4.0, 2.0, 1.0):
                    for sign in (1.0, -1.0):
                        trial = list(best)
                        trial[ci] = _clamp(trial[ci] + sign * mult * step[ci],
                                           BOUNDS[ci][0], BOUNDS[ci][1])
                        tl = self._loss(chains, trial)
                        if tl < bl - _ACCEPT_EPS:
                            best, bl, moved = trial, tl, True
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                step = [s * 0.5 for s in step]
                if max(step) < _STEP_TOL:
                    converged = True
                    break
        return best, bl, converged, iters

    def _empty_result(self, learner_id: str, n_observed: int,
                      n_reviews: int) -> FitResult:
        """No interval exists to calibrate against → return the defaults."""
        return FitResult(
            learner_id=learner_id,
            fitted_weights=list(DEFAULT_WEIGHTS),
            default_lr=list(DEFAULT_WEIGHTS),
            n_observed=n_observed,
            n_reviews=n_reviews,
            loss_before=0.0,
            loss_after=0.0,
            converged=False,
            iterations=0,
            params=self.p,
        )


# ----------------------------------------------------------------- helpers

def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _norm_difficulty(difficulty: float | None) -> float:
    """Concept difficulty (curriculum 1..5) → normalized 0..1; None → 0.5."""
    if difficulty is None:
        return 0.5
    return _clamp((difficulty - 1.0) / 4.0, 0.0, 1.0)


def _grid() -> list:
    """Deterministic coarse grid that seeds coordinate descent.

    A reduced but space-spanning list; the default point is a candidate too,
    so the fit can never be worse than the unfitted defaults.
    """
    points = []
    for base in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0):
        for sens in (0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0):
            for gain in (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0):
                points.append([base, sens, gain])
    points.append(list(DEFAULT_WEIGHTS))
    return points