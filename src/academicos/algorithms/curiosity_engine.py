"""Algorithm 7 — Curiosity Engine ("what next to pique?").

Suggests the next concept to study by ranking a candidate list against four
evidence-grounded drives:

  * novelty              — Berlyne (1960): collative variables (novelty,
                           complexity, surprise) drive epistemic curiosity;
                           brand-new concepts get the strongest pull.
  * desired_difficulty   — Bjork (1994): *desirable difficulties* predict that
                           material is best learned when it is hard enough to
                           require effort but not so hard it overwhelms. A
                           difficulty band with a Gaussian peak models that
                           "just-past-easy" sweet spot (mu 0.65, sd 0.15).
  * interleaving         — interleaving adjacent-study topics (Bjork, 1994;
                           Rohrer et al.) strengthens discriminative learning;
                           a cross-domain candidate earns a bonus.
  * interest momentum    — recent success on a concept fuels continued pull on
                           it; untouched concepts sit at a neutral 0.5 pull.

The composite is a validated, weighted sum of the four cues (see
docs/research/curiosity.md). OE most weight falls on novelty + desired
difficulty, the two strongest findings in the literature; spacing is handled
operationally by steering *away* from already-mastered concepts and applying a
"seen" penalty so a never-quite-nailed concept is not over-recommended.

Pure, deterministic, no APIs — same contract as the other algorithms in this
library (version, research-shaped defaults, `_params`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .learner_model import ConceptState, LearnerModel
from .mastery import KnowledgeMastery

_KM = KnowledgeMastery()

_NEUTRAL_MOMENTUM = 0.5


@dataclass
class CuriosityParams:
    """Calibratable constants for the curiosity suggestion engine."""
    model: str = "berlyne-bjork"
    w_novelty: float = 0.35            # Berlyne collative novelty pull
    w_desired_difficulty: float = 0.35  # Bjork desirable-difficulty cue
    w_interleaving: float = 0.15       # cross-domain interleave bonus
    w_interest_momentum: float = 0.15  # recent-success inertia
    novelty_bonus: float = 0.2         # extra lift for never-seen concepts
    desired_difficulty_mu: float = 0.65     # Gaussian peak of the sweet spot
    desired_difficulty_sigma: float = 0.15  # width of the desired band
    desired_difficulty_band: list = field(
        default_factory=lambda: [0.5, 0.85])   # hard window; outside = 0
    penalty_seen: float = 0.2                 # damp already-worked concepts
    momentum_window: int = 5                 # recent answers for momentum
    mastery_threshold: float = 0.8           # mastery at/above -> skipped


@dataclass
class CandidateScore:
    """One ranked suggestion with its decoded cue breakdown."""
    concept_id: str
    score: float            # final curiosity, 0..1 (this is the rank key)
    overall: float          # === score; explicit reading of "overall score"
    novelty: float          # 1.0 if never seen, else 0.0
    interleave: bool        # cross-domain with respect to the source concept
    momentum: float         # recent-success incentive (0..1, 0.5 neutral)
    reasons: str            # human-readable composition detail


@dataclass
class CuriosityResult:
    source_concept: str | None
    ranked: list[CandidateScore] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    version: str = "1.0"


def _desired(difficulty: float, params: CuriosityParams) -> float:
    """Desirable-difficulty cue: Gaussian peak inside the difficulty band."""
    lo, hi = params.desired_difficulty_band
    if difficulty < lo or difficulty > hi:
        return 0.0
    z = (difficulty - params.desired_difficulty_mu) / params.desired_difficulty_sigma
    return math.exp(-0.5 * z * z)


def _domain_of(item: object) -> str | None:
    """Domain of an id (or concept-dict) — allows override with a `domain` key."""
    if isinstance(item, dict):
        d = item.get("domain")
        if d is not None and str(d).strip():
            return str(d).strip()
        cid = item.get("concept_id")
    else:
        cid = item
    s = str(cid or "")
    for sep in (":", "."):
        if sep in s:
            return s.split(sep, 1)[0]
    return s or None


def _recent_momentum(state: ConceptState | None, window: int) -> float:
    """Recent success = correct/total over the last `window` answer events."""
    if state is None or state.attempts <= 0:
        return _NEUTRAL_MOMENTUM
    answers = [i for i in state.history if i.kind == "answer" and i.outcome is not None]
    recent = answers[-window:] if window > 0 else answers
    if not recent:
        return _NEUTRAL_MOMENTUM
    return sum(1 for i in recent if i.outcome >= 0.5) / len(recent)


class CuriosityEngine:
    """Rank future-concept candidates by four Berlyne/Bjork curiosity cues.

    v1 (deterministic):

        curiosity = w_novelty * novelty
                  + w_desired_difficulty * desired(difficulty)
                  + w_interleaving * interleave_flag
                  + w_interest_momentum * momentum
                  + novelty_bonus (if novel)
                  - penalty_seen (if already seen)

        Novelty = 1 if the concept is absent from the LearnerModel, else 0.
        Desired uses a Gaussian peaked at desired_difficulty_mu with the band
        acting as a hard gate. Already-mastered concepts are skipped entirely
        (spacing). Ranks are highest-first; ties break alphabetically on id.
    """
    version = "1.0"

    def __init__(self, params: CuriosityParams | None = None):
        self.p = params or CuriosityParams()

    def rank(self, model: LearnerModel, candidates: list[dict],
             current_concept: str | None = None,
             now: str | None = None) -> CuriosityResult:
        current_domain = self._current_domain(candidates, current_concept)
        scored: list[CandidateScore] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cid = cand.get("concept_id")
            if cid is None or str(cid) == "":
                continue
            cid = str(cid)
            difficulty = cand.get("difficulty")
            difficulty = max(0.0, min(1.0, float(difficulty if difficulty is not None else 0.5)))

            state = model.concepts.get(cid)
            novel = state is None
            # Mastery is computed at use, not read off ``state.mastery``: that
            # cache is never populated on real learners (see learner_model.py's
            # docstring on the record() path), so gating on it never skipped a
            # mastered concept.
            if state is not None and _KM.score(model, cid, now=now).mastery >= self.p.mastery_threshold:
                continue  # already mastered -> do not re-suggest

            n_term = self.p.w_novelty * (1.0 if novel else 0.0)
            d_term = self.p.w_desired_difficulty * _desired(difficulty, self.p)

            cand_domain = _domain_of(cand)
            interleaved = (current_domain is not None and cand_domain is not None
                           and cand_domain != current_domain)
            i_term = self.p.w_interleaving * (1.0 if interleaved else 0.0)

            momentum = _recent_momentum(state, self.p.momentum_window)
            m_term = self.p.w_interest_momentum * momentum

            bonus = self.p.novelty_bonus if novel else 0.0
            penalty = 0.0 if novel else self.p.penalty_seen

            raw = n_term + d_term + i_term + m_term + bonus - penalty
            score = round(max(0.0, min(1.0, raw)), 4)

            scored.append(CandidateScore(
                concept_id=cid,
                score=score,
                overall=score,
                novelty=1.0 if novel else 0.0,
                interleave=interleaved,
                momentum=round(momentum, 4),
                reasons=_desired_reason(cid, difficulty, n_term, d_term,
                                        i_term, m_term, bonus, penalty, raw),
            ))
        scored.sort(key=lambda c: (-c.score, c.concept_id))
        return CuriosityResult(
            source_concept=current_concept,
            ranked=scored,
            params=self._params(),
            version=self.version,
        )

    def _current_domain(self, candidates: list[dict], source: str | None) -> str | None:
        if not source:
            return None
        hit = next((c for c in candidates
                    if isinstance(c, dict) and str(c.get("concept_id", "")) == str(source)), None)
        return _domain_of(hit) if hit is not None else _domain_of(source)

    def _params(self) -> dict:
        return {
            "model": self.p.model,
            "w_novelty": self.p.w_novelty,
            "w_desired_difficulty": self.p.w_desired_difficulty,
            "w_interleaving": self.p.w_interleaving,
            "w_interest_momentum": self.p.w_interest_momentum,
            "novelty_bonus": self.p.novelty_bonus,
            "desired_difficulty_mu": self.p.desired_difficulty_mu,
            "desired_difficulty_sigma": self.p.desired_difficulty_sigma,
            "desired_difficulty_band": list(self.p.desired_difficulty_band),
            "penalty_seen": self.p.penalty_seen,
            "momentum_window": self.p.momentum_window,
            "mastery_threshold": self.p.mastery_threshold,
        }


def _desired_reason(cid: str, difficulty: float, n: float, d: float,
                    i: float, m: float, bonus: float, penalty: float, raw: float) -> str:
    return (
        f"{cid} d={difficulty:.2f}: novelty={n:.4f} desired={d:.4f} "
        f"interleave={i:.4f} momentum={m:.4f} bonus={bonus:.4f} "
        f"penalty={penalty:.4f} raw={raw:.4f}"
    )