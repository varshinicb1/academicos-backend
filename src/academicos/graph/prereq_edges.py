"""Prerequisite edge enrichment: materialize voted prerequisite pairs as
typed PREREQUISITE_OF edges with estimator-derived properties.

The voting estimator (CurriculumEstimator.vote_prerequisites) produces a
bare {later: [(prereq, score, va)]} map; this module turns that into first-
class graph edges so retrieval and planning can traverse, threshold and
explain prerequisites like any other relationship.

Edge properties (stored in Edge.attributes):
  score                  normalized winning-margin vote (same as estimator
                         output, in [theta, 1])
  confidence             participation-weighted margin in [0, 1]; combines
                         how much of the possible criterion weight actually
                         voted (va+vb / active weight) with how decisive the
                         margin was (0.5 + 0.5*|s|)
  criteria               {criterion: +1/-1} breakdown of the winning side
  n_criteria             number of criteria that voted
  va, vb                 weighted vote mass for and against the direction
  transitive             True for edges produced by transitive closure
                         (no direct pair evidence)
  difficulty_jump        difficulty(later) - difficulty(prereq), positive
                         when the prereq is genuinely easier
  recommended_spacing_days  review-spacing heuristic: base 3 days scaled by
                         confidence and the difficulty gap (see docs)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..models.common import Confidence, ExtractionMethod, Provenance, SourceSpan
from ..models.enums import EdgeType
from ..models.graph import Edge
from .curriculum import CRITERION_WEIGHTS, VOTE_THETA
from .schema import edge_id, new_edge
from .store import GraphStore

log = logging.getLogger(__name__)

# Review-spacing heuristic: a fresh mastered concept needs ~3 days before it
# is worth revisiting (forgetting-curve base), stretched by uncertainty
# (low confidence) and by how far the learner has to climb (difficulty gap).
SPACING_BASE_DAYS = 3.0


def confidence_from_votes(va: float, vb: float, active_weight: float) -> float:
    """Participation-weighted margin in [0, 1].

    conf = participation * (0.5 + 0.5 * margin), where
      participation = (va + vb) / active_weight  (fraction of the criterion
                     weight that actually voted for this pair)
      margin        = |va - vb| / (va + vb)      (how one-sided the vote was)

    A pair decided by many criteria at a wide margin is confident; a pair
    decided by one criterion at a narrow margin is not.
    """
    if active_weight <= 0 or (va + vb) <= 0:
        return 0.0
    participation = min(1.0, (va + vb) / active_weight)
    margin = abs(va - vb) / (va + vb)
    return round(participation * (0.5 + 0.5 * margin), 4)


def recommended_spacing_days(confidence: float, difficulty_jump: float) -> int:
    """Days until the later concept should be reviewed after mastering the
    prerequisite. Base interval stretched by the climb:"""
    days = SPACING_BASE_DAYS * (1.0 + (1.0 - confidence)) * (1.0 + max(0, difficulty_jump))
    return max(1, round(days))


def build_prerequisite_edges(
    prereqs: dict[str, list[tuple[str, float, int]]],
    label_to_id: dict[str, str],
    pair_detail: Optional[list[dict[str, Any]]] = None,
    difficulty: Optional[dict[str, int]] = None,
    active_criteria: Optional[tuple[str, ...]] = None,
    weights: Optional[dict[str, int]] = None,
) -> list[Edge]:
    """Convert voted prereq pairs into PREREQUISITE_OF Edge objects.

    prereqs:     {later_label: [(prereq_label, score, va), ...]} exactly as
                 produced by vote_prerequisites / transitive_prereqs.
    label_to_id: label -> concept node id (the id written into the store).
    pair_detail: optional list of dicts with keys prereq, later, score, va,
                 vb, votes ({criterion: +1/-1}) — emitted by
                 vote_prerequisites(..., pair_detail=...) for accepted pairs.
    difficulty:  optional {label: 1..5}; enables difficulty_jump.
    active_criteria / weights: match the estimator run so confidence can be
                 normalized by the weight that could have voted.
    """
    w = weights or CRITERION_WEIGHTS
    active = tuple(active_criteria) if active_criteria else tuple(w.keys())
    active_weight = float(sum(w.get(c, 1) for c in active))
    detail: dict[tuple[str, str], dict[str, Any]] = {}
    for d in pair_detail or []:
        detail[(d["prereq"], d["later"])] = d

    edges: list[Edge] = []
    for later, pairs in prereqs.items():
        if later not in label_to_id:
            continue
        for prereq, score, va in pairs:
            if prereq not in label_to_id:
                continue
            src, tgt = label_to_id[prereq], label_to_id[later]
            d = detail.get((prereq, later))
            if d is not None:
                votes = d.get("votes") or {}
                vb = d.get("vb", 0.0)
                attrs: dict[str, Any] = {
                    "score": round(float(score), 4),
                    "confidence": confidence_from_votes(float(va), float(vb), active_weight),
                    "criteria": votes,
                    "n_criteria": len(votes),
                    "va": round(float(va), 2),
                    "vb": round(float(vb), 2),
                }
            else:
                attrs = {
                    "score": round(float(score), 4),
                    "confidence": min(1.0, abs(float(score))),
                    "criteria": {},
                    "n_criteria": 0,
                    "transitive": True,
                }
            if difficulty and prereq in difficulty and later in difficulty:
                attrs["difficulty_jump"] = difficulty[later] - difficulty[prereq]
            attrs["recommended_spacing_days"] = recommended_spacing_days(
                attrs["confidence"], attrs.get("difficulty_jump", 0))
            edges.append(new_edge(
                EdgeType.PREREQUISITE_OF, src, tgt, weight=attrs["score"],
                attributes=attrs,
                provenance=_voting_provenance(score, attrs["confidence"]),
            ))
    return edges


def write_prerequisite_edges(store: GraphStore, edges: list[Edge]) -> int:
    """Upsert prerequisite edges; returns the number written."""
    for e in edges:
        store.upsert_edge(e)
    return len(edges)


def _voting_provenance(score: float, confidence: float) -> Provenance:
    return Provenance(
        source=SourceSpan(document_id="curriculum:voting", page=1),
        method=ExtractionMethod.RULE_BASED,
        confidence=Confidence(
            method=ExtractionMethod.RULE_BASED,
            score=min(1.0, max(0.0, confidence)),
            rationale=f"multi-criteria voting, score={score:.3f}",
        ),
    )
