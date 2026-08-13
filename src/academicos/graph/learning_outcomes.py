"""Learning-outcome ontology layer: canonical LO statements derived from
concept profiles, aligned to concepts via MAPS_TO edges, with prerequisite
relations inherited from the concept graph.

Why derived, not extracted: the NCERT corpus in production (curiosity /
ganita_prakash / poorvi) contains no structured "Learning Outcomes" sections
(the NCF boxes live in syllabus documents, which are not in the corpus).
This layer is the deterministic scaffold — every profiled concept with a
Bloom level yields one canonical LO statement via per-level templates — so
that LO-centric planning (lo_prereq_plan) works today. When CBSE/NCF syllabus
documents are ingested (Phase 2), verbatim LO extraction replaces the
template derivation for those documents and aligns via the same MAPS_TO
machinery; template LOs remain for concepts without a syllabus mention.

LO statement template: <verb phrase> <concept>. Verb phrases are the
Anderson & Krathwohl signatures already used by classify_bloom, so an
"apply" concept becomes "apply fractions to solve problems" — the standard
NCF-style competency phrasing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..models.common import Confidence, ExtractionMethod, Provenance, SourceSpan
from ..models.enums import EdgeType, NodeType
from ..models.graph import Edge, Node
from .schema import edge_id, new_edge, new_node, node_id
from .store import GraphStore

log = logging.getLogger(__name__)

# Bloom level -> LO verb phrase (first-person, competency style).
LO_VERB_PHRASE: dict[str, str] = {
    "remember": "recall and describe",
    "understand": "explain and compare",
    "apply": "apply ... to solve problems",
    "analyze": "analyze problems and situations involving",
    "evaluate": "evaluate and justify decisions involving",
    "create": "design and create solutions involving",
}

# Phrase templates: {placeholder} = concept label. Each level keeps a
# concrete, assessable verb (the paper's taxonomy verbs).
LO_TEMPLATES: dict[str, str] = {
    "remember": "recall and describe the concept of {c}",
    "understand": "explain and compare the concept of {c}",
    "apply": "apply the concept of {c} to solve problems",
    "analyze": "analyze problems and situations involving {c}",
    "evaluate": "evaluate and justify decisions involving {c}",
    "create": "design and create solutions involving {c}",
}

_LO_CACHE: dict[str, str] = {}


def lo_statement(concept_label: str, bloom: str) -> str:
    """Canonical LO statement for a concept at a Bloom level."""
    return LO_TEMPLATES.get(bloom, LO_TEMPLATES["understand"]).format(c=concept_label)


def lo_node_id(concept_label: str, bloom: str) -> str:
    return node_id(NodeType.LEARNING_OUTCOME, f"{bloom}:{concept_label}")


def lo_node(concept_label: str, bloom: str, difficulty: Optional[int] = None,
            learning_hours: Optional[float] = None,
            provenance: Optional[Provenance] = None) -> Node:
    """Canonical LO node: statement is the label; concept/bloom in attrs."""
    attrs: dict[str, Any] = {
        "concept": concept_label,
        "bloom": bloom,
        "source": "template",
    }
    if difficulty is not None:
        attrs["difficulty"] = difficulty
    if learning_hours is not None:
        attrs["learning_hours"] = learning_hours
    return new_node(
        NodeType.LEARNING_OUTCOME, lo_statement(concept_label, bloom),
        id_=lo_node_id(concept_label, bloom), attributes=attrs,
        provenance=provenance)


def _rule_provenance(rationale: str) -> Provenance:
    return Provenance(
        source=SourceSpan(document_id="curriculum:lo", page=1),
        method=ExtractionMethod.RULE_BASED,
        confidence=Confidence(method=ExtractionMethod.RULE_BASED, score=1.0,
                              rationale=rationale),
    )


def build_lo_layer(store: GraphStore,
                   profiles: dict[str, dict[str, Any]],
                   prereqs: Optional[dict[str, list[tuple[str, float, int]]]] = None,
                   concept_ids: Optional[dict[str, str]] = None,
                   ) -> tuple[int, int, int]:
    """Create LO nodes + MAPS_TO concept->LO edges + inherited PREREQUISITE_OF
    LO->LO edges.

    profiles:  {concept_label: {bloom, difficulty?, learning_hours?}} —
               the node attributes the rule profiler writes per concept.
    prereqs:   {later_label: [(prereq_label, score, n)]} — the estimator's
               voted prereq pairs (post transitive closure). LO-level
               prerequisites inherit from concept-level pairs, carrying the
               same score (confidence is dropped: the derived edge is a
               projection, not a vote).
    concept_ids: {label: stored node id} — the ids actually present in the
               store (the graph builder case-folds ids, so recomputing
               cbse:concept:{label} can miss); falls back to computing.

    Returns (lo_nodes, maps_to_edges, prereq_edges).
    """
    concept_ids = concept_ids or {}
    concept_ids = dict(concept_ids)
    lo_ids: dict[str, str] = {}
    for label, p in profiles.items():
        bloom = p.get("bloom") or "understand"
        if bloom not in LO_TEMPLATES:
            bloom = "understand"
        concept_ids.setdefault(label, node_id(NodeType.CONCEPT, label))
        lo_ids[label] = lo_node_id(label, bloom)

    n_nodes = 0
    for label, p in profiles.items():
        prov = _rule_provenance(f"LO derived from concept profile ({label})")
        store.upsert_node(lo_node(
            label, p.get("bloom") or "understand",
            difficulty=p.get("difficulty"), learning_hours=p.get("learning_hours"),
            provenance=prov))
        n_nodes += 1

    n_maps = 0
    for label in profiles:
        store.upsert_edge(new_edge(
            EdgeType.MAPS_TO, concept_ids[label], lo_ids[label],
            weight=1.0, provenance=_rule_provenance("concept-LO alignment")))
        n_maps += 1

    n_prereq = 0
    for later, pairs in (prereqs or {}).items():
        if later not in lo_ids:
            continue
        for prereq, score, _ in pairs:
            if prereq not in lo_ids:
                continue
            store.upsert_edge(new_edge(
                EdgeType.PREREQUISITE_OF, lo_ids[prereq], lo_ids[later],
                weight=min(1.0, float(score)),
                attributes={"derived": True, "score": round(float(score), 4)},
                provenance=_rule_provenance("inherited from concept prereq")))
            n_prereq += 1
    return n_nodes, n_maps, n_prereq
