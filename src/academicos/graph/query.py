"""Graph query helpers: typed traversals used by retrieval and agents."""
from __future__ import annotations

import json
from typing import Any, Optional

from ..algorithms.mastery import KnowledgeMastery
from ..models.enums import EdgeType, NodeType
from .store import GraphStore


def find_concept_chain(store: GraphStore, concept_label: str) -> dict[str, Any]:
    """Return concept node + its prerequisite graph (2-hop) and related docs."""
    nodes = store.query_nodes(NodeType.CONCEPT, label_contains=concept_label, limit=5)
    if not nodes:
        return {"concept": None, "prereqs": [], "related": []}
    nid = nodes[0].id
    nb = store.neighbors(nid, max_depth=2)
    prereqs = [x for x in nb if x["edge"].type == EdgeType.PREREQUISITE_OF]
    related = [x for x in nb if x["edge"].type in (EdgeType.RELATED_TO, EdgeType.SUPPORTED_BY)]
    return {"concept": nodes[0], "prereqs": prereqs, "related": related}


def prereq_plan(store: GraphStore, concept_label: str,
                min_confidence: float = 0.0,
                max_depth: int = 4,
                ntype: NodeType = NodeType.CONCEPT,
                learner=None,
                mastery_gate: float = 0.5) -> dict[str, Any]:
    """Study plan for a concept from the enriched PREREQUISITE_OF edges.

    Walks prerequisite edges *backward* (target = concept being prepared for)
    and returns a topological study order: foundational concepts first.

    Edges carry the estimator's properties (score, confidence, spacing);
    consumers can gate on min_confidence (see docs/research: conf >= 0.40
    trades recall for precision). Pass ntype=NodeType.LEARNING_OUTCOME to plan
    the LO layer. Pass a LearnerModel (e.g. replayed from the event store)
    plus mastery_gate to drop prerequisites the learner has already mastered
    (mastery >= gate); LO-layer plans resolve mastery through their MAPS_TO
    concept. Returns:
      {"concept": label, "steps": [{label, mastered, mastery, prereq_of,
        difficulty, learning_hours, bloom, score, confidence, spacing_days,
        transitive, criteria}, ...]}  — steps sorted so every prereq precedes
        its later; mastered steps are omitted from the study path.
    """
    nodes = store.query_nodes(ntype, label_contains=concept_label, limit=5)
    if not nodes:
        return {"concept": None, "steps": []}
    target = nodes[0]

    # edges: (prereq_node, later_node, attrs) collected backward from target
    pr_edges: list[tuple[Any, Any, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    frontier = [(target, 0)]
    while frontier:
        later, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        rows = store.conn.execute(
            """SELECT source, target, attributes FROM edges
               WHERE type=? AND target=? AND status='draft'""",
            (EdgeType.PREREQUISITE_OF.value, later.id)).fetchall()
        for r in rows:
            attrs = json.loads(r["attributes"]) if r["attributes"] else {}
            if attrs.get("confidence", 1.0) < min_confidence:
                continue
            src = store.get_node(r["source"])
            if src is None:
                continue
            key = (src.id, later.id)
            if key in seen:
                continue
            seen.add(key)
            pr_edges.append((src, later, attrs))
            frontier.append((src, depth + 1))

    # topological order: Kahn's algorithm over prereq -> later edges.
    # A node is ready once all its prerequisites have been processed.
    in_deg: dict[str, int] = {}
    out: dict[str, list[Any]] = {}
    by_id: dict[str, Any] = {target.id: target}
    for src, later, attrs in pr_edges:
        by_id.setdefault(src.id, src)
        by_id.setdefault(later.id, later)
        in_deg[later.id] = in_deg.get(later.id, 0) + 1
        out.setdefault(src.id, []).append((src, later, attrs))
    ready = [nid for nid in by_id if in_deg.get(nid, 0) == 0]
    order: list[str] = []
    while ready:
        ready.sort()
        nid = ready.pop(0)
        order.append(nid)
        for src, later, attrs in out.get(nid, []):
            in_deg[later.id] -= 1
            if in_deg[later.id] == 0:
                ready.append(later.id)

    mastery_of: dict[str, float] = {}
    if learner is not None:
        for nid, node in by_id.items():
            m = _mastery(learner, store, nid, node.label, ntype)
            if m is not None:
                mastery_of[nid] = m

    steps = []
    for nid in order:
        if nid == target.id:
            continue
        node = by_id[nid]
        mastered = mastery_of.get(nid, 0.0) >= mastery_gate
        if mastered:
            continue
        a = node.attributes or {}
        step = {
            "label": node.label,
            "mastered": False,
            "mastery": round(mastery_of.get(nid, 0.0), 4) or None,
            "difficulty": a.get("difficulty"),
            "learning_hours": a.get("learning_hours"),
            "bloom": a.get("bloom"),
            "spacing_days": None,
            "score": None,
            "confidence": None,
            "transitive": False,
            "criteria": {},
        }
        # attach the strongest edge from this node toward the plan target
        for src, later, attrs in pr_edges:
            if src.id != nid:
                continue
            if later.id not in order and later.id != target.id:
                continue
            cand = (attrs.get("score") or 0.0, attrs.get("confidence") or 0.0)
            if cand[0] > (step["score"] or 0.0):
                step["score"] = cand[0]
                step["confidence"] = cand[1]
                step["spacing_days"] = attrs.get("recommended_spacing_days")
                step["transitive"] = bool(attrs.get("transitive"))
                step["criteria"] = attrs.get("criteria") or {}
        steps.append(step)
    target_mastered = mastery_of.get(target.id, 0.0) >= mastery_gate
    steps.append({"label": target.label,
                  "mastered": target_mastered,
                  "mastery": round(mastery_of.get(target.id, 0.0), 4) or None,
                  "difficulty": (target.attributes or {}).get("difficulty"),
                  "learning_hours": (target.attributes or {}).get("learning_hours"),
                  "bloom": (target.attributes or {}).get("bloom"),
                  "score": None, "confidence": None, "spacing_days": None,
                  "transitive": False, "criteria": {}, "target": True})
    return {"concept": target.label, "steps": steps}


def _mastery(learner, store: GraphStore, nid: str, label: str,
             ntype: NodeType) -> float | None:
    """Learner mastery for a graph node: exact node id, then label, then
    (for LO nodes) the MAPS_TO concept's mastery.

    Mastery is *computed* here via ``KnowledgeMastery`` rather than read off
    ``ConceptState.mastery``: that attribute is a cache only ``demo()`` ever
    fills in, so on any real learner (``EventStore.replay`` -> ``observe`` ->
    ``record``) it stays 0.0 forever. Reading it made every prereq look
    unmastered (or, for hand-seeded states, silently trusted a stale value).
    Same defect class as the daily_loop calibration-advice gate and the
    RevisionScheduler importance weight; see docs/compliance.md.
    """
    _km = KnowledgeMastery()
    for key in (nid, label):
        cs = learner.concepts.get(key)
        if cs is not None:
            return _km.score(learner, key).mastery
    if ntype == NodeType.LEARNING_OUTCOME:
        rows = store.conn.execute(
            """SELECT source FROM edges WHERE type=? AND target=?""",
            (EdgeType.MAPS_TO.value, nid)).fetchall()
        for r in rows:
            for key in (r["source"], store.get_node(r["source"]).label if store.get_node(r["source"]) else None):
                if not key:
                    continue
                cs = learner.concepts.get(key)
                if cs is not None:
                    return _km.score(learner, key).mastery
    return None


def exam_pattern_for(store: GraphStore, subject: str, year: str) -> list[dict[str, Any]]:
    """Blueprint + paper + scheme objects for a subject/year."""
    out: list[dict[str, Any]] = []
    for ntype in (NodeType.ASSESSMENT_BLUEPRINT, NodeType.QUESTION_PAPER, NodeType.ANSWER_SCHEME):
        for n in store.query_nodes(ntype, label_contains=subject, limit=50):
            attrs = n.attributes or {}
            if attrs.get("academic_year") and year not in str(attrs["academic_year"]):
                continue
            out.append({"node": n, "attrs": attrs})
    return out


def evidence_chain(store: GraphStore, node_id: str) -> list[dict[str, Any]]:
    """Walk EVIDENCED_BY / DERIVED_FROM edges to the source documents."""
    return store.neighbors(node_id, max_depth=3)


def subgraph_for_query(store: GraphStore, query: str, max_concepts: int = 3,
                       max_depth: int = 2) -> dict[str, Any]:
    """Concept-anchored sub-graph expansion (HyKGE-style).

    1. Extract candidate concept labels from the query (noun phrases via heuristics).
    2. Find matching Concept nodes (label contains / token overlap).
    3. Expand each to a 2-hop neighborhood; keep the highest-weight union.
    Returns {"concepts": [Node], "nodes": [Node], "edges": [Edge], "paths": [path ids]}.
    """
    from ..models.enums import NodeType

    labels = candidate_concepts(query)
    concepts = []
    for label in labels[:max_concepts]:
        for n in store.query_nodes(NodeType.CONCEPT, label_contains=label, limit=3):
            if n not in concepts:
                concepts.append(n)
    if not concepts:
        return {"concepts": [], "nodes": [], "edges": [], "paths": []}

    seen_nodes: dict[str, Any] = {}
    seen_edges: dict[str, Any] = {}
    for c in concepts:
        nb = store.neighbors(c.id, max_depth=max_depth)
        for hit in nb:
            seen_nodes.setdefault(hit["node"].id, hit["node"])
            seen_edges.setdefault(hit["edge"].id, hit["edge"])
    return {
        "concepts": [c.label for c in concepts],
        "nodes": list(seen_nodes.values()),
        "edges": list(seen_edges.values()),
        "paths": [[c.label, *[e.type.value for e in _walk(store, c.id, depth=max_depth)]] for c in concepts],
    }


def candidate_concepts(query: str) -> list[str]:
    """Crude noun-phrase extraction for concept anchoring: capitalized phrases and
    long token sequences that are not stopwords. Heuristic; LLM NER lands Phase 4."""
    import re

    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)
    stop = {"the", "and", "for", "with", "what", "how", "why", "when", "which",
            "explain", "define", "list", "state", "describe", "marks", "class",
            "question", "paper", "section", "part", "difference", "between"}
    return [w for w in words if w.lower() not in stop][:5]


def _walk(store: GraphStore, nid: str, depth: int) -> list[Any]:
    nb = store.neighbors(nid, max_depth=depth, max_nodes=8)
    return [h["edge"] for h in nb]
