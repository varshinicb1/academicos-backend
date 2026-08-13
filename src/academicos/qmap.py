"""P1.9 — Question → Concept mapping (LLM-assisted, verifier-checked).

Given a question (as found in a workbook / board paper / an LMS log), decide
*which knowledge-graph concepts it exercises*. This unlocks per-question
tracing: every solve records outcomes against concrete `ConceptState`s, so
mastery, retention and revision all become measurable at question level.

Two stages, by design (proposal → verification):

1. **Proposal.** An LLM (Sarvam 30b/105b) reads the question and the
   candidate concept spine and returns `[{name, bloom, confidence}]`.
   When no LLM is available the module falls back to a lexical
   proposer (token-overlap over graph concept labels) so the whole
   pipeline works offline and is testable without a key.
2. **Verification.** Every proposed concept must resolve to a real node in
   the graph (`CONCEPT` nodes). A proposed concept that has no graph node
   is *not* silently accepted: it is returned with `verified=False` and
   `out_of_graph=True`, and the caller decides (create the node via
   concept-profile later, or drop it). Verified mappings carry the resolved
   `node_id`, so downstream tracing uses canonical ids (case-folded like
   the graph builder does).

The Bloom level LLM-proposed is cross-checked against the node's profiled
`bloom` when available; conflicts keep the graph-profiled value and flag
`bloom_conflict=True`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models.enums import NodeType
from .graph.store import GraphStore

PROPOSE_SYSTEM = (
    "You are an exam question → concept mapper for an Indian CBSE study system. "
    "Given a question and a concept glossary, return the concepts the question "
    "actually exercises (testing, not just mentioning). Respond ONLY with JSON:\n"
    "{\"concepts\": [{\"name\": \"exact glossary concept name\", \"bloom\": "
    "\"remember|understand|apply|analyze|evaluate|create\", "
    "\"confidence\": 0.0-1.0}]}"
)

CONCEPT_SEP = " • "


def _canonical(label: str) -> str:
    """Normalize like the graph builder (`_entity_part`): lower, collapse spaces."""
    return " ".join(label.split()).lower()


@dataclass
class QuestionConcept:
    name: str
    bloom: str | None = None
    confidence: float = 0.0
    verified: bool = False
    node_id: str | None = None
    out_of_graph: bool = False
    bloom_conflict: bool = False


@dataclass
class QuestionMap:
    question: str
    concepts: list[QuestionConcept] = field(default_factory=list)
    method: str = "lexical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "method": self.method,
            "concepts": [
                {"name": c.name, "bloom": c.bloom, "confidence": round(c.confidence, 3),
                 "verified": c.verified, "node_id": c.node_id,
                 "out_of_graph": c.out_of_graph, "bloom_conflict": c.bloom_conflict}
                for c in self.concepts
            ],
        }


class QuestionMapper:
    """Propose + verify question→concept links against the knowledge graph."""

    def __init__(self, store: GraphStore, llm=None,
                 min_lexical_score: float = 0.18, max_concepts: int = 6):
        self.store = store
        self.llm = llm  # .available, .chat_json (duck-typed; None => lexical)
        self.min_lexical_score = min_lexical_score
        self.max_concepts = max_concepts
        self._nodes: list | None = None  # real CONCEPT Nodes from the store

    def map(self, question: str) -> QuestionMap:
        proposals = (self._propose_llm(question)
                     if self._llm_ok() else self._propose_lexical(question))
        return self._verify(question, proposals)

    def _llm_ok(self) -> bool:
        try:
            return bool(self.llm and self.llm.available)
        except Exception:
            return False

    # ---------------- proposal ----------------
    def _propose_llm(self, question: str) -> list[tuple[str, str | None, float]]:
        glossary = self._concept_glossary()
        if not glossary:
            return []
        out = self.llm.chat_json(
            [{"role": "system", "content": PROPOSE_SYSTEM},
             {"role": "user", "content":
                 f"Concept glossary (propose only concepts from this list):\n"
                 f"{CONCEPT_SEP.join(glossary)}\n\nQuestion:\n{question}"}],
            temperature=0.0, max_tokens=200,
        )
        results: list[tuple[str, str | None, float]] = []
        for c in (out or {}).get("concepts", [])[:self.max_concepts]:
            name = str(c.get("name", "")).strip()
            if not name:
                continue
            bloom = str(c.get("bloom", "")).strip().lower() or None
            conf = float(c.get("confidence", 0.5))
            results.append((name, bloom, conf))
        return results

    def _concept_glossary(self) -> list[str]:
        return [n.label for n in self._graph_nodes()]

    def _graph_nodes(self) -> list:
        if self._nodes is None:
            nodes = self.store.query_nodes(NodeType.CONCEPT, limit=5000)
            self._nodes = [n for n in nodes if n.label and len(n.label) > 2]
        return self._nodes

    def _propose_lexical(self, question: str) -> list[tuple[str, str | None, float]]:
        """Stopword-resistant: concept labels whose phrase overlaps the question
        (token-jaccard ≥ min_lexical_score)."""
        q_tokens = _tokenize(question)
        scored: list[tuple[float, str | None, str, dict]] = []
        for n in self._graph_nodes():
            label_tokens = set(_tokenize(n.label))
            if not label_tokens:
                continue
            overlap = len(q_tokens & label_tokens) / (len(label_tokens | q_tokens) or 1)
            if overlap >= self.min_lexical_score:
                bloom = (n.attributes or {}).get("bloom", "")
                scored.append((overlap, n.label, str(bloom).lower() or None,
                               n.attributes or {}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(label, bloom, score) for score, label, bloom, _ in scored
                [:self.max_concepts]]

    # ---------------- verification ----------------
    def _verify(self, question: str, proposals) -> QuestionMap:
        concepts: list[QuestionConcept] = []
        seen: set[str] = set()
        for name, bloom, conf in proposals:
            key = _canonical(name)
            if key in seen:
                continue
            seen.add(key)
            node = self._resolve(name, key)
            verified = node is not None
            bloom_conflict = False
            if verified and bloom:
                node_bloom = (node.attributes or {}).get("bloom", "").lower()
                if node_bloom and bloom != node_bloom:
                    bloom_conflict = True
                    bloom = node_bloom
            concepts.append(QuestionConcept(
                name=name, bloom=bloom, confidence=conf, verified=verified,
                node_id=node.id if node else None,
                out_of_graph=not verified,
                bloom_conflict=bloom_conflict,
            ))
        method = "llm+verify" if self._llm_ok() else "lexical+verify"
        return QuestionMap(question=question, concepts=concepts, method=method)

    def _resolve(self, name: str, key: str):
        """Exact case-folded label match first, then containment either way."""
        for n in self._graph_nodes():
            if _canonical(n.label) == key:
                return n
        for n in self._graph_nodes():
            lower = n.label.lower()
            if key in lower or lower in key:
                return n
        return None


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS}


_STOPWORDS = frozenset(
    "a an the is are was were be been being in on at to of and or for with "
    "from by as what which when where who whom how why if then than so into "
    "using use used does do did not no yes about between without".split()
)