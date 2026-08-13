"""Graph builder: canonical objects -> versioned nodes/edges.

Maps the AcademicObject layer onto the graph with content-addressed ids so
re-ingestion after content changes produces new versions instead of dupes.
"""
from __future__ import annotations

import logging
from typing import Any

from ..models.academic import (
    AcademicObject,
    AnswerScheme,
    Chapter,
    Competency,
    Concept,
    LearningOutcome,
    MarkingPoint,
    Question,
    QuestionPaper,
    Subject,
    Topic,
)
from ..models.enums import EdgeType, NodeType
from .schema import edge_id, new_edge, new_node, node_id
from .store import GraphStore

log = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, store: GraphStore):
        self.store = store

    def upsert_object(self, obj: AcademicObject) -> str:
        ntype = _NODE_TYPE.get(type(obj).__name__, NodeType.DOCUMENT)
        node = new_node(
            ntype, obj.title[:200],
            id_=_canonical_id(obj),
            attributes=_attrs(obj),
            provenance=obj.evidence.top(1)[0] if obj.evidence.items else None,
        )
        self.store.upsert_node(node)
        return node.id

    def link(self, source: str, target: str, etype: EdgeType, weight: float = 1.0,
             attrs: dict[str, Any] | None = None) -> None:
        self.store.upsert_edge(new_edge(etype, source, target, weight=weight, attributes=attrs))

    def build_question(self, q: Question) -> str:
        qid = self.upsert_object(q)
        for p in q.parts:
            pid = self.upsert_object(p)
            self.link(qid, pid, EdgeType.CONTAINS)
        return qid

    def build_answer_scheme(self, scheme: AnswerScheme) -> str:
        sid = self.upsert_object(scheme)
        for mp in scheme.marking_points:
            mpid = self.upsert_object(mp)
            self.link(sid, mpid, EdgeType.HAS_MARKING_POINT)
        return sid

    def build_paper(self, paper: QuestionPaper) -> str:
        pid = self.upsert_object(paper)
        for qid in paper.questions:
            self.link(pid, qid, EdgeType.CONTAINS)
        return pid

    def build_triples(self, document_id: str, triples: list[tuple[str, str, str]],
                      page: int = 0) -> tuple[int, int]:
        """OpenIE triples -> Concept nodes + RELATED_TO edges (HippoRAG-style)."""
        from ..models.common import Confidence, Provenance, SourceSpan
        from ..models.enums import ExtractionMethod

        prov = Provenance(
            source=SourceSpan(document_id=document_id, page=page),
            method=ExtractionMethod.LLM,
            confidence=Confidence(method=ExtractionMethod.LLM, score=0.9),
        )
        entity_ids: dict[str, str] = {}
        node_count = 0
        edge_count = 0
        for subj, pred, obj in triples:
            for ent in (subj, obj):
                eid = node_id(NodeType.CONCEPT, _entity_part(ent))
                if eid not in entity_ids:
                    existing = self.store.get_node(eid)
                    self.store.upsert_node(new_node(
                        NodeType.CONCEPT, ent, id_=eid,
                        attributes={"aliases": [ent]} if not existing else None,
                        provenance=prov,
                    ))
                    entity_ids[ent] = eid
                    node_count += 1
            s, t = entity_ids[subj], entity_ids[obj]
            if s == t:
                continue
            old = self.store.get_edge(edge_id(EdgeType.RELATED_TO, s, t))
            attrs = {"predicates": [pred], "occurrences": 1}
            if old:
                attrs = dict(old.attributes or {})
                attrs["occurrences"] = int(attrs.get("occurrences", 1)) + 1
                if pred not in attrs.get("predicates", []):
                    attrs.setdefault("predicates", []).append(pred)
            self.store.upsert_edge(new_edge(
                EdgeType.RELATED_TO, s, t, weight=1.0,
                attributes=attrs,
                provenance=prov,
            ))
            edge_count += 1
        return node_count, edge_count


def _canonical_id(obj: AcademicObject) -> str:
    return obj.canonical_id


def _entity_part(name: str) -> str:
    """Normalize an entity name for use in a content-addressed node id.

    Case-folded so 'Fractions' / 'fractions' / 'FRACTIONS' collapse to one node.
    """
    cleaned = " ".join(str(name).split()).replace(":", "-")
    return (cleaned.lower()[:120] or "entity")


def _attrs(obj: AcademicObject) -> dict[str, Any]:
    out = {
        "grade": obj.grade,
        "subject": obj.subject,
        "academic_year": obj.academic_year,
    }
    for k in ("code", "chapter_no", "seq", "marks", "q_no", "section", "question_type",
              "cognitive", "difficulty", "marking_points", "expected_answer",
              "total_marks", "duration_min", "unit_allocations", "circular_no", "date"):
        v = getattr(obj, k, None)
        if v is not None:
            out[k] = v
    if isinstance(obj, Question) and obj.cognitive:
        out["cognitive"] = obj.cognitive.value
    return out


_NODE_TYPE: dict[str, NodeType] = {
    "Subject": NodeType.SUBJECT,
    "Chapter": NodeType.CHAPTER,
    "Topic": NodeType.TOPIC,
    "Subtopic": NodeType.SUBTOPIC,
    "LearningOutcome": NodeType.LEARNING_OUTCOME,
    "Competency": NodeType.COMPETENCY,
    "Concept": NodeType.CONCEPT,
    "Prerequisite": NodeType.PREREQUISITE,
    "Question": NodeType.QUESTION,
    "QuestionPart": NodeType.QUESTION_PART,
    "QuestionPaper": NodeType.QUESTION_PAPER,
    "AnswerScheme": NodeType.ANSWER_SCHEME,
    "MarkingPoint": NodeType.MARKING_POINT,
    "AssessmentBlueprint": NodeType.ASSESSMENT_BLUEPRINT,
    "Circular": NodeType.CIRCULAR,
    "SyllabusItem": NodeType.SYLLABUS_ITEM,
}
