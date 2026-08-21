"""Agents: curriculum-doubt-solver, examiner-answer-scorer, paper-analyst.

Deterministic (tool-driven) in v0.1, with an optional Self-RAG-style reflection
critic (Sarvam LLM) that decides whether to retrieve and scores each
evidence path on relevance/support/utility. Answers always cite evidence
from the corpus.
"""
from __future__ import annotations

from ..graph.store import GraphStore
from ..llm.sarvam import SarvamLLM
from ..models.common import Confidence, Evidence, Provenance, SourceSpan
from ..models.enums import ExtractionMethod
from ..retrieval.hybrid import HybridRetriever
from .base import EvidencePack, SearchTool, Tool, ToolResult


class Agent:
    name = "agent"

    def __init__(self, retriever: HybridRetriever, graph: GraphStore | None = None,
                 critic=None):
        self.retriever = retriever
        self.graph = graph
        self.critic = critic
        self.tools: list[Tool] = [SearchTool(retriever)]
        if graph:
            from .base import ConceptLookupTool
            self.tools.append(ConceptLookupTool(graph))

    def _search(self, query: str, limit: int = 6) -> list[dict]:
        for t in self.tools:
            if t.name == "search":
                r = t.run(query=query, limit=limit)
                return r.payload if r.ok else []
        return []

    def _make_evidence(self, hits: list[dict]) -> Evidence:
        ev = Evidence()
        for h in hits[:5]:
            ev.add(Provenance(
                source=SourceSpan(document_id=h["document_id"], page=h.get("page", 1)),
                method=ExtractionMethod.PDF_NATIVE,
                confidence=Confidence(method=ExtractionMethod.PDF_NATIVE, score=1.0),
            ))
        return ev


class DoubtSolver(Agent):
    """Given a doubt, retrieve syllabus/chapter/LO context and answer with citations.

    With a critic attached: decides on-demand whether retrieval is needed
    (Self-RAG IsRel), then scores each retrieved path on relevance/support/
    utility and returns the best one (segment-wise path selection).
    """
    name = "doubt-solver"

    def solve(self, doubt: str) -> dict:
        reflections = None
        if self.critic is not None:
            reflections = self._solve_with_critic(doubt)
            if reflections is not None:
                return reflections
        hits = self._search(doubt, limit=8)
        return {
            "agent": self.name,
            "doubt": doubt,
            "answer": _answer_from_hits(hits),
            "citations": [_cite(h) for h in hits],
            "evidence": [{"document_id": h["document_id"], "page": h.get("page")} for h in hits[:5]],
            "reflection": {"used_critic": False},
        }

    def _solve_with_critic(self, doubt: str) -> dict | None:
        from .critic import ReflectionScores

        if not self.critic.llm.available:
            return None
        retrieved = self.critic.decide_retrieve(doubt)
        if not retrieved:
            return {
                "agent": self.name,
                "doubt": doubt,
                "answer": "No retrieval needed for this doubt; answerable from general knowledge. "
                          "(Self-RAG IsRel = no retrieval.)",
                "citations": [],
                "evidence": [],
                "reflection": {"used_critic": True, "retrieved": False},
            }
        hits = self._search(doubt, limit=6)
        if not hits:
            return {
                "agent": self.name,
                "doubt": doubt,
                "answer": "No evidence found in the corpus for this doubt.",
                "citations": [],
                "evidence": [],
                "reflection": {"used_critic": True, "retrieved": True, "paths": []},
            }
        candidate = _answer_from_hits(hits)
        scored: list[tuple[float, dict, ReflectionScores]] = []
        for h in hits[:3]:
            answer_variant = (
                f"Based on the corpus, the most relevant passage is:\n"
                f"[{h['document_id']} p.{h.get('page', '?')}] {h['text'][:400]}"
            )
            rs = self.critic.score(doubt, answer_variant, h["text"][:600])
            scored.append((rs.final(self.critic.w_rel, self.critic.w_sup, self.critic.w_use), h, rs))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_hit, best_rs = scored[0]
        paths = [
            {
                "document_id": h["document_id"],
                "page": h.get("page"),
                "score": round(s, 3),
                "relevance": rs.relevance,
                "support": rs.support,
                "utility": rs.utility,
            }
            for s, h, rs in scored
        ]
        return {
            "agent": self.name,
            "doubt": doubt,
            "answer": (
                f"Based on the corpus, the most relevant passage is:\n"
                f"[{best_hit['document_id']} p.{best_hit.get('page', '?')}] {best_hit['text'][:500]}"
            ),
            "citations": [_cite(h) for _, h, _ in scored],
            "evidence": [{"document_id": h["document_id"], "page": h.get("page")} for _, h, _ in scored[:3]],
            "reflection": {
                "used_critic": True,
                "retrieved": True,
                "best_score": round(best_score, 3),
                "weights": {"relevance": self.critic.w_rel, "support": self.critic.w_sup, "utility": self.critic.w_use},
                "paths": paths,
            },
        }


class ExaminerScorer(Agent):
    """Score a student answer against marking points from the corpus scheme."""
    name = "examiner-scorer"

    def score(self, question: str, student_answer: str, marks_available: float = 0.0) -> dict:
        hits = self._search(f"marking scheme {question}", limit=8)
        scheme = [h for h in hits if "marking" in (h.get("document_id", "") + h.get("heading", "")).lower()] or hits
        return {
            "agent": self.name,
            "score": {"earned": None, "available": marks_available, "scheme_found": bool(scheme)},
            "scheme_citations": [_cite(h) for h in scheme[:5]],
            "note": "v0.1 heuristic scorer; LLM rubric application lands in Phase 4.",
        }


class PaperAnalyst(Agent):
    """Analyze a question paper: structure, marks distribution, bloom coverage."""
    name = "paper-analyst"

    def analyze(self, paper_id: str) -> dict:
        hits = self._search(f"question paper {paper_id} marks", limit=10)
        sections = {}
        for h in hits:
            head = h.get("heading", "")
            if head:
                sections.setdefault(head, []).append(h["text"][:120])
        return {
            "agent": self.name,
            "paper_id": paper_id,
            "sections_found": list(sections),
            "top_chunks": [_cite(h) for h in hits[:8]],
        }


def _answer_from_hits(hits: list[dict]) -> str:
    if not hits:
        return "No evidence found in the corpus for this doubt. Try rephrasing or check the syllabus documents."
    return (
        "Based on the corpus, the following passages are most relevant to your doubt:\n\n"
        + "\n\n".join(f"[{i + 1}] ({h['document_id']} p.{h.get('page', '?')}) {h['text'][:400]}"
                      for i, h in enumerate(hits[:3]))
    )


def _cite(h: dict) -> dict:
    return {"document_id": h["document_id"], "page": h.get("page", 1), "heading": h.get("heading", "")}
