"""Agent toolkit: tool protocol + built-in tools over retrieval/graph."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..graph.store import GraphStore
from ..models.common import Evidence
from ..retrieval.hybrid import HybridRetriever


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...


@dataclass
class ToolResult:
    tool: str
    ok: bool
    payload: Any = None
    error: str | None = None


class SearchTool(Tool):
    name = "search"
    description = "Semantic + lexical search over the CBSE corpus. Params: query (str), limit (int, default 8)."

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, query: str, limit: int = 8, **_: Any) -> ToolResult:
        try:
            hits = self.retriever.search(query, limit=limit)
            payload = [{
                "document_id": h.document_id,
                "page": h.chunk.page,
                "heading": h.chunk.heading,
                "score": round(h.score, 3),
                "text": h.chunk.text[:600],
            } for h in hits]
            return ToolResult(tool=self.name, ok=True, payload=payload)
        except Exception as e:
            return ToolResult(tool=self.name, ok=False, error=str(e))


class ConceptLookupTool(Tool):
    name = "concept_lookup"
    description = "Find a concept in the knowledge graph plus its prerequisites. Params: concept (str)."

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run(self, concept: str, **_: Any) -> ToolResult:
        from ..graph.query import find_concept_chain
        try:
            out = find_concept_chain(self.graph, concept)
            if out["concept"] is None:
                return ToolResult(tool=self.name, ok=False, error=f"concept not found: {concept}")
            return ToolResult(tool=self.name, ok=True, payload=out)
        except Exception as e:
            return ToolResult(tool=self.name, ok=False, error=str(e))


class EvidencePack:
    """Evidence bundle collected by an agent run (for governance/audit)."""
    def __init__(self) -> None:
        self.evidence = Evidence()
        self.tool_calls: list[dict[str, Any]] = []
        self.messages: list[dict[str, str]] = field(default_factory=list)
