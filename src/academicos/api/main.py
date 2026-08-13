"""FastAPI app: the AcademicOS public surface (v1).

Endpoints implemented: search, agent-doubt, agent-score, agent-analyze, graph
neighbors/paths, registry stats, ingest trigger. Storage and graph are
constructed once at startup from Config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agents.orchestrator import DoubtSolver, ExaminerScorer, PaperAnalyst
from ..assessment import mobile_routes, mobile_scan, pillar_routes
from ..assessment import routes as assessment_routes
from ..config import Config
from ..graph.store import GraphStore
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.index import ChunkIndex
from ..storage.registry import SourceRegistry

app = FastAPI(title="AcademicOS", version="0.1.0",
              description="CBSE/NCERT Academic Brain — evidence-grounded retrieval + reasoning")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(assessment_routes.router)
app.include_router(pillar_routes.router)
app.include_router(mobile_routes.router)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=10, ge=1, le=50)


class DoubtRequest(BaseModel):
    doubt: str = Field(min_length=2)


class ScoreRequest(BaseModel):
    question: str
    student_answer: str = ""
    marks_available: float = 0.0


class AnalyzeRequest(BaseModel):
    paper_id: str


class NodeQuery(BaseModel):
    node_id: str
    max_depth: int = Field(default=2, ge=1, le=6)


class PathQuery(BaseModel):
    source: str
    target: str
    max_depth: int = Field(default=4, ge=1, le=8)


class GraphResult(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


_retriever: Optional[HybridRetriever] = None
_graph: Optional[GraphStore] = None
_registry: Optional[SourceRegistry] = None
_critic: Optional[Any] = None


def init_runtime(config: Optional[Config] = None) -> None:
    global _retriever, _graph, _registry, _critic
    cfg = config or Config.load()
    index = ChunkIndex(cfg.index_db)
    _graph = GraphStore(cfg.graph_db)
    _retriever = HybridRetriever(index, _graph)
    _registry = SourceRegistry(cfg.registry_db)
    assessment_routes.init(cfg)
    pillar_routes.init(cfg)
    mobile_scan.configure_workdir(cfg.data_root / "scan-sessions")
    if cfg.llm_api_key:
        from ..agents.critic import SarvamCritic
        from ..llm.sarvam import SarvamLLM

        llm = SarvamLLM(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url,
                        model=cfg.llm_model, timeout=cfg.llm_timeout)
        if llm.available:
            _critic = SarvamCritic(llm, **cfg.critic_weights,
                                   retrieve_threshold=cfg.critic_retrieve_threshold)


@app.on_event("startup")
def _startup() -> None:
    init_runtime()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/registry/stats")
def registry_stats() -> dict:
    if not _registry:
        raise HTTPException(503, "runtime not initialized")
    return {"count": _registry.count()}


@app.post("/v1/search")
def search(req: SearchRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    hits = _retriever.search(req.query, limit=req.limit)
    return {
        "query": req.query,
        "hits": [{
            "document_id": h.document_id,
            "page": h.chunk.page,
            "heading": h.chunk.heading,
            "score": round(h.score, 4),
            "text": h.chunk.text[:800],
            "sources": h.sources,
        } for h in hits],
    }


@app.post("/v1/agent/doubt")
def agent_doubt(req: DoubtRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    return DoubtSolver(_retriever, _graph, critic=_critic).solve(req.doubt)


@app.post("/v1/agent/score")
def agent_score(req: ScoreRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    return ExaminerScorer(_retriever, _graph).score(req.question, req.student_answer, req.marks_available)


@app.post("/v1/agent/analyze")
def agent_analyze(req: AnalyzeRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    return PaperAnalyst(_retriever, _graph).analyze(req.paper_id)


@app.post("/v1/graph/neighbors")
def graph_neighbors(req: NodeQuery) -> GraphResult:
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    return GraphResult(nodes=[{"id": x["node"].id, "label": x["node"].label,
                               "type": x["node"].type.value, "depth": x["depth"],
                               "edge": x["edge"].type.value} for x in _graph.neighbors(req.node_id, req.max_depth)])


@app.post("/v1/graph/paths")
def graph_paths(req: PathQuery) -> dict:
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    return {"paths": [[e.type.value for e in p] for p in _graph.paths(req.source, req.target, req.max_depth)]}
