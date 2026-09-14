"""FastAPI app: the AcademicOS public surface (v1).

Endpoints implemented: search, agent-doubt, agent-score, agent-analyze, graph
neighbors/paths, registry stats, ingest trigger, revise, plan, question-solve.
Storage and graph are constructed once at startup from Config.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agents.orchestrator import DoubtSolver, ExaminerScorer, PaperAnalyst
from ..assessment import auth_routes, ingest_routes, mobile_routes, mobile_scan, pillar_routes
from ..assessment import routes as assessment_routes
from ..config import Config
from ..curriculum import routes as curriculum_routes
from ..graph.store import GraphStore
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.index import ChunkIndex
from ..storage.event_store import EventStore
from ..storage.question_map import QuestionMapStore
from ..storage.registry import SourceRegistry

app = FastAPI(title="AcademicOS", version="0.1.0",
              description="CBSE/NCERT Academic Brain — evidence-grounded retrieval + reasoning")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(assessment_routes.router)
app.include_router(pillar_routes.router)
app.include_router(mobile_routes.router)
app.include_router(ingest_routes.router)
app.include_router(auth_routes.router)
app.include_router(curriculum_routes.router)


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


class ReviseRequest(BaseModel):
    learner: str = "default"
    model: str = Field(default="fsrs", pattern="^(fsrs|exponential)$")


class StudyPlanRequest(BaseModel):
    target: str = Field(min_length=1)
    learner: Optional[str] = None
    model: str = Field(default="fsrs", pattern="^(fsrs|exponential)$")
    capacity_minutes: Optional[int] = Field(default=None, ge=1)
    lo: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mastery_gate: float = Field(default=0.5, ge=0.0, le=1.0)


class QuestionSolveRequest(BaseModel):
    question: str = Field(min_length=2)


_retriever: Optional[HybridRetriever] = None
_graph: Optional[GraphStore] = None
_registry: Optional[SourceRegistry] = None
_events: Optional[EventStore] = None
_qmap_store: Optional[QuestionMapStore] = None
_critic: Optional[Any] = None


def init_runtime(config: Optional[Config] = None) -> None:
    global _retriever, _graph, _registry, _events, _qmap_store, _critic
    cfg = config or Config.load()
    index = ChunkIndex(cfg.index_db)
    _graph = GraphStore(cfg.graph_db)
    _retriever = HybridRetriever(index, _graph)
    _registry = SourceRegistry(cfg.registry_db)
    _events = EventStore(cfg.events_db)
    _qmap_store = QuestionMapStore(cfg.question_map_db)
    assessment_routes.init(cfg)
    pillar_routes.init(cfg)
    ingest_routes.init(cfg)
    auth_routes.init(cfg)
    curriculum_routes.init(cfg)
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


# Durability-critical stores (see AGENTS.md's storage inventory and
# curriculum/store.py's module docstring): AssessmentStore, PaperStore,
# PracticeStore, GradedStore, ScanSessionStore, TemplateStore, UserStore
# and AuditLog use the per-row SupabaseTable fallback pattern; EventStore
# joined them in this pass; CurriculumStore uses whole-file snapshot/
# restore via SupabaseStorage instead (too relational to wrap per-table).
_DURABLE_STORES = [
    "AssessmentStore", "PaperStore", "PracticeStore", "GradedStore",
    "ScanSessionStore", "TemplateStore", "UserStore", "AuditLog",
    "EventStore", "CurriculumStore (snapshot/restore)",
]


@app.get("/health/storage")
def health_storage() -> dict:
    """Answers "is this actually live" without guessing from Render's
    dashboard or grepping env vars on the host -- whether Supabase is
    configured at all, and (only if so) whether it's reachable right now
    via one cheap, unfiltered, limit-1 select."""
    cfg = Config.load()
    reachable: Optional[bool] = None
    if cfg.supabase_enabled:
        from ..assessment.supabase_kv import SupabaseTable
        try:
            SupabaseTable("assessments").select(limit=1)
            reachable = True
        except requests.exceptions.RequestException:
            reachable = False
    return {
        "supabase_configured": cfg.supabase_enabled,
        "supabase_reachable": reachable,
        "durable_stores": _DURABLE_STORES,
    }


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


@app.post("/v1/revise")
def revise(req: ReviseRequest) -> dict:
    """P5.2 — what to revise today (FSRS-backed, from the learner's event store)."""
    if not _events:
        raise HTTPException(503, "runtime not initialized")
    from ..algorithms.forgetting import ForgettingModel, ForgettingParams, RevisionScheduler

    model = _events.replay(req.learner)
    if not model.concepts:
        raise HTTPException(404, f"no interactions recorded for learner '{req.learner}'")
    sched = RevisionScheduler(ForgettingModel(ForgettingParams(model=req.model)))
    plan = sched.plan(model)
    return {
        "learner": model.learner_id,
        "due_today": [{"concept_id": cid, "retention": round(retention, 4), "priority": round(priority, 4)}
                      for cid, retention, priority in plan.due_today],
        "next_reviews": [{"concept_id": cid, "due": due} for cid, due in plan.next_reviews],
        "totals": plan.totals,
        "confident": plan.confident,
        "minutes_per_concept": sched.minutes_per_concept,
    }


@app.post("/v1/plan")
def study_plan(req: StudyPlanRequest) -> dict:
    """P5.2 — today's study session: prereqs in order, FSRS-due items first,
    bounded by a daily capacity window."""
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    from ..algorithms.forgetting import ForgettingModel, ForgettingParams
    from ..algorithms.study_planner import StudyPlanner

    learner = _events.replay(req.learner) if req.learner and _events else None
    planner = StudyPlanner(
        _graph, ForgettingModel(ForgettingParams(model=req.model)),
        default_capacity_minutes=req.capacity_minutes or 15)
    plan = planner.plan(req.target, capacity_minutes=req.capacity_minutes,
                        learner=learner, lo=req.lo,
                        min_confidence=req.min_confidence,
                        mastery_gate=req.mastery_gate)
    if not plan.totals:
        raise HTTPException(404, f"no plan path for '{req.target}' (is it in the graph?)")
    return {
        "target": plan.target,
        "session": [asdict(i) for i in plan.session],
        "upcoming": [asdict(i) for i in plan.upcoming],
        "totals": plan.totals,
        "params": plan.params,
    }


@app.post("/v1/question/solve")
def question_solve(req: QuestionSolveRequest) -> dict:
    """P5.2 — question-solve with evidence: map a question to graph concepts
    (LLM-assisted, verifier-checked) and persist the mapping for tracing."""
    if not _graph or not _qmap_store:
        raise HTTPException(503, "runtime not initialized")
    from ..qmap import QuestionMapper

    qmap = QuestionMapper(_graph).map(req.question)
    _qmap_store.append(qmap.to_dict())
    return qmap.to_dict()
