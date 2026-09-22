"""FastAPI app: the AcademicOS public surface (v1).

Endpoints implemented: search, agent-doubt, agent-score, agent-analyze, graph
neighbors/paths, registry stats, ingest trigger, revise, plan, question-solve.
Storage and graph are constructed once at startup from Config.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agents.orchestrator import DoubtSolver, ExaminerScorer, PaperAnalyst
from ..assessment import auth_routes, consent_routes, ingest_routes, mobile_routes, mobile_scan, pillar_routes
from ..assessment import grade_by_question
from ..assessment import paper_template_routes
from ..assessment import qbank_routes
from ..assessment import routes as assessment_routes
from ..assessment.supabase_kv import SupabaseUnavailable
from ..assessment.api_keys import ScopeError
from ..assessment.llm_evaluate import LLMEvaluationError
from ..assessment.qbank_store import ReviewStateError
from ..llm.sarvam import LLMProviderError
from ..assessment.postgres_kv import durable_table
from ..assessment.auth_routes import get_current_user
from ..assessment.authz import require_school_owns_student
from ..assessment.users import User
from ..config import (Config, LLMNotEnabled, build_identity, enforce_production_config,
                      get_config, is_production)
from ..curriculum import routes as curriculum_routes
from ..graph.store import GraphStore
from ..llm.budget import LLMBudgetExceeded, llm_budget
from ..llm.telemetry import TELEMETRY
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.index import ChunkIndex
from ..storage.event_store import EventStore
from ..storage.question_map import QuestionMapStore
from ..storage.registry import SourceRegistry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Request bounds and the per-request LLM budget
# --------------------------------------------------------------------------- #
# OWASP's LLM Top 10 (2026) ranks Unbounded Consumption sixth, up four places,
# because the evidence caught up with the theory: their survey found a confirmed
# denial-of-wallet path in 12 of 14 applications with an AI surface. The shape
# is always the same -- an endpoint that costs the caller one HTTP request and
# costs the operator an unbounded amount of inference.
#
# These limits were absent. `DoubtRequest.doubt` was `Field(min_length=2)` with
# no upper bound, and `ScoreRequest.student_answer` had no `Field` at all, so an
# authenticated caller could post an arbitrarily large body that went straight
# into a model prompt. The bounds below are generous enough that no real request
# meets them (the largest legitimate answer sheet is well under 20k characters)
# and tight enough that a hostile one cannot be expensive.
MAX_QUERY_CHARS = 4_000
MAX_ANSWER_CHARS = 20_000
MAX_ID_CHARS = 128
MAX_CAPACITY_MINUTES = 240

# Ceiling on provider calls for one inbound request, enforced by
# llm/budget.py. The deepest current path is 5 calls (1 retrieval decision + 3
# evidence-path scores + 1 question mapping), so 12 leaves room to grow while
# still stopping a runaway loop in single digits rather than at the request
# timeout. Overridable per deployment via ACOS_MAX_LLM_CALLS_PER_REQUEST.
_DEFAULT_MAX_LLM_CALLS = 12


def _max_llm_calls() -> int:
    raw = os.environ.get("ACOS_MAX_LLM_CALLS_PER_REQUEST", "").strip()
    if not raw:
        return _DEFAULT_MAX_LLM_CALLS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ACOS_MAX_LLM_CALLS_PER_REQUEST=%r is not an integer; using default", raw)
        return _DEFAULT_MAX_LLM_CALLS
    return max(1, value)


app = FastAPI(title="AcademicOS", version="0.1.0",
              description="CBSE/NCERT Academic Brain — evidence-grounded retrieval + reasoning")


@app.exception_handler(LLMBudgetExceeded)
def _on_llm_budget_exceeded(_request: Any, exc: LLMBudgetExceeded) -> JSONResponse:
    """A spent budget is 429, not 500.

    The request was well-formed and the caller is authenticated; it simply asked
    for more model work than one request may consume. 429 with `Retry-After` is
    the honest answer, and it is actionable -- a 500 would read as our bug and
    would invite the caller to retry immediately, which is exactly what we do
    not want.
    """
    logger.warning("LLM budget exhausted: %s", exc)
    return JSONResponse(
        status_code=429,
        content={"detail": "This request exceeded its model-call budget. "
                           "Narrow the request or retry with a smaller scope."},
        headers={"Retry-After": "5"},
    )


# Compression. Measured on this repo's Flutter web build, gzip takes the
# assets that dominate first load from 19.4 MB to 5.0 MB (-74%):
#   main.dart.js        5.43 MB -> 1.53 MB  (-72%)
#   questions.json      6.76 MB -> 0.57 MB  (-92%)
#   canvaskit.wasm      7.23 MB -> 2.90 MB  (-60%)
# Nothing compressed it before -- Cloud Run does not gzip responses on your
# behalf -- so every user downloaded all of it raw, on every visit.
# Added last so it is the outermost middleware and also compresses the
# CORS-wrapped responses registered below.
app.add_middleware(GZipMiddleware, minimum_size=1024)

_cfg = get_config()
_cors_origins = getattr(_cfg, "cors_origins", ["*"])
if "*" in _cors_origins:
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        # Any *.web.app is ANY Firebase site, not this school's. In production
        # only the origins CI passes in ACOS_CORS_ORIGINS (the project's own
        # web.app / firebaseapp.com) are answered; the wildcards stay for dev.
        allow_origin_regex=None if is_production() else r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.onrender\.com$|^https://.*\.web\.app$|^https://.*\.firebaseapp\.com$|^https://.*\.github\.io$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(assessment_routes.router)
app.include_router(pillar_routes.router)
app.include_router(mobile_routes.router)
app.include_router(ingest_routes.router)
app.include_router(auth_routes.router)
app.include_router(consent_routes.router)
app.include_router(curriculum_routes.router)
app.include_router(paper_template_routes.router)
# The question-bank API carries its own `/v1/...` paths and its own key auth,
# so it is mounted at the root rather than under `/api/v1` -- its paths are
# part of a published contract that third parties will build against, and
# `docs/question-bank-api.md` fixes them as `GET /v1/questions` and friends.
app.include_router(qbank_routes.router)
app.include_router(grade_by_question.router)


@app.exception_handler(SupabaseUnavailable)
def _on_supabase_unavailable(_request: Any, exc: SupabaseUnavailable) -> JSONResponse:
    """One place so a live Supabase failure is never a bare 500 again. The
    stores that fall back to local SQLite (AssessmentStore, EventStore)
    catch SupabaseUnavailable themselves before it reaches this handler --
    this is for everything else (auth, papers, templates, ...): a real 503
    naming the table and the PostgREST error, logged with a traceback
    server-side. See SupabaseUnavailable's docstring for the production
    incident this closes."""
    logger.error("Supabase call failed: %s", exc, exc_info=exc)
    return JSONResponse(status_code=503,
                        content={"detail": f"storage backend unavailable: {exc}"})


# --- domain exceptions that used to surface as bare 500s -------------------
#
# Four exception classes were raised but never caught anywhere, so every one of
# them reached the client as a 500. A 500 says "we broke"; three of these say
# something the caller can act on, and one says the provider did. Mapping them
# here rather than in each route means a new call site cannot forget.

@app.exception_handler(ScopeError)
def _on_scope_error(_request: Any, exc: ScopeError) -> JSONResponse:
    """A scope outside the vocabulary is the CALLER's mistake: 400.

    Raised by `validate_scopes` when a key is defined with an unknown scope,
    with one permanently refused because it would reach student data, or with
    none at all. The message is safe to return and is the useful part -- it
    names the offending scopes and lists the vocabulary -- so a caller can fix
    the request without reading our source.
    """
    logger.info("scope validation rejected a request: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ReviewStateError)
def _on_review_state_error(_request: Any, exc: ReviewStateError) -> JSONResponse:
    """An illegal state transition is a CONFLICT: 409.

    The request is well-formed and the resource exists; what fails is the
    transition against its current state (already `published`, `draft -> retired`
    is not legal, a no-op that would make the audit trail ambiguous).

    Known imprecision, recorded rather than hidden: this class also covers
    "unknown question", which would ideally be 404. Splitting it needs two
    exception classes, not a cleverer handler, and 409 is correct for the
    majority. `str(exc)` is returned because it names the states involved.
    """
    logger.info("review-state transition rejected: %s", exc)
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(LLMProviderError)
def _on_llm_provider_error(_request: Any, exc: LLMProviderError) -> JSONResponse:
    """503: the model provider is unreachable or not configured.

    Deliberately does NOT echo `str(exc)`. Most of these are "SARVAM_API_KEY is
    not set", but the class also wraps provider HTTP failures, and an operator
    message is not the caller's business. Name the subsystem, log the detail.
    """
    logger.error("LLM provider unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "the language model provider is unavailable"},
    )


@app.exception_handler(LLMEvaluationError)
def _on_llm_evaluation_error(_request: Any, exc: LLMEvaluationError) -> JSONResponse:
    """502: the provider answered, but not usably.

    Distinct from 503 on purpose. 502 means the upstream was reached and
    returned something we cannot use (no JSON object in the response); 503
    means we could not reach it. Retrying is sensible for one and pointless for
    the other, so collapsing them would lose the distinction the caller needs.

    `str(exc)` is NOT returned: these carry raw model output
    (`f"no JSON object in model response: {raw[:200]!r}"`), and echoing model
    output to a client is how prompt content leaks.
    """
    logger.error("LLM evaluation failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "the language model returned an unusable response"},
    )


@app.exception_handler(LLMNotEnabled)
def _on_llm_not_enabled(_request: Any, exc: LLMNotEnabled) -> JSONResponse:
    """501: this deployment has no AI provider key. Not 503 -- retrying will
    never help until an administrator configures one -- and a distinct `code`
    so a client can show "not enabled for this school" rather than an error."""
    logger.info("LLM feature requested with no provider key configured")
    return JSONResponse(status_code=501,
                        content={"detail": str(exc), "code": "llm_not_enabled"})


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=MAX_QUERY_CHARS)
    limit: int = Field(default=10, ge=1, le=50)


class DoubtRequest(BaseModel):
    doubt: str = Field(min_length=2, max_length=MAX_QUERY_CHARS)


class ScoreRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    student_answer: str = Field(default="", max_length=MAX_ANSWER_CHARS)
    marks_available: float = Field(default=0.0, ge=0.0, le=1000.0)


class AnalyzeRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)


class NodeQuery(BaseModel):
    node_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    max_depth: int = Field(default=2, ge=1, le=6)


class PathQuery(BaseModel):
    source: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    target: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    max_depth: int = Field(default=4, ge=1, le=8)


class GraphResult(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class ReviseRequest(BaseModel):
    learner: str = Field(default="default", max_length=MAX_ID_CHARS)
    model: str = Field(default="fsrs", pattern="^(fsrs|exponential)$")


class StudyPlanRequest(BaseModel):
    target: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    learner: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    model: str = Field(default="fsrs", pattern="^(fsrs|exponential)$")
    # Upper bound added with the lower one: `capacity_minutes` feeds a planning
    # loop, so an unbounded value is a request that asks the server to compute
    # for an arbitrary length of time.
    capacity_minutes: Optional[int] = Field(default=None, ge=1, le=MAX_CAPACITY_MINUTES)
    lo: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mastery_gate: float = Field(default=0.5, ge=0.0, le=1.0)


class QuestionSolveRequest(BaseModel):
    question: str = Field(min_length=2, max_length=MAX_QUERY_CHARS)


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
    consent_routes.init(cfg.data_root)
    curriculum_routes.init(cfg)
    paper_template_routes.init(cfg)
    mobile_scan.configure_workdir(cfg.data_root / "scan-sessions")
    # The public question-bank API. Its own key store and its own scope
    # vocabulary, deliberately separate from the user auth above: an API key is
    # a machine credential for content, and must never carry a user's identity.
    # See assessment/api_keys.py for why student data is unexpressible here.
    qbank_routes.init(cfg.data_root / "assessment" / "api_keys.sqlite",
                      cfg.data_root / "syllabus" / "questions.json")

    # Grade-by-question borrows pillar_routes' stores and guards rather than
    # opening its own, so a bulk award goes through exactly the same
    # authorization and clamping the per-student path uses. `questions_by_id`
    # is best-effort: the marking scheme is an aid to grading, and a teacher
    # must still be able to mark a question the bank has never seen.
    grade_by_question.init(
        graded=pillar_routes._graded,
        require=pillar_routes._require,
        require_school_owns_assessment=pillar_routes._require_school_owns_assessment,
        questions_by_id=lambda qid: (qbank_routes._bank.get(qid)
                                     if qbank_routes._bank is not None else None),
        audit=pillar_routes._audit,
        consent=pillar_routes._consents,
    )
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
    cfg = Config.load()
    # First, before any store opens: a misconfigured production service must
    # fail to start (the Cloud Run revision never goes ready and the deploy
    # goes red) rather than serve a school on placeholder secrets.
    enforce_production_config(cfg)
    from ..assessment.pdf import register_unicode_font
    logger.info("PDF body font: %s", register_unicode_font())
    init_runtime(cfg)


@app.on_event("shutdown")
def _shutdown() -> None:
    """Cloud Run sends SIGTERM and allows 10 s before SIGKILL; uvicorn runs
    this inside that window. CurriculumStore debounces its snapshot upload
    (storage/snapshot_sync.py), so the last edits before a deploy can still
    be waiting -- upload them now or they die with the container."""
    from ..storage.snapshot_sync import flush_all_snapshots
    flush_all_snapshots()


@app.get("/health")
def health() -> dict:
    """Liveness plus build identity. The 2026-09-21 audit found the live
    service a build 4+ days behind its source and unable to say so; the deploy
    job now fails unless `commit` is the commit it just deployed."""
    return {"status": "ok", **build_identity()}


@app.get("/health/llm")
def health_llm() -> dict:
    """Recent provider-call behaviour for this process.

    Unauthenticated, like `/health`, and safe to be: it reports counts, timings,
    model names and failure rates, never prompt or completion content. That is
    the component level of agent observability -- enough to answer "is the
    provider degrading" or "did the retry rate move after that deploy", which is
    precisely what a discarded log line cannot tell you.

    The window is the in-process ring buffer (llm/telemetry.py), so on a
    multi-instance deployment this describes one instance. That is a stated
    limitation, not an oversight: a "global" view needs a metrics backend, and
    this exists so the question is answerable at all before one is added.
    """
    return TELEMETRY.summary()


# Durability-critical stores (see AGENTS.md's storage inventory and
# curriculum/store.py's module docstring): AssessmentStore, PaperStore,
# PracticeStore, GradedStore, ScanSessionStore, TemplateStore, UserStore
# and AuditLog use the per-row SupabaseTable fallback pattern; EventStore
# joined them in this pass; CurriculumStore uses whole-file snapshot/
# restore via SupabaseStorage instead (too relational to wrap per-table).
_DURABLE_STORES = [
    "AssessmentStore", "PaperStore", "PracticeStore", "GradedStore",
    "ScanSessionStore", "TemplateStore", "UserStore", "AuditLog",
    "EventStore", "KnowledgeStore", "ConsentStore",
    "CurriculumStore (snapshot/restore)",
]

# One SupabaseTable per _DURABLE_STORES row above (keep the two lists in
# sync): the real table each store reads/writes, so the probe below covers
# every durability-critical table, not just one of them.
#
# This is also THE list of tables the service requires, on either backend:
# tests/test_health_storage_schema.py pins it to exactly the tables
# deploy/gcp/schema.sql creates (schema.sql itself is not in the image, so it
# cannot be read at runtime). parental_consents joined 2026-09-22: ConsentStore
# has used it since the DPDP pass, but the probe never looked, so a Cloud SQL
# instance missing it would have reported healthy.
_DURABLE_TABLES = [
    "assessments", "papers", "practice_sets", "graded_evaluations",
    "scan_sessions", "school_templates", "users", "sessions", "invites",
    "audit_log", "learner_events", "learner_models", "parental_consents",
]

# How long one /health/storage answer stays good. The 2026-09-15 stress
# test showed why this exists: the uncached probe ran 10 sequential
# Supabase selects per call (~0.5-1s holding a threadpool thread), managed
# only ~4-6 rps on the free tier, and any burst of callers starved real
# traffic of threads. 30s of staleness on a manual diagnostic endpoint is
# the right trade -- but note it when verifying a fix: after provisioning
# a missing table, allow one TTL before trusting a still-red answer.
_STORAGE_PROBE_TTL_S = 30.0
_probe_cache: dict[str, Any] = {"at": 0.0, "body": None}
# Single-flight: exactly one refresh runs at a time; concurrent callers
# get the last-known body instead of queueing behind up to 10 sequential
# upstream timeouts each.
_probe_lock = threading.Lock()


def _table_missing(exc: SupabaseUnavailable) -> bool:
    """True only when the error says the table itself does not exist -- not
    for an outage, a timeout or a permissions problem, which say nothing
    about the schema. Cloud SQL: psycopg's UndefinedTable, SQLSTATE 42P01,
    which postgres_kv chains as the PostgresUnavailable's cause. Supabase:
    PostgREST's PGRST205 ("Could not find the table")."""
    if getattr(exc.__cause__, "sqlstate", None) == "42P01":
        return True
    return getattr(exc, "status", None) == 404 and "PGRST205" in str(exc)


def _probe_table(table: str) -> tuple[str, str, Optional[bool]]:
    """(table, "ok" | "error: ...", exists). `exists` is None when the probe
    could not tell -- the backend did not answer."""
    try:
        durable_table(table).select(limit=1)
        return table, "ok", True
    except SupabaseUnavailable as exc:
        return table, f"error: {exc}", (False if _table_missing(exc) else None)


def _probe_all_tables() -> tuple[dict[str, str], dict[str, Optional[bool]]]:
    tables: dict[str, str] = {}
    exists: dict[str, Optional[bool]] = {}
    with ThreadPoolExecutor(max_workers=len(_DURABLE_TABLES)) as pool:
        for table, status, present in pool.map(_probe_table, _DURABLE_TABLES):
            tables[table] = status
            exists[table] = present
    return tables, exists


# Where /health/storage sends an operator for each backend when a required
# table is missing. On Cloud SQL nothing applies schema.sql automatically --
# not the workflow, not bootstrap.sh -- so a table added to schema.sql (as
# learner_models was) is missing on every instance provisioned before it.
_SCHEMA_FIX = {
    "cloudsql": "apply deploy/gcp/schema.sql with scripts/apply_cloudsql_schema.sh "
                "(docs/gcp-deployment.md section 6)",
    "supabase": "run docs/supabase-setup.sql in the Supabase SQL editor",
}


def _overall_status(backend: str, tables: dict[str, str],
                    exists: dict[str, Optional[bool]]) -> tuple[str, str, list[str]]:
    """(status, status_detail, missing_tables). "ok" only when every required
    table answered. A missing table outranks an unreachable one: it is the
    one an operator can fix in one command, and it does not heal itself."""
    missing = [t for t in _DURABLE_TABLES if exists.get(t) is False]
    if missing:
        return ("schema_missing",
                f"required tables missing: {', '.join(missing)}; "
                f"{_SCHEMA_FIX.get(backend, 'apply the schema')}", missing)
    if all(status == "ok" for status in tables.values()):
        return "ok", "every required table answered", missing
    broken = [t for t, status in tables.items() if status != "ok"]
    return "unreachable", f"tables did not answer: {', '.join(broken)}", missing


def _durable_backend(cfg) -> str:
    """Which durability backend the stores are actually bound to right now.

    Cloud SQL wins when configured (see assessment/postgres_kv.py's
    durable_table), then Supabase, else none -- the same precedence the
    stores themselves resolve.
    """
    from ..assessment.postgres_kv import postgres_configured
    if postgres_configured():
        return "cloudsql"
    return "supabase" if cfg.supabase_enabled else "none"


def _blob_backend() -> str:
    """Where scan media and the curriculum snapshot actually go right now
    (gcs | supabase | local) -- see storage/blobs.py. Reported separately from
    durability_backend because the two moved to GCP independently: a service
    can be on Cloud SQL and still be writing blobs to container disk."""
    from ..storage.blobs import blob_backend
    return blob_backend()


def _refresh_storage_probe(cfg) -> dict:
    tables, exists = _probe_all_tables()
    reachable = all(status == "ok" for status in tables.values())
    backend = _durable_backend(cfg)
    status, detail, missing = _overall_status(backend, tables, exists)
    body = {
        "status": status,
        "status_detail": detail,
        "missing_tables": missing,
        "table_exists": exists,
        "durability_backend": backend,
        "blob_backend": _blob_backend(),
        "durability_reachable": reachable,
        "supabase_configured": cfg.supabase_enabled,
        # Legacy key: the Flutter client and the live cutover runbook both
        # read it. Means "the Supabase backend is reachable"; null when
        # Supabase is not the active backend (e.g. after the Cloud SQL move),
        # so it can never be mistaken for a healthy Supabase.
        "supabase_reachable": reachable if backend == "supabase" else None,
        "tables": tables,
        "durable_stores": _DURABLE_STORES,
    }
    _probe_cache.update(at=time.monotonic(), body=body)
    return body


@app.get("/health/storage")
def health_storage() -> dict:
    """Answers "is this actually live" without guessing from the hosting
    dashboard or grepping env vars on the host -- which durability backend
    the stores are bound to, and (only if one is configured) whether every
    durability-critical table is actually reachable right now, one limit-1
    select each. `blob_backend` says where scan media and the curriculum
    snapshot go (gcs | supabase | local).

    `blob_status` (storage/blobs.py) adds this instance's snapshot conflicts,
    the conflict copies that hold edits awaiting a merge, and failed blob
    uploads. It is live, never cached: in-process counters, and a conflict
    must show the moment it happens, not one probe TTL later. The table probe
    is cached and single-flighted (see _storage_probe)."""
    from ..storage.blobs import blob_status
    return {**_storage_probe(), "blob_status": blob_status()}


def _storage_probe() -> dict:
    """Answers "is this actually live" without guessing from the hosting
    dashboard or grepping env vars on the host -- which durability backend
    the stores are bound to, and (only if one is configured) whether every
    durability-critical table is actually reachable right now, one limit-1
    select each.

    `durability_reachable` is true only when ALL of them answer. Probing a
    single table (this route's original behavior) is exactly how a live
    deployment spent its whole life reporting "reachable" while the
    `users`/`sessions` tables were missing and every auth call 500'd --
    the per-table detail below is the point: it names the broken one.

    `status` is "ok" only when every required table answered;
    "schema_missing" names the tables that do not exist (`missing_tables`,
    `table_exists`) and the one command that creates them -- a Cloud SQL
    instance provisioned before a table was added to schema.sql is exactly
    that, and nothing applies schema.sql on deploy. "unreachable" is an
    outage, which says nothing about the schema. The HTTP status stays 200:
    this is a diagnostic, and the body is the verdict.

    Answers are cached for _STORAGE_PROBE_TTL_S seconds and refreshes are
    single-flighted (see above): this endpoint is 10x-amplified upstream
    traffic and must never be able to saturate the worker pool itself."""
    cfg = Config.load()
    backend = _durable_backend(cfg)
    if backend == "none":
        return {
            "status": "unconfigured",
            "status_detail": "no durability backend configured: every store is "
                             "on local disk",
            "missing_tables": [],
            "table_exists": {},
            "durability_backend": "none",
            "blob_backend": _blob_backend(),
            "durability_reachable": None,
            "supabase_configured": False,
            "supabase_reachable": None,
            "tables": {},
            "durable_stores": _DURABLE_STORES,
        }
    cached = _probe_cache["body"]
    if cached is not None and time.monotonic() - _probe_cache["at"] < _STORAGE_PROBE_TTL_S:
        return cached
    if _probe_lock.acquire(blocking=False):
        try:
            return _refresh_storage_probe(cfg)
        finally:
            _probe_lock.release()
    # A refresh is already in flight: serve last-known state rather than
    # queue behind it. Only when no probe has ever succeeded do we wait
    # for the in-flight one instead.
    cached = _probe_cache["body"]
    if cached is not None:
        return cached
    with _probe_lock:
        return _refresh_storage_probe(cfg)


@app.get("/v1/registry/stats", dependencies=[Depends(get_current_user)])
def registry_stats() -> dict:
    if not _registry:
        raise HTTPException(503, "runtime not initialized")
    return {"count": _registry.count()}


@app.post("/v1/search", dependencies=[Depends(get_current_user)])
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


@app.post("/v1/agent/doubt", dependencies=[Depends(get_current_user)])
def agent_doubt(req: DoubtRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    # The budget is entered *inside* the body, not in a dependency, on purpose.
    # A budget must be visible to the code that calls the model, and it is
    # carried in a context variable; FastAPI runs dependencies and sync
    # endpoints on threadpool workers with separately-copied contexts, so a
    # value set in a dependency would not reliably reach this frame. Entering it
    # here guarantees the same thread that makes the call is the one that set
    # the ceiling.
    with llm_budget(_max_llm_calls(), label="agent.doubt"):
        return DoubtSolver(_retriever, _graph, critic=_critic).solve(req.doubt)


@app.post("/v1/agent/score", dependencies=[Depends(get_current_user)])
def agent_score(req: ScoreRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    with llm_budget(_max_llm_calls(), label="agent.score"):
        return ExaminerScorer(_retriever, _graph).score(req.question, req.student_answer, req.marks_available)


@app.post("/v1/agent/analyze", dependencies=[Depends(get_current_user)])
def agent_analyze(req: AnalyzeRequest) -> dict:
    if not _retriever:
        raise HTTPException(503, "runtime not initialized")
    with llm_budget(_max_llm_calls(), label="agent.analyze"):
        return PaperAnalyst(_retriever, _graph).analyze(req.paper_id)


@app.post("/v1/graph/neighbors", dependencies=[Depends(get_current_user)])
def graph_neighbors(req: NodeQuery) -> GraphResult:
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    return GraphResult(nodes=[{"id": x["node"].id, "label": x["node"].label,
                               "type": x["node"].type.value, "depth": x["depth"],
                               "edge": x["edge"].type.value} for x in _graph.neighbors(req.node_id, req.max_depth)])


@app.post("/v1/graph/paths", dependencies=[Depends(get_current_user)])
def graph_paths(req: PathQuery) -> dict:
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    return {"paths": [[e.type.value for e in p] for p in _graph.paths(req.source, req.target, req.max_depth)]}


def _require_learner_access(learner_id: str, current: User) -> None:
    """The learner in a /v1 request body is a student id, and a caller may
    only name one they are entitled to: a student of their own school, and
    for a student caller, themselves. Until 2026-09-21 these routes checked
    only that the caller was logged in, and a school_B teacher received a
    school_A student's due concepts and retention from /v1/revise. A learner
    id that is no user at all is a 404, which is what these routes already
    said for a learner with no events."""
    require_school_owns_student(auth_routes._require(), learner_id, current)


@app.post("/v1/revise")
def revise(req: ReviseRequest, current: User = Depends(get_current_user)) -> dict:
    """P5.2 — what to revise today (FSRS-backed, from the learner's event store)."""
    if not _events:
        raise HTTPException(503, "runtime not initialized")
    _require_learner_access(req.learner, current)
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
def study_plan(req: StudyPlanRequest, current: User = Depends(get_current_user)) -> dict:
    """P5.2 — today's study session: prereqs in order, FSRS-due items first,
    bounded by a daily capacity window."""
    if not _graph:
        raise HTTPException(503, "runtime not initialized")
    # Without a learner the plan is the concept graph alone, which any
    # caller may see; with one, it reads that student's event log.
    if req.learner:
        _require_learner_access(req.learner, current)
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


@app.post("/v1/question/solve", dependencies=[Depends(get_current_user)])
def question_solve(req: QuestionSolveRequest) -> dict:
    """P5.2 — question-solve with evidence: map a question to graph concepts
    (LLM-assisted, verifier-checked) and persist the mapping for tracing."""
    if not _graph or not _qmap_store:
        raise HTTPException(503, "runtime not initialized")
    from ..qmap import QuestionMapper

    with llm_budget(_max_llm_calls(), label="question.solve"):
        qmap = QuestionMapper(_graph).map(req.question)
    _qmap_store.append(qmap.to_dict())
    return qmap.to_dict()


# --- Static web frontend mounting (SPA) ---
from starlette.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class SPAStaticFiles(StaticFiles):
    """Serves compiled Flutter web static assets with client-side SPA fallback.
    Any non-file GET path that does not start with /api/ or /v1/ falls back to index.html."""

    # canvaskit/ is pinned to the Flutter engine version, so a filename there
    # never means two different things -- safe to cache forever. Everything
    # else keeps Starlette's ETag but must revalidate (`no-cache` = "check
    # before using", not "don't store"): Flutter web asset filenames are NOT
    # content-hashed, so `main.dart.js` and `assets/**` keep their names
    # across releases. Long-caching those would serve a stale app forever.
    # Measured on this repo's build: canvaskit.wasm is 7.2 MB, main.dart.js
    # 5.4 MB, assets/corpus/questions.json 6.8 MB -- re-downlading those on
    # every visit is exactly the "loading time" that matters.
    _IMMUTABLE_PREFIX = "canvaskit/"

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as ex:
            if ex.status_code == 404:
                norm = path.replace("\\", "/").strip("/")
                if norm.startswith("api/") or norm.startswith("v1/") or norm == "health":
                    raise
                response = await super().get_response("index.html", scope)
                response.headers.setdefault("Cache-Control", "no-cache")
                return response
            raise

        norm = path.replace("\\", "/").lstrip("/")
        if norm.startswith(self._IMMUTABLE_PREFIX):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers.setdefault("Cache-Control", "no-cache")
        return response


def _mount_web_frontend(fastapi_app: FastAPI) -> None:
    candidates = [
        Path.cwd() / "academicos-data" / "web",
        Path(__file__).resolve().parents[3] / "academicos-data" / "web",
        Path("/app/academicos-data/web"),
        Path(__file__).resolve().parents[3] / "frontend" / "build" / "web",
    ]
    for p in candidates:
        if p.is_dir() and (p / "index.html").is_file():
            fastapi_app.mount("/", SPAStaticFiles(directory=str(p), html=True), name="web")
            logger.info("Mounted AcademicOS web frontend from %s", p)
            return


_mount_web_frontend(app)

