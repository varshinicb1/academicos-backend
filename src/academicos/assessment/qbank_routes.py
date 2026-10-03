"""The public question-bank API: a resource surface over the enriched bank.

`docs/question-bank-api.md` section 4 asks for this specifically, and gives the
reasons the contract is shaped the way it is:

    GET  /v1/questions                 list + filter (the workhorse)
    GET  /v1/questions/{id}            one, with full provenance and scheme
    GET  /v1/questions/{id}/scheme     just the marking scheme
    GET  /v1/subtopics/{id}/questions  curriculum-driven listing
    GET  /v1/facets                    filter vocabulary and counts
    GET  /v1/coverage                  answer-key coverage per subject x class
    POST /v1/papers                    blueprint-driven paper generation

Design decisions taken from that document, all of them deliberate:

  * **Cursor pagination, not offset.** "The corpus is written once and read
    forever; offset paging over a large filtered set degrades and duplicates
    rows under concurrent edits." Keyset on `id`, which is stable and unique.
  * **Query parameters, not a query DSL.** "A DSL is a second language to
    document and to get wrong."
  * **Facets on list responses.** One round trip instead of list-then-fetch.
  * **RFC 9457 problem+json errors.** "Standard, machine-readable, survives an
    SDK generation."
  * **Rate-limit headers on every response.** "Consumers cannot behave well
    without them."

Authentication is a scoped API key (`api_keys.py`), and the route depends on the
scope it needs, so a `facets:read` key cannot read a marking scheme it was not
granted. Student data is unreachable here by construction: no scope in the
vocabulary grants it. A key may also be limited to some classes and subjects;
every route then serves only records inside them, and a request naming a class
or subject outside them is a 403 (`_limits`, `_refuse_outside`).

A query parameter a route does not declare is a 422 naming it (`_require`),
not ignored: `?subjct=Science` used to answer with the whole bank.

Honesty about what is not here: `POST /v1/papers` is **not** implemented in this
pass -- paper generation already exists at `POST /api/v1/papers/generate` behind
user auth, and re-exposing it needs a decision about whether an API key may
consume the blueprint engine. It is listed in the doc and omitted here rather
than half-built.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .api_keys import ApiKey, ApiKeyStore, QuotaExceeded
# The engine lives in `qbank_engine.py`, which imports no web framework.
# `QuestionBank`, the answer-key rule and the cursor were defined HERE, and
# `mcp/qbank_server.py` and `assessment/pool.py` both had to reach into this
# module -- and so into FastAPI -- to get them, which is why `pip install -e
# ".[mcp]"` could not run the MCP server and a core install could not build a
# paper pool. Re-exported under the old names so existing importers
# (`syllabus/bank_health.py`, the tests) keep working unchanged.
from .qbank_engine import (  # noqa: F401  (re-exported for importers)
    DEFAULT_LIMIT,
    KEY_PROVENANCE,
    MAX_LIMIT,
    TRUST_LEVELS,
    InvalidCursor,
    QuestionBank,
    book_of,
    decode_cursor,
    encode_cursor,
    key_provenance_of,
    key_tier,
    trust_of,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["question-bank"])

_store: Optional[ApiKeyStore] = None
_bank: Optional[QuestionBank] = None


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #

def init(db_path: Path | str, bank_path: Path | str) -> None:
    """Build the key store and load the corpus. Called once at startup.

    A missing corpus is a warning, not a failure. The bank is a data file, not a
    code dependency, and raising here would take the entire API down over a
    missing asset -- every other route would 500 with it. With `_bank` left
    None the question-bank surface answers 503 (honest: it genuinely is not
    initialised) and everything else boots normally.
    """
    global _store
    _store = ApiKeyStore(db_path, durable=True)
    # API-6: partners' webhooks, beside the keys they belong to.
    from . import webhooks
    webhooks.init(Path(db_path).with_name("webhooks.sqlite"), _store)
    load_bank(bank_path)


def load_bank(bank_path: Path | str) -> Optional[QuestionBank]:
    """(Re)load the served bank from its file. The one place `_bank` changes,
    so the one place partners' webhooks hear of new questions (API-6): the
    answer-keyed ids are compared with the last load's and each chapter of
    new ones becomes a `questions.added` event (webhooks.py)."""
    global _bank
    path = Path(bank_path)
    if not path.exists():
        _bank = None
        log.warning("question-bank corpus missing at %s; the /v1 question "
                    "surface will answer 503", path)
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _bank = None
        log.warning("question-bank corpus at %s is unreadable (%s); the /v1 "
                    "question surface will answer 503", path, exc)
        return None

    _bank = QuestionBank(payload.get("questions") or [])
    log.info("question-bank API ready: %d questions", len(_bank))
    from . import webhooks
    webhooks.bank_loaded_safely(_bank)
    return _bank


class AuthFailure(Exception):
    """A presented key was refused, with the answer both surfaces must give.

    An exception rather than an `HTTPException` because the MCP server's
    network transports authenticate against this same store and scopes
    (rule Q5, and the audit's 7.2: sse and streamable-http had no
    authentication at all while the HTTP surface of the same product was
    key-gated). They are not FastAPI, so the refusal has to be expressible
    without it; `_require` re-raises it as an `HTTPException` unchanged.
    """

    def __init__(self, status: int, detail: str,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.headers = headers or {}


def presented_key(authorization: str | None, x_api_key: str | None) -> str:
    """The key a caller presented, by either accepted header."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    return ""


def authorize(store: ApiKeyStore, presented: str, scope: str) -> ApiKey:
    """Authenticate a key, require one scope, and spend one unit of quota.

    The whole gate, in one function, so the HTTP routes and the MCP network
    transports cannot come to different conclusions about the same key.
    """
    key = store.authenticate(presented)
    if key is None:
        raise AuthFailure(401, "a valid API key is required",
                          {"WWW-Authenticate": "Bearer"})
    if not key.may(scope):
        # 403, not 404: the key is valid, the permission is not held.
        raise AuthFailure(403, f"this key does not hold the {scope!r} scope")
    try:
        store.check_quota(key)
    except QuotaExceeded as exc:
        raise AuthFailure(429, str(exc),
                          {"Retry-After": str(exc.retry_after)}) from exc
    return key


# The two headers `presented_key` accepts, declared as security schemes so the
# published OpenAPI says how to authenticate. The document carried only the
# user-session `HTTPBearer`, and the key headers appeared as two optional
# plain parameters, so a client generated from it sent no key and met a 401
# (audit API-6). `auto_error=False` on both: either header will do, and the
# refusal stays `authorize`'s own 401, the one the MCP transports give too.
KEY_HEADER = APIKeyHeader(
    name="X-API-Key", scheme_name="QuestionBankApiKey", auto_error=False,
    description="A question-bank API key (`acos_qb_...`), minted by a "
                "principal on the web console's API keys page.")
KEY_BEARER = HTTPBearer(
    scheme_name="QuestionBankApiKeyBearer", bearerFormat="acos_qb_...",
    auto_error=False,
    description="The same API key, sent as `Authorization: Bearer acos_qb_...`. "
                "A user's session token is not a key and is refused here.")


def _declared_query(route: Any) -> frozenset[str]:
    """Every query parameter a route declares, its own and its dependencies'.

    Read from the route FastAPI matched rather than kept as a second list per
    route, so a filter added to a signature is accepted the moment it exists.
    """
    names: set[str] = set()
    stack = [route.dependant]
    while stack:
        dependant = stack.pop()
        names.update(p.alias for p in dependant.query_params)
        stack.extend(dependant.dependencies)
    return frozenset(names)


def _refuse_unknown_query(request: Request) -> None:
    """A query parameter the route does not take is a 422 naming it.

    FastAPI drops an undeclared parameter without a word, so `?subjct=Science`
    answered with all 5,319 records (audit D7) and `?book=`, `?topic_id=`,
    `?competency=`, `?source=` and `?trust=` with the whole class (N-67-12):
    a full page that reads as the answer to the question asked, given to a
    different one. The 422 has the shape of every other validation error
    (`detail[]` with `loc`, `msg`, `type`), so a client parses one shape.
    """
    known = _declared_query(request.scope["route"])
    unknown = [name for name in request.query_params.keys() if name not in known]
    if unknown:
        accepted = ", ".join(sorted(known)) or "no query parameters"
        raise HTTPException(422, detail=[{
            "type": "extra_forbidden", "loc": ["query", name],
            "msg": f"unknown query parameter {name!r}; this route takes {accepted}",
            "input": request.query_params.get(name),
        } for name in unknown])


def _require(scope: str):
    """Dependency factory: authenticate a key, require one scope, and refuse a
    query parameter the route does not know.

    Returning a closure means the required scope is visible in the route
    signature rather than buried in the body, so a reviewer can see the whole
    permission surface by reading the decorators.

    The order is the HTTP surface's whole contract: 503 (not loaded), then
    401, 403 and 429 from `authorize`, then the unknown-parameter 422. A caller
    with no key learns nothing about the parameters, and a 422 costs one unit
    of quota exactly as a malformed value always has.
    """
    def dependency(
        request: Request,
        # Declared for the OpenAPI document; the key itself is read below by
        # `presented_key`, the one parser the MCP transports share, so the two
        # surfaces cannot read the same header two ways.
        _bearer: HTTPAuthorizationCredentials | None = Security(KEY_BEARER),
        _header: str | None = Security(KEY_HEADER),
    ) -> ApiKey:
        if _store is None or _bank is None:
            raise HTTPException(503, detail="question-bank API not initialised")
        try:
            key = authorize(_store, presented_key(request.headers.get("authorization"),
                                                  request.headers.get("x-api-key")),
                            scope)
        except AuthFailure as exc:
            raise HTTPException(exc.status, detail=exc.detail,
                                headers=exc.headers or None) from exc
        _refuse_unknown_query(request)
        return key
    return dependency


# --------------------------------------------------------------------------- #
# a key's classes and subjects (API-3, audit N-67-10)
# --------------------------------------------------------------------------- #

def _limits(key: ApiKey) -> dict[str, Any]:
    """The key's limits as engine filters: `None` where there is no limit."""
    return {"grades": key.grades or None, "subjects": key.subject_keys or None}


def _refuse_outside(key: ApiKey, *, grade: int | None = None,
                    subject: str | None = None) -> None:
    """403 when a request NAMES a class or subject outside the key's limits.

    An empty page would also be true, and is the wrong answer: it reads as "the
    bank has no class 9 questions" when the truth is "this key may not read
    class 9", and only the second tells the partner what to fix.
    """
    if grade is not None and not key.admits_grade(grade):
        raise HTTPException(403, detail=(
            f"this key is limited to classes {sorted(key.grades)}; "
            f"class {grade} is outside it"))
    if subject is not None and not key.admits_subject(subject):
        raise HTTPException(403, detail=(
            f"this key is limited to subjects {sorted(key.subjects)}; "
            f"{subject!r} is outside it"))


def _refuse_record_outside(key: ApiKey, rec: dict[str, Any]) -> None:
    """The same 403 for one record asked for by id. Strict where
    `_refuse_outside` is lenient: a record with no class is not inside a key
    limited to class 10."""
    if not (key.admits_grade(rec.get("grade")) and key.admits_subject(rec.get("subject"))):
        raise HTTPException(403, detail=(
            f"question {rec.get('id')!r} is {rec.get('subject')} class "
            f"{rec.get('grade')}, outside this key's limits (classes "
            f"{sorted(key.grades) or 'any'}, subjects {sorted(key.subjects) or 'any'})"))


# --------------------------------------------------------------------------- #
# the documented errors (audit API-6)
# --------------------------------------------------------------------------- #

class V1Error(BaseModel):
    """The body of a 401, 403, 404, 429 or 503 from `/v1`: one sentence."""
    detail: str = Field(examples=["this key does not hold the 'questions:read' scope"])


class V1Problem(BaseModel):
    """RFC 9457's fields, carried under `detail` by a 400."""
    type: str = Field(examples=["about:blank"])
    title: str = Field(examples=["invalid cursor"])
    status: int = Field(examples=[400])
    detail: str = Field(examples=["the cursor is not decodable; omit it to start again"])


class V1BadRequest(BaseModel):
    """A 400: a cursor that does not decode, a `key_provenance` or `trust`
    outside its vocabulary, or two filters no record can satisfy together."""
    detail: V1Problem


class V1InvalidInput(BaseModel):
    loc: list[str | int] = Field(examples=[["query", "subjct"]])
    msg: str = Field(examples=["unknown query parameter 'subjct'; this route takes ..."])
    type: str = Field(examples=["extra_forbidden"])
    input: Any = None


class V1ValidationError(BaseModel):
    """A 422: a query parameter the route does not take (`type`
    `extra_forbidden`), a value of the wrong type or out of range, or a body
    that does not validate. The shape of every FastAPI validation error."""
    detail: list[V1InvalidInput]


def _errors(*extra: int) -> dict[int | str, dict[str, Any]]:
    """The responses every `/v1` route can give, plus the route's own.

    Every one of these was real and none was in the OpenAPI document, which
    listed only 200 and 422 for the seven routes (audit API-6).
    """
    out: dict[int | str, dict[str, Any]] = {
        401: {"model": V1Error, "description": (
            "No key, an unknown key, or a revoked one. Send the key as "
            "`X-API-Key: acos_qb_...` or `Authorization: Bearer acos_qb_...`."),
            "headers": {"WWW-Authenticate": {"schema": {"type": "string"},
                                             "description": "`Bearer`"}}},
        403: {"model": V1Error, "description": (
            "The key is valid but does not hold this route's scope, or the "
            "request (or the record asked for) is outside the classes or "
            "subjects the key is limited to.")},
        422: {"model": V1ValidationError, "description": (
            "A query parameter this route does not take, or a parameter or "
            "body that does not validate. `loc` names it.")},
        429: {"model": V1Error, "description": (
            "The key has spent its requests for this minute. Wait `Retry-After` "
            "seconds: the time left in the current one-minute window."),
            "headers": {"Retry-After": {"schema": {"type": "integer"},
                                        "description": "seconds until the window reopens"}}},
        503: {"model": V1Error, "description": (
            "The question bank is not loaded on this server.")},
    }
    described = {
        400: {"model": V1BadRequest, "description": (
            "A cursor that does not decode, a `key_provenance` or `trust` "
            "outside its vocabulary, or two filters no record can satisfy "
            "together. RFC 9457's fields, under `detail`.")},
        404: {"model": V1Error, "description": "No question with that id."},
    }
    for status in extra:
        out[status] = described[status]
    return out


def _rate_headers(key: ApiKey) -> dict[str, str]:
    return {
        "RateLimit-Limit": str(key.quota_per_minute),
        "RateLimit-Policy": f"{key.quota_per_minute};w=60",
        "X-RateLimit-Limit": str(key.quota_per_minute),
    }


def _slim(rec: dict[str, Any]) -> dict[str, Any]:
    """The list projection.

    The full scheme is deliberately excluded: a list of 25 questions each
    carrying four value points is several times the payload for information
    nobody reads in a list. `/{id}/scheme` exists for the caller who wants it.
    """
    provenance = key_provenance_of(rec)
    return {
        "id": rec.get("id"),
        "subject": rec.get("subject"),
        "grade": rec.get("grade"),
        "marks": rec.get("marks"),
        "type": rec.get("type"),
        "difficulty": rec.get("difficulty"),
        "bloomLevel": rec.get("bloomLevel"),
        "stem": rec.get("stem"),
        "chapterIds": rec.get("chapterIds") or [],
        "tags": rec.get("tags") or [],
        "language": rec.get("language"),
        "reviewState": rec.get("reviewState"),
        "version": rec.get("version"),
        # The label AND the content, the same rule `has_answer_key` applies:
        # the label alone is what let 259 empty CBE schemes read as CBSE's own,
        # and `/v1/coverage` counts only keyed records, so a consumer adding up
        # this flag would disagree with it (review of 2026-09-23).
        "hasOfficialScheme": (QuestionBank.has_answer_key(rec)
                              and provenance == "cbse_marking_scheme"),
        # `hasOfficialScheme` alone stopped being the answer to "can I mark
        # this?" when the Exemplar records arrived: 4,192 of the 5,427 served
        # records carry NCERT's own answer, which is a published key and is not
        # a CBSE marking scheme. Both are reported, plus the label itself, so a
        # consumer can tell which kind of key it is holding.
        "hasAnswerKey": QuestionBank.has_answer_key(rec),
        "keyProvenance": provenance,
        # "published" (CBSE's or NCERT's own) or "checked" (grounded in the
        # textbook, two agreeing solves, or approved by a teacher): which kind
        # of key this is, in one word (qbank_engine.CHECKED_PROVENANCE). What
        # `?trust=` selects on, by the same function.
        "keyTier": trust_of(rec),
        # What `?source=` and `?book=` select on, so a listed question shows
        # the value that would find it again.
        "source": rec.get("source"),
        "book": book_of(rec),
    }


def _q1_filter(has_scheme: bool | None, include_unkeyed: bool) -> bool | None:
    """Rule Q1 as a default on the query, not as a deletion from the bank.

    PRD Q1: *without an answer key, no question.* The audit found
    `/v1/questions` serving all 3,286 records by default, 345 of which had no
    answer content at all -- a consumer paging the bank was handed questions
    nobody can mark, which is the exact failure Q1 names.

    Two ways to see the rest, both explicit, mirroring the MCP's
    `answer_key_state`:

      * `has_scheme=false` -- ONLY the unanswerable set, for auditing what the
        relink has not reached.
      * `include_unkeyed=true` -- the whole bank, for a client that was written
        against the old default and needs the wider set.
    """
    if has_scheme is not None:
        return has_scheme
    return None if include_unkeyed else True


KEY_PROVENANCE_PARAM = Query(
    default=None,
    description="narrow to ONE kind of key: 'cbse_marking_scheme' for the "
                "board's own schemes, 'ncert_exemplar_answer' for NCERT's "
                "Exemplar answers, 'ncert_textbook_answer' for a textbook's "
                "printed answers; or a checked key: 'textbook_grounded', "
                "'two_model_solved', 'teacher_verified'. Narrows within the "
                "answer-keyed set; it never widens past rule Q1, so combining "
                "it with has_scheme=false is a 400 rather than an empty page.")


def _checked_key_provenance(value: str | None) -> str | None:
    """Reject a provenance that is not a published key, rather than return [].

    `has_scheme=true` used to mean `provenance == "cbse_marking_scheme"`. It is
    now the general answer-key rule, so this parameter is the only way left to
    ask for the board's own schemes -- and `/v1/coverage` reports
    `withOfficialCbseScheme`, a number that would otherwise name a set no
    request could select. Anything outside `KEY_PROVENANCE` (a teacher's
    answer, an unlabelled scheme) can never be served under Q1, so a typo like
    `?key_provenance=cbse` gets a 400 naming the accepted values instead of a
    silent empty page that reads like "the bank has none of those".
    """
    if value is None:
        return None
    if value not in KEY_PROVENANCE:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "unknown key provenance",
            "status": 400,
            "detail": ("key_provenance must be one of "
                       + ", ".join(sorted(KEY_PROVENANCE))),
        })
    return value


def _checked_key_filters(has_scheme: bool | None,
                         key_provenance: str | None) -> str | None:
    """The same refusal, for a pair that contradicts itself.

    `?has_scheme=false&key_provenance=cbse_marking_scheme` asks for the records
    with no answer key whose answer key is CBSE's. The two filters cannot both
    hold -- `key_provenance` narrows within the keyed set by construction --
    so the request answered 200 with an empty page, and an auditor combining
    the flags was told the unanswerable set contains no CBSE records. True, and
    meaningless: it is the same "silent empty page that reads like 'the bank
    has none of those'" that `_checked_key_provenance` above exists to prevent,
    arrived at from the other direction.

    Only an explicitly false `has_scheme` conflicts. `include_unkeyed=true`
    widens and `key_provenance` narrows back to the keyed records of that kind,
    which is a set the API can actually hand over.
    """
    if key_provenance is not None and has_scheme is False:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "contradictory filters",
            "status": 400,
            "detail": ("has_scheme=false selects questions with NO answer key, "
                       "and key_provenance selects a kind of answer key; no "
                       "record can satisfy both. Drop one of them: "
                       "has_scheme=false for the unanswerable set, or "
                       "key_provenance alone for the board's own schemes."),
        })
    return _checked_key_provenance(key_provenance)


TRUST_PARAM = Query(
    default=None,
    description="narrow to one trust label (QB-5): 'published' for a key CBSE "
                "or NCERT published, 'checked' for one checked before it was "
                "served (grounded in the textbook, two agreeing solves, or a "
                "teacher's approval). The list's `keyTier`. Narrows within the "
                "answer-keyed set, so with has_scheme=false it is a 400.")


def _checked_trust(has_scheme: bool | None, trust: str | None) -> str | None:
    """`trust`, refused the way `key_provenance` is: a closed vocabulary, so a
    typo is a 400 naming the two labels rather than an empty page, and asking
    for a trusted key among records with no key is a contradiction."""
    if trust is None:
        return None
    if trust not in TRUST_LEVELS:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "unknown trust label",
            "status": 400,
            "detail": "trust must be one of " + ", ".join(sorted(TRUST_LEVELS)),
        })
    if has_scheme is False:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "contradictory filters",
            "status": 400,
            "detail": ("has_scheme=false selects questions with NO answer key, "
                       "and trust selects how far a key is trusted; no record "
                       "can satisfy both. Drop one of them."),
        })
    return trust


def _page(bank: QuestionBank, **filters: Any):
    """`QuestionBank.page`, with a bad cursor turned into the promised 400.

    The engine is framework-free, so it raises `InvalidCursor`; the HTTP
    contract in `docs/question-bank-api.md` promises problem+json, and that
    translation belongs here rather than in the engine.
    """
    try:
        return bank.page(**filters)
    except InvalidCursor as exc:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "invalid cursor",
            "status": 400,
            "detail": "the cursor is not decodable; omit it to start again",
        }) from exc


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@router.get("/questions", responses=_errors(400))
def list_questions(
    request: Request,
    key: ApiKey = Depends(_require("questions:read")),
    subject: str | None = None,
    grade: int | None = None,
    marks: int | None = None,
    question_type: str | None = Query(default=None, alias="type"),
    difficulty: str | None = None,
    bloom: str | None = None,
    chapter_id: str | None = None,
    subtopic_id: str | None = None,
    topic_id: str | None = Query(
        default=None,
        description="exact topic id, one of a record's `topicIds` (e.g. "
                    "'science-10/life-processes/nutrition')."),
    competency: str | None = Query(
        default=None,
        description="exact competency id, one of a record's `competencyIds`. "
                    "No record carried one when this filter was added "
                    "(2026-10-01), so it answers an empty page until the bank "
                    "is tagged; it exists so a client written now keeps working."),
    source: str | None = Query(
        default=None,
        description="where the question comes from, as the record's `source` "
                    "names it: 'cbse_board_paper', 'cbse_sample_paper', "
                    "'cbse_question_bank', 'ncert_exemplar', ... Case is "
                    "ignored; the `source` facet lists what the bank holds."),
    trust: str | None = TRUST_PARAM,
    book: str | None = Query(
        default=None,
        description="the publication the question was printed in, as its "
                    "rights attribution names it (e.g. 'NCERT, Exemplar "
                    "Problems, Class X Science'). Case and spacing are "
                    "ignored; the `book` facet lists the exact names."),
    has_scheme: bool | None = Query(
        default=None,
        description="true: only answer-keyed questions (the default). "
                    "false: only the unanswerable set, for auditing. Since "
                    "the Q1 pass this is the shared answer-key rule (published "
                    "provenance AND real content), not the CBSE label alone; "
                    "use key_provenance for the board's own schemes."),
    key_provenance: str | None = KEY_PROVENANCE_PARAM,
    review_state: str | None = None,
    keyword: str | None = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    cursor: str | None = None,
    facets: bool = Query(default=False, description="include facet counts"),
    include_unkeyed: bool = Query(
        default=False,
        description="serve questions with no answer key as well. Rule Q1 "
                    "excludes them by default; this is the explicit opt-in."),
) -> JSONResponse:
    """Paged, filtered question list. The workhorse endpoint.

    **Answer-keyed by default (rule Q1).** See `_q1_filter` for the two ways
    to ask for the rest, and `key_provenance` to narrow to one kind of key.
    A parameter not listed here is a 422, and a class or subject outside the
    key's limits is a 403.
    """
    assert _bank is not None
    _refuse_outside(key, grade=grade, subject=subject)
    filters = dict(subject=subject, grade=grade, marks=marks, type_=question_type,
                   difficulty=difficulty, bloom=bloom, chapter_id=chapter_id,
                   subtopic_id=subtopic_id,
                   has_scheme=_q1_filter(has_scheme, include_unkeyed),
                   key_provenance=_checked_key_filters(has_scheme, key_provenance),
                   review_state=review_state, keyword=keyword,
                   topic_id=topic_id, competency=competency, source=source,
                   trust=_checked_trust(has_scheme, trust), book=book,
                   **_limits(key))

    items, next_cursor, total = _page(_bank, cursor=cursor, limit=limit, **filters)
    body: dict[str, Any] = {
        "items": [_slim(r) for r in items],
        "count": len(items),
        "totalMatching": total,
        "nextCursor": next_cursor,
    }
    if facets:
        body["facets"] = _bank.facets(**filters)

    headers = _rate_headers(key)
    if next_cursor:
        # The same query with the cursor moved on. It was the bare path plus
        # `?cursor=`, so a client following `Link` paged the whole bank from
        # page two, every filter dropped.
        headers["Link"] = '<{}>; rel="next"'.format(
            request.url.include_query_params(cursor=next_cursor))
    return JSONResponse(body, headers=headers)


@router.get("/questions/{question_id}/scheme", responses=_errors(404))
def get_scheme(
    question_id: str,
    key: ApiKey = Depends(_require("questions:read")),
) -> JSONResponse:
    """Just the marking scheme.

    Its own resource because it is the scarce half of the bank: a consumer
    scoring an answer needs the value points and the provenance that says they
    are official, and nothing else on the record.
    """
    assert _bank is not None
    rec = _bank.get(question_id)
    if rec is None:
        raise HTTPException(404, detail=f"no question {question_id!r}")
    _refuse_record_outside(key, rec)

    scheme = rec.get("answerScheme") or {}
    return JSONResponse({
        "questionId": question_id,
        "totalMarks": scheme.get("totalMarks"),
        "markingPoints": scheme.get("markingPoints") or [],
        "modelAnswer": scheme.get("modelAnswer") or "",
        "hasPartialCredit": scheme.get("hasPartialCredit", False),
        # The consumer needs to know whether this is an official CBSE scheme or
        # an empty placeholder, because two thirds of the bank is the latter.
        # The label is reported as it is, and the flag says whether anything
        # can actually be marked with it -- as the MCP's get_answer_key does.
        "provenance": scheme.get("provenance") or "none",
        "hasAnswerKey": QuestionBank.has_answer_key(rec),
        "sourcePaperCode": scheme.get("sourcePaperCode") or "",
        "sourceDocumentId": scheme.get("sourceDocumentId") or "",
    }, headers=_rate_headers(key))


@router.get("/questions/{question_id}", responses=_errors(404))
def get_question(
    question_id: str,
    key: ApiKey = Depends(_require("questions:read")),
) -> JSONResponse:
    """One question, with full provenance and scheme."""
    assert _bank is not None
    rec = _bank.get(question_id)
    if rec is None:
        raise HTTPException(404, detail=f"no question {question_id!r}")
    _refuse_record_outside(key, rec)
    return JSONResponse(rec, headers=_rate_headers(key))


# `:path`, because every real subtopic id carries slashes
# ("science-10/life-processes/nutrition/nutrition-in-human-beings"). A plain
# `{subtopic_id}` stops at the first one, so all 87 real ids were a 404 here --
# percent-encoded or not, since the server decodes `%2F` before routing --
# while `/v1/questions?subtopic_id=` found every one of them (audit D52). The
# OpenAPI path is unchanged; only the matching is.
@router.get("/subtopics/{subtopic_id:path}/questions", responses=_errors(400))
def questions_for_subtopic(
    subtopic_id: str,
    key: ApiKey = Depends(_require("questions:read")),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    cursor: str | None = None,
    has_scheme: bool | None = Query(
        default=None,
        description="true: only answer-keyed questions (the default). "
                    "false: only the unanswerable set, for auditing this "
                    "subtopic."),
    key_provenance: str | None = KEY_PROVENANCE_PARAM,
    include_unkeyed: bool = Query(default=False),
) -> JSONResponse:
    """Curriculum-driven listing: everything tagged to one subtopic.

    Answer-keyed by default, like `/v1/questions` -- a rule that held on one
    listing route and not its neighbour would not be a rule. **And the same
    two opt-ins**: `has_scheme` was documented as accepted by all three
    listing routes but was never declared here, so FastAPI dropped it and
    `?has_scheme=false` answered 200 with the exact OPPOSITE set -- an auditor
    asking which questions of a subtopic the relink has not reached was handed
    the ones it already covers, with nothing in the response to say so.
    """
    assert _bank is not None
    # The same `_page(subtopic_id=...)` as `?subtopic_id=`, so the path and the
    # query can only disagree if the route stops matching the id -- which is
    # what the `:path` above is for.
    items, next_cursor, total = _page(
        _bank, subtopic_id=subtopic_id, cursor=cursor, limit=limit,
        has_scheme=_q1_filter(has_scheme, include_unkeyed),
        key_provenance=_checked_key_filters(has_scheme, key_provenance),
        **_limits(key))
    return JSONResponse({
        "subtopicId": subtopic_id,
        "items": [_slim(r) for r in items],
        "count": len(items),
        "totalMatching": total,
        "nextCursor": next_cursor,
    }, headers=_rate_headers(key))


@router.get("/coverage", responses=_errors())
def get_coverage(
    key: ApiKey = Depends(_require("facets:read")),
) -> JSONResponse:
    """How much of the bank is answerable, per subject and class.

    The audit found no way to ask this over HTTP (`/v1/coverage` was a 404)
    while the MCP server had `coverage_report` -- so the number a buyer could
    read and the number an agent could read came from different places, or in
    the HTTP case from nowhere. Both now return `QuestionBank.coverage()`
    verbatim (rule Q5).

    Every count here names a set a request can actually be handed:
    `withOfficialCbseScheme` is `?key_provenance=cbse_marking_scheme`, and per
    subject and class `markValues` lists the distinct mark denominations while
    `marksTotal` is what the answer-keyed questions add up to. The field was
    called `marksAvailable` and held the denominations, which on a coverage
    endpoint reads as the total a buyer would size a purchase on.

    Aggregate counts, so it takes `facets:read` rather than `questions:read`:
    no question text or marking scheme crosses this route. A key limited to
    some classes or subjects is told the coverage of those.
    """
    assert _bank is not None
    return JSONResponse(_bank.coverage(**_limits(key)), headers=_rate_headers(key))


@router.get("/facets", responses=_errors(400))
def get_facets(
    key: ApiKey = Depends(_require("facets:read")),
    subject: str | None = None,
    grade: int | None = None,
    has_scheme: bool | None = Query(
        default=None,
        description="true: count only answer-keyed questions (the default). "
                    "false: count only the unanswerable set, for auditing."),
    key_provenance: str | None = KEY_PROVENANCE_PARAM,
    include_unkeyed: bool = Query(
        default=False,
        description="count questions with no answer key as well. Rule Q1 "
                    "excludes them by default; this is the explicit opt-in."),
) -> JSONResponse:
    """The filter vocabulary and its counts under the current filter.

    **Under the same Q1 default as `/v1/questions`, and the same two opt-ins.**
    These counts exist to size a query, so counting a wider set than the query
    will serve is a lie about the page the consumer is about to request --
    whichever door the count comes through. The inline facets on
    `/v1/questions` took the default and this route did not, which put the two
    counts of one thing 8 records apart on the fixture bank (20 against 12) and
    is the exact divergence Q1 and Q5 were written to end.
    """
    assert _bank is not None
    _refuse_outside(key, grade=grade, subject=subject)
    return JSONResponse(
        _bank.facets(subject=subject, grade=grade,
                     has_scheme=_q1_filter(has_scheme, include_unkeyed),
                     key_provenance=_checked_key_filters(has_scheme,
                                                         key_provenance),
                     **_limits(key)),
        headers=_rate_headers(key))


# --------------------------------------------------------------------------- #
# homework sets (API-2)
# --------------------------------------------------------------------------- #

MAX_SET = 50     # questions in one homework set, whatever the mix


class HomeworkSetRequest(BaseModel):
    """N answer-keyed questions for a class, from chapters, topics or
    subtopics, in a mix of marks or of types, avoiding questions the caller
    has already used. The API knows no students (API-5), so "already had" is
    the caller's list of ids.

    API-2 asks for "this mix of marks and types"; the audit found neither a
    marks mix nor a topic filter, and `count` silently ignored whenever
    `types` was sent (N-67-11). The three ways to ask:

      * `count` alone: that many, spread across the chosen chapters.
      * a mix -- `types` as `{"mcq": 5, "short_answer": 3}` or `marks` as
        `{"1": 4, "3": 2}` -- that many of each. `count`, if sent too, must
        agree with the mix's total; it is a 422 rather than one of the two
        quietly winning.
      * `types` as a list, `["mcq", "short_answer"]`: `count` questions of
        those types, spread across them. With a `marks` mix it chooses the
        types inside each mark value.

    A types mix and a marks mix together are refused: honouring both at once
    is an allocation problem a greedy pick gets wrong without saying so.
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    subject: str
    grade: int = Field(ge=1, le=12)
    chapter_ids: list[str] = Field(default_factory=list, alias="chapterIds", max_length=30)
    topic_ids: list[str] = Field(
        default_factory=list, alias="topicIds", max_length=50,
        description="only questions carrying one of these topic ids (a record's `topicIds`)")
    subtopic_ids: list[str] = Field(
        default_factory=list, alias="subtopicIds", max_length=50,
        description="only questions carrying one of these subtopic ids (a record's `subtopicIds`)")
    count: int = Field(default=10, ge=1, le=MAX_SET,
                       description="how many questions, when no mix is given")
    types: dict[str, Annotated[int, Field(ge=0, le=MAX_SET)]] | list[str] = Field(
        default_factory=dict,
        description='a mix, {"mcq": 5, "short_answer": 3}: that many of each type; '
                    'or a list, ["mcq", "short_answer"]: `count` questions of those '
                    'types, spread across them')
    marks: dict[int, Annotated[int, Field(ge=0, le=MAX_SET)]] = Field(
        default_factory=dict,
        description='a mix of marks, {"1": 4, "3": 2}: four 1-mark and two 3-mark '
                    'questions. Give `types` as a list to choose the types inside it.')
    exclude_ids: list[str] = Field(default_factory=list, alias="excludeIds", max_length=5000)
    seed: str | None = None

    def mix(self) -> dict[str, tuple[Callable[[dict[str, Any]], bool], int]] | None:
        """The buckets asked for, by the label `shortfall` reports them under,
        or None when no mix was given and `count` decides."""
        if isinstance(self.types, dict) and self.types:
            return {t: ((lambda r, t=t: r.get("type") == t), n) for t, n in self.types.items()}
        if self.marks:
            return {str(m): ((lambda r, m=m: r.get("marks") == m), n) for m, n in self.marks.items()}
        return None

    @model_validator(mode="after")
    def _one_mix_that_adds_up(self) -> "HomeworkSetRequest":
        if isinstance(self.types, dict) and self.types and self.marks:
            raise ValueError(
                "ask for a mix of marks or a mix of types, not both; to choose the "
                "types inside a marks mix, send types as a list, e.g. "
                '"types": ["mcq", "short_answer"]')
        if any(m < 1 for m in self.marks):
            raise ValueError("a mark value in `marks` must be 1 or more")
        buckets = self.mix()
        if buckets is not None:
            total = sum(n for _, n in buckets.values())
            if not 1 <= total <= MAX_SET:
                raise ValueError(f"the mix asks for {total} questions; a set holds 1 to {MAX_SET}")
            if "count" in self.model_fields_set and self.count != total:
                raise ValueError(
                    f"count is {self.count} but the mix adds up to {total}; send "
                    "one of them, or make them agree")
        return self


def _candidates(bank: QuestionBank, req: HomeworkSetRequest, key: ApiKey) -> list[dict[str, Any]]:
    chapters = req.chapter_ids or [None]
    seen: dict[str, dict[str, Any]] = {}
    for chapter in chapters:
        cursor = None
        while True:
            items, cursor, _ = _page(bank, subject=req.subject, grade=req.grade, chapter_id=chapter,
                                     has_scheme=True, cursor=cursor, limit=MAX_LIMIT, **_limits(key))
            for r in items:
                seen.setdefault(r["id"], r)
            if not cursor:
                break
    excluded = set(req.exclude_ids)
    topics, subtopics = set(req.topic_ids), set(req.subtopic_ids)
    allowed_types = set(req.types) if isinstance(req.types, list) else set()
    return [r for r in seen.values()
            if r["id"] not in excluded
            and (not topics or topics & set(r.get("topicIds") or []))
            and (not subtopics or subtopics & set(r.get("subtopicIds") or []))
            and (not allowed_types or r.get("type") in allowed_types)]


def _chapter_of(rec: dict[str, Any]) -> str:
    return rec.get("taxonomyChapterId") or next(iter(rec.get("chapterIds") or []), "-")


def _groups(records: list[dict[str, Any]],
            by: Callable[[dict[str, Any]], Any]) -> list[list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        out.setdefault(by(r), []).append(r)
    return list(out.values())


def _round_robin(groups: list[list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    """The first of each group in turn, until `n` are taken or none are left,
    so one big chapter (or type) cannot crowd out the rest."""
    queues = [list(g) for g in groups]
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return picked


def _spread(records: list[dict[str, Any]], n: int, *, by_type: bool) -> list[dict[str, Any]]:
    """`n` of `records`, spread across chapters -- and first across types,
    when the caller listed the types they want rather than counting them."""
    if not by_type:
        return _round_robin(_groups(records, _chapter_of), n)
    per_type = [_round_robin(_groups(g, _chapter_of), len(g))
                for g in _groups(records, lambda r: r.get("type"))]
    return _round_robin(per_type, n)


@router.post("/homework-sets", responses=_errors())
def homework_set(req: HomeworkSetRequest, key: ApiKey = Depends(_require("questions:read"))) -> JSONResponse:
    """A homework set (API-2): `count` questions, or a mix of types or of
    marks (see `HomeworkSetRequest`), from the chosen chapters, topics and
    subtopics. The same request with the same `seed` returns the same set;
    `shortfall` names each bucket the bank could not fill and by how many
    (`"any"` when no mix was given)."""
    assert _bank is not None
    _refuse_outside(key, grade=req.grade, subject=req.subject)
    pool = _candidates(_bank, req, key)
    rng = random.Random(req.seed or f"{req.subject}:{req.grade}:{','.join(sorted(req.chapter_ids))}")
    pool.sort(key=lambda r: r["id"])
    rng.shuffle(pool)
    by_type = isinstance(req.types, list) and bool(req.types)
    picked: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    buckets = req.mix()
    if buckets is None:
        picked = _spread(pool, req.count, by_type=by_type)
        if len(picked) < req.count:
            shortfall["any"] = req.count - len(picked)
        requested = req.count
    else:
        taken: set[str] = set()
        for label, (wanted, n) in buckets.items():
            got = _spread([r for r in pool if wanted(r) and r["id"] not in taken], n, by_type=by_type)
            taken.update(r["id"] for r in got)
            picked += got
            if len(got) < n:
                shortfall[label] = n - len(got)
        requested = sum(n for _, n in buckets.values())
    body = {"items": [_slim(r) for r in picked], "count": len(picked),
            "requested": requested, "available": len(pool), "shortfall": shortfall}
    return JSONResponse(body, headers=_rate_headers(key))
