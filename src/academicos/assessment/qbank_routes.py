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
vocabulary grants it.

Honesty about what is not here: `POST /v1/papers` is **not** implemented in this
pass -- paper generation already exists at `POST /api/v1/papers/generate` behind
user auth, and re-exposing it needs a decision about whether an API key may
consume the blueprint engine. It is listed in the doc and omitted here rather
than half-built.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

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
    InvalidCursor,
    QuestionBank,
    decode_cursor,
    encode_cursor,
    key_provenance_of,
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
    global _store, _bank
    _store = ApiKeyStore(db_path)

    path = Path(bank_path)
    if not path.exists():
        _bank = None
        log.warning("question-bank corpus missing at %s; the /v1 question "
                    "surface will answer 503", path)
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _bank = None
        log.warning("question-bank corpus at %s is unreadable (%s); the /v1 "
                    "question surface will answer 503", path, exc)
        return

    _bank = QuestionBank(payload.get("questions") or [])
    log.info("question-bank API ready: %d questions", len(_bank))


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


def _require(scope: str):
    """Dependency factory: authenticate a key and require one scope.

    Returning a closure means the required scope is visible in the route
    signature rather than buried in the body, so a reviewer can see the whole
    permission surface by reading the decorators.
    """
    def dependency(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> ApiKey:
        if _store is None or _bank is None:
            raise HTTPException(503, detail="question-bank API not initialised")
        try:
            return authorize(_store, presented_key(authorization, x_api_key),
                             scope)
        except AuthFailure as exc:
            raise HTTPException(exc.status, detail=exc.detail,
                                headers=exc.headers or None) from exc
    return dependency


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
    description="narrow to ONE kind of published key: 'cbse_marking_scheme' "
                "for the board's own schemes, 'ncert_exemplar_answer' for "
                "NCERT's Exemplar answers. Narrows within the answer-keyed "
                "set; it never widens past rule Q1, so combining it with "
                "has_scheme=false is a 400 rather than an empty page.")


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

@router.get("/questions")
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
    """
    assert _bank is not None
    filters = dict(subject=subject, grade=grade, marks=marks, type_=question_type,
                   difficulty=difficulty, bloom=bloom, chapter_id=chapter_id,
                   subtopic_id=subtopic_id,
                   has_scheme=_q1_filter(has_scheme, include_unkeyed),
                   key_provenance=_checked_key_filters(has_scheme, key_provenance),
                   review_state=review_state, keyword=keyword)

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
        headers["Link"] = '<{}?cursor={}>; rel="next"'.format(
            str(request.url).split("?")[0], next_cursor)
    return JSONResponse(body, headers=headers)


@router.get("/questions/{question_id}/scheme")
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


@router.get("/questions/{question_id}")
def get_question(
    question_id: str,
    key: ApiKey = Depends(_require("questions:read")),
) -> JSONResponse:
    """One question, with full provenance and scheme."""
    assert _bank is not None
    rec = _bank.get(question_id)
    if rec is None:
        raise HTTPException(404, detail=f"no question {question_id!r}")
    return JSONResponse(rec, headers=_rate_headers(key))


@router.get("/subtopics/{subtopic_id}/questions")
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
    items, next_cursor, total = _page(
        _bank, subtopic_id=subtopic_id, cursor=cursor, limit=limit,
        has_scheme=_q1_filter(has_scheme, include_unkeyed),
        key_provenance=_checked_key_filters(has_scheme, key_provenance))
    return JSONResponse({
        "subtopicId": subtopic_id,
        "items": [_slim(r) for r in items],
        "count": len(items),
        "totalMatching": total,
        "nextCursor": next_cursor,
    }, headers=_rate_headers(key))


@router.get("/coverage")
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
    no question text or marking scheme crosses this route.
    """
    assert _bank is not None
    return JSONResponse(_bank.coverage(), headers=_rate_headers(key))


@router.get("/facets")
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
    return JSONResponse(
        _bank.facets(subject=subject, grade=grade,
                     has_scheme=_q1_filter(has_scheme, include_unkeyed),
                     key_provenance=_checked_key_filters(has_scheme,
                                                         key_provenance)),
        headers=_rate_headers(key))
