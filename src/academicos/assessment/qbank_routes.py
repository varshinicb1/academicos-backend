"""The public question-bank API: a resource surface over the enriched bank.

`docs/question-bank-api.md` section 4 asks for this specifically, and gives the
reasons the contract is shaped the way it is:

    GET  /v1/questions                 list + filter (the workhorse)
    GET  /v1/questions/{id}            one, with full provenance and scheme
    GET  /v1/questions/{id}/scheme     just the marking scheme
    GET  /v1/subtopics/{id}/questions  curriculum-driven listing
    GET  /v1/facets                    filter vocabulary and counts
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

import base64
import binascii
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .api_keys import ApiKey, ApiKeyStore, QuotaExceeded

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["question-bank"])

_store: Optional[ApiKeyStore] = None
_bank: Optional["QuestionBank"] = None

# Hard ceiling on page size. A consumer asking for 10,000 items is either
# scraping or has misunderstood; either way the answer is no, not 10,000.
MAX_LIMIT = 200
DEFAULT_LIMIT = 25


# --------------------------------------------------------------------------- #
# the in-memory bank
# --------------------------------------------------------------------------- #

class QuestionBank:
    """The enriched corpus, loaded once and queried in memory.

    Same posture as `pool.py`: built once at startup, cheap at this size
    (3,286 records, ~7 MB). `docs/question-bank-api.md` section 5 puts a read
    replica and Redis in front of this for the real deployment; that is a
    deployment concern, not a code one, and this does not pretend otherwise.
    """

    def __init__(self, records: list[dict[str, Any]], *,
                 require_answer_key: bool = False):
        """`require_answer_key` DEFAULTS OFF, and that is deliberate.

        Q1 says "without an answer key, no question", and this was briefly
        enforced by dropping unanswerable records here. That was wrong: it
        destroyed a designed feature. The MCP surface lets an agent AUDIT the
        unanswerable set (`answer_key_state: "none"`, documented as *"wants the
        unanswerable set has to say so in those words"*), and
        `coverage_report` counts it under Q4. Filtering at construction made
        both impossible while reporting a healthy record count.

        Q1 is enforced where it belongs -- at QUERY time, which is where an
        examination board would enforce it too. Every default path (search,
        build_paper) excludes records with no scheme; asking for them explicitly
        is the audit path, and it stays open.

        The parameter remains so a caller who genuinely wants a pre-filtered
        bank can ask for one; nothing does today.
        """
        if require_answer_key:
            records = [r for r in records if self.has_answer_key(r)]
        self.records = records
        self._by_id = {str(r.get("id")): r for r in records if r.get("id")}
        self._sorted_ids = sorted(self._by_id)

    @staticmethod
    def has_answer_key(rec: dict[str, Any]) -> bool:
        """An answer scheme with actual content -- not just a provenance label.

        A record can carry `provenance: "cbse_marking_scheme"` and still have
        nothing in it; 259 CBE multiple-choice records did exactly that before
        the builder was fixed. The check is on the CONTENT, so a label alone can
        never make an unanswerable question look answerable.
        """
        scheme = rec.get("answerScheme") or {}
        if (scheme.get("modelAnswer") or "").strip():
            return True
        # Both field names, deliberately. The canonical shape uses
        # `description`, but one builder briefly emitted `text` -- and a gate
        # that knows only one of them filters a whole corpus out while
        # reporting a healthy record count, which is exactly what happened.
        # Accepting either is cheap insurance against the next shape change.
        return any(
            ((pt.get("description") or pt.get("text") or "")).strip()
            for pt in (scheme.get("markingPoints") or [])
        )

    def __len__(self) -> int:
        return len(self.records)

    def get(self, question_id: str) -> dict[str, Any] | None:
        return self._by_id.get(question_id)

    def page(
        self,
        *,
        subject: str | None = None,
        grade: int | None = None,
        marks: int | None = None,
        type_: str | None = None,
        difficulty: str | None = None,
        bloom: str | None = None,
        chapter_id: str | None = None,
        subtopic_id: str | None = None,
        has_scheme: bool | None = None,
        review_state: str | None = None,
        keyword: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        """One keyset page. Returns `(items, next_cursor, total_matching)`.

        Ordered by `id`, which is stable and unique, so a cursor is simply "the
        last id I sent you" and paging cannot skip or duplicate a row when the
        bank is edited between requests.
        """
        after = decode_cursor(cursor) if cursor else None
        needle = (keyword or "").strip().lower()

        items: list[dict[str, Any]] = []
        total = 0
        last_included: str | None = None
        more_available = False

        for qid in self._sorted_ids:
            rec = self._by_id[qid]
            if not _matches(rec, subject=subject, grade=grade, marks=marks,
                            type_=type_, difficulty=difficulty, bloom=bloom,
                            chapter_id=chapter_id, subtopic_id=subtopic_id,
                            has_scheme=has_scheme, review_state=review_state,
                            needle=needle):
                continue
            # `totalMatching` counts everything the filter matches, not just
            # what remains after the cursor, so a caller can size the result set
            # from any page.
            total += 1
            if after is not None and qid <= after:
                continue
            if len(items) < limit:
                items.append(rec)
                last_included = qid
            else:
                more_available = True

        # The cursor is the LAST INCLUDED id, not the first excluded one. Sending
        # the first excluded id makes the next request skip it -- which silently
        # dropped one record per page, seven of sixty in the test that caught it.
        next_cursor = encode_cursor(last_included) if (more_available and last_included) else None
        return items, next_cursor, total

    def facets(self, **filters: Any) -> dict[str, Any]:
        """Counts for the filter vocabulary, computed over the filtered set.

        `question-bank-api.md`: "Include `facets` on list responses -- one round
        trip instead of list-then-fetch-counts."
        """
        counts: dict[str, dict[str, int]] = {
            "subject": {}, "grade": {}, "marks": {}, "type": {},
            "difficulty": {}, "bloomLevel": {}, "reviewState": {},
        }
        for rec in self._by_id.values():
            if not _matches(
                rec,
                subject=filters.get("subject"), grade=filters.get("grade"),
                marks=filters.get("marks"), type_=filters.get("type_"),
                difficulty=filters.get("difficulty"), bloom=filters.get("bloom"),
                chapter_id=filters.get("chapter_id"),
                subtopic_id=filters.get("subtopic_id"),
                has_scheme=filters.get("has_scheme"),
                review_state=filters.get("review_state"),
                needle=(filters.get("keyword") or "").strip().lower(),
            ):
                continue
            for key, field in (
                ("subject", "subject"), ("grade", "grade"), ("marks", "marks"),
                ("type", "type"), ("difficulty", "difficulty"),
                ("bloomLevel", "bloomLevel"), ("reviewState", "reviewState"),
            ):
                value = rec.get(field)
                if value is None:
                    continue
                counts[key][str(value)] = counts[key].get(str(value), 0) + 1
        return {
            k: [{"value": v, "count": c}
                for v, c in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]
            for k, d in counts.items()
        }


def _matches(rec: dict[str, Any], *, subject, grade, marks, type_, difficulty,
             bloom, chapter_id, subtopic_id, has_scheme, review_state,
             needle: str) -> bool:
    if subject and str(rec.get("subject") or "").lower() != subject.lower():
        return False
    if grade is not None and rec.get("grade") != grade:
        return False
    if marks is not None and rec.get("marks") != marks:
        return False
    if type_ and str(rec.get("type") or "") != type_:
        return False
    if difficulty and str(rec.get("difficulty") or "") != difficulty:
        return False
    if bloom and str(rec.get("bloomLevel") or "") != bloom:
        return False
    if chapter_id and chapter_id not in (rec.get("chapterIds") or []):
        return False
    if subtopic_id and subtopic_id not in (rec.get("subtopicIds") or []):
        return False
    if review_state and str(rec.get("reviewState") or "") != review_state:
        return False
    if has_scheme is not None:
        scheme = rec.get("answerScheme") or {}
        is_official = scheme.get("provenance") == "cbse_marking_scheme"
        if bool(is_official) != bool(has_scheme):
            return False
    if needle:
        haystack = " ".join([
            str(rec.get("stem") or ""),
            " ".join(str(t) for t in (rec.get("tags") or [])),
        ]).lower()
        if needle not in haystack:
            return False
    return True


# --------------------------------------------------------------------------- #
# cursor
# --------------------------------------------------------------------------- #

def encode_cursor(last_id: str) -> str:
    """Opaque base64url cursor. Opaque so the encoding can change without
    breaking a consumer that treated it as an offset."""
    return base64.urlsafe_b64encode(last_id.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> str:
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(400, detail={
            "type": "about:blank", "title": "invalid cursor",
            "status": 400,
            "detail": "the cursor is not decodable; omit it to start again",
        }) from exc


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

        presented = ""
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        elif x_api_key:
            presented = x_api_key.strip()

        key = _store.authenticate(presented)
        if key is None:
            raise HTTPException(
                401, detail="a valid API key is required",
                headers={"WWW-Authenticate": "Bearer"})
        if not key.may(scope):
            # 403, not 404: the key is valid, the permission is not held.
            raise HTTPException(
                403, detail=f"this key does not hold the {scope!r} scope")
        try:
            _store.check_quota(key)
        except QuotaExceeded as exc:
            raise HTTPException(
                429, detail=str(exc),
                headers={"Retry-After": str(exc.retry_after)}) from exc
        return key
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
    scheme = rec.get("answerScheme") or {}
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
        "hasOfficialScheme": scheme.get("provenance") == "cbse_marking_scheme",
    }


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
    has_scheme: bool | None = None,
    review_state: str | None = None,
    keyword: str | None = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    cursor: str | None = None,
    facets: bool = Query(default=False, description="include facet counts"),
) -> JSONResponse:
    """Paged, filtered question list. The workhorse endpoint."""
    assert _bank is not None
    filters = dict(subject=subject, grade=grade, marks=marks, type_=question_type,
                   difficulty=difficulty, bloom=bloom, chapter_id=chapter_id,
                   subtopic_id=subtopic_id, has_scheme=has_scheme,
                   review_state=review_state, keyword=keyword)

    items, next_cursor, total = _bank.page(cursor=cursor, limit=limit, **filters)
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
        "provenance": scheme.get("provenance") or "none",
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
) -> JSONResponse:
    """Curriculum-driven listing: everything tagged to one subtopic."""
    assert _bank is not None
    items, next_cursor, total = _bank.page(
        subtopic_id=subtopic_id, cursor=cursor, limit=limit)
    return JSONResponse({
        "subtopicId": subtopic_id,
        "items": [_slim(r) for r in items],
        "count": len(items),
        "totalMatching": total,
        "nextCursor": next_cursor,
    }, headers=_rate_headers(key))


@router.get("/facets")
def get_facets(
    key: ApiKey = Depends(_require("facets:read")),
    subject: str | None = None,
    grade: int | None = None,
) -> JSONResponse:
    """The filter vocabulary and its counts under the current filter."""
    assert _bank is not None
    return JSONResponse(
        _bank.facets(subject=subject, grade=grade),
        headers=_rate_headers(key))
