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

# The provenances that make a scheme a PUBLISHED answer key: CBSE's own marking
# scheme, and NCERT's answer from an Exemplar book (4,192 of the 5,427 served
# records). Everything else -- a teacher's answer, a sandbox fabrication, an
# unlabelled scheme of unknown origin -- may be a perfectly good answer, but it
# is not the board's, and Q1 is about what the bank may serve as authoritative.
KEY_PROVENANCE: frozenset[str] = frozenset({
    "cbse_marking_scheme",
    "ncert_exemplar_answer",
})


# --------------------------------------------------------------------------- #
# the in-memory bank
# --------------------------------------------------------------------------- #

class QuestionBank:
    """The enriched corpus, loaded once and queried in memory.

    Same posture as `pool.py`: built once at startup, cheap at this size
    (5,427 records, ~7 MB). `docs/question-bank-api.md` section 5 puts a read
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
        examination board would enforce it too. Every default path excludes
        records with no answer key -- the MCP's search and build_paper, and
        (since the audit caught it serving all 3,286 records) `/v1/questions`
        and `/v1/subtopics/{id}/questions` too. Asking for them explicitly is
        the audit path, and it stays open.

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
        """**The** answer-key rule: a published provenance AND real content.

        This is the single definition rule Q1 is enforced with, on both
        surfaces. It used to be two, and they disagreed: `_matches` tested the
        provenance label (2,930 records of the live bank), this function tested
        the content (2,941), and `/v1/questions` applied neither, serving all
        3,286 including 345 with no answer content at all.

        Both halves earn their place:

          * **Content**, because a record can carry
            `provenance: "cbse_marking_scheme"` and hold nothing -- 259 CBE
            multiple-choice records did exactly that before the builder was
            fixed, and a key whose points all carry 0 marks was refused for
            the same reason (41 board records, Task 121).
          * **Provenance**, because content alone would let an unattributed or
            teacher-authored answer be served with the authority of a board
            marking scheme. `KEY_PROVENANCE` names the two that are published
            keys.
        """
        scheme = rec.get("answerScheme") or {}
        if scheme.get("provenance") not in KEY_PROVENANCE:
            return False
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

    def coverage(self) -> dict[str, Any]:
        """The honest state of the bank, reported rather than implied.

        One computation for both surfaces (Q5): `GET /v1/coverage` and the
        MCP's `coverage_report` return this same dict, so a buyer reading the
        HTTP API and an agent reading the MCP cannot be told different numbers
        about the same corpus. Counted over EVERY record, keyed or not --
        coverage that silently excluded the unanswerable set would always
        report 100%.
        """
        total = len(self.records)
        with_key = sum(1 for r in self.records if self.has_answer_key(r))
        official = sum(1 for r in self.records
                       if (r.get("answerScheme") or {}).get("provenance")
                       == "cbse_marking_scheme")

        by_subject: dict[str, dict[str, int]] = {}
        by_pair: dict[tuple[str, int], dict[str, Any]] = {}
        for r in self.records:
            subject = str(r.get("subject") or "unknown")
            grade = int(r.get("grade") or 0)
            keyed = self.has_answer_key(r)

            d = by_subject.setdefault(subject, {"total": 0, "withAnswerKey": 0})
            d["total"] += 1
            row = by_pair.setdefault((subject, grade), {
                "subject": subject, "grade": grade,
                "questions": 0, "withAnswerKey": 0, "_marks": set(),
            })
            row["questions"] += 1
            if keyed:
                d["withAnswerKey"] += 1
                row["withAnswerKey"] += 1
                # Marks a paper can actually be built from, so only the keyed
                # records count: a 3-mark question that cannot be marked does
                # not make 3 marks available.
                row["_marks"].add(int(r.get("marks") or 0))

        by_subject_class = [
            {"subject": row["subject"], "grade": row["grade"],
             "questions": row["questions"], "withAnswerKey": row["withAnswerKey"],
             "marksAvailable": sorted(row["_marks"])}
            for _, row in sorted(by_pair.items())
        ]
        return {
            "questions": total,
            "withAnswerKey": with_key,
            "withOfficialCbseScheme": official,
            "answerKeyCoverage": round(with_key / total, 4) if total else 0.0,
            "subjects": dict(sorted(by_subject.items())),
            "grades": sorted({int(r.get("grade") or 0) for r in self.records}),
            "bySubjectClass": by_subject_class,
            "note": ("Questions without an answer key are excluded from every "
                     "search by default. They exist in the corpus and are "
                     "reported here rather than hidden."),
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
        # The ONE rule, not a second one. This tested the provenance label
        # alone, which counted an official label over an empty scheme as
        # "has a scheme" and disagreed with the MCP surface by 11 records.
        if QuestionBank.has_answer_key(rec) != bool(has_scheme):
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
        # `hasOfficialScheme` alone stopped being the answer to "can I mark
        # this?" when the Exemplar records arrived: 4,192 of the 5,427 served
        # records carry NCERT's own answer, which is a published key and is not
        # a CBSE marking scheme. Both are reported, plus the label itself, so a
        # consumer can tell which kind of key it is holding.
        "hasAnswerKey": QuestionBank.has_answer_key(rec),
        "keyProvenance": scheme.get("provenance") or "none",
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
                    "false: only the unanswerable set, for auditing."),
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
    to ask for the rest.
    """
    assert _bank is not None
    filters = dict(subject=subject, grade=grade, marks=marks, type_=question_type,
                   difficulty=difficulty, bloom=bloom, chapter_id=chapter_id,
                   subtopic_id=subtopic_id,
                   has_scheme=_q1_filter(has_scheme, include_unkeyed),
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
    include_unkeyed: bool = Query(default=False),
) -> JSONResponse:
    """Curriculum-driven listing: everything tagged to one subtopic.

    Answer-keyed by default, like `/v1/questions` -- a rule that held on one
    listing route and not its neighbour would not be a rule.
    """
    assert _bank is not None
    items, next_cursor, total = _bank.page(
        subtopic_id=subtopic_id, cursor=cursor, limit=limit,
        has_scheme=_q1_filter(None, include_unkeyed))
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
) -> JSONResponse:
    """The filter vocabulary and its counts under the current filter."""
    assert _bank is not None
    return JSONResponse(
        _bank.facets(subject=subject, grade=grade),
        headers=_rate_headers(key))
