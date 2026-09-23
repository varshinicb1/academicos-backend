"""MCP server for the question bank.

Two consumers, one surface
--------------------------
`docs/PRD.md` section 4 names paper generation as a core loop, and the AI agents
in `src/academicos/agents/` need the same corpus the HTTP API serves. Rather
than build a second query engine for agents, this exposes the *same*
`QuestionBank` over the Model Context Protocol, so an agent and the backend
cannot disagree about what the bank contains.

Built on the official SDK (`mcp>=2,<3`), which supports the 2026-07-28 spec and
speaks stdio, SSE and Streamable HTTP. `--transport stdio` is the default
because that is what desktop MCP clients launch.

The answer-key rule, enforced here rather than documented
---------------------------------------------------------
The requirement is explicit: **without an answer key, no question.** So every
tool defaults to `require_answer_key=True`, and a question with no published
answer is invisible unless a caller explicitly asks for it by name. This is not
a filter applied at the edge -- it is the default of the query, so an agent
cannot accidentally build a paper it cannot mark.

`coverage_report` exists so the shortfall is visible rather than silent: much of
the older board-paper corpus has no attached scheme, and the honest thing is to
report that number rather than quietly serve the subset that does.

One bank, resolved the way everything else resolves it
------------------------------------------------------
This read three corpora (CBE + SQP + served, 5,840 records) where the HTTP API
and paper generation read only the served bank (3,286), so the two surfaces
could not agree on what the bank contained -- rule Q5's exact failure. Since
the merge (`corpus/bank_merge.py`) the served bank CONTAINS the eligible CBE
and SQP records, so reading them separately would both double-count them and
serve records the merge deliberately excluded. It now reads the served bank
alone, at `$ACOS_DATA_ROOT/syllabus/questions.json` -- the path was
cwd-relative, so running the server from anywhere but the repo root silently
served an empty bank.

Authentication
--------------
`stdio` is unauthenticated by construction: the client launches this process
and owns both ends of the pipe, so there is no network peer to authenticate
and no way for a desktop host to hold a key. The network transports (`sse`,
`streamable-http`) take the SAME scoped API key as the HTTP surface -- same
store, same scope vocabulary, same quota counter (`assessment/api_keys.py`) --
because the same product must not be key-gated on one port and open on
another.

Run
---
    python -m academicos.mcp.qbank_server                  # stdio
    python -m academicos.mcp.qbank_server --transport sse --port 8081
                                                           # a web client, key required
    python -m academicos.mcp.qbank_server --bank PATH      # a specific corpus
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

# The engine, from the FastAPI-free module. Imported at MODULE scope on
# purpose: it used to come from `assessment/qbank_routes.py`, which imports
# `fastapi` at module scope, so `pip install -e ".[mcp]"` raised ImportError on
# the first stdio request -- and deferring the import into the method bodies,
# as this file did, moved that failure from import time to request time rather
# than removing it. `[mcp]` now genuinely runs stdio; the network transports
# still need `[api]`, for uvicorn and for the key gate below.
from ..assessment.qbank_engine import (
    QuestionBank,
    has_answer_key,
    topics_of as _topics_of,
)

if TYPE_CHECKING:  # the SDK and the key store are both optional at import time
    from ..assessment.api_keys import ApiKey, ApiKeyStore

log = logging.getLogger(__name__)

# Relative to the data root, never to the working directory.
SERVED_BANK = Path("syllabus") / "questions.json"
DEFAULT_DATA_ROOT = Path("academicos-data")
KEY_DB = Path("assessment") / "api_keys.sqlite"

# The scope a network MCP client needs. `search_questions` and `get_question`
# return question text and marking schemes, so this is the questions scope the
# HTTP surface requires for exactly the same data.
TRANSPORT_SCOPE = "questions:read"

# The three things a caller may say about answer keys. Closed, because an
# unrecognised value used to fall through every branch and silently turn the
# Q1 default off.
ANSWER_KEY_STATES: frozenset[str] = frozenset({"any", "official", "none"})


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

class Corpus:
    """The loaded bank plus the queries the tools need.

    Wraps `assessment.qbank_engine.QuestionBank` rather than reimplementing
    filtering, so the MCP surface and the HTTP surface answer identically. Any
    divergence between them would be a bug that only appears for one kind of
    consumer.

    That claim used to be false where it mattered most: `search` ran its own
    copy of the whole filter chain, and its keyword haystack was stem + tags +
    `provenance.sourceDocumentId` where the HTTP `_matches` haystack was stem +
    tags. `keyword="CBSE2023"` therefore returned a different set over MCP than
    over HTTP for one bank. Every filter below is now a call into
    `QuestionBank.page`, and the source document is a named parameter rather
    than a term smuggled into one surface's keyword search.

    What remains here is translation, not filtering: MCP's `answer_key_state`
    vocabulary onto the engine's `has_scheme`/`key_provenance`, and the
    per-topic reporting the tools shape for an agent.
    """

    def __init__(self, records: list[dict[str, Any]]):
        self._bank = QuestionBank(records)
        self.records = self._bank.records

    @classmethod
    def load(cls, path: Path | str) -> "Corpus":
        """Load one bank file -- the served bank (Q5), not a union of corpora.

        A missing file is a warning and an empty corpus rather than a crash,
        for the same reason `qbank_routes.init` tolerates it: the bank is a
        data file, and an empty bank reports itself honestly through
        `coverage_report`.

        This used to de-duplicate ids itself (`seen[rid] = rec`, blank ids
        skipped). It no longer does, because `QuestionBank.__init__` does it
        for every surface and logs what it dropped -- doing it twice, in two
        places, with only one of them logging, is how the two surfaces came to
        count differently in the first place.
        """
        p = Path(path)
        if not p.exists():
            log.warning("served bank not found at %s; serving an empty corpus", p)
            return cls([])
        payload = json.loads(p.read_text(encoding="utf-8"))
        records = payload.get("questions") or []
        log.info("loaded %d records from %s", len(records), p)
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def has_answer_key(rec: dict[str, Any]) -> bool:
        """Delegates to the shared engine, so the two cannot drift apart.

        This was a second copy of the rule. Two copies of "what makes a question
        answerable" is exactly how the MCP surface and the HTTP surface start
        disagreeing -- the divergence Q5 exists to prevent -- and the copy here
        did not check the same fields as the builder.
        """
        return has_answer_key(rec)

    def search(
        self,
        *,
        subject: str | None = None,
        grade: int | None = None,
        topic: str | None = None,
        marks: int | None = None,
        min_marks: int | None = None,
        max_marks: int | None = None,
        question_type: str | None = None,
        difficulty: str | None = None,
        keyword: str | None = None,
        source_document_id: str | None = None,
        limit: int = 20,
        require_answer_key: bool = True,
        answer_key_state: str = "any",
    ) -> list[dict[str, Any]]:
        """Filter the bank -- through `QuestionBank.page`, not beside it.

        Every predicate below is the engine's, so a question asked over MCP and
        the same question asked over HTTP select the same records. This method
        used to reimplement the chain, and the copy had drifted: its keyword
        haystack was stem + tags + `provenance.sourceDocumentId` where HTTP's
        was stem + tags, so `keyword="CBSE2023"` answered differently on the
        two surfaces. Selecting by source paper survives as
        `source_document_id`, its own parameter, which is what it always was.

        `answer_key_state`:
          * `"any"` (default) -- require_answer_key is honoured
          * `"official"`      -- only answer-keyed questions whose key is
                                 CBSE's own marking scheme. The HTTP surface's
                                 `?key_provenance=cbse_marking_scheme`, record
                                 for record (Q5).
          * `"none"`          -- only questions with no attached scheme, for
                                 auditing what the relink has not reached

        `"official"` tested the CBSE label ALONE, which let a record carrying
        that label over an empty scheme through -- the 259-record failure the
        content half of the rule exists for, and an asymmetry with HTTP in the
        opposite direction from the one the Q1 pass closed. It now narrows
        within the answer-keyed set rather than sidestepping it.

        There is no "include unanswered" flag that quietly widens the default.
        An agent asking for questions gets answerable ones, and a caller who
        wants the unanswerable set has to say so in those words.

        An unknown `answer_key_state` is an error rather than a no-op: it used
        to fall through every branch, so a typo (`"offical"`) disabled the Q1
        default and handed the caller the unanswerable set.
        """
        if answer_key_state not in ANSWER_KEY_STATES:
            raise ValueError(
                "answer_key_state must be one of "
                + ", ".join(sorted(ANSWER_KEY_STATES))
                + f"; got {answer_key_state!r}")

        # The MCP vocabulary, translated onto the engine's. `"official"` is
        # `?key_provenance=cbse_marking_scheme`, which narrows WITHIN the keyed
        # set -- it never widens past Q1.
        has_scheme: bool | None
        key_provenance: str | None = None
        if answer_key_state == "official":
            has_scheme = True
            key_provenance = "cbse_marking_scheme"
        elif answer_key_state == "none":
            has_scheme = False
        else:
            has_scheme = True if require_answer_key else None

        items, _, _ = self._bank.page(
            subject=subject, grade=grade, topic=topic, marks=marks,
            min_marks=min_marks, max_marks=max_marks, type_=question_type,
            difficulty=difficulty, keyword=keyword,
            source_document_id=source_document_id,
            has_scheme=has_scheme, key_provenance=key_provenance,
            limit=max(0, limit))
        return items

    def topics(self, *, subject: str | None = None, grade: int | None = None,
               require_answer_key: bool = True) -> list[dict[str, Any]]:
        """Every topic in the bank, with its question count and marks spread.

        This is the requirement's "all chapters, and topics inside the chapter,
        with every question mapped to a topic" read back out. A topic with zero
        answerable questions is reported with a zero rather than dropped --
        the gap is the information.
        """
        buckets: dict[tuple[str, int, str], dict[str, Any]] = {}
        for rec in self.records:
            if subject and str(rec.get("subject") or "").lower() != subject.lower():
                continue
            if grade is not None and rec.get("grade") != grade:
                continue
            if require_answer_key and not self.has_answer_key(rec):
                continue
            for topic in _topics_of(rec):
                key = (str(rec.get("subject") or ""), int(rec.get("grade") or 0), topic)
                b = buckets.setdefault(key, {
                    "subject": key[0], "grade": key[1], "topic": topic,
                    "count": 0, "marks": {}, "withAnswerKey": 0,
                })
                b["count"] += 1
                m = str(rec.get("marks") or 0)
                b["marks"][m] = b["marks"].get(m, 0) + 1
                if self.has_answer_key(rec):
                    b["withAnswerKey"] += 1
        out = []
        for b in buckets.values():
            out.append({
                **b,
                "marksDistribution": [
                    {"marks": int(k), "count": v}
                    for k, v in sorted(b["marks"].items(), key=lambda kv: int(kv[0]))
                ],
            })
        out.sort(key=lambda b: (b["subject"], b["grade"], b["topic"]))
        return out

    def coverage(self) -> dict[str, Any]:
        """The honest state of the bank. Reported, never implied.

        Delegates, like `has_answer_key` does: this was a second computation,
        and `GET /v1/coverage` returning a different number from
        `coverage_report` over the same records is precisely the divergence Q5
        exists to prevent.
        """
        return self._bank.coverage()


# `_topics_of` and `_topic_matches` used to live here. They moved into
# `assessment/qbank_engine.py` (as `topics_of`/`topic_matches`) when `?topic=`
# became part of the one shared filter chain -- a topic predicate that only the
# MCP surface could run is how the two surfaces start disagreeing again.
# `_topics_of` is imported above under its old name, because the reporting
# tools below read better with it.


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #

def build_server(corpus: Corpus):
    """Construct the MCP server over a loaded corpus."""
    from mcp.server import MCPServer

    mcp = MCPServer(
        name="academicos-question-bank",
        title="AcademicOS Question Bank",
        version="0.1.0",
        instructions=(
            "Authentic CBSE questions with their official marking schemes. "
            "Every question is mapped to a curriculum topic and carries a mark "
            "value, so you can assemble papers by topic and marks distribution. "
            "Questions without an answer key are excluded by default: an "
            "examination board does not issue a question it cannot mark. Use "
            "coverage_report to see how much of the corpus is answerable."
        ),
    )

    # -- tools ------------------------------------------------------------- #

    @mcp.tool(
        name="search_questions",
        description=(
            "Find answerable CBSE questions by subject, class, topic, marks, "
            "type or keyword. keyword searches the question text and its tags; "
            "use source_document_id to ask for one source paper. Returns "
            "questions WITH their marking schemes by default. Set "
            "answer_key_state='none' to audit which questions have no scheme."
        ),
    )
    def search_questions(
        subject: Optional[str] = None,
        grade: Optional[int] = None,
        topic: Optional[str] = None,
        marks: Optional[int] = None,
        min_marks: Optional[int] = None,
        max_marks: Optional[int] = None,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        keyword: Optional[str] = None,
        source_document_id: Optional[str] = None,
        limit: int = 20,
        require_answer_key: bool = True,
        answer_key_state: str = "any",
    ) -> dict:
        try:
            rows = corpus.search(
                subject=subject, grade=grade, topic=topic, marks=marks,
                min_marks=min_marks, max_marks=max_marks,
                question_type=question_type, difficulty=difficulty,
                keyword=keyword, source_document_id=source_document_id,
                limit=max(1, min(limit, 200)),
                require_answer_key=require_answer_key,
                answer_key_state=answer_key_state,
            )
        except ValueError as exc:
            # An agent that misspells a filter is told so, rather than handed
            # a set it did not ask for.
            return {"error": str(exc)}
        return {"count": len(rows), "questions": [_brief(r) for r in rows]}

    @mcp.tool(
        name="get_question",
        description="One question in full: stem, options, marking scheme, "
                    "provenance (source paper and page), rights, and version.",
    )
    def get_question(question_id: str) -> dict:
        for rec in corpus.records:
            if str(rec.get("id")) == question_id:
                return rec
        return {"error": f"no question {question_id!r}"}

    @mcp.tool(
        name="get_answer_key",
        description="Just the marking scheme for one question: the value points "
                    "and whether they are an official CBSE scheme or absent.",
    )
    def get_answer_key(question_id: str) -> dict:
        for rec in corpus.records:
            if str(rec.get("id")) == question_id:
                scheme = rec.get("answerScheme") or {}
                return {
                    "questionId": question_id,
                    "totalMarks": scheme.get("totalMarks"),
                    "markingPoints": scheme.get("markingPoints") or [],
                    "modelAnswer": scheme.get("modelAnswer") or "",
                    "provenance": scheme.get("provenance") or "none",
                    "sourcePaperCode": scheme.get("sourcePaperCode") or "",
                    "hasAnswerKey": corpus.has_answer_key(rec),
                }
        return {"error": f"no question {question_id!r}"}

    @mcp.tool(
        name="list_topics",
        description="Every curriculum topic in the bank, with its question "
                    "count and marks distribution. Use this before building a "
                    "paper to see what is available.",
    )
    def list_topics(subject: Optional[str] = None, grade: Optional[int] = None,
                    require_answer_key: bool = True) -> dict:
        topics = corpus.topics(subject=subject, grade=grade,
                               require_answer_key=require_answer_key)
        return {"count": len(topics), "topics": topics}

    @mcp.tool(
        name="marks_distribution",
        description=(
            "For one topic, how many questions exist at each mark value -- the "
            "range a paper needs in order to be balanced. Reports which mark "
            "values are missing as well as which exist."
        ),
    )
    def marks_distribution(subject: str, grade: int, topic: str,
                           require_answer_key: bool = True) -> dict:
        rows = corpus.search(subject=subject, grade=grade, topic=topic,
                             limit=1000, require_answer_key=require_answer_key)
        spread: dict[int, int] = {}
        for r in rows:
            m = int(r.get("marks") or 0)
            spread[m] = spread.get(m, 0) + 1
        present = sorted(spread)
        available = set(range(1, 7))
        return {
            "subject": subject, "grade": grade, "topic": topic,
            "total": len(rows),
            "distribution": [{"marks": m, "count": spread[m]} for m in present],
            "missingMarks": sorted(available - set(present)),
            "note": ("missingMarks lists mark values in the 1-6 range the "
                     "curriculum uses that have no question for this topic. A "
                     "paper needing one of them cannot be built from this topic "
                     "alone."),
        }

    @mcp.tool(
        name="coverage_report",
        description="How much of the corpus is answerable, by subject. Use this "
                    "to judge whether the bank can support a paper before "
                    "attempting to build one.",
    )
    def coverage_report() -> dict:
        return corpus.coverage()

    @mcp.tool(
        name="build_paper",
        description=(
            "Assemble a paper from the bank: pick topics, then fill each "
            "section's mark value and question count. Only questions with a "
            "marking scheme are used. Returns the paper plus every gap it could "
            "not fill, rather than silently returning a short paper."
        ),
    )
    def build_paper(
        subject: str,
        grade: int,
        sections: list[dict],
        topic: Optional[str] = None,
    ) -> dict:
        """`sections` is a list of `{"marks": 1, "count": 5}`.

        Paper generation already exists at `POST /api/v1/papers/generate` with
        the full blueprint engine. This is the bank-level view: choose by topic
        and mark value, and be told exactly what could not be filled. It does
        not attempt Bloom or difficulty balancing -- that is the blueprint
        engine's job, and duplicating it here would give two answers to the
        same question.
        """
        chosen: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        used: set[str] = set()

        for section in sections or []:
            want_marks = int(section.get("marks") or 0)
            want_count = int(section.get("count") or 0)
            pool = corpus.search(subject=subject, grade=grade, topic=topic,
                                 marks=want_marks, limit=500)
            picked = 0
            for rec in pool:
                rid = str(rec.get("id"))
                if rid in used:
                    continue
                used.add(rid)
                chosen.append(_brief(rec))
                picked += 1
                if picked >= want_count:
                    break
            if picked < want_count:
                gaps.append({
                    "marks": want_marks, "requested": want_count, "found": picked,
                    "shortfall": want_count - picked,
                })
        total = sum(int(r.get("marks") or 0) for r in chosen)
        return {
            "subject": subject, "grade": grade, "topic": topic,
            "questions": chosen, "questionCount": len(chosen), "totalMarks": total,
            "gaps": gaps,
            "complete": not gaps,
        }

    # -- resources --------------------------------------------------------- #

    @mcp.resource("qbank://catalog", name="Question bank catalog",
                  description="Subjects, classes and answer-key coverage.",
                  mime_type="application/json")
    def catalog() -> str:
        return json.dumps(corpus.coverage(), indent=1)

    @mcp.resource("qbank://topics/{subject}/{grade}",
                  name="Topics for a subject and class",
                  description="Every topic with its question count and marks spread.",
                  mime_type="application/json")
    def topics_resource(subject: str, grade: str) -> str:
        try:
            g = int(grade)
        except ValueError:
            return json.dumps({"error": f"grade {grade!r} is not a number"})
        return json.dumps(corpus.topics(subject=subject, grade=g), indent=1)

    return mcp


def _brief(rec: dict[str, Any]) -> dict[str, Any]:
    """The list projection, including the scheme.

    Unlike the HTTP list, the scheme travels here. An agent assembling a paper
    needs to know the answer is present and what it says, and making it fetch
    each question separately would burn a round trip per question for data it
    always needs next.
    """
    return {
        "id": rec.get("id"),
        "subject": rec.get("subject"),
        "grade": rec.get("grade"),
        "marks": rec.get("marks"),
        "type": rec.get("type"),
        "difficulty": rec.get("difficulty"),
        "stem": rec.get("stem"),
        "topics": _topics_of(rec),
        # The shared rule, not a third inline copy of it: this one counted a
        # marking-points list of empty descriptions as an answer key, so a
        # record the search had already excluded would have been reported
        # answerable if it ever reached here by id.
        "hasAnswerKey": Corpus.has_answer_key(rec),
        "answerScheme": rec.get("answerScheme") or {},
        "provenance": rec.get("provenance") or {},
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def data_root() -> Path:
    """`ACOS_DATA_ROOT`, like every other module resolves it."""
    return Path(os.environ.get("ACOS_DATA_ROOT") or DEFAULT_DATA_ROOT)


def load_corpus(bank: str | None = None) -> Corpus:
    """The served bank, and only the served bank (Q5)."""
    return Corpus.load(Path(bank) if bank else data_root() / SERVED_BANK)


# --------------------------------------------------------------------------- #
# authentication for the network transports
# --------------------------------------------------------------------------- #

def needs_transport_auth(transport: str) -> bool:
    """stdio is local by construction; a network peer must present a key."""
    return transport != "stdio"


class TransportAuth:
    """The HTTP surface's key gate, in front of a network MCP transport.

    Holds an `ApiKeyStore` and defers every decision to
    `qbank_routes.authorize`, so "which keys work, which scopes they need and
    how much quota they spend" has one answer for both surfaces of the
    product rather than two implementations that drift.
    """

    def __init__(self, store: "ApiKeyStore", scope: str = TRANSPORT_SCOPE) -> None:
        self.store = store
        self.scope = scope

    def authenticate(self, headers: Mapping[str, str]) -> "ApiKey":
        """Return the authenticated key, or raise `qbank_routes.AuthFailure`."""
        from ..assessment.qbank_routes import authorize, presented_key

        presented = presented_key(headers.get("authorization"),
                                  headers.get("x-api-key"))
        return authorize(self.store, presented, self.scope)


class ApiKeyMiddleware:
    """Pure-ASGI gate: no request reaches the MCP app without a valid key.

    ASGI rather than the SDK's OAuth `TokenVerifier` because the credential
    this product issues is a scoped API key, not an OAuth token, and the
    refusal must be the same 401/403/429 the HTTP surface gives for the same
    key. Non-HTTP scopes (`lifespan`, `websocket`) pass through untouched --
    the session manager's lifespan has to run for the transport to work at
    all.
    """

    def __init__(self, app, auth: TransportAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope, receive, send) -> None:
        from ..assessment.qbank_routes import AuthFailure

        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        try:
            self.auth.authenticate(headers)
        except AuthFailure as refused:
            await _send_refusal(send, refused)
            return
        await self.app(scope, receive, send)


async def _send_refusal(send, failure) -> None:
    body = json.dumps({"error": failure.detail, "status": failure.status}).encode()
    headers = [(b"content-type", b"application/json"),
               (b"content-length", str(len(body)).encode())]
    headers += [(k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in (failure.headers or {}).items()]
    await send({"type": "http.response.start", "status": failure.status,
                "headers": headers})
    await send({"type": "http.response.body", "body": body})


def transport_app(mcp, auth: TransportAuth, *, transport: str,
                  host: str = "127.0.0.1"):
    """The transport's ASGI app, wrapped in the key gate."""
    app = (mcp.streamable_http_app(host=host) if transport == "streamable-http"
           else mcp.sse_app(host=host))
    return ApiKeyMiddleware(app, auth)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AcademicOS question-bank MCP server")
    ap.add_argument("--transport", default="stdio",
                    choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--bank", default=None, help="a specific bank JSON to serve")
    ap.add_argument("--keys", default=None,
                    help="the API-key store the network transports authenticate "
                         "against (default $ACOS_DATA_ROOT/assessment/api_keys.sqlite)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--list-tools", action="store_true", help="print tools and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    corpus = load_corpus(args.bank)
    mcp = build_server(corpus)

    if args.list_tools:
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        print(f"corpus: {len(corpus)} questions")
        for t in tools:
            print(f"  {t.name:<22} {t.description.splitlines()[0][:70]}")
        return 0

    if not needs_transport_auth(args.transport):
        print(f"serving {len(corpus)} questions over stdio", file=sys.stderr)
        mcp.run(transport="stdio")
        return 0

    # A network transport. Fail closed: an absent key store authenticates
    # nobody, which is the right answer for a port -- the alternative, serving
    # the corpus unauthenticated, is the audit finding this replaces.
    import uvicorn

    from ..assessment.api_keys import ApiKeyStore

    db = Path(args.keys) if args.keys else data_root() / KEY_DB
    if not db.exists():
        log.warning("no API-key store at %s; every network request will be "
                    "refused until a key is minted there", db)
    db.parent.mkdir(parents=True, exist_ok=True)
    auth = TransportAuth(ApiKeyStore(db))
    app = transport_app(mcp, auth, transport=args.transport, host=args.host)
    print(f"serving {len(corpus)} questions over {args.transport} on "
          f"{args.host}:{args.port}, API key required", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
