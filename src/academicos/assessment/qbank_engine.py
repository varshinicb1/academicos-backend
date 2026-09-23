"""The question-bank engine: the corpus, the answer-key rule, the query.

Named `qbank_engine` rather than `question_bank` because that name is taken,
by the module that relinks official answers onto records. This one is the read
side: what the bank will serve, and under which rule.

**This module must never import FastAPI, and that is the whole point of it
existing.** It was carved out of `qbank_routes.py`, which imports `fastapi` at
module scope. Three callers need the engine and only one of them is an HTTP
route:

  * `qbank_routes.py` -- the HTTP surface, an `[api]` install.
  * `mcp/qbank_server.py` -- the MCP server, an `[mcp]` install. Its stdio
    transport has no web framework in it at all, yet `Corpus.__init__` and
    `Corpus.has_answer_key` reached into `qbank_routes` for `QuestionBank`, so
    `pip install -e ".[mcp]"` raised `ImportError: No module named 'fastapi'`
    on the first stdio request. The import was deferred into the method bodies,
    which moves the failure from import time to request time rather than
    removing it.
  * `assessment/pool.py` -- paper generation, a core path with no web
    framework in its own dependency set. It reached for the same class, so a
    non-`[api]` install could no longer build a pool from the baked bank.

Keeping the engine here means the answer-key rule has exactly one definition
(rule Q1) shared by all three surfaces (rule Q5), and the two non-HTTP ones
can import it directly, at module scope, without pulling a web framework in.

`qbank_routes` re-exports `KEY_PROVENANCE`, `QuestionBank`, `encode_cursor` and
`decode_cursor` so existing importers keep working unchanged.
"""
from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

log = logging.getLogger(__name__)

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


class InvalidCursor(ValueError):
    """A cursor that does not decode.

    A plain `ValueError` rather than an `HTTPException` because this module is
    framework-free; `qbank_routes` catches it and returns the 400 the HTTP
    contract promises.
    """


def has_answer_key(rec: dict[str, Any]) -> bool:
    """**The** answer-key rule: a published provenance AND real content.

    This is the single definition rule Q1 is enforced with, on every surface.
    It used to be several, and they disagreed: `_matches` tested the provenance
    label (2,930 records of the live bank), this function tested the content
    (2,941), `pool.PoolQuestion.has_answer` tested `correct_option or
    answer_text`, and `/v1/questions` applied none of them, serving all 3,286
    including 345 with no answer content at all.

    Both halves earn their place:

      * **Content**, because a record can carry
        `provenance: "cbse_marking_scheme"` and hold nothing -- 259 CBE
        multiple-choice records did exactly that before the builder was fixed,
        and a key whose points all carry 0 marks was refused for the same
        reason (41 board records, Task 121).
      * **Provenance**, because content alone would let an unattributed or
        teacher-authored answer be served with the authority of a board
        marking scheme. `KEY_PROVENANCE` names the two that are published keys.
    """
    scheme = rec.get("answerScheme") or {}
    if scheme.get("provenance") not in KEY_PROVENANCE:
        return False
    if (scheme.get("modelAnswer") or "").strip():
        return True
    # Both field names, deliberately. The canonical shape uses `description`,
    # but one builder briefly emitted `text` -- and a gate that knows only one
    # of them filters a whole corpus out while reporting a healthy record
    # count, which is exactly what happened. Accepting either is cheap
    # insurance against the next shape change.
    return any(
        ((pt.get("description") or pt.get("text") or "")).strip()
        for pt in (scheme.get("markingPoints") or [])
    )


def topics_of(rec: dict[str, Any]) -> list[str]:
    """Every topic label a question carries.

    Chapters come from `chapterIds` (the keyword/embedding tagger) and the
    learning-ladder reference from the CBSE CBE import; both are real mappings,
    so both are exposed. Lives here rather than in `mcp/qbank_server.py`
    because `?topic=` is now part of the one filter chain both surfaces run.
    """
    out = [str(c) for c in (rec.get("chapterIds") or []) if c]
    for key in ("topic", "contentCode", "contentReference"):
        v = rec.get(key)
        if v:
            out.append(str(v))
    return list(dict.fromkeys(out)) or ["unmapped"]


def topic_matches(rec: dict[str, Any], topic: str) -> bool:
    """Substring, case-insensitive, against any of a record's topic labels.

    Deliberately looser than `chapter_id`, which is an exact id: an agent asks
    for "Algebra" and means the chapter, the content code and the ladder
    reference alike.
    """
    needle = topic.strip().lower()
    return any(needle in t.lower() for t in topics_of(rec))


def source_document_of(rec: dict[str, Any]) -> str:
    """The id of the document a question was extracted from.

    Its own filter, because it used to be a silent third term in the MCP
    surface's keyword haystack: `keyword=CBSE2023` matched a source document id
    over MCP and nothing over HTTP, which is one keyword with two meanings on
    one corpus. Asking for a source paper is a real need, so it keeps working
    -- by name.
    """
    return str((rec.get("provenance") or {}).get("sourceDocumentId") or "")


def key_provenance_of(rec: dict[str, Any]) -> str:
    """The label on a record's scheme, or `"none"`.

    Reported next to `hasAnswerKey` rather than instead of it: the rule says
    whether the record is servable, the label says which board published it.
    """
    return (rec.get("answerScheme") or {}).get("provenance") or "none"


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
        raise InvalidCursor(
            "the cursor is not decodable; omit it to start again") from exc


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
        unanswerable set has to say so in those words"*), and `coverage_report`
        counts it under Q4. Filtering at construction made both impossible
        while reporting a healthy record count.

        Q1 is enforced where it belongs -- at QUERY time, which is where an
        examination board would enforce it too. Every default path excludes
        records with no answer key -- the MCP's search and build_paper, and
        (since the audit caught it serving all 3,286 records) `/v1/questions`
        and `/v1/subtopics/{id}/questions` too. Asking for them explicitly is
        the audit path, and it stays open.

        The parameter remains so a caller who genuinely wants a pre-filtered
        bank can ask for one; nothing does today.

        **One collection, not two.** `_by_id` was built by a comprehension
        (`{str(r.get("id")): r for r in records if r.get("id")}`) that dropped
        id-less records and kept only the last of a duplicated id, while
        `self.records` kept every one of them. `page()` and `facets()` iterate
        the first, `coverage()` counted the second, so a bank with one repeated
        id made `/v1/coverage` advertise a question `/v1/questions` could not
        serve -- Q5's failure in a new axis, and the MCP's old `Corpus.load`
        (`seen[rid] = rec`, blank ids skipped) used to prevent it. `records` is
        now that same de-duplicated collection, so every count in this class is
        counted over the records it can actually hand over.

        A dropped record is a data bug in the bank file, and load is the only
        place it can be noticed, so it is logged rather than swallowed. It is
        not raised: the bank is a data file, a single bad record must not take
        the API down with it, and the honest response is to serve the rest and
        say what was dropped.
        """
        if require_answer_key:
            records = [r for r in records if self.has_answer_key(r)]

        by_id: dict[str, dict[str, Any]] = {}
        without_id = 0
        duplicated: list[str] = []
        for rec in records:
            rid = str(rec.get("id") or "").strip()
            if not rid:
                without_id += 1
                continue
            if rid in by_id:
                duplicated.append(rid)
            # Last one wins, which is what the comprehension and the MCP's old
            # loader both did; the point of this rewrite is that it is now
            # counted, not that it resolves differently.
            by_id[rid] = rec
        if without_id:
            log.warning("question bank: dropped %d record(s) with no id", without_id)
        if duplicated:
            log.warning(
                "question bank: %d duplicate id(s), keeping the last of each "
                "(first few: %s)", len(duplicated), ", ".join(duplicated[:5]))

        self._by_id = by_id
        self.records = list(by_id.values())
        self._sorted_ids = sorted(by_id)

    # The module-level rule, bound here so the many existing
    # `QuestionBank.has_answer_key(rec)` call sites keep reading the way they
    # did. One function, one behaviour, two spellings of the same name.
    has_answer_key = staticmethod(has_answer_key)

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
        min_marks: int | None = None,
        max_marks: int | None = None,
        type_: str | None = None,
        difficulty: str | None = None,
        bloom: str | None = None,
        chapter_id: str | None = None,
        subtopic_id: str | None = None,
        topic: str | None = None,
        source_document_id: str | None = None,
        has_scheme: bool | None = None,
        key_provenance: str | None = None,
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
                            min_marks=min_marks, max_marks=max_marks,
                            type_=type_, difficulty=difficulty, bloom=bloom,
                            chapter_id=chapter_id, subtopic_id=subtopic_id,
                            topic=topic, source_document_id=source_document_id,
                            has_scheme=has_scheme,
                            key_provenance=key_provenance,
                            review_state=review_state, needle=needle):
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
                marks=filters.get("marks"),
                min_marks=filters.get("min_marks"),
                max_marks=filters.get("max_marks"),
                type_=filters.get("type_"),
                difficulty=filters.get("difficulty"), bloom=filters.get("bloom"),
                chapter_id=filters.get("chapter_id"),
                subtopic_id=filters.get("subtopic_id"),
                topic=filters.get("topic"),
                source_document_id=filters.get("source_document_id"),
                has_scheme=filters.get("has_scheme"),
                key_provenance=filters.get("key_provenance"),
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

        **Every count here is a set the API can actually hand over.**
        `withOfficialCbseScheme` used to count the CBSE label alone, which
        included records carrying that label over an empty scheme -- a number
        no query could return, because every serving path applies
        `has_answer_key`. It now counts keyed records whose key is CBSE's, which
        is exactly what `?key_provenance=cbse_marking_scheme` returns.

        Counted over `_by_id.values()` -- the collection `page()` and
        `facets()` serve from -- and not over a second list that might hold
        records those two cannot reach. `__init__` makes `self.records` the
        same collection; this spells out which one is authoritative, so the
        two cannot drift apart again if that ever changes.
        """
        served = list(self._by_id.values())
        total = len(served)
        with_key = sum(1 for r in served if has_answer_key(r))
        official = sum(1 for r in served
                       if has_answer_key(r)
                       and key_provenance_of(r) == "cbse_marking_scheme")

        by_subject: dict[str, dict[str, int]] = {}
        by_pair: dict[tuple[str, int], dict[str, Any]] = {}
        for r in served:
            subject = str(r.get("subject") or "unknown")
            grade = int(r.get("grade") or 0)
            keyed = has_answer_key(r)

            d = by_subject.setdefault(subject, {"total": 0, "withAnswerKey": 0})
            d["total"] += 1
            row = by_pair.setdefault((subject, grade), {
                "subject": subject, "grade": grade,
                "questions": 0, "withAnswerKey": 0, "marksTotal": 0,
                "_marks": set(),
            })
            row["questions"] += 1
            if keyed:
                d["withAnswerKey"] += 1
                row["withAnswerKey"] += 1
                # Marks a paper can actually be built from, so only the keyed
                # records count: a 3-mark question that cannot be marked does
                # not make 3 marks available.
                row["_marks"].add(int(r.get("marks") or 0))
                row["marksTotal"] += int(r.get("marks") or 0)

        # `markValues`, not `marksAvailable`. The field holds the distinct mark
        # DENOMINATIONS this subject and class offer -- `[8, 9]` for Assamese
        # class 12, over two questions -- and on a coverage endpoint
        # "marksAvailable" reads as "how many marks I can build a paper from",
        # which is the number a buyer sizes a purchase on. That number is
        # `marksTotal`, and it is now reported next to it rather than left to be
        # misread off the other one.
        by_subject_class = [
            {"subject": row["subject"], "grade": row["grade"],
             "questions": row["questions"], "withAnswerKey": row["withAnswerKey"],
             "markValues": sorted(row["_marks"]),
             "marksTotal": row["marksTotal"]}
            for _, row in sorted(by_pair.items())
        ]
        return {
            "questions": total,
            "withAnswerKey": with_key,
            "withOfficialCbseScheme": official,
            "answerKeyCoverage": round(with_key / total, 4) if total else 0.0,
            "subjects": dict(sorted(by_subject.items())),
            "grades": sorted({int(r.get("grade") or 0) for r in served}),
            "bySubjectClass": by_subject_class,
            "note": ("Questions without an answer key are excluded from every "
                     "search by default. They exist in the corpus and are "
                     "reported here rather than hidden. Per subject and class, "
                     "markValues lists the distinct mark denominations on offer "
                     "and marksTotal is the marks those answer-keyed questions "
                     "add up to."),
        }


def _matches(rec: dict[str, Any], *, subject, grade, marks, type_, difficulty,
             bloom, chapter_id, subtopic_id, has_scheme, review_state,
             needle: str, key_provenance: str | None = None,
             min_marks: int | None = None, max_marks: int | None = None,
             topic: str | None = None,
             source_document_id: str | None = None) -> bool:
    """**The** filter chain, for every surface.

    `min_marks`, `max_marks`, `topic` and `source_document_id` are here rather
    than in the MCP server because `Corpus.search` used to run its own copy of
    this function, and a copy is how two surfaces start answering one question
    differently -- which they did, on `keyword`. Only some of these are exposed
    as HTTP query parameters; the chain is shared whether or not both surfaces
    spell every filter.
    """
    if subject and str(rec.get("subject") or "").lower() != subject.lower():
        return False
    if grade is not None and rec.get("grade") != grade:
        return False
    if marks is not None and rec.get("marks") != marks:
        return False
    if min_marks is not None and int(rec.get("marks") or 0) < min_marks:
        return False
    if max_marks is not None and int(rec.get("marks") or 0) > max_marks:
        return False
    if topic and not topic_matches(rec, topic):
        return False
    if source_document_id:
        if source_document_id.lower() not in source_document_of(rec).lower():
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
        if has_answer_key(rec) != bool(has_scheme):
            return False
    if key_provenance:
        # Which KIND of published key, narrowing WITHIN the keyed set -- never
        # widening past Q1. `has_scheme=true` used to mean "official CBSE
        # scheme" and is now the general rule, so this is the only way left to
        # ask for the board's own schemes; without it the API reported
        # `withOfficialCbseScheme` for a set no request could select.
        if not has_answer_key(rec):
            return False
        if key_provenance_of(rec) != key_provenance:
            return False
    if needle:
        haystack = " ".join([
            str(rec.get("stem") or ""),
            " ".join(str(t) for t in (rec.get("tags") or [])),
        ]).lower()
        if needle not in haystack:
            return False
    return True
