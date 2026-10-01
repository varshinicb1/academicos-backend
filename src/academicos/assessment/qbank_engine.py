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
from dataclasses import dataclass
from typing import Any, Callable, Collection

from .chapter_filing import UNMAPPED, ChapterFiling, name_key

log = logging.getLogger(__name__)

# Hard ceiling on page size. A consumer asking for 10,000 items is either
# scraping or has misunderstood; either way the answer is no, not 10,000.
MAX_LIMIT = 200
DEFAULT_LIMIT = 25

# The provenances that make a scheme a PUBLISHED answer key: CBSE's own marking
# scheme, NCERT's answer from an Exemplar book (4,192 of the 5,427 served
# records), and an answer a textbook prints (its answers page).
PUBLISHED_PROVENANCE: frozenset[str] = frozenset({
    "cbse_marking_scheme",
    "ncert_exemplar_answer",
    "ncert_textbook_answer",
})
# Answers no board published but that were CHECKED before they were served
# (corpus/textbook.py): every value point quoting the chapter, the quote found
# in the chapter's own text; a worked answer two independent solves agree on;
# an answer a teacher of the school reviewed and approved. Classes 1-5 have no
# published keys at all, so without this tier they could have no bank.
# Everything else -- a teacher's own answer not yet reviewed, a sandbox
# fabrication, an unlabelled scheme -- is not a key. Every surface reports
# which tier a key is (`key_tier`), and `?key_provenance=` narrows to one.
CHECKED_PROVENANCE: frozenset[str] = frozenset({
    "textbook_grounded",
    "two_model_solved",
    "teacher_verified",
})
KEY_PROVENANCE: frozenset[str] = PUBLISHED_PROVENANCE | CHECKED_PROVENANCE

# The trust labels a served record can carry (`key_tier`): QB-5's "trust
# label", one word per record. `"none"` is deliberately absent -- a record with
# no answer key is not served by default (rule Q1), and asking for it is what
# `has_scheme=false` is for, so `?trust=none` would be a second spelling of an
# audit path rather than a level of trust.
TRUST_LEVELS: frozenset[str] = frozenset({"published", "checked"})


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


def topics_of(rec: dict[str, Any], chapter: str | None = None) -> list[str]:
    """Every topic label a question carries.

    `chapter` is the chapter the record is filed under
    (`QuestionBank.chapter_of`), and every caller with the bank to hand passes
    it. Without it the labels are the record's raw `chapterIds`, which for six
    of the ten class 6-10 Mathematics/Science pairs name a chapter of the
    pre-2024 book: MCP listed 631 of 857 Mathematics 8 questions as "unmapped"
    while the builder filed 491 of them under a chapter (audit D10, D39).
    The learning-ladder reference from the CBSE CBE import is a real mapping
    too, so it is exposed beside the chapter. Lives here rather than in
    `mcp/qbank_server.py` because `?topic=` is part of the one filter chain
    both surfaces run.
    """
    if chapter:
        out = [chapter]
    else:
        out = [str(c) for c in (rec.get("chapterIds") or []) if c]
    for key in ("topic", "contentCode", "contentReference"):
        v = rec.get(key)
        if v:
            out.append(str(v))
    return list(dict.fromkeys(out)) or [UNMAPPED]


def topic_matches(rec: dict[str, Any], topic: str, chapter: str | None = None) -> bool:
    """Substring, case-insensitive, against any of a record's topic labels and
    its raw `chapterIds`.

    The loose end of `QuestionBank`'s topic filter, reached only when the topic
    is neither a chapter of the class nor a label it lists: an agent asks for
    "Algebra" and means the chapter, the content code and the ladder reference
    alike. The raw ids stay in the haystack so an old slug a caller saved keeps
    finding the questions that carry it.
    """
    needle = topic.strip().lower()
    labels = topics_of(rec, chapter) + [str(c) for c in (rec.get("chapterIds") or []) if c]
    return any(needle in t.lower() for t in labels)


def _class_key(rec: dict[str, Any]) -> tuple[str, int]:
    return str(rec.get("subject") or ""), int(rec.get("grade") or 0)


def _chapter_claims(rec: dict[str, Any]) -> tuple[str | None, list[str]]:
    """(taxonomy tag, own chapter ids): what `ChapterFiling` files a record by."""
    return rec.get("taxonomyChapterId"), [str(c) for c in (rec.get("chapterIds") or []) if c]


@dataclass(frozen=True)
class _ClassChapters:
    """One class's chapters, filed the way the builder's picker files them.

    `ids` is every chapter the picker can list for the class -- the syllabus's
    and those its questions are filed under -- keyed lower-case, so a caller
    need not match the slug's case. `names` maps a chapter's printed name to
    its id, which is how an agent asks ("Life Processes", audit D10).
    `labels` is every topic label the class's questions carry, lower-cased.
    """
    filing: ChapterFiling
    filed: dict[str, str]
    ids: dict[str, str]
    names: dict[str, str]
    labels: frozenset[str]

    @classmethod
    def build(cls, subject: str, grade: int,
              records: list[dict[str, Any]]) -> "_ClassChapters":
        try:
            filing = ChapterFiling.for_class(subject, grade)
        except (OSError, ValueError):
            # A subject whose name cannot be a file name has no syllabus; with
            # nothing known, the filing keeps each record's own first claim.
            filing = ChapterFiling(known=frozenset())
        filed = {str(r.get("id")): filing.chapter_of(*_chapter_claims(r)) for r in records}
        chapters = set(filing.known) | set(filed.values())
        names = dict(filing.by_name)
        for cid in sorted(chapters - set(filing.known) - {UNMAPPED}):
            # The picker's name for a chapter the syllabus does not list.
            printed = filing.book_names.get(cid) or cid.replace("-", " ").title()
            names.setdefault(name_key(printed), cid)
        labels = frozenset(t.lower() for r in records
                           for t in topics_of(r, filed[str(r.get("id"))]))
        return cls(filing=filing, filed=filed,
                   ids={c.lower(): c for c in chapters}, names=names, labels=labels)


def source_document_of(rec: dict[str, Any]) -> str:
    """The id of the document a question was extracted from.

    Its own filter, because it used to be a silent third term in the MCP
    surface's keyword haystack: `keyword=CBSE2023` matched a source document id
    over MCP and nothing over HTTP, which is one keyword with two meanings on
    one corpus. Asking for a source paper is a real need, so it keeps working
    -- by name.
    """
    return str((rec.get("provenance") or {}).get("sourceDocumentId") or "")


def key_tier(rec: dict[str, Any]) -> str:
    """"published", "checked", or "none" -- how far a record's key can be
    trusted, reported next to the key on every surface."""
    p = key_provenance_of(rec)
    return "published" if p in PUBLISHED_PROVENANCE else "checked" if p in CHECKED_PROVENANCE else "none"


def key_provenance_of(rec: dict[str, Any]) -> str:
    """The label on a record's scheme, or `"none"`.

    Reported next to `hasAnswerKey` rather than instead of it: the rule says
    whether the record is servable, the label says which board published it.
    """
    return (rec.get("answerScheme") or {}).get("provenance") or "none"


def trust_of(rec: dict[str, Any]) -> str:
    """The trust label a surface reports and `?trust=` selects on.

    `key_tier` reads the label alone, so a record carrying CBSE's label over an
    empty scheme would read "published" -- the 259-record failure again. The
    label counts only when the record has a key at all, which is what the list
    projection always reported; the filter and the facet now say the same.
    """
    return key_tier(rec) if has_answer_key(rec) else "none"


def book_of(rec: dict[str, Any]) -> str:
    """The publication a question was printed in, as its rights record names it.

    "NCERT, Exemplar Problems, Class X Science", "Central Board of Secondary
    Education, sample question paper and marking scheme", and so on: every
    record carries `rights.attribution`, and it is the only field that names
    the book rather than the chapter. The NCERT textbook a chapter belongs to
    is not on the record (it would come from `ncert_books.json`, which the
    served tree does not carry), so `?book=` selects the book the question
    came FROM; `?chapter_id=` already selects the book chapter it is filed
    under.
    """
    return " ".join(str((rec.get("rights") or {}).get("attribution") or "").split())


def _folded(value: Any) -> str:
    """Case and runs of whitespace ignored, so a book name pasted from a facet
    or typed by hand compares equal to the one on the record."""
    return " ".join(str(value or "").split()).lower()


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
        # Built on the first chapter or topic question, not here: the pool
        # builds a bank to read answer keys and never asks one.
        self._classes: dict[tuple[str, int], _ClassChapters] | None = None

    # The module-level rule, bound here so the many existing
    # `QuestionBank.has_answer_key(rec)` call sites keep reading the way they
    # did. One function, one behaviour, two spellings of the same name.
    has_answer_key = staticmethod(has_answer_key)

    def __len__(self) -> int:
        return len(self.records)

    def get(self, question_id: str) -> dict[str, Any] | None:
        return self._by_id.get(question_id)

    # -- chapters: the builder's filing, not the record's raw ids (D39) ------

    def _chapter_index(self) -> dict[tuple[str, int], _ClassChapters]:
        if self._classes is None:
            groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
            for rec in self._by_id.values():
                groups.setdefault(_class_key(rec), []).append(rec)
            self._classes = {key: _ClassChapters.build(key[0], key[1], recs)
                             for key, recs in groups.items()}
        return self._classes

    def chapter_of(self, rec: dict[str, Any]) -> str:
        """The chapter the builder's picker files this record under.

        `assessment/chapter_filing.py` is the rule; this is the bank applying
        it, so `/v1`, the MCP server and the paper builder name one chapter
        for one question. Until 2026-09-28 the first two matched raw
        `chapterIds` and reached 1,266 of the 2,297 questions the builder
        draws from the 124 named class 6-10 Mathematics/Science chapters
        (audit D39).
        """
        cls = self._chapter_index().get(_class_key(rec))
        if cls is not None and str(rec.get("id")) in cls.filed:
            return cls.filed[str(rec.get("id"))]
        return ChapterFiling(known=frozenset()).chapter_of(*_chapter_claims(rec))

    def topics_of(self, rec: dict[str, Any]) -> list[str]:
        """`topics_of`, with the chapter the builder files the record under."""
        return topics_of(rec, self.chapter_of(rec))

    def _in_chapter(self, chapter_id: str) -> Callable[[dict[str, Any]], bool]:
        """`?chapter_id=`: in the chapter exactly when the builder would draw
        it for a teacher who chose that chapter -- `ChapterFiling.selects`,
        including its allowance for a pre-2024 slug a caller saved."""
        tests = {key: c.filing.selects([chapter_id], set(c.filed.values()))
                 for key, c in self._chapter_index().items()}
        return lambda rec: tests[_class_key(rec)](*_chapter_claims(rec))

    def _on_topic(self, topic: str) -> Callable[[dict[str, Any]], bool]:
        """`topic`, read in the record's own class, first match wins:

          1. a chapter id of the class -- exactly that chapter, as the builder
             files it;
          2. a label the class's questions carry -- exactly that label, so
             `list_topics` and a search for what it listed agree ("10.1.1" was
             listed at 1 and searched at 6, because "10.1.10" and "10.1.11"
             contain it);
          3. a chapter's printed name -- that chapter ("Life Processes" found 0
             while "life-processes" found 69);
          4. anything else -- a substring of the labels and the raw ids, the
             deliberately loose search this has always been.
        """
        needle = topic.strip().lower()
        key = name_key(topic)
        rules: dict[tuple[str, int], tuple[str, str | None]] = {}
        for ck, c in self._chapter_index().items():
            if needle in c.ids:
                rules[ck] = ("chapter", c.ids[needle])
            elif needle in c.labels:
                rules[ck] = ("label", None)
            elif key and key in c.names:
                rules[ck] = ("chapter", c.names[key])
            else:
                rules[ck] = ("substring", None)

        def test(rec: dict[str, Any]) -> bool:
            how, chapter = rules[_class_key(rec)]
            if how == "chapter":
                return self.chapter_of(rec) == chapter
            if how == "label":
                return any(t.lower() == needle for t in self.topics_of(rec))
            return topic_matches(rec, topic, self.chapter_of(rec))

        return test

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
        topic_id: str | None = None,
        competency: str | None = None,
        source: str | None = None,
        trust: str | None = None,
        book: str | None = None,
        grades: Collection[int] | None = None,
        subjects: Collection[str] | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        """One keyset page. Returns `(items, next_cursor, total_matching)`.

        Ordered by `id`, which is stable and unique, so a cursor is simply "the
        last id I sent you" and paging cannot skip or duplicate a row when the
        bank is edited between requests.

        `grades` and `subjects` are sets, not one value: they are an API key's
        limits (`ApiKey.grades`, `.subjects`), applied under whatever single
        `grade` or `subject` the caller asked for.
        """
        after = decode_cursor(cursor) if cursor else None
        needle = (keyword or "").strip().lower()
        in_chapter = self._in_chapter(chapter_id) if chapter_id else None
        on_topic = self._on_topic(topic) if topic else None

        items: list[dict[str, Any]] = []
        total = 0
        last_included: str | None = None
        more_available = False

        for qid in self._sorted_ids:
            rec = self._by_id[qid]
            if not _matches(rec, subject=subject, grade=grade, marks=marks,
                            min_marks=min_marks, max_marks=max_marks,
                            type_=type_, difficulty=difficulty, bloom=bloom,
                            in_chapter=in_chapter, subtopic_id=subtopic_id,
                            on_topic=on_topic, source_document_id=source_document_id,
                            has_scheme=has_scheme,
                            key_provenance=key_provenance,
                            review_state=review_state, needle=needle,
                            topic_id=topic_id, competency=competency,
                            source=source, trust=trust, book=book,
                            grades=grades, subjects=subjects):
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
            # The vocabulary of the three open-ended filters added with
            # `?source=`, `?book=` and `?trust=`: a consumer cannot guess a
            # book's exact name, so the facet is where it is read from.
            "source": {}, "book": {}, "keyTier": {},
        }
        chapter_id, topic = filters.get("chapter_id"), filters.get("topic")
        in_chapter = self._in_chapter(chapter_id) if chapter_id else None
        on_topic = self._on_topic(topic) if topic else None
        for rec in self._by_id.values():
            if not _matches(
                rec,
                subject=filters.get("subject"), grade=filters.get("grade"),
                marks=filters.get("marks"),
                min_marks=filters.get("min_marks"),
                max_marks=filters.get("max_marks"),
                type_=filters.get("type_"),
                difficulty=filters.get("difficulty"), bloom=filters.get("bloom"),
                in_chapter=in_chapter,
                subtopic_id=filters.get("subtopic_id"),
                on_topic=on_topic,
                source_document_id=filters.get("source_document_id"),
                has_scheme=filters.get("has_scheme"),
                key_provenance=filters.get("key_provenance"),
                review_state=filters.get("review_state"),
                needle=(filters.get("keyword") or "").strip().lower(),
                topic_id=filters.get("topic_id"),
                competency=filters.get("competency"),
                source=filters.get("source"), trust=filters.get("trust"),
                book=filters.get("book"),
                grades=filters.get("grades"), subjects=filters.get("subjects"),
            ):
                continue
            for key, value in (
                ("subject", rec.get("subject")), ("grade", rec.get("grade")),
                ("marks", rec.get("marks")), ("type", rec.get("type")),
                ("difficulty", rec.get("difficulty")),
                ("bloomLevel", rec.get("bloomLevel")),
                ("reviewState", rec.get("reviewState")),
                ("source", rec.get("source")), ("book", book_of(rec) or None),
                ("keyTier", trust_of(rec)),
            ):
                if value is None:
                    continue
                counts[key][str(value)] = counts[key].get(str(value), 0) + 1
        return {
            k: [{"value": v, "count": c}
                for v, c in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]
            for k, d in counts.items()
        }

    def coverage(self, *, grades: Collection[int] | None = None,
                 subjects: Collection[str] | None = None) -> dict[str, Any]:
        """The honest state of the bank, reported rather than implied.

        `grades` and `subjects` are an API key's limits: a key limited to
        class 10 Science is told the coverage of class 10 Science, not of the
        bank it may not read. The MCP's `coverage_report` passes neither, so
        Q5's "the same numbers on both surfaces" holds for every unlimited key.

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
        served = [r for r in self._by_id.values()
                  if _within(r, grades=grades, subjects=subjects)]
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


def _within(rec: dict[str, Any], *, grades: Collection[int] | None,
            subjects: Collection[str] | None) -> bool:
    """Inside an API key's limits. `subjects` arrives lower-cased.

    A limit that is set refuses a record that lacks the field: a record with
    no grade is not "in class 10", and treating it as inside would hand a
    limited key exactly the records nobody could place.
    """
    if grades is not None and rec.get("grade") not in grades:
        return False
    if subjects is not None and str(rec.get("subject") or "").lower() not in subjects:
        return False
    return True


def _matches(rec: dict[str, Any], *, subject, grade, marks, type_, difficulty,
             bloom, in_chapter: Callable[[dict[str, Any]], bool] | None,
             subtopic_id, has_scheme, review_state,
             needle: str, key_provenance: str | None = None,
             min_marks: int | None = None, max_marks: int | None = None,
             on_topic: Callable[[dict[str, Any]], bool] | None = None,
             source_document_id: str | None = None,
             topic_id: str | None = None, competency: str | None = None,
             source: str | None = None, trust: str | None = None,
             book: str | None = None,
             grades: Collection[int] | None = None,
             subjects: Collection[str] | None = None) -> bool:
    """**The** filter chain, for every surface.

    `min_marks`, `max_marks`, `topic` and `source_document_id` are here rather
    than in the MCP server because `Corpus.search` used to run its own copy of
    this function, and a copy is how two surfaces start answering one question
    differently -- which they did, on `keyword`. Only some of these are exposed
    as HTTP query parameters; the chain is shared whether or not both surfaces
    spell every filter.

    The chapter and the topic arrive as tests the bank built
    (`QuestionBank._in_chapter`, `._on_topic`), because both depend on the
    class's chapter filing, which a single record cannot see.

    `topic_id`, `competency`, `source`, `trust` and `book` are API-1's filters
    over fields every record already carries (`topicIds`, `competencyIds`,
    `source`, the key's tier, `rights.attribution`). Before they were here,
    `/v1/questions?book=...` dropped the parameter and answered with the whole
    class (audit N-67-12). `topic_id` and `competency` are exact ids, like
    `subtopic_id`: `topic` above is the deliberately loose search, and one name
    meaning two strictness levels is how "10.1.1" came to match "10.1.10".
    """
    if not _within(rec, grades=grades, subjects=subjects):
        return False
    if subject and str(rec.get("subject") or "").lower() != subject.lower():
        return False
    if grade is not None and rec.get("grade") != grade:
        return False
    if topic_id and topic_id not in (rec.get("topicIds") or []):
        return False
    if competency and competency not in (rec.get("competencyIds") or []):
        return False
    if source and str(rec.get("source") or "").lower() != source.lower():
        return False
    if trust and trust_of(rec) != trust:
        return False
    if book and _folded(book_of(rec)) != _folded(book):
        return False
    if marks is not None and rec.get("marks") != marks:
        return False
    if min_marks is not None and int(rec.get("marks") or 0) < min_marks:
        return False
    if max_marks is not None and int(rec.get("marks") or 0) > max_marks:
        return False
    if on_topic is not None and not on_topic(rec):
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
    if in_chapter is not None and not in_chapter(rec):
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
