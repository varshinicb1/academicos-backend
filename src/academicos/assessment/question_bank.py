"""Question bank: the enriched record, the answer-key relink, and the
curriculum-to-question surface.

Why this module exists
----------------------
The bank holds 3,286 questions and only 1,093 of them (33.3%) carried a marking
scheme, which is the half `docs/question-bank-api.md` calls the moat. That
document concluded the answers were scarce.

They are not scarce. They are **unlinked**. A separate store,
`academicos-data/answer_keys.db`, holds 4,483 official answers with real value
points, and 88.6% of those rows sit under a filename carrying the CBSE paper
code. The questions carry a *content hash* of their source PDF and a question
number, and nothing joined the two.

The link, verified end to end against real files:

    question id      cbse:q:src:af50a3f7….…:4
    parse tree       academicos-data/parse/src_af50a3f7….….json
                     -> page headers carry the paper code 31/1/1
    answer key       answer_keys.db, source "X_086_31-1-1 to 3 Science_MS.pdf"
                     -> KeyedAnswer for question 4

So this module walks that chain, attaches the official scheme where it exists,
and records the provenance that makes the attachment checkable. It never
invents an answer: a question with no matching scheme is left without one and
reported, not filled with a guess.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from ..corpus.scheme_verify import (
    AGREE_SHARE,
    CONFLICTING_KEYS,
    OUTRANKED,
    SIBLING_SHARE,
    Verdict,
    idf_weights,
    join_score,
    join_terms,
    join_verdict,
    medium_of_text,
    norm_of,
    outright_option,
    question_medium,
    question_options,
    row_text,
    share,
    verify,
)
from .answer_key import _SUBJECT_BY_CODE, normalize_paper_code
from .marking import parse_options
from .schemas import (
    AnswerSchemeSchema,
    CalibrationSchema,
    MarkingPointSchema,
    ProvenanceSchema,
    RightsSchema,
    RubricLevelSchema,
)

log = logging.getLogger(__name__)

# Recover the hash and question number back out of a canonical question id.
# `cbse:q:src:<24 hex>:<q_no>` is produced by `pool.py`; this is its inverse.
_QUESTION_ID = re.compile(r"^[a-z]+:q:src:([0-9a-f]{8,}):(\d+)$")

# A CBSE marking scheme's running header prints the code; `normalize_paper_code`
# handles the separators. This pattern is only used to decide whether a page or
# a filename is a *candidate* carrier, so it stays deliberately loose -- but it
# must accept every separator the corpus uses (`.`, `_`, `-`, `/`), because
# gating on a narrower set silently drops whole papers.
#
# The series number is one to three digits. Science and Mathematics lead with
# two (`31/1/1`), but English leads with one (`1/4/1`, `2/5/1`) -- requiring two
# skipped every English paper in the corpus, which is why all 207 English
# questions reached the bank with no marking scheme. The year guard is the
# lookbehind, not the digit count: `2025-1-1` still yields nothing because the
# `2` is preceded by no separator and `025` is preceded by a digit.
_CODE_HINT = re.compile(
    r"(?<!\d)\d{1,3}\s*[/\-._]\s*\d\s*[/\-._]\s*\d(?!\d)")


def parse_question_id(qid: str) -> tuple[str, int] | None:
    """`cbse:q:src:<hash>:<q_no>` -> `(hash, q_no)`, or None if it is not that shape."""
    m = _QUESTION_ID.match((qid or "").strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _collapse(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Applied to whole page texts, not just to search probes. Without it a probe
    containing a line break never matched the page text it came from.
    """
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# source documents
# --------------------------------------------------------------------------- #

@dataclass
class ParsedSource:
    """One parsed source document, with the paper code found in its headers."""

    document_id: str
    doc_type: str = ""
    paper_code: str = ""
    page_count: int = 0
    _pages: list[tuple[int, str]] = field(default_factory=list, repr=False)

    def page_containing(self, needle: str, *, window: int = 60) -> int | None:
        """First page whose text contains the first `window` characters of `needle`.

        Used for provenance only. A miss is not an error -- it means the stem
        was reassembled across a page break, and returning a wrong page would
        be worse than returning none.

        Whitespace is collapsed on BOTH sides. Normalising only the probe, as
        the first version did, matched almost nothing: the page text keeps the
        PDF's line breaks, so a 60-character probe containing a newline never
        appeared in it verbatim and every page came back as None.
        """
        probe = " ".join((needle or "").split())[:window]
        if len(probe) < 24:
            return None
        for page_no, text in self._pages:
            if probe in text:
                return page_no
        return None


def load_parsed_source(parse_dir: Path, doc_hash: str) -> ParsedSource | None:
    """Read `parse/src_<hash>.json` and find its paper code.

    The code lives in the running header, so only the first few hundred
    characters of each page are scanned: a code appearing mid-page is far more
    likely to be a question number than the paper identifier.
    """
    path = Path(parse_dir) / f"src_{doc_hash}.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("unreadable parse tree: %s", path)
        return None

    pages: list[tuple[int, str]] = []
    code = ""
    for page in doc.get("pages") or []:
        page_no = int(page.get("page_no") or len(pages) + 1)
        text = " ".join(
            str(page.get(k) or "") for k in ("text", "content", "body")
        ).strip()
        if text:
            pages.append((page_no, _collapse(text)))
            if not code:
                head = text[:300]
                if _CODE_HINT.search(head):
                    code = normalize_paper_code(head)

    return ParsedSource(
        document_id=str(doc.get("document_id") or f"src:{doc_hash}"),
        doc_type=str(doc.get("doc_type") or ""),
        paper_code=code,
        page_count=len(pages),
        _pages=pages,
    )


# --------------------------------------------------------------------------- #
# answer keys
# --------------------------------------------------------------------------- #

@dataclass
class OfficialAnswer:
    q_no: int
    correct_option: str | None
    answer_text: str
    value_points: list[tuple[str, float]] = field(default_factory=list)
    marks: float = 0.0
    # The answer_keys.db row's source file, and the medium of the scheme it
    # came from ("en", "hi", or "" when neither the filename nor the text
    # says). One key often holds both media of the same paper.
    source: str = ""
    medium: str = ""
    # The paper code the row is filed under. The content join draws on every
    # set of a series, so the row's own code can differ from the question's.
    code: str = ""

    def text(self) -> str:
        """The answer text and every value point it does not already contain.

        answer_keys.db usually stores the points joined as the answer text as
        well; judging "X X" instead of "X" would double every word count.
        """
        parts = [self.answer_text.strip()] if self.answer_text.strip() else []
        for point, _ in self.value_points:
            if point and not any(point in p for p in parts):
                parts.append(point)
        return " ".join(parts)


# A scheme's medium, from its filename: "55-1-1,2,3 Hindi Version.pdf",
# "XII_042_..._hindi_med.pdf", "32-1-1 H.pdf", "MS-087-32-3-1-E.pdf",
# "XII_029_MS_64-1-1  ENG UNSIGNED.pdf".
_HINDI_NAME = re.compile(r"(?i:hindi)|(?:^|[\s_\-])H(?=[\s_\-.]|$)|\d\s?H(?=\.pdf$)")
_ENGLISH_NAME = re.compile(r"(?i:english|\beng\b|_eng)|(?:^|[\s_\-])E(?=[\s_\-.]|$)|\dE(?=\.pdf$)")


def answer_medium(source: str, text: str) -> str:
    """'hi' or 'en' from the filename, failing that from the script of the text."""
    name = Path(source or "").name
    if _HINDI_NAME.search(name):
        return "hi"
    if _ENGLISH_NAME.search(name):
        return "en"
    return medium_of_text(text)


def verify_answer(record: dict[str, Any], answer: OfficialAnswer) -> Verdict:
    """`scheme_verify.verify` for one index candidate."""
    return verify(record, answer.text(), option=answer.correct_option)


# Reasons the relink adds to `scheme_verify.REASONS`. Those judge one answer
# against its question; `scheme_verify.JOIN_REASONS` (near-tie,
# conflicting-keys, unanchored, not-an-option, outranked, other-number) judge
# the content join's pick and the rows under the question's own number it passed over;
# this one judges what the index says about a scheme the record already has.
# A scheme for which the index holds no accepted row. An objective one cannot
# be checked at all: `build_official_scheme` wrote the option's own text as
# its model answer, so the key "verifies" against itself. A descriptive one
# the verifier accepts is on the question's topic, which is not the board's
# answer: since Task 151 (user decision 2026-09-22, correctness first) an
# unverified scheme is not served at all.
UNBACKED_KEY = "unbacked-key"
# A scheme without the official label that verifies but names another answer
# than the join's accepted pick. It is no key, so the official replaces it;
# it goes to review whole rather than being counted as the record's own
# scheme verified (Task 152's review).
SUPERSEDED_BY_OFFICIAL = "superseded-by-official"

# Where a record withheld for conflicting official keys keeps its own key:
# `answerScheme.metadata.conflictingKeys.existing`, the whole scheme. Without
# it the next relink sees a record with no scheme, nothing to compare, and
# serves the join's pick as newly attached -- and the pipeline's own order
# (enrich, merge, enrich) runs the relink twice (Task 152's review). Nested,
# never the top-level `correctOption` the scorers read.
CONFLICT_HELD = "conflictingKeys"

# The verifier a relink ran, stamped on the bank it wrote (`schemeVerifier`,
# written by scripts/enrich_question_bank.py). merge_question_banks.py serves
# the board records this relink verified, so it refuses a served bank without
# this stamp: merged before the relink, it would serve labels no verifier has
# seen (Task 151's pipeline-order concern). Change it whenever `verify`,
# `join_verdict`, `AnswerKeyIndex.choose` or the relink's own rules change
# what is served, so a bank relinked under the old rules is refused too.
SCHEME_VERIFIER = "scheme-verify/content-join/conflicting-keys-held 2026-09-22"


def row_code(stored: str, source: str) -> str:
    """The paper code an answer_keys row is filed under, or "".

    The writer's `paper_code` column is favoured over the filename (see
    `AnswerKeyIndex._load`), except where the column is led by a SUBJECT code:
    English Core rows sit under "301/1/1" for "XII_301_1_1_3_MS_unsigned
    (1).pdf", which is paper 1/1/3. The writer read three parts from the left
    of "301/1-1-3" and dropped the set, so sets 1-3 of a series collapsed
    onto one key no question carries (794 rows; 184/2/* English Language and
    Literature rows the same way, 34). The filename keeps the set; "XII_301_
    English Core_MS_Set 6-3.pdf", which names no code, is paper 1/6/3.
    """
    if not (stored or "").strip():
        if not _CODE_HINT.search(source or ""):
            return ""
        return normalize_paper_code(source)
    code = normalize_paper_code(stored) or stored.strip()
    parts = code.split("/")
    if len(parts) != 3 or parts[0] not in _SUBJECT_BY_CODE:
        return code
    from_name = normalize_paper_code(source or "")
    if from_name and from_name.split("/")[0] not in _SUBJECT_BY_CODE:
        return from_name
    m = re.search(r"Set[\s_]*(\d)\s*[_\-]\s*(\d)(?!\d)", source or "", re.I)
    if m and m.group(1) == parts[2]:
        return f"{parts[1]}/{m.group(1)}/{m.group(2)}"
    return ""


def family_of(code: str) -> str:
    """The paper series a code belongs to: "55/1" for 55/1/1, 55/1/2, 55/1/3.
    One marking scheme answers all of them."""
    norm = normalize_paper_code(code) or code
    return norm.rsplit("/", 1)[0] if norm.count("/") == 2 else ""


@dataclass
class Choice:
    """What the index made of one question's candidates.

    `rejected` holds the content join's refused pick (and, when two keys name
    different options, its runner-up) with the reason, then the verifier's
    verdict on every other row filed under the question's own (code, q_no) --
    the rows a question-number join would have tried, which is how the relink
    recognises the row a labelled scheme was built from."""

    best: OfficialAnswer | None
    rejected: list[tuple[OfficialAnswer, Verdict]] = field(default_factory=list)
    verdict: Verdict | None = None      # the best candidate's
    # The runner-up's score over the pick's, and whether the pick sits at the
    # question's own number or a sibling's. None when nothing was ranked.
    ratio: float | None = None
    anchored: bool = False
    # The row the join picked, accepted or refused: `best` once accepted
    # (with the option letter its words are), the refused pick otherwise.
    pick: OfficialAnswer | None = None


@dataclass
class _Pool:
    """One scheme family in one medium, ready to rank."""

    answers: list[OfficialAnswer]
    texts: list[str]            # `row_text` of each answer: what is judged
    terms: list[frozenset[str]]
    idf: dict[str, float]
    norms: list[float]


class AnswerKeyIndex:
    """`answer_keys.db` indexed by normalised paper code, keeping every candidate.

    Keyed by code rather than by filename because the code is the only
    identifier both a question paper and its marking scheme reliably carry.
    Filenames vary wildly -- "32-3-2 H.pdf", "MS Science (086) 31-8-1 to 3) in
    HINDI.pdf" -- and content hashes identify a file, not a paper.

    A key holds a LIST. 3,117 of the 11,787 (code, q_no) keys carry answers
    from more than one source file (measured 2026-09-22): "55-1-1,2,3 English
    Version.pdf" and "55-1-1,2,3 Hindi Version.pdf" are the two media of one
    Physics paper, "XII_042_Physics_MS_55_1_1,2,3.pdf" a second edition of the
    English one whose question numbers do not line up with it. Keeping one
    entry per key kept whichever row loaded last. `lookup` now chooses among
    them, and only among those `scheme_verify` accepts for the question.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._by_code: dict[str, dict[int, list[OfficialAnswer]]] = {}
        self._by_family: dict[str, list[OfficialAnswer]] = {}
        self._pools: dict[tuple[str, str], _Pool] = {}
        self.rows_total = 0
        self.rows_keyed = 0
        if self.db_path.exists():
            self._load()

    def _load(self) -> None:
        con = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        # `busy_timeout` applies here; `journal_mode=WAL` does NOT, and is
        # deliberately absent rather than forgotten. This connection is opened
        # read-only to a database another process owns, and setting the journal
        # mode is a write (`attempt to write a readonly database`). The read
        # still needs a lock timeout, or it fails instantly against a writer.
        con.execute("PRAGMA busy_timeout=60000")
        try:
            rows = list(con.execute(
                "SELECT paper_code, subject, grade, q_no, correct_option, "
                "answer_text, value_points, marks, source FROM answer_keys"
            ))
        finally:
            con.close()

        for r in rows:
            self.rows_total += 1
            # The writer put the canonical code in `paper_code`, derived from
            # the page headers -- the authoritative place it lives. Favouring
            # the column over the filename matters: 79 real rows sit under
            # names like "Science(MS) Set 1.pdf" that carry no code, and
            # re-deriving from the filename silently dropped those papers
            # (found 2026-09-20: the board's 31/1/2, 31/2/2, ... 55/*/2,
            # 55/*/3 variants were keyed from the header at write time and
            # unreachable at read time). `row_code` makes the one exception.
            # A row with no code in the column or the filename may still be
            # reachable by content hash, which is a separate pass; no code is
            # guessed here.
            code = row_code(str(r["paper_code"] or ""), str(r["source"] or ""))
            if not code:
                continue
            try:
                q_no = int(r["q_no"])
            except (TypeError, ValueError):
                continue
            points = _parse_value_points(r["value_points"])
            source = str(r["source"] or "")
            text = str(r["answer_text"] or "")
            entry = OfficialAnswer(
                q_no=q_no,
                correct_option=(r["correct_option"] or None),
                answer_text=text,
                value_points=points,
                marks=float(r["marks"] or 0.0),
                source=source,
                medium=answer_medium(source, text + " " + " ".join(p for p, _ in points)),
                code=code,
            )
            self._by_code.setdefault(code, {}).setdefault(q_no, []).append(entry)
            self._by_family.setdefault(family_of(code), []).append(entry)
            self.rows_keyed += 1

    @property
    def codes(self) -> set[str]:
        return set(self._by_code)

    # There is no subject-code map. The one that stood here sent 55/* to
    # 301/* ("English: corpus uses 55") and 64/* to 55/* ("Physics: corpus
    # uses 64"), and both premises were misreadings. 55/* IS Physics -- the
    # rows the audit took for English XII come from "55-x-1,2,3 English
    # Version.pdf", the English-medium Physics scheme, which the writer filed
    # under subject "English". 64/* is Geography ("XII_029_MS_64-1-1 ENG
    # UNSIGNED.pdf", 496 rows under its own code). The map's only effect was
    # to give 7 served Geography questions Hindi-medium Physics answers.
    # (The 301/* keys were English Core schemes mis-keyed from filenames like
    # "XII_301_1_1_3_MS_unsigned (1).pdf", which is paper 1/1/3; `row_code`
    # now files them there.)

    @property
    def families(self) -> set[str]:
        return set(self._by_family)

    def candidates(self, code: str, q_no: int) -> list[OfficialAnswer]:
        """Every official answer filed under this code and question number."""
        if not code:
            return []
        norm = normalize_paper_code(code) or code
        return list(self._by_code.get(norm, {}).get(q_no, []))

    def _pool(self, family: str, medium: str) -> _Pool:
        """Every row of a scheme family in `medium`, or naming none: an
        English-medium question never ranks a Hindi-medium answer."""
        key = (family, medium)
        if key not in self._pools:
            answers = [a for a in self._by_family.get(family, []) if a.medium in (medium, "")]
            texts = [row_text(a.text()) for a in answers]
            terms = [join_terms(t) for t in texts]
            idf = idf_weights(terms)
            self._pools[key] = _Pool(answers, texts, terms, idf,
                                     [norm_of(t, idf) for t in terms])
        return self._pools[key]

    def choose(self, code: str, q_no: int, record: dict[str, Any], *,
               siblings: Iterable[tuple[str, int]] = ()) -> Choice:
        """The content join: the family row that answers `record`, and why
        nothing was taken otherwise.

        Every row of the paper's series in the question's medium is scored
        against the stem, whatever its number: a set-2 question's answer is
        often filed under set 1's numbering. The pick is the best-scoring row
        `scheme_verify.join_verdict` would take -- the verifier accepts it and
        it carries evidence of its own: an objective key whose words are an
        option, a subjective row that is anchored or restates the question.
        The runner-up is the best such row that is not an edition of the pick
        (`AGREE_SHARE`), and the pick must beat it by `JOIN_MARGIN`.

        Rows without that evidence rank but do not compete. The best-scoring
        row overall answered the question in 20 of 50 questions read (fixture
        stratum join-random) -- a series holds every chapter's answers -- so
        letting it veto was letting noise veto: 32/5/2 Q29 lost its own
        answer ("How primary, secondary and tertiary sectors are dependent on
        each other? ...") to set 1's tertiary-sector answer, 0.93 of whose
        score it had.

        A pick filed under the question's OWN code at another number answers
        another question of the same paper -- one set's scheme numbers as
        that set's paper does -- and is refused (OTHER_NUMBER) unless it is
        anchored by itself: a row at an anchor naming the same option, for an
        objective key; for a subjective one, the row sitting at an anchor. An
        edition of the own row does not anchor it: "XII_029_MS_64-4-1 ENG
        UNSIGNED.pdf" prints one Visually Impaired alternative under Q22 and
        Q29, and Q22's was taken for Q29 as its edition (judged wrong).

        `siblings` are the (code, q_no) of the same question served from
        other sets of the series. A row is anchored when it, or an edition of
        it, sits at the question's own number or a sibling's: "55-1-1,2,3
        English Version.pdf" and "XII_042_Physics_MS_55_1_1,2,3.pdf" print one
        answer under different numbers. Ties keep load order and are refused.

        An objective question takes only keys whose words are one of its
        options, and ANY key of the series naming another of them is a tie
        (ratio 1.0), whatever it scores: a key's score is the rarity of its
        option's words in the series, not evidence that it answers this
        question. 66/1/1 Q19 (Beenu's bookstore) has "Public relations" under
        its own number and set 3's "Sales promotion" -- Mehta Sons' key --
        names option C; in a three-row pool the wrong key outscored the right
        one by 0.76. A key that shares no word or number with the stem ("x =
        y", "(– 3, 0)") scores nothing and is taken only when anchored.
        """
        norm = normalize_paper_code(code) or code
        own = self.candidates(norm, q_no)
        family = family_of(norm)
        pool = self._pool(family, question_medium(record)) if family else None
        if pool is None or not pool.answers:
            return Choice(best=None, rejected=[(c, _own_verdict(record, c)) for c in own])
        anchors = {(norm, q_no), *siblings}
        at_anchor = [i for i, a in enumerate(pool.answers) if (a.code, a.q_no) in anchors]
        query = join_terms(str(record.get("stem") or ""))
        scores = [join_score(query, t, pool.idf, n) for t, n in zip(pool.terms, pool.norms)]
        ranked = sorted((i for i, sc in enumerate(scores) if sc > 0), key=lambda i: -scores[i])
        options = question_options(record)

        pick: int | None = None
        runner: int | None = None
        ratio = 0.0
        anchored = False
        if options:
            letters = {i: letter for i, text in enumerate(pool.texts)
                       if (letter := outright_option(text, options)) is not None
                       and verify(record, text).accepted}

            def keyed_anchor(i: int) -> bool:
                return any(letters.get(j) == letters[i] for j in at_anchor)

            eligible = [i for i in ranked if i in letters]
            eligible += [i for i in letters if scores[i] <= 0 and keyed_anchor(i)]
            if eligible:
                pick = eligible[0]
                anchored = keyed_anchor(pick)
                others = [i for i, letter in letters.items() if letter != letters[pick]]
                if others:
                    runner = max(others, key=lambda i: scores[i])
                    ratio = 1.0
        else:
            def anchor_of(i: int) -> bool:
                return i in at_anchor or any(
                    share(pool.terms[i], pool.terms[j]) >= AGREE_SHARE for j in at_anchor)

            for i in ranked:
                if pick is not None and share(pool.terms[pick], pool.terms[i]) >= AGREE_SHARE:
                    continue            # an edition of the pick
                anch = anchor_of(i)
                if not join_verdict(record, pool.texts[i], ratio=0.0, anchored=anch).accepted:
                    continue
                if pick is None:
                    pick, anchored = i, anch
                else:
                    runner = i
                    ratio = scores[i] / scores[pick]
                    break
        if pick is None and ranked:
            # Nothing eligible: report the best-scoring row and why it is not.
            pick = ranked[0]
            anchored = (pool.answers[pick].code, pool.answers[pick].q_no) in anchors
        if pick is None:
            return Choice(best=None, rejected=[(c, _own_verdict(record, c)) for c in own])

        best = pool.answers[pick]
        other_number = best.code == norm and best.q_no != q_no
        if other_number and not options:
            anchored = pick in at_anchor
        verdict = join_verdict(record, pool.texts[pick], ratio=ratio, anchored=anchored,
                               other_number=other_number)
        refused: list[tuple[OfficialAnswer, Verdict]] = []
        if not verdict.accepted:
            refused.append((best, verdict))
            if verdict.reason == CONFLICTING_KEYS and runner is not None:
                refused.append((pool.answers[runner], verdict))
        listed = [a for a, _ in refused] + ([best] if verdict.accepted else [])
        rejected = refused + [(c, _own_verdict(record, c)) for c in own
                              if not any(c is a for a in listed)]
        if not verdict.accepted:
            return Choice(best=None, rejected=rejected, ratio=ratio, anchored=anchored,
                          pick=best)
        # The letter is the option the key's words are, never the row's own
        # letter column (another set's row may order the options differently);
        # a subjective pick carries none, or the scheme would be built objective.
        best = replace(best, correct_option=verdict.option)
        return Choice(best=best, rejected=rejected, verdict=verdict, ratio=ratio,
                      anchored=anchored, pick=best)

    def lookup(self, code: str, q_no: int, record: dict[str, Any]) -> OfficialAnswer | None:
        """The official answer to label for this question, or None."""
        return self.choose(code, q_no, record).best

    def stats(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "rows_keyed": self.rows_keyed,
            "distinct_codes": len(self._by_code),
        }


def _own_verdict(record: dict[str, Any], answer: OfficialAnswer) -> Verdict:
    """Why a row under the question's own number was not attached: the
    verifier's reason, or OUTRANKED when the verifier alone would take it."""
    verdict = verify_answer(record, answer)
    return verdict if not verdict.accepted else Verdict(OUTRANKED)


def _parse_value_points(raw: Any) -> list[tuple[str, float]]:
    """`value_points` is stored as a JSON array of [text, marks] pairs."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    out: list[tuple[str, float]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and item:
                text = str(item[0] or "").strip()
                marks = float(item[1]) if len(item) > 1 and item[1] is not None else 0.0
                if text:
                    out.append((text, marks))
            elif isinstance(item, str) and item.strip():
                out.append((item.strip(), 0.0))
    return out


# --------------------------------------------------------------------------- #
# the relink
# --------------------------------------------------------------------------- #

@dataclass
class RelinkReport:
    """What the relink actually did. Printed by the pipeline, asserted in tests."""

    total: int = 0
    # Board records that had a scheme end in exactly one of the next three:
    # verified as it was, replaced, or removed. (Nothing is kept unlabelled
    # any more: a scheme no official row verifies is not served.)
    already_had_scheme: int = 0
    # The record's own scheme passed, and a verified official ends on it.
    verified_existing_scheme: int = 0
    # The record's own scheme was quarantined, and a verified official
    # candidate was attached in its place. Counted apart from the above: the
    # scheme the record had is the one removed, not the one verified.
    replaced_after_quarantine: int = 0
    # The record's own scheme was quarantined and nothing verified replaced it.
    existing_scheme_removed: int = 0
    newly_attached: int = 0
    # Verified officials the content join found somewhere other than the
    # question's own (code, q_no): another set's numbering, or another number.
    recovered_elsewhere: int = 0
    # CBE/SQP records: their ids are not cbse:q:src:<hash>:<n>, so they never
    # reach the answer-key index, but their builder already attached the
    # official scheme. Counting them under no_source_hash made coverage read
    # 50.4% when the real figure was 93.9%.
    carried_official: int = 0
    no_source_hash: int = 0
    no_parse_tree: int = 0
    no_paper_code: int = 0
    # The code's series has no row in answer_keys.db, or none in the
    # question's medium that shares a word or number with its stem and none
    # under its own number either.
    code_not_in_answer_keys: int = 0
    no_matching_candidate: int = 0
    # Records the join ranked a candidate for (or that have a row under their
    # own number), and those among them for which nothing was accepted.
    found: int = 0
    all_candidates_rejected: int = 0
    # Scheme content REMOVED from a record because the verifier rejected it,
    # by reason; `quarantine` holds one entry per removal for a person to
    # review, with the whole scheme. `rejected` counts the join's refused
    # picks and the rows under a question's own number never attached.
    quarantined: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    # Records whose own official (cbse_marking_scheme) scheme verified while
    # the content join accepted a DIFFERENT answer: two official keys, nothing
    # to say which is right, so
    # neither is served (both are in `quarantine` as conflicting-keys). Each
    # entry names the record and both answers, for a person to settle.
    conflicting_keys: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    # Of those, conflicts an earlier relink found and held on the record
    # (`CONFLICT_HELD`): still two answers, still neither served.
    conflicts_held: int = 0
    # Records whose unlabelled scheme verified but named another answer than
    # the join's accepted pick: quarantined as superseded-by-official, the
    # official served in its place (counted in replaced_after_quarantine).
    # Each entry names the record and both answers.
    superseded: list[dict[str, Any]] = field(default_factory=list)
    # Records whose scheme held a Symbol-font code point the answer-key store
    # carries and `_restore_scheme_symbols` turned back into the character it
    # stands for -- the merge's `symbol-font-restored`, applied to the scheme
    # the relink writes rather than to the record the merge read.
    symbol_font_restored: int = 0
    # And the other half: records whose scheme still holds a private-use
    # character the table CANNOT name, in a field a scorer reads. Nothing here
    # guesses it, and nothing downstream sees it until the next merge withholds
    # the whole question as private-use-glyph -- so the run reports it.
    symbol_font_unnameable: int = 0

    @property
    def verified(self) -> int:
        """Records that end with an official scheme the verifier accepted --
        the only ones labelled cbse_marking_scheme."""
        return (self.verified_existing_scheme + self.replaced_after_quarantine
                + self.newly_attached + self.carried_official)

    @property
    def with_scheme(self) -> int:
        """Records carrying any scheme. Only a verified official is served,
        so this is `verified`; kept so a consumer reading it is not misled."""
        return self.verified

    def coverage(self) -> float:
        """Share of records with a VERIFIED official scheme."""
        return (100.0 * self.verified / self.total) if self.total else 0.0

    def lines(self) -> list[str]:
        out = [
            "questions                      : {:,}".format(self.total),
            "had a scheme before            : {:,}".format(self.already_had_scheme),
            "  its scheme verified, official: {:,}".format(self.verified_existing_scheme),
            "  its scheme quarantined, replaced by a verified official: {:,}".format(
                self.replaced_after_quarantine),
            "  its scheme quarantined, nothing in its place: {:,}".format(
                self.existing_scheme_removed),
            "    conflicting official keys (neither served): {:,}".format(
                self.conflicting_keys),
            "      (held from an earlier relink: {:,})".format(self.conflicts_held),
            "    its unlabelled scheme named another answer, superseded by the official: {:,}"
            .format(len(self.superseded)),
            "verified official attached new : {:,}".format(self.newly_attached),
            "  (verified officials found away from the question's own number: {:,})".format(
                self.recovered_elsewhere),
            "  carried an official scheme from its builder, verified : {:,}".format(
                self.carried_official),
            "with a VERIFIED official scheme: {:,}  ({:.1f}%)".format(
                self.verified, self.coverage()),
            "with any scheme                : {:,}".format(self.with_scheme),
            "symbol-font code points restored in the scheme: {:,}".format(
                self.symbol_font_restored),
            "  schemes left holding a symbol the table cannot name: {:,}".format(
                self.symbol_font_unnameable),
            "",
            "quarantined (scheme removed from the record): {:,}".format(
                sum(self.quarantined.values())),
            "  (of which replaced by a verified official: {:,})".format(
                self.replaced_after_quarantine),
        ]
        out += ["  {:<20}: {:,}".format(r, n) for r, n in sorted(self.quarantined.items())]
        out += ["official candidates rejected (never attached): {:,}".format(
            sum(self.rejected.values()))]
        out += ["  {:<20}: {:,}".format(r, n) for r, n in sorted(self.rejected.items())]
        out += [
            "",
            "did not reach an answer key:",
            "  id is not a cbse:q:src:<hash>:<n> : {:,}".format(self.no_source_hash),
            "  no parse tree for the source      : {:,}".format(self.no_parse_tree),
            "  no paper code in the parse tree   : {:,}".format(self.no_paper_code),
            "  series absent from answer_keys.db : {:,}".format(self.code_not_in_answer_keys),
            "  series present, nothing matched   : {:,}".format(self.no_matching_candidate),
            "  (an answer key found for)         : {:,}".format(self.found),
            "  every candidate rejected          : {:,}".format(self.all_candidates_rejected),
        ]
        return out


def build_official_scheme(
    answer: OfficialAnswer, *, marks: int, question_id: str, code: str,
    document_id: str, options: dict[str, str] | None = None,
) -> AnswerSchemeSchema:
    """Turn a KeyedAnswer into the API's scheme shape.

    An official key that carries a `correct_option` letter is an objective
    scheme: one all-or-nothing point naming the option, a two-level rubric,
    the other options as distractors, and `metadata.objective` so the scoring
    path (`evaluate._evaluate_objective`, which reads `metadata.correctOption`)
    can grade it. The `options` dict comes from the question stem, the only
    place the option text lives; when it is absent the letter itself is the
    scheme (keyword and synonym), never empty.

    A descriptive key is rebuilt from its value points. Marks per value point
    are left at 0 when the source did not carry them: a marking scheme's mark
    column is emitted separately from its text in a flattened PDF, so attaching
    a number would be inventing one.
    """
    options = options or {}
    # Which scheme file the answer came from, so a reviewer can open it: the
    # question's own document id says nothing about which of a key's
    # candidates was chosen.
    source = ({"schemeSource": answer.source, "schemeMedium": answer.medium,
               "schemeCode": answer.code, "schemeQNo": answer.q_no}
              if answer.source else {})
    if answer.correct_option:
        key = str(answer.correct_option).strip().upper()
        option_text = options.get(key, "")
        point_text = option_text or f"Correct option {key}"
        points = [
            MarkingPointSchema(
                id=f"{question_id}:mp1",
                description=(f"Correct option {key}: {option_text}".strip()
                             if option_text else point_text),
                marks=marks or 1,
                keyword=option_text or key,
                is_required=True,
                synonyms=[key],
            )
        ]
        rubric = [
            RubricLevelSchema(
                level=1, label="Correct",
                min_marks=marks or 1, max_marks=marks or 1,
                description="Correct option identified.",
            ),
            RubricLevelSchema(
                level=0, label="Incorrect", min_marks=0, max_marks=0,
                description="Wrong or no option selected.",
            ),
        ]
        return AnswerSchemeSchema(
            total_marks=marks or 1,
            marking_points=points,
            rubric_levels=rubric,
            common_errors=[f"Chose distractor {lbl}" for lbl in options if lbl != key],
            alternative_answers=[],
            model_answer=option_text or answer.answer_text,
            has_partial_credit=False,
            metadata={"objective": True, "options": options, "correctOption": key, **source},
            provenance="cbse_marking_scheme",
            source_paper_code=code,
            source_document_id=document_id,
        )

    points = [
        MarkingPointSchema(
            id=f"{question_id}:mp{i + 1}",
            description=text,
            marks=int(round(m)) if m else 0,
        )
        for i, (text, m) in enumerate(answer.value_points)
    ]
    return AnswerSchemeSchema(
        total_marks=marks,
        marking_points=points,
        model_answer=answer.answer_text,
        has_partial_credit=len(points) > 1,
        metadata=dict(source),
        provenance="cbse_marking_scheme",
        source_paper_code=code,
        source_document_id=document_id,
    )


def _scheme_content(scheme: dict[str, Any]) -> tuple[str, str | None]:
    """A record's scheme as the verifier reads a candidate: its text, and
    its option letter when it is an objective scheme."""
    meta = scheme.get("metadata") or {}
    letter = meta.get("correctOption") if meta.get("objective") else None
    if letter:
        # An objective scheme's one point is `build_official_scheme`'s own
        # "Correct option C: harder" -- generated text, not the key's words.
        # The key's words, if it had any, are the model answer.
        return str(scheme.get("modelAnswer") or "").strip(), str(letter)
    parts = [str(scheme.get("modelAnswer") or "").strip()]
    for mp in scheme.get("markingPoints") or []:
        desc = str((mp or {}).get("description") or "").strip()
        if desc and not any(desc in p for p in parts):
            parts.append(desc)
    return " ".join(p for p in parts if p), None


def builder_key_reason(rec: dict[str, Any]) -> str | None:
    """Why the relink removes a non-board record's own scheme, or None.

    One rule for both ends of the pipeline: `relink_answer_schemes` applies
    it to every record with no board-paper id, and the merge
    (`bank_merge.compose`) refuses a CBE, SQP or Exemplar record it fails as
    key-rejected. Before, the merge served keys the relink after it removed --
    67 at 8e92ed9, left served with no key.

    The options are the ones the builder resolved when it wrote an objective
    scheme (`metadata.options`, `bank_merge._objective_scheme`), not the
    record re-read: the verifier's own reader takes the numbered statements
    before cbe:q:Science10MS2's (A)-(D) for its options. A scheme that is not
    objective, on a record not typed mcq, has no options to name: its
    builder did not make it an MCQ, and read as one, the "(a) ... (b) ..."
    blanks of exemplar:q:6:mathematics:1:-:99 were four options its key named
    none of. A record with no scheme content is not this rule's to judge.
    """
    scheme = rec.get("answerScheme") or {}
    if not _has_content(scheme):
        return None
    text, letter = _scheme_content(scheme)
    meta = scheme.get("metadata") or {}
    resolved = meta.get("options") if letter else None
    if isinstance(resolved, dict) and resolved:
        options: dict[str, str] | None = {str(k): str(v) for k, v in resolved.items()}
    elif not letter and rec.get("type") != "mcq":
        options = {}
    else:
        options = None
    return verify(rec, text, option=letter, judge_content=False, options=options).reason


def _source_row(scheme: dict[str, Any],
                rejected: list[tuple[OfficialAnswer, Verdict]]) -> Verdict | None:
    """The rejection of the index row a record's scheme was built from, if
    one of the rejected candidates is recognisably that row: the same option
    letter, or the same answer text."""
    letter = str((scheme.get("metadata") or {}).get("correctOption") or "").upper()
    model = str(scheme.get("modelAnswer") or "").strip()
    first = str(((scheme.get("markingPoints") or [{}])[0] or {}).get("description") or "").strip()
    for cand, verdict in rejected:
        if letter and str(cand.correct_option or "").upper() == letter:
            return verdict
        text = cand.answer_text.strip()
        points = [p for p, _ in cand.value_points]
        if (text and text in (model, first)) or (first and first in points):
            return verdict
    return None


def _has_content(scheme: dict[str, Any]) -> bool:
    return bool(scheme.get("markingPoints") or str(scheme.get("modelAnswer") or "").strip())


def _quarantine(rec: dict[str, Any], report: RelinkReport, reason: str, text: str,
                code: str, q_no: int | None, **extra: Any) -> None:
    """Remove a rejected scheme from the record and log it for review.

    Removed, not merely unlabelled: another question's answer in the record
    is harmful whatever its label, because the scorer and the teacher both
    read it. The record keeps an empty scheme of its own mark value.

    The entry keeps the WHOLE removed scheme -- every marking point, the
    model answer, the option letter and the scheme file it came from -- so a
    person can review it and restore it. It used to keep 200 characters of
    text, and after one run the served bank no longer holds the rest.
    """
    removed = dict(rec.get("answerScheme") or {})
    meta = removed.get("metadata") or {}
    report.quarantined[reason] += 1
    report.quarantine.append({
        "id": rec.get("id"), "code": code, "q_no": q_no, "reason": reason, "text": text,
        "correctOption": meta.get("correctOption"),
        "schemeSource": meta.get("schemeSource") or "",
        "scheme": removed,
        **extra,
    })
    rec["answerScheme"] = AnswerSchemeSchema(
        total_marks=int(rec.get("marks") or 0)).model_dump(mode="json", by_alias=True)


def _answer_of(scheme: dict[str, Any]) -> dict[str, Any]:
    """What a scheme says the answer is, and which scheme file says it."""
    text, letter = _scheme_content(scheme)
    meta = scheme.get("metadata") or {}
    return {"option": letter, "text": text, "source": meta.get("schemeSource") or "",
            "code": meta.get("schemeCode") or "", "qNo": meta.get("schemeQNo")}


def _agreeable(text: str) -> str:
    """One answer's text as it compares with another's.

    Whitespace collapsed, case folded, and the symbol-font code points the
    merge restores (corpus/symbol_font.py) read as the characters they stand
    for: the served bank carries cbse:q:src:784bd959b5f48fd5620f57e6:27 as
    "2⋅4 g/litre" and the answer-key store still holds the extraction's
    "2<U+F0D7>4 g/litre". They are one answer, and a short key whose only
    difference is the symbol shares too few words for the share rule below to
    decide -- it would read as a conflict and the record would lose its key.
    A private-use character the table cannot name is left as it is: nothing
    knows what it was, so it stays a difference.
    """
    # Imported here, not at the top: corpus imports this module.
    from ..corpus.symbol_font import restore_symbol_font

    return restore_symbol_font(_collapse(text or "")).casefold()


def _same_answer(ours: dict[str, Any], pick: dict[str, Any],
                 options: dict[str, str]) -> bool:
    """Two schemes name the same answer, by the join's own tests.

    An objective answer is its option: the scheme's letter, or for a scheme
    built without one the option its words are (`outright_option` -- a
    pre-Task-15 scheme keeps "(B)/ Amorphous, colloidal ..." as text). Two
    letters agree when they are the same letter. Otherwise two texts agree
    when they are editions of one answer, which is what `choose` calls a row
    sharing `AGREE_SHARE` of the pick's join terms: two sets of a series
    print one descriptive answer in different words and layout. On the
    served bank at 7747f7f exact text found 20 disagreements, 7 of them one
    answer (57/5/2 Q9 "(B)/ Amorphous, ..." against B; five descriptive
    editions such as 67/1/1 Q22 against 67/1/2 Q21); this rule finds 13."""
    a, b = _answer_of(ours), _answer_of(pick)
    letters = [str(x["option"] or outright_option(x["text"], options) or "").strip().upper()
               for x in (a, b)]
    if all(letters):
        return letters[0] == letters[1]
    if _agreeable(a["text"]) == _agreeable(b["text"]):
        return True
    ta, tb = row_text(a["text"]), row_text(b["text"])
    # Share is blind to what flips an answer: `join_terms` drops "not", "no",
    # "yes" and single digits, so "It is a rational number" and "It is not a
    # rational number" share every term. Editions of one answer keep the
    # same polarity words and the same numbers; anything else is a conflict,
    # which fails closed.
    if _polarity(ta) != _polarity(tb) or _all_numbers(ta) != _all_numbers(tb):
        return False
    return share(join_terms(ta), join_terms(tb)) >= AGREE_SHARE


# Words that turn one answer into its opposite, each mapped to its sense so
# "isn't" and "not", "increases" and "increased" count alike.
_POLARITY_WORD = re.compile(
    r"\b(not|no|never|none|nor|neither|cannot|yes|true|false|"
    r"increas\w*|decreas\w*|higher|lower|more|less|positive|negative)\b"
    r"|n['’]t\b", re.I)
_POLARITY_SENSE = {"never": "not", "none": "not", "nor": "not", "neither": "not",
                   "cannot": "not", "n't": "not", "n’t": "not"}


def _polarity(text: str) -> frozenset[str]:
    senses = set()
    for m in _POLARITY_WORD.finditer(text or ""):
        w = m.group(0).casefold()
        if w.startswith("increas"):
            w = "increase"
        elif w.startswith("decreas"):
            w = "decrease"
        senses.add(_POLARITY_SENSE.get(w, w))
    return frozenset(senses)


_ANY_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _all_numbers(text: str) -> frozenset[str]:
    """Every number, single digits too (`join_terms` keeps two characters up)."""
    return frozenset(n.lstrip("0") or "0" for n in _ANY_NUMBER.findall(text or ""))


def _serve_neither(rec: dict[str, Any], report: RelinkReport, pick: dict[str, Any],
                   code: str, q_no: int | None) -> None:
    """The record's own verified official scheme and the join's accepted
    pick name different answers. Both passed the verifier, so the verifier cannot say
    which is the board's; serving either would be a coin toss labelled
    official. Until Task 152 the join's pick simply overwrote the record's
    scheme and was counted verified_existing_scheme. On the served bank at
    7747f7f 13 of those 53 records name two answers: 11 option letters
    (30/5/2 Q17 A '90°' against D '30°', 430/2/3 Q6 C '60°' against A '30°',
    nine MCQs of 66/1/1 such as Q16 'Personal selling' against 'Sales
    promotion') and 2 descriptive keys whose join row restates the question
    first (430/5/1 Q24, 430/5/2 Q28) -- likely one answer, but `_same_answer`
    cannot tell, and a conflict fails closed. A 14th since `_same_answer`
    also requires the same polarity words and numbers: 5884c691 Q31 (lens of
    +4D), whose join row is cut to "Focal length of lens, f(m) =" against our
    "... Real and inverted Magnification = -1". Both schemes are quarantined
    whole, each naming the other, and the record keeps no scheme (the merge
    withholds it as no-verified-key) -- only its own key, held under
    `metadata.conflictingKeys` for the next relink to compare again."""
    whole = dict(rec.get("answerScheme") or {})
    ours = _answer_of(whole)
    theirs = _answer_of(pick)
    report.conflicting_keys += 1
    report.conflicts.append({"id": rec.get("id"), "code": code, "q_no": q_no,
                             "existing": ours, "join": theirs})
    _quarantine(rec, report, CONFLICTING_KEYS, ours["text"], code, q_no,
                conflictsWith=theirs)
    # Held on the record, so the next relink compares the two again rather
    # than finding no scheme and serving the pick.
    rec["answerScheme"]["metadata"] = {CONFLICT_HELD: {"existing": whole, "join": theirs}}
    # The pick was never on the record: a refused candidate, kept for review.
    report.rejected[CONFLICTING_KEYS] += 1
    report.quarantine.append({
        "id": rec.get("id"), "code": code, "q_no": q_no, "reason": CONFLICTING_KEYS,
        "text": theirs["text"], "correctOption": theirs["option"],
        "schemeSource": theirs["source"], "scheme": pick, "conflictsWith": ours,
    })


def relink_answer_schemes(
    records: list[dict[str, Any]],
    index: AnswerKeyIndex,
    *,
    parse_dir: Path,
    attach_provenance: bool = True,
) -> RelinkReport:
    """Attach official marking schemes to question records, in place.

    `records` are the camelCase dicts as exported (the wire shape), because that
    is what `questions.json` is made of and rewriting the pipeline's shape here
    would put a second schema in the path. Fields written: `answerScheme`,
    and when `attach_provenance`, `provenance`.

    Every record's scheme is verified first, including one already labelled
    cbse_marking_scheme: the old shortcut counted a labelled scheme verified
    and skipped it, and the audit found 30.5% of those detectably wrong. A
    rejected scheme is removed and quarantined. Then the index's content join
    is asked for the family row that answers the question (`AnswerKeyIndex.
    choose`); only an accepted one is labelled official. A record with no
    accepted row keeps no scheme at all: an unverified scheme is not served.
    A record whose own verified cbse_marking_scheme names another answer
    than the accepted row keeps none either: two official keys that disagree
    are a conflict, not an overwrite (`_serve_neither`), and the record holds
    its own key (`CONFLICT_HELD`) so every later relink compares the two
    again; it is served only when the join's pick agrees with it. An
    unlabelled scheme is no key, so the accepted row replaces it: kept as
    verified when it names the same answer, quarantined as
    superseded-by-official and reported when it names another.
    """
    report = RelinkReport(total=len(records))
    sources: dict[str, ParsedSource | None] = {}
    had_before: list[dict[str, Any]] = []

    def located(rec: dict[str, Any]) -> tuple[tuple[str, int] | None, ParsedSource | None, str]:
        parsed = parse_question_id(rec.get("id") or "")
        if parsed is None:
            return None, None, ""
        doc_hash = parsed[0]
        if doc_hash not in sources:
            sources[doc_hash] = load_parsed_source(Path(parse_dir), doc_hash)
        src = sources[doc_hash]
        return parsed, src, (src.paper_code if src else "")

    siblings = _siblings(records, located)

    for rec in records:
        parsed_id, source, code = located(rec)
        q_no = parsed_id[1] if parsed_id else None

        scheme = rec.get("answerScheme") or {}
        had_scheme = _has_content(scheme)
        held = None
        if not had_scheme and parsed_id is not None:
            held = ((scheme.get("metadata") or {}).get(CONFLICT_HELD) or {}).get("existing")
            if isinstance(held, dict) and _has_content(held):
                # An earlier relink withheld this record for two disagreeing
                # official keys. Its own key is compared again, as it was:
                # re-verified first, so a key the verifier now rejects goes
                # to review and no longer blocks the join's pick.
                text, letter = _scheme_content(held)
                verdict = verify(rec, text, option=letter, judge_content=True)
                if not verdict.accepted:
                    rec["answerScheme"] = held
                    _quarantine(rec, report, verdict.reason, text, code, q_no)
                    held = None
            else:
                held = None
        kept = had_scheme
        if had_scheme:
            text, letter = _scheme_content(scheme)
            # A CBE/SQP/Exemplar scheme came from its own item, so there is no
            # key to misalign; the content rule is for board rows (see
            # `verify`), and the options are the ones its builder resolved.
            reason = (builder_key_reason(rec) if parsed_id is None else
                      verify(rec, text, option=letter, judge_content=True).reason)
            if reason is not None:
                _quarantine(rec, report, reason, text, code, q_no)
                kept = False
        labelled = kept and scheme.get("provenance") == "cbse_marking_scheme"

        if parsed_id is None:
            # Not a board-paper id, so there is no parse tree to relink from:
            # a CBE/SQP record keeps its builder's scheme if it passed.
            if labelled:
                report.carried_official += 1
            else:
                report.no_source_hash += 1
            continue
        if had_scheme:
            report.already_had_scheme += 1
            had_before.append(rec)

        def settle_without_candidate() -> None:
            if not kept:
                return
            # No accepted index row backs this scheme. A labelled one was
            # built from a row that is gone or no longer accepted, and an
            # option letter cannot be checked against anything: an MCQ scheme
            # "verifies" on its own trivially, because `build_official_scheme`
            # wrote the option's text as its model answer. Dropping only the
            # label is not enough -- the scorer reads `correctOption` whatever
            # the label says -- and in hand checks of such served keys 13 of
            # 21, then 4 of 10, were wrong. Unlabelled descriptive content went
            # the same way in Task 151: correctness first, so what no official
            # row verifies is not served (the merge then excludes the record
            # as no-verified-key).
            _quarantine(rec, report, UNBACKED_KEY, _scheme_content(scheme)[0], code, q_no)

        if source is None:
            report.no_parse_tree += 1
            settle_without_candidate()
            continue

        if attach_provenance:
            page = source.page_containing(rec.get("stem") or "")
            if page is not None or source.page_count:
                rec["provenance"] = ProvenanceSchema(
                    source_document_id=source.document_id,
                    page_number=page,
                    method="pdf_native",
                    confidence=round(1.0 if page else 0.6, 2),
                ).model_dump(mode="json", by_alias=True)

        if not code:
            report.no_paper_code += 1
            settle_without_candidate()
            continue

        choice = index.choose(code, q_no, rec, siblings=siblings.get(id(rec), ()))
        for _, verdict in choice.rejected:
            report.rejected[verdict.reason] += 1
        if choice.best is None:
            if choice.rejected:
                report.found += 1
                report.all_candidates_rejected += 1
                source_row = _source_row(scheme, choice.rejected) if labelled else None
                if source_row is not None:
                    # The labelled scheme was built from a row the verifier
                    # has just rejected. Standing alone it can look right --
                    # `build_official_scheme` wrote the option's own text as
                    # its model answer -- but it is only as good as that row.
                    _quarantine(rec, report, source_row.reason, _scheme_content(scheme)[0],
                                code, q_no)
                    kept = labelled = False
            elif family_of(code) in index.families:
                report.no_matching_candidate += 1
            else:
                report.code_not_in_answer_keys += 1
            settle_without_candidate()
            continue

        report.found += 1
        stem = str(rec.get("stem") or "")
        options = question_options(rec) or parse_options(stem)
        pick = build_official_scheme(
            choice.best,
            marks=int(rec.get("marks") or 0),
            question_id=str(rec.get("id")),
            code=code,
            document_id=source.document_id,
            options=options,
        ).model_dump(mode="json", by_alias=True)
        if held is not None:
            if not _same_answer(held, pick, options):
                rec["answerScheme"] = held
                report.conflicts_held += 1
                _serve_neither(rec, report, pick, code, q_no)
                continue
        elif labelled and not _same_answer(scheme, pick, options):
            _serve_neither(rec, report, pick, code, q_no)
            continue
        elif kept and not _same_answer(scheme, pick, options):
            # An unlabelled scheme is no key: the official replaces it. But
            # it names another answer, so it is not the record's own scheme
            # verified -- it goes to review whole, and is reported.
            ours = _answer_of(scheme)
            theirs = _answer_of(pick)
            report.superseded.append({"id": rec.get("id"), "code": code, "q_no": q_no,
                                      "existing": ours, "join": theirs})
            _quarantine(rec, report, SUPERSEDED_BY_OFFICIAL, ours["text"], code, q_no,
                        conflictsWith=theirs)
            kept = False
        rec["answerScheme"] = pick
        if (choice.best.code, choice.best.q_no) != (normalize_paper_code(code) or code, q_no):
            report.recovered_elsewhere += 1
        if not had_scheme:
            report.newly_attached += 1
        elif kept:
            report.verified_existing_scheme += 1
        else:
            report.replaced_after_quarantine += 1

    for rec in records:
        restored, unnameable = _restore_scheme_symbols(rec)
        report.symbol_font_restored += restored
        report.symbol_font_unnameable += unnameable
    report.existing_scheme_removed = sum(
        not _has_content(r.get("answerScheme") or {}) for r in had_before)
    return report


def _restore_scheme_symbols(rec: dict[str, Any]) -> tuple[bool, bool]:
    """Restore the Symbol-font code points in the scheme the relink just wrote,
    in place; whether it changed anything, and whether a private-use character
    the table CANNOT name is left in a field a scorer reads.

    The merge repairs a record's text before it serves it
    (`bank_merge.repair_symbol_font`), but the scheme the relink writes does
    not come from the record -- it is built from the answer-key store's own
    rows (`build_official_scheme`), which hold the text as the marking-scheme
    PDF was extracted, symbol-font code points and all. Task 11's runbook runs
    the relink AFTER the merge (plans/2026-09-21-question-bank-merge.md, step 3
    merge, step 4 enrich), so without this the documented pipeline puts the
    holes straight back: on the bank at 64dd21a it reverted three served
    records -- cbse:q:src:f01c7b6fd6fca7ef7d9987dc:27's "x cm ∴ altitude" to
    "x cm  altitude", 6df8d2b3a2f6c8cce1428573:31's "• Deriving the
    expression" to " Deriving...", and 784bd959b5f48fd5620f57e6:27's
    "2⋅4 g/litre" to "24 g/litre" -- with no gate to catch it, since the
    private-use gate only runs inside `compose`.

    The quarantine entries and the `conflictingKeys` hold are left raw: they
    are provenance, a copy of what was removed as it was removed, and a person
    reviewing one needs the characters the extractor actually produced. They
    are not counted as unnameable either, for the same reason the merge's gate
    does not read them (`bank_merge.printed_scheme`): nobody scores them.

    What the table cannot name is not guessed and not dropped -- it is
    REPORTED. `restore_symbol_font` covers the codes Adobe's Symbol encoding
    names and leaves the rest, and 46 of the 15,605 rows in answer_keys.db
    carry one it cannot (U+F09F x17, U+F0A7 x16, U+F0E7 x4, U+F076 x3,
    U+F0BE x2, U+F0F8 x2, U+F0FB, U+F0E0). A scheme built from such a row goes
    into a served record's `modelAnswer` as a hole a scorer reads, and the only
    thing that would ever notice is the NEXT merge -- which then withholds the
    whole question rather than just its key. A wrong answer shipped silently is
    worse than an error, so the run says so: `RelinkReport.symbol_font_
    unnameable`, printed beside the restored count.
    """
    from ..corpus.symbol_font import private_use_glyph, restore_symbol_font

    changed = False

    def fix(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, str):
            out = restore_symbol_font(value)
            changed = changed or out != value
            return out
        if isinstance(value, dict):
            return {k: (v if k == CONFLICT_HELD else fix(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v) for v in value]
        return value

    def unnameable(value: Any) -> bool:
        if isinstance(value, str):
            return private_use_glyph(value) is not None
        if isinstance(value, dict):
            return any(unnameable(v) for k, v in value.items() if k != CONFLICT_HELD)
        if isinstance(value, list):
            return any(unnameable(v) for v in value)
        return False

    scheme = rec.get("answerScheme")
    if not isinstance(scheme, dict):
        return False, False
    scheme = fix(scheme)
    rec["answerScheme"] = scheme
    return changed, unnameable(scheme)


def _siblings(records: list[dict[str, Any]], located) -> dict[int, list[tuple[str, int]]]:
    """For each board record (by `id()`), the (code, q_no) of the same
    question served from another set of its series: a stem sharing
    `SIBLING_SHARE` of its join terms, in the same medium.

    A pick at a sibling's own number is the answer the question-number join
    gives the sibling, found again by content from this set -- two signals
    agreeing, which is what `join_verdict` calls anchored.
    """
    groups: dict[tuple[str, str], list[tuple[int, str, int, frozenset[str]]]] = {}
    for rec in records:
        parsed, _, code = located(rec)
        family = family_of(code) if parsed and code else ""
        if not family:
            continue
        norm = normalize_paper_code(code) or code
        groups.setdefault((family, question_medium(rec)), []).append(
            (id(rec), norm, parsed[1], join_terms(str(rec.get("stem") or ""))))
    out: dict[int, list[tuple[str, int]]] = {}
    for members in groups.values():
        for key, code, q_no, terms in members:
            out[key] = [(c, q) for k, c, q, t in members
                        if k != key and c != code and terms and share(terms, t) >= SIBLING_SHARE]
    return out


# --------------------------------------------------------------------------- #
# rights and calibration defaults
# --------------------------------------------------------------------------- #

def rights_for_cbse_board_paper() -> RightsSchema:
    """Every question in the bank today comes from a real CBSE board paper.

    `redistribution` stays `unknown` because no written position exists --
    `docs/question-bank-api.md` section 2 says the legal question gates
    distribution and nothing here is qualified to answer it. Defaulting to
    unknown means an unsettled decision fails closed.
    """
    return RightsSchema(
        origin="CBSE",
        redistribution="unknown",
        basis="",
        attribution="Central Board of Secondary Education, previous year question paper",
    )


def assigned_calibration() -> CalibrationSchema:
    """The difficulty field is derived from mark value and Bloom level.

    That is a heuristic, not a measurement, and the payload says so.
    """
    return CalibrationSchema(measured=False, sample_size=0, basis="assigned")


def enrich_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Fill the standalone-asset fields on one wire-shaped record.

    Idempotent, and never overwrites a value that is already set -- so running
    the pipeline twice does not bump `version` twice or reset a rights decision
    someone has since recorded.

    Fields written: `rights`, `calibration`, `version`, `reviewState`,
    `language`. `provenance` is written by the relink, which is the pass that
    knows where the question sits in its source.
    """
    if not rec.get("rights"):
        rec["rights"] = rights_for_cbse_board_paper().model_dump(
            mode="json", by_alias=True)

    if not rec.get("calibration"):
        rec["calibration"] = assigned_calibration().model_dump(
            mode="json", by_alias=True)

    # Version 1 is the initial extraction. A correction is a new version, never
    # an in-place rewrite, so a consumer can reproduce what they were served.
    rec.setdefault("version", 1)
    rec.setdefault("supersededBy", None)

    # `published` because these are real, reviewed board questions already in
    # use by teachers. `draft` would be a lie that hides them from every
    # consumer filtering on state.
    rec.setdefault("reviewState", "published")

    # Already present on every record, but set explicitly so a record built by
    # some future path cannot silently omit it.
    rec.setdefault("language", "en")

    return rec


def enrich_all(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Apply `enrich_record` across a corpus; returns a small census."""
    counts = {"total": 0, "rights_set": 0, "published": 0, "with_provenance": 0,
              "with_page": 0, "with_official_scheme": 0}
    for rec in records:
        counts["total"] += 1
        enrich_record(rec)
        if rec.get("rights"):
            counts["rights_set"] += 1
        if rec.get("reviewState") == "published":
            counts["published"] += 1
        prov = rec.get("provenance") or {}
        if prov.get("sourceDocumentId"):
            counts["with_provenance"] += 1
        if prov.get("pageNumber"):
            counts["with_page"] += 1
        if (rec.get("answerScheme") or {}).get("provenance") == "cbse_marking_scheme":
            counts["with_official_scheme"] += 1
    return counts
