"""Question candidate pool: loads parsed pilot papers, extracts questions,
tags chapters. In-memory, built once at API startup — cheap for a pilot-sized
corpus (tens of questions); swap for a graph/DB-backed query when the corpus
grows past what fits in memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..extract.academic import extract_questions
from ..models.academic import Question
from ..models.document import ParsedDocument
from ..models.enums import DocType
from . import notation
from .chapters import chapter_name, tag_chapter

log = logging.getLogger(__name__)

_MIN_QUESTION_CHARS = 25
_ADMIN_MARKERS = ("general instructions", "candidates must", "read the following instructions")

# Phrases meaning "the answer lives in a picture we did not extract". Including
# such a question in a generated paper produces something a student cannot
# answer, so they are excluded from the selectable pool until a diagram library
# exists to carry the figure through.
_FIGURE_DEPENDENT = (
    "shown in option", "in the given figure", "in the figure given", "given diagram",
    "following diagram", "figure shown", "in the diagram", "shown in the graph",
    "given circuit", "following circuit diagram", "in the given map", "given table",
    "following table", "shown below",
)


def needs_missing_figure(text: str) -> bool:
    """True when the stem refers to a figure/diagram that was not extracted."""
    low = text.lower()
    if not any(marker in low for marker in _FIGURE_DEPENDENT):
        return False
    # If the options themselves are present as text, the question is self-contained.
    has_options = low.count("(a)") and low.count("(b)") and low.count("(c)")
    return not has_options


# A stem that trails off into a colon is an MCQ whose options never made it
# through extraction (they were a table or an image). Printing it gives the
# student a 1-mark question with nothing to choose from.
_DANGLING_LEAD_IN = re.compile(
    r"(?:\bis|\bare|\bfollowing|respectively|option|options|correct|statements?)\s*[:\-–]\s*$",
    re.I)
_TRAILING_COLON = re.compile(r"[:\-–]\s*$")


# Phrases that promise a list of choices. If they appear and no (A)-(D) options
# survived extraction, the choices were a table or an image and the question is
# unanswerable as printed.
_MCQ_LEAD_IN = re.compile(
    r"\b(?:select the correct option|which (?:one )?of the following|"
    r"the correct (?:option|answer)|choose the correct|"
    r"from the following\s*:|the appropriate term)", re.I)


def looks_truncated(text: str) -> bool:
    """True when the stem promises options/content that are not present."""
    has_options = bool(text.count("(A)")) or (text.count("(a)") and text.count("(b)"))
    if has_options:
        return False
    if _MCQ_LEAD_IN.search(text):
        return True
    if not _TRAILING_COLON.search(text):
        return False
    return bool(_DANGLING_LEAD_IN.search(text))


# Fractions and roots in MCQ options are laid out in the source PDF as a
# numerator over a denominator ("50" over "√3"); flat text extraction turns
# that into the two numbers on their own lines with the bar and radical lost.
# `clean_question_text` drops bare-number lines as page-furniture noise, which
# is right for page numbers but destroys these values, leaving e.g. option
# (a) as an empty stub. Two or more such lines inside an options block means
# the numeric content is gone, not just reformatted.
_OPTION_ORDER = ("(a)", "(b)", "(c)", "(d)")


def has_broken_fraction_options(raw: str) -> bool:
    """True when the raw (pre-clean) stem shows the stacked-fraction pattern."""
    if "(a)" not in raw.lower():
        return False
    bare_lines = sum(1 for line in raw.splitlines() if _BARE_NUMBER.match(line.strip()))
    return bare_lines >= 2


_OPTION_MARKER_RE = re.compile(r"\(([a-dA-D])\)")
# An option marker with nothing real after it ("(D) " followed straight by
# the next marker, or the end of the stem) — a shorter, non-fraction sibling
# of the stacked-fraction case: the value was a single token that got dropped
# entirely rather than split across lines.
_MIN_OPTION_CONTENT_CHARS = 2


def has_broken_options(text: str) -> bool:
    """True when the cleaned stem carries a *partial* or *empty* option list.

    A legitimate CBSE "(a) ... OR (b) ..." internal choice always survives
    cleaning as a lone leading "(a)" (the OR split keeps only the first
    alternative), so that case must stay allowed. What must not: any option
    letter appearing without every letter before it also present — that
    ordering is physically impossible in the source and only happens when
    extraction destroyed the earlier options (see `has_broken_fraction_options`)
    — or an option marker present but immediately followed by the next marker
    (or the end of the stem) with no content in between.
    """
    low = text.lower()
    present = [m for m in _OPTION_ORDER if m in low]
    if not present:
        return False
    expected = _OPTION_ORDER[: _OPTION_ORDER.index(present[-1]) + 1]
    if any(m not in low for m in expected):
        return True

    matches = list(_OPTION_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return False
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end():end].strip(" .:;")
        if len(content) < _MIN_OPTION_CONTENT_CHARS:
            return True
    return False


def _is_mostly_non_latin(text: str) -> bool:
    """CBSE papers print each question in Hindi then English; the extractor's
    q_no regex matches both, producing two Question objects with the same
    canonical id. Skip the Hindi copy so ids stay unique and the pool stays
    English-only for now."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if ord(c) > 0x2FF)
    return non_latin / len(letters) > 0.3


# Characters that belong in ordinary English question text. Anything outside
# this set is "noise" for the purposes of the mangled-encoding check below.
_CLEAN_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " .,;:'\"()[]/-–—+=%°?!*&#\n\t→×÷±≤≥√∞²³"
)
# Sigils that show up constantly in legacy-font Devanagari but almost never in
# real English prose: `{H$gr Xn©U`, `à{H«$`m`, `_hmoX`.
_MANGLE_SIGILS = set("${}|~^\\`©¡«»¥¤§¨ª¬¯µ¶·¸¹º½¾")


def looks_mangled(text: str) -> bool:
    """True for Devanagari rendered through a legacy 8-bit font.

    Those PDFs encode Hindi in a custom Latin mapping, so the extracted string
    is ASCII/Latin-1 and slips past `_is_mostly_non_latin` — but it is unreadable
    and must never reach a generated paper.
    """
    if not text:
        return True
    sample = text[:600]
    noise = sum(1 for c in sample if c not in _CLEAN_CHARS)
    if noise / len(sample) > 0.15:
        return True
    sigils = sum(1 for c in sample if c in _MANGLE_SIGILS)
    if sigils / len(sample) > 0.03:
        return True
    # Real English has few tokens containing $ or { ; mangled Hindi is full of them.
    tokens = sample.split()
    if tokens:
        odd = sum(1 for t in tokens if any(ch in t for ch in "${}|"))
        if odd / len(tokens) > 0.08:
            return True
    return False


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Page furniture the block extractor sweeps up between question numbers:
# paper codes (*31/1/1*), mirrored footers (# *ECNEICS*), bare page numbers, P.T.O.
_STAR_MARKER = re.compile(r"\*[A-Za-z0-9/\-]+\*")
_BARE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_PTO = re.compile(r"^\s*P\.?\s*T\.?\s*O\.?\s*\.?\s*$", re.I)
_LEADING_QNO = re.compile(r"^\s*\d{1,3}\s*[.)\]]\s*")
_TRAILING_OR = re.compile(r"\s*\b(\d+\s+)?OR\s*$", re.I)

# Paper furniture that sits *between* questions and gets swept into the previous
# one: section banners and the rubric that introduces the next block of questions.
_SECTION_BANNER = re.compile(
    r"\s*\b(?:SECTION|खण्ड)\s*[-–—]?\s*[A-EIVX]\b.*$", re.I | re.S)
# "… respectively : For Questions number 17 to 20, two statements are given …"
# The lead-in word ("For"/"In") belongs to the rubric, so consume it too —
# otherwise a stray "For" is left dangling at the end of the question.
_QUESTION_RANGE = re.compile(
    r"\s*(?:\b(?:For|In|Read)\s+)?Question(?:s)?\s*(?:no\.?|number)?\s*\d+\s*"
    r"(?:to|and|-|–)\s*\d+.*$", re.I | re.S)
_TRAILING_MARK_DIGIT = re.compile(r"\s+\d{1,2}\s*$")
# Bare paper codes (31/2/2) and running-header fragments (H 4 H, # 12 #) that the
# star-marker rule misses because this printing has no asterisks around them.
_PAPER_CODE = re.compile(r"\b\d{2,3}/\d{1,2}/\d{1,2}\b")
_HEADER_FRAGMENT = re.compile(r"\s+(?:[A-Z#]\s+\d{1,3}\s+[A-Z#]|#\s*\d{1,3}|\d{1,3}\s*#)(?=\s|$)")
# Running-header debris left after the code is removed: repeated marker letters
# and a stray "Page"/"Set" label ("… B(5, 6) ? 430/2/1 JJJJ Page").
_HEADER_DEBRIS = re.compile(
    r"\s+(?:([A-Z])\1{2,}|Page|Set|SET)(?=\s|$)")
# "… get rusted. 3 OR (ii) Write a test …" — CBSE prints internal-choice
# alternatives inline. Keep the first alternative; the second is a separate
# question the teacher can swap in.
_INTERNAL_CHOICE = re.compile(r"\s+\d{0,2}\s*\bOR\b\s+", re.I)

# Source PDFs draw arrows/maths with the Adobe Symbol font; the extractor surfaces
# those glyphs in the Private Use Area (U+F0xx), which no normal font can render —
# they came out as black boxes in the exported paper. Map the ones that carry
# meaning, drop the rest.
_SYMBOL_PUA = {
    "": "←", "": "→", "": "↑", "": "↓",
    "": "×", "": "÷", "": "°", "": "±",
    "": "∞", "": "√", "": "≠", "": "≤",
    "": "≥", "": "→", "": "←",
}
# Arrow-shaft / line-extension pieces that only exist to lengthen a glyph.
_ARROW_SHAFT = re.compile(r"[⎯]+")
_PUA = re.compile(r"[-]")
_MULTI_ARROW = re.compile(r"(?:\s*→\s*){2,}")


def normalize_symbols(text: str) -> str:
    """Turn Symbol-font PUA codepoints into real Unicode; drop undecodable ones."""
    for pua, real in _SYMBOL_PUA.items():
        text = text.replace(pua, real)
    text = _ARROW_SHAFT.sub("→", text)   # a run of shaft pieces *is* an arrow
    text = _MULTI_ARROW.sub(" → ", text)  # collapse shaft+head into one arrow
    text = _PUA.sub("", text)                  # anything still unmapped is undrawable
    return text


def clean_question_text(raw: str) -> str:
    """Strip page furniture and the bilingual Hindi tail from an extracted stem.

    The block-level extractor accumulates every text block between two question
    numbers, which on a real CBSE paper means the English question *plus* page
    headers/footers and the Hindi rendering of the same question. Keep the
    English question only.
    """
    kept: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The Hindi rendering follows the English one — stop at the first line of it.
        if _DEVANAGARI.search(stripped):
            break
        if _STAR_MARKER.search(stripped):
            continue
        if _BARE_NUMBER.match(stripped) or _PTO.match(stripped):
            continue
        kept.append(stripped)

    text = " ".join(kept)
    text = normalize_symbols(text)
    text = _PAPER_CODE.sub(" ", text)          # 31/2/2 printed in the running head
    text = _HEADER_FRAGMENT.sub(" ", text)
    text = _HEADER_DEBRIS.sub(" ", text)
    text = _LEADING_QNO.sub("", text)          # we renumber questions per section
    # Cut at the next section banner / question-range rubric: everything after it
    # belongs to the paper, not to this question.
    text = _SECTION_BANNER.sub("", text)
    text = _QUESTION_RANGE.sub("", text)
    text = _INTERNAL_CHOICE.split(text)[0]     # keep the first choice alternative
    text = _TRAILING_OR.sub("", text)          # dangling internal-choice marker
    text = _TRAILING_MARK_DIGIT.sub("", text)  # the mark value printed after the stem
    text = " ".join(text.split()).strip()
    # Restore subscripts/superscripts flattened by PDF text extraction, so
    # chemistry and physics read correctly on the printed paper.
    return notation.restore(text)


@dataclass
class PoolQuestion:
    question: Question
    subject: str
    grade: str
    academic_year: str | None
    chapter_id: str | None
    chapter_name: str | None
    chapter_confidence: float
    text_hash: str
    # Board paper this came from ("31/1/1"), the join key back to the official
    # marking scheme. Empty when the paper's code could not be read.
    paper_code: str = ""
    # Filled from the marking scheme when one covers this paper. Without it a
    # question can be printed but not auto-scored.
    correct_option: str | None = None
    answer_text: str = ""
    value_points: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.question.canonical_id

    @property
    def marks(self) -> int:
        return int(self.question.marks or 0)

    @property
    def has_answer(self) -> bool:
        return bool(self.correct_option or self.answer_text)


@dataclass
class QuestionPool:
    questions: list[PoolQuestion] = field(default_factory=list)

    def filter(self, *, subject: str | None = None, grade: str | None = None,
               chapter_ids: list[str] | None = None) -> list[PoolQuestion]:
        out = self.questions
        if subject:
            out = [q for q in out if q.subject.lower() == subject.lower()]
        if grade:
            out = [q for q in out if _grade_matches(q.grade, grade)]
        if chapter_ids:
            wanted = set(chapter_ids)
            out = [q for q in out if q.chapter_id in wanted]
        return out


def _grade_matches(pool_grade: str, requested_grade: str) -> bool:
    """Compares e.g. pool grade 'X' against requested grade '10' or '10th'."""
    roman = {"10": "X", "10th": "X", "x": "X"}
    return pool_grade.upper() == roman.get(requested_grade.strip().lower(), requested_grade.strip().upper())


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


_DEDUPE_STOPWORDS = {
    "the", "a", "an", "of", "in", "is", "are", "to", "and", "or", "for", "on", "at",
    "it", "its", "this", "that", "which", "with", "from", "by", "as", "be", "following",
}
NEAR_DUPLICATE_THRESHOLD = 0.75


def _signature(text: str) -> frozenset[str]:
    """Content-word set used for near-duplicate comparison."""
    words = re.findall(r"[a-z][a-z0-9\-]+", text.lower())
    return frozenset(w for w in words if w not in _DEDUPE_STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


class _NearDuplicateIndex:
    """Rejects questions that restate one already in the pool.

    CBSE issues several variants of the same paper (31-1-1, 31-1-2, 31-1-3) whose
    questions differ only in wording or are truncated differently by extraction.
    Exact-hash dedupe lets those through, and a generated paper then asks the same
    thing twice. Compares against accepted questions bucketed by shared rare words
    so this stays far cheaper than all-pairs.
    """

    def __init__(self, threshold: float = NEAR_DUPLICATE_THRESHOLD):
        self.threshold = threshold
        self._buckets: dict[str, list[frozenset[str]]] = {}

    def is_duplicate(self, text: str) -> bool:
        sig = _signature(text)
        if len(sig) < 4:
            return False
        # Compare only against questions sharing one of this question's rarest words.
        keys = sorted(sig)[:8]
        seen: set[int] = set()
        for k in keys:
            for other in self._buckets.get(k, ()):
                if id(other) in seen:
                    continue
                seen.add(id(other))
                if _jaccard(sig, other) >= self.threshold:
                    return True
        for k in keys:
            self._buckets.setdefault(k, []).append(sig)
        return False


def build_pool(cfg: Config, *, subject: str = "Science", grade: str = "X") -> QuestionPool:
    """Loads every parsed question_paper for (subject, grade) from the registry,
    extracts + chapter-tags its questions, dedupes near-identical stems."""
    import sqlite3

    pool = QuestionPool()
    seen_hashes: set[str] = set()
    skipped_figure = 0
    skipped_mangled = 0
    skipped_near_dup = 0
    skipped_truncated = 0
    skipped_broken_options = 0
    near_dupes = _NearDuplicateIndex()

    reg = sqlite3.connect(cfg.registry_db)
    reg.row_factory = sqlite3.Row
    rows = reg.execute(
        "SELECT source_id, doc_type, subject, grade, academic_year FROM sources "
        "WHERE doc_type='question_paper' AND subject=? AND grade=?",
        (subject, grade),
    ).fetchall()
    reg.close()

    # Official answers, if the marking schemes have been parsed. Absent store =
    # questions still load, they just cannot be auto-scored.
    keys = _open_answer_keys(cfg)
    keyed = 0

    for row in rows:
        source_id = row["source_id"]  # e.g. "src:af50a3f7e666fdea9cee608f"
        parse_path = cfg.parse_dir / f"{source_id.replace(':', '_')}.json"
        if not parse_path.exists():
            log.warning("no parse output for %s, skipping", source_id)
            continue
        doc = ParsedDocument.model_validate(json.loads(parse_path.read_text(encoding="utf-8")))
        if doc.doc_type != DocType.QUESTION_PAPER:
            doc = doc.model_copy(update={"doc_type": DocType.QUESTION_PAPER})
        paper_code = _paper_code_of(doc)

        for q in extract_questions(doc, DocType.QUESTION_PAPER):
            raw = (q.source_text or q.title or "").strip()
            if _is_mostly_non_latin(raw):
                continue
            if has_broken_fraction_options(raw):
                skipped_broken_options += 1
                continue
            text = clean_question_text(raw)
            if looks_mangled(text):
                skipped_mangled += 1
                continue
            if len(text) < _MIN_QUESTION_CHARS:
                continue
            if has_broken_options(text):
                skipped_broken_options += 1
                continue
            low = text.lower()
            if any(marker in low for marker in _ADMIN_MARKERS):
                continue
            if needs_missing_figure(text):
                skipped_figure += 1
                continue
            if looks_truncated(text):
                skipped_truncated += 1
                continue

            text_hash = hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()[:16]
            if text_hash in seen_hashes:
                continue
            if near_dupes.is_duplicate(text):
                skipped_near_dup += 1
                continue
            seen_hashes.add(text_hash)

            q.source_text = text
            # tag_chapter's keyword set is Class X Science only (see chapters.py's
            # docstring) — running it on other subjects produces false positives:
            # "base" (geometry) matching "Acids, Bases and Salts", "reflection"
            # (coordinate geometry) matching the Light chapter, etc. 61/752 real
            # Mathematics/X questions were mistagged with a Science chapter this
            # way before this guard was added.
            is_science = (row["subject"] or subject).strip().lower() == "science"
            cid, conf = tag_chapter(text) if is_science else (None, 0.0)
            pq = PoolQuestion(
                question=q,
                subject=row["subject"] or subject,
                grade=row["grade"] or grade,
                academic_year=row["academic_year"],
                chapter_id=cid,
                chapter_name=chapter_name(cid),
                chapter_confidence=conf,
                text_hash=text_hash,
                paper_code=paper_code,
            )
            if keys and paper_code and q.q_no:
                official = keys.lookup(paper_code, int(q.q_no),
                                       subject=pq.subject, grade=pq.grade)
                if official is not None:
                    pq.correct_option = official.correct_option
                    pq.answer_text = official.answer_text
                    pq.value_points = [p for p, _ in official.value_points]
                    keyed += 1
            pool.questions.append(pq)

    if keys:
        keys.close()

    log.info("question pool built: %d questions from %d papers, %d with official answers "
             "(skipped: %d need a figure, %d unreadable Hindi, %d near-duplicates, "
             "%d truncated/missing options, %d broken option lists)",
             len(pool.questions), len(rows), keyed, skipped_figure, skipped_mangled,
             skipped_near_dup, skipped_truncated, skipped_broken_options)
    return pool


# Paper code as printed on a board paper ("31/1/1"), used to find its marking
# scheme. Only the first page is searched: later pages repeat it in a running
# header, but question text can contain look-alike numbers.
_PAPER_CODE_RE = re.compile(r"\b(\d{2,3})\s*[/-]\s*(\d)\s*[/-]\s*(\d)\b")


def _paper_code_of(doc: ParsedDocument) -> str:
    if not doc.pages:
        return ""
    m = _PAPER_CODE_RE.search(doc.pages[0].text[:1500])
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def _open_answer_keys(cfg: Config):
    """Opens the answer-key store, or returns None if it has not been built."""
    from .answer_key import AnswerKeyStore, default_db

    path = default_db(cfg)
    if not path.exists():
        log.warning("no answer-key store at %s — questions will have no marking "
                    "scheme; run scripts/build_answer_keys.py --run", path)
        return None
    return AnswerKeyStore(path)


_CACHE: dict[tuple[str, str], QuestionPool] = {}


def get_pool(cfg: Config, *, subject: str = "Science", grade: str = "X") -> QuestionPool:
    key = (subject, grade)
    if key not in _CACHE:
        _CACHE[key] = build_pool(cfg, subject=subject, grade=grade)
    return _CACHE[key]
