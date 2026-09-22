"""Question candidate pool: loads parsed pilot papers, extracts questions,
tags chapters. In-memory, built once at API startup — cheap for a pilot-sized
corpus (tens of questions); swap for a graph/DB-backed query when the corpus
grows past what fits in memory.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from ..config import Config
from ..extract.academic import extract_questions
from ..models.academic import Question
from ..models.document import ParsedDocument
from ..models.enums import DocType
from . import grades, notation
from .chapters import chapter_name, tag_chapter
from .embedding_chapter_tagger import ChapterTagCache, tag_questions

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
    "following table", "shown below", "figure given below",
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
    # The served-bank record this came from, when it came from the baked bank.
    # Kept whole because that record already IS the API wire shape; rebuilding
    # it from the stem lost the source, the parts and every SQP MCQ's key.
    record: dict | None = None

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
    """Same grade, whatever form each side is in -- 'IX', '9', '9th'. See grades.py."""
    return grades.matches(pool_grade, requested_grade)


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


# Normalised-text similarity (difflib ratio) at or above which two stems are
# one question on a printed paper. Measured on the served bank (class 10,
# 2026-09-22): Science has no keyed pair at or above 0.9, Mathematics 26 --
# the same MCQ in two board papers (cbse:q:src:f319c93abeb1185b0864d200:9 /
# cbse:q:src:b23bc2e537566a04afbf4415:15), but also "prove that root 5 / root
# 3 is irrational" and an assertion vs its reason, which `near_duplicate` now
# tells apart by their numbers and polarity (`_same_exact_tokens`). Just below
# it sit different questions that share a template: a concave lens vs a
# concave mirror problem (0.873), LCM of 576 and 512 vs HCF of 660 and 704
# (0.851).
STEM_SIMILARITY_THRESHOLD = 0.9


# What a board paper's page adds to a question the extractor ran on into:
# the footer ("5| P a g e", "# 17| P a g e"), "P.T.O." and a trailing page
# number ("... (D) 24 ´ 53 15-"). Removed before the numbers are read, or the
# same MCQ from pages 5 and 9 would count as two questions.
_PAGE_NOISE = re.compile(
    r"#?\s*\d+\s*\|\s*p\s*a\s*g\s*e|p\.\s*t\.\s*o\.?|\b\d+\s*-\s*$")
# Superscript and subscript digits as ASCII, and every dash extraction
# produces as '-': the served stems carry "(x – 2)₂" and "x²", which `\d`
# does not read, so "x² + 7x" and "x³ + 7x" had the same numbers.
_TOKEN_FOLD = str.maketrans({
    **{c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
    **{c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")},
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
})
# The tokens that make two otherwise identical sentences different questions,
# kept in the order they appear: numbers ("root 5" vs "root 3"), signs and
# relations ("x2 + 7x" vs "x2 - 7x"; a hyphen inside a word such as
# "non-leap" is not a sign), function names ("= sec A - tan A" vs "= cot A";
# LCM vs HCF), polarity ("leap" vs "non-leap") and an assertion-reason item's
# role.
_EXACT_TOKENS = re.compile(
    r"\d+(?:\.\d+)?"
    r"|[+=<>≤≥]|(?<![a-z])-|-(?![a-z])"
    r"|\b(?:sin|cos|tan|cot|sec|cosec|log|lcm|hcf)\b"
    r"|\b(?:not|non|no|never|cannot)\b"
    r"|\b(?:assertion|reason)(?=\s*\()")


class _ExactTokens(NamedTuple):
    """A stem's exact tokens with where each sits in its page-stripped text,
    so a shorter copy can be checked to be the longer one cut short."""

    tokens: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    text: str


def _exact_tokens(norm: str) -> _ExactTokens:
    text = _PAGE_NOISE.sub(" ", norm.translate(_TOKEN_FOLD))
    found = list(_EXACT_TOKENS.finditer(text))
    return _ExactTokens(tuple(m.group() for m in found),
                        tuple(m.start() for m in found),
                        tuple(m.end() for m in found), text)


# How many exact tokens the shorter stem needs before one extra token in the
# middle of the longer is read as extraction noise. The different questions
# the measure must keep apart differ by one to three tokens out of at most
# eight (RADICAL: 1 against 4; ROOT_K: 8 against 8); the one real duplicate
# differing mid-stem shares 39.
_MIDDLE_INSERT_MIN_TOKENS = 10


def _one_insertion(short: tuple[str, ...], long_: tuple[str, ...]) -> bool:
    """`long_` is `short` with one stray NUMBER added somewhere.

    Only a number: that is what extraction noise mid-stem looks like (a stray
    "12" in the Hindi-medium case study). A sign, relation, function name,
    polarity word or assertion/reason label changes the question --
    "2x - 4y = 24" and "= -24" are different (Task 906's review)."""
    i = next((k for k, (x, y) in enumerate(zip(short, long_)) if x != y), len(short))
    return short[i:] == long_[i + 1:] and long_[i][:1].isdigit()


def _same_exact_tokens(a: _ExactTokens, b: _ExactTokens) -> bool:
    """Equal, or one the other cut short: extraction cuts a question short
    at the end ("(D) 2x + x 3 =" in one paper, "... = 5" in the other), it
    does not change a number in the middle. So the shorter's tokens must
    open the longer's, and the shorter's text past its last token must be
    what the longer has before its first extra token: "Prove that 3 is an
    irrational number." against "Prove that 3 + 2 5 is an irrational
    number." (root 3 against 3 + 2 root 5, the radicals lost) shares its
    first number, but its text runs on where the other has "+ 2 5", so the
    extra numbers sit in the middle, not after a cut. A stem with none of
    them against one with some is not a cut copy either: "Prove that is an
    irrational number" lost its radicand, and matched both "root 5" and
    "root 3" otherwise.

    One stray token in the middle of a long run is a copy too: the
    Hindi-medium frequency-table case study, its words lost, reads "... 20
    -24 1 x y 12 x y" in one paper and "... 20 -24 1 x y x y" in two others
    (Task 906's code review). 39 tokens agreeing in order but one is
    extraction noise, not a different question; one token against four
    ("3" against "3 + 2 5") is. So the exception needs the shorter to carry
    `_MIDDLE_INSERT_MIN_TOKENS` and the longer exactly one more."""
    short, long_ = (a, b) if len(a.tokens) <= len(b.tokens) else (b, a)
    n = len(short.tokens)
    if not n:
        return not long_.tokens
    if (n >= _MIDDLE_INSERT_MIN_TOKENS and len(long_.tokens) == n + 1
            and _one_insertion(short.tokens, long_.tokens)):
        return True
    if long_.tokens[:n] != short.tokens:
        return False
    if n == len(long_.tokens):
        return True
    tail = short.text[short.ends[-1]:].strip()
    gap = long_.text[long_.ends[n - 1]:long_.starts[n]].strip()
    return gap.startswith(tail)


@lru_cache(maxsize=16384)
def _stem_key(text: str) -> tuple[str, frozenset[str], Counter, _ExactTokens]:
    """Normalised text, content-word signature, character counts and exact
    tokens, once per stem. The counts are difflib's `quick_ratio` bound
    precomputed: building them per pair was 90% of a swap's time on the
    class 10 Science bank."""
    norm = _normalize(text)
    return norm, _signature(text), Counter(norm), _exact_tokens(norm)


def near_duplicate(a: str, b: str) -> bool:
    """Would a teacher see these two stems as the same question on one paper?

    The pool's own gate (content-word Jaccard >= NEAR_DUPLICATE_THRESHOLD)
    counts, and so does the whole normalised text: the gate skips stems with
    fewer than four content words, which is most of a mathematics paper
    ("Which of the following is not a quadratic equation ? (A) (x - 2)^2 ..."
    has three), so two copies of one MCQ passed it and could print as Q3 and
    Q9. The cheap upper bounds on the ratio run first, so most pairs never
    reach the full comparison.

    Words alone over-flag: over every pair in the class 10 Mathematics pool
    (645 questions) they flagged 28, among them "If 3/2 is a root of kx^2 -
    x - 2 = 0, find k" vs "If 2/1 is a root of x^2 + kx - 4/5 = 0, find k",
    "root 5" vs "root 3 is irrational" and an assertion about leap years vs
    its reason about non-leap years. So the numbers, function names and
    polarity must also agree (`_same_exact_tokens`). Re-measured with it
    (2026-09-22): Mathematics 17 pairs, every one the same question as it
    prints (an MCQ in two papers, a page footer or a cut copy apart);
    Science 0 of 408 questions. With signs, folded superscripts and the
    cut-at-the-end check: Mathematics 15, Science 0, which let one
    Hindi-medium frequency-table question print twice (a stray "12" in the
    middle of one copy). With one mid-stem token allowed in a long run
    (`_MIDDLE_INSERT_MIN_TOKENS`): Mathematics 17 (the
    15 plus that question against its two other copies,
    0b075a8b722f16f272c0cc4d:34 against db32b7ab98169b625243c09e:32 and
    c42982ff54131ed0d5b7d9b0:35, and nothing else), Science 0."""
    na, sa, ca, xa = _stem_key(a)
    nb, sb, cb, xb = _stem_key(b)
    if na == nb:
        return True
    if not _same_exact_tokens(xa, xb):
        return False
    if len(sa) >= 4 and len(sb) >= 4 and _jaccard(sa, sb) >= NEAR_DUPLICATE_THRESHOLD:
        return True
    t = STEM_SIMILARITY_THRESHOLD
    total = len(na) + len(nb)
    # difflib's real_quick_ratio and quick_ratio, both upper bounds on ratio.
    if 2 * min(len(na), len(nb)) < t * total:
        return False
    if 2 * sum((ca & cb).values()) < t * total:
        return False
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio() >= t


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


def _registry_paper_rows(cfg: Config, subject: str, grade: str) -> list:
    """The registry's question papers for (subject, grade), whatever spelling
    the registry used for the grade.

    sources.grade holds '6'..'9' for classes 6-9 but 'X' and 'XII' for 10 and
    12 (registry.sqlite, 2026-09-21), so a raw `grade=?` against the pool's
    roman key would miss a class 8 paper stored as '8'. Fetch the subject's
    papers and compare grades through grades.py instead -- at most ~70 rows.
    """
    import sqlite3

    if not cfg.registry_db.exists():
        return []
    try:
        reg = sqlite3.connect(cfg.registry_db)
        reg.row_factory = sqlite3.Row
        # AGENTS.md section 3: every connection sets both pragmas.
        reg.execute("PRAGMA journal_mode=WAL")
        reg.execute("PRAGMA busy_timeout=60000")
        rows = reg.execute(
            "SELECT source_id, doc_type, subject, grade, academic_year FROM sources "
            "WHERE doc_type='question_paper' AND subject=?",
            (subject,),
        ).fetchall()
        reg.close()
    except sqlite3.OperationalError:
        # Found live 2026-09-18: this used to swallow the error into
        # rows = [] -- indistinguishable from "no question papers exist
        # for this subject/grade", which get_pool() then caches forever
        # (keyed on (subject, grade), populated once). A transient lock
        # at exactly the wrong moment silently and *permanently* zeroed
        # out that subject/grade's question pool for the rest of the
        # process's life, with nothing in any log to explain why. Log
        # loudly and re-raise instead: get_pool()'s cache assignment
        # then never runs, so the next real request retries the query
        # instead of replaying a poisoned empty result.
        log.error(
            "registry query failed for subject=%s grade=%s -- refusing to "
            "silently treat this as 'no question papers exist' (that "
            "result would be cached forever by get_pool)", subject, grade,
            exc_info=True,
        )
        raise
    return [r for r in rows if grades.matches(r["grade"], grade)]


def _baked_bank_paths(cfg: Config) -> list[Path]:
    """Where the baked-in bank may live, in the order it is tried."""
    root = Path(__file__).parents[3]
    return [
        root / "academicos-data" / "syllabus" / "questions.json",
        root / "academicos-data" / "questions.json",
        root / "frontend" / "assets" / "corpus" / "questions.json",
        cfg.data_root / "questions.json",
        cfg.data_root / "syllabus" / "questions.json",
    ]


def build_pool(cfg: Config, *, subject: str = "Science", grade: str) -> QuestionPool:
    """Loads every parsed question_paper for (subject, grade) from the registry,
    extracts + chapter-tags its questions, dedupes near-identical stems.

    Chapter tagging: Science keeps chapters.py's hand-curated keyword tagger
    (already validated, free, no network call). Every other subject with a
    real syllabus chapter list (see syllabus/cbse_syllabus.py) is tagged via
    embedding_chapter_tagger.tag_questions -- sentence-embedding cosine
    similarity against the real chapter names, cached persistently, silently
    untagged (not crashed) if no syllabus data exists yet for that subject
    (e.g. Hindi has none as of this writing). Chosen over an LLM-JSON
    approach after a live comparison: an LLM asked to emit structured JSON
    failed ~40-50% of batches for Social Science's longer (21-chapter)
    candidate list, where embeddings have no generation step to fail at all
    -- see embedding_chapter_tagger.py's docstring for the full comparison."""
    grade = grades.to_roman(grade)   # idempotent; direct callers get the same check as get_pool

    pool = QuestionPool()
    seen_hashes: set[str] = set()
    skipped_figure = 0
    skipped_mangled = 0
    skipped_near_dup = 0
    skipped_truncated = 0
    skipped_broken_options = 0
    skipped_invalid = 0
    near_dupes = _NearDuplicateIndex()

    rows = _registry_paper_rows(cfg, subject, grade)

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
            # chapters.py's keyword set is Class X Science only -- running it on
            # other subjects produces false positives: "base" (geometry) matching
            # "Acids, Bases and Salts", "reflection" (coordinate geometry) matching
            # the Light chapter, etc. 61/752 real Mathematics/X questions were
            # mistagged with a Science chapter this way before this guard was
            # added. Other subjects are tagged below, after the loop, via the LLM
            # classifier (batched) instead of reusing the Science keyword set.
            is_science = (row["subject"] or subject).strip().lower() == "science"
            cid, conf = tag_chapter(text) if is_science else (None, 0.0)
            pq = PoolQuestion(
                question=q,
                subject=row["subject"] or subject,
                # The pool key, not row["grade"]: the registry spells classes
                # 6-9 as '8' where the key and the baked path say 'VIII', and
                # this label is the grade keys.lookup matches on below.
                grade=grade,
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

    is_science_subject = subject.strip().lower() == "science"
    if not is_science_subject and pool.questions:
        untagged = [pq for pq in pool.questions if pq.chapter_id is None]
        if untagged:
            from ..syllabus.cbse_syllabus import load_syllabus
            from .mapping import grade_to_int
            syllabus = load_syllabus(subject, grade_to_int(grade))
            names_by_id = ({c.id: c.name for _, c in syllabus.all_chapters()}
                           if syllabus is not None else {})

            cache = ChapterTagCache(cfg.chapter_tags_db)
            try:
                items = [(pq.text_hash, pq.question.source_text or "") for pq in untagged]
                tags = tag_questions(cache, subject=subject, grade_label=grade,
                                     grade_int=grade_to_int(grade), items=items)
            finally:
                cache.close()
            embed_tagged = 0
            for pq in untagged:
                cid, conf = tags.get(pq.text_hash, (None, 0.0))
                if cid is not None:
                    pq.chapter_id = cid
                    pq.chapter_name = names_by_id.get(cid)
                    pq.chapter_confidence = conf
                    embed_tagged += 1
            log.info("Embedding chapter tagging: %d/%d %s questions tagged", embed_tagged, len(untagged), subject)

    if not pool.questions:
        if "PYTEST_CURRENT_TEST" not in os.environ or getattr(cfg, "_load_baked_questions", False):
            for p in _baked_bank_paths(cfg):
                if p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        # Board papers first, stably, and *before* the gates: the
                        # near-duplicate gate keeps whichever copy it sees first,
                        # and SQPs reuse board questions. A class 10/12 paper that
                        # composes today must not be silently re-composed from the
                        # SQP and CBE items the merged bank adds -- whatever order
                        # the file happens to hold them in.
                        raw_qs = sorted(
                            data.get("questions", []),
                            key=lambda it: 0 if it.get("source") == "cbse_board_paper" else 1)
                        from .schemas import QuestionSchema
                        for item in raw_qs:
                            item_subject = item.get("subject", "")
                            try:
                                item_grade_roman = grades.to_roman(item.get("grade"))
                            except ValueError:
                                continue  # a record with no readable grade cannot be served to any class
                            if subject and item_subject.strip().lower() != subject.strip().lower():
                                continue
                            if not _grade_matches(item_grade_roman, grade):
                                continue
                            # The same gates the registry path applies above: a
                            # served record is no more exempt from printing a
                            # figure it lacks, or options that never arrived.
                            stem = item["stem"]
                            if has_broken_options(stem):
                                skipped_broken_options += 1
                                continue
                            if needs_missing_figure(stem):
                                skipped_figure += 1
                                continue
                            if looks_truncated(stem):
                                skipped_truncated += 1
                                continue
                            # The record is served as-is (mapping.to_question_schema),
                            # so it must be a valid QuestionSchema. The phone contract
                            # bank_merge checks is a separate check; a record that
                            # passes it but not this would otherwise raise inside
                            # every paper request for this subject and grade.
                            try:
                                QuestionSchema.model_validate(item)
                            except ValueError as err:
                                skipped_invalid += 1
                                log.warning("baked bank record %s is not a valid QuestionSchema, "
                                            "skipped: %s", item.get("id"), err)
                                continue
                            if near_dupes.is_duplicate(stem):
                                skipped_near_dup += 1
                                continue
                            q = Question(
                                canonical_id=item["id"],
                                title=item["stem"][:100],
                                source_text=item["stem"],
                                marks=float(item.get("answerScheme", {}).get("totalMarks", 1)),
                                question_type=item.get("type", "short_answer"),
                            )
                            pq = PoolQuestion(
                                question=q,
                                subject=item_subject,
                                grade=item_grade_roman,
                                academic_year="2024",
                                chapter_id=item["chapterIds"][0] if item.get("chapterIds") else None,
                                chapter_name=None,
                                chapter_confidence=0.9,
                                text_hash=hashlib.sha256(item["stem"].encode("utf-8")).hexdigest()[:16],
                                paper_code="",
                                answer_text=item.get("answerScheme", {}).get("modelAnswer", ""),
                                record=item,
                            )
                            pool.questions.append(pq)
                        if pool.questions:
                            log.info("Loaded %d questions for %s Grade %s from baked-in question bank at %s", len(pool.questions), subject, grade, p)
                            break
                    except Exception as e:
                        log.warning("Failed loading baked-in question bank from %s: %s", p, e)

    log.info("question pool built: %d questions from %d papers, %d with official answers "
             "(skipped: %d need a figure, %d unreadable Hindi, %d near-duplicates, "
             "%d truncated/missing options, %d broken option lists, %d invalid records)",
             len(pool.questions), len(rows), keyed, skipped_figure, skipped_mangled,
             skipped_near_dup, skipped_truncated, skipped_broken_options, skipped_invalid)
    return pool


def _paper_code_of(doc: ParsedDocument) -> str:
    """The paper code printed on a board paper ("31/1/1", "1/4/1").

    Only the first page is searched: later pages repeat the code in a running
    header, but question text can contain look-alike numbers.

    This used to carry its own regex -- a third copy of the pattern, alongside
    `answer_key.normalize_paper_code` and `question_bank._CODE_HINT`. All three
    required a two-or-three digit series, so none could read an English paper
    (`1/4/1`), and fixing the other two left this one -- the copy paper
    generation actually uses -- still returning "". Delegating to the canonical
    normaliser means a corpus shape only has to be taught once.
    """
    if not doc.pages:
        return ""
    from .answer_key import normalize_paper_code

    return normalize_paper_code(doc.pages[0].text[:1500])


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


def get_pool(cfg: Config, *, subject: str = "Science", grade: str) -> QuestionPool:
    grade = grades.to_roman(grade)   # 'X', '10', '10th' are one pool, built once; raises on junk
    key = (subject, grade)
    if key not in _CACHE:
        _CACHE[key] = build_pool(cfg, subject=subject, grade=grade)
    return _CACHE[key]
