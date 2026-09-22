"""Parser for NCERT's *Exemplar Problems* books, classes 6-10, Maths and Science.

Why this source
---------------
Classes 6-9 served 15-64 questions per subject (measured with
``bank_merge.compose()`` on 2026-09-22), mostly 1-markers, so no real paper
could be filled. NCERT publishes Exemplar Problems for Mathematics and Science
at every class from 6 to 10 -- and, unlike almost every other free question
source, the same books carry NCERT's own answers. That is what rule Q1 needs:
without an answer key, no question.

Where the answers are, established by reading the 165 files
----------------------------------------------------------
Every book keeps ALL its answers in one separate file at the end (``*an.pdf``,
class 7 maths ``gemp1a1.pdf``); no chapter file carries its own. Each answers
file is cut into blocks by its own headings, and pairing follows the block:

  Maths 6-8     unit files ``(C) Exercise`` numbered 1..N across the unit;
                answers under ``Unit N``. Section kinds are the book's own
                instructions ("In questions 39 to 98, state whether ... true
                (T) or false (F)"); one that names no range runs until the
                next instruction or the first question it does not fit
                (`_close_open_kinds`).
  Maths 9-10    chapter files ``EXERCISE N.1`` .. ``N.4``, numbering afresh in
                each; answers under the same ``EXERCISE N.k``. Kinds from the
                ``(B) Multiple Choice`` / ``(C) Short Answer Questions with
                Reasoning`` / ``(D) Short Answer`` / ``(E) Long Answer`` heads.
  Science 6-10  chapter files numbered 1..N through ``MULTIPLE CHOICE``, ``VERY
                SHORT ANSWER``, ``SHORT ANSWER``, ``LONG ANSWER`` sections;
                answers per chapter, each block opening with the MCQ heading.
                Classes 9 and 10 print the chapter number only as display type
                that PyMuPDF orders at the END of the chapter's first answer
                page, so a block takes the one chapter number printed on the
                page where its MCQ heading stands.

Not parsed, deliberately: the maths solved examples (their answers are inline
prose, not keyed), the "(D) Activities / Applications, Games and Puzzles"
parts, and the sample question papers and appendices of classes 9 and 10
(`NOT_PARSED` says which, and why).

Pairing rule
------------
An answer is attached only when its (chapter, exercise, number) matches the
question's exactly, and only when that number occurs once in its block. A
number seen twice is ambiguous and pairs with nothing; so is an answer that
swallowed the text of a number the parser could not see (``_numbered_entries``).
An MCQ answer must be option letters, and every letter must name an option the
question actually has; anything else is ``answer-not-an-option``. A chapter
file is paired with an answer block only when the file itself proves its
chapter number (``_verify_chapter``). A confidently wrong key is worse than no
question, so every doubt resolves to exclusion, and every exclusion is counted
by reason rather than dropped.

Text fidelity
-------------
Plain ``page.get_text()`` loses the structure maths and science text depend on:
``1.9 x 10^11`` extracts as ``1.9 x 1011`` and ``cm^3`` as ``cm3``, so the stem
reads as a confidently wrong number. This module renders pages from
``get_text("dict")`` spans instead:

  * a span set smaller and raised is a superscript (``10^11`` -> ``10¹¹``), set
    smaller and lowered a subscript (``H2O`` -> ``H₂O``);
  * each font's encoding is decided from its whole text (`classify_fonts`):
    Private-Use-Area ASCII, Adobe Symbol, or a font shifted below ASCII;
  * what cannot be linearised faithfully -- a built-up fraction (same-size
    spans stacked on different baselines), bracket pieces (U+F8EB-F8FE), an
    unmapped glyph, a run of spaces where a formula was typeset separately --
    is replaced by U+FFFD, and any question or answer containing U+FFFD is
    excluded as ``symbol-loss`` instead of being served with a formula missing;
  * so is a line that a small vector drawing sits on (`_mark_drawn_math`):
    classes 9-10 draw the radical sign and the repeating-decimal bar, and
    without this "2√3 + √3" was served as "2 3+ 3"; and so are the two
    short lines on either side of a drawn fraction bar (`_lose_fraction`),
    which the stacked-line test misses when their boxes do not overlap;
  * text in a tinted feature panel, or emitted out of reading order, is
    marked (`_out_of_order_blocks`) and excludes whatever question takes it
    in (``reading-order``).

PDF library
-----------
Everything above rests on the span boxes, baselines and drawings PyMuPDF
reports, and those differ between releases. Loss detection is where it shows:
the stacked-fraction test compares line boxes, and under PyMuPDF 1.27.2.3 it
misses fractions that 1.28.2 catches -- 3 of 55 tests fail and the build
writes 10 more records whose fractions were silently flattened ("7 over 8 - x"
served as "7 8 – x"). Nothing in the output says which library made it, so
`read_pdf` refuses to run on any version but `VERIFIED_PYMUPDF`, the one the
bank was checked on; pyproject.toml pins the same version. Moving to another
release means re-verifying the bank on it, then changing the constant.

A question that leans on data it does not carry -- a figure, a table, "In
question 42 above", the data block of a range instruction -- is excluded
(``figure-unavailable``, ``shared-stimulus``), as is a fill-in-the-blank whose
blank was a drawn rule (``blank-lost``). So is a question whose book answer
only points elsewhere -- "See pages 223 and 224 of ... textbook", "Take help
of elders or use internet" -- because it is no answer key
(``answer-is-a-pointer``), and one whose answer has a labelled part with
nothing in it, "(a) (b) Football", because part (a) was drawn
(``answer-part-missing``).

Book versions
-------------
Exemplar for classes 6-8 was written for the OLD NCERT books; schools now teach
the new ones (Curiosity, Ganita Prakash). Chapters are therefore not forced
onto the current syllabus: every record carries its Exemplar chapter as
``metadata.exemplarChapter`` and gets ``chapterIds`` only when that title
equals a chapter name in ``academicos-data/syllabus/<Subject>_<grade>.json``
(`map_chapter`). Mapping onto the current books' topics is the tagger's job.

Marks (a declared convention, NOT from the book)
------------------------------------------------
The Exemplar books do not print marks. So paper blueprints can use these items,
marks are assigned by section kind at CBSE-typical values: MCQ, fill-in-the-
blank and true/false 1; VSA -- including maths 9-10 "short answer with
reasoning" -- 2; SA 3; LA 5. Maths 6-8 exercise questions outside the book's
objective ranges are SA. Every record says so in
``metadata.marksBasis = "assigned_by_section_kind"``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

# See "PDF library" in the module docstring. pyproject.toml and uv.lock pin
# the same version.
VERIFIED_PYMUPDF = "1.28.2"

SOURCE = "ncert_exemplar"
PROVENANCE = "ncert_exemplar_answer"
LOSS = "�"

# --------------------------------------------------------------------------- #
# section kinds and marks
# --------------------------------------------------------------------------- #

MCQ_SINGLE = "mcq_single"
MCQ_MULTIPLE = "mcq_multiple"
FILL_BLANK = "fill_blank"
TRUE_FALSE = "true_false"
VSA = "vsa"
SA = "sa"
LA = "la"
MATCHING = "matching"

OPTION_KINDS = frozenset({MCQ_SINGLE, MCQ_MULTIPLE})

# See "Marks" in the module docstring: a declared convention, not the book's.
MARKS_BY_KIND = {
    MCQ_SINGLE: 1, MCQ_MULTIPLE: 1, FILL_BLANK: 1, TRUE_FALSE: 1,
    VSA: 2, SA: 3, LA: 5, MATCHING: 3,
}

_TYPE_BY_KIND = {
    MCQ_SINGLE: "mcq", MCQ_MULTIPLE: "mcq",
    FILL_BLANK: "very_short_answer", TRUE_FALSE: "very_short_answer",
    VSA: "very_short_answer", SA: "short_answer", LA: "long_answer",
    MATCHING: "short_answer",
}


@dataclass
class ExemplarQuestion:
    """One question from an Exemplar book, with its official answer if paired."""

    grade: int
    subject: str                     # "Mathematics" | "Science"
    chapter_number: int              # the Exemplar's own unit/chapter number
    chapter_name: str
    section_kind: str
    section_label: str
    number: str                      # as printed within its scope: "12"
    stem: str
    source_file: str
    page: int
    scope: str = ""                  # maths 9-10 exercise, "2.3"; "" elsewhere
    options: list[str] = field(default_factory=list)
    option_labels: list[str] = field(default_factory=list)
    options_error: str = ""          # why the options could not be trusted
    answer_letters: list[str] = field(default_factory=list)   # "a".."e"
    answer_text: str = ""
    answer_file: str = ""
    answer_page: Optional[int] = None
    answer_error: str = ""           # why an answer found was refused
    chapter_verified: bool = True    # the file proved its chapter number
    shared_stimulus: bool = False    # needs a table/figure printed above its range
    instruction: str = ""            # the book's range instruction, put before the stem
    open_run: int = 0                # >0: kind from an instruction naming no range (`_close_open_kinds`)
    raw_lines: list[str] = field(default_factory=list, repr=False)

    @property
    def key(self) -> tuple[int, str, str]:
        """What an answer must match exactly: (chapter, exercise, number)."""
        return (self.chapter_number, self.scope, self.number)

    @property
    def has_answer(self) -> bool:
        if self.section_kind in OPTION_KINDS:
            return bool(self.answer_letters)
        return bool(self.answer_text.strip())

    @property
    def record_id(self) -> str:
        return (f"exemplar:q:{self.grade}:{self.subject.lower()}:{self.chapter_number}:"
                f"{self.scope or '-'}:{self.number}")


# --------------------------------------------------------------------------- #
# text layer: dict spans -> faithful lines
# --------------------------------------------------------------------------- #


# Adobe Symbol encoding for the glyphs PyMuPDF surfaces as U+F0xx. Only the
# code points measured in the class 11 Exemplar files (where this table was
# built) plus their obvious neighbours; anything else becomes U+FFFD rather
# than a guess.
_SYMBOL = {
    0x20: " ", 0x21: "!", 0x28: "(", 0x29: ")", 0x2B: "+", 0x2C: ",",
    0x2D: "−", 0x2E: ".", 0x2F: "/", 0x3A: ":", 0x3B: ";", 0x3C: "<",
    0x3D: "=", 0x3E: ">",
    **{0x30 + i: str(i) for i in range(10)},
    0x41: "Α", 0x42: "Β", 0x43: "Χ", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ",
    0x47: "Γ", 0x4C: "Λ", 0x50: "Π", 0x51: "Θ", 0x53: "Σ", 0x57: "Ω",
    0x5C: "∴", 0x5B: "[", 0x5D: "]",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ",
    0x67: "γ", 0x68: "η", 0x69: "ι", 0x6B: "κ", 0x6C: "λ", 0x6D: "μ",
    0x6E: "ν", 0x6F: "ο", 0x70: "π", 0x71: "θ", 0x72: "ρ", 0x73: "σ",
    0x74: "τ", 0x75: "υ", 0x77: "ω", 0x78: "ξ", 0x79: "ψ", 0x7A: "ζ",
    0xA2: "′", 0xA3: "≤", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→",
    0xAF: "↓", 0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×",
    0xB5: "∝", 0xB6: "∂", 0xB7: "•", 0xB8: "÷", 0xB9: "≠", 0xBA: "≡",
    0xBB: "≈", 0xBC: "…", 0xC5: "⊕", 0xC6: "∅", 0xD7: "·", 0xDB: "⇔",
    0xDE: "⇒",
    # 0xBE is the horizontal arrow extender. Chemistry stretches an arrow
    # only to set reaction conditions over and under it ("Mo₂O₃", "523 K,
    # 100 atm"), and those extract as small text shuffled after the product
    # ("→ HCHO + H2O Mo O"), so a stretched arrow means lost conditions.
    0xBE: LOSS,
}
# U+F8E7 is the Apple/Adobe arrow extender, same role as Symbol 0xBE.
_PUA_DROP = {0xF8E7}
# U+F8EB..F8FE are the pieces of tall brackets and braces -- they only occur
# around built-up (2-D) mathematics, which linear text cannot represent.
_PUA_BUILT_UP = range(0xF8E0, 0xF900)

_SUP = str.maketrans("0123456789+-−–=()n*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁻⁼⁽⁾ⁿ*")
_SUB = str.maketrans("0123456789+-−–=()aeoxhklmnpst",
                     "₀₁₂₃₄₅₆₇₈₉₊₋₋₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ")
_SUP_OK = set("0123456789+-−–=()n* ")
_SUB_OK = set("0123456789+-−–=()aeoxhklmnpst ")
_ORDINAL = {"st", "nd", "rd", "th"}

# A formula typeset outside the text line leaves a gap of spaces where it sat
# (measured: physics 2.8 "report correct value for            as:" -- the
# square root sits in a separate block). Three spaces occur between ordinary
# tokens ("H2SO4   +  2NaOH"); six never did in running prose.
_GAP = re.compile(r"(?<=\S) {6,}(?=\S)")


# How a font's text is encoded, decided per document by `classify_fonts`.
ENC_SYMBOL = "symbol"      # Adobe Symbol: U+F0xx through `_SYMBOL`
ENC_ASCII = "ascii"        # U+F0xx is ASCII moved into the PUA (U+F055 U+F04E = "UN")
ENC_SHIFT30 = "shift30"    # every character 0x1E below its real one ("/WNVKRNG" = "Multiple")
ENC_SHIFT29 = "shift29"    # every character 0x1D below its real one ("&21752/" = "CONTROL")
ENC_UNKNOWN = "unknown"    # PUA glyphs with no established meaning: lost

_PUA_SPACE = chr(0xF020)
_SHIFTS = {ENC_SHIFT30: 0x1E, ENC_SHIFT29: 0x1D}
# Code points where the Symbol encoding and ASCII agree (digits, and the
# punctuation Symbol leaves alone).
_SYMBOL_EQUALS_ASCII = frozenset(b"0123456789!#%&()+,./:;<=>?[]_{|}")
# Words that tell the right shift from the wrong one when a font's text has
# no space character to decide it (a heading font setting only "CHAPTER").
_SHIFT_VOCABULARY = ("chapter", "answer", "question", "choice", "multiple", "short",
                     "long", "the ", " and ", " of ", "which", "science")


def _unshift(text: str, shift: int) -> str:
    """Undo a shifted font. A real space (U+0020) is kept: PyMuPDF inserts it
    between words of any font, and in these fonts the space glyph is 0x02 or
    0x03. A character above ASCII has no known meaning here and is lost."""
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == " " or ch in "\t\n\r":
            out.append(ch)
        elif 0x01 <= cp <= 0x7E - shift:
            out.append(chr(cp + shift))
        else:
            out.append(LOSS)
    return "".join(out)


def _shift_encoding(t: str) -> Optional[str]:
    """ENC_SHIFT30 / ENC_SHIFT29 if this font's whole text is shifted, else None.

    Two shifts occur (measured on class 10 science): the section-heading
    fonts use 0x1E ("/WNVKRNG" = "Multiple"), and chapter 7's body font --
    every question of the chapter -- uses 0x1D ("&21752/" = "CONTROL"). The
    space glyph (0x02 or 0x03) says which; without one, the shift that spells
    the book's own words wins, and a font that spells none is not treated as
    shifted at all.
    """
    cands = [enc for enc, s in _SHIFTS.items()
             if all(0x01 <= ord(c) <= 0x7E - s or c.isspace() or ord(c) >= 0x80 for c in t)]
    if not cands:
        return None
    # Control characters never occur in real text. Without them the shifted
    # capitals are digits and punctuation ("#059'45" is "ANSWERS", "%*#26'4"
    # "CHAPTER"): accept only a font whose text has no letters of its own.
    ctrl = any(ord(c) < 0x20 and c not in "\t\n\r" for c in t)
    capitals = (not re.search(r"[A-Za-z]", t)
                and sum(c in "#$%&'*" for c in t) >= 2)
    if not (ctrl or capitals):
        return None
    if ctrl and "\x02" in t and "\x03" not in t and ENC_SHIFT30 in cands:
        return ENC_SHIFT30
    if ctrl and "\x03" in t and "\x02" not in t and ENC_SHIFT29 in cands:
        return ENC_SHIFT29
    best, score = None, 0
    for enc in cands:
        low = _unshift(t, _SHIFTS[enc]).lower()
        hits = sum(low.count(w) for w in _SHIFT_VOCABULARY)
        if hits > score:
            best, score = enc, hits
    return best


def classify_fonts(dicts: Iterable[dict]) -> dict[str, str]:
    """Each font's encoding, from all the text it sets in one document.

    Why per font and per document: the class 6-10 books set most headings and
    every maths display title in subset TrueType fonts whose glyphs sit in the
    Private Use Area at ASCII + 0xF000 -- U+F055 U+F04E U+F049 U+F054 is
    "UNIT". The class-11 draft read every U+F0xx through the Symbol encoding,
    which turned those headings into Greek ("UNIT" -> "ΥΝΙΤ", measured on all
    three class 6-8 maths books) and would have done the same to any stem set
    in such a font. Class 10 science sets headings, and the whole of chapter
    7, in fonts shifted below ASCII (`_shift_encoding`), which only the whole
    font's text reveals: "%*#26'4" alone ("CHAPTER") looks like punctuation.

    A font counts as ASCII-in-PUA if all its PUA glyphs are printable ASCII
    positions that spell at least one three-letter word, or if they are all
    digits and punctuation on which Symbol and ASCII agree; as Symbol-encoded
    if its name says Symbol or it uses the upper half (U+F080-F0FF: ×, ÷, ≠,
    ∴ ...). Anything else -- single PUA letters that could be Latin or Greek
    (U+F06C: "l" or "λ"?) -- is unknown, and its glyphs become U+FFFD rather
    than a guess.
    """
    texts: dict[str, list[str]] = {}
    for d in dicts:
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    texts.setdefault(span.get("font", ""), []).append(span.get("text", ""))
    enc: dict[str, str] = {}
    for font, parts in texts.items():
        t = "".join(parts)
        if "symbol" in font.lower():
            enc[font] = ENC_SYMBOL
            continue
        shifted = _shift_encoding(t)
        if shifted:
            enc[font] = shifted
            continue
        pua = [c for c in t if 0xF000 <= ord(c) <= 0xF0FF and c != _PUA_SPACE]
        if not pua:
            continue
        # The same font often sets plain ASCII too (measured: class 6 maths
        # TT371T00 sets "UNIT 2" in the PUA on one page and plain on the
        # next), which is consistent with -- not evidence against -- identity.
        ascii_ = "".join(chr(ord(c) - 0xF000) for c in pua)
        if (all(0xF021 <= ord(c) <= 0xF07E for c in pua)
                and re.search(r"[A-Za-z]{3,}", ascii_)):
            enc[font] = ENC_ASCII
        elif all(ord(c) - 0xF000 in _SYMBOL_EQUALS_ASCII for c in pua):
            # Digits and the punctuation both encodings share: whichever the
            # font is, PUA "1" "2" is "12". Measured: class 9 science prints
            # its chapter numbers this way, and calling the font unknown lost
            # the number that ties a chapter file to its answers.
            enc[font] = ENC_ASCII
        elif any(ord(c) >= 0xF080 for c in pua):
            enc[font] = ENC_SYMBOL
        else:
            enc[font] = ENC_UNKNOWN
    return enc


def map_glyphs(text: str, encoding: Optional[str] = ENC_SYMBOL) -> str:
    """Resolve a span's glyphs under its font's encoding; U+FFFD where no
    faithful mapping exists."""
    if encoding in _SHIFTS:
        return _unshift(text, _SHIFTS[encoding])
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == _PUA_SPACE:
            out.append(" ")
        elif 0xF000 <= cp <= 0xF0FF:
            if encoding == ENC_ASCII and 0xF021 <= cp <= 0xF07E:
                out.append(chr(cp - 0xF000))
            elif encoding == ENC_SYMBOL:
                out.append(_SYMBOL.get(cp - 0xF000, LOSS))
            else:
                out.append(LOSS)
        elif cp in _PUA_DROP:
            continue
        elif cp == 0x23AF:                 # the same extender, outside the PUA
            out.append(LOSS)
        elif cp in _PUA_BUILT_UP:
            out.append(LOSS)
        elif 0xE000 <= cp <= 0xF8FF:
            out.append(LOSS)
        else:
            out.append(ch)
    return "".join(out)


def _script(text: str, sup: bool) -> str:
    """Render a raised/lowered span in Unicode, or mark it lost."""
    t = text.strip()
    if not t:
        return text
    if sup:
        if t.lower() in _ORDINAL:
            return t                       # "14th": the superscript is cosmetic
        if t in ("o", "°"):
            return "°"                     # a raised small o is a degree sign
        if set(t) <= _SUP_OK:
            return t.translate(_SUP)
        # Letters in an exponent (e^{-γt}, 10^{x}) have no Unicode form; the
        # caret keeps the meaning instead of silently gluing base and exponent.
        return f"^({t})"
    if set(t) <= _SUB_OK:
        return t.translate(_SUB)
    return f"_{t}" if len(t) == 1 else f"_({t})"


@dataclass
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    base: float
    text: str
    block: int


def _main_size(spans: list[dict]) -> float:
    """The line's body size: the largest size carrying letters or digits.

    Weighting by character count is wrong for chemistry: ``C9H18O9`` has more
    subscript characters than body characters, which made the body look like
    the anomaly and marked every formula lost.
    """
    sizes = [s["size"] for s in spans if any(c.isalnum() for c in s["text"])]
    return max(sizes) if sizes else max((s["size"] for s in spans), default=0.0)


def _baseline(spans: list[dict], main: float) -> float:
    bases = sorted(s["origin"][1] for s in spans
                   if s["size"] >= 0.95 * main and s["text"].strip())
    return bases[len(bases) // 2] if bases else (spans[0]["origin"][1] if spans else 0.0)


def _keep_ws(raw: str, rendered: str) -> str:
    lead = raw[:len(raw) - len(raw.lstrip())]
    trail = raw[len(raw.rstrip()):]
    return lead + rendered + trail


def _span_role(span: dict, main: float, base: float) -> str:
    size, y, text = span["size"], span["origin"][1], span["text"]
    if not text.strip():
        return "body"
    small = size < 0.85 * main
    if small and (span["flags"] & 1 or y < base - 0.15 * main):
        return "sup"
    if small and y > base + 0.08 * main:
        return "sub"
    if not small and abs(y - base) > 0.3 * main:
        return "stacked"          # same size, other baseline: a fraction part
    return "body"


def _render_spans(spans: list[dict], main: float, base: float,
                  encodings: Optional[dict[str, str]] = None) -> str:
    """Render spans, treating consecutive raised (or lowered) spans as ONE run.

    Run-grouping matters: maths prints the exponent of ``2^(p-1)`` as two
    spans, ``p`` and ``– 1``; scripting them separately gave ``2^(p) ⁻ ¹``,
    which reads as a different number.
    """
    out: list[str] = []
    run_role, run_text = "body", ""

    def flush() -> None:
        nonlocal run_text
        if not run_text:
            return
        if run_role in ("sup", "sub"):
            out.append(_keep_ws(run_text, _script(run_text, sup=run_role == "sup")))
        elif run_role == "stacked":
            out.append(LOSS + run_text)
        else:
            out.append(run_text)
        run_text = ""

    for span in spans:
        text = map_glyphs(span["text"], _encoding(span, encodings))
        role = _span_role(span, main, base)
        if not text.strip() and run_role in ("sup", "sub"):
            role = run_role               # a space inside an exponent run
        if role != run_role:
            flush()
            run_role = role
        run_text += text
    flush()
    return _GAP.sub(LOSS, "".join(out))


def _encoding(span: dict, encodings: Optional[dict[str, str]]) -> Optional[str]:
    if encodings is None:
        return ENC_SYMBOL
    return encodings.get(span.get("font", ""), ENC_UNKNOWN)


def render_page(page, encodings: Optional[dict[str, str]] = None,
                d: Optional[dict] = None) -> str:
    """One page as faithful text lines, in PyMuPDF's reading order.

    Beyond the per-span handling above, two whole-line rules:

      * a line that starts where an earlier line ends, on the same row, is the
        rest of that row and is rendered against the row's baseline (measured:
        chemistry prints ``mol L`` and ``–1`` as two separate lines, and
        ``3.12 g mL`` / ``–1, the mass of...`` likewise);
      * two same-size lines of one block that overlap vertically by a PART of
        their height are the stacked halves of a built-up expression -- a row
        of text overlaps fully or not at all -- and both are marked lost.
    """
    if d is None:
        d = page.get_text("dict")
    emitted: list[_Line] = []
    for bi, block in enumerate(d.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"]]
            if not any(s["text"].strip() for s in spans):
                continue
            if line.get("dir", (1, 0))[1] != 0:
                continue                  # rotated sidebar furniture
            x0, y0, x1, y1 = line["bbox"]
            main = _main_size(spans)
            host = _row_host(emitted, x0, y0, y1)
            if host is not None and main <= 1.15 * host.size:
                host.text += _render_spans(spans, host.size, host.base, encodings)
                host.x1 = max(host.x1, x1)
                continue
            base = _baseline(spans, main)
            emitted.append(_Line(x0, y0, x1, y1, main, base,
                                 _render_spans(spans, main, base, encodings), bi))
    _mark_stacked(emitted)
    _mark_drawn_math(emitted, page)
    aside =_out_of_order_blocks(d, _panels(page, page.rect.height))
    # A figure marks every line level with it: the question it belongs to has
    # text beside it or labels inside it, wherever PyMuPDF orders those lines.
    # Where no line is level with it, it goes after the last line above it --
    # in these books a figure follows the stem that cites it.
    after: set[int] = set()
    lead = False
    ph = page.rect.height
    for x0, y0, x1, y1 in _figure_regions(page, d):
        if y0 < 60 or y1 > ph - 60:
            continue                      # heading / footer decoration band
        inside = [l for l in emitted
                  if l.x0 < x1 and l.x1 > x0 and l.y0 < y1 and l.y1 > y0]
        if any(re.match(r"^\s*(?:Chapter|CHAPTER|Unit)\b", l.text) for l in inside):
            continue                      # the chapter-title panel, not a figure
        level = [i for i, l in enumerate(emitted) if l.y0 < y1 + 2 and l.y1 > y0 - 2]
        if level:
            after.update(level)
        else:
            slot = _figure_slot(emitted, (x0, y0, x1, y1))
            if slot == 0:
                lead = True
            else:
                after.add(slot - 1)
    lines = [FIGURE_MARK] if lead else []
    for i, l in enumerate(emitted):
        # Display type (chapter titles at 21-110pt, drawn as several offset
        # copies for a shadow) is never question text; marked so parsers can
        # read a title from it and otherwise drop it.
        if l.size >= 20:
            lines.append(f"{DISPLAY_MARK} {l.text}")
        else:
            lines.append(f"{ASIDE_MARK} {l.text}" if l.block in aside else l.text)
        if i in after:
            lines.append(FIGURE_MARK)
    return "\n".join(lines)


# A figure, structure or table the product has no asset for. Chemistry's
# structural formulae are vector drawings with no caption and no "Fig." in the
# stem ("Number of π bonds and σ bonds in the following structure is–"), so a
# text-only figure test served them without the structure. The marker puts the
# drawing into the question's text, where the figure gate sees it.
FIGURE_MARK = "[[figure]]"
DISPLAY_MARK = "[[display]]"
ASIDE_MARK = "[[aside]]"


def _inside(bbox, rect, share: float = 0.8) -> bool:
    x0, y0, x1, y1 = bbox
    rx0, ry0, rx1, ry1 = rect
    w, h = max(x1 - x0, 0.1), max(y1 - y0, 0.1)
    ix = max(0.0, min(x1, rx1) - max(x0, rx0))
    iy = max(0.0, min(y1, ry1) - max(y0, ry0))
    return ix * iy >= share * w * h


def _panels(page, height: float) -> list[tuple[float, float, float, float]]:
    """Tinted, outlined boxes in the page body: the "Key Concept" / "Think
    and Discuss" features of the class 7-8 maths books (measured: a light
    blue fill with a blue rule, "fs" in PyMuPDF's drawing types). White and
    black fills are page furniture and watermarks, not panels."""
    out = []
    try:
        drawings = page.get_drawings()
    except Exception:                      # pragma: no cover - defensive
        return out
    for dr in drawings:
        r, fill = dr.get("rect"), dr.get("fill")
        if r is None or dr.get("type") != "fs" or not fill:
            continue
        if all(c > 0.97 for c in fill) or all(c < 0.03 for c in fill):
            continue
        if r.width < 150 or r.height < 30 or r.y0 < 85 or (height and r.y1 > height - 50):
            continue
        out.append((r.x0, r.y0, r.x1, r.y1))
    return out


def _out_of_order_blocks(d: dict, panels: list[tuple[float, float, float, float]] = ()) -> set[int]:
    """Text blocks that are not the question stream: those inside a tinted
    panel (`_panels`), and those PyMuPDF emits after a block that sits below
    them in the same column. Measured: class 8 maths unit 4 prints a "one-step
    equation" panel under question 14, whose option (d) then read "3(x + 3) A
    one-step equation is ..."; class 7 maths unit 11 emits its top-of-page
    "Key Concept" panel after question 17. The lines are marked, not dropped:
    a question or answer that takes one in is excluded (``reading-order``),
    since dropping could as easily cut a real line out of it.
    """
    out: set[int] = set()
    for bi, block in enumerate(d.get("blocks", [])):
        if block.get("type") == 0 and any(_inside(block["bbox"], r) for r in panels):
            out.add(bi)
    prev = None
    height = d.get("height", 0) or 0
    for bi, block in enumerate(d.get("blocks", [])):
        if block.get("type") != 0 or not any(
                s.get("text", "").strip() for l in block.get("lines", []) for s in l.get("spans", [])):
            continue
        x0, y0, x1, y1 = block["bbox"]
        # Running heads, page numbers and print dates come in any order and
        # are dropped as furniture anyway; they neither count as out of
        # order nor set the frontier.
        if y1 < 85 or (height and y0 > height - 110):
            continue
        if prev is not None:
            px0, py0, px1, py1 = prev
            overlap = min(x1, px1) - max(x0, px0)
            if y1 < py0 - 5 and overlap > 0.3 * min(x1 - x0, px1 - px0):
                out.add(bi)
        prev = (x0, y0, x1, y1)
    return out


def _figure_regions(page, d: dict) -> list[tuple[float, float, float, float]]:
    """Regions of the page drawn as vector paths or placed as images.

    Excluded, because every page has them: the full-page background and the
    centred watermark images, page-wide rules, and decorations in the top and
    bottom 60pt (running heads, the footer band).
    """
    pw, ph = page.rect.width, page.rect.height
    rects = []
    for b in d.get("blocks", []):
        if b.get("type") == 1:
            x0, y0, x1, y1 = b["bbox"]
            area = (x1 - x0) * (y1 - y0)
            if area < 400 or area > 0.3 * pw * ph:
                continue
            rects.append((x0, y0, x1, y1))
    try:
        drawings = page.get_drawings()
    except Exception:                      # pragma: no cover - defensive
        drawings = []
    paths = []
    for dr in drawings:
        r = dr.get("rect")
        if r is None or r.is_empty or r.is_infinite:
            continue
        if r.width > 0.6 * pw or r.y1 < 60 or r.y0 > ph - 60:
            continue
        if r.width <= 1 and r.height <= 1:
            continue
        paths.append((r.x0, r.y0, r.x1, r.y1))
    # Cluster paths that touch or nearly touch; a structure is dozens of short
    # bonds, a fill-in blank or an underline is one path on its own.
    clusters: list[list[float]] = []
    for x0, y0, x1, y1 in sorted(paths, key=lambda r: (r[1], r[0])):
        for c in clusters:
            if x0 <= c[2] + 8 and x1 >= c[0] - 8 and y0 <= c[3] + 8 and y1 >= c[1] - 8:
                c[0], c[1], c[2], c[3] = min(c[0], x0), min(c[1], y0), max(c[2], x1), max(c[3], y1)
                c[4] += 1
                break
        else:
            clusters.append([x0, y0, x1, y1, 1])
    for x0, y0, x1, y1, n in clusters:
        if n >= 3 and (x1 - x0) * (y1 - y0) >= 400:
            rects.append((x0, y0, x1, y1))
    return rects


def _figure_slot(emitted: list[_Line], region: tuple[float, float, float, float]) -> int:
    """Where in reading order a figure sits: before the first line inside it,
    else after the last line above its middle."""
    x0, y0, x1, y1 = region
    for i, l in enumerate(emitted):
        if l.x0 < x1 and l.x1 > x0 and l.y0 < y1 and l.y1 > y0:
            return i
    mid = (y0 + y1) / 2
    above = [i for i, l in enumerate(emitted) if l.y1 <= mid]
    return (max(above, key=lambda i: emitted[i].y1) + 1) if above else 0


def _row_host(emitted: list[_Line], x0: float, y0: float, y1: float) -> Optional[_Line]:
    mid = (y0 + y1) / 2
    for cand in reversed(emitted[-6:]):
        if cand.y0 - 1 <= mid <= cand.y1 + 1 and -1.5 <= x0 - cand.x1 <= 4:
            return cand
    return None


def _mark_drawn_math(lines: list[_Line], page) -> None:
    """Mark lost every line that a small vector drawing sits on.

    The class 9-10 maths books draw the radical sign and the repeating-decimal
    bar as vector paths, not glyphs, so the text layer keeps only what is
    under them. Measured on class 9 maths chapter 1: "2√3 + √3 is equal to"
    extracted as "2 3+ 3 is equal to", "√10 × √15" as "10× 15", and the
    options "0.14", "0.14̄16", "0.1̄416" as three near-duplicates -- each a
    confidently wrong question, with no loss mark at all. Two drawing shapes
    are taken as mathematics:

      * a small path with a slanted segment (the radical's tick and strokes;
        a table rule or a box is only horizontal and vertical lines);
      * a short horizontal rule over the upper part of a text line (a
        vinculum or a fraction bar). A rule at the foot of a line is an
        underline or a drawn answer blank, and is left alone.
    """
    try:
        drawings = page.get_drawings()
    except Exception:                      # pragma: no cover - defensive
        return
    for dr in drawings:
        r = dr.get("rect")
        items = dr.get("items") or []
        if r is None or not items or any(it[0] not in ("l",) for it in items):
            continue
        if r.width > 150 or r.height > 60:
            continue
        slanted = any(abs(it[1].x - it[2].x) > 0.5 and abs(it[1].y - it[2].y) > 0.5
                      for it in items)
        if slanted and len(items) <= 12:
            for l in lines:
                if l.x0 < r.x1 + 1 and l.x1 > r.x0 - 1 and l.y0 < r.y1 + 1 and l.y1 > r.y0 - 1:
                    _lose(l)
        elif not slanted and r.height < 1.5 and 3 <= r.width <= 150:
            for l in lines:
                h = l.y1 - l.y0
                overlap = min(l.x1, r.x1) - max(l.x0, r.x0)
                if overlap >= 0.5 * r.width and l.y0 - 1.5 <= r.y0 <= l.y0 + 0.6 * h:
                    _lose(l)
            _lose_fraction(lines, r)


def _lose_fraction(lines: list[_Line], r) -> None:
    """A rule with a short line just above it and one just below it is a
    fraction bar: numerator and denominator are both lost.

    Measured on gemp110 p.24, question 93: "7" over "8 – x" with the bar at
    y 103.4, in the gap between the two lines, so neither the vinculum rule
    (a rule over the upper part of a line) nor the stacked-line rule (it needs
    the two lines' boxes to overlap, which depends on the PyMuPDF build) saw
    it, and "7 8 – x" read as a whole number. A body line is much wider than a
    bar, so an underline between two lines of prose is not taken.
    """
    def near(l: _Line) -> bool:
        overlap = min(l.x1, r.x1) - max(l.x0, r.x0)
        return overlap >= 0.3 * min(r.width, l.x1 - l.x0) and (l.x1 - l.x0) <= 2 * r.width + 6

    above = [l for l in lines if near(l) and -1.0 <= r.y0 - l.y1 <= 5.0]
    below = [l for l in lines if near(l) and -1.0 <= l.y0 - r.y0 <= 5.0]
    above = [l for l in above if l not in below]
    if not (above and below):
        return
    # A bar is as wide as the wider of its two parts; a table rule or a
    # figure's edge is much wider than the cell text or label next to it.
    widest = max(l.x1 - l.x0 for l in above + below)
    if r.width <= 1.6 * widest + 6:
        for l in above + below:
            _lose(l)


def _lose(line: _Line) -> None:
    if not line.text.startswith(LOSS):
        line.text = LOSS + line.text


def _mark_stacked(lines: list[_Line]) -> None:
    by_block: dict[int, list[_Line]] = {}
    for l in lines:
        by_block.setdefault(l.block, []).append(l)
    for group in by_block.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if min(a.size, b.size) < 0.8 * max(a.size, b.size):
                    continue
                overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
                h = min(a.y1 - a.y0, b.y1 - b.y0)
                if h <= 0 or overlap <= 0:
                    continue
                if 0.15 <= overlap / h <= 0.75:
                    for l in (a, b):
                        if not l.text.startswith(LOSS):
                            l.text = LOSS + l.text


def _pymupdf():
    try:
        import pymupdf as fitz
    except ImportError:                    # pragma: no cover - older installs
        import fitz
    return fitz


def require_verified_pymupdf() -> None:
    """Refuse to render under any PyMuPDF but `VERIFIED_PYMUPDF`.

    See "PDF library" in the module docstring: another version changes which
    fractions are detected as lost, and so changes the bank without a word.
    """
    running = str(getattr(_pymupdf(), "VersionBind", "unknown"))
    if running != VERIFIED_PYMUPDF:
        raise RuntimeError(
            f"PyMuPDF {running} is installed, but the Exemplar parser was verified on "
            f"PyMuPDF {VERIFIED_PYMUPDF} only: under 1.27.2.3 it wrote 10 records whose "
            f"stacked fractions were flattened into plain digits. Install "
            f"pymupdf=={VERIFIED_PYMUPDF} (pyproject.toml, uv.lock), or re-verify the "
            f"bank on the new version and change VERIFIED_PYMUPDF.")


def read_pdf(path: Path) -> list[str]:
    """Every page of a PDF through `render_page`. The one PDF-touching function."""
    require_verified_pymupdf()
    fitz = _pymupdf()
    doc = fitz.open(str(path))
    try:
        pages = list(doc)
        dicts = [p.get_text("dict") for p in pages]
        encodings = classify_fonts(dicts)
        return [render_page(p, encodings, d) for p, d in zip(pages, dicts)]
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# shared line handling
# --------------------------------------------------------------------------- #

_PAGE_NO = re.compile(r"^\s*\d{1,3}\s*$")
# The print date every page carries: "11.4.2018", "12/04/18", "16-04-2018".
_DATE = re.compile(r"^\s*\d{1,2}[./-]\d{1,2}[./-](?:20)?1[5-9]\s*$")


def _norm(text: str) -> str:
    """Letters and digits only, lowercased: the identity used for matching."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", text).lower())


def _is_running_head(line: str, *names: str) -> bool:
    """A running head, alone or row-joined with its page number.

    Measured: class 6 maths prints "NUMBER SYSTEM  3" and "2  EXEMPLAR
    PROBLEMS" as one row each, which a name-only test missed.
    """
    bare = re.sub(r"^\s*\d{1,3}\s+|\s+\d{1,3}\s*$", "", line)
    return any(n and _norm(bare) == _norm(n) for n in names)


def _join(lines: Iterable[str]) -> str:
    """Join extracted lines into prose; a line-end hyphen joins without space."""
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif text.endswith("-") and not text.endswith(" -"):
            text += line
        else:
            text += " " + line
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# A label on a line of its own is structure, not a formula fragment: science
# prints "(i)" / "Wheat" / "(ii)" / "Ghee" as four lines (class 6, feep202).
_LIST_LABEL = re.compile(r"^\(?(?:[a-hA-H]|i{1,3}|iv|vi{0,3}|ix|x)\)\.?$")
_SHORT_WORDS = {"or", "and", "the", "of", "to", "is", "in", "on", "at", "by", "as",
                "it", "an", "no", "yes", "not"}


def _is_orphan(line: str) -> bool:
    """A continuation line that is a stray fragment of built-up mathematics.

    Measured: a fraction or a big operator that PyMuPDF splits into separate
    blocks leaves lines like ``2``, ``/``, ``x``, ``i=`` between ordinary text.
    Prose never continues on a line of three characters or fewer -- apart from
    list labels and a few short words ("or" between two alternative answers) --
    so such a line in the middle of a question means a formula did not survive
    linearisation, or a table's cells were read one per line.
    """
    t = line.strip()
    if not 0 < len(t) <= 3:
        return False
    return not (re.fullmatch(r"[.,;:?)\]]+", t) or _LIST_LABEL.match(t)
                or t.lower() in _SHORT_WORDS)


@dataclass
class _Tagged:
    """One kept line. A leading loss mark is held aside in `lossy`.

    The stacked-line rule marks a whole line lost by prefixing U+FFFD, and
    that line is often a question number or an option label. Structure is
    matched on `text`, and `content` -- what a question or answer is built
    from -- carries the mark on, so the loss still excludes the question
    instead of breaking the numbering around it.
    """

    page: int
    text: str
    lossy: bool = False
    aside: bool = False              # printed out of reading order (`render_page`)

    @property
    def prefix(self) -> str:
        return (LOSS if self.lossy else "") + (ASIDE_MARK + " " if self.aside else "")

    @property
    def content(self) -> str:
        return self.prefix + self.text


def _page_number_lines(lines: list[str],
                       furniture: Optional[Callable[[str], bool]] = None) -> set[int]:
    """Indices of a page's page-number lines: the leading run of bare numbers
    and bare numbers among the last three lines.

    Position matters. A bare number anywhere else is content -- an option's
    value or an answer ("98." / "T" / "99." / "60") is exactly such a line.
    Running-head lines (`furniture`) do not end the leading run: class 6
    science prints "COMPONENTSOF F" / "OMPONENTSOF" ... / "7" before the
    text, and the "7" was read into question 4 as a stray fragment.
    """
    lines = [_unaside(l)[0] for l in lines]
    idx = [i for i, l in enumerate(lines) if l.strip()]
    out: set[int] = set()
    for i in idx:
        if _PAGE_NO.match(lines[i]):
            out.add(i)
        elif (lines[i].strip() != FIGURE_MARK and not lines[i].strip().startswith(DISPLAY_MARK)
              and not (furniture is not None and furniture(lines[i].strip()))):
            break
    for i in idx[-3:]:
        if _PAGE_NO.match(lines[i]) and not _prev_is_number_label(lines, idx, i):
            out.add(i)
    # A running head's page number can sit right after it: "EXEMPLAR
    # PROBLEMS" / "8". Only after a head-like line (words, not an option
    # label), so "(B)" / "120" at the top of a page stays content.
    if len(idx) > 1 and _PAGE_NO.match(lines[idx[1]]):
        first = lines[idx[0]].strip()
        if not first.startswith("(") and len(re.findall(r"[A-Za-z]{2,}", first)) >= 2:
            out.add(idx[1])
    return out


def _prev_is_number_label(lines: list[str], idx: list[int], i: int) -> bool:
    """The bare number is the answer of the "N." line just before it
    (answers files print "38." / "(C)" / "39." / "7"), not a page number."""
    pos = idx.index(i)
    return pos > 0 and re.fullmatch(r"\s*\d{1,3}\.\s*", lines[idx[pos - 1]]) is not None


# A line that stops mid-sentence: it ends on a lower-case word or a comma.
_MID_SENTENCE = re.compile(r"(?:\b[a-z]+|,)\s*$")
_HEAD_BAND = 3          # a running head sits among a page's first or last 3 lines


def _head_slots(raw: list[str], numbers: set[int]) -> set[int]:
    """Where a running head can stand on a page: among its first and last
    few text lines, or next to its page number."""
    idx = [j for j, l in enumerate(raw)
           if l.strip() and l.strip() != FIGURE_MARK
           and not _DATE.match(_unaside(l)[0].strip())]
    slots = set(idx[:_HEAD_BAND]) | set(idx[-_HEAD_BAND:])
    for k, j in enumerate(idx):
        if (k > 0 and idx[k - 1] in numbers) or (k + 1 < len(idx) and idx[k + 1] in numbers):
            slots.add(j)
    return slots


def _reads_as_head(line: str, prev: Optional[str]) -> bool:
    """A title-matching line that reads as a running head, not as the rest of
    a sentence. Heads are set in capitals or title case with no sentence
    punctuation; a line that starts in lower case, ends like a sentence, or
    follows a line cut mid-sentence continues a question ("... require more
    than a bucket of" / "water.")."""
    s = line.strip()
    if not s or s[0].islower() or re.search(r"[.?!;:,]\s*$", s):
        return False
    return not (prev is not None and _MID_SENTENCE.search(prev))


def _tag(pages: list[str], drop: Callable[[str], bool],
         title: Optional[Callable[[str], bool]] = None) -> list[_Tagged]:
    """Every page's kept lines. `drop` is furniture wherever it stands.
    `title` (a chapter-title match) is dropped only where a running head
    stands (`_head_slots`) and reads as one (`_reads_as_head`)."""
    out = []
    furniture = drop if title is None else (lambda l: drop(l) or title(l))
    for i, page in enumerate(pages, start=1):
        raw = page.split("\n")
        numbers = _page_number_lines(raw, furniture)
        slots = _head_slots(raw, numbers) if title is not None else set()
        prev: Optional[str] = None            # the last kept text line on this page
        for j, line in enumerate(raw):
            if j in numbers:
                continue
            line, aside = _unaside(line)
            body = line.strip()
            lossy = body.startswith(LOSS)
            body = body.lstrip(LOSS + " ") if lossy else body
            if not body:
                if lossy:
                    out.append(_Tagged(i, LOSS))
                continue
            if _DATE.match(body) or body.startswith(DISPLAY_MARK) or drop(body):
                continue
            if (title is not None and j in slots and title(body)
                    and _reads_as_head(body, prev)):
                continue
            out.append(_Tagged(i, body, lossy, aside))
            if body != FIGURE_MARK:
                prev = body
    return out


def _unaside(line: str) -> tuple[str, bool]:
    if line.startswith(ASIDE_MARK):
        return line[len(ASIDE_MARK):].lstrip(), True
    return line, False


def _head(first: Optional[str], prefix: str) -> list[str]:
    """A question's first content lines, from the text sharing its number line;
    `prefix` carries the number line's own marks (loss, out of order)."""
    lines = [first] if first and first.strip() else []
    return ([prefix] if prefix else []) + lines


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #

_LETTERS = "abcdef"


def _option_style(lines: list[str]) -> str:
    """How this question labels its options: the first label line decides.
    Maths 6-7 and 9-10 print (A)-(D); maths 8 and all science (a)-(d)."""
    for line in lines:
        m = re.match(rf"^\s*{LOSS}*\s*\(([A-Da-d])\)", line)
        if m:
            return "paren_upper" if m.group(1).isupper() else "paren_lower"
    return "paren_lower"


def _split_options(lines: list[str], style: str) -> tuple[list[str], list[str], list[str], str]:
    """Split a question's lines into (stem lines, option texts, labels, error).

    `style` is ``paren_lower`` ``(a)`` or ``paren_upper`` ``(A)``. Labels must
    appear once each, in order, starting at the first; a restart or a skip
    means something else (a table header, a list inside the stem) is using the
    same marks, and the options are refused rather than guessed -- except the
    two-column layout `_permuted_options` can prove.
    """
    if style == "paren_upper":
        labels = [c.upper() for c in _LETTERS]
        pat = r"\(([A-F])\)"
    else:
        labels = list(_LETTERS)
        pat = r"\(([a-f])\)"
    start = re.compile(rf"^\s*({LOSS}*)\s*{pat}\s*(.*)$")
    inline = re.compile(rf"(?:(?<=[\s{LOSS}])|^){pat}\s*")

    stem: list[str] = []
    options: list[list[str]] = []
    seen: list[str] = []
    error = ""
    for line in lines:
        m = start.match(line)
        if m and (not seen and m.group(2) == labels[0]
                  or seen and len(seen) < len(labels) and m.group(2) == labels[len(seen)]):
            seen.append(m.group(2))
            options.append([])
            rest = m.group(1) + m.group(3)
        elif m and seen:
            error = f"option label {m.group(2)!r} out of order after {seen[-1]!r}"
            break
        elif m and not seen:
            error = f"options start at {m.group(2)!r}, not {labels[0]!r}"
            break
        elif not seen:
            stem.append(line)
            continue
        else:
            rest = line
        # Several options on one printed line: "(A) 10 (B) 11".
        while True:
            nxt = labels[len(seen)] if len(seen) < len(labels) else None
            im = None
            if nxt is not None:
                for cand in inline.finditer(rest):
                    if (cand.group(1) == nxt and cand.start() > 0
                            and rest[:cand.start()].strip(LOSS + " ")):
                        im = cand
                        break
            if im is None:
                options[-1].append(rest)
                break
            options[-1].append(rest[:im.start()])
            seen.append(im.group(1))
            options.append([])
            rest = rest[im.end():]
    if error.startswith("option label"):
        permuted = _permuted_options(lines, labels, start)
        if permuted is not None:
            return permuted
    texts = []
    for opt in options:
        body = [l for l in opt if l.strip()]
        # The first body line of an option may be short ("Na", "B"); any
        # later short line is a fragment.
        if any(_is_orphan(l) for l in body[1:]):
            texts.append(LOSS + _join(body))
        else:
            texts.append(_join(body))
    if not error and any(not t for t in texts):
        error = "an option has no text"
    if not error and 0 < len(texts) < 4:
        error = f"only {len(texts)} options"
    return stem, texts, seen, error


def _permuted_options(lines: list[str], labels: list[str], start: re.Pattern):
    """Options printed in two columns, read row by row: (a) (c) (b) (d).

    Measured: class 8 science sets most MCQs this way ("(a) Bacteria" /
    "(c) Amoeba" / "(b) Virus" / "(d) Fungus."). Accepted only when every
    label occurs exactly once, they are the first k labels (k >= 4), and each
    option is exactly its own label line -- so no continuation line can have
    been read into the wrong column.
    """
    idx = [(i, m) for i, l in enumerate(lines) if (m := start.match(l))]
    if len(idx) < 4:
        return None
    got = [m.group(2) for _, m in idx]
    k = len(got)
    if sorted(got, key=labels.index) != labels[:k]:
        return None
    ends = [i for i, _ in idx[1:]] + [len(lines)]
    texts: dict[str, str] = {}
    for (i, m), end in zip(idx, ends):
        own = (m.group(1) + m.group(3)).strip()
        if not own.strip(LOSS + " ") or any(l.strip() for l in lines[i + 1:end]):
            return None
        texts[m.group(2)] = own
    return lines[:idx[0][0]], [texts[l] for l in labels[:k]], labels[:k], ""


def _letters_from_answer(text: str, n_options: int) -> tuple[list[str], str, str]:
    """(option letters, explanatory note, error) for an MCQ answer.

    The answer must OPEN with option labels -- "(b)", "(C)", "b", "(b), (c)"
    -- and may be followed by the book's explanation ("(c) Hint— The
    substance which oxidises..."). If the explanation itself names another
    option label, the key is no longer only the leading letters, so it is
    refused.
    """
    # A loss mark beside a key comes from the stacked-line rule firing on the
    # answer column's layout; the letters themselves are intact. A loss inside
    # the explanation is caught later by the symbol-loss gate on answer_text.
    t = text.strip().lstrip(LOSS + " ").strip()
    lab = r"(?:\([a-fA-F]\)|\b[a-fA-F]\b)"
    m = re.match(rf"^((?:{lab}[\s,.;]*(?:and\s+)?)+)(.*)$", t, re.S)
    letters = [x.lower() for x in re.findall(r"([a-fA-F])", m.group(1))] if m else []
    if not m or not letters:
        return [], "", f"answer {t[:40]!r} does not open with option letters"
    note = m.group(2).strip()
    if note and re.search(r"\((?:[a-fA-F]|i|ii|iii|iv|v)\)", note):
        return [], "", f"answer {t[:40]!r} names options beyond its leading letters"
    if note and "(" not in m.group(1) and re.match(r"^[a-z]", note):
        # "a" followed by lower-case prose is a word, not a key.
        return [], "", f"answer {t[:40]!r} runs letters into prose"
    if len(set(letters)) != len(letters):
        return [], "", f"answer {t!r} repeats a letter"
    if any(_LETTERS.index(l) >= n_options for l in letters):
        return [], "", f"answer {t!r} names an option the question does not have"
    # A fragment ("(b) 4") is residue beside the key, not a note.
    return letters, (note if len(note) >= 20 else ""), ""


# --------------------------------------------------------------------------- #
# numbered entries (questions and answers)
# --------------------------------------------------------------------------- #

_ENTRY_LINE = re.compile(r"^\s*\d{1,3}\s?\.(?!\d)")
# "103. 1650 104. 1290000": an answers row carrying several answers. Between
# two answers the gap is spaces, or the loss mark `_GAP` left for a wide one
# ("109. 5,23,78,401�110. L").
_MIDLINE_AFTER_ENTRY = re.compile(rf"(?<=\S)(?:\s+|\s*{LOSS}\s*)(?=\d{{1,3}}\.\s)")
_MIDLINE_ANY = re.compile(rf"(?<=\S)(?:\s{{2,}}|\s*{LOSS}\s*)(?=\d{{1,3}}\.\s)")


def _numbered_entries(tagged: list[_Tagged], start_re: re.Pattern,
                      max_step: int = 25, limit: Optional[int] = None,
                      stop: Optional[Callable[[str], bool]] = None,
                      ) -> tuple[list[tuple[int, int, list[str]]], set[int]]:
    """Split answer lines into (number, page, lines) entries.

    A line is an entry start only if its number exceeds the previous entry's
    by 1..`max_step`, and a start found in the middle of a row only if it is
    exactly the next number: a sub-list "1." inside an answer, or a value like
    "2.5" on its own line, cannot silently start a new entry. The step is wide
    because the books skip answers in runs (class 10 maths answers exercise
    1.3 from 8, its proofs having none). Three things make a number ambiguous
    -- returned so it pairs with nothing:

      * it recurs close to where the scan is (within three of the running
        number: far behind, "1." is a numbered list inside an answer);
      * a number the scan skipped is visible inside the previous entry
        ("101. (a) 1000 ... 102.1": the "102." was glued to its answer, so
        101 swallowed it and would pair a wrong answer);
      * it was split out of the middle of a row whose split later proved
        wrong (its number recurs), which also taints the row it came from.
    """
    entries: list[tuple[int, int, list[str]]] = []
    seen: dict[int, int] = {}
    ambiguous: set[int] = set()
    split_from: dict[int, int] = {}
    last = 0
    for t in tagged:
        if stop is not None and stop(t.text):
            break
        splitter = _MIDLINE_AFTER_ENTRY if _ENTRY_LINE.match(t.text) else _MIDLINE_ANY
        pieces = splitter.split(t.text)
        for k, piece in enumerate(pieces):
            prefix = t.prefix if k == 0 else ""
            m = start_re.match(piece)
            if m:
                n = int(m.group(1))
                ok = (n == last + 1) if k else (last < n <= last + max_step)
                if ok and (limit is None or n <= limit):
                    if k and entries:
                        split_from[n] = entries[-1][0]
                    entries.append((n, t.page, _head(m.group(2), prefix)))
                    seen[n] = seen.get(n, 0) + 1
                    last = n
                    continue
                # A number seen again near the running position could be a
                # second answer for it; far behind ("1." while at 30) it is
                # a numbered list inside the current answer. Measured: the
                # strict rule threw away the first 7-17 keys of 9 chapters.
                if n in seen and n != last and n >= last - 3:
                    ambiguous.add(n)
            if entries:
                entries[-1][2].append(prefix + piece)
    for n, c in seen.items():
        if c > 1:
            ambiguous.add(n)
    for n in list(ambiguous):
        if n in split_from:
            ambiguous.add(split_from[n])
    # A skipped number printed inside the entry before the gap.
    for (n, _, body), nxt in zip(entries, entries[1:] + [(None, 0, [])]):
        upto = nxt[0] if nxt[0] is not None else n + 2
        text = " ".join(body)
        for missing in range(n + 1, upto):
            if re.search(rf"(?<![\d.]){missing}\s?\.(?!\d)", text):
                ambiguous.add(n)
                break
    return entries, ambiguous


_EXTRA_QUESTION = re.compile(r"^\s*Extra\s+Questions?\b", re.I)


def _answer_text(lines: list[str]) -> str:
    body = [l for l in lines if l.strip()]
    # gemp1a1 p.45 prints "123. d" then "Extra Question:-" and a new question
    # before "124.": that question and its text are not part of answer 123.
    cut = next((i for i, l in enumerate(body) if _EXTRA_QUESTION.match(l)), None)
    if cut is not None:
        body = body[:cut]
    text = _join(body)
    if any(_is_orphan(l) for l in body[1:]):
        text = LOSS + text
    return text


# --------------------------------------------------------------------------- #
# the ten books
# --------------------------------------------------------------------------- #

LAYOUT_MATHS_UNIT = "maths_unit"          # maths 6-8
LAYOUT_MATHS_EXERCISE = "maths_exercise"  # maths 9-10
LAYOUT_SCIENCE = "science"                # science 6-10


@dataclass(frozen=True)
class Book:
    grade: int
    subject: str
    folder: str               # under the exemplar root: "classVI/mathematics"
    answers_file: str
    layout: str
    current_textbook: bool    # False: written for the pre-2024 NCERT books


BOOKS: tuple[Book, ...] = (
    Book(6, "Mathematics", "classVI/mathematics", "feep1an.pdf", LAYOUT_MATHS_UNIT, False),
    Book(6, "Science", "classVI/science", "feep2an.pdf", LAYOUT_SCIENCE, False),
    Book(7, "Mathematics", "classVII/mathematics", "gemp1a1.pdf", LAYOUT_MATHS_UNIT, False),
    Book(7, "Science", "classVII/science", "geep1an.pdf", LAYOUT_SCIENCE, False),
    Book(8, "Mathematics", "classVIII/mathematics", "heep2an.pdf", LAYOUT_MATHS_UNIT, False),
    Book(8, "Science", "classVIII/science", "heep1an.pdf", LAYOUT_SCIENCE, False),
    Book(9, "Mathematics", "classIX/mathematics", "ieep2an.pdf", LAYOUT_MATHS_EXERCISE, True),
    Book(9, "Science", "classIX/science", "ieep1an.pdf", LAYOUT_SCIENCE, True),
    Book(10, "Mathematics", "classX/mathematics", "jeep2an.pdf", LAYOUT_MATHS_EXERCISE, True),
    Book(10, "Science", "classX/science", "jeep1an.pdf", LAYOUT_SCIENCE, True),
)

# Files that are not chapters, and why they are not parsed.
NOT_PARSED = {
    ("classIX/mathematics", "ieep215.pdf"): "design of the question paper, set I",
    ("classIX/mathematics", "ieep216.pdf"): "design of the question paper, set II",
    ("classIX/science", "ieep116.pdf"): "sample question paper I",
    ("classIX/science", "ieep117.pdf"): "sample question paper II",
    ("classX/mathematics", "jeep214.pdf"): "design of the question paper, set I",
    ("classX/mathematics", "jeep215.pdf"): "design of the question paper, set II",
    ("classX/science", "jeep117.pdf"): "sample question paper I",
    ("classX/science", "jeep118.pdf"): "sample question paper II",
    ("classX/science", "jeep119.pdf"): "appendix: design of the sample paper",
    ("classX/science", "jeep120.pdf"): "appendix: definitions and symbols",
    ("classX/science", "jeep121.pdf"): "appendix: table of elements",
}

# The Exemplar's own chapter titles, as each file prints them (display title
# or running head). Typed from the files and checked against them on every
# build (`_title_in_file`): the files' own title text is shadow-printed,
# shredded across display lines or set in shifted fonts often enough that no
# single extraction rule reads all 125 of them. A title the check cannot find
# is reported, never silently used.
CHAPTER_TITLES: dict[tuple[int, str], dict[int, str]] = {
    (6, "Mathematics"): {
        1: "Number System", 2: "Geometry", 3: "Integers", 4: "Fractions and Decimals",
        5: "Data Handling", 6: "Mensuration", 7: "Algebra", 8: "Ratio and Proportion",
        9: "Symmetry and Practical Geometry",
    },
    (6, "Science"): {
        1: "Food: Where Does It Come From?", 2: "Components of Food", 3: "Fibre to Fabric",
        4: "Sorting Materials and Groups", 5: "Separation of Substances",
        6: "Changes Around Us", 7: "Getting to Know Plants", 8: "Body Movement",
        9: "The Living Organisms and their Surroundings",
        10: "Motion and Measurement of Distances", 11: "Light",
        12: "Electricity and Circuits", 13: "Fun with Magnets", 14: "Water",
        15: "Air Around Us", 16: "Garbage in, Garbage out",
    },
    (7, "Mathematics"): {
        1: "Integers", 2: "Fractions and Decimals", 3: "Data Handling",
        4: "Simple Equations", 5: "Lines and Angles", 6: "Triangles",
        7: "Comparing Quantities", 8: "Rational Numbers", 9: "Perimeter and Area",
        10: "Algebraic Expressions", 11: "Exponents and Powers",
        12: "Practical Geometry, Symmetry and Visualising Solid Shapes",
    },
    (7, "Science"): {
        1: "Nutrition in Plants", 2: "Nutrition in Animals", 3: "Fibre to Fabric",
        4: "Heat", 5: "Acids, Bases and Salts", 6: "Physical and Chemical Changes",
        7: "Weather, Climate and Adaptation of Animals to Climate",
        8: "Wind, Storm and Cyclone", 9: "Soil", 10: "Respiration in Organisms",
        11: "Transportation in Animals and Plants", 12: "Reproduction in Plants",
        13: "Motion and Time", 14: "Electric Current and Its Effects", 15: "Light",
        16: "Water: A Precious Resource", 17: "Forests: Our Lifeline",
        18: "Wastewater Story",
    },
    (8, "Mathematics"): {
        1: "Rational Numbers", 2: "Data Handling",
        3: "Square-Square Root and Cube-Cube Root", 4: "Linear Equations in One Variable",
        5: "Understanding Quadrilaterals and Practical Geometry",
        6: "Visualising Solid Shapes", 7: "Algebraic Expressions, Identities and Factorisation",
        8: "Exponents and Powers", 9: "Comparing Quantities",
        10: "Direct and Inverse Proportions", 11: "Mensuration",
        12: "Introduction to Graphs", 13: "Playing with Numbers",
    },
    (8, "Science"): {
        1: "Crop Production and Management", 2: "Microorganisms: Friend and Foe",
        3: "Synthetic Fibres and Plastics", 4: "Materials: Metals and Non-Metals",
        5: "Coal and Petroleum", 6: "Combustion and Flame",
        7: "Conservation of Plants and Animals", 8: "Cell—Structure and Functions",
        9: "Reproduction in Animals", 10: "Reaching the Age of Adolescence",
        11: "Force", 12: "Friction", 13: "Sound", 14: "Chemical Effects of Electric Current",
        15: "Some Natural Phenomena", 16: "Light", 17: "Stars and Solar System",
        18: "Pollution of Air and Water",
    },
    (9, "Mathematics"): {
        1: "Number Systems", 2: "Polynomials", 3: "Coordinate Geometry",
        4: "Linear Equations in Two Variables", 5: "Introduction to Euclid's Geometry",
        6: "Lines and Angles", 7: "Triangles", 8: "Quadrilaterals",
        9: "Areas of Parallelograms and Triangles", 10: "Circles", 11: "Constructions",
        12: "Heron's Formula", 13: "Surface Areas and Volumes",
        14: "Statistics and Probability",
    },
    (9, "Science"): {
        1: "Matter in Our Surroundings", 2: "Is Matter Around Us Pure",
        3: "Atoms and Molecules", 4: "Structure of the Atom",
        5: "The Fundamental Unit of Life", 6: "Tissues", 7: "Diversity in Living Organisms",
        8: "Motion", 9: "Force and Laws of Motion", 10: "Gravitation",
        11: "Work and Energy", 12: "Sound", 13: "Why Do We Fall Ill",
        14: "Natural Resources", 15: "Improvement in Food Resources",
    },
    (10, "Mathematics"): {
        1: "Real Numbers", 2: "Polynomials", 3: "Pair of Linear Equations in Two Variables",
        4: "Quadratic Equations", 5: "Arithmetic Progressions", 6: "Triangles",
        7: "Coordinate Geometry", 8: "Introduction to Trigonometry and Its Applications",
        9: "Circles", 10: "Construction", 11: "Area Related to Circles",
        12: "Surface Areas and Volumes", 13: "Statistics and Probability",
    },
    (10, "Science"): {
        1: "Chemical Reactions and Equations", 2: "Acids, Bases and Salts",
        3: "Metals and Non-metals", 4: "Carbon and its Compounds",
        5: "Periodic Classification of Elements", 6: "Life Processes",
        7: "Control and Coordination", 8: "How do Organisms Reproduce?",
        9: "Heredity and Evolution", 10: "Light – Reflection and Refraction",
        11: "The Human Eye and the Colourful World", 12: "Electricity",
        13: "Magnetic Effects of Electric Current", 14: "Sources of Energy",
        15: "Our Environment", 16: "Management of Natural Resources",
    },
}


def _title_in_file(title: str, pages: list[str]) -> bool:
    """The title's letters occur in the file's own text, display type included.

    Shadow printing splits a head into fragments ("COMPONENTSOF F" /
    "OMPONENTS OF" / "FOOD"), so the check is on letters and digits only,
    first against the whole of the first two pages, then against the
    most-repeated line of the rest (the running head).
    """
    want = _norm(title)
    if not want:
        return False
    head = _norm(" ".join(p.replace(DISPLAY_MARK, " ") for p in pages[:2]))
    if want in head:
        return True
    lines: dict[str, int] = {}
    for page in pages:
        for line in {l.replace(DISPLAY_MARK, "").strip() for l in page.split("\n")}:
            n = _norm(re.sub(r"^\d{1,3}\s+|\s+\d{1,3}$", "", line))
            if n:
                lines[n] = lines.get(n, 0) + 1
    if any(want == n for n, c in lines.items() if c >= 2):
        return True
    # Shredded display titles ("FORCE" / "ORCE AND LAWS" / "OF MOTION"):
    # every word of the title, in order, inside the page-1 display text.
    display = _norm(" ".join(l.replace(DISPLAY_MARK, " ") for p in pages[:2]
                             for l in p.split("\n") if l.startswith(DISPLAY_MARK)))
    at = 0
    for word in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", title).lower()):
        at = display.find(word, at)
        if at < 0:
            return False
        at += len(word)
    return True


_FURNITURE = {_norm(s) for s in (
    "EXEMPLAR PROBLEMS", "EXEMPLAR PROBLEMS – SCIENCE", "EXEMPLAR PROBLEMS SCIENCE",
    "MATHEMATICS", "SCIENCE", "ANSWERS", "Answers",
)}


def _is_furniture(line: str, titles: Iterable[str]) -> bool:
    """Running heads, their shadow copies, and the book's own labels.

    Class 6-8 science shadow-prints every head as five offset copies that
    extract as fragments ("XEMPLAR", "OMPONENTSOF", "NSWERS"); a line that is
    all capitals and whose letters are a piece of a head is one of them.
    """
    s = line.strip()
    n = _norm(s)
    if not n:
        return False
    names = [t for t in titles if t]
    if n in _FURNITURE or _is_running_head(s, *names, "EXEMPLAR PROBLEMS",
                                           "EXEMPLAR PROBLEMS – SCIENCE", "ANSWERS",
                                           "MATHEMATICS"):
        return True
    return _is_shadow_fragment(s, names)


def _is_shadow_fragment(line: str, titles: Iterable[str]) -> bool:
    """All capitals, and its letters a piece of a head or of the title."""
    s = line.strip()
    n = _norm(s)
    if re.search(r"[a-z]", s) or len(n) < 3 or not re.search(r"[A-Z]", s):
        return False
    heads = [_norm(t) for t in titles if t] + ["exemplarproblems", "answers", "mathematics"]
    return any(n in h for h in heads)


# Section heads, matched on letters only so spacing and case do not matter.
_SCIENCE_SECTION = {
    "multiplechoicequestions": (MCQ_SINGLE, "Multiple Choice Questions"),
    "veryshortanswerquestions": (VSA, "Very Short Answer Questions"),
    "veryshortanswertypequestions": (VSA, "Very Short Answer Questions"),
    "shortanswerquestions": (SA, "Short Answer Questions"),
    "shortanswertypequestions": (SA, "Short Answer Questions"),
    "longanswerquestions": (LA, "Long Answer Questions"),
    "longanswertypequestions": (LA, "Long Answer Questions"),
}
_MATHS_SECTION = {
    "multiplechoicequestions": (MCQ_SINGLE, "Multiple Choice Questions"),
    "shortanswerquestionswithreasoning": (VSA, "Short Answer Questions with Reasoning"),
    "shortanswerquestions": (SA, "Short Answer Questions"),
    "longanswerquestions": (LA, "Long Answer Questions"),
}
_MATHS_SECTION_LINE = re.compile(r"^\s*\(([A-F])\)\s*(\S.*)$")
_EXERCISE = re.compile(r"^\s*EXERCISE\s+(\d{1,2})\s*\.\s*(\d)\s*$", re.I)
_UNIT_HEAD = re.compile(r"^\s*UNIT\s*[-−–]?\s*(\d{1,2})\s*$", re.I)
_CHAPTER_HEAD = re.compile(r"^\s*(?:\[\[display\]\]\s*)?CHAPTER\s*(\d{1,2})\s*$", re.I)
_QUESTION_START = re.compile(r"^\s*(\d{1,3})\s?\.(?!\d)\s*(.*)$")
# "In questions 39 to 98, state whether ...": the maths 6-8 section kinds.
# Wherever the words sit: "In each of the questions, 1 to 24, ...", "State
# whether the statements given in questions 12 to 20 are true", "Using the
# following frequency table, answer question 20-22" (all measured).
_RANGE = re.compile(r"\bquestions?\s*,?\s*(?:from\s+)?(\d{1,3})\s*,?\s*(?:to|and|-|–|—)\s*"
                    r"(\d{1,3})\b", re.I)
# An instruction opens its line with one of these words (measured over all
# 120-odd range instructions of the class 6-8 maths books). Requiring it
# keeps "... as in questions 5 and 6" inside a question from being read as
# an instruction that would cut that question short.
_INSTRUCTION_START = re.compile(
    r"^\s*(?:In|State|Translate|Encircle|Using|Now|Work|Answer|Fill|Write|Choose|What|"
    r"Express|Replace|Solve|Observe|The following|Based)\b")
# The range as an instruction words it -- "In questions 39 to 98, ", "In each
# of the following, ", "... given in questions 32 to 41 ..." -- which means
# nothing once the question stands alone in a paper.
_RANGE_WORDS = re.compile(
    r"^\s*in\s+(?:each\s+of\s+)?(?:the\s+)?(?:following|questions?\s*,?\s*(?:from\s+)?"
    r"\d{1,3}\s*,?\s*(?:to|and|-|–|—)\s*\d{1,3})\b\s*,?"
    r"|\s*\b(?:given\s+)?(?:in|for)\s+(?:the\s+)?questions?\s*,?\s*(?:from\s+)?"
    r"\d{1,3}\s*,?\s*(?:to|and|-|–|—)\s*\d{1,3}\b", re.I)
_UNIT_EXERCISE = re.compile(r"^\s*(?:\(C\)\s*)?Exercises?\s*$", re.I)
_UNIT_END_WORDS = re.compile(r"^\s*(?:\(D\)\s*)?(?:Activit|Applicat|Games|Think)", re.I)


def _science_section(line: str) -> Optional[tuple[str, str]]:
    return _SCIENCE_SECTION.get(_norm(line))


def _maths_section(line: str) -> Optional[tuple[str, str]]:
    m = _MATHS_SECTION_LINE.match(line)
    return _MATHS_SECTION.get(_norm(m.group(2))) if m else None


def _range_kind(text: str) -> Optional[str]:
    """The section kind an instruction names, or None (then: SA)."""
    low = text.lower()
    if "fill in the blank" in low:
        return FILL_BLANK
    if "true" in low and "false" in low:
        return TRUE_FALSE
    if "option" in low or "correct answer" in low:
        return MCQ_SINGLE
    return None


def _chapter_lines(pages: list[str], book: Book, chapter: int) -> list[_Tagged]:
    """The chapter's kept lines. The book's own labels and shadow fragments
    are dropped wherever they stand; a line equal to the chapter title only
    where a running head stands -- anywhere else it is question text:
    "... where x and y are" / "rational numbers." (heep201 p.16)."""
    titles = [CHAPTER_TITLES.get((book.grade, book.subject), {}).get(chapter, "")]

    def drop(line: str) -> bool:
        return (_is_furniture(line, ()) or _is_shadow_fragment(line, titles)
                or _UNIT_HEAD.match(line) is not None
                or _CHAPTER_HEAD.match(line) is not None)

    def title(line: str) -> bool:
        return _is_running_head(line, *titles)

    return _tag(pages, drop, title)


def _verify_chapter(pages: list[str], book: Book, chapter: int) -> bool:
    """The file prints its own chapter number where the layout puts it.

    Maths 6-8: "UNIT 2" / "UNIT-2"; maths 9-10: "EXERCISE 2.1"; science: the
    chapter number as display type on page 1 ("[[display]] 2", or with the
    title raised beside it, "14 ^(Water)"), or plain "CHAPTER" beside it.
    """
    lines = [_unaside(l.strip())[0] for p in pages for l in p.split("\n")]
    if book.layout == LAYOUT_MATHS_UNIT:
        return any((m := _UNIT_HEAD.match(l)) and int(m.group(1)) == chapter for l in lines)
    if book.layout == LAYOUT_MATHS_EXERCISE:
        return any((m := _EXERCISE.match(l)) and int(m.group(1)) == chapter for l in lines)
    first = [l.replace(DISPLAY_MARK, "").strip() for l in pages[0].split("\n")
             if l.startswith(DISPLAY_MARK)] if pages else []
    return any(re.match(rf"^{chapter}(?:\s+\^\(.*\))?$", l) for l in first)


def parse_chapter_pages(pages: list[str], book: Book, source_file: str,
                        chapter: int) -> list[ExemplarQuestion]:
    """Every question of one chapter (or unit) file, answers not yet paired."""
    tagged = _chapter_lines(pages, book, chapter)
    name = CHAPTER_TITLES.get((book.grade, book.subject), {}).get(chapter, "")
    verified = _verify_chapter(pages, book, chapter)
    if book.layout == LAYOUT_MATHS_UNIT:
        qs = _parse_maths_unit(tagged, book, source_file, chapter, name)
    elif book.layout == LAYOUT_MATHS_EXERCISE:
        qs = _parse_maths_exercises(tagged, book, source_file, chapter, name)
    else:
        qs = _parse_science(tagged, book, source_file, chapter, name)
    for q in qs:
        q.chapter_verified = verified
    return qs


class _Collector:
    """Accumulates the current question and closes it into `questions`."""

    def __init__(self, book: Book, source_file: str, chapter: int, name: str):
        self.book, self.source_file, self.chapter, self.name = book, source_file, chapter, name
        self.questions: list[ExemplarQuestion] = []
        self.cur: Optional[list] = None       # [number, page, kind, label, scope, lines]

    def open(self, n: int, page: int, kind: str, label: str, scope: str,
             first: Optional[str], prefix: str) -> None:
        self.close()
        self.cur = [n, page, kind, label, scope, _head(first, prefix)]

    def add(self, t: _Tagged) -> None:
        if self.cur is not None:
            self.cur[5].append(t.content)

    def close(self) -> None:
        if self.cur is None:
            return
        n, page, kind, label, scope, lines = self.cur
        self.cur = None
        self.questions.append(_build(self.book, self.chapter, self.name, kind, label, str(n),
                                     lines, self.source_file, page, scope))


def _parse_science(tagged, book, source_file, chapter, name) -> list[ExemplarQuestion]:
    col = _Collector(book, source_file, chapter, name)
    kind = label = None
    last = 0
    for t in tagged:
        sec = _science_section(t.text)
        if sec is not None:
            col.close()
            kind, label = sec
            continue
        m = _QUESTION_START.match(t.text)
        if m and kind is not None and last < int(m.group(1)) <= last + 2:
            last = int(m.group(1))
            col.open(last, t.page, kind, label, "", m.group(2), t.prefix)
            continue
        col.add(t)
    col.close()
    return col.questions


def _parse_maths_unit(tagged, book, source_file, chapter, name) -> list[ExemplarQuestion]:
    """``(C) Exercise`` to ``(D) Activities``; kinds from "In questions a to b".

    Every line naming a question range is an instruction, and so are the
    lines after it up to the next question: they belong to no question
    (appending them to the question before is how "In questions 39 to 98
    state whether..." became the text of option (D) of question 38). Three
    kinds of instruction are the book's section kinds (MCQ, fill in the
    blanks, true/false); the true/false and fill-in ones also lead every stem
    they cover, without their range (`_lead_with_instruction`), because the
    served type cannot say "true or false" or "using <, = or >". Any other
    short instruction -- "In questions 56 to
    74, choose a letter x, y, z ... and write the corresponding expressions"
    -- is part of every question it covers and is put in front of each stem.
    An instruction that carries data (a table, a figure, more than three
    lines: "Using the following frequency table, answer question 20-22")
    makes its questions unanswerable without it: ``shared-stimulus``.
    """
    at = next((i for i, t in enumerate(tagged) if _UNIT_EXERCISE.match(t.text)), None)
    if at is None:
        return []
    body = tagged[at + 1:]
    end = len(body)
    for i, t in enumerate(body):
        nxt = body[i + 1].text if i + 1 < len(body) else ""
        if (re.match(r"^\s*\(D\)\s*\S", t.text) and _UNIT_END_WORDS.match(t.text)
                or t.text.strip() == "(D)" and _UNIT_END_WORDS.match(nxt)):
            end = i
            break
    body = body[:end]

    col = _Collector(book, source_file, chapter, name)
    # (first, last or None, lines): None is an instruction that names no range
    # and runs to the next instruction.
    ranges: list[tuple[int, Optional[int], list[str]]] = []
    last = 0
    preamble: Optional[list[str]] = []        # lines of the instruction being read
    for i, t in enumerate(body):
        m = _QUESTION_START.match(t.text)
        if m and last < int(m.group(1)) <= last + 2:
            last = int(m.group(1))
            col.open(last, t.page, SA, "Exercise", "", m.group(2), t.prefix)
            preamble = None
            continue
        r = None
        open_ended = False
        if not m and _INSTRUCTION_START.match(t.text):
            # The range may wrap: "... Question 91 to" / "94 into words."
            nxt = body[i + 1].text if i + 1 < len(body) else ""
            r = _RANGE.search(f"{t.text} {nxt}")
            if r and r.start() >= len(t.text):
                r = None
            # No range: "In each of the following, state whether the statements
            # are true (T) or false (F)." (heep201 p.15). Appended to the
            # question before, it left the 30 true/false questions after it
            # typed as 3-mark short answers. Taken as an instruction only when
            # it names a kind whose questions show whether they belong to it
            # (true/false by the answer, fill-in by the blank; see
            # `_close_open_kinds`) and the next question follows at once.
            # Not from a feature panel ("[[aside]] Express 5 + 7n in words"),
            # and not borrowing its kind from the next line when that line is
            # an instruction or a question of its own (gemp104 p.8).
            follow = ("" if _INSTRUCTION_START.match(nxt) or _QUESTION_START.match(nxt)
                      else nxt)
            # And only after a finished sentence: feep105 p.8 prints "(f)" /
            # "State whether true or false: The total number ..." as part
            # (f) of question 37, which is no instruction.
            before = body[i - 1].text.strip() if i else ""
            open_ended = (r is None and not t.aside
                          and (not before or re.search(r"[.?!]$", before) is not None)
                          and not re.search(r":\s*\S", t.text)
                          and _range_kind(f"{t.text} {follow}") in (TRUE_FALSE, FILL_BLANK)
                          and _next_question_within(body, i, last, 3))
        if r or open_ended:
            col.close()
            preamble = [t.content]
            if r:
                ranges.append((int(r.group(1)), int(r.group(2)), preamble))
            else:
                ranges.append((last + 1, None, preamble))
            continue
        if preamble is not None:
            preamble.append(t.content)
        else:
            col.add(t)
    col.close()

    labels = {MCQ_SINGLE: "Multiple Choice Questions", FILL_BLANK: "Fill in the blanks",
              TRUE_FALSE: "True or False"}
    for k, (a, b, lines) in enumerate(ranges):
        text = _join(lines)
        kind = _range_kind(text)
        open_end = b is None
        if b is None:
            later = [r[0] for r in ranges[k + 1:] if r[0] >= a]
            b = min(later) - 1 if later else 10 ** 6
        covered = [q for q in col.questions if a <= int(q.number) <= b]
        for q in covered:
            if kind is not None:
                q.section_kind, q.section_label = kind, labels[kind]
                q.open_run = k + 1 if open_end else 0
                if kind in OPTION_KINDS:
                    _reoption(q)
                else:
                    _lead_with_instruction(q, text)
            elif (len(lines) > 3 or FIGURE_MARK in text or LOSS in text
                  or ASIDE_MARK in text or _POINTS_BACK.match(text)):
                q.shared_stimulus = True
            else:
                q.stem = f"{text} {q.stem}".strip()
                q.raw_lines = [text] + q.raw_lines
                q.instruction = text
    return col.questions


# "Now answer Questions 62 to 65:" (gemp109 p.22) answers questions about
# what was printed before it -- there, the triangles of Fig. 9.28. Served
# alone, "All triangles have the same base and the same altitude." was keyed
# True, which without the figure is false.
_POINTS_BACK = re.compile(r"^\s*Now\b", re.I)


def _question_instruction(text: str) -> str:
    """A range instruction as one question's lead-in, in the book's words:
    "In questions 39 to 98, state whether the given statements are true (T)
    or false (F)." -> "State whether the given statements are true (T) or
    false (F):". Only the range goes; every other word is the book's."""
    s = re.sub(r"\s+", " ", _RANGE_WORDS.sub(" ", text)).strip(" ,")
    s = re.sub(r"[\s.:;,]+$", "", s)
    return f"{s[0].upper()}{s[1:]}:" if s else ""


def _lead_with_instruction(q: ExemplarQuestion, text: str) -> None:
    """Put a true/false or fill-in-the-blank instruction in front of the stem.

    The kind alone does not reach the student: the served type is
    ``very_short_answer`` for both (no true/false type exists in the schema
    or the app), so "XXIX = 31" keyed "False" was served with nothing asking
    for true or false, and "0 _______ 1" keyed "<" without "using <, = or >"
    (feep103 p.8). Before this, 568 true/false and 418 fill-in records of
    the committed bank carried no instruction at all.
    """
    lead = _question_instruction(text)
    if lead:
        q.instruction = lead
        q.stem = f"{lead} {q.stem}".strip()


def _drop_instruction(q: ExemplarQuestion) -> None:
    """Undo `_lead_with_instruction` for a question the kind turned out not to fit."""
    if q.instruction and q.stem.startswith(q.instruction):
        q.stem = q.stem[len(q.instruction):].lstrip()
    q.instruction = ""


def _next_question_within(body: list[_Tagged], i: int, last: int, n: int) -> bool:
    """The next question (``last + 1``) starts within `n` lines after line `i`."""
    for t in body[i + 1:i + 1 + n]:
        m = _QUESTION_START.match(t.text)
        if m and int(m.group(1)) == last + 1:
            return True
    return False


def _reoption(q: ExemplarQuestion) -> None:
    """Re-split a question built as SA once its range says it is an MCQ."""
    lines = q.raw_lines
    stem_lines, options, labels, err = _split_options(lines, _option_style(lines))
    if not options and not err:
        err = "no options found"
    stem = _join(stem_lines)
    if any(_is_orphan(l) for l in stem_lines[1:]):
        stem = LOSS + stem
    q.stem, q.options, q.option_labels, q.options_error = stem, options, labels, err


def _parse_maths_exercises(tagged, book, source_file, chapter, name) -> list[ExemplarQuestion]:
    """``EXERCISE N.k`` blocks; kind from the (B)-(E) head above each."""
    col = _Collector(book, source_file, chapter, name)
    kind = label = None
    scope = None
    last = 0
    preamble = False
    for t in tagged:
        sec = _maths_section(t.text)
        if sec is not None or re.match(r"^\s*Sample\s+Questions?\b", t.text, re.I):
            col.close()
            scope = None                      # solved samples until the next EXERCISE
            if sec is not None:
                kind, label = sec
            continue
        ex = _EXERCISE.match(t.text)
        if ex:
            col.close()
            scope = f"{ex.group(1)}.{ex.group(2)}" if int(ex.group(1)) == chapter else None
            last = 0
            preamble = True
            continue
        if scope is None or kind is None:
            continue
        m = _QUESTION_START.match(t.text)
        if m and last < int(m.group(1)) <= last + 2:
            last = int(m.group(1))
            col.open(last, t.page, kind, label, scope, m.group(2), t.prefix)
            preamble = False
            continue
        if not preamble:
            col.add(t)
    col.close()
    return col.questions


# --------------------------------------------------------------------------- #
# answers files
# --------------------------------------------------------------------------- #

@dataclass
class AnswerKey:
    """Answers by (chapter, exercise, number), and the numbers that are not safe."""

    answers: dict[tuple[int, str, str], tuple[str, int]] = field(default_factory=dict)
    ambiguous: set[tuple[int, str, str]] = field(default_factory=set)
    blocks: list[dict] = field(default_factory=list)    # report: what was found where


def _page_chapter_numbers(page: str) -> set[int]:
    """Chapter numbers a page prints as a heading.

    Classes 6-8: "Chapter 3" (plain or display). Classes 9-10: a bare display
    number beside a "Chapter" / "C hapter" / "CHAPTER" label on the same page.
    """
    lines = [_unaside(l.strip())[0] for l in page.split("\n")]
    out = {int(m.group(1)) for l in lines if (m := _CHAPTER_HEAD.match(l))}
    labelled = any(_norm(l.replace(DISPLAY_MARK, "")) == "chapter" for l in lines)
    if labelled:
        out |= {int(m.group(1)) for l in lines
                if (m := re.match(rf"^{re.escape(DISPLAY_MARK)}\s*(\d{{1,2}})$", l))}
    return out


def parse_answer_pages(pages: list[str], book: Book,
                       limits: Optional[dict[tuple[int, str], int]] = None) -> AnswerKey:
    """Cut an answers file into blocks and number the answers in each.

    `limits` is the highest question number of each (chapter, exercise): an
    answer line that opens with a value like "14. 5 cm" can then never start
    an entry beyond the block's last question.
    """
    # The answers files carry no chapter-title running heads. Measured over
    # all ten answers files, the only lines a title matched were answer text:
    # feep2an "Water" (answer 8 of chapter 2) and "water." / geep1an "soil."
    # (last lines of answers, which were served cut short). So titles are
    # not furniture here.
    def drop(line: str) -> bool:
        return _is_furniture(line, ())

    tagged = _tag(pages, drop)
    key = AnswerKey()
    blocks: list[tuple[Optional[int], str, int, list[_Tagged]]] = []  # chapter, scope, page, lines
    if book.layout == LAYOUT_SCIENCE:
        chapters_on = {i: _page_chapter_numbers(p) for i, p in enumerate(pages, start=1)}
        for t in tagged:
            sec = _science_section(t.text)
            if sec is not None and sec[0] == MCQ_SINGLE:
                on_page = chapters_on.get(t.page, set())
                blocks.append((next(iter(on_page)) if len(on_page) == 1 else None, "", t.page, []))
                continue
            if sec is not None or _CHAPTER_HEAD.match(t.text):
                continue                      # a section head is not answer text
            if blocks:
                blocks[-1][3].append(t)
    elif book.layout == LAYOUT_MATHS_UNIT:
        stopped = False
        for t in tagged:
            m = _UNIT_HEAD.match(t.text)
            if m:
                blocks.append((int(m.group(1)), "", t.page, []))
                stopped = False
                continue
            if re.match(r"^\s*\(D\)\s*\S", t.text) and _UNIT_END_WORDS.match(t.text):
                stopped = True                # answers to the games, not the exercise
                continue
            if blocks and not stopped:
                blocks[-1][3].append(t)
    else:
        for t in tagged:
            m = _EXERCISE.match(t.text)
            if m:
                blocks.append((int(m.group(1)), f"{m.group(1)}.{m.group(2)}", t.page, []))
                continue
            if blocks:
                blocks[-1][3].append(t)

    seen_blocks: dict[tuple[int, str], int] = {}
    for chapter, scope, _, _ in blocks:
        if chapter is not None:
            seen_blocks[(chapter, scope)] = seen_blocks.get((chapter, scope), 0) + 1
    start = re.compile(r"^\s*(\d{1,3})\s?\.(?!\d)\s*(.*)$")
    for chapter, scope, page, lines in blocks:
        report = {"chapter": chapter, "scope": scope, "page": page, "answers": 0}
        key.blocks.append(report)
        if chapter is None:
            report["refused"] = "chapter number not printed once on the block's page"
            continue
        if seen_blocks[(chapter, scope)] > 1:
            report["refused"] = "block heading occurs twice"
            continue
        entries, amb = _numbered_entries(lines, start, limit=(limits or {}).get((chapter, scope)))
        for n, p, body in entries:
            key.answers[(chapter, scope, str(n))] = (_answer_text(body), p)
        key.ambiguous |= {(chapter, scope, str(n)) for n in amb}
        report["answers"] = len(entries)
        report["ambiguous"] = len(amb)
    return key


def pair_answers(questions: list[ExemplarQuestion], key: AnswerKey, answer_file: str) -> None:
    """Attach answers by (chapter, exercise, number) -- exact, never nearest."""
    for q in questions:
        if not q.chapter_verified or q.key in key.ambiguous or q.key not in key.answers:
            continue
        text, page = key.answers[q.key]
        if (q.section_kind not in OPTION_KINDS and q.section_kind != MATCHING
                and re.fullmatch(r"\s*\(?[a-dA-D]\)?\.?\s*", text)):
            # "Encircle the odd one of the following (Questions 26 to 30)",
            # keyed "(c)": an MCQ the instructions did not call one. Only if
            # its own lines split into a clean set of options.
            before = (q.stem, q.options, q.option_labels, q.options_error)
            _reoption(q)
            if q.options_error or len(q.options) < 4:
                q.stem, q.options, q.option_labels, q.options_error = before
                continue                      # a bare letter is no answer to prose
            q.section_kind, q.section_label = MCQ_SINGLE, "Multiple Choice Questions"
            if q.instruction and not q.stem.startswith(q.instruction):
                q.instruction = ""        # a true/false or fill-in lead-in; not an MCQ's
        if q.section_kind in OPTION_KINDS:
            if q.options_error:
                continue
            letters, note, err = _letters_from_answer(text, len(q.options))
            if err:
                q.answer_error = err
                continue
            if q.section_kind != MCQ_MULTIPLE and len(letters) != 1:
                q.answer_error = f"{len(letters)} letters for a single-correct question"
                continue
            q.answer_letters = letters
            labels = q.option_labels or list(_LETTERS)
            q.answer_text = "; ".join(
                f"({labels[_LETTERS.index(l)]}) {q.options[_LETTERS.index(l)]}" for l in letters)
            if note:
                q.answer_text += ". " + note
        else:
            text = text.strip()
            if not text:
                continue
            if q.section_kind == TRUE_FALSE:
                # The book keys true/false as a bare letter.
                text = {"T": "True", "F": "False"}.get(text.rstrip("."), text)
            q.answer_text = text
        q.answer_file, q.answer_page = answer_file, page
    _close_open_kinds(questions)


_TF_ANSWER = re.compile(r"\s*(?:T|F|True|False)\s*\.?\s*", re.I)


def _fits_kind(q: ExemplarQuestion) -> Optional[bool]:
    """Whether a question shows it belongs to its kind; None when it cannot
    tell (a true/false question with no answer). Loss marks are ignored:
    heep2an keys question 48 "� �False", question 100 "�� �8 9 6 4 1 0 ..."."""
    if q.section_kind == TRUE_FALSE:
        text = q.answer_text.replace(LOSS, " ").strip()
        return bool(_TF_ANSWER.fullmatch(text)) if text else None
    if q.section_kind == FILL_BLANK:
        return bool(_BLANK.search(q.stem[len(q.instruction):]))
    return True


def _close_open_kinds(questions: list[ExemplarQuestion]) -> None:
    """End an instruction that named no range where its questions stop fitting.

    "In each of the following, state whether the statements are true (T) or
    false (F)." (heep201 p.15) covers questions 48-99, and question 100,
    "Solve the following: Select the rational numbers ...", follows with no
    instruction between. The first question the kind does not fit ends the
    run: it and every later question of that run go back to short answer.
    """
    ended: set[tuple[str, int]] = set()
    for q in questions:
        if not q.open_run:
            continue
        run = (q.source_file, q.open_run)
        if run not in ended and _fits_kind(q) is False:
            ended.add(run)
        if run in ended:
            q.section_kind, q.section_label, q.open_run = SA, "Exercise", 0
            _drop_instruction(q)


# --------------------------------------------------------------------------- #
# building a question and deciding whether it may be served
# --------------------------------------------------------------------------- #

_MATCHING = re.compile(r"^\s*Match\b|\bColumn\s*(?:I|A|1)\b", re.I)

# The question sentence printed AFTER the options, read into the last one:
# heep108 p.1 "(d) cell membrane, ribosome, mitochondria, chloroplast." then
# "The correct combination of terms with reference to an animal cell is _____."
_TRAILING_QUESTION = re.compile(
    r"^(?P<option>.*?\S)\.\s+(?P<question>[A-Z][^.?]*?(?:\?|_{2,}\s*\.?))\s*$")


def _lift_trailing_question(stem: str, options: list[str]) -> tuple[str, list[str]]:
    """Move a question sentence that closes the last option back to the stem.

    Only a whole sentence of at least four words that asks (ends "?" or holds
    a blank) and follows a full stop; anything less stays where it is.
    """
    m = _TRAILING_QUESTION.match(options[-1])
    if not m or len(m.group("question").split()) < 4:
        return stem, options
    return (f"{stem} {m.group('question')}".strip(),
            [*options[:-1], m.group("option")])


def _build(book: Book, chapter: int, name: str, kind: str, label: str, number: str,
           lines: list[str], source_file: str, page: int, scope: str) -> ExemplarQuestion:
    lines = [l for l in lines if l.strip()]
    if kind in OPTION_KINDS:
        stem_lines, options, labels, err = _split_options(lines, _option_style(lines))
        if not options and not err:
            err = "no options found"
    else:
        stem_lines, options, labels, err = lines, [], [], ""
    stem = _join(stem_lines)
    if any(_is_orphan(l) for l in stem_lines[1:]):
        stem = LOSS + stem
    if options and not err:
        stem, options = _lift_trailing_question(stem, options)
    if _MATCHING.search(stem):
        kind, label = MATCHING, "Matching"
    return ExemplarQuestion(
        grade=book.grade, subject=book.subject, chapter_number=chapter, chapter_name=name,
        section_kind=kind, section_label=label, number=number, stem=stem,
        source_file=source_file, page=page, scope=scope, options=options,
        option_labels=labels, options_error=err, raw_lines=lines)


# A question that points at a figure, graph or diagram the product does not
# have. "significant figures" is vocabulary, not a figure reference, so the
# plural alone does not count. "the following number line" is a drawing
# (gemp101 p.9 Q7: options X, Y, Z, W are points marked on it); "On the
# number line, ..." is not.
_FIGURE = re.compile(
    r"\bFigs?\.|\bFigs?\s*\d|\bfigure\b|\bdiagrams?\b|\bgraphs?\b|\bsketch\b|"
    r"\bshown (?:in|below|above|alongside|here)\b|\bas shown\b|\bpictures?\b|"
    r"\bgiven below\b.*\b(?:figure|diagram|graph)\b|\bplot\b|\bdraw\b|\bpictograph\b|"
    r"\bbar graph\b|\bconstruct\b|\bmaps?\b|\bpie chart\b|"
    r"\b(?:following|given|above|below) table\b|\btable (?:given|below|above)\b|"
    r"\b(?:following|given|above|below)\s+number\s+lines?\b|"
    + re.escape(FIGURE_MARK), re.I)

# "The table shows ..." names a table without saying where it is. When the
# table was set as text its rows follow in the stem (gemp101 p.18 Q113: every
# continent and its temperature; gemp107 p.30 Q138: ten deserts and areas);
# when it was drawn, nothing follows (heep208 p.18 Q129-130: the planets'
# masses and distances are not in the text). Rows of such a table are numbers,
# so fewer than four numbers after the phrase means the table is missing.
_TABLE_LEAD = re.compile(r"\bthe table (?:shows|lists|gives)\b", re.I)
_TABLE_ROWS_MIN = 4


def needs_table(stem: str) -> bool:
    """True when the stem talks about "the table" but the table is not in it."""
    m = _TABLE_LEAD.search(stem)
    return bool(m) and len(re.findall(r"\d[\d,.]*", stem[m.end():])) < _TABLE_ROWS_MIN


# An answer whose sub-part label is followed straight away by the next label
# lost that part: feep1an p.9 "26. (a)" / "(b) Football (c) Tennis." -- (a) was
# the tally table, drawn. "(a) and (b)" and "(a), (b)" are lists of labels, not
# parts, so only whitespace may separate the two labels.
_SUB_LABEL = re.compile(r"(?<![\w)])\(?([a-z]|[ivx]{1,4})\)", re.I)
_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii")


def _next_label(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a in _ROMAN and b in _ROMAN and _ROMAN.index(b) == _ROMAN.index(a) + 1:
        return True
    return len(a) == len(b) == 1 and ord(b) == ord(a) + 1


def has_empty_subpart(answer: str) -> bool:
    """True when a labelled part of the answer has nothing in it."""
    labels = list(_SUB_LABEL.finditer(answer))
    return any(not answer[x.end():y.start()].strip() and _next_label(x.group(1), y.group(1))
               for x, y in zip(labels, labels[1:]))

# "Which of the following figures ..." is a figure reference unless the
# options name the figures in words ("Rectangle", "Kite", ...): heep205 p.17
# asks it with options P, Q, R, S (shapes drawn on the page), gemp112 p.29
# with bare labels "(a) (b) (c) (d)" whose options are the pictures.
_PLURAL_FIGURES = re.compile(r"\b(?:following|given|these|those|above|below)\s+figures\b", re.I)
_EMPTY_LABELS = re.compile(r"\(([a-dA-D])\)\s*\((?!\1)[a-dA-D]\)\s*\([a-dA-D]\)")


def needs_figure(stem: str, options: list[str]) -> bool:
    """True when the question can only be answered by looking at a figure."""
    if _EMPTY_LABELS.search(stem):
        return True
    if _PLURAL_FIGURES.search(stem):
        worded = options and all(len(re.sub(r"\W", "", o)) > 2 for o in options)
        return not worded
    return False


# The reaction arrow of the class 10 science answers extracts as the letter o
# (jeep1an p.9 "Zn + 2HCl o ZnCl₂ + H₂"): a lost symbol, not a word.
_LOST_ARROW = re.compile(r"(?<=[A-Za-z0-9)₀-₉])\s+o\s+(?=[A-Z0-9(])")


# An "answer" that sends the reader elsewhere or asks for an opinion instead of
# answering: "Hint: See pages 223 and 224 of Chapter 18 of NCERT Science
# textbook" (geep1an), "Hint: Take help of elders or use internet." (feep2an).
# Faithful to the book, useless as a key. A sentence that points is dropped;
# if fewer than three words of answer remain, the whole answer is a pointer.
_POINTER_SENTENCE = re.compile(
    r"\bsee\s+pages?\s+\d|\bpages?\s+\d+.*\btextbook\b|\btake help of\b|"
    r"\buse (?:the )?internet\b|\bask (?:your )?elders\b|^\(?\s*Activity\s+\d+\.\d+\b|"
    r"\baccording to your (?:own )?(?:stand|opinion|view)|"
    r"^students may (?:come up|design|discuss)\b|"
    r"^(?:Describe|Discuss|Explain)\s+\w+(?:\s+(?:and|or)\s+\w+)?\s*\.?$", re.I)
_HINT = re.compile(r"^\s*Hint\s*[:—–-]?\s*", re.I)


def is_pointer_answer(text: str) -> bool:
    body = _HINT.sub("", text.strip())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", body) if s.strip()]
    kept = [s for s in sentences if not _POINTER_SENTENCE.search(_HINT.sub("", s))]
    if len(kept) == len(sentences):
        return False
    return len(re.findall(r"\w+", " ".join(kept))) < 3


_BLANK = re.compile(r"_{2,}|\.{4,}|…|-{3,}")

# A question that leans on another question or on data printed elsewhere
# ("In question 42 above, ...", "Which quantities in the previous question
# ...", "Now answer Questions 12 to 15:") cannot stand alone in a paper.
_CROSS_REFERENCE = re.compile(
    r"\b(?:in|of|to|from)\s+questions?\s+(?:no\.?\s*)?\d|\b(?:above|previous|preceding)\s+"
    r"questions?\b|\bquestions?\s+(?:that\s+follow|given\s+above)\b|\banswer\s+questions?\s+\d",
    re.I)

EXCLUSION_REASONS = (
    "chapter-unverified", "shared-stimulus", "options-unparsed", "no-answer", "answer-not-an-option",
    "figure-unavailable", "symbol-loss", "reading-order", "blank-lost", "table-layout",
    "stem-too-short", "answer-is-a-pointer", "answer-part-missing",
)


def exclusion_reason(q: ExemplarQuestion) -> Optional[str]:
    """Why this question may not be served, or None. First reason wins."""
    if not q.chapter_verified:
        return "chapter-unverified"
    if q.shared_stimulus or _CROSS_REFERENCE.search(q.stem[len(q.instruction):]):
        return "shared-stimulus"
    if q.section_kind in OPTION_KINDS and q.options_error:
        return "options-unparsed"
    if q.answer_error:
        return "answer-not-an-option"
    if not q.has_answer:
        return "no-answer"
    everything = " ".join([q.stem, *q.options, q.answer_text])
    if _FIGURE.search(everything) or needs_figure(q.stem, q.options) or needs_table(q.stem):
        return "figure-unavailable"
    if LOSS in everything or _LOST_ARROW.search(everything):
        return "symbol-loss"
    if ASIDE_MARK in everything:
        return "reading-order"
    # The question without a true/false or fill-in lead-in (`_lead_with_instruction`),
    # which would otherwise supply the blank or the length it lacks. Any other
    # instruction is part of the question: "Encircle the odd one of the
    # following (Questions 26 to 30)" is the whole stem of five MCQs.
    own = (q.stem[len(q.instruction):] if q.section_kind in (TRUE_FALSE, FILL_BLANK)
           else q.stem)
    if q.section_kind == FILL_BLANK and not _BLANK.search(own):
        # The blank was a drawn rule, not text: "Two squares are congruent,
        # if they have same ." cannot be answered from the text.
        return "blank-lost"
    if q.section_kind == MATCHING:
        return "table-layout"
    # Counted with symbols: "(– 19) × (– 11) = 19 × 11" is a whole question
    # in eight letters and digits.
    if len(re.sub(r"\s", "", own)) < 6:
        return "stem-too-short"
    if q.section_kind not in OPTION_KINDS and is_pointer_answer(q.answer_text):
        return "answer-is-a-pointer"
    if q.section_kind not in OPTION_KINDS and has_empty_subpart(q.answer_text):
        return "answer-part-missing"
    return None


# --------------------------------------------------------------------------- #
# wire shape
# --------------------------------------------------------------------------- #

_BUILT_AT = "2026-09-22T00:00:00Z"
_ROMAN_CLASS = {6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}


def _difficulty_for(marks: int) -> str:
    """Same marks rule the CBE and board corpora use, so sources compare."""
    return "easy" if marks <= 2 else "medium" if marks <= 3 else "hard"


def to_bank_record(q: ExemplarQuestion, chapter_id: str = "",
                   current_textbook: bool = True) -> dict:
    """The served bank's camelCase shape (`frontend/.../question.dart`).

    Options go in ``parts[0].options`` with ``correctOption`` the letter(s):
    ``"b"`` for single-correct, ``"b,c"`` for multiple-correct, which also
    carries ``answerType: "multipleChoice"`` and ``metadata.multipleCorrect``
    so paper generation never marks it as single-correct.
    """
    qid = q.record_id
    marks = MARKS_BY_KIND[q.section_kind]
    parts = []
    if q.section_kind in OPTION_KINDS:
        parts.append({
            "id": f"{qid}:p1", "partNumber": 1, "text": "", "textLatex": "",
            "marks": marks,
            "answerType": "multipleChoice" if q.section_kind == MCQ_MULTIPLE else "singleChoice",
            "options": list(q.options),
            "correctOption": ",".join(q.answer_letters),
            "expectedAnswer": q.answer_text,
            "alternativeAnswers": [],
        })
    folder = f"class{q.grade}-{q.subject.lower()}"
    doc_id = f"ncert-exemplar:{folder}/{q.source_file}"
    answer_doc = f"ncert-exemplar:{folder}/{q.answer_file}"
    return {
        "id": qid,
        "questionBankId": doc_id,
        "subject": q.subject,
        "grade": q.grade,
        "chapterIds": [chapter_id] if chapter_id else [],
        "competencyIds": [],
        "bloomLevel": "understand",
        "difficulty": _difficulty_for(marks),
        "type": _TYPE_BY_KIND[q.section_kind],
        "stem": q.stem,
        "stemLatex": "",
        "parts": parts,
        "answerScheme": {
            "totalMarks": marks,
            "markingPoints": [{
                "id": f"{qid}:mp1", "description": q.answer_text, "marks": marks,
                "keyword": "", "isRequired": True, "synonyms": [],
            }],
            "rubricLevels": [],
            "commonErrors": [],
            "alternativeAnswers": [],
            "modelAnswer": q.answer_text,
            "modelAnswerLatex": "",
            "hasPartialCredit": marks > 1,
            "metadata": {"answerPage": q.answer_page, "answerFile": q.answer_file},
            "provenance": PROVENANCE,
            "sourcePaperCode": "",
            "sourceDocumentId": q.answer_file,
        },
        "estimatedTimeMinutes": max(1, marks * 2),
        "marks": marks,
        "language": "en",
        "source": SOURCE,
        "qualityScore": 0.9,
        "tags": [t for t in (q.chapter_name, q.section_label) if t],
        "createdAt": _BUILT_AT,
        "updatedAt": _BUILT_AT,
        "metadata": {
            "exemplarChapter": {"number": q.chapter_number, "title": q.chapter_name,
                                "exercise": q.scope or None},
            "exemplarTextbook": "current" if current_textbook else "pre-2024",
            "sectionKind": q.section_kind,
            "sectionLabel": q.section_label,
            "questionNumber": q.number,
            "optionLabels": list(q.option_labels),
            "correctOptions": list(q.answer_letters),
            "multipleCorrect": q.section_kind == MCQ_MULTIPLE,
            "marksBasis": "assigned_by_section_kind",
            "answerDocumentId": answer_doc,
        },
        "diagramAssetId": None, "mapAssetId": None,
        "graphAssetId": None, "tableAssetId": None,
        "provenance": {
            "sourceDocumentId": doc_id,
            "pageNumber": q.page,
            "boundingBox": None,
            "method": "pdf_native",
            "confidence": 0.9,
        },
        "rights": {
            "origin": "NCERT",
            "redistribution": "unknown",
            "basis": "",
            "attribution": (f"NCERT, Exemplar Problems, Class {_ROMAN_CLASS.get(q.grade, q.grade)} "
                            f"{q.subject}"),
        },
        "calibration": {"measured": False, "sampleSize": 0, "facility": None,
                        "discrimination": None, "basis": "assigned"},
        "version": 1,
        "supersededBy": None,
        "reviewState": "published",
    }


# --------------------------------------------------------------------------- #
# chapter mapping (rule Q2)
# --------------------------------------------------------------------------- #

_STOP = {"and", "the", "of", "in", "a", "an", "their", "its", "our"}


def _tokens(name: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", name).lower())
    # "Proportion" / "Proportions", "Construction" / "Constructions".
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w
                     for w in words if w not in _STOP)


def map_chapter(name: str, syllabus: list[tuple[str, str]]) -> Optional[str]:
    """The syllabus chapter id whose NAME equals this title, or None.

    Near-exact only -- the same words, ignoring case, punctuation, "and/the/
    of" and a plural s. Classes 6-8 Exemplar follows the old books, whose
    chapters mostly do not exist in the new ones; a containment match
    ("Light" inside "Light – Reflection and Refraction") would tag questions
    to a chapter that teaches something else. Two syllabus chapters with the
    same name map to nothing.
    """
    want = _tokens(name)
    if not want:
        return None
    exact = [cid for cid, cname in syllabus if _tokens(cname) == want]
    return exact[0] if len(exact) == 1 else None


def load_syllabus(path: Path) -> list[tuple[str, str]]:
    """(chapter id, chapter name) pairs of one syllabus file, [] if absent."""
    import json
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(c["id"], c["name"]) for u in data.get("units", []) for c in u.get("chapters", [])
            if c.get("id") and c.get("name")]


# --------------------------------------------------------------------------- #
# whole corpus
# --------------------------------------------------------------------------- #

def _chapter_of(path: Path) -> Optional[int]:
    """NCERT names chapter files <book code><NN>.pdf: feep102 is chapter 2."""
    m = re.match(r"^[a-z]{4}\d(\d\d)$", path.stem)
    return int(m.group(1)) if m else None


def parse_book(root: Path, book: Book, reader=read_pdf) -> tuple[list[ExemplarQuestion], dict]:
    """Every chapter question of one book, answers paired, plus a report."""
    folder = Path(root) / book.folder
    report: dict = {"grade": book.grade, "subject": book.subject, "files": [], "answers": {}}
    questions: list[ExemplarQuestion] = []
    for path in sorted(folder.glob("*.pdf")):
        if path.name == book.answers_file:
            continue
        reason = NOT_PARSED.get((book.folder, path.name))
        chapter = _chapter_of(path)
        if reason or chapter is None:
            report["files"].append({"file": path.name, "status": reason or "not a chapter file"})
            continue
        pages = reader(path)
        qs = parse_chapter_pages(pages, book, path.name, chapter)
        title = CHAPTER_TITLES.get((book.grade, book.subject), {}).get(chapter, "")
        report["files"].append({
            "file": path.name, "chapter": chapter, "title": title,
            "titleVerified": _title_in_file(title, pages),
            "chapterVerified": _verify_chapter(pages, book, chapter),
            "questions": len(qs), "status": "chapter" if qs else "no questions found"})
        questions += qs
    answer_path = folder / book.answers_file
    if answer_path.exists():
        limits: dict[tuple[int, str], int] = {}
        for q in questions:
            k = (q.chapter_number, q.scope)
            limits[k] = max(limits.get(k, 0), int(q.number))
        key = parse_answer_pages(reader(answer_path), book, limits)
        pair_answers(questions, key, book.answers_file)
        report["answers"] = {"file": book.answers_file, "answers": len(key.answers),
                             "ambiguous": len(key.ambiguous), "blocks": key.blocks}
    else:
        report["answers"] = {"file": book.answers_file, "missing": True}
    return questions, report


def parse_all(root: Path, reader=read_pdf,
              books: Iterable[Book] = BOOKS) -> tuple[list[ExemplarQuestion], list[dict]]:
    """Every book under `root` (the ``exemplar/`` folder)."""
    out: list[ExemplarQuestion] = []
    reports = []
    for book in books:
        qs, rep = parse_book(root, book, reader)
        out += qs
        reports.append(rep)
    return out, reports
