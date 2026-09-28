"""Chapter -> topic -> subtopic headings from the NCERT textbooks. No AI.

Why
---
The bank has to be tagged per chapter, topic and subtopic, and on 2026-09-22
zero questions were linked to any subtopic: the only subtopics in the product
were one demo school's 417, school-scoped. The user chose one board-level tree
built from the section headings of the books the schools actually teach from,
read deterministically from the PDF text layer -- a heading is a line the book
typesets as a heading, never a line a model thought sounded like one.

How a heading is recognised
---------------------------
Every line is rebuilt from PyMuPDF characters (`lines_from_rawdict`) and keeps
its size, font, weight and colour. Within one chapter:

  numbered      "2.1 Diversity in Plants and Animals" is a topic and "2.2.1 How
                to group plants?" a subtopic, when the first number is the
                chapter's and the line is set apart from the body (bold, larger
                or coloured). This is how the maths and science books of every
                class, old and new, mark their sections.
  section       Class 10 History numbers its sections afresh in each chapter:
                "1 The First World War, Khilafat ..." then "1.1 The Idea of
                Satyagraha" (jess302.pdf p2). Tried when the chapter-prefixed
                scheme finds fewer than two topics.
  unnumbered    The new Social Science books (classes 6-9) and Class 10
                Democratic Politics number nothing. Their headings are bold
                lines at least 1.5pt above the body size; the largest such size
                in the chapter is the topic level and the next one down the
                subtopic level (Exploring Society, class 6: 17pt ExtraBold and
                15pt Bold, fees102.pdf pp3, 7). Marked ``headingKind:
                "unnumbered"`` because it is the weaker signal. A heading
                lettered "a)", "a.", "(a)" or "A." in a run from a/A is a
                subtopic, named without its letter ("a) Temperature",
                gees102.pdf p5).

What is never a topic, whatever its type: activities, examples, figures,
tables, exercises, boxes ("LET'S EXPLORE", "Think and Reflect", "Probe and
ponder"), and the end-of-chapter matter ("Before we move on", "In a Nutshell",
"What you have learnt"), matched up to the first non-letter so trailing "?",
"!" or "..." do not hide them. A numbered line whose text starts in lower case
is a cross-reference in running text ("2.4 and 2.5. They also ...",
class8_curiosity.pdf p36), not a heading -- unless it is bold or larger than
body and carries exactly the next number of the chapter ("5.3 nth Term of an
AP", jemh105.pdf p8).

The measured precision and recall of all this, against headings read off the
pages by hand, are in ``academicos-data/syllabus/taxonomy/_manifest.json``
(`scripts/build_taxonomy.py`).

The second extractor: `extract_chapters`
----------------------------------------
This module holds a SECOND, independent reader, added by feat/product-calendar
and kept whole below: `extract_chapters` / `extract_headings`, which open the
PDFs themselves (PyMuPDF) and return `ExtractedChapter` -> `ExtractedTopic` ->
`SectionHeading`. It is source (b) of the decomposition templates
(`scripts/export_decomposition_templates.py`); the reader above is what
`scripts/build_taxonomy.py` uses to build the board-level taxonomy from
pre-read lines. They answer different questions and neither replaces the
other, so both rule sets are kept in full. Nothing is shared between the two
halves: the second half's line record, heading record, text cleaner, bold-face
list and section-number pattern are all its own (`_Line`, `SectionHeading`,
`_clean_run`, `_SECTION_BOLD_NAMES`, `_SECTION_NUMBERED`), so a change to one
reader cannot quietly move the other.
"""
from __future__ import annotations

import collections
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

@dataclass
class Line:
    """One printed line as the reader sees it, rebuilt from PyMuPDF characters."""
    page: int            # 1-based page in the PDF file
    text: str
    size: float          # the size most of the line's characters are set in
    font: str
    bold: bool
    color: int
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Line":
        return cls(**d)


_BOLD_NAMES = ("bold", "black", "heavy", "semibold", "demi", "halbfett", "extrab")


def _is_bold(font: str, flags: int) -> bool:
    return bool(flags & 16) or any(k in font.lower() for k in _BOLD_NAMES)


def _printed_near(seen: dict, c: str, x: float, y: float) -> bool:
    """The same character already printed within 1.2pt: an overprint copy.

    The copies are nudged a fraction of a point apart (that is how the bold
    is faked), so exact positions do not match; a character is never set
    twice within 1.2pt on purpose (the narrowest glyph, a 6pt superscript
    "l", advances ~1.7pt).
    """
    rx, ry = round(x), round(y)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for sx, sy in seen.get((c, rx + dx, ry + dy), ()):
                if abs(sx - x) < 1.2 and abs(sy - y) < 1.2:
                    return True
    return False


def lines_from_rawdict(page_no: int, raw: dict) -> list[Line]:
    """Rebuild a page's lines from ``page.get_text("rawdict")``.

    Two things in the books defeat PyMuPDF's own lines. Class 10 Science
    fakes bold by printing a heading five times at nearly the same spot, and
    kerned capitals split it mid-word, so ``1.1 CHEMICAL EQUATIONS`` arrives as
    ``1.1 CHEMIC`` x4, ``1.1 CHEMICAL EQUA``, ``AL EQUA`` x4, ``TIONS`` x4
    (jesc101.pdf page 2). So a character already printed at the same place is
    dropped, and what survives on one baseline in one size is joined back.
    """
    seen: dict[tuple, list] = {}
    pieces = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                chars = []
                for ch in span["chars"]:
                    c = ch["c"]
                    x, y = ch["origin"]
                    if _printed_near(seen, c, x, y):
                        continue
                    seen.setdefault((c, round(x), round(y)), []).append((x, y))
                    chars.append((c, ch["bbox"]))
                if not "".join(c for c, _ in chars).strip():
                    continue
                pieces.append({
                    "chars": chars, "size": round(span["size"], 2),
                    "font": span["font"], "flags": span["flags"],
                    "color": span["color"], "base": round(span["origin"][1], 1),
                })
    return _join_pieces(page_no, pieces)


def _join_pieces(page_no: int, pieces: list[dict]) -> list[Line]:
    """Join same-baseline, same-size pieces that touch horizontally."""
    groups: list[list[dict]] = []
    for p in pieces:
        ink = [b for c, b in p["chars"] if not c.isspace()]
        p["x0"] = min(b[0] for b in ink)
        p["x1"] = max(b[2] for b in ink)
        for g in groups:
            last = g[-1]
            if (abs(last["base"] - p["base"]) <= 1.0
                    and abs(last["size"] - p["size"]) <= 0.6
                    and -0.5 * p["size"] <= p["x0"] - last["x1"]
                    <= max(p["size"], last["size"]) * 1.2):
                g.append(p)
                break
        else:
            groups.append([p])
    out = []
    for g in groups:
        g.sort(key=lambda p: p["x0"])
        text = ""
        prev_x1 = None
        weight: dict[tuple, int] = {}
        for p in g:
            for c, b in p["chars"]:
                if (prev_x1 is not None and b[0] - prev_x1 > p["size"] * 0.2
                        and not text.endswith(" ") and c != " "):
                    text += " "
                text += c
                prev_x1 = b[2]
            k = (p["size"], p["font"], p["flags"], p["color"])
            weight[k] = weight.get(k, 0) + sum(1 for c, _ in p["chars"] if not c.isspace())
        size, font, flags, color = max(weight, key=weight.get)
        boxes = [b for p in g for _, b in p["chars"]]
        out.append(Line(
            page=page_no, text=" ".join(text.split()), size=size, font=font,
            bold=_is_bold(font, flags), color=color,
            x0=round(min(b[0] for b in boxes), 1), y0=round(min(b[1] for b in boxes), 1),
            x1=round(max(b[2] for b in boxes), 1), y1=round(max(b[3] for b in boxes), 1)))
    return out


def read_lines(path: Path) -> list[Line]:
    """Every line of every page of one PDF. The one PDF-touching function."""
    from academicos.corpus.ncert_exemplar import _pymupdf, require_verified_pymupdf
    require_verified_pymupdf()
    fitz = _pymupdf()
    doc = fitz.open(str(path))
    try:
        out: list[Line] = []
        for i, page in enumerate(doc, 1):
            out.extend(lines_from_rawdict(i, page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT)))
        return out
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# headings within one chapter
# --------------------------------------------------------------------------- #

@dataclass
class Heading:
    # 1 = topic, 2 = subtopic, 3 = a lettered heading below the subtopic
    # level: the tree has no third level, so `build_topics` leaves it out and
    # the build lists it as not represented rather than dropping it unseen.
    level: int
    name: str
    page: int
    number: str | None = None   # "2.1", "2.2.1"; None when the book prints none
    kind: str = "numbered"      # numbered | section | unnumbered
    # Index, in the reading-ordered lines it came from, of the line the heading
    # starts on. Merging by this, not by searching for the text again, keeps a
    # heading that wraps onto a second line in its place (-1: not recorded).
    line: int = -1


def body_style(lines: list[Line]) -> tuple[float, str]:
    """The (size, font) most characters are set in."""
    sizes: collections.Counter = collections.Counter()
    fonts: collections.Counter = collections.Counter()
    for ln in lines:
        n = len(ln.text)
        sizes[ln.size] += n
        fonts[ln.font] += n
    if not sizes:
        return 0.0, ""
    return sizes.most_common(1)[0][0], fonts.most_common(1)[0][0]


def _whitish(color: int) -> bool:
    return all(((color >> s) & 0xFF) >= 0xE0 for s in (16, 8, 0))


def _blackish(color: int) -> bool:
    return all(((color >> s) & 0xFF) <= 0x40 for s in (16, 8, 0))


# Box labels, feature panels and end matter. Matched against the start of the
# heading, case- and apostrophe-insensitively, up to a non-letter: "Do you
# know?", "Let's Explore!" and "Before we move on..." all match, "Sources of
# Energy" does not match "source".
_NOT_TOPICS = (
    "activity", "activities", "example", "examples", "fig ", "fig.", "figure", "table",
    "exercise", "exercises", "box", "project", "projects", "discuss", "source",
    "let's explore", "let us explore", "think about it", "don't miss out",
    "before we move on", "questions, activities", "questions and activities",
    "the big questions", "big questions", "keywords", "key words", "glossary",
    "in a nutshell", "let us enhance", "exploratory projects", "probe and ponder",
    "ever heard of", "know a scientist", "at a glance", "revise, reflect",
    "the journey beyond", "chapter summary", "summary", "figure it out",
    "think and reflect", "what you have learnt", "overview", "do you know",
    "did you know", "fun facts", "try this", "try these", "math talk", "puzzle time",
    "notes for", "a note on", "write in brief", "points to ponder", "remember",
    "happy investigating", "questions", "let's work", "let us work", "summing up",
    "sources for information", "chapter", "additional project", "additional projects",
    "additional activity", "additional activities", "suggested reading",
    "suggested readings", "notes", "class activity", "answer the following",
    # the recurring panels of the new books (Curiosity 6: "More to know!" 22
    # times, class6_curiosity.pdf pp84-270; Ganita Prakash 6: "Teacher's Note",
    # class6_ganita_prakash.pdf pp135-213), set in the sub-heading type
    "more to know", "more to do", "think it over", "teacher's note", "teachers' note",
    "discussion", "let's play a game", "let us play a game", "puzzle",
    "true or false", "steps to follow",
)


def _norm_words(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).lower().replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


_TRAILING = re.compile(r"[\s?!.:;,…]+$")


# Panels that Ganita Prakash sometimes numbers as a section of their own
# ("7.3 Mind the Mistake, Mend the Mistake", gegp207.pdf p18) and sometimes
# sets as an unnumbered box (class6_ganita_prakash.pdf p69). Numbered, the
# book counts it a section, and so does the tree.
_UNNUMBERED_PANELS = ("mind the mistake",)


def _not_a_topic(name: str, numbered: bool = False) -> bool:
    t = _TRAILING.sub("", _norm_words(name))
    for k in _NOT_TOPICS + (() if numbered else _UNNUMBERED_PANELS):
        if not t.startswith(k):
            continue
        rest = t[len(k):]
        # the key ends at a word boundary: nothing after it, or a non-letter
        if not rest or not rest[0].isalpha() or k.endswith((" ", ".")):
            return True
    return False


# A heading does not end on one of these: "Steps to follow: Start with a"
# (class6_ganita_prakash.pdf p97) and "More to" (class6_curiosity.pdf p96)
# are the first line of a sentence or of a label cut off at its line end.
_DANGLING = {"a", "an", "the", "to", "of", "with", "and", "or", "for", "from", "into"}


def _dangling(name: str) -> bool:
    words = re.findall(r"[\w']+", name)
    return bool(words) and not re.search(r"[?!.)…]$", name) and words[-1].lower() in _DANGLING


_NUMBERED = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?\.?\s*(.*)$")
_SECTION = re.compile(r"^(\d{1,2})\s+(\S.*)$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def _clean(text: str) -> str:
    """One line of heading text as printed: control characters out, ligatures
    spelt ("Reﬂ ection" -> "Reflection", class7_curiosity.pdf p179), no space
    before a comma ("Trade , trade routes", gees105.pdf p9)."""
    # composed form, so "Kauṭilya" is one string whichever way the PDF built it
    text = unicodedata.normalize("NFC", _CONTROL.sub(" ", text))
    for lig, letters in _LIGATURES.items():
        text = re.sub(lig + r"\s?(?=[a-z])", letters, text).replace(lig, letters)
    text = re.sub(r"\s+([,;:?!])", r"\1", text)
    # a raised ordinal run into the next word: "11thand 12th Centuries" (gees204.pdf p1)
    text = re.sub(r"(\d(?:st|nd|rd|th))(?=[a-z]{2})", r"\1 ", text)
    return " ".join(text.split()).strip(" -–—:")


# A line opening with one of these is the second line of something else (a
# table caption "... INDIA AND ITS NEIGHBOURS" / "FOR 2023", jess201.pdf p12).
_FRAGMENT_START = {"and", "or", "of", "for", "with", "to"}


def _starts_like_a_title(text: str) -> bool:
    t = text.lstrip("‘'\"“(")
    if not t or t.split()[0].lower() in _FRAGMENT_START:
        return False
    return t[0].isupper() or t[0].isdigit() or not t[0].isascii()


# "a) Temperature", "a. What are the Vedas?", "(c) Floods", "A. Primary
# activities": one letter, then ")" or ".", then the name. Not "E. F. Hayward"
# (initials: another letter label follows) and not "A Tale ..." (no mark).
_LETTER_LABEL = re.compile(r"^\(?([A-Za-z])[.)]\s+(?![A-Za-z][.)](?:\s|$))(\S.*)$")


def letter_label(text: str) -> tuple[str | None, str]:
    """(letter, name) of a lettered heading; (None, text) for any other."""
    m = _LETTER_LABEL.match(text)
    return (m.group(1), m.group(2)) if m else (None, text)


def _in_letter_run(label: str, prev: str | None) -> bool:
    """``label`` opens a run (a/A) or follows ``prev`` in the same case.

    A letter counts as a label only inside such a run: a lone "O. Henry" in
    heading type is an author, not item O of a list.
    """
    if label in "aA":
        return True
    return (prev is not None and prev.islower() == label.islower()
            and ord(label) == ord(prev) + 1)


def _latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ch.isascii() for ch in letters) / len(letters) if letters else 0.0


def _foreign_script(name: str, lines: list[Line]) -> bool:
    """A Devanagari epigraph in an English book (Curiosity 6 opens chapter 9
    with a Kabir doha in 18pt bold, class6_curiosity.pdf p181) is not a
    heading; a heading shares the script of the book's running text."""
    book_latin = _latin_share("".join(ln.text for ln in lines[:400])) >= 0.5
    return book_latin != (_latin_share(name) >= 0.5)


def _set_apart(ln: Line, body: float, body_font: str) -> bool:
    """Typeset unlike running text: bold, larger, or coloured in another font."""
    return (ln.bold or ln.size >= body + 0.9
            or (not _blackish(ln.color) and ln.font != body_font))


def _continuation(prev: Line, ln: Line, hang: float = 0.0) -> bool:
    """``ln`` carries on the heading ``prev`` on the next printed line.

    ``hang`` is how far a numbered heading's second line may sit right of its
    first: it hangs under the title, past the number ("11.7.1 Practical
    Applications of Heating Effect of" / "Electric Current" 42.7pt in,
    jesc111.pdf p20).
    """
    return (ln.page == prev.page and abs(ln.size - prev.size) < 0.3
            and ln.font == prev.font and ln.color == prev.color
            and -0.5 * ln.size <= ln.y0 - prev.y1 <= 0.8 * ln.size
            and (-max(40.0, 3 * ln.size) <= ln.x0 - prev.x0 <= max(40.0, 3 * ln.size) + hang
                 or abs(ln.x0 + ln.x1 - prev.x0 - prev.x1) <= 6 * ln.size))  # centred


def _with_continuations(lines: list[Line], i: int, stop, hang: float = 0.0) -> tuple[str, int]:
    """Text of the heading at ``lines[i]`` plus its wrapped lines below.

    Looks a few lines ahead, because in reading order a margin note at the
    same height can sit between a heading and its second line.
    """
    text, j = lines[i].text, i
    while len(text) < 160:
        nxt = next((k for k in range(j + 1, min(j + 5, len(lines)))
                    if _continuation(lines[j], lines[k], hang if j == i else 0.0)), None)
        if nxt is None or stop(lines[nxt].text):
            break
        j = nxt
        text += " " + lines[j].text
    return text, j


def reading_order(lines: list[Line]) -> list[Line]:
    """Page by page; top to bottom, column by column on a two-column page.

    PyMuPDF's content-stream order put "2.1 The Aristocracy and the New
    Middle Class" before its topic "2 The Making of Nationalism in Europe"
    (jess301.pdf p8, y=535 vs 55), so lines are re-sorted by height. But a
    plain top-to-bottom sort reads across columns: Democratic Politics ends
    its left column with "Caste and politics" (y=607) and opens the right
    one with its first sub-section "Caste inequalities" (y=71, jess403.pdf
    p10). A page is two-column when at least 8 body-size lines end before
    68% of its text width and 8 start after 60%.
    """
    body, _ = body_style(lines)
    by_page: dict[int, list[Line]] = collections.defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln)
    out: list[Line] = []
    for page in sorted(by_page):
        pl = by_page[page]
        width = max(ln.x1 for ln in pl)
        body_lines = [ln for ln in pl if abs(ln.size - body) < 0.6]
        left = [ln for ln in body_lines if ln.x1 < 0.68 * width]
        right = [ln for ln in body_lines if ln.x0 > 0.6 * width]
        if len(left) >= 8 and len(right) >= 8:
            split = min(ln.x0 for ln in right) - 5
            key = lambda ln: (ln.x0 >= split, round(ln.y0), ln.x0)
        else:
            key = lambda ln: (round(ln.y0), ln.x0)
        out.extend(sorted(pl, key=key))
    return out


def numbered_headings(lines: list[Line], chapter: int) -> list[Heading]:
    """"N.k Title" topics and "N.k.m Title" subtopics of chapter N, in order.

    A number is taken once, at its first appearance, and only if it comes
    after the previous one: running heads and cross-references repeat numbers
    out of order, headings do not.

    The lower-case and "Of ..."/"And ..." guards (`_starts_like_a_title`) are
    for cross-references and wrapped captions. They are waived for a line
    that is bold or larger than body and carries exactly the next number in
    the chapter's sequence: that is a heading which happens to open with
    "nth" ("5.3 nth Term of an AP", jemh105.pdf p8), "pH" ("2.4.2 pH of
    Salts", jesc102.pdf p13) or "Of" ("1.3 Of Crores and Crores!",
    class7_ganita_prakash.pdf p44; "5.1 Of Questions and Statements",
    gegp205.pdf p1).
    """
    body, body_font = body_style(lines)
    out: list[Heading] = []
    seen: set[tuple] = set()
    last: tuple = (0, 0)
    for i, ln in enumerate(lines):
        m = _NUMBERED.match(ln.text)
        if not m or int(m.group(1)) != chapter or not _set_apart(ln, body, body_font):
            continue
        k = int(m.group(2))
        sub = int(m.group(3)) if m.group(3) else 0
        key = (k, sub)
        # the number's width, roughly: 0.6em a character
        hang = 0.6 * ln.size * (len(ln.text) - len(m.group(4)))
        text, _ = _with_continuations(lines, i, lambda t: bool(_NUMBERED.match(t)), hang)
        name = _clean(_NUMBERED.match(text).group(4))
        next_in_sequence = key in ((last[0] + 1, 0), (last[0], last[1] + 1))
        strong = next_in_sequence and (ln.bold or ln.size >= body + 0.9)
        if (not name or not (strong or _starts_like_a_title(name)) or _not_a_topic(name, True)
                or len(name) > 150 or sum(ch.isalpha() for ch in name) < 3):
            continue
        if key in seen or key <= last:
            continue
        seen.add(key)
        last = key
        number = f"{chapter}.{k}" + (f".{sub}" if sub else "")
        out.append(Heading(level=2 if sub else 1, name=name, page=ln.page, number=number,
                           line=i))
    return out


def unnumbered_subtopics(lines: list[Line], topics: list[Heading]) -> list[Heading]:
    """Unnumbered sub-headings under numbered topics, merged in page order.

    Ganita Prakash numbers its topics ("5.4 Prime Factorisation", 17pt red)
    but not the sub-sections under them ("Divisibility by 4", 14pt blue,
    class6_ganita_prakash.pdf p184). Taken only when a chapter has no
    numbered subtopic: bold, not white, set between body and topic size, and
    only the one (size, font, colour) most used for such lines -- a third,
    smaller level (13pt black "Idli-Vada Game") stays out.

    The topics must carry the index of the line they were read from
    (`Heading.line`, set by `numbered_headings` on these same lines): a topic
    is never looked for again by its text, because a heading that wraps
    ("5.4 Parallel and Perpendicular Lines in Paper" / "Folding",
    class7_ganita_prakash.pdf p216) is in no single line, and a topic that
    cannot be placed would take the wrong subtopics.
    """
    if not topics:
        return topics
    missing = [t.number or t.name for t in topics if not 0 <= t.line < len(lines)]
    if missing:
        raise ValueError(f"topics without the line they were read from: {missing}")
    body, _ = body_style(lines)
    top = lines[topics[0].line].size
    cands = []
    for i, ln in enumerate(lines):
        if (ln.bold and not _whitish(ln.color) and body + 0.9 <= ln.size <= top - 0.9
                and not _NUMBERED.match(ln.text) and not ln.text[:1].isdigit()):
            text, _ = _with_continuations(lines, i, lambda t: False)
            name = _clean(text)
            if (sum(ch.isalpha() for ch in name) >= 3 and len(name) <= 110
                    and _starts_like_a_title(name) and not _not_a_topic(name)
                    and not _dangling(name)
                    and not name.endswith((",", ";", ".")) and not _foreign_script(name, lines)):
                cands.append((ln, name, i))
    # A text printed more than once in the chapter is a panel label, not a
    # section: it does not vote for the sub-heading style, nor is it taken.
    repeats = collections.Counter(_TRAILING.sub("", _norm_words(n)) for _, n, _ in cands)
    cands = [c for c in cands if repeats[_TRAILING.sub("", _norm_words(c[1]))] == 1]
    if not cands:
        return topics
    # The largest style used at least twice: the level just under the topic.
    # Not the most used one -- Ganita Prakash sets more run-in heads and game
    # titles (13pt black: "Matha Pachchi!", "Split and rejoin") than sections
    # (14pt blue: "Perimeter of a rectangle" .. "Perimeter of a regular
    # polygon", class6_ganita_prakash.pdf pp200-206).
    votes = collections.Counter((ln.size, ln.font, ln.color) for ln, _, _ in cands)
    recurring = [st for st in votes if votes[st] >= 2]
    style = (max(recurring, key=lambda st: (st[0], votes[st])) if recurring
             else max(votes, key=lambda st: (votes[st], st[0])))
    # a continuation line was also a candidate on its own: keep the first
    taken, subs = set(), []
    for ln, name, i in cands:
        if (ln.size, ln.font, ln.color) != style or i in taken:
            continue
        _, j = _with_continuations(lines, i, lambda t: False)
        taken.update(range(i, j + 1))
        subs.append((ln, Heading(2, name, ln.page, None, "numbered", line=i)))
    # interleave by position in reading order
    merged = sorted(topics + [h for _, h in subs], key=lambda h: h.line)
    return merged


def section_headings(lines: list[Line]) -> list[Heading]:
    """"1 Title" topics and "1.1 Title" subtopics numbered afresh per chapter.

    The topic line must be well above body size (jess302.pdf p2: 18pt against
    11.5pt), or a list item "1 Power is shared among ..." would count.
    """
    body, body_font = body_style(lines)
    out: list[Heading] = []
    topic = 0
    subs: set[int] = set()
    for i, ln in enumerate(lines):
        m = _SECTION.match(ln.text)
        if m and ln.bold and ln.size >= body + 3 and int(m.group(1)) == topic + 1:
            text, _ = _with_continuations(lines, i, lambda t: bool(_SECTION.match(t)))
            name = _clean(_SECTION.match(text).group(2))
            if _starts_like_a_title(name) and not _not_a_topic(name):
                topic += 1
                subs = set()
                out.append(Heading(1, name, ln.page, str(topic), "section"))
            continue
        m = _NUMBERED.match(ln.text)
        if (m and not m.group(3) and topic and int(m.group(1)) == topic
                and _set_apart(ln, body, body_font)):
            k = int(m.group(2))
            text, _ = _with_continuations(
                lines, i, lambda t: bool(_NUMBERED.match(t) or _SECTION.match(t)))
            name = _clean(_NUMBERED.match(text).group(4))
            if k in subs or (subs and k < max(subs)) or not _starts_like_a_title(name):
                continue
            if _not_a_topic(name):
                continue
            subs.add(k)
            out.append(Heading(2, name, ln.page, f"{topic}.{k}", "section"))
    return out


def _captions_a_table(head: Line, last: Line, page_lines: list[Line], body: float) -> bool:
    """``head`` (ending on ``last``) sits directly above the header row of a
    grid: three or more bold pieces of text size on one row within two lines
    below it, each separated by at least two body widths of space.

    Democratic Politics sets its table captions in the sub-section type,
    13pt orange bold, over "Activities | Men | Women" (jess403.pdf p4) and
    "Caste and Community groups | Rural | Urban" (p13), each cell bold. The
    gap is what tells a grid from a justified line spread into words ("of
    the  Class  Representative", 14pt apart, hees105.pdf p7); bold at text
    size is what tells it from a body line beside a margin note ("... winds.
    The" | "Monsoon:", class9 Understanding Society p73) and from the 8pt
    degree labels of a map under a heading (gees202.pdf p6 "India and
    Pakistan", "50°E  60°E  70°E").
    """
    below = sorted((ln for ln in page_lines
                    if last.y1 - 2 <= ln.y0 <= last.y1 + 2 * head.size and ln is not last),
                   key=lambda ln: ln.y0)
    rows: list[list[Line]] = []
    for ln in below:
        if rows and ln.y0 - rows[-1][0].y0 <= 1.5:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    for row in rows:
        row.sort(key=lambda ln: ln.x0)
        if (len(row) >= 3 and all(ln.bold and ln.size >= body - 1 for ln in row)
                and all(b.x0 - a.x1 >= 2 * body for a, b in zip(row, row[1:]))):
            return True
    return False


def unnumbered_headings(lines: list[Line], skip_pages: frozenset = frozenset()) -> list[Heading]:
    """Bold headings of a book that numbers nothing, in two size levels.

    Candidates are bold, not white (white type sits on a picture or a panel),
    at least 1.5pt above body size and below chapter-title size (24pt). The
    largest candidate size in the chapter is the topic level; the next size at
    least 0.9pt smaller is the subtopic level; anything smaller is ignored.

    A heading lettered as an item of a list -- "a) Temperature" .. "e)
    Humidity" under "Weather Instruments" (gees102.pdf pp5-11), "A. Primary
    activities" .. "C. Tertiary activities" (fees114.pdf pp2-7) -- is a
    subtopic of the heading above it, named without its letter. Read as a
    title, the lower-case letter dropped 31 such headings in the Social
    Science books of classes 6-9 (21 gees, 6 fees, 4 Understanding Society
    9). It is a subtopic whichever of the two levels it is set in: gees207.pdf
    p8 sets "b) Indian railway network" in the 17pt topic type between "a)"
    and "c)" at 15pt. That recovers 27 of the 31. The other 4 sit below the
    subtopic level: class9_understanding_society_part1.pdf pp103-107 sets
    "a. The Sumerians" .. "d. The Babylonians" in 13pt bold italic under the
    15pt "Mesopotamian Civilisation". The tree has no third level, so they
    come back as level 3, for the build to list as not represented; an
    unlettered heading at a third size is still ignored.
    """
    body, body_font = body_style(lines)
    family = re.split(r"[-,]", body_font)[0]
    by_page: dict[int, list[Line]] = collections.defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln)

    def acceptable(name: str) -> bool:
        return (sum(ch.isalpha() for ch in name) >= 3 and len(name) <= 110
                and not _not_a_topic(name) and not name.endswith((",", ";"))
                and not _dangling(name)
                and _starts_like_a_title(name) and not _foreign_script(name, lines))

    cands: list[tuple[Line, str, bool]] = []     # (line, name, lettered)
    prev_letter: str | None = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Bold, or -- where the PDF names the heading font "Regular" although
        # it prints heavy (fees105.pdf p3: "How Indians Named India", 17pt
        # NotoSerif-Regular in red on 13pt body) -- the text face, upright,
        # coloured and 3pt up. The text face, because a calendar label in a
        # display face ("Chaitra", Oswald, class8_curiosity.pdf p25) is not.
        heading_type = (ln.bold and ln.size >= body + 1.5) or (
            not _blackish(ln.color) and ln.size >= body + 3 and "italic" not in ln.font.lower()
            and re.split(r"[-,]", ln.font)[0] == family)
        if (ln.page not in skip_pages and heading_type and not _whitish(ln.color)
                and ln.size < 24 and not ln.text[:1].isdigit()):
            text, j = _with_continuations(lines, i, lambda t: False)
            name = _clean(text)
            label, rest = letter_label(name)
            lettered = (label is not None and _in_letter_run(label, prev_letter)
                        and acceptable(rest))
            if ((lettered or acceptable(name))
                    and not _captions_a_table(ln, lines[j], by_page[ln.page], body)):
                cands.append((ln, rest if lettered else name, lettered))
                if lettered:
                    prev_letter = label
            i = j + 1
            continue
        i += 1
    # Sizes within 0.6pt are one level: gees105.pdf sets the same sub-heading
    # style at 15pt and 15.43pt (pp6, 17).
    def two_levels(sizes: set[float]) -> tuple[set[float], set[float]]:
        levels: list[list[float]] = []
        for size in sorted(sizes, reverse=True):
            if levels and levels[-1][0] - size <= 0.6:
                levels[-1].append(size)
            else:
                levels.append([size])
        if not levels:
            return set(), set()
        second = (set(levels[1]) if len(levels) > 1 and levels[0][-1] - levels[1][0] >= 0.9
                  else set())
        return set(levels[0]), second

    # The two sizes are the book's unlettered headings'. A lettered heading is
    # a subtopic whatever its size, so its size says nothing about the levels:
    # counted in, a lettered run set between the topic and subtopic sizes
    # became the subtopic level and the real subtopics fell to a third level
    # and out of the tree; a run set above the topics demoted them. Only when
    # the chapter has no unlettered subtopics do the lettered sizes decide
    # where the subtopic level starts.
    top, second = two_levels({ln.size for ln, _, lettered in cands if not lettered})
    everything_top, everything_second = two_levels({ln.size for ln, _, _ in cands})
    if not top:
        top, second = everything_top, everything_second
    if not top:
        return []
    if second:
        sub_floor = min(second)
    elif everything_second and min(everything_second) < min(top):
        sub_floor = min(everything_second)
    else:
        sub_floor = min(top)
    out = []
    for ln, name, lettered in cands:
        if lettered and ln.size >= sub_floor:
            out.append(Heading(2, name, ln.page, None, "unnumbered"))
        elif lettered:
            out.append(Heading(3, name, ln.page, None, "unnumbered"))
        elif ln.size in top:
            out.append(Heading(1, name, ln.page, None, "unnumbered"))
        elif ln.size in second:
            out.append(Heading(2, name, ln.page, None, "unnumbered"))
    return out


@dataclass(frozen=True)
class HeadingStyle:
    """A book's own heading type, for a book the size ranking misreads.

    Understanding Economic Development (Class 10) sets its sections as 14pt
    white capitals on a coloured band and its sub-sections in 18pt dark type
    (jess203.pdf pp2, 4: "MONEY AS A MEDIUM OF EXCHANGE", "Cheque Payments"),
    so the larger size is the lower level and the topic level is white --
    both of which the automatic rule exists to refuse.
    """
    min_size: float
    max_size: float
    white: bool = False
    caps: bool = False


def _matches(ln: Line, st: HeadingStyle) -> bool:
    return (st.min_size <= ln.size <= st.max_size and _whitish(ln.color) == st.white
            and (not st.caps or ln.text.upper() == ln.text))


def _next_line_in_style(prev: Line, ln: Line) -> bool:
    """``ln`` is the next line of ``prev``'s title, however it is aligned."""
    return (ln.page == prev.page and ln.size == prev.size and ln.font == prev.font
            and ln.color == prev.color and 0 <= ln.y0 - prev.y0 <= 1.5 * ln.size)


def styled_headings(lines: list[Line], topic: HeadingStyle, sub: HeadingStyle | None = None,
                    skip_pages: frozenset = frozenset()) -> list[Heading]:
    """Headings in a declared type style (see `HeadingStyle`), in reading order."""
    out: list[Heading] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        level = (1 if _matches(ln, topic) else 2 if sub and _matches(ln, sub) else 0)
        if level and ln.page not in skip_pages and not ln.text[:1].isdigit():
            text, j = _with_continuations(lines, i, lambda t: False)
            name = _clean(text)
            if (sum(ch.isalpha() for ch in name) >= 3 and not _not_a_topic(name)
                    and _starts_like_a_title(name) and not _foreign_script(name, lines)):
                out.append(Heading(level, name, ln.page, None, "styled", line=i))
            elif j + 1 < len(lines) and _next_line_in_style(lines[j], lines[j + 1]):
                # the rest of a refused title: "Example 2: Exhaustion of" /
                # "Natural Resources", 18pt, the second line offset right
                # (jess201.pdf p14)
                j += 1
            i = j + 1
            continue
        i += 1
    return out


def chapter_headings(lines: list[Line], chapter: int, skip_pages: frozenset = frozenset(),
                     styles: tuple[HeadingStyle, HeadingStyle | None] | None = None
                     ) -> tuple[list[Heading], str]:
    """The best-supported heading scheme for one chapter, and its name.

    Numbered (the strongest signal) or per-chapter section numbers, then
    unnumbered bold headings; ``"none"`` when nothing qualifies.
    """
    lines = reading_order(lines)
    if styles:
        found = styled_headings(lines, styles[0], styles[1], skip_pages)
        return found, ("styled" if found else "none")
    found = numbered_headings(lines, chapter)
    sec = section_headings(lines)
    # History numbers sections afresh, so chapter 2's "2.1 The Movement in the
    # Towns" also reads as chapter-prefixed; the section scheme, which sees
    # the "1 ..", "2 .." topics too, finds more (jess302.pdf: 4+7 against 3).
    if sum(h.level == 1 for h in sec) >= 2 and len(sec) > len(found):
        return sec, "section"
    # One numbered topic is enough: after "Summary" is dropped, Class 10
    # Maths chapter 9 has only "9.1 Heights and Distances" (jemh109.pdf).
    if any(h.level == 1 for h in found):
        if not any(h.level == 2 for h in found):
            found = unnumbered_subtopics(lines, found)
        return found, "numbered"
    un = unnumbered_headings(lines, skip_pages)
    if un:
        return un, "unnumbered"
    return [], "none"


# --------------------------------------------------------------------------- #
# the tree
# --------------------------------------------------------------------------- #

def slug(text: str) -> str:
    """A stable, readable id segment: ASCII-folded, lower case, hyphenated.

    Devanagari is kept as it is (folding would erase it); everything else is
    reduced to letters and digits so the id does not change with the book's
    punctuation or a curly apostrophe.
    """
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch) or "ऀ" <= ch <= "ॿ")
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[^\wऀ-ॿ]+", "-", t).strip("-_")
    return t[:60].rstrip("-") or "untitled"


def _unique(base: str, used: set[str]) -> str:
    sid, n = base, 2
    while sid in used:
        sid, n = f"{base}-{n}", n + 1
    used.add(sid)
    return sid


def build_topics(chapter_id: str, headings: list[Heading],
                 below: list[dict] | None = None) -> tuple[list[dict], int]:
    """Nest subtopics under the topic before them. Returns (topics, orphans).

    An orphan is a subtopic with no topic before it in the chapter, or whose
    number names another topic; it is counted and dropped rather than hung
    under a guessed parent. A level-3 heading (below the subtopic level) is
    not part of the tree: it is appended to ``below``, when given, as
    {name, page, under} -- ``under`` the id of the topic or subtopic placed
    last before it (None if none) -- so the caller can list it.
    """
    topics: list[dict] = []
    used_t: set[str] = set()
    used_s: dict[str, set] = {}
    orphans = 0
    last: str | None = None
    for h in headings:
        if h.level > 2:
            if below is not None:
                below.append({"name": h.name, "page": h.page, "under": last})
            continue
        if h.level == 1:
            tid = _unique(f"{chapter_id}/{slug(h.name)}", used_t)
            t = {"id": tid, "name": h.name, "page": h.page}
            if h.number:
                t["number"] = h.number
            t["subtopics"] = []
            used_s[tid] = set()
            topics.append(t)
            last = tid
            continue
        if not topics or (h.number and topics[-1].get("number")
                          and not h.number.startswith(topics[-1]["number"] + ".")):
            orphans += 1
            continue
        t = topics[-1]
        s = {"id": _unique(f'{t["id"]}/{slug(h.name)}', used_s[t["id"]]),
             "name": h.name, "page": h.page}
        if h.number:
            s["number"] = h.number
        t["subtopics"].append(s)
        last = s["id"]
    return topics, orphans


# --------------------------------------------------------------------------- #
# chapters of a whole book, from its contents page
# --------------------------------------------------------------------------- #

@dataclass
class ContentsEntry:
    title: str
    printed_page: int
    number: int | None = None      # "Chapter N" when the contents says so
    title_parts: list[str] = field(default_factory=list)


_CHAPTER_MARK = re.compile(r"^chapter\s+(\d{1,2})\b\s*(.*)$", re.I)
_UNIT_MARK = re.compile(r"^unit\s+\d+\b", re.I)
_ARABIC = re.compile(r"^\d{1,3}$")
_ROMAN = re.compile(r"^[ivxlc]+$", re.I)


def _rows(lines: list[Line]) -> list[list[Line]]:
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.y1, l.x0)):
        if rows and abs(rows[-1][0].y1 - ln.y1) <= 3.5:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [sorted(r, key=lambda l: l.x0) for r in rows]


def parse_contents(lines: list[Line], pages: list[int]) -> list[ContentsEntry]:
    """Entries of a printed contents page: title, printed start page, number.

    A row that ends in an arabic page number closes an entry; "Chapter N"
    opens one. A plain row before the closing row is the start of a wrapped
    title (Poorvi 8: "A Tale of Valour: Major Somnath Sharma" / "and the
    Battle of Badgam  49"); a plain row in lower case straight after a closed
    entry is the end of one (Ganita Manjari: "Predicting What Comes Next:
    Exploring Sequences  174" / "and Progressions"). Bold rows with no page
    number head the contents' own units ("Unit 2: Wit and Humour",
    "गद्य खंड"). Only the leftmost text of a row is title: the class 9 Hindi
    contents prints the author in a second column.
    """
    entries: list[ContentsEntry] = []
    open_: ContentsEntry | None = None
    pending: list[str] = []
    for p in pages:
        for row in _rows([ln for ln in lines if ln.page == p]):
            texts = [_clean(ln.text) for ln in row]
            row = [ln for ln, t in zip(row, texts) if t]
            texts = [t for t in texts if t]
            if not texts or (len(texts) == 1 and (
                    texts[0].lower().replace(" ", "") in ("contents", "content")
                    or _ARABIC.match(texts[0]) or _ROMAN.match(texts[0]))):
                continue
            num = None
            if len(row) > 1 and _ARABIC.match(texts[-1]):
                num = int(texts[-1])
                row, texts = row[:-1], texts[:-1]
            elif len(row) > 1 and _ROMAN.match(texts[-1]):
                pending = []
                continue            # front matter: Foreword, About the Book
            m = _CHAPTER_MARK.match(texts[0])
            if m:
                open_ = ContentsEntry("", 0, int(m.group(1)))
                pending = []
                rest = " ".join(t for t in [m.group(2)] + texts[1:] if t)
                if rest:
                    open_.title_parts.append(rest)
                if num is not None:
                    open_.printed_page = num
                    entries.append(open_)
                    open_ = None
                continue
            first = texts[0]
            if num is None:
                if row[0].bold or _UNIT_MARK.match(first):
                    pending = []
                elif open_ is not None:
                    open_.title_parts.append(first)
                elif entries and not pending and first[:1].islower():
                    entries[-1].title_parts.append(first)
                else:
                    pending.append(first)
                continue
            if open_ is not None:
                open_.title_parts.append(first)
                open_.printed_page = num
                entries.append(open_)
                open_ = None
            else:
                entries.append(ContentsEntry("", num, None, pending + [first]))
            pending = []
    for e in entries:
        e.title = _clean(" ".join(e.title_parts))
    return entries


def _letters(text: str) -> str:
    return re.sub(r"[^a-zऀ-ॿ]", "", unicodedata.normalize("NFKD", text).lower())


def printed_page_numbers(lines: list[Line]) -> dict[int, int]:
    """Printed page number -> PDF page, from the folios in the page margins.

    A constant offset does not hold: Ganita Prakash 6 puts five unnumbered
    pages between chapters 1 and 2 (printed 12 is PDF 32, printed 14 is PDF
    39). A folio is a line of digits only, within 70pt of the top or bottom
    edge, no larger than body type; a number claimed by two pages is dropped.
    """
    body, _ = body_style(lines)
    bottom = max((ln.y1 for ln in lines), default=0)
    seen: dict[int, set] = collections.defaultdict(set)
    for ln in lines:
        t = ln.text.strip()
        if (_ARABIC.match(t) and ln.size <= body + 0.5
                and (ln.y0 >= bottom - 70 or ln.y1 <= 70)):
            seen[int(t)].add(ln.page)
    return {n: next(iter(ps)) for n, ps in seen.items() if len(ps) == 1}


def locate_chapter_starts(lines: list[Line], entries: list[ContentsEntry],
                          after_page: int) -> list[tuple[int | None, str]]:
    """The PDF page each entry starts on, and how it was found.

    ``title``: the first page after the previous chapter's start whose
    display type (>= body + 2.5pt) contains the first 14 letters of the
    entry's title. ``folio``: failing that, the page printed one before the
    entry's start page, plus one. ``None`` when neither is there.
    """
    body, _ = body_style(lines)
    display: dict[int, str] = collections.defaultdict(str)
    for ln in reading_order(lines):
        if ln.page > after_page and ln.size >= body + 2.5:
            display[ln.page] += _letters(ln.text)
    folios = printed_page_numbers(lines)
    out: list[tuple[int | None, str]] = []
    prev = after_page
    for e in entries:
        key = _letters(e.title)[:14]
        found = None
        if len(key) >= 4:
            found = next((p for p in sorted(display) if p > prev and key in display[p]), None)
        if found is not None:
            out.append((found, "title"))
        elif e.printed_page - 1 in folios and folios[e.printed_page - 1] + 1 > prev:
            found = folios[e.printed_page - 1] + 1
            out.append((found, "folio"))
        else:
            out.append((None, "not-found"))
        if found is not None:
            prev = found
    return out


def title_case(text: str) -> str:
    """Display type in capitals ("GEOMETRIC TWINS") as the contents prints it."""
    if not text.isupper():
        return text
    small = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}

    def word(i: int, w: str) -> str:
        if len(w) > 1 and not re.search(r"[AEIOUY]", w):
            return w                       # an acronym: GDP, not Gdp
        w = w.lower()
        return w if (i and w in small) else w[:1].upper() + w[1:]
    return " ".join(word(i, w) for i, w in enumerate(text.split()))


def display_title(lines: list[Line], page: int) -> str:
    """A chapter-PDF's title: its display lines (>= 24pt) on the opening page.

    Numerals, the word "CHAPTER"/"Chapter N"/"Chapter I" and single letters
    (drop caps) are not title. White type is: Ganita Prakash 8 part 2 sets
    its titles white on a picture (hegp201.pdf p1).
    """
    rows = []
    for ln in lines:
        t = ln.text.strip()
        if (ln.page == page and ln.size >= 24 and len(t) > 1
                and not re.fullmatch(r"[\d\W]+", t)
                and not re.match(r"^chapter(\s+(\d+|[ivxlc]+))?$", t, re.I)):
            rows.append(ln)
    rows.sort(key=lambda ln: (ln.y0, ln.x0))
    # Lines in one size are one phrase; a change of size is a kicker and its
    # title: "Grassroots Democracy – Part 1" (24pt) over "Governance" (26pt),
    # fees110.pdf p1, reads "Grassroots Democracy – Part 1: Governance".
    seen: set[str] = set()
    groups: list[tuple[float, list[str]]] = []
    for ln in rows:
        if ln.text in seen:
            continue
        seen.add(ln.text)
        if groups and abs(groups[-1][0] - ln.size) <= 0.6:
            groups[-1][1].append(ln.text)
        else:
            groups.append((ln.size, [ln.text]))
    # a line broken after a hyphen joins without a space ("Baudhāyana-" /
    # "Pythagoras Theorem", hegp202.pdf p1)
    return ": ".join(title_case(_clean(" ".join(parts).replace("- ", "-")))
                     for _, parts in groups)


# =========================================================================== #
# SECOND EXTRACTOR -- straight from the PDFs (feat/product-calendar)
# =========================================================================== #
#
# Deterministic extraction of an NCERT textbook's own chapter and section
# headings, straight from the PDFs with PyMuPDF.
#
# This is source (b) of the decomposition templates (see
# `scripts/export_decomposition_templates.py`). The syllabus JSONs stop at
# chapter level, so a freshly seeded school has nothing below Chapter; the
# textbooks themselves carry the missing level as printed headings -- "2.2 How to
# Group Plants and Animals?", "2.2.1 How to group plants?" -- and those are the
# book's own words, not a model's guess at them.
#
# Why font metrics and not a regex over plain text: the same numeric shape occurs
# constantly in body prose. Measured on class6_curiosity.pdf 2026-09-23, matching
# `^\d{1,2}(\.\d{1,2}){1,2}\s+\S` over the raw page lines returns 105 lines, of
# which 33 carry a number that is no section of that book ("2.6 that the plants
# and animals", "35.0 g", "10.4 cm-1.0 cm = 9.4 cm", "37.0 degC") and the rest
# repeat the real headings. Every printed heading in that book is set in a BOLD
# face at a size above the body size and no body line is, so adding "bold and
# larger than body text" to the numbering rule takes the same book to 71
# headings, each of which `tests/test_textbook_headings.py` finds printed
# verbatim on the page this module reports, with its number immediately before
# it.
#
# Two heading styles, because NCERT prints two:
#
#   * NUMBERED (Curiosity, Ganita Prakash, Exploration): "2.2", "2.2.1". The
#     number gives the nesting, so nothing has to be inferred.
#   * UNNUMBERED (the Class 6-8 Social Science books, which ship as one PDF per
#     chapter): headings are distinguished only by their face and size. Those are
#     read as SIZE TIERS, and only for a book that prints no numbered heading at
#     all -- see `_tiers` for the two rules that keep a recurring box label
#     ("LET'S EXPLORE", "Questions") from being mistaken for a section.
#
# Deliberately no fuzzier fallback. A heading list scraped from the Contents page
# or a similarity match on titles would put text under a heading it does not
# belong to, which is worse for the principal reviewing the proposal than a
# chapter that honestly reports having no breakdown.
#
# Nothing here writes anything: it reads read-only PDFs and returns a tree.
# `scripts/export_decomposition_templates.py` maps that tree onto syllabus
# chapters and commits it.
#
#     from academicos.syllabus.textbook_headings import extract_chapters
#     for ch in extract_chapters(Path(".../textbooks/science/class6_curiosity.pdf")):
#         print(ch.number, ch.name, [t.number for t in ch.topics])

# PyMuPDF's span flag bit for a bold face. Some NCERT titles are set in faces
# whose flag is not set but whose name says what they are ("NotoSans-SemiBold",
# "NotoSerif-ExtraBold"), so the name is checked too.
_FLAG_BOLD = 1 << 4
_SECTION_BOLD_NAMES = ("bold", "black", "heavy", "semib", "extrab")

# "2.1", "2.2.1" -- at most three levels, because NCERT never prints a fourth
# and a longer run of dotted numbers in a textbook is a figure reference or a
# measurement, not a heading.
_SECTION_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})[\s \t.:)]+(\S.*)$")

# A chapter opener prints its number alone at display size beside the title.
_PURE_INT = re.compile(r"^\d{1,2}$")

# Both ratios are floors, not descriptions of a typical book, and both were too
# high until 2026-09-23. Measured across the books this repo actually reads:
# Curiosity 6 prints its numeral at 5.5x the body size, Exploration 9 at 5.9x,
# the Social Science chapter PDFs at 5.5x -- but Ganita Manjari 9 prints its at
# 22.6pt against an 11.0pt body (2.05x) and its titles at 16.0pt (1.45x). At
# 2.5x/1.5x that book yielded ONE chapter, its front cover, and none of its
# eight real chapters; Poorvi 6, Ganga 9 and Understanding Society 9 yielded
# nothing at all. The floors exist to keep a bold run-in from being read as a
# chapter opener, and 1.8x/1.4x still sits far above every heading size in
# these books; the opener rules below (one lone numeral on the page, a title in
# a single size class, nothing before chapter 1) are what actually does the
# separating.
_MIN_CHAPTER_NUMERAL_RATIO = 1.8   # x body size
_MIN_TITLE_RATIO = 1.4             # x body size
# A chapter title may be set in two sizes ("Health:" at 26pt over "The Ultimate
# Treasure" at 24pt, Curiosity 8) but not in three unrelated ones. Anything
# printed below this share of the largest title line on the opener page is
# furniture, not part of the name: Exploration 9's literal word "Chapter"
# (21.2pt against a 26pt title, 0.82) and the Social Science books' "Questions"
# box label (18pt against 30pt, 0.60) were both being folded into the title.
_MIN_HEADING_RATIO = 1.05          # x body size; headings sit just above body
_TITLE_SIZE_BAND = 0.9             # x the largest title line on the page
# A numbered line that is NOT bold is a heading only if it is set well clear of
# the body. Ganita Manjari 9 is the one book here that sets its sections
# unbolded -- "3.5 Irrational Numbers" is NotoSerif-Regular at 14pt over an
# 11pt body, while its SUBsections are NotoSerif-Bold at 12pt -- so the
# bold-only rule read its 27 subsections and none of its 53 sections, leaving
# every chapter with zero topics. Measured across the other five numbered books
# (Curiosity 6/7/8, Exploration 9, Ganita Prakash 6): not one line in any of
# them matches the numbering, clears the body size and is unbolded, so this
# branch admits nothing anywhere else.
_MIN_UNBOLD_HEADING_RATIO = 1.2    # x body size

# Unnumbered mode only (see `_tiers`).
_MIN_TIER_DISTINCT = 0.5           # share of occurrences that must be unique
_MIN_TIER_CHAPTER_SHARE = 0.25     # share of chapters the tier must appear in
_MIN_TIER_OCCURRENCES = 5          # a face used three times is a flourish


def _clean_run(text: str) -> str:
    """Collapse whitespace and drop the control codepoints the NCERT PDFs carry
    inside a heading's text run: the bullet glyph in "6.3.1 Observe and
    identify..." extracts as U+0007, which would otherwise end up stored in a
    topic name and shown to a principal."""
    return " ".join("".join(
        " " if unicodedata.category(c) in ("Cc", "Cf") else c for c in text).split())


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    font: str
    bold: bool
    file: str
    page: int
    block: int
    # Position in the PDF's own content stream (file, then page, block, line).
    # Chapter titles are ordered by this and not by bbox: where the display
    # numeral sits inside the title's line, that line's bbox is stretched to
    # the numeral's full height and would sort above the line printed above it.
    index: int = 0
    # (text, size) per span, kept because a chapter opener sometimes sets the
    # display numeral as one span INSIDE the title line ("Storms, and
    # Cyclones 6" in class8_curiosity) rather than on a line of its own.
    spans: tuple[tuple[str, float], ...] = ()

    @property
    def where(self) -> tuple[str, int]:
        return (self.file, self.page)

    def without(self, display_size: float) -> str:
        """This line's text with any span set at `display_size` removed -- the
        chapter numeral, when it shares the title's line."""
        return _clean_run("".join(t for t, s in self.spans if s != display_size))

    def size_without(self, display_size: float) -> float:
        """The size this line is really set in once the display numeral's spans
        are taken out. `size` is the LEAD span's size, and Poorvi's "Unit 1"
        drop-caps the U at the numeral's own 27.1pt, so the line reports 27.1
        while the words on it are 19pt -- enough to make a page's furniture
        outrank its title."""
        rest = [s for t, s in self.spans if s != display_size and t.strip()]
        return max(rest) if rest else self.size


@dataclass(frozen=True)
class SectionHeading:
    """One printed section heading, with the file and page it was printed on so
    a reviewer can go and look at it. `number` is None for a book that does not
    number its sections."""
    title: str
    page: int                      # 0-based page index, as PyMuPDF counts them
    source_file: str
    number: Optional[str] = None

    @property
    def chapter_number(self) -> Optional[int]:
        return int(self.number.split(".")[0]) if self.number else None

    @property
    def level(self) -> int:
        """1 for "2.2" (a topic), 2 for "2.2.1" (a subtopic)."""
        return self.number.count(".") if self.number else 1

    @property
    def parent_number(self) -> Optional[str]:
        if self.number and self.number.count(".") > 1:
            return self.number.rsplit(".", 1)[0]
        return None


@dataclass(frozen=True)
class ExtractedTopic:
    name: str
    page: int
    number: Optional[str] = None
    subtopics: tuple[SectionHeading, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExtractedChapter:
    number: int
    name: str
    page: int
    source_file: str
    topics: tuple[ExtractedTopic, ...] = field(default_factory=tuple)


# ---------------- reading the PDFs ----------------

def _open(pdf_path: Path):
    try:
        import pymupdf                      # PyMuPDF >= 1.24 name
    except ImportError:                     # pragma: no cover - older wheels
        import fitz as pymupdf
    return pymupdf.open(pdf_path)


def pdf_files(source: Path) -> list[Path]:
    """One PDF, or every PDF in a directory in filename order -- NCERT ships
    the Social Science books as one file per chapter (`fees101.pdf` ...), and
    the whole directory is one book."""
    if source.is_dir():
        return sorted(p for p in source.glob("*.pdf"))
    return [source]


def _lines(paths: Sequence[Path], root: Optional[Path] = None) -> list[_Line]:
    """Every text line across the book, with the font facts the heading rules
    need. Continuation lines are NOT merged here -- `_merge_wrapped` does that,
    because it needs to know which line started a heading."""
    out: list[_Line] = []
    for path in paths:
        name = path.relative_to(root).as_posix() if root else path.name
        doc = _open(path)
        try:
            for page_no in range(doc.page_count):
                blocks = doc[page_no].get_text("dict")["blocks"]
                for block_no, block in enumerate(blocks):
                    if block.get("type") != 0:      # 0 = text, 1 = image
                        continue
                    for line in block["lines"]:
                        spans = [s for s in line["spans"] if s["text"].strip()]
                        if not spans:
                            continue
                        text = _clean_run("".join(s["text"] for s in spans))
                        if not text:
                            continue
                        lead = spans[0]
                        font = lead["font"]
                        out.append(_Line(
                            text=text, size=round(lead["size"], 1), font=font,
                            bold=bool(lead["flags"] & _FLAG_BOLD)
                                 or any(n in font.lower() for n in _SECTION_BOLD_NAMES),
                            file=name, page=page_no, block=block_no, index=len(out),
                            spans=tuple((s["text"], round(s["size"], 1)) for s in spans)))
        finally:
            doc.close()
    return out


def body_size(lines: Iterable[_Line]) -> float:
    """The book's running-text size: the size that the most CHARACTERS are set
    in. Counting lines instead would be swayed by the many short lines in
    captions and margin notes; counting characters lands on the prose."""
    weight: dict[float, int] = {}
    for ln in lines:
        weight[ln.size] = weight.get(ln.size, 0) + len(ln.text)
    return max(weight, key=lambda s: weight[s]) if weight else 0.0


# ---------------- chapter openers ----------------

def _opener_title(page_lines: list[_Line], numeral_size: float,
                  base: float) -> Optional[tuple[str, _Line]]:
    """The chapter title printed on one opener page, or None.

    Size only, no bold test: Ganita Prakash and Poorvi set their chapter titles
    in display faces (LuckiestGuy) that PyMuPDF reports as neither bold-flagged
    nor bold-named, and requiring bold found no chapter at all in either book.
    On a page that already prints a lone display numeral, size is the signal --
    but it has to be read as a CLASS, not as a threshold, because an opener page
    carries three kinds of large text:

      * the title, in one size or two adjacent ones;
      * page furniture at a distinctly smaller size -- the word "Chapter"
        (Exploration 9), "CHAPTER" (Social Science), "Probe and ponder"
        (Curiosity 8);
      * unrelated large text elsewhere on the page -- the "Questions" box
        heading, Poorvi's unit theme printed a dozen blocks below the lesson
        title.

    So: take every line that has words at title size, keep the ones within
    `_TITLE_SIZE_BAND` of the largest (which drops the furniture), and then keep
    only the run of CONSECUTIVE lines that contains that largest one (which
    drops the unrelated text further down the page). Both halves are needed --
    Curiosity 7 splits one title across two blocks at 30pt and 28pt, so a
    single-size or single-block rule truncates it to "Basic, and Neutral".
    """
    cands = []
    for ln in sorted(page_lines, key=lambda l: l.index):
        text = ln.without(numeral_size)
        if sum(c.isalpha() for c in text) < 3:
            continue
        size = ln.size_without(numeral_size)
        if size >= base * _MIN_TITLE_RATIO:
            cands.append((ln, text, size))
    if not cands:
        return None

    top = max(size for _, _, size in cands)
    runs: list[list[tuple[_Line, str, float]]] = []
    for cand in cands:
        if cand[2] < top * _TITLE_SIZE_BAND:
            continue
        if runs and cand[0].index == runs[-1][-1][0].index + 1:
            runs[-1].append(cand)
        else:
            runs.append([cand])
    run = next(r for r in runs if any(size == top for _, _, size in r))
    return " ".join(" ".join(text for _, text, _ in run).split()), run[0][0]


def _chapter_openers(lines: list[_Line]) -> tuple[dict[int, _Line], dict[int, str]]:
    """({chapter number: the line its title starts on}, {chapter number: title}).

    An NCERT chapter opens on a page that prints its number alone at display
    size (72pt against 13pt body in Curiosity) beside the chapter title. Both
    facts are required: front matter ("Foreword", "Contents") is set at title
    size too but never carries a lone display numeral, and the big decorative
    "?" that opens several spreads carries no letters.
    """
    base = body_size(lines)
    by_page: dict[tuple[str, int], list[_Line]] = {}
    for ln in lines:
        by_page.setdefault(ln.where, []).append(ln)

    found: list[tuple[int, str, _Line]] = []
    for page_lines in sorted(by_page.values(), key=lambda g: g[0].index):
        # The numeral is looked for per SPAN, not per line: class8_curiosity
        # sets "6" inside the title's own line, class6_curiosity gives it a line
        # of its own, and both are the same chapter opener.
        numerals = [(ln, text.strip(), size) for ln in page_lines
                    for text, size in ln.spans
                    if _PURE_INT.match(text.strip())
                    and size >= base * _MIN_CHAPTER_NUMERAL_RATIO]
        if len(numerals) != 1:
            continue
        _, numeral_text, numeral_size = numerals[0]
        title = _opener_title(page_lines, numeral_size, base)
        if title is None:
            continue
        found.append((int(numeral_text), title[0], title[1]))

    found = _drop_front_matter(found)
    out: dict[int, tuple[str, _Line]] = {}
    for number, title, line in found:
        # A number printed twice at display size (a part divider reusing it)
        # keeps its first, i.e. opening, page.
        out.setdefault(number, (title, line))
    return ({n: ln for n, (_, ln) in out.items()},
            {n: t for n, (t, _) in out.items()})


def _drop_front_matter(found: list[tuple[int, str, _Line]]) -> list[tuple[int, str, _Line]]:
    """Everything printed before the chapter numbered 1 is front matter.

    Ganita Manjari 9's cover sets the GRADE -- "9" -- alone at 43.8pt beside
    "GANITA MANJARI", which is a lone display numeral next to a title and so is
    indistinguishable from a chapter opener by the page's own metrics. It is
    distinguishable by position: it is printed 20 pages before chapter 1. Books
    whose chapter 1 is never recognised keep every candidate, because then
    there is no evidence to cut on.
    """
    first = next((i for i, (number, _, _) in enumerate(found) if number == 1), None)
    return found[first:] if first is not None else found


# ---------------- numbered headings ----------------

def _merge_wrapped(lines: list[_Line], start: int) -> tuple[str, int]:
    """A printed heading that runs over two lines ("2.3 Plants and Animals in
    Different" / "Surroundings") is one heading. PyMuPDF keeps both lines in the
    same block at the same size and font, so the continuation rule is exactly
    that -- same block, same size, same font, and not itself a new heading."""
    head = lines[start]
    parts = [head.text]
    i = start + 1
    while (i < len(lines) and lines[i].where == head.where
           and lines[i].block == head.block and lines[i].size == head.size
           and lines[i].font == head.font
           and not _SECTION_NUMBERED.match(lines[i].text)):
        parts.append(lines[i].text)
        i += 1
    return " ".join(" ".join(parts).split()), i


def _numbered_section_headings(lines: list[_Line]) -> list[SectionHeading]:
    """Every numbered section heading the book prints, in reading order.

    The rule, and only this rule: the line starts with a dotted section number
    and is set apart from the body text -- bold and a touch larger, or, for a
    book that does not bold its sections, plainly larger (see
    `_MIN_UNBOLD_HEADING_RATIO`).
    """
    base = body_size(lines)
    cutoff = base * _MIN_HEADING_RATIO
    unbold_cutoff = base * _MIN_UNBOLD_HEADING_RATIO
    out: list[SectionHeading] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _SECTION_NUMBERED.match(ln.text)
        apart = ln.size > cutoff if ln.bold else ln.size >= unbold_cutoff
        if not (m and apart):
            i += 1
            continue
        text, nxt = _merge_wrapped(lines, i)
        m = _SECTION_NUMBERED.match(text) or m
        number, title = m.group(1), m.group(2).strip()
        # The same number printed twice is the chapter's own recap list, not a
        # second section. First occurrence wins -- that is the one on the page
        # where the section actually begins.
        if number not in seen and title:
            seen.add(number)
            out.append(SectionHeading(number=number, title=title, page=ln.page,
                               source_file=ln.file))
        i = nxt
    return out


# ---------------- unnumbered headings (size tiers) ----------------

def _tiers(lines: list[_Line], openers: dict[int, _Line]) -> list[tuple[str, float]]:
    """The one or two (font, size) classes that a book WITHOUT numbered
    sections uses for its headings, largest first.

    Three rules separate a heading class from the furniture that a textbook
    also sets bold and large, measured on the Class 6 Social Science book:

      * a heading is written once. A class whose lines repeat the same words
        over and over is a box label -- "LET'S EXPLORE" (117 lines, 3 distinct
        strings), "Questions" (13 lines, 1 string), "CHAPTER".
      * a heading class is used again and again. The decorative pull-quote
        words that open a spread ("Ocean", "Life", 27pt) are unique strings but
        appear three times in the whole book.
      * a heading class is used across chapters, not inside one.

    The classes are counted from the pages that are NOT chapter openers: an
    opener prints the chapter title, its subtitle and the word "CHAPTER" in the
    title face, and those are exactly the lines a size rule cannot tell from a
    section heading. Headings themselves are still collected from every page,
    because a chapter that starts on its opener page prints its first heading
    there ("Family", in fees109.pdf).
    """
    base = body_size(lines)
    opener_pages = {ln.where for ln in openers.values()}
    title_size = max((ln.size for ln in openers.values()), default=float("inf"))
    chapters = {ln.file for ln in openers.values()} or {ln.file for ln in lines}

    buckets: dict[tuple[str, float], list[_Line]] = {}
    for ln in lines:
        if ln.bold and base < ln.size < title_size and ln.where not in opener_pages:
            buckets.setdefault((ln.font, ln.size), []).append(ln)

    kept: list[tuple[str, float]] = []
    for key, group in buckets.items():
        if len(group) < _MIN_TIER_OCCURRENCES:
            continue
        distinct = len({ln.text.lower() for ln in group}) / len(group)
        share = len({ln.file for ln in group}) / len(chapters)
        if distinct >= _MIN_TIER_DISTINCT and share >= _MIN_TIER_CHAPTER_SHARE:
            kept.append(key)
    kept.sort(key=lambda k: k[1], reverse=True)
    return kept[:2]


def _tiered_section_headings(lines: list[_Line],
                     openers: dict[int, _Line]) -> list[tuple[int, SectionHeading]]:
    """(level, heading) in reading order, where level 1 is a topic and level 2
    a subtopic. An unnumbered book carries no number to nest by, so the tier's
    rank IS the level."""
    tiers = _tiers(lines, openers)
    if not tiers:
        return []
    levels = {key: i + 1 for i, key in enumerate(tiers)}
    opener_indexes = {o.index for o in openers.values()}
    out: list[tuple[int, SectionHeading]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        level = levels.get((ln.font, ln.size))
        if level is None or ln.index in opener_indexes:
            i += 1
            continue
        # Same continuation rule as the numbered path.
        parts = [ln.text]
        j = i + 1
        while (j < len(lines) and lines[j].where == ln.where
               and lines[j].block == ln.block and lines[j].size == ln.size
               and lines[j].font == ln.font):
            parts.append(lines[j].text)
            j += 1
        out.append((level, SectionHeading(number=None, title=" ".join(" ".join(parts).split()),
                                   page=ln.page, source_file=ln.file)))
        i = j
    return out


# ---------------- public API ----------------

def extract_headings(source: Path, root: Optional[Path] = None) -> list[SectionHeading]:
    """Every section heading the book prints, in reading order. `source` is one
    PDF or a directory of one-chapter PDFs."""
    lines = _lines(pdf_files(source), root)
    numbered = _numbered_section_headings(lines)
    if numbered:
        return numbered
    openers, _ = _chapter_openers(lines)
    return [h for _, h in _tiered_section_headings(lines, openers)]


def extract_chapters(source: Path, root: Optional[Path] = None) -> list[ExtractedChapter]:
    """The book as Chapter -> Topic -> Subtopic, from its own printed headings.

    A chapter the book opens but prints no heading inside (the literature
    readers, Curiosity's own chapters 1 and 9) comes back with an empty
    `topics` rather than being dropped: "this chapter has no sub-structure to
    extract" and "this chapter was missed" are different answers, and the
    export's manifest reports them differently.
    """
    paths = pdf_files(source)
    lines = _lines(paths, root)
    opener_lines, opener_titles = _chapter_openers(lines)

    numbered = _numbered_section_headings(lines)
    if numbered:
        return _nest_numbered(numbered, opener_lines, opener_titles)
    return _nest_tiered(_tiered_section_headings(lines, opener_lines), opener_lines, opener_titles)


def _chapter_order(openers: dict[int, _Line]) -> list[int]:
    return [n for n, _ in sorted(openers.items(), key=lambda kv: kv[1].index)]


def _nest_numbered(headings: list[SectionHeading], openers: dict[int, _Line],
                   titles: dict[int, str]) -> list[ExtractedChapter]:
    by_chapter: dict[int, list[SectionHeading]] = {}
    for h in headings:
        # A heading whose chapter the book never opened is not attached to a
        # guess; it is dropped, and the count difference is visible in the
        # export's report.
        if h.chapter_number in openers:
            by_chapter.setdefault(h.chapter_number, []).append(h)

    out: list[ExtractedChapter] = []
    for number in _chapter_order(openers):
        subs: dict[str, list[SectionHeading]] = {}
        for h in by_chapter.get(number, []):
            if h.parent_number:
                subs.setdefault(h.parent_number, []).append(h)
        topics = tuple(
            ExtractedTopic(number=h.number, name=h.title, page=h.page,
                           subtopics=tuple(subs.get(h.number or "", ())))
            for h in by_chapter.get(number, []) if h.level == 1)
        line = openers[number]
        out.append(ExtractedChapter(number=number, name=titles[number], page=line.page,
                                    source_file=line.file, topics=topics))
    return out


def _nest_tiered(headings: list[tuple[int, SectionHeading]], openers: dict[int, _Line],
                 titles: dict[int, str]) -> list[ExtractedChapter]:
    """Unnumbered books carry no chapter number in the heading, so a heading
    belongs to the chapter whose file it is printed in -- NCERT ships these
    books as one PDF per chapter, which is why the unnumbered path only has to
    handle that shape."""
    by_file: dict[str, int] = {ln.file: n for n, ln in openers.items()}
    grouped: dict[int, list[tuple[int, SectionHeading]]] = {}
    for level, h in headings:
        number = by_file.get(h.source_file)
        if number is not None:
            grouped.setdefault(number, []).append((level, h))

    out: list[ExtractedChapter] = []
    for number in _chapter_order(openers):
        topics: list[ExtractedTopic] = []
        for level, h in grouped.get(number, []):
            if level == 1:
                topics.append(ExtractedTopic(number=None, name=h.title, page=h.page))
            elif topics:      # a subtopic before any topic has no parent: dropped
                last = topics[-1]
                topics[-1] = ExtractedTopic(
                    number=None, name=last.name, page=last.page,
                    subtopics=last.subtopics + (h,))
        line = openers[number]
        out.append(ExtractedChapter(number=number, name=titles[number], page=line.page,
                                    source_file=line.file, topics=tuple(topics)))
    return out
