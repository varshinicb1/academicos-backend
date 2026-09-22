"""Parse a CBSE Sample Question Paper and its Marking Scheme, and join them.

SQP and MS are separate documents numbered the same way, so the link between a
question and its answer is authority-provided. That is what the board-paper
corpus could not give us: there, 1,658 questions had a matching paper code and
no answer for their question number, and the offset that would have joined them
proved unmeasurable.

Traps characterised against the real Class X Social Science pair:

1. MARKS ARE A LONE DIGIT ON THEIR OWN LINE, after the question text. There is
   no "N marks" string anywhere in the document -- measured at zero occurrences
   -- so marks cannot be read from prose and must be read positionally.

2. `OR` MEANS AN INTERNAL CHOICE. `17A` and `17B` are not two questions; they
   are one question with two options. Counting rows naively inflates the paper,
   so B is recorded as an alternative of A rather than as a sibling. Stripped
   on both sides now (2026-09-18) -- the scheme-side parser used to leave the
   literal word "OR" sitting inside the preceding alternative's answer text.

3. PAGE NUMBERS IN THE MARGIN OR HEADER/FOOTER BAND ARE FURNITURE, not
   content, and are shaped exactly like a bare mark value or a question
   header missing its trailing letter. Removed by requiring BOTH the value to
   equal the page index AND a furniture position (margin or band) -- value
   alone would delete a real mark that happens to equal the page index (a
   3-mark question on page 3); position alone let centred page numbers
   through (they were never in the horizontal margin), which is how a real
   production defect once turned page 10's number into a 10-mark question.
   (An earlier version of this docstring blamed a stray "17." on page-number
   bleed; it doesn't need one -- see point 5 below for what it actually was.)

4. SECTION GATING MUST TOLERATE THE HEADER SITTING INSIDE A TABLE ROW.
   Everything before the paper's first section header is General Instructions,
   numbered 1-N -- the same shape as a question header -- so nothing before it
   may be read as a question. But Section A's own header does not print as a
   bare "SECTION A" line in this paper: it prints as a table row, "Sr.No
   SECTION A Marks", sharing a line with the table's other column headers. An
   anchored, line-start-only match never finds it, so gating never engaged for
   Section A at all, and every one of its nine questions (1-4, 5A/5B, 6A/6B,
   7A) was silently read as still being part of the instructions. Matching
   unanchored (search, not match) finds it -- but then also finds "SECTION A"
   and "SECTION B" used descriptively inside Instruction item 8 ("...Q9. In
   Section A-History (2 marks) and / Q19. In Section B -Geography (3
   marks)"), which is prose referencing a section, not heading one. Excluded
   by the one discriminator the evidence actually supports: every real header
   is either at the start of a line or beside other table-header text; every
   false positive found is preceded by the word "in"/"In".

5. A NUMBER AND ITS PART-LETTER CAN BE SEPARATED BY WHITESPACE, OR THE LETTER
   CAN BE MISSING ITS TRAILING PERIOD, PURELY AS A RENDERING INCONSISTENCY IN
   THE SAME DOCUMENT. Question 6 prints "6 A." / "6 B." (a space where
   5A./5B./17A./17B. have none); the scheme's answer for 7B prints "7B 1."
   with no period after B at all -- its own trailing period appears to have
   been swallowed by the value-point numbering that immediately follows. The
   first is tolerated (optional whitespace between the digit and the letter);
   the second is not yet, and 7B is a known unjoined key as a result. Neither
   is a "page number" -- an earlier version of this docstring's point 3
   attributed a bare "17." inside 7A's own multi-part answer to page-number
   bleed ("...in 2016- 17. Panchpatmali..."). It is not: the marking scheme is
   11 pages long, so there is no page 17. It is "2016-17", a school-year
   range, wrapped across a PDF line break -- and it was already handled
   correctly (the forward-only key invariant in parse_scheme rejects a
   same-or-lower number as a new header, so this text is correctly absorbed
   into the answer it belongs to, not split into a phantom entry).

6. A MATCHING-COLUMN MCQ CAN NUMBER ITS OPTIONS 1-4 INSTEAD OF LETTERING THEM
   A-D, because A-D already label the matching table's rows. Question 1's four
   candidate orderings print as their own lines, "1.A-4, B-1, C-2, D-3"
   through "4.A-4, B-1, C-3, D-4" -- each shaped exactly like a question
   header. Recognised by the option content itself (comma-separated
   `letter-digit` pairs), not the leading number, so real questions 2, 3 and 4
   are never mistaken for a fifth and sixth option of question 1.

The rule this module holds: it never pairs a question with an answer it cannot
key on exactly. Anything unmatched is reported as unmatched.

Current measured state, the real Class X 2025-26 Social Science pair
(2026-09-18, after points 4-6 above): all 38 of the paper's stated questions
are recognised (was 32); 34 of 44 question/alternative entries carry a mark
value; 37 of 44 join to a scheme answer (was 30 of 32). The 7 that don't:
questions 1-4's scheme answers use inconsistent separators ("1- A-4...", "2
B-...", no period) that `_HEADER` does not yet recognise; 7B's scheme entry is
missing its trailing period (point 5); 27A/27B have no scheme entry at all
under any key -- genuinely absent from the source document, not a parser gap.
Still not wired into any build script.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Iterator

try:
    import pymupdf as _fitz
except ImportError:                                # pragma: no cover
    import fitz as _fitz

# A page number is a bare number equal to the page index, sitting in a
# furniture position: vertically in the header/footer band, or horizontally in
# an outer margin. The value test alone would delete a real mark that happens to
# equal the page index (a 3-mark question on page 3); the position test alone
# would delete a question number that happens to be indented.
#
# MEASURED, and this was a real defect rather than a hypothetical one: the first
# version required only the HORIZONTAL margin, and the page numbers in this
# paper are centred, so they were never caught. They then fell through to the
# marks rule and were assigned to questions as mark values -- page 10's number
# became a 10-mark question, page 11's an 11-mark question. Twelve of the
# fourteen lone digits in the document were page numbers, and the marks
# distribution showed values (6,7,8,9,10,11) that no Social Science question
# could carry. Verified line by line against the page index before being fixed.
_MARGIN_FRACTION = 0.16
_TOP_BAND = 0.08
_BOTTOM_BAND = 0.92

# A mark value is right-aligned in a column of its own, so it is separated from
# the last word of the question text by a gap much wider than the normal space
# between words. Measured on the real paper: ordinary inter-word spacing is a
# few points, and the gap before the marks column is an order of magnitude more.
#
# The marks digit is NOT on its own line. Rows are grouped by y-position and the
# marks sit on the same visual row as the last line of the question text:
#
#     "...vibrant pre-modern trade and cultural links 2"
#     "...was not the result of a sudden upheaval 5"
#
# 21 lines have this shape and a lone-digit rule misses every one. Rather than
# teach the marks rule about gaps, the gap is used here to SPLIT the row, so the
# value arrives as a lone digit line and one rule handles both shapes.
#
# The gap is what stops a question that legitimately ends in a number from being
# misread as carrying that many marks.
_MARK_GAP = 18.0


def _is_page_number(text: str, page_number: int, *, x0: float, x1: float,
                    y0: float, y1: float, width: float, height: float) -> bool:
    """Is this word the page number rather than content?

    Both conditions are required. See the note above for why the horizontal
    margin alone was insufficient and what it cost.
    """
    # The trailing period is deliberately NOT stripped. A page number in this
    # corpus is a bare digit; a question header is `12.` with the period. So the
    # period separates them, and stripping it made `12.` at the top of page 12
    # indistinguishable from the page number -- which destroyed the header, the
    # exact failure the band was originally blamed for.
    stripped = (text or "").strip()
    # `.isdigit()` is True for characters `int()` cannot parse -- `²`.isdigit()
    # is True and int('²') raises ValueError. Found by parsing the wider
    # corpus: a formula containing x² crashed the parser. ASCII check, not isdigit.
    if not (stripped.isascii() and stripped.isdigit()):
        return False
    if int(stripped) != page_number:
        return False
    if width <= 0 or height <= 0:
        return False
    in_margin = (x0 < width * _MARGIN_FRACTION
                 or x1 > width * (1.0 - _MARGIN_FRACTION))
    in_band = y0 < height * _TOP_BAND or y1 > height * _BOTTOM_BAND
    return in_margin or in_band


# `15.` or `17A.` at the start of a line, optionally followed by text.
#
# The whitespace between the number and its optional part-letter is
# deliberate, not cosmetic. Found live 2026-09-18: this paper's question 6
# prints as "6 A." and "6 B." (a space where 5A./5B./17A./17B. have none) --
# a real kerning/rendering inconsistency in the source PDF, not a different
# convention. Without tolerating it, `_HEADER` never matches either line, so
# both fall through as body text and get silently appended onto question 5B,
# and question 6 never exists at all. The trailing `\.` immediately after the
# single captured letter keeps this safe: ordinary prose starting "6 Aluminium
# is..." still needs a period right after a single letter to match, which
# "Aluminium" (many letters, no period after the first) never satisfies.
_HEADER = re.compile(r"^\s*(\d{1,2})\s*([A-Z])?\.\s*(.*)$")

# A lone digit is the mark value for the preceding question.
_MARKS = re.compile(r"^\s*(\d{1,2})\s*$")

# The word that marks an internal choice.
_OR = re.compile(r"^\s*OR\s*$", re.I)

# `SECTION A`, `SECTION B`, ... Every paper opens with a General Instructions
# list that is numbered 1, 2, 3 -- the SAME shape as a question header. Without
# gating on the first section header those instructions are read as questions
# 1-6, which both invents six questions and swallows the real questions 1-7.
# Measured: 38 questions were still parsed correctly by count, but the numbering
# was shifted, so the join against the marking scheme was wrong for the first
# seven questions -- the exact silent error this module exists to avoid.
#
# Found live 2026-09-18, diagnosing "6 questions still missed": this used to be
# anchored (`^...`) and required "SECTION" to open the line. Section A's own
# header never matches that shape in the real Social Science SQP -- it prints
# as a table row, "Sr.No SECTION A Marks", with the column headers on either
# side on the SAME visual row. An anchored match against "SECTION" never finds
# it, so `started` never flips for Section A, and every question in it --
# including 5A/5B/6A/6B/7A -- was silently skipped as if it were still part of
# the General Instructions. Sections B/C/D happen to print as bare lines with
# no table-row prefix, which is why only Section A's nine questions were lost
# and the defect looked like "six missing" rather than "gating never engages
# at all" for a while.
#
# Unanchored `.search()` finds "SECTION A" wherever it sits on the row. The
# trailing section-letter group is deliberately NOT case-insensitive (unlike
# "SECTION" itself, via the inline (?i:...) flag) -- re.IGNORECASE on the
# whole pattern let `[A-Z]` match a lowercase letter too, so the instructions'
# own prose, "has Four Sections \u2013 A-History...", matched on "Section" + the
# lowercase "s" of "Sections" satisfying `[A-Z]` case-insensitively. Every
# real CBSE section header is capitalised ("SECTION A"); nothing before it
# ever is, so requiring an actual uppercase letter here removes that false
# positive without needing any position-based test at all.
#
# A second false positive surfaced once `.search()` was unanchored: General
# Instruction item 8 reads "...Q9. In Section A-History (2 marks) and / Q19.
# In Section B -Geography (3 marks)" -- genuinely referencing sections by
# name mid-sentence, not heading one, and both "A-History"/"B -Geography"
# still end in a bare capital letter satisfying the pattern above. Measured:
# every real section header in this paper is either the start of its own
# line, or immediately preceded by the table's other column headers
# ("Sr.No ... Marks"); every false positive found so far is preceded by the
# word "in"/"In". The lookbehind is the narrow, evidenced exclusion for that,
# not a guess at what else might precede a real header.
_SECTION = re.compile(r"(?<![Ii]n\s)(?i:SECTION)\s*[-\u2013\u2014]?\s*([A-E])\b")

# The largest question number a paper could plausibly reach. Guards against a
# stray large number being read as a header.
_MAX_Q = 60

# A "match the following, choose the correct option" MCQ numbers its four
# candidate orderings 1/2/3/4 rather than lettering them A-D, because A-D are
# already the matching-table's row labels. Found live 2026-09-18, once the
# SECTION-gating fix (above) finally let Section A's real content through:
# question 1's four options print as their own lines, "1.A-4, B-1, C-2, D-3"
# through "4.A-4, B-1, C-3, D-4" -- each one individually shaped exactly like
# a question header, so numbers 2, 3 and 4 were read as three new phantom
# questions and question 1 itself absorbed none of its own options. The
# option content itself has a shape no real question stem does -- four
# `letter-digit` pairs, comma-separated -- so that content, not the leading
# number, is what identifies these lines.
_MATCH_OPTION = re.compile(r"^[A-D]-\d+(?:,\s*[A-D]-\d+){2,}$")


@dataclasses.dataclass
class Question:
    number: int
    part: str = ""                     # "" for 15, "A"/"B" for 17A/17B
    text: str = ""
    marks: int = 0
    options: list[str] = dataclasses.field(default_factory=list)
    alternative_of: int | None = None   # set on a B that follows an A
    page: int = 0

    @property
    def key(self) -> str:
        return f"{self.number}{self.part}"


def _body_lines(pdf_path: Path) -> list[tuple[int, str]]:
    """`(page, line)` for the body text of every page, page numbers removed.

    Lines are rebuilt from positioned words rather than taken from
    `get_text()`, because a pageless `get_text()` splice puts the page number in
    the middle of a sentence -- "...in 2016- 17. Panchpatmali deposits..." --
    where no line-based rule can find it. Working from positioned words lets the
    page number be identified and dropped on its own merits.
    """
    out: list[tuple[int, str]] = []
    doc = _fitz.open(pdf_path)
    try:
        for index in range(doc.page_count):
            page = doc[index]
            width = page.rect.width or 1.0
            height = page.rect.height or 1.0
            try:
                words = page.get_text("words")
            except Exception:                      # pragma: no cover
                continue
            rows: dict[float, list] = {}
            for w in words:
                x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                if _is_page_number(text, index + 1, x0=x0, x1=x1, y0=y0, y1=y1,
                                   width=width, height=height):
                    continue
                rows.setdefault(round(y0 / 3.0), []).append((x0, x1, text))
            for key in sorted(rows):
                items = sorted(rows[key])
                # A trailing 1-6 set off by a wide gap is the marks column.
                if len(items) >= 2:
                    last_x0, _last_x1, last_text = items[-1]
                    if ((last_text.isascii() and last_text.isdigit())
                            and 1 <= int(last_text) <= 6
                            and (last_x0 - items[-2][1]) >= _MARK_GAP):
                        body = " ".join(t for _a, _b, t in items[:-1]).strip()
                        if body:
                            out.append((index + 1, body))
                        out.append((index + 1, last_text))
                        continue
                line = " ".join(t for _a, _b, t in items).strip()
                if line:
                    out.append((index + 1, line))
    finally:
        doc.close()
    return out


def parse_questions(pdf_path: Path) -> tuple[list[Question], list[str]]:
    """Questions from a Sample Question Paper, with marks and options."""
    lines = _body_lines(pdf_path)
    questions: list[Question] = []
    current: Question | None = None
    last_primary: Question | None = None
    errors: list[str] = []
    started = False

    for page, line in lines:
        # Nothing before the first SECTION header is a question. See _SECTION.
        if not started:
            if _SECTION.search(line):
                started = True
            continue

        marks_hit = _MARKS.match(line)
        header = _HEADER.match(line)

        # A lone digit is the mark value for the question just read. Checked
        # before the header, because a bare `1` is both a marks value and a
        # valid header shape.
        if marks_hit and current is not None and not current.marks:
            value = int(marks_hit.group(1))
            if 1 <= value <= 20:
                current.marks = value
                pending_marks = current
                continue

        # A matching-question option ("1.A-4, B-1, C-2, D-3") is shaped exactly
        # like a header but is never one -- see _MATCH_OPTION. Checked before
        # the header branch so it never gets a chance to start a phantom
        # question.
        if header and _MATCH_OPTION.match(header.group(3).strip()):
            if current is not None:
                current.options.append(line.strip())
            continue

        if header and int(header.group(1)) <= _MAX_Q and not (marks_hit and header.group(3) == ""):
            number = int(header.group(1))
            part = header.group(2) or ""
            rest = header.group(3).strip()
            # A header must move forward, or be an alternative part of the
            # question just seen. Anything else is body text that happens to
            # start with a number.
            if questions:
                last = questions[-1]
                forward = number > last.number or (number == last.number and part > last.part)
                alternative = (part == "B" and number == last.number and last.part == "A")
                if not (forward or alternative):
                    if current is not None:
                        current.text += " " + line
                    continue
            q = Question(number=number, part=part, page=page)
            if part == "B" and questions and questions[-1].number == number:
                q.alternative_of = number
                last_primary = questions[-1]
            questions.append(q)
            current = q
            if part == "":
                last_primary = q
            if rest:
                current.text = rest
            continue

        if _OR.match(line):
            continue
        if current is not None:
            current.text += " " + line
            # Option lines in an MCQ start "A." "B." "C." "D."
            if re.match(r"^\s*[A-D]\.\s", line):
                current.options.append(line.strip())

    for q in questions:
        q.text = " ".join(q.text.split())
    if not questions:
        errors.append(f"{pdf_path.name}: no questions parsed")
    return questions, errors


def parse_scheme(pdf_path: Path) -> tuple[dict[str, str], dict[str, int], list[str]]:
    """`question key -> answer text` and `question key -> marks`.

    Keyed the same way as the questions (`"15"`, `"17A"`) so the join is exact.
    A key that cannot be matched is simply absent, never guessed.
    """
    lines = _body_lines(pdf_path)
    answers: dict[str, str] = {}
    marks: dict[str, int] = {}
    current_key: str | None = None
    errors: list[str] = []
    last_number = 0

    for _page, line in lines:
        header = _HEADER.match(line)
        marks_hit = _MARKS.match(line)

        if marks_hit and current_key is not None and current_key not in marks:
            value = int(marks_hit.group(1))
            if 1 <= value <= 20:
                marks[current_key] = value
                continue

        if header:
            number = int(header.group(1))
            part = header.group(2) or ""
            rest = header.group(3).strip()
            if number <= _MAX_Q:
                forward = number > last_number or (number == last_number and part)
                if forward:
                    current_key = f"{number}{part}"
                    answers.setdefault(current_key, "")
                    last_number = max(last_number, number)
                    if rest:
                        answers[current_key] += " " + rest
                    continue
        # An "OR" marker between two alternatives is structural, not part of
        # either answer's text -- parse_questions already strips it (_OR);
        # this side never did, so the literal word "OR" was leaking into
        # every alternative-question's stored answer text.
        if _OR.match(line):
            continue
        if current_key is not None:
            answers[current_key] += " " + line

    for k in answers:
        answers[k] = " ".join(answers[k].split())
    if not answers:
        errors.append(f"{pdf_path.name}: no marking scheme entries parsed")
    return answers, marks, errors


@dataclasses.dataclass
class SqpQuestion:
    """A question that HAS an answer key. Without one it is not emitted."""

    question: Question
    answer: str
    scheme_marks: int


def parse_pair(sqp_path: Path, ms_path: Path) -> tuple[list[SqpQuestion], dict]:
    """Join a paper to its scheme by question number.

    Reports what it could not join, rather than emitting a question with an
    empty or borrowed answer.
    """
    questions, q_errors = parse_questions(sqp_path)
    answers, marks, ms_errors = parse_scheme(ms_path)

    joined: list[SqpQuestion] = []
    unjoined: list[str] = []
    for q in questions:
        text = answers.get(q.key, "").strip()
        if not text:
            unjoined.append(q.key)
            continue
        joined.append(SqpQuestion(question=q, answer=text,
                                  scheme_marks=marks.get(q.key, 0)))

    stats = {
        "questions": len(questions),
        "schemeEntries": len(answers),
        "joined": len(joined),
        "unjoined": unjoined,
        "errors": q_errors + ms_errors,
    }
    return joined, stats
