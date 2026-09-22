"""Official answer keys, lifted from CBSE marking schemes.

The question papers say what was asked; only the marking schemes say what a
correct answer looks like. Without them every MCQ evaluates to "no correct
option recorded" and every descriptive answer is scored against an empty
rubric, so the teacher review queue is noise.

A marking scheme PDF is laid out as a table:

    Q.No.   EXPECTED ANSWERS / VALUE POINTS                      Marks  Total
    1.      D /1: 8                                                  1      1
    2.      B / Al2O3 and MgO                                        1      1
    ...
    21.     * Evolution of gas                                       1
            * Change / Rise in temperature                           1      2

Section A (objective) gives a letter and the option text. Sections B onwards
give value points — the individual scoring steps an examiner ticks — each with
its own mark. Both are worth capturing: the letter drives automatic MCQ
scoring, the value points become the marking points a teacher sees.

One PDF often covers several paper variants (31/1/1 through 31/1/3) with the
question numbering restarting at each. Every page carries its own paper code in
the header, so pages are grouped by code before parsing rather than treating
the file as one document.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config

log = logging.getLogger(__name__)

# "31/1/1", "X_086_ 31/1/1", "Paper Code: 30-1-1". Two-to-three digit series,
# then set, then variant.
#
# Separators widened 2026-09-17 after measuring the real marking-scheme corpus.
# CBSE writes the code with whatever separator the filename happens to use, and
# real files on disk carry all of these:
#     31-1-1     X_086_31-1-1 to 3 Science_MS.pdf
#     31.1.1     X_086_31.1.1 to 31.1.3 in HINDI.pdf
#     31_1-1     X_086_31_1-1 to 3_Science _HINDI_Med.pdf
#     55_7_1     XII_042_Physics_MS_55_7_1,2,3_hindi_med.pdf
# The old `[/-]`-only pattern indexed 1,040 of 4,483 answer-key rows (23%); it
# now indexes the ones a filename can actually name.
#
# `\b` was also wrong: it does not fire between "_" and a digit, so
# "X_086_31_1-1..." was unmatched even for the separators it did allow. The
# digit lookarounds express the real constraint -- do not splice a code out of a
# longer run of digits, such as the year in "2025_086_...".
_PAPER_CODE = re.compile(
    r"(?<!\d)(\d{1,3})\s*[/\-._]\s*(\d)\s*[/\-._]\s*(\d)(?!\d)")

# A CBSE marking scheme names the subject code immediately before the paper
# code, both in its header ("MS-English Core/301/1-5-1") and in its filename
# ("XII_301_1_1_1_MS.pdf"). Reading three components from the left then takes
# the SUBJECT code plus the paper code's first two parts and drops its last --
# `301/1/5` for a paper that is really `1/5/1`. That invents a code that does
# not exist, and collapses sets 1, 2 and 3 of one paper onto a single key, so
# each ingested set silently overwrote the previous one. Matching the four-part
# shape first, and discarding the lead only when it is a subject code we
# recognise, keeps the real code intact.
_SUBJECT_THEN_CODE = re.compile(
    r"(?<!\d)(\d{3})\s*[/\-._]\s*(\d{1,3})\s*[/\-._]\s*(\d)\s*[/\-._]\s*(\d)(?!\d)")

# Subject codes as they appear in CBSE marking-scheme filenames and headers.
_SUBJECT_BY_CODE = {
    "086": "Science",
    "041": "Mathematics",
    "241": "Mathematics",
    "087": "Social Science",
    "184": "English",
    "101": "English",
    "301": "English",      # English Core, class XII
    "042": "Physics",
    "043": "Chemistry",
    "044": "Biology",
    "055": "Accountancy",
    "054": "Business Studies",
    "030": "Economics",
    "027": "History",
    "029": "Geography",
    "028": "Political Science",
}

# Lower-cased subject-name fallbacks for scheme files whose path lacks a code.
# "buisness" is not a typo here -- CBSE 2022 XII schemes ship as
# `Buisness_Studies/`, misspelled upstream.
_SUBJECT_BY_NAME = (
    "social science", "science", "mathematics", "math", "english",
    "business studies", "business", "buisness studies", "accountancy",
    "economics", "physics", "chemistry", "biology", "history", "geography",
    "political science",
)

# Marking-scheme boilerplate: the examiner instructions that precede the actual
# answers. A "1." inside these pages is an instruction number, not a question.
# Hindi-medium schemes label the same boundary "खण्ड - क" (Section A), so cut
# there too or their preamble's numbered instructions are mistaken for answers.
_PREAMBLE_END = re.compile(
    r"(?:SECTION|खण्ड)\s*[-–—]?\s*(?:A|क)\b", re.I)

# Section A objective answer, two layouts seen across subjects:
#   Science: "1.\nD / 1: 8"                  — letter and text on one line
#   Math:    "1.\n...\nSol.\n(d) answer"      — letter in parens after "Sol."
# Both end the answer at the mark column (a line holding only a mark).
_MCQ_SLASH = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*\n"          # 1.
    r"[ \t]*([A-D])[ \t]*/[ \t]*"                  # D /
    r"(.*?)"                                       # answer text (may wrap)
    r"(?=\n[ \t]*(?:1|2|½|\d{1,2}\s*\.[ \t]*\n))",  # up to marks or next Q
    re.S,
)
_MCQ_SOL = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*\n"          # 1.
    r"(?:[ \t]*\n)*"                               # blank lines before Sol.
    r"[ \t]*Sol\.?[ \t]*\n"                        # Sol.
    r"[ \t]*\(([a-dA-D])\)[ \t]*"                  # (d)
    r"(.*?)"                                       # answer text
    r"(?=\n[ \t]*(?:1|2|½|\d{1,2}\s*\.[ \t]*\n))",
    re.S,
)

# Section A objective answer, third layout — the letter alone, without either
# a "D /" prefix or a "Sol." lead-in. Hindi-medium schemes and some English
# "set" files write it as the question number, then the letter in parens on
# its own line, then the marks column:
#     2.
#     (c)
#     1
#     1
# Moved into the DB as a bare option so automatic scoring still keys a letter.
# The marks line must immediately follow the letter: a value-point part label
# ("14.\n(a)\n• In a few reptiles...") also looks like a lone (a) and must NOT
# be read as an MCQ option, so the lookahead pins the letter to the mark column
# (1 / 1 1 / ½) that only a real marks-only layout owns. answer_text stays empty.
_MCQ_PAREN = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*\n"          # 2.
    r"[ \t]*\(([a-dA-D])\)[ \t]*\n"               # (c) alone on the line
    r"(?=(?:[ \t]*[12½](?:[ \t]*\n|[ \t]+[12½][ \t]*\n)))"   # marks line follows
    r"()",                                             # no answer text in this layout
    re.S,
)

# Fifth layout — Hindi-style bare numbers without period:
#     1
#     (a)
#     1
#     1
_MCQ_PAREN_BARE = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\n"                   # 2 (bare, no period)
    r"[ \t]*\(([a-dA-D])\)[ \t]*\n"               # (c) alone on the line
    r"(?=(?:[ \t]*[12½](?:[ \t]*\n|[ \t]+[12½][ \t]*\n)))"   # marks line follows
    r"()",                                             # no answer text
    re.S,
)

# Section A objective answer, fourth layout -- a bare running list with no
# question numbers:
#     Answer (B) 5 units
#     Ans. (c)
#     2800
#     Ans.. (A) x2 - x - 6
# CBSE uses this in Mathematics Basic (430/*) and some XII schemes; the list
# always runs q1..qN in paper order, with the marks column at the line end,
# a dashed rule under each answer, and the answer text beside or below the
# header. Question numbers are recovered from the list position.
_ANSWER_LIST_HEADER = re.compile(
    r"^[ \t]*(?:Answer|Ans\.{0,2}|Solution|Sol\.)[ \t]*\(([a-dA-D])\)[ \t]*(.*)$")

# A right-aligned marks column that PyMuPDF merges onto the answer line, e.g.
# "Answer (C) P(E) + P( E ) = 1                           1". Only strip it when it
# is separated by two+ runes of space so an answer text that itself ends in a
# digit ("(D) - 6", "x = 2") is never mangled.
_TRAILING_MARK = re.compile(r"[ \t]{2,}(?:½|[0-9]+(?:\s*½)?)[ \t]*$")

# A line that is nothing but the marks column, standing alone (the "Ans."
# files put the mark on its own line under the answer). Section A marks are
# 1, 2 or ½ -- never three figures, so "2800" (a real answer) is kept.
_MARK_TOKEN_LINE = re.compile(
    r"^[ \t]*(?:½|[1-3](?:[ \t]*½)?)(?:[ \t]+(?:½|[1-3]))*[ \t]*$")

# The dashed rule @u2026 that separates consecutive answers in the "Ans." files.
_RULE_LINE = re.compile(r"^[ \t]*[_\-–—]{8,}[ \t]*$")

# A value point: a bulleted/dashed scoring step in sections B onwards.
_VALUE_POINT = re.compile(r"(?m)^[ \t]*[•▪●➢*•][ \t]*(.+)$")

# A mark token standing alone in the marks column.
_MARK_TOKEN = re.compile(r"^\s*(½|\d+(?:\s*½)?)\s*$")


def normalize_paper_code(raw: str) -> str:
    """Renders any of 31/1/1, 31-1-1, '31 / 1 / 1' as the canonical 31/1/1.

    A leading subject code is dropped rather than spliced in -- see
    `_SUBJECT_THEN_CODE` for why that mattered.
    """
    m = _SUBJECT_THEN_CODE.search(raw)
    if m and m.group(1) in _SUBJECT_BY_CODE:
        return f"{m.group(2)}/{m.group(3)}/{m.group(4)}"
    m = _PAPER_CODE.search(raw)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def subject_from_path(path: Path) -> str | None:
    """Infers the subject from a marking-scheme path.

    CBSE nests these as `.../Science/086 Science (English Medium)/X_086_...pdf`,
    so the subject code in the filename is the most reliable signal; the folder
    name is the fallback for files that omit it.
    """
    text = str(path)
    for code, subject in _SUBJECT_BY_CODE.items():
        if re.search(rf"[_\s/\\]{code}[_\s]", text):
            return subject
    lowered = text.lower()
    for subject in _SUBJECT_BY_NAME:
        if subject in lowered:
            return "Mathematics" if subject == "math" else (
                "Business Studies" if "buisness" in subject else subject.title())
    return None


def grade_from_path(path: Path) -> str | None:
    """Class X vs XII, from the `/2025/X/` segment CBSE uses."""
    for part in path.parts:
        if part in ("X", "XII"):
            return part
    if re.search(r"[\\/]XII[_\s]", str(path)):
        return "XII"
    if re.search(r"[\\/]X[_\s]", str(path)):
        return "X"
    return None


@dataclass
class KeyedAnswer:
    """The official answer to one question of one paper variant."""
    q_no: int
    correct_option: str | None          # "D" for objective questions
    answer_text: str                    # option text, or the joined value points
    value_points: list[tuple[str, float]] = field(default_factory=list)
    marks: float = 0.0


@dataclass
class PaperKey:
    paper_code: str
    subject: str | None
    grade: str | None
    answers: dict[int, KeyedAnswer] = field(default_factory=dict)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip(" .:/")


def _pages_by_paper_code(pdf_path: Path) -> dict[str, str]:
    """Groups a marking scheme's pages by the paper code in each page header.

    A single PDF routinely covers 31/1/1 through 31/1/3 with question numbers
    restarting at 1 for each, so parsing the file as one blob would collide the
    variants' answers.
    """
    import fitz

    grouped: dict[str, list[str]] = {}
    current = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            # The code lives in the running header, so only look at the top.
            code = normalize_paper_code(text[:300])
            if code:
                current = code
            if not current:
                continue
            grouped.setdefault(current, []).append(text)
    return {code: "\n".join(pages) for code, pages in grouped.items()}


def _parse_objective(body: str) -> dict[int, KeyedAnswer]:
    """Pulls Section A letter answers out of one paper variant's text."""
    answers: dict[int, KeyedAnswer] = {}
    for pattern in (_MCQ_SLASH, _MCQ_SOL, _MCQ_PAREN, _MCQ_PAREN_BARE):
        for m in pattern.finditer(body):
            q_no = int(m.group(1))
            # Section A is questions 1-20; a later match with a low number means
            # the regex wandered into a different section, so ignore it.
            if not 1 <= q_no <= 20 or q_no in answers:
                continue
            answers[q_no] = KeyedAnswer(
                q_no=q_no,
                correct_option=m.group(2).upper(),
                answer_text=_clean(m.group(3)),
                marks=1.0,
            )
    return answers


def _parse_answer_list(body: str) -> dict[int, KeyedAnswer]:
    """Pulls Section A letter answers from the numbered-less running list.

    The four numbered layouts above anchor a question number, so they can
    never read a list that carries none. CBSE's Mathematics Basic schemes
    (430/*) and some XII schemes write Section A as:
        Answer (B) 5 units
        Ans. (c)
        2800
        Ans.. (A) x2 - x - 6
    in paper order, q1..qN, each answer closed by a dashed rule and its marks
    column at the line end. Only use this pass when the numbered layouts found
    nothing -- a paper that has both would otherwise key its Section A twice.
    """
    lines = body.splitlines()
    headers: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = _ANSWER_LIST_HEADER.match(line)
        if m:
            headers.append((i, m.group(1).upper(), m.group(2)))
    if len(headers) < 2:
        return {}
    # Cut the list where the scheme switches to the value-point (Solution:)
    # layout or passes into another section.
    end = len(lines)
    for i, line in enumerate(lines):
        if re.match(r"^[ \t]*(?:Sol(?:ution)?\s*[:.]|SECTION\s*[BCbc])", line):
            end = i
            break

    answers: dict[int, KeyedAnswer] = {}
    q_no = 0
    for idx, (i, letter, inline) in enumerate(headers):
        if i >= end:
            break
        q_no += 1
        if q_no > 20:
            break
        parts: list[str] = []
        # Inline text sometimes carries the marks column: strip it before
        # cleaning collapses the separating whitespace to one space.
        raw_inline = _TRAILING_MARK.sub("", inline).rstrip()
        if raw_inline:
            parts.append(_clean(raw_inline))
        # The "Ans." layouts put the answer on the lines under the header,
        # up to the marks-only line or the next header.
        j = i + 1
        while j < end:
            line = lines[j]
            if _ANSWER_LIST_HEADER.match(line):
                break
            if _RULE_LINE.match(line):
                break
            if _MARK_TOKEN_LINE.match(line):
                break
            if line.strip():
                parts.append(_clean(line))
            j += 1
        text = " ".join(p.strip(" .:") for p in parts if p.strip(" .:"))
        answers[q_no] = KeyedAnswer(
            q_no=q_no,
            correct_option=letter,
            answer_text=text,
            marks=1.0,
        )
    return answers


# A SECTION header line. CBSE prints it as "SECTION B", "Section A", or with a
# separating dash/period; Hindi-medium schemes keep खण्ड (handled via the पreamble
# cut). The value sections the section-aware pass cares about are the English
# ones, so the marker is matched case-insensitively without the Hindi forms.
# Hindi section headers: "खंड—अ", "खंड ख", "खंड – ग", "खंड – घ", "खंड – ड",
# "खंड - ग", "खंड- क", "खंड-ख" etc.  Letters: अ क ख ग घ ड (A-E).
_HINDI_SECTION_LETTERS = "अकखगघड"
_HINDI_TO_EN = {"अ": "A", "क": "A", "ख": "B", "ग": "C", "घ": "D", "ड": "E"}

def _sec_label(m: re.Match) -> str:
    """Returns the section label as A-E (English) from either regex group."""
    if m.group(1):
        return m.group(1).upper()
    return _HINDI_TO_EN.get(m.group(2), "?")

_SECTION_HEADER = re.compile(
    r"(?m)^[ \t]*(?:SECTION[ \t—-]+([A-E])"
    r"|खंड[ \t—\-—]+([" + _HINDI_SECTION_LETTERS + r"]))"
    r"[ \t]*[:.\-—]?[ \t]*$", re.I)


def _section_blocks(block: str) -> list[str]:
    """Splits one section's answer run into per-question answer TEXT blocks.

    Prefers `Solution:`/`Sol.` line breaks (the Mathematics Basic 430/* layout:
    each solution block holds exactly one global question, with OR and (a)/(b)
    parts inside it). Falls back to question-number lines: returns the text
    BETWEEN consecutive number lines (i.e. the answer content), not the numbers.
    Matches both "N." and bare "N" on their own line.
    """
    if re.search(r"(?m)^[ \t]*Sol(?:ution)?\s*[:.]", block):
        return re.split(r"(?m)^[ \t]*Sol(?:ution)?\s*[:.]", block)[1:]
    parts = re.split(r"(?m)^[ \t]*(\d{1,2})\s*(?:\.|$)", block)
    # parts = [pre, num1, text1, num2, text2, ...] -> return [text1, text2, ...]
    return parts[2::2]


def _value_points_from_block(block: str) -> list[str]:
    """Collects the scoring steps inside one value-point block.

    The 430/* solution blocks are unbulleted prose, so the bullet pass finds
    nothing there; the fallback keeps the first substantial lines exactly as
    the legacy parser did.
    """
    points = [_clean(p) for p in _VALUE_POINT.findall(block)]
    points = [p for p in points if len(p) > 8][:8]
    if not points:
        lines = [_clean(l) for l in block.splitlines()]
        points = [l for l in lines if len(l) > 25][:4]
    return points


def _section_a_len(body: str, header_iter: list[re.Match]) -> int:
    """Counts the questions in Section A from the scheme's own section text.

    The objective passes can return nothing for the English "1.\n(B) answer"
    inline layout, so this measures the section directly. Section A runs from
    its own header (or the body start when it carries no header, as the Science
    31/* value-passed schemes do) up to the next SECTION marker:
      - the `Answer (x)` / `Ans.` list count (Mathematics Basic 430/*), or
      - the largest standalone question number in the section (everything else).
    """
    first_label = _sec_label(header_iter[0]) if header_iter else "?"
    if header_iter and first_label == "A":
        a_start = header_iter[0].start()
        a_end = (header_iter[1].start()
                 if len(header_iter) > 1 else len(body))
        a_text = body[a_start:a_end]
    else:
        a_text = body[:header_iter[0].start()] if header_iter else body
    headers = [m for ln in a_text.splitlines()
               if (m := _ANSWER_LIST_HEADER.match(ln))]
    if len(headers) >= 2:
        return len(headers)
    # Fallback: try objective MCQ patterns (incl. bare parens for Hindi)
    obj = _parse_objective(a_text)
    if obj:
        return len(obj)
    nums = [int(m.group(1))
            for m in re.finditer(r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*$", a_text)]
    return max(nums) if nums else 0


def _parse_value_points(body: str, section_a_count: int = 0) -> dict[int, KeyedAnswer]:
    """Pulls sections B+ value points, keyed by the GLOBAL question number.

    CBSE's new-layout schemes renumber questions per section, or drop the number
    entirely beneath `Sections A-E` headers, so a literal "17." split cannot key
    them. When the body carries real SECTION headers the sections are walked in
    order and each question block advances a running counter seeded by Section
    A's length -- the only numbering the question paper itself agrees with.

    The legacy literal-number split runs as a gap-fill underneath: some headed
    schemes (Social Science 32/7/*, some Science 31/*) head sections but still
    number their value questions globally, and the section walk would otherwise
    lose them entirely.

    Bodies with no SECTION headers keep the legacy literal-number split alone
    (the Science/Math standard schemes write their value questions globally
    1..N).

    Marks per point are not reliably alignable to their text in a flattened PDF
    (the mark column is emitted separately), so each point carries 0.0 and the
    teacher allocates -- better than inventing a number.
    """
    answers: dict[int, KeyedAnswer] = {}
    header_iter = list(_SECTION_HEADER.finditer(body))

    def from_blocks(blocks: list[str]) -> None:
        for i in range(1, len(blocks) - 1, 2):
            try:
                q_no = int(blocks[i])
            except ValueError:
                continue
            if q_no in answers:  # Section A (1-20) keyed by the objective passes
                continue
            points = _value_points_from_block(blocks[i + 1])
            if not points:
                continue
            answers[q_no] = KeyedAnswer(
                q_no=q_no,
                correct_option=None,
                answer_text=" ".join(points)[:1200],
                value_points=[(p, 0.0) for p in points],
            )

    legacy_blocks = re.split(r"(?m)^[ \t]*(\d{1,2})\s*(?:\.|Q\.)[ \t]*", body)

    if len(header_iter) == 0:
        # No SECTION headers at all: pure legacy literal-number split.
        from_blocks(legacy_blocks)
        return answers

    # Section-aware pass, then legacy as gap-fill for whatever it missed.
    run = max(section_a_count, _section_a_len(body, header_iter))
    seen_sections = set()
    for hi in range(len(header_iter)):
        start = header_iter[hi].start()
        end = (header_iter[hi + 1].start()
               if hi + 1 < len(header_iter) else len(body))
        sec_label = _sec_label(header_iter[hi])
        if sec_label in seen_sections:
            continue
        seen_sections.add(sec_label)
        sec_text = body[start:end]
        if sec_label == "A" or not sec_text.strip():
            continue
        blocks = _section_blocks(sec_text)
        if not blocks:
            continue
        # Prefer the printed numbers only when they continue the running counter
        # (global numbering, e.g. English 55/1/1: B starts at 17 after A's 16).
        # Filter to numbers >= run+1 to avoid picking up marks (1, 2, 3...).
        raw_numbered = re.findall(r"(?m)^[ \t]*(\d{1,2})\s*(?:\.|$)", sec_text)
        # Extract the longest consecutive prefix starting from run+1.
        # This avoids marks/page numbers (e.g. 38, 60) that are >= run+1 but not consecutive.
        numbered = []
        expected = run + 1
        for n in raw_numbered:
            val = int(n)
            if val == expected:
                numbered.append(n)
                expected += 1
            elif val > expected:
                # Gap detected - stop looking for consecutive questions
                break
        global_numbers = (
            numbered and int(numbered[0]) == run + 1
            and all(int(numbered[i]) == int(numbered[0]) + i
                    for i in range(len(numbered)))
        )
        if global_numbers:
            # Split section text by the question numbers to get answer blocks
            # that align with the numbered list (avoids splitting on marks).
            parts = re.split(r"(?m)^[ \t]*(\d{1,2})\s*(?:\.|$)", sec_text)
            # parts = [pre, num1, text1, num2, text2, ...]
            aligned_blocks = parts[2::2]
            # Filter aligned_blocks to only those corresponding to numbered >= run+1
            filtered_blocks = []
            for n, b in zip(raw_numbered, parts[2::2]):
                if int(n) >= run + 1:
                    filtered_blocks.append(b)
            if len(filtered_blocks) == len(numbered):
                for qno_text, block in zip(numbered, filtered_blocks):
                    points = _value_points_from_block(block)
                    if not points:
                        continue
                    q_no = int(qno_text)
                    if q_no in answers:
                        continue
                    answers[q_no] = KeyedAnswer(
                        q_no=q_no, correct_option=None,
                        answer_text=" ".join(points)[:1200],
                        value_points=[(p, 0.0) for p in points])
                    run = max(run, q_no)
                continue
            # Fall back to generic blocks if alignment fails
            blocks = filtered_blocks
        # Otherwise local numbering (or none): advance the running counter.
        for block in blocks:
            points = _value_points_from_block(block)
            if not points:
                continue
            run += 1
            if run in answers:
                continue
            answers[run] = KeyedAnswer(
                q_no=run, correct_option=None,
                answer_text=" ".join(points)[:1200],
                value_points=[(p, 0.0) for p in points])

    # Handle "SECTION A only" schemes where value questions continue globally
    # after A (e.g. 66/1/1 Business Studies: A ends at 20, then 21 Q., 22 Q....).
    if len(header_iter) == 1 and _sec_label(header_iter[0]) == "A":
        a_end = header_iter[0].end()
        tail = body[a_end:]
        if tail.strip():
            # Try numbered blocks first (global numbering: "21 Q.", "22 Q." etc.)
            num_blocks = re.split(r"(?m)^[ \t]*(\d{1,2})\s*[.Qq]", tail)
            if len(num_blocks) > 2:
                for i in range(1, len(num_blocks) - 1, 2):
                    try:
                        q_no = int(num_blocks[i])
                    except ValueError:
                        continue
                    if q_no <= run or q_no in answers:
                        continue
                    points = _value_points_from_block(num_blocks[i + 1])
                    if not points:
                        continue
                    answers[q_no] = KeyedAnswer(
                        q_no=q_no, correct_option=None,
                        answer_text=" ".join(points)[:1200],
                        value_points=[(p, 0.0) for p in points])
                    run = max(run, q_no)

    # Gap-fill: section-walked schemes that ALSO carry global question numbers
    # (Social Science 32/7/*, numbered Science 31/*) would otherwise lose every
    # value answer. The literal split keys those directly and cannot disturb a
    # key the section walk already claimed.
    from_blocks(legacy_blocks)
    return answers


def parse_marking_scheme(pdf_path: Path) -> list[PaperKey]:
    """Extracts every paper variant's answer key from one marking-scheme PDF.

    Raises whatever `_pages_by_paper_code` raises (a corrupt or image-only PDF
    typically fails inside PyMuPDF) rather than swallowing it. This used to
    catch here and return an empty list, logging only a warning -- silently
    indistinguishable, to every caller, from "this PDF genuinely has zero
    parseable answer keys." Found live 2026-09-18: `build_answer_keys` (below)
    had no error handling of its own and would have crashed on the first bad
    file if this ever raised, and `scripts/build_answer_keys.py` already had a
    try/except around this exact call, with a `failed` counter and a final
    "N unreadable" report -- code that could never fire because this function
    never let anything through it to catch. Both callers below now count and
    report failures explicitly instead. A corrupt or image-only PDF still
    cannot abort a batch run; it just no longer disappears into a truthful-
    looking zero.
    """
    subject = subject_from_path(pdf_path)
    grade = grade_from_path(pdf_path)
    keys: list[PaperKey] = []
    grouped = _pages_by_paper_code(pdf_path)

    for code, body in grouped.items():
        # Drop the examiner-instruction preamble so its numbered list is not
        # mistaken for answers.
        cut = _PREAMBLE_END.search(body)
        answers_body = body[cut.start():] if cut else body
        answers = _parse_objective(answers_body)
        if not answers:
            answers = _parse_answer_list(answers_body)
        # Value points key questions by number too, but stray "1."/"16." lines
        # that survive inside Section B+ body text can collide with a question
        # already keyed objectively -- the objective answer must win (setdefault,
        # not update).
        for q_no, keyed in _parse_value_points(answers_body).items():
            answers.setdefault(q_no, keyed)
        if answers:
            keys.append(PaperKey(paper_code=code, subject=subject,
                                 grade=grade, answers=answers))
    return keys


def discover_marking_schemes(root: Path) -> list[Path]:
    """Every marking-scheme PDF under a corpus root, archives already expanded."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


class AnswerKeyStore:
    """SQLite-backed lookup of official answers by (subject, grade, paper, q_no)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self._migrate()

    def _migrate(self) -> None:
        # Paper codes ("31/1/1") are only unique WITHIN a subject/grade — CBSE
        # reuses the same numeric code space across subjects (Science and
        # Social Science both had "32/1/1" papers), so a key of paper_code+q_no
        # alone silently overwrites one subject's answers with another's.
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS answer_keys (
                subject        TEXT NOT NULL DEFAULT '',
                grade          TEXT NOT NULL DEFAULT '',
                paper_code     TEXT NOT NULL,
                q_no           INTEGER NOT NULL,
                correct_option TEXT,
                answer_text    TEXT NOT NULL,
                value_points   TEXT NOT NULL DEFAULT '[]',
                marks          REAL NOT NULL DEFAULT 0,
                source         TEXT,
                PRIMARY KEY (subject, grade, paper_code, q_no)
            );
            CREATE INDEX IF NOT EXISTS idx_keys_subject
                ON answer_keys (subject, grade);
            """
        )
        self.conn.commit()

    def put(self, key: PaperKey, source: str) -> int:
        import json

        subject = key.subject or ""
        grade = key.grade or ""
        rows = [
            (subject, grade, key.paper_code, a.q_no, a.correct_option,
             a.answer_text, json.dumps(a.value_points), a.marks, source)
            for a in key.answers.values()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO answer_keys (subject, grade, paper_code, q_no, "
            "correct_option, answer_text, value_points, marks, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def lookup(self, paper_code: str, q_no: int, *,
               subject: str | None = None, grade: str | None = None) -> KeyedAnswer | None:
        import json

        if subject is not None:
            row = self.conn.execute(
                "SELECT * FROM answer_keys WHERE paper_code=? AND q_no=? "
                "AND subject=? AND (grade=? OR grade='')",
                (paper_code, q_no, subject, grade or ""),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM answer_keys WHERE paper_code=? AND q_no=?",
                (paper_code, q_no),
            ).fetchone()
        if row is None:
            return None
        return KeyedAnswer(
            q_no=row["q_no"],
            correct_option=row["correct_option"],
            answer_text=row["answer_text"],
            value_points=[tuple(p) for p in json.loads(row["value_points"])],
            marks=row["marks"] or 0.0,
        )

    def stats(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) total, "
            "SUM(CASE WHEN correct_option IS NOT NULL THEN 1 ELSE 0 END) objective, "
            "COUNT(DISTINCT paper_code) papers FROM answer_keys"
        ).fetchone()
        return {"total": row["total"] or 0,
                "objective": row["objective"] or 0,
                "papers": row["papers"] or 0}

    def close(self) -> None:
        self.conn.close()


def default_db(cfg: Config) -> Path:
    return cfg.data_root / "answer_keys.db"


def build_answer_keys(cfg: Config, roots: list[Path]) -> dict[str, int]:
    """Parses every marking scheme under `roots` into the answer-key store.

    A single corrupt or image-only PDF must not abort the whole run, but
    (2026-09-18) it must not silently vanish into the totals either --
    `failed` in the returned dict is the count of files parse_marking_scheme
    could not read at all, separate from files that were read fine and
    genuinely had nothing to extract.

    `papers_written` is this run's count of paper variants written to the
    store, kept distinct from `**stats`'s `papers` (the store's cumulative
    distinct-paper-code count across every run ever, including previous
    ones) -- these two used to share the key "papers" in the returned dict,
    with `**stats` silently winning and this run's own count discarded
    (found live 2026-09-18 while fixing the failure-counting bug above, in
    the same dict literal).
    """
    store = AnswerKeyStore(default_db(cfg))
    files = [p for root in roots for p in discover_marking_schemes(root)]
    written = papers_written = failed = 0
    for path in files:
        try:
            keys = parse_marking_scheme(path)
        except Exception as exc:
            failed += 1
            log.warning("cannot read marking scheme %s: %s", path.name, exc)
            continue
        for key in keys:
            written += store.put(key, source=path.name)
            papers_written += 1
    stats = store.stats()
    store.close()
    log.info("answer keys: %d answers, %d paper variants written this run; "
             "store now holds %d answers across %d paper variants total, "
             "from %d files (%d unreadable)",
             written, papers_written, stats["total"], stats["papers"], len(files), failed)
    return {"files": len(files), "written": written, "papers_written": papers_written,
            "failed": failed, **stats}
