"""Parser for the official CBSE CBE item banks.

What these documents are, and why they are the right source
----------------------------------------------------------
CBSE publishes 15 item banks (Classes 6-10 x Mathematics, Science, English) at
`cbseacademic.nic.in/cbe/assessment.html`. They were produced with the British
Council and Alpha Plus, and they are the artefact the requirement actually
describes: a board writing questions against a curriculum, with an answer key.

Every item carries, in a fixed layout:

    Item identity        Maths6AS1
    AO1 marks            1
    AO2 marks
    C/N/E*               N                      (calculator rule)
    Content reference    6A1a Form and use algebraic expressions (up to 2 variables...)
    Marks                1
    Item purpose         prose
    Question(s)          the question, its options, and (N mark)
    Mark scheme          Answer | Guidance table, i.e. the ANSWER KEY

So the topic mapping is not inferred from keywords -- it is a curriculum code
CBSE wrote (`6A1a`), plus a named strand (`Algebra`). And the answer key is not
reconstructed from the text -- it is the published one.

Format notes that the parser depends on, all established by reading the real
files rather than assumed:

  * A running header (`www.britishcouncil.org`, the page number) must be
    stripped before anything else, or every field parse picks up the page
    number.
  * An item BEGINS on a page whose first non-blank line is the item identity
    alone, and may CONTINUE onto following pages. Items therefore have to be
    accumulated page-by-page, not matched within a page.
  * `pymupdf` (imported as `fitz`) emits the metadata table cells in a
    different order from their visual layout, so fields are located by their
    content (the code pattern) rather than by position.

Deliberately NOT parsed: the diagram images. Items that depend on a figure are
flagged (`needs_figure`) rather than silently exported without it, because a
figure-dependent question with no figure is unanswerable -- the same reason
`assessment/pool.py` excludes them from paper generation.

English's items are not all self-contained the way the layout above assumes
(2026-09-18) -- see `_group_sub_item_ids` and `parse_group_items` below for
the passage-linked-group shape that was the actual cause of English being a
materially weaker subject than Maths/Science, and the positional-join
mechanism used to recover it. Measured effect: English Class 6 went from 4
items (all with an answer key) to 12 (all with an answer key); Class 9 from
31 items (20 with an answer key) to 77 (54 with an answer key). Not every
group is recoverable -- one Class 6 group (`English6SN2`) has 11 real,
separately-marked questions under only 10 declared sub-item ids (a 5(a)/5(b)
split that, in this one document, doesn't get its own id the way it does in
every other group observed), and is correctly refused rather than guessed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

log = logging.getLogger(__name__)

# "Maths6AS1", "Science10YP3", "English9JV4a", and -- importantly -- "English6SB"
# and "English8IN", which have NO trailing digit.
#
# The digit was required in the first version, so every id shaped
# `{Subject}{Class}{Letters}` was rejected and English Class 6 and Class 8
# parsed to 3 and 2 items against the 49+ each document holds. A parser that
# silently finds 6% of a document looks like a thin document, not a broken
# pattern, which is why the count is worth asserting rather than eyeballing.
# `[a-z]{0,3}` rather than `[a-z]?`: Science sub-question ids carry roman
# suffixes -- `Science9SRN41bi`, `...1bii`, `...1biii` -- and allowing a single
# lowercase letter rejected every one of them, which is a quiet slice of the
# Science corpus.
ITEM_ID = re.compile(r"^([A-Z][A-Za-z]+)(\d{1,2})([A-Z]{2,4}\d*[a-z]{0,3})$")

# The learning-ladder content reference. Two shapes exist, and both are real:
#   "6A1a", "10A2b", "9C1a"   -- Mathematics and English (digit, letter, digit)
#   "9.1.13", "10.3.9"        -- Science (dotted numeric)
# A pattern that only accepts the first silently yields 0% topic coverage on
# every Science document, which is exactly what happened before this was
# widened -- and 0% looks like "the source has no codes" rather than "the regex
# is wrong", so it is worth being explicit about.
#
# The second alternative is deliberately tight (`\d{1,2}[A-Z]\d{1,2}[a-z]?`).
# A looser version (`[A-Z]?\d{0,2}[A-Z]\d+[a-z]?`) matched fragments of ordinary
# words and produced codes that do not exist in the curriculum, which then
# overrode the authoritative index value.
CONTENT_CODE = re.compile(
    r"(?<![\w.])(\d{1,2}(?:\.\d{1,2}){2}|\d{1,2}[A-Z]\d{1,2}[a-z]?)(?![\w.])")

# Running furniture that must never become content.
_RUNNING_HEAD = re.compile(r"^\s*(www\.britishcouncil\.org|britishcouncil\.org)\s*$", re.I)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_CNE = re.compile(r"\b([CNE])\b")
_MARKS_LINE = re.compile(r"\((\d{1,2})\s*marks?\)", re.I)
_TOTAL_MARKS = re.compile(r"Total marks\s*(\d{1,2})", re.I)
_ITEM_PURPOSE = re.compile(r"Item purpose\s*(.+?)(?=\n\s*(?:Sources|Source|Question|Mark scheme)\b)",
                           re.S | re.I)
_OPTION = re.compile(r"^\s*([A-D])[.)]\s*(.+)$", re.M)
_ANSWER_ROW = re.compile(r"^\s*([A-D])[.)]\s*(.+)$", re.M)

# A mark-scheme entry inside a passage-linked GROUP (see _group_sub_item_ids
# below) is keyed by a bare sequential number, not an item id -- "1", "2",
# ... "5 (a)", "5 (b)". Anchored to line start; `re.M` in the fullmatch call
# site keeps this from firing on numbers embedded mid-sentence.
_SCHEME_ENTRY = re.compile(r"^\s*(\d{1,2})\s*(\([a-zA-Z]\))?\s+(?=\S)", re.M)

SUBJECT_BY_PREFIX = {
    "maths": "Mathematics", "mathematics": "Mathematics",
    "science": "Science", "english": "English",
}


@dataclass
class CbeItem:
    """One assessment item from a CBSE CBE item bank."""

    item_id: str
    question_id: str                 # the specific part, e.g. Maths6AS1b
    subject: str
    grade: int
    content_code: str = ""            # learning-ladder reference, e.g. "6A1a"
    content_reference: str = ""       # the human description of that code
    topic: str = ""                   # the named strand, e.g. "Algebra"
    ao1_marks: int = 0
    ao2_marks: int = 0
    # Only the English index carries AO3/AO4 columns; zero everywhere else.
    ao3_marks: int = 0
    ao4_marks: int = 0
    marks: int = 0
    calculator: str = ""              # "C" required, "N" not allowed, "E" either
    purpose: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)
    answer: str = ""                  # the official answer, from the mark scheme
    guidance: str = ""
    needs_figure: bool = False
    # Set when this item came from a passage-linked group that could not be
    # split. Its `marks` is then the GROUP's stated total, not this question's,
    # and `question` is at best the group's first sub-part -- so neither field
    # can be trusted. Recorded rather than silently carried.
    marks_unresolved: bool = False
    group_marks_stated: int = 0
    passage: str = ""                 # shared reading passage, for a linked-passage item
    source_pdf: str = ""
    pages: list[int] = field(default_factory=list)

    @property
    def has_answer_key(self) -> bool:
        """The hard gate. A question with no published answer is not a question.

        An examination board does not issue a question it cannot mark, and a
        bank that mixes answerable and unanswerable items silently degrades
        every paper built from it.
        """
        return bool(self.answer.strip())

    @property
    def ao_marks(self) -> dict[str, int]:
        out = {}
        if self.ao1_marks:
            out["AO1"] = self.ao1_marks
        if self.ao2_marks:
            out["AO2"] = self.ao2_marks
        if self.ao3_marks:
            out["AO3"] = self.ao3_marks
        if self.ao4_marks:
            out["AO4"] = self.ao4_marks
        return out

    def to_dict(self) -> dict:
        return {
            "itemId": self.item_id,
            "questionId": self.question_id,
            "subject": self.subject,
            "grade": self.grade,
            "contentCode": self.content_code,
            "contentReference": self.content_reference,
            "topic": self.topic,
            "aoMarks": self.ao_marks,
            "marks": self.marks,
            "calculator": self.calculator,
            "purpose": self.purpose,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "guidance": self.guidance,
            "hasAnswerKey": self.has_answer_key,
            "needsFigure": self.needs_figure,
            "marksUnresolved": self.marks_unresolved,
            "groupMarksStated": self.group_marks_stated,
            "passage": self.passage,
            "sourcePdf": self.source_pdf,
            "pages": self.pages,
        }


# --------------------------------------------------------------------------- #
# page-level text
# --------------------------------------------------------------------------- #

def _clean_page(text: str) -> list[str]:
    """Drop running furniture, return meaningful lines."""
    out: list[str] = []
    for line in text.splitlines():
        if _RUNNING_HEAD.match(line) or _PAGE_NUMBER.match(line):
            continue
        out.append(line.rstrip())
    # drop leading/trailing blanks
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _page_starts_item(lines: list[str]) -> str | None:
    """An item begins when the first meaningful line is an item identity alone.

    The item markers are required as well as the identity. Without them the
    Item Index pages qualify: their rows also begin with an identifier, so
    every index row was emitted as a question whose text was the rest of the
    table -- which is how the English documents produced items with no
    question and no answer.
    """
    if not lines:
        return None
    first = lines[0].strip()
    if not ITEM_ID.match(first):
        return None
    joined = " ".join(lines[:12]).lower()
    if not any(marker in joined for marker in
               ("item identity", "item purpose", "mark scheme", "question")):
        return None
    return first


def _stream(doc) -> list[tuple[int, str]]:
    """The whole document as `(page_number, line)` pairs.

    Items do NOT align to page boundaries: a 1-mark Mathematics item occupies a
    few lines, so several share a page, and a long English item spans ten. A
    page-based split therefore misses every item that does not happen to start
    at the top of a page -- which is how Maths Class 6 yielded 14 items against
    the 58 the document states, and English Class 6 yielded 5 against 46.
    """
    out: list[tuple[int, str]] = []
    for index in range(len(doc)):
        for line in _clean_page(doc[index].get_text()):
            out.append((index + 1, line))
    return out


def _is_metadata_header(stream: list[tuple[int, str]], i: int) -> bool:
    """True when position `i` opens an item's metadata table.

    The header is `Item identity`, but the PDF splits it across lines in some
    documents (`Item` / `identity`) and keeps it on one line in others. Both
    are treated as the boundary, because every item has it and nothing else
    does -- which makes it a far more reliable anchor than the item id, since
    the Item Index also contains item ids but no metadata header.
    """
    if i >= len(stream):
        return False
    line = stream[i][1].strip().lower()
    if line == "item identity":
        return True
    if line != "item":
        return False
    for j in range(i + 1, min(i + 4, len(stream))):
        nxt = stream[j][1].strip().lower()
        if not nxt:
            continue
        return nxt == "identity"
    return False


def _lookahead_nonblank(stream: list[tuple[int, str]], i: int, n: int) -> list[str]:
    out: list[str] = []
    for j in range(i + 1, min(i + 20, len(stream))):
        t = stream[j][1].strip()
        if t:
            out.append(t)
            if len(out) >= n:
                break
    return out


def _starts_item(stream: list[tuple[int, str]], i: int) -> bool:
    """Is position `i` the first line of an item?

    Two anchors, because the documents are not internally consistent:

      * **A content code follows.** This is the Mathematics and Science shape.
        A 1-mark item is four lines long and its metadata table may omit the
        `Item identity` header entirely, going straight to
        `Maths6AS1 / Maths6AS1 / 6A1a / Algebra`. Anchoring on the header alone
        found 10 of Maths Class 6's stated 58 items.
      * **An `Item identity` header follows.** This is the English shape, where
        the id is followed by the full metadata table.

    Accepting either recovered both, and the id-line is required in both cases
    so the Item Index -- which also contains item ids -- is still excluded, since
    an index row is id/id/marks with no code and no header after it.
    """
    ident = stream[i][1].strip()
    if not ITEM_ID.match(ident):
        return False

    look = _lookahead_nonblank(stream, i, 6)
    if not look:
        return False

    if any(CONTENT_CODE.fullmatch(t) or CONTENT_CODE.match(t) and len(t) < 14
           for t in look[:4]):
        return True

    joined = " ".join(look).lower()
    if "item identity" in joined or "identity" in joined:
        return True
    return False


def split_items(pdf_path: Path) -> Iterator[tuple[str, list[int], str]]:
    """Yield `(item_id, page_numbers, text)` for each item in one document."""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        stream = _stream(doc)

        # Item ids are not unique lines -- an id appears at a page top, again in
        # its own metadata table, and its sub-question ids follow. Only lines
        # that open an item are accepted, then consecutive repeats collapse.
        starts: list[tuple[int, str]] = []
        for i in range(len(stream)):
            if _starts_item(stream, i):
                ident = stream[i][1].strip()
                if starts and starts[-1][1] == ident:
                    continue          # same item, seen at the page top
                starts.append((i, ident))

        for n, (pos, ident) in enumerate(starts):
            end = starts[n + 1][0] if n + 1 < len(starts) else len(stream)
            body = "\n".join(line for _page, line in stream[pos + 1:end])
            pages = sorted({page for page, _line in stream[pos:end]})
            yield ident, pages, body
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# field extraction
# --------------------------------------------------------------------------- #

def _extract_metadata(block: str) -> dict:
    """Pull the metadata table fields out of an item block.

    The PDF emits the table cells in a different order from the visual layout,
    so each field is found by its own shape rather than by position. The
    content reference is located by the code pattern, which is the only field
    in the table with a distinctive form.
    """
    out: dict = {"content_code": "", "content_reference": "", "marks": 0,
                 "ao1_marks": 0, "ao2_marks": 0, "calculator": ""}

    # The block opens with the identity, then "Item identity", then the AO
    # marks. Everything before "Item purpose" is the table.
    head = block.split("Item purpose", 1)[0]

    m = CONTENT_CODE.search(head)
    if m:
        out["content_code"] = m.group(1)
        # The description runs from just after the code up to the next
        # standalone integer (the Marks cell) or the end of the table.
        rest = head[m.end():]
        desc = re.split(r"\n\s*\d{1,2}\s*\n", rest, maxsplit=1)[0]
        desc = " ".join(desc.split())
        out["content_reference"] = desc.strip(" .;*")[:300]

    total = _TOTAL_MARKS.search(block)
    if total:
        out["marks"] = int(total.group(1))
    else:
        per = _MARKS_LINE.search(block)
        if per:
            out["marks"] = int(per.group(1))

    # AO marks: the table lists AO1 and AO2 columns before the content
    # reference. Their values are small integers on their own lines.
    aomatch = re.search(r"Item\s*identity\s*(.*?)(?=[CEN]\b|Content reference|"
                        r"reference\(s\)|reference from)", head, re.S | re.I)
    if aomatch:
        nums = [int(n) for n in re.findall(r"^\s*(\d{1,2})\s*$", aomatch.group(1), re.M)]
        if nums:
            out["ao1_marks"] = nums[0]
        if len(nums) > 1:
            out["ao2_marks"] = nums[1]

    cne = re.search(r"\b([CNE])\b\s*(?:Content|reference)", head)
    if cne:
        out["calculator"] = cne.group(1)
    return out


def _extract_question_and_answer(block: str) -> tuple[str, list[str], str, str, bool]:
    """Split the block into `(question, options, answer, guidance, needs_figure)`."""
    qsplit = re.split(r"\n\s*Question(?:\(s\))?\s*\n", block, maxsplit=1, flags=re.I)
    body = qsplit[1] if len(qsplit) > 1 else block
    msplit = re.split(r"\n\s*Mark scheme\s*\n", body, maxsplit=1, flags=re.I)
    question_zone = msplit[0]
    scheme_zone = msplit[1] if len(msplit) > 1 else ""

    options = [f"{m.group(1)}. {m.group(2).strip()}"
               for m in _OPTION.finditer(question_zone)]
    question = _OPTION.split(question_zone)[0]
    question = " ".join(question.split())
    # Drop the trailing mark annotation; marks are carried as their own field.
    question = _MARKS_LINE.sub("", question).replace("(Total marks", "").strip()

    answer, guidance = "", ""
    if scheme_zone:
        # The mark scheme restates the question, then an "Answer | Guidance"
        # table. In flattened text the answer is the option letter followed by
        # its text, or a short answer string before the guidance.
        after = re.split(r"\n\s*Answer\s*\n\s*Guidance\s*\n", scheme_zone, maxsplit=1)
        if len(after) > 1:
            tail = after[1]
            am = _ANSWER_ROW.search(tail)
            if am:
                answer = f"{am.group(1)}. {am.group(2).strip()}"
            else:
                answer = " ".join(tail.split()[:40])
            guidance = " ".join(tail.split())[len(answer):].strip()[:400]
        else:
            # "Answer" and "Guidance" may be on one line or reordered.
            flat = " ".join(scheme_zone.split())
            gm = re.search(r"Answer\s+Guidance\s+(.*)$", flat)
            if gm:
                tail = gm.group(1)
                am = _ANSWER_ROW.match(tail.strip())
                answer = (f"{am.group(1)}. {am.group(2).strip()}" if am
                          else " ".join(tail.split()[:40]))
                guidance = " ".join(tail.split())[len(answer):].strip()[:400]

    figure_phrases = ("shown in the figure", "in the given figure", "following diagram",
                      "the diagram below", "figure below", "shown below", "given figure")
    low = question.lower()
    options_text = " ".join(o.lower() for o in options)
    needs_figure = any(p in low for p in figure_phrases) and not options
    return question, options, answer, guidance, needs_figure


# --------------------------------------------------------------------------- #
# passage-linked groups (2026-09-18) -- English's weak-subject cause
# --------------------------------------------------------------------------- #
#
# English's reading-comprehension items are not self-contained the way a
# Mathematics or Science item is. One shared passage carries several
# sub-questions, e.g. English6SB1 through English6SB9, and the item-identity
# table lists them as a bare run of ids with no marks/content-code attached
# to any of them individually -- unlike a Maths/Science multi-part item
# (Maths6MG4a/Maths6MG4b), where each sub-item's own AO flag and content code
# immediately follow it, which is exactly what the existing `_starts_item`
# lookahead already recognises. Split_items() therefore correctly finds
# Maths/Science sub-items as separate items, and just as correctly finds an
# English passage group as ONE giant blob -- there is nothing in the bare id
# list for the existing per-item anchors to catch.
#
# The questions and their answers are recoverable, but only by POSITION, not
# by key: the `Question(s)` zone holds N questions in order, each closed by
# its own "(N mark(s))" annotation (the same marker used everywhere else in
# this module); the `Mark scheme` zone holds N answers in the SAME order,
# keyed by a bare sequential number (1, 2, 3, ... 5 (a), 5 (b)) rather than
# the item id. This module's rule elsewhere is "never pair a question with an
# answer it cannot key exactly" -- there is no exact key here, so the
# positional join is only trusted when the sub-item-id count, the parsed
# question count, and the parsed scheme-entry count all agree exactly.
# Disagreement means refuse the whole group rather than guess an alignment,
# consistent with that rule's intent even though the mechanism differs.

def _group_sub_item_ids(block: str) -> list[str]:
    """The sub-item ids sharing one passage, if this block is a passage-
    linked GROUP. Returns [] for an ordinary single (or Maths/Science
    multi-part) item, which must not be routed through this path.

    The discriminator: two ITEM_ID lines in a row, with nothing but blank
    lines between them, is only possible in English's bare-list shape --
    Maths/Science always has real content (a C/N/E flag, then a content
    code) immediately after each sub-item's own id.
    """
    head = block.split("Item purpose", 1)[0]
    lines = [l.strip() for l in head.split("\n")]

    start = None
    for i, l in enumerate(lines):
        if l.lower() == "item identity":
            start = i
            break
        if l.lower() == "item" and i + 1 < len(lines) and lines[i + 1].lower() == "identity":
            start = i + 1
            break
    if start is None:
        return []

    tail = lines[start + 1:]
    id_positions = [i for i, l in enumerate(tail) if ITEM_ID.match(l)]
    if len(id_positions) < 2:
        return []

    ids: list[str] = []
    for pos in id_positions:
        k = pos + 1
        while k < len(tail) and not tail[k]:
            k += 1
        nxt = tail[k] if k < len(tail) else ""
        if not (ITEM_ID.match(nxt) or nxt.lower() == "total marks"):
            # This id has real inline content (a flag, a code) -- the
            # Maths/Science shape, already handled correctly elsewhere.
            return []
        ids.append(tail[pos])
    return ids


def _extract_group_passage(block: str) -> str:
    """The shared reading passage, so each split-out sub-question is still
    answerable on its own once served -- a linked-passage question with no
    passage is exactly as unanswerable as a figure-dependent one with no
    figure (see needs_figure above)."""
    m = re.search(r"Source\(s\)\s*:?\s*\n(.*?)\n\s*Question(?:\(s\))?\s*\n",
                 block, re.S | re.I)
    if not m:
        return ""
    return " ".join(m.group(1).split())[:4000]


def _split_group_questions(question_zone: str) -> list[tuple[str, list[str], int]]:
    """One `(question_text, options, marks)` triple per sub-item, in document
    order.

    Each question in a group's shared `Question(s)` zone ends with its own
    "(N mark(s))" annotation -- the same _MARKS_LINE marker already used to
    strip a single item's trailing mark note. Used here as a right boundary
    instead: whatever text precedes it, back to the previous boundary, is one
    sub-item's full question.
    """
    marks_matches = list(_MARKS_LINE.finditer(question_zone))
    result: list[tuple[str, list[str], int]] = []
    pos = 0
    for m in marks_matches:
        chunk = _MARKS_LINE.sub("", question_zone[pos:m.end()])
        options = [f"{om.group(1)}. {om.group(2).strip()}" for om in _OPTION.finditer(chunk)]
        stem = _OPTION.split(chunk)[0]
        result.append((" ".join(stem.split()), options, int(m.group(1))))
        pos = m.end()
    return result


def _split_group_scheme(scheme_zone: str) -> list[str]:
    """One answer string per sub-item, in document order, keyed by the bare
    sequential numbering CBSE's own layout uses here (see module note above)
    -- NOT the item id, which does not appear in this zone at all."""
    matches = list(_SCHEME_ENTRY.finditer(scheme_zone))
    answers: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(scheme_zone)
        chunk = scheme_zone[m.end():end]
        am = re.search(r"Answer\s*\n?\s*Guidance\s*\n?\s*(.*)", chunk, re.S | re.I)
        answers.append(" ".join(am.group(1).split())[:400] if am else "")
    return answers


def parse_group_items(sub_ids: list[str], pages: list[int], block: str, *,
                      subject: str, grade: int, source_pdf: str) -> list["CbeItem"]:
    """Split one passage-linked group block into its real sub-items.

    Returns [] rather than a partial or misaligned list if the sub-item
    count, the parsed question count and the parsed scheme-entry count do
    not all agree -- see the module note above for why a positional join is
    only trusted under that exact agreement.
    """
    # A block from split_items() can span more than one real group: found
    # live 2026-09-18 on English6IN2, which immediately follows English6SB
    # with no "Item identity" sub-id table of its own that this module's
    # anchors can find, so its entire body -- including its OWN
    # Question(s)/Mark scheme cycle, restarting the same 1-2-3... numbering
    # -- lands inside SB's block. Without truncating, SB's scheme_zone ran
    # on into IN2's, doubling every entry count. The group's own real
    # boundary is its second "Item purpose": the first is this group's own,
    # a second means a different group's content has bled in after it.
    purpose_positions = [m.start() for m in re.finditer(r"\n\s*Item purpose\s*\n", block, re.I)]
    if len(purpose_positions) >= 2:
        block = block[:purpose_positions[1]]

    question_markers = list(re.finditer(r"\n\s*Question(?:\(s\))?\s*\n", block, re.I))
    scheme_markers = list(re.finditer(r"\n\s*Mark scheme\s*\n", block, re.I))
    if not question_markers or not scheme_markers:
        return []

    if len(question_markers) == 1:
        # SB-style: one shared "Question(s)" zone and one shared "Mark
        # scheme" zone, each internally holding all N sub-items -- split
        # positionally within each zone (_split_group_questions/_scheme).
        question_zone = block[question_markers[0].end():scheme_markers[0].start()]
        scheme_zone = block[scheme_markers[0].end():]
        questions = _split_group_questions(question_zone)
        answers = _split_group_scheme(scheme_zone)
    else:
        # SN2-style: N independent "Question" / "Mark scheme" cycles, one
        # sub-item per cycle, each already isolated by the document itself --
        # found live 2026-09-18, a real second layout CBSE uses for the same
        # passage-linked-group concept. Paired by simple adjacency: the
        # scheme marker immediately following each question marker is that
        # question's own answer.
        questions, answers = [], []
        for i, qm in enumerate(question_markers):
            sm = next((s for s in scheme_markers if s.start() > qm.end()), None)
            if sm is None:
                break
            q_text = block[qm.end():sm.start()]
            marks_m = _MARKS_LINE.search(q_text)
            marks_val = int(marks_m.group(1)) if marks_m else 0
            q_text = _MARKS_LINE.sub("", q_text)
            q_text = re.sub(r"\(Total marks?\s*\d+\)", "", q_text, flags=re.I)
            options = [f"{om.group(1)}. {om.group(2).strip()}" for om in _OPTION.finditer(q_text)]
            stem = _OPTION.split(q_text)[0]
            questions.append((" ".join(stem.split()), options, marks_val))

            next_q = question_markers[i + 1].start() if i + 1 < len(question_markers) else len(block)
            scheme_chunk = block[sm.end():next_q]
            am = re.search(r"Answer\s*\n?\s*Guidance\s*\n?\s*(.*)", scheme_chunk, re.S | re.I)
            answers.append(" ".join(am.group(1).split())[:400] if am else "")

    if not (len(sub_ids) == len(questions) == len(answers)):
        log.warning(
            "passage group %s: %d sub-items but %d questions and %d scheme "
            "entries -- refusing to guess an alignment",
            sub_ids[0] if sub_ids else "?", len(sub_ids), len(questions), len(answers))
        return []

    passage = _extract_group_passage(block)
    purpose_m = _ITEM_PURPOSE.search(block)
    purpose = " ".join(purpose_m.group(1).split())[:300] if purpose_m else ""

    items = []
    for sub_id, (question, options, marks), answer in zip(sub_ids, questions, answers):
        items.append(CbeItem(
            item_id=sub_id, question_id=sub_id, subject=subject, grade=grade,
            marks=marks, purpose=purpose, question=question, options=options,
            answer=answer, passage=passage, source_pdf=source_pdf, pages=pages,
        ))
    return items


def parse_item(item_id: str, pages: list[int], block: str, *, subject: str,
               grade: int, source_pdf: str) -> CbeItem:
    meta = _extract_metadata(block)
    question, options, answer, guidance, needs_figure = _extract_question_and_answer(block)
    purpose_m = _ITEM_PURPOSE.search(block)
    return CbeItem(
        item_id=item_id,
        question_id=item_id,
        subject=subject,
        grade=grade,
        content_code=meta["content_code"],
        content_reference=meta["content_reference"],
        ao1_marks=meta["ao1_marks"],
        ao2_marks=meta["ao2_marks"],
        marks=meta["marks"],
        calculator=meta["calculator"],
        purpose=" ".join(purpose_m.group(1).split())[:300] if purpose_m else "",
        question=question,
        options=options,
        answer=answer,
        guidance=guidance,
        needs_figure=needs_figure,
        source_pdf=source_pdf,
        pages=pages,
    )


# --------------------------------------------------------------------------- #
# the topic index
# --------------------------------------------------------------------------- #

# The index table emits one cell per line, not one row per line:
#
#     6A1a
#     Algebra
#     Maths6AS1
#     Maths6AS1
#     1
#
# so a single-line row regex matches nothing. The rows are reconstructed by
# scanning for a content code and taking the following cells positionally.
_CELL = re.compile(r"^\s*(.+?)\s*$")
# Widths match ITEM_ID: a trailing digit is optional ("English6SB" is real).
_CODE_CELL = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){2}|\d{0,2}[A-Z]\d{1,2}[a-z]?)$")
_QID_CELL = re.compile(r"^[A-Za-z]{3,}\d{1,2}[A-Z]{2,4}\d*(?:[a-z]|\d)?$")


# The English Item Index carries no content-code column -- every other subject's
# does. It gives a "Text type" instead ("Poetry", "Speech transcript"), so the
# strand is synthesised from that and prefixed `ENG` so it can never be mistaken
# for a code CBSE actually issued.
_ENGLISH_STRANDS = (
    (("poetry", "poem"), 1),
    (("drama", "play"), 2),
    (("autobiography", "biography", "diary", "memoir"), 3),
    (("travel",), 4),
    # "informative" is listed separately because it is not a substring of
    # "information": every "Informative newspaper/magazine article" fell to the
    # `.0` fallback without it -- 53 rows across classes 6-8.
    (("information", "informative", "speech", "transcript"), 5),
    (("advertisement", "advert"), 6),
    (("non-fiction", "nonfiction"), 7),
    (("prose", "short story", "novel", "fiction"), 8),
)


def _english_pseudo_code(grade: int, text_type: str) -> str:
    """A per-grade strand code for an English item -- `ENG9.1` for class 9 poetry.

    The grade is part of the code because `coarse_code` keeps only the part
    before the dot: without it, class 6 poetry and class 10 poetry collapse into
    one strand and the topic mapper can map an item onto another grade's topic.

    An unrecognised text type falls back to `.0` rather than to no code at all.
    It is still an English item of a known grade, and dropping it would shrink
    coverage silently.
    """
    lowered = (text_type or "").lower()
    for keywords, idx in _ENGLISH_STRANDS:
        if any(k in lowered for k in keywords):
            return f"ENG{int(grade)}.{idx}"
    return f"ENG{int(grade)}.0"


def _cell(row: list, i: int) -> str:
    """One extracted table cell as single-line text, or "" when absent."""
    if len(row) <= i or row[i] is None:
        return ""
    return str(row[i]).strip().replace("\n", " ")


def _int_cell(row: list, i: int) -> int:
    """A table cell as an int, or 0 when it is blank or not a number."""
    raw = _cell(row, i)
    return int(raw) if raw.isdigit() else 0


# Header labels of the English index, mapped to the names used below. The
# layout is NOT fixed: Class 6 omits AO3, so its table is seven columns wide
# where Classes 7-10 are eight. Reading a hardcoded column index therefore ran
# one place right on Class 6 and took `Source description` -- a sentence of
# prose -- as the text type.
_ENGLISH_COLUMNS = {
    "question id": "qid",
    "text type": "text_type",
    "source description": "source",
    "ao1": "ao1",
    "ao2": "ao2",
    "ao3": "ao3",
    "ao4": "ao4",
}


def _english_header_map(row: list) -> dict[str, int] | None:
    """Column name -> position, read off an English index header row.

    Returns None when this is not that header, which is also how a continuation
    page (the same table repeated with no header) is recognised.
    """
    cells = [" ".join(_cell(row, i).lower().split()) for i in range(len(row))]
    if "question id" not in cells:
        return None
    out: dict[str, int] = {}
    for i, name in enumerate(cells):
        key = _ENGLISH_COLUMNS.get(name)
        if key and key not in out:
            out[key] = i
    return out if {"qid", "text_type"} <= set(out) else None


def _ao_cell(row: list, cols: dict[str, int], key: str) -> int:
    """One AO column's marks, or 0 when this layout has no such column."""
    i = cols.get(key)
    return _int_cell(row, i) if i is not None else 0


def _parse_english_index(pdf_path: Path, grade: int) -> dict[str, dict]:
    """The English Item Index: question id -> strand, text type, AO marks.

    English ships a different layout from the other subjects -- no content-code
    column, and a `Text type` label instead -- so it is read with table
    extraction rather than the line-per-cell scan `parse_index` uses elsewhere.

    Three things about this table are not obvious and each one silently lost
    rows before it was handled:

      * `Text type` and `Source description` are VERTICALLY MERGED cells, one
        per passage. The extractor returns their text on the first row of a
        passage group and "" for every sibling question beneath it, so
        requiring a non-empty text type kept only one question per passage --
        65 of 350 rows across the five banks. Both are carried down instead,
        and across a page break, because a group can span one.
      * The column positions vary by grade (Class 6 has no AO3), so they are
        resolved from the header rather than hardcoded.
      * Continuation pages repeat the table with no header. Judging one by its
        first row alone threw away a whole page when that row's id did not
        match -- 18 valid rows on Class 9 page 6 -- so the page is accepted if
        it holds ANY id row.
    """
    import fitz

    doc = fitz.open(pdf_path)
    out: dict[str, dict] = {}
    cols: dict[str, int] | None = None
    carried_text_type = carried_source = ""
    try:
        for index in range(min(14, len(doc))):
            for table in doc[index].find_tables():
                rows = table.extract()
                if not rows:
                    continue
                header = _english_header_map(rows[0])
                if header:
                    cols, carried_text_type, carried_source = header, "", ""
                    data_rows = rows[1:]
                elif cols is not None and any(
                        _QID_CELL.match(_cell(r, cols["qid"])) for r in rows if r):
                    data_rows = rows              # continuation page, no header
                else:
                    continue

                for row in data_rows:
                    if not row:
                        continue
                    qid = _cell(row, cols["qid"])
                    if not qid or not _QID_CELL.match(qid):
                        continue
                    # Both passage-level columns are merged cells, so both are
                    # carried down to the sibling questions of a passage.
                    carried_text_type = _cell(row, cols["text_type"]) or carried_text_type
                    if "source" in cols:
                        carried_source = _cell(row, cols["source"]) or carried_source
                    out[qid] = {
                        "content_code": _english_pseudo_code(grade, carried_text_type),
                        "topic": carried_text_type,
                        "content_reference": carried_source,
                        "ao1_marks": _ao_cell(row, cols, "ao1"),
                        "ao2_marks": _ao_cell(row, cols, "ao2"),
                        "ao3_marks": _ao_cell(row, cols, "ao3"),
                        "ao4_marks": _ao_cell(row, cols, "ao4"),
                    }
    finally:
        doc.close()
    return out


def parse_index(pdf_path: Path) -> dict[str, dict]:
    """Read the Item Index: question id -> content code, topic strand, AO marks.

    This is the topic mapping the requirement asks for, and it is CBSE's own --
    not a keyword guess. The index gives the *strand* (`Algebra`, `Living
    world`), while each item's metadata gives the fuller content reference.
    """
    import fitz

    # English's index has a different layout and no content code; it needs the
    # grade, because its strand codes are synthesised per grade.
    subject, grade = subject_and_grade_for(pdf_path)
    if subject.lower() == "english":
        return _parse_english_index(pdf_path, grade)

    doc = fitz.open(pdf_path)
    out: dict[str, dict] = {}
    try:
        for index in range(min(14, len(doc))):
            lines = [l.strip() for l in _clean_page(doc[index].get_text())]
            i = 0
            while i < len(lines):
                if not _CODE_CELL.match(lines[i]):
                    i += 1
                    continue
                # code, topic, filename, question id, ao1, ao2 -- with the
                # optionals possibly absent, so walk forward rather than
                # indexing blindly.
                code = lines[i]
                j = i + 1
                topic = lines[j] if j < len(lines) else ""
                j += 1
                # skip the filename cell if the next cell is also an id
                qid = ""
                if j < len(lines) and _QID_CELL.match(lines[j]):
                    qid = lines[j]
                    j += 1
                    if j < len(lines) and _QID_CELL.match(lines[j]):
                        qid = lines[j]
                        j += 1
                if not qid or not topic or _CODE_CELL.match(topic):
                    i += 1
                    continue
                ao1 = ao2 = 0
                # `.isdigit()` is True for characters `int()` rejects (e.g. the
                # superscript in x²), so this is an ASCII check.
                def _is_int(s: str) -> bool:
                    return s.isascii() and s.isdigit()
                if j < len(lines) and _is_int(lines[j]):
                    ao1 = int(lines[j]); j += 1
                if j < len(lines) and _is_int(lines[j]):
                    ao2 = int(lines[j]); j += 1
                out[qid] = {
                    "content_code": code,
                    "topic": " ".join(topic.split()),
                    "ao1_marks": ao1,
                    "ao2_marks": ao2,
                }
                i = j
    finally:
        doc.close()
    return out


def subject_and_grade_for(path: Path) -> tuple[str, int]:
    m = re.search(r"Item-Bank-+([A-Za-z]+)-+Class-?(\d+)", path.name, re.I)
    if not m:
        return "", 0
    raw = m.group(1).lower()
    return SUBJECT_BY_PREFIX.get(raw, m.group(1).title()), int(m.group(2))


# --------------------------------------------------------------------------- #
# converting an item into a question-bank record
# --------------------------------------------------------------------------- #

# CBE items do not print a Bloom level; they print assessment objectives, which
# measure a different thing. Inferring Bloom from AO would be inventing data, so
# every imported item is marked `understand` -- the neutral default the rest of
# the bank uses -- and carries its real AO split alongside.
_DEFAULT_BLOOM = "understand"


def _difficulty_for(marks: int) -> str:
    """Marks are the only difficulty signal these documents carry.

    Same rule `assessment/mapping.py` uses for the board-paper corpus, so the
    two sources are comparable rather than each inventing a scale.
    """
    return "easy" if marks <= 2 else "medium" if marks <= 3 else "hard"


def _question_type_for(item: "CbeItem") -> str:
    if item.options:
        return "mcq"
    return {1: "very_short_answer", 2: "short_answer"}.get(item.marks, "long_answer")


def to_bank_record(item: "CbeItem") -> dict:
    """Convert a parsed CBE item into the question-bank wire shape.

    Every field the bank already uses is populated, plus the four the CBE
    source adds and the rest of the bank does not have: `contentCode`,
    `contentReference`, `topic` and `aoMarks`. Those are the topic mapping and
    the marks-by-assessment-objective split, and dropping them on import would
    throw away the most valuable thing this source provides.

    The answer scheme carries `provenance: "cbse_marking_scheme"` because that is
    literally what it is -- the scheme CBSE published for this item. It is the
    only source in the bank that can say that, and the API's `hasScheme` filter
    depends on the distinction.
    """
    qid = f"cbe:q:{item.item_id}"
    marks = item.marks or 0
    # A passage group that could not be split has no knowable per-question mark
    # value, so it is emitted with marks=0. `scripts/build_cbe_bank.py`'s existing
    # marks gate then excludes it and lists it in `_unanswerable.json` alongside
    # the items with no answer key -- no second exclusion mechanism, and the
    # stated group total survives in `metadata.groupMarksStated` rather than
    # being lost or passed off as this question's weight.
    if item.marks_unresolved:
        marks = 0
    points = []
    if item.answer:
        points.append({
            "id": f"{qid}:mp1",
            "description": item.answer,
            "marks": marks,
            "keyword": "",
            "isRequired": True,
            "synonyms": [],
        })

    metadata = {
        "itemId": item.item_id,
        "needsFigure": item.needs_figure,
        "marksUnresolved": item.marks_unresolved,
        "groupMarksStated": item.group_marks_stated,
        "contentReference": item.content_reference,
        "aoMarks": item.ao_marks,
    }
    # The shared reading passage for a linked-passage item. The parser
    # extracts it (60 of 151 English class 6-9 items carry one) and this
    # record used to drop it, so an English comprehension question ("Give two
    # symptoms of the writer's fear as she went diving") reached the bank
    # answerable by no one. Omitted, not emptied, when there is none.
    passage = (item.passage or "").strip()
    if passage:
        metadata["passage"] = passage

    return {
        "id": qid,
        "questionBankId": f"cbe:{item.source_pdf}",
        "subject": item.subject,
        "grade": item.grade,
        "chapterIds": [item.content_code] if item.content_code else [],
        # The named strand, when the index supplied one. `topic` is the field
        # the MCP topic tools read.
        "topic": item.topic or "",
        "contentCode": item.content_code,
        "contentReference": item.content_reference,
        "aoMarks": item.ao_marks,
        "calculator": item.calculator,
        "competencyIds": [],
        "bloomLevel": _DEFAULT_BLOOM,
        "difficulty": _difficulty_for(marks),
        "type": _question_type_for(item),
        "stem": item.question,
        "stemLatex": "",
        "parts": [{"id": f"{qid}:p1", "partNumber": 1, "text": o, "marks": 0,
                   "answerType": "singleChoice", "alternativeAnswers": []}
                  for o in item.options],
        "answerScheme": {
            "totalMarks": marks,
            # An MCQ's mark scheme is the correct option, so it is a marking
            # point like any other. Without this the record's only answer
            # content was an empty `modelAnswer`.
            "markingPoints": (points or (
                [{"text": item.answer, "marks": marks}] if item.answer else [])),
            "rubricLevels": [],
            "commonErrors": [],
            "alternativeAnswers": [],
            # ALWAYS the answer. This was `"" if item.options else item.answer`,
            # which discarded the correct option for every multiple-choice
            # question: `parts` lists the options but nothing said which one was
            # right, so the record had no answer content at all while its
            # `provenance` still claimed "cbse_marking_scheme". That is a false
            # attribution -- a question that reads as answerable and is not --
            # and it affected 259 served records. The option is printed for an
            # MCQ precisely because that IS the answer.
            "modelAnswer": item.answer,
            "modelAnswerLatex": "",
            "hasPartialCredit": False,
            "metadata": {"guidance": item.guidance, "purpose": item.purpose,
                         "calculator": item.calculator, "aoMarks": item.ao_marks},
            "provenance": (
                "cbse_marking_scheme"
                if (item.answer or "").strip() else "none"),
            "sourcePaperCode": "",
            "sourceDocumentId": f"cbe:{item.source_pdf}",
        },
        "estimatedTimeMinutes": max(1, marks * 2),
        "marks": marks,
        "language": "en",
        "source": "cbse_question_bank",
        "qualityScore": 0.95,          # board-published, not extracted
        "tags": [t for t in (item.topic, item.content_code) if t],
        "createdAt": "2026-09-17T00:00:00Z",
        "updatedAt": "2026-09-17T00:00:00Z",
        "metadata": metadata,
        "diagramAssetId": None, "mapAssetId": None,
        "graphAssetId": None, "tableAssetId": None,
        # An item whose stem refers to a figure that was not extracted is
        # unanswerable as printed. Flagged rather than silently exported, and
        # `assessment/pool.py` excludes these from paper generation.
        "rights": {
            "origin": "CBSE",
            "redistribution": "unknown",
            "basis": "",
            "attribution": ("Central Board of Secondary Education, "
                            "CBE item bank (CBSE / British Council)"),
        },
        "provenance": {
            "sourceDocumentId": f"cbe:{item.source_pdf}",
            "pageNumber": item.pages[0] if item.pages else None,
            "boundingBox": None,
            "method": "pdf_native",
            "confidence": 0.95,
        },
        "calibration": {"measured": False, "sampleSize": 0, "facility": None,
                        "discrimination": None, "basis": "assigned"},
        "version": 1,
        "reviewState": "published",
        "supersededBy": None,
    }


def parse_all(directory: Path) -> tuple[list["CbeItem"], list[str]]:
    """Parse every item bank in a directory. Returns `(items, errors)`."""
    items: list[CbeItem] = []
    errors: list[str] = []
    for pdf in sorted(Path(directory).glob("*.pdf")):
        try:
            items.extend(parse_item_bank(pdf))
        except Exception as exc:                               # noqa: BLE001
            errors.append(f"{pdf.name}: {type(exc).__name__}: {exc}")
    return items, errors


def parse_item_bank(pdf_path: Path) -> list[CbeItem]:
    """Parse one item-bank PDF end to end.

    Anchoring on either the content code or the `Item identity` header finds
    every real item, and also fragments: an id line inside an item's own
    mark scheme, or a sub-question id, can satisfy the same test. The result was
    2,760 candidates against 893 items the documents state.

    So candidates are collapsed by id, keeping the best-scoring one. "Best" is
    deliberately not "first": a fragment can precede the real item, and a
    fragment rarely carries both a question and a mark scheme. Scoring on those
    two fields picks the real one and discards the rest, which recovered the
    counts without needing the anchors to be perfect.
    """
    subject, grade = subject_and_grade_for(pdf_path)
    index = parse_index(pdf_path)

    def _score(item: CbeItem) -> int:
        return ((1 if item.question else 0) + (2 if item.has_answer_key else 0)
                + (1 if item.marks else 0) + (1 if item.content_code else 0))

    best: dict[str, tuple[int, CbeItem]] = {}
    for item_id, pages, block in split_items(pdf_path):
        sub_ids = _group_sub_item_ids(block)
        if sub_ids:
            # A passage-linked group (see the note above parse_group_items).
            group_items = parse_group_items(sub_ids, pages, block, subject=subject,
                                            grade=grade, source_pdf=pdf_path.name)
            if group_items:
                # The bare group blob itself is never a real answerable item,
                # so it is not also parsed via the single-item path below --
                # doing both would leave a harmless-looking but meaningless
                # extra "English6SB" entry with no coherent single question,
                # on top of its real sub-items.
                for sub_item in group_items:
                    current = best.get(sub_item.item_id)
                    score = _score(sub_item)
                    if current is None or score > current[0]:
                        best[sub_item.item_id] = (score, sub_item)
                continue
            # Real regression found 2026-09-18: a block can look like a
            # bare-list group (_group_sub_item_ids > 0) without actually
            # being one that parse_group_items can safely split (e.g. its
            # sub-item count doesn't match the real question/scheme count).
            # Unconditionally skipping the single-item fallback here dropped
            # Class 7/8 items that were already parsing correctly the old
            # way -- falling through instead of `continue`-ing keeps that
            # working path as the safety net it always was.

        item = parse_item(item_id, pages, block, subject=subject, grade=grade,
                          source_pdf=pdf_path.name)
        if sub_ids:
            # We only reach here when `parse_group_items` REFUSED to split this
            # block (its own count check failed), so this single item is a whole
            # passage group wearing one question's clothes. Its `marks` is the
            # group total and its `question` is the group's first sub-part.
            #
            # Measured 2026-09-18: this produced English items stamped 14 and 15
            # marks -- `English8IN`'s entire "question" was the group header,
            # "Text A Informative Newspaper Article - Fit India School Week
            # Item", and `English9PM4` was sub-part 1(a) carrying all 15 marks of
            # its passage. Class X English has no 15-mark question, so any
            # marks-wise report built on this was wrong, and the stem was not the
            # question that answer belonged to.
            #
            # The marks are therefore marked UNRESOLVED rather than carried.
            # Zero would be a lie in the other direction -- the count is known,
            # it is just not this question's -- so the stated total is preserved
            # in `group_marks_stated` for a future splitter to use.
            item.marks_unresolved = True
            item.group_marks_stated = item.marks
            # `marks` is deliberately NOT zeroed here: `_score` below uses it to
            # prefer the real item over a fragment, and zeroing first would hand
            # the id to the fragment. The record builder resolves it instead.
        score = _score(item)
        current = best.get(item_id)
        if current is None or score > current[0]:
            best[item_id] = (score, item)

    items = [item for _score, item in best.values()]

    for item in items:
        row = index.get(item.question_id)
        if row:
            # The INDEX is authoritative for the content code and the strand;
            # the item's own metadata is authoritative only for the fuller
            # reference text. Getting this precedence backwards let a loose
            # in-item regex produce codes that do not exist in the curriculum
            # and override the correct one -- 638 items collapsed onto seven
            # distinct codes, which is how the bug was spotted.
            item.topic = row["topic"] or item.topic
            if row["content_code"]:
                item.content_code = row["content_code"]
            if not item.ao1_marks:
                item.ao1_marks = row["ao1_marks"]
            if not item.ao2_marks:
                item.ao2_marks = row["ao2_marks"]
            # AO3/AO4 and the source description exist only in the English
            # index, hence `.get`. The item's own reference text still wins
            # when it has one, per the precedence above -- the index fills
            # the gap, it does not overwrite.
            if not item.ao3_marks:
                item.ao3_marks = row.get("ao3_marks", 0)
            if not item.ao4_marks:
                item.ao4_marks = row.get("ao4_marks", 0)
            if not item.content_reference:
                item.content_reference = row.get("content_reference", "")
    return items
