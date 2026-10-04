"""One NCERT chapter in, CBSE-format questions out.

The chain is deliberately short, because every extra hop is somewhere a question
can quietly stop following the board's rules:

  chapter PDF -> text with page numbers -> a plan from `cbse_blueprint` ->
  one generation call per section -> a validator that refuses what does not fit ->
  a bank record

Three things here are load-bearing.

*Chapter text is read from the PDF, not from a summary.* `pages()` keeps the page
number on every line, so a generated question can say where in the book it came
from, and so a later audit can go back to the printed page and check it.

*One call per section, not one per question.* The chapter text is the expensive
part of the prompt; asking for a section's worth of questions at once pays for it
once. It also lets the model see the other questions in its own section, which is
what stops it writing four questions that all test the same sentence.

*The validator is not the model.* A language model asked to follow a format will
report having followed it whether or not it did. So after the call, the marks,
the type, the option count, the Bloom level and the answerability are all checked
against the blueprint again, here, where a failure is a refusal with a reason
rather than a bad question in the bank.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import cbse_blueprint as bp
from . import mcq_shape as ms
from ..syllabus import textbook_headings as th

# The free model on Cloudflare Workers AI. 131k context, so a whole chapter
# fits in one prompt and this needs no retrieval step at all -- the chapter *is*
# the retrieval corpus, which is the cheapest correct design at this size.
DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

PAGE_MARK = re.compile(r"^\[page (\d+)\]$")

# What an answer may not contain. These are the tells that make a question unusable
# in a paper: the model hedging, apologising, or narrating that it is an AI.
BANNED_IN_STEM = (
    "as an ai", "language model", "i cannot", "i can't", "i'm unable",
    "however, it is not possible", "this question", "the following question",
    "based on the text above", "according to the text", "in the chapter",
    "the text mentions", "as per the", "slightly different", "generally speaking",
    "it depends", "there are many", "note that", "please note", "furthermore,",
    "overall,", "in conclusion", "it is important to note", "one might say",
)


@dataclass(frozen=True)
class Chapter:
    """A chapter as the generator needs it: the text, and where it printed."""

    book_code: str
    subject: str
    grade: int
    language: str
    number: int
    title: str
    pages: tuple[tuple[int, str], ...]   # (printed page, line)

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.pages)

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(p for p, _ in self.pages)

    def topic_window(self, topic: str, topics: list[str]) -> list[tuple[int, str]]:
        """The lines belonging to one topic: its heading up to the next topic's.

        Bounded by the headings themselves rather than by guessing where prose
        starts. Guessing fails badly on the primary books, which are verse and
        activity pages where every line is short and unpunctuated -- a
        prose heuristic ends the window after two lines and leaves the model
        nothing to ask about. The headings are the book's own division of the
        chapter, and between one heading and the next is exactly the material a
        teacher would set from.
        """
        if not topic or not topics:
            return list(self.pages)
        start = self._heading_line(topic)
        if start is None:
            return list(self.pages)
        # The next heading, in the book's order, closes the window.
        try:
            position = topics.index(topic)
        except ValueError:
            position = -1
        end = len(self.pages)
        for candidate in topics[position + 1:] + topics[:position]:
            if candidate == topic:
                continue
            line = self._heading_line(candidate)
            if line is not None and line > start:
                end = line
                break
        return self.pages[start:end]

    def _heading_line(self, topic: str) -> int | None:
        needle = _norm(topic)
        if not needle:
            return None
        for i, (_, line) in enumerate(self.pages):
            if needle in _norm(line):
                return i
        return None


def _norm(text: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def _looks_like_body(line: str) -> bool:
    """Whether a line is prose rather than a heading. A heading is short, has no
    terminal full stop, and is not a numbered exercise."""
    stripped = line.strip()
    if len(stripped) > 90 or stripped.endswith((".", ":", "?")):
        return True
    if re.match(r"^(exercise|question|ex\s*\d|eg\.|example|activity|fill in|choose|"
                 r"tick|write|read the|answer|observe|look at|draw|colour)", stripped, re.I):
        return True
    return False


def read_chapter(pdf: Path, *, book_code: str, subject: str, grade: int,
                 language: str, number: int, title: str) -> Chapter:
    """The chapter as printed, one line per entry, each carrying its page.

    Read through `textbook_headings`, not `get_text()`, and that is not a detail.
    An NCERT heading wraps: the chapter title prints as "Chemical Reactions" on one
    line and "and Equations" on the next, and drop capitals arrive as a bare "C"
    followed by "onsider". Raw `get_text` returns those as separate lines, so a
    heading never matches itself and every topic silently falls back to the whole
    chapter. `read_lines` rebuilds each line from its characters and joins the
    wrapped pieces, and `printed_page_numbers` gives the page the book prints
    rather than the page's position in the file -- the difference between citing
    "page 7" and citing something no marker can find.
    """
    lines = th.reading_order(th.read_lines(pdf))
    printed = th.printed_page_numbers(lines)
    # printed_page_numbers is printed -> PDF page; invert it for the walk.
    by_pdf = {pdf_page: printed_no for printed_no, pdf_page in printed.items()}

    pages: list[tuple[int, str]] = []
    for line in lines:
        text = (line.text or "").strip()
        if not text:
            continue
        pages.append((by_pdf.get(line.page, line.page), text))

    return Chapter(book_code=book_code, subject=subject, grade=grade,
                   language=language, number=number, title=title,
                   pages=tuple(pages))


def headings_in_chapter(pdf: Path, number: int) -> list[str]:
    """The topics a chapter prints in its own typography.

    The fallback for a chapter no taxonomy tree covers. `chapter_headings` picks
    the heading scheme the chapter itself uses -- numbered for the senior books,
    unnumbered for the primary ones -- so this works across grades 1-10 without
    being told which kind of book it is holding.
    """
    try:
        lines = th.reading_order(th.read_lines(pdf))
        found, _name = th.chapter_headings(lines, number)
    except Exception:                      # a chapter with no supported scheme
        return []
    out: list[str] = []
    for heading in found:
        name = (getattr(heading, "name", None) or getattr(heading, "title", None)
                or "").strip()
        if name and name not in out:
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #

SYSTEM = """You are a CBSE question paper maker. You set papers for {subject}, Class {grade}, \
in {language_name}. You are given the text of one chapter, and for one section of the paper \
you write the questions that section holds.

The paper is {total_marks} marks. You are writing Section {section_key}, "{section_name}".

Rules, all of which the board enforces:
- Every question in this section is worth exactly {marks} mark(s).
- The question type is {qtype}. Write it in the form the board uses for that type.
{type_rules}{kind_rules}- Set the question in a real situation a Class {grade} student meets, not a \
definition to memorise. The board calls this competency-based, and it is the reason the \
paper exists.
- Use only what the chapter text below actually says. If the chapter does not settle it, \
write a different question; never write one the text cannot answer.
{naming_rule}- Write in {language_name}. {script_note}
- Distractors, where the type has them, must be plausible to a student who has read the \
chapter and wrong for a reason. No joke options, no "all of the above", no "none of these".
{bloom_note}Return only the questions, as JSON matching the schema."""


TYPE_RULES = {
    bp.MCQ: """- One sentence of stem, then four options labelled A, B, C, D. Exactly four.
- Every option is the same kind of thing: all words, or all numbers, or all dates.
- Exactly one option is correct. Vary which letter is correct; do not favour A.
""",
    bp.ASSERTION_REASON: """- Two numbered statements, an option, and the reason.
- Options are the four standard pairs: both true and reason explains; both true but
  reason does not explain; assertion true, reason false; both false.
""",
    bp.VSA: """- A question the student answers in one line. No part (a)/(b), no sub-questions.
- The answer is a single word, number, or short phrase. Not a sentence.
""",
    bp.SA: """- A question the student answers in about {words} words.
- At most two parts, and only if the section's marks allow it.
- Keep the model answer to about {words} words. It is a mark scheme, not an essay.
""",
    bp.LA: """- A question the student answers in about {words} words, in steps.
- Show the working the student would write, not a paragraph of prose.
- Keep the model answer to about {words} words. It is a mark scheme, not an essay.
""",
    bp.CASE: """- Open with a short real-life passage of 80-120 words that the student reads.
- Then ask {n} short questions on it, of which the last is a higher-order one.
- The passage is part of the question, not a separate item.
- Keep the model answer to about {words} words in total.
""",
    bp.MAP: """- A map-pointing question: "On an outline political map of India, locate and label the
  following:" and then five features, (a) to (e), each worth one mark, that the chapter names
  (a dam, a port, a power plant, a mineral belt, a state where something happened, the place of
  a session or a movement). Say "outline map of India" or "outline map of the world", never "the
  given map", "the figure" or "the map shown": the paper prints no map, the student marks the
  school's blank outline map.
- Name each feature exactly as the chapter does, with nothing the student could not find from
  the chapter alone.
- `modelAnswer` lists each feature with where it lies (state or region) as the chapter says.
  `markingPoints` is one point per feature.
""",
}


def has_passage(section: bp.Section) -> bool:
    """A reading or extract question prints a passage copied from the book above its stem. A case
    study writes its own passage, and a map section "carries a source" only in the sense that the
    map sits above the question: neither takes the copied-passage rules."""
    return section.carries_source and section.qtype not in (bp.CASE, bp.MAP)


NAMING_RULE = """- Never mention the chapter, the text, the passage or yourself in the question. A student \
must not be able to tell it was written by reading the question.
"""
SOURCE_NAMING_RULE = """- The question is printed under its `passage`, so it may say "the passage" or "the extract". \
Never mention the chapter, the lesson, the book or yourself.
"""

# What each kind of language question is. The paper's section names the kind (assessment/
# template_presets); the generator writes it so a section never has to guess what a 1-mark
# question is. {words} is the section's word limit.
KIND_RULES = {
    "reading": """- `passage`: copy 50-110 words from the chapter text exactly as printed, whole sentences in
  order. It is printed above the question, so the question must not repeat it.
- The question can be answered only by reading the passage: an inference, the writer's purpose
  or tone, what a word or phrase means in that place, or a detail.
""",
    "extract": """- `passage`: copy 30-80 words from the chapter text exactly as printed (a stanza, a speech or
  a paragraph), whole lines in order. It is printed above the question, so the question must not
  repeat it.
- Ask about the extract: its meaning, a word or image in it, who speaks, why, what it shows.
""",
    "grammar": """- Base the question on one sentence copied from the chapter text, and put that sentence in
  `evidence`. Test one grammar point that sentence shows (tense, voice, narration, a determiner,
  a modal, a clause, a connector, a punctuation mark; in Hindi a sandhi, samas, alankar, vakya
  bhed or pad parichay). The stem carries the sentence or one made from it, with four options.
- The answer must follow from the rule, not from taste.
""",
    "writing": """- Set one writing task of the kind the paper asks for (a letter, an e-mail, a notice, a
  paragraph, a speech, an article, an advertisement, a message) on a situation drawn from the
  chapter's own theme. Say the format, the topic and the word limit ({words} words).
- `markingPoints`: content, format, organisation, accuracy and expression, as separate points.
  `modelAnswer`: a short sample in the right format.
""",
    "literature": """- Ask about the lesson or poem: a character's reason, the theme, what a line means, how
  something changes, why something happened. The answer is in the chapter text.
- The model answer says it in about {words} words, and each marking point is one idea.
""",
}


def build_prompt(chapter: Chapter, section: bp.Section, topic: str, *,
                 count: int, bank_subject: str, topics: list[str] | None = None
                 ) -> tuple[str, dict, str]:
    """The system text, the JSON schema, and the user text for one section.

    The schema is what makes the model obey the format rather than describe it.
    Every constrained value is an enum taken from the blueprint, so a question that
    came back with the wrong marks or the wrong type cannot parse.

    `count` asks for several questions in one call rather than one per call, and it
    is the largest cost lever in the pipeline. The chapter text is the bulk of the
    prompt, so nine one-question calls pay for the chapter nine times where three
    three-question calls pay for it three times. It is a quality lever too: a model
    writing the fourth question in a section can see the first three and stops
    restating the same sentence, which is the failure that made a first run return
    four questions testing one line of text.
    """
    topics = topics or []
    language_name = {"en": "English", "hi": "Hindi"}.get(chapter.language, chapter.language)
    script_note = {
        # An English-medium book is printed in Latin numerals, and a Devanagari
        # digit in an English paper is a marking-scheme error, not a stylistic one.
        "en": "Use Latin numerals (1, 2, 3), as the book does.",
        "hi": "Use Devanagari numerals and the book's own Devanagari punctuation, "
              "not Latin digits.",
    }.get(chapter.language, "")
    target_bloom = section_levels(section)

    system = SYSTEM.format(
        subject=bank_subject, grade=chapter.grade, language_name=language_name,
        total_marks=bp.for_paper(bank_subject, chapter.grade).total_marks,
        section_key=section.key, section_name=section.name, marks=section.marks,
        qtype=section.qtype,
        type_rules=TYPE_RULES.get(section.qtype, "").format(
            words=section.answer_words or section.marks * 2,
            n=max(2, min(count, 4))),
        script_note=script_note,
        naming_rule=SOURCE_NAMING_RULE if has_passage(section) else NAMING_RULE,
        kind_rules=KIND_RULES.get(section.kind, "").format(
            words=section.answer_words or section.marks * 2),
        bloom_note=(f"- Aim for this cognitive level: {target_bloom}.\n"),
    )

    schema = _schema(section, count)
    user = _user_text(chapter, topic, section, count, topics)
    return system, schema, user


def section_levels(section: bp.Section) -> str:
    """The cognitive level this section's questions should sit at."""
    if section.qtype in (bp.CASE, bp.MAP):
        return "apply, then analyze for the closing question"
    if section.qtype in (bp.LA, bp.SA):
        return "understand to apply"
    if section.qtype == bp.ASSERTION_REASON:
        return "understand and analyze"
    return "remember to apply"


def _schema(section: bp.Section, count: int) -> dict:
    """JSON Schema for one section's questions, restricted to what both providers
    accept.

    Both Cloudflare Workers AI and Gemini will hold a model to a schema, which is
    the only reason a model can be made to follow the board's format rather than
    describe it. But their subsets differ, and the intersection is narrower than
    either:

    * `const` is Workers AI only; Gemini 400s on it.
    * `enum` on a non-string is Workers AI only; Gemini 400s with
      "(TYPE_STRING), 4" -- it will not put a number in an enum at all.
    * `minItems` / `maxItems` are Workers AI only.

    So every enum here is a list of *strings*, including the marks, which travels
    as `"4"` and is converted back on the way in. The alternative -- dropping the
    marks pin so the schema is simpler -- would mean a wrong-marks question is
    produced and then caught, instead of being unable to exist. `validate` checks
    marks and marking-point count either way, so this is belt as well as braces.

    `tests/test_cbse_generate.py` walks the built schema and fails on any key
    outside the shared subset, so this cannot rot silently.
    """
    item = {
        "type": "object",
        "properties": {
            "qtype": {"type": "string", "enum": [section.qtype]},
            # A string enum on purpose: see the docstring.
            "marks": {"type": "string", "enum": [str(section.marks)]},
            "bloomLevel": {"type": "string", "enum": list(bp.BLOOM_ORDER)},
            "stem": {"type": "string"},
            "option": {"type": "integer", "description": "1-based position in the section"},
        },
        "required": ["qtype", "marks", "bloomLevel", "stem", "option"],
    }
    if section.qtype in (bp.MCQ, bp.ASSERTION_REASON):
        item["properties"]["options"] = {
            "type": "object",
            "properties": {k: {"type": "string"} for k in "ABCD"},
            "required": ["A", "B", "C", "D"],
        }
        item["properties"]["correctOption"] = {"type": "string", "enum": list("ABCD")}
        item["required"] += ["options", "correctOption"]
    if section.qtype in (bp.SA, bp.LA, bp.VSA, bp.MAP):
        item["properties"]["modelAnswer"] = {"type": "string"}
        # No minItems/maxItems: Gemini rejects them. `validate` refuses a
        # marking-point count that does not match the marks.
        item["properties"]["markingPoints"] = {
            "type": "array", "items": {"type": "string"}}
        item["required"] += ["modelAnswer", "markingPoints"]
    if section.qtype == bp.CASE:
        item["properties"]["passage"] = {"type": "string"}
        item["properties"]["modelAnswer"] = {"type": "string"}
        item["required"] += ["passage", "modelAnswer"]
    elif has_passage(section):
        # A reading or extract question prints the passage it was set on above its own stem.
        item["properties"]["passage"] = {"type": "string"}
        item["required"] += ["passage"]

    return {
        "type": "object",
        # No minItems here either: Gemini rejects it, and an empty array is caught
        # by the loop that reads it, which simply produces no questions.
        "properties": {"questions": {"type": "array", "items": item}},
        "required": ["questions"],
    }


def parse_item(raw: dict) -> dict:
    """A generated item with `marks` back as an int.

    The schema carries marks as a string enum because that is the only enum form
    both providers accept. Everything downstream -- `validate`, the record -- wants
    an int, so it is converted here, once, at the boundary.
    """
    item = dict(raw)
    marks = item.get("marks")
    if isinstance(marks, str):
        try:
            item["marks"] = int(marks.strip())
        except ValueError:
            item["marks"] = None
    return item


def _user_text(chapter: Chapter, topic: str, section: bp.Section,
               count: int, topics: list[str]) -> str:
    # Cost is set by the prompt, not the model: the whole plan for a chapter is one
    # call per section item, and each call re-sends the chapter text. So the window
    # is the topic's own span, kept tight, and the chapter is only used whole when
    # the topic cannot be located at all.
    lines = chapter.topic_window(topic, topics)
    # A language question is set on passages, so it needs the lesson itself and not a thin slice of
    # an activity page: more lines, which the free models handle without cost.
    window = 320 if section.kind else 90
    if section.kind and section.kind != "grammar":
        # The lesson comes first in a chapter file and its exercises after it. A topic's centred
        # slice of an English chapter was an exercise page ("Fill in the blanks with the past
        # perfect"), which is no passage to read or extract to ask about.
        lines = list(chapter.pages[:window])
    elif len(lines) < 20 or len(lines) > window:
        # Too thin to set a question from, or so wide it is costing tokens: take a
        # centred slice of the topic's own span rather than the entire chapter.
        pool = lines if len(lines) >= 20 else chapter.pages
        if len(pool) > window:
            start = max(0, len(pool) // 2 - window // 2)
            lines = pool[start:start + window]
        else:
            lines = pool
    if not lines:
        lines = chapter.pages[:window]
    body = "\n".join(line for _, line in lines)
    pages = sorted({p for p, _ in lines})
    where = f"pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"page {pages[0]}"
    heading = f"Topic: {topic}\n" if topic else ""
    return (f"{heading}Chapter {chapter.number}: {chapter.title} "
            f"({chapter.book_code}, {where})\n\n"
            f"Write {count} question(s) for Section {section.key}, each asking about "
            f"something different.\n\n"
            f"--- chapter text ---\n{body}\n--- end ---")


# --------------------------------------------------------------------------- #
# the validator
# --------------------------------------------------------------------------- #

def _numbers_in(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text or "")]


def _ladder_options(options: dict, correct: str) -> bool:
    """Whether the wrong options are the right one walked up and down a number.

    A live run produced, for the factorisation of 91, the options 7x13, 7x14,
    7x15 and 7x16. All four are distinct, all four are the same length, and
    `mcq_shape.answerable` waves them through -- but a student who cannot factorise
    anything can still pick, because three of the four sit in an arithmetic
    sequence. The distractors have to be wrong for a reason, not wrong by
    adjacency.

    So: if every wrong option is the correct option with exactly one number in it
    moved by a small constant, the item is refused.
    """
    right = _numbers_in(options.get(correct) or "")
    if not right:
        return False
    wrong = [options[k] for k in "ABCD" if k != correct]
    if len(wrong) != 3:
        return False

    deltas: set[int] = set()
    for option in wrong:
        numbers = _numbers_in(option)
        if len(numbers) != len(right):
            return False
        # Same multiset of numbers, so the difference is positional.
        if sorted(numbers) == sorted(right):
            deltas.add(0)
            continue
        moved = [abs(a - b) for a, b in zip(numbers, right) if a != b]
        if not moved or len(moved) != 1:
            return False
        deltas.add(moved[0])
    # Every wrong option differs by the same small step, or not at all.
    return bool(deltas) and max(deltas) <= 3


def _is_a_question(stem: str) -> bool:
    """Whether the stem actually asks something.

    A stem that trails off -- "The prime factorisation of the number 12 is" -- is
    a statement with the question mark left off, and it reads as broken on a
    printed paper. A stem that opens by telling the student what is wanted, as in
    "State the prime factorisation of 12", asks for a form rather than testing
    anything, and the board's competency questions do not read that way.
    """
    text = (stem or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    # । and ॥ are the danda and double danda: how a Hindi sentence ends.
    if not text.endswith((".", "!", ":", "।", "॥")):
        return False
    lowered = text.lower()
    if lowered.startswith(("state ", "write ", "list ", "name ", "mention ",
                           "give ", "tell ")):
        return False
    return True


# "1.", "Q3.", "I.", "(ii)", "a)" -- a question number the model copied from the
# shape of a paper. A live run returned "1. Find the HCF of 6, 72 and 120" and
# "I. Is 5 - sqrt(3) rational or irrational?", because CBSE papers are numbered and
# the model is imitating a paper rather than writing an item.
_LEADING_NUMBER = re.compile(
    r"^\s*(?:q(?:uestion)?\.?\s*)?\(?(?:\d{1,2}|[ivxlc]{1,4}|[a-d])\)?\s*[.)\-:]\s+",
    re.I)


def strip_question_number(stem: str) -> str:
    """The stem as it would be numbered on a paper, not as it was numbered here."""
    stem = stem or ""
    # "(a) ... (b) ..." is a question in parts, and its first label is part of it: stripping the
    # "(a)" left a case study asking one part and then "(b)" (ai:cbse:jess1:01:E:1), which the
    # stem gate refuses as a broken option list.
    if re.match(r"^\s*\(?a\)", stem, re.I) and re.search(r"\(\s*b\s*\)", stem, re.I):
        return stem.strip()
    cleaned = _LEADING_NUMBER.sub("", stem, count=1).strip()
    return cleaned or stem.strip()


def _still_multi_part(stem: str) -> bool:
    """Whether the stem is really several questions in one.

    A live run returned "I. Is 5 - sqrt(3) rational or irrational? II. What property
    of irrational numbers is being used in this proof?" for a single 2-mark item. A
    bank entry is one question; a paper numbers them.
    """
    parts = re.split(r"\s(?:I{1,3}V?|IV|V|\(?[a-e]\))\s*[.)\-]\s+", stem or "")
    return len([p for p in parts if p.strip()]) > 1


def _contradicts_its_stem(item: dict) -> bool:
    """Whether a model answer contradicts what the stem claims to be true.

    A stem that says "show that", "prove that" or "verify that" is asserting a
    result. If the model answer then says the result does not hold, the two cannot
    both go on a paper: the student is asked to prove something that is not so, and
    the mark scheme would have to mark the right answer wrong.

    Cheap and narrow on purpose -- it looks for the words, not for arithmetic. It
    catches the case a language model reliably falls into, which is inventing a
    nice question whose premise it then has to quietly contradict.
    """
    stem = (item.get("stem") or "").lower()
    if not re.search(r"\b(show|prove|verify|establish|demonstrate)\b.{0,40}"
                     r"\b(that|whether)\b", stem, re.S):
        return False
    answer = (item.get("modelAnswer") or "").lower()
    if not answer:
        return False
    # A denial, in the words a worked answer uses.
    return bool(re.search(
        r"(is not|are not|does not|do not|cannot|≠|\bisn't\b|doesn't\b|"
        r"not equal|is false|incorrect)", answer)) and not re.search(
        r"(therefore|hence|thus|so,|which proves|as required)", answer)


def audit_key_distribution(records: list[dict]) -> dict:
    """Where the key lands across a run of MCQs, and the worst offenders.

    Not a per-question refusal, because a single question with key A is not a
    fault. But a model asked to "not favour A" will favour A in most of its
    output if left unchecked -- a live run put the key at A in both of its
    questions -- and a paper where the key is A more often than not is a paper a
    student can score in without reading the questions. Reported so the run can be
    retried or the offending items dropped.
    """
    keys: list[str] = []
    for record in records:
        meta = (record.get("answerScheme") or {}).get("metadata") or {}
        if meta.get("correctOption"):
            keys.append(meta["correctOption"])
    counts = {k: keys.count(k) for k in "ABCD" if k in keys}
    total = len(keys)
    return {"total": total, "counts": counts,
            "share": {k: (v / total if total else 0.0) for k, v in counts.items()},
            "largest": max(counts, key=counts.get) if counts else None,
            # 25% is chance. Anything much above it is a bias a student can use.
            "biased": bool(total >= 4 and max(counts.values()) / total > 0.40)}


def validate(item: dict, section: bp.Section) -> str | None:
    """Why this question does not belong in the paper, or None if it does.

    Deliberately harsher than the schema. The schema stops a malformed answer
    arriving; this stops a well-formed answer that is not a question for this
    section -- the failure mode a schema cannot see.
    """
    stem = (item.get("stem") or "").strip()
    if not stem:
        return "empty-stem"
    if len(stem) < 12:
        return "stem-too-short"

    low = stem.lower()
    for tell in BANNED_IN_STEM:
        if tell in low:
            return "stem-names-the-text"

    if item.get("qtype") != section.qtype:
        return "wrong-type"
    # Marks may arrive as a string: the schema pins it with a string enum, which
    # is the only enum form Gemini accepts. Compare on int() where possible so a
    # `"4"` and a `4` are the same answer, and refuse anything non-numeric.
    marks = item.get("marks")
    try:
        marks = int(str(marks).strip())
    except (TypeError, ValueError):
        return "wrong-marks"
    if marks != section.marks:
        return "wrong-marks"

    bloom = item.get("bloomLevel")
    if bloom not in bp.BLOOM_ORDER:
        return "bad-bloom"

    if section.qtype in (bp.MCQ, bp.ASSERTION_REASON):
        options = item.get("options") or {}
        if sorted(options) != ["A", "B", "C", "D"]:
            return "options-not-abcd"
        if not ms.answerable(options):
            return "options-not-answerable"
        correct = item.get("correctOption")
        if correct not in "ABCD" or not (options.get(correct) or "").strip():
            return "no-correct-option"
        if _ladder_options(options, correct):
            return "distractor-ladder"

    # A writing task ("Write a letter to ...") and a grammar item ("Choose the correct form ...") are
    # given as an instruction; only the other kinds have to read as a question.
    if section.kind not in ("writing", "grammar") and not _is_a_question(stem):
        return "stem-is-not-a-question"
    if _still_multi_part(stem):
        return "stem-is-several-questions"

    # A premise the book cannot support is the one fault no schema sees. A live run
    # asked the student to "verify that LCM x HCF = product of the two numbers" for
    # 140 and 156, which is false -- that identity only holds for coprime pairs --
    # and the model answered the question correctly by showing it was false. The
    # arithmetic was right and the question was still unaskable. So the model
    # answer is checked against the stem: if the stem says "show that" or "verify
    # that" and the model answer concludes otherwise, the item is refused rather
    # than printed.
    if item.get("qtype") in (bp.SA, bp.LA, bp.VSA) and _contradicts_its_stem(item):
        return "premise-not-supported"

    if section.qtype in (bp.VSA, bp.SA, bp.LA, bp.MAP):
        if not (item.get("modelAnswer") or "").strip():
            return "no-model-answer"
        points = item.get("markingPoints") or []
        # A marking scheme is worth the section's marks; how many points express
        # that is the board's business, not ours. An earlier version of this
        # required one point per mark, which refused 12 of 27 completions on a
        # single chapter because a 5-mark long answer was scored in three steps --
        # "2 for the factorisation, 2 for the product, 1 for the conclusion" is
        # how CBSE writes a 5-mark answer. The count is bounded so a padded scheme
        # still cannot slip through, and `to_record` distributes the marks so the
        # stored scheme is worth exactly the section's marks.
        if not points:
            return "no-marking-points"
        if len(points) > section.marks:
            return "too-many-marking-points"

    if has_passage(section):
        passage = (item.get("passage") or "").strip()
        words = len(passage.split())
        if words < 25:
            return "passage-too-short"
        if words > 160:
            return "passage-too-long"
        if passage in stem:
            return "stem-repeats-the-passage"

    if section.qtype == bp.CASE:
        passage = (item.get("passage") or "").strip()
        # CBSE case passages run 100-150 words; the prompt asks for 80-120. A
        # floor of 60 catches the model returning a sentence with a question
        # bolted on, which is not a case-based section at all.
        if len(passage.split()) < 60:
            return "case-passage-too-short"

    return None


def _marking_points(texts: list[str], total: int) -> list[dict]:
    """A marking scheme whose points sum to exactly the section's marks.

    The model returns the points as strings and does not say what each is worth,
    because that is what the board decides. So the marks are shared out evenly and
    the remainder goes to the first points, and a scheme that says "2 for this, 1
    for that" is expressed as a point of 2 and a point of 1 rather than being
    rejected for having two entries.

    The number of points is capped at the number of marks by the validator, so a
    scheme never has more mark-bearing steps than marks and never has to invent a
    zero-mark point. The invariant that matters downstream -- that a question's
    points add up to its marks -- is asserted in `test_cbse_generate.py` for every
    section of every blueprint, so this cannot quietly produce a scheme worth the
    wrong number.
    """
    texts = [t for t in (str(x or "").strip() for x in texts) if t]
    if not texts:
        return []
    texts = texts[:max(1, total)]
    base, extra = divmod(total, len(texts))
    out: list[dict] = []
    for index, text in enumerate(texts):
        marks = base + (1 if index < extra else 0)
        out.append({"description": text, "marks": marks, "keyword": "",
                    "isRequired": True, "synonyms": []})
    return out


def rebalance_keys(records: list[dict], tolerance: float = 0.34) -> dict:
    """Level the key across a run by rotating option labels, losslessly.

    A free-tier run of 78 MCQs came back with the key on B 6 times, C 38 and D 34
    -- and never once on A. Nothing in the prompt forbids that, the schema cannot
    express it, and per-question validation cannot see it: each of those 78
    questions is individually well formed. It is only visible in aggregate, which
    is what `audit_key_distribution` is for.

    It is also the kind of fault that costs the product its credibility. A teacher
    who notices the answer is never A stops trusting the paper, and a student who
    notices scores full marks without reading anything.

    So the options are relabelled: the correct option moves to whichever letter is
    currently under-used, and every other option travels with its text. Nothing
    about the question changes -- the stem, the option texts, the distractors and
    the answer itself are all identical afterwards. Only the letters move.

    Reports what it did, because a silent rewrite of a third of a bank's keys is
    not something to do quietly.
    """
    with_options = [r for r in records
                    if (r.get("answerScheme") or {}).get("metadata", {}).get("correctOption")]
    if not with_options:
        return {"mcqs": 0, "rotated": 0}

    def keys_now() -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in with_options:
            letter = record["answerScheme"]["metadata"]["correctOption"]
            counts[letter] = counts.get(letter, 0) + 1
        return counts

    # A list, not a dict keyed by id. Keying by id looked tidy and silently
    # collapsed 78 records to 37, because ids collide when the model supplies the
    # position and two sections both return position 1. A rebalance that quietly
    # discards half the bank is worse than the bias it was fixing.
    pending = with_options
    counts = keys_now()
    total = len(pending)
    limit = total * tolerance
    rotated = 0

    # Walk the over-represented letters down to the tolerance, one rotation at a
    # time, re-counting each pass: a single pass cannot see the effect of its own
    # changes, and a second letter may become over-represented as a result.
    for _ in range(len("ABCD") * total):
        counts = keys_now()
        over = max(counts, key=counts.get)
        if counts[over] <= limit:
            break
        # Find the least-used letter that is not the one being relieved.
        under = min((k for k in "ABCD" if k != over), key=lambda k: counts.get(k, 0))
        candidates = [r for r in pending
                      if r["answerScheme"]["metadata"]["correctOption"] == over]
        if not candidates:
            break
        target = candidates[0]
        scheme = target["answerScheme"]
        options = dict(scheme["metadata"]["options"])
        if under not in options or over not in options:
            break
        options[under], options[over] = options[over], options[under]
        scheme["metadata"]["options"] = options
        scheme["metadata"]["correctOption"] = under
        scheme["modelAnswer"] = options[under]
        target["provenance"]["keyRotated"] = f"{over}->{under}"
        rotated += 1

    return {"mcqs": total, "rotated": rotated,
            "after": dict(sorted(keys_now().items()))}


def _repaired(item: dict) -> dict:
    """`item` with its Hindi text mended (corpus/hindi_ocr.repair_devanagari): the model copies the
    book's words, and the book's text layer is what the extraction broke."""
    from .hindi_ocr import repair_devanagari as fix

    def walk(value):
        if isinstance(value, str):
            return fix(value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return {k: (walk(v) if k in ("stem", "passage", "options", "modelAnswer", "markingPoints")
                else v) for k, v in item.items()}


def to_record(item: dict, section: bp.Section, chapter: Chapter, topic: str,
              *, chapter_id: str, topic_id: str, model: str,
              also_chapter_ids: tuple[str, ...] = ()) -> dict:
    """A validated question as the bank stores it.

    `source` is `ai_generated` and the provenance says so in as many words, so a
    generated question can never be mistaken for one lifted out of a board paper
    or a book. The evidence carries the page it was written from, which is what
    lets anyone check it later.
    """
    if chapter.language == "hi":
        item = _repaired(item)
    scheme: dict = {"totalMarks": section.marks, "markingPoints": [], "rubricLevels": [],
                    "commonErrors": [], "alternativeAnswers": [],
                    "modelAnswer": item.get("modelAnswer") or "",
                    "modelAnswerLatex": "", "hasPartialCredit": section.marks >= 3,
                    "metadata": {"answerSource": "generated",
                                 "evidence": [],
                                 "objective": section.qtype in (bp.MCQ, bp.ASSERTION_REASON)}}
    if section.qtype in (bp.VSA, bp.SA, bp.LA, bp.MAP):
        scheme["markingPoints"] = _marking_points(item.get("markingPoints") or [],
                                                 section.marks)
    if section.qtype in (bp.MCQ, bp.ASSERTION_REASON):
        options = item.get("options") or {}
        scheme["metadata"]["options"] = options
        scheme["metadata"]["correctOption"] = item.get("correctOption") or ""
        # An MCQ's model answer is the text of the correct option. Leaving it
        # blank means every consumer that reads `modelAnswer` -- the answer
        # display, the marking view, the export -- shows nothing for an objective
        # question, even though the key is sitting in the metadata.
        scheme["modelAnswer"] = options.get(item.get("correctOption") or "", "")
    if section.qtype == bp.CASE:
        scheme["metadata"]["passage"] = item.get("passage") or ""

    stem = strip_question_number(item.get("stem") or "")
    if section.qtype in (bp.MCQ, bp.ASSERTION_REASON):
        # The bank, the paper and the merge's MCQ gate all read the options out of the
        # stem, as every served board question carries them: "... (A) x (B) y (C) z (D) w".
        # Left only in the metadata, the question prints with no choices.
        options = item.get("options") or {}
        if options and not re.search(r"\(A\)", stem):
            stem = f"{stem} " + " ".join(f"({k}) {options[k]}" for k in "ABCD" if k in options)
    if (section.qtype == bp.CASE or has_passage(section)) and (item.get("passage") or "").strip():
        # A case study, a reading question and an extract question are their passage and the
        # question on it; the stem is what prints.
        stem = item["passage"].strip() + "\n\n" + stem

    record = {
        "id": f"ai:cbse:{chapter.book_code}:{chapter.number:02d}:{section.key}:{item.get('option', 1)}",
        "questionBankId": f"ai-cbse:{chapter.book_code}",
        "subject": chapter.subject,
        "grade": chapter.grade,
        "chapterIds": [chapter_id, *also_chapter_ids],
        "taxonomyChapterId": chapter_id,
        # A topic named like the chapter is no topic label: a label equal to the chapter's name is read by
        # the bank's topic search as that label alone (8 of Science 9 "Tissues in Action"'s 52 questions)
        # instead of as the chapter (`qbank_engine._on_topic`, rule 2 before rule 3).
        "topic": "" if (topic or "").strip().lower() == (chapter.title or "").strip().lower() else (topic or ""),
        "difficulty": "medium",
        "bloomLevel": item.get("bloomLevel") or "understand",
        # A passage-based reading or extract question is what CBSE calls competency-based (source
        # based; the presets head the reading section 100% competency), and the competency rule counts only a type the source printed
        # (assessment/competency.COMPETENCY_TYPES). The stem still carries its four options, which is
        # all the printer and the answer key read.
        "type": "competency_based" if section.kind in ("reading", "extract") else section.qtype,
        # The paper supplies the number; the stem must not carry one.
        "stem": stem,
        "stemLatex": "",
        "parts": [],
        "answerScheme": scheme,
        "estimatedTimeMinutes": max(1, section.marks),
        "marks": section.marks,
        "source": "ai_generated",
        "provenance": {
            "method": "cbse-blueprint-generation",
            "model": model,
            "section": f"{section.key} {section.name}",
            "chapter": f"{chapter.book_code} ch{chapter.number}",
            "pages": sorted(set(chapter.page_numbers)),
            "promptVersion": 1,
        },
    }
    if section.kind:
        # What a template section that names the kind (Reading, Grammar, Writing...) matches on.
        record["metadata"] = {"paperSection": section.kind}
    if topic_id:
        record["taxonomyTopicId"] = topic_id
    return record


def plan_for_chapter(book_subject: str, grade: int, topics: list[str],
                     mark_budget: int) -> list[tuple[bp.Section, str]]:
    return bp.plan_chapter(bp.for_paper(book_subject, grade), topics, mark_budget)
