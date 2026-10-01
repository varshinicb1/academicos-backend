"""Split an official answer into the value points its source printed.

Why this exists
---------------
The re-audit of 2026-09-23 measured every one of the 2,449 served multi-mark
questions carrying exactly ONE marking point worth all the marks. The printed
key (`assessment/paper.py`) then reads

    - <the whole answer> [5m]

so a teacher cannot give 3 of 5, and `assessment/evaluate.py` can only score 0
or 5 against it. A marking scheme exists so a teacher can give 3 of 5; one
value point worth all 5 marks is a model answer wearing a scheme's clothes.

The sources print the split. A CBSE marking scheme writes "1+1=2", "1X2=2",
"Any other, Any two", "Award 1 mark for each point"; a CBE maths scheme codes
every step of the working, "M1 4 x 2(12+8) ... M1 their 160 x 5 ... A1 Rs
800"; an NCERT Exemplar answer prints its parts, "(a) ... (b) ... (c) ...".
The importers threw all of it away. This module reads it back -- and NOTHING
else. Measured over the 2,449 served multi-mark answers with the rules as they
stand (`python scripts/check_marking_split.py --reasons`), 123 of them split
and 2,326 keep their single point, which is what those answers say:

   1,606  prose: no list, no mark annotation, nothing to read
     273  a list whose count does not divide the question's marks
     213  two answers in one key -- an internal choice
     168  a CBE group item, whose key covers several sub-questions
      22  a printed per-point value that does not fit the list
      13  answer content outside the list no point would account for
       9  the board prints half a mark a point, which `marks` cannot carry
       8  a statement in front of the list that no point accounts for
       3  an "any N" that does not divide the marks or over-awards the list
       3  a level-of-response rubric, whose bullets are not value points
       3  named points that do not account for the question's marks
       3  printed marks or addends that do not line up with the list
       1  "any one of the following", where one point carries every mark
       1  a case study, whose parts the paper weights 1, 1 and 2

Every one of those keeps its single point and SAYS so (`allOrNothing`), so a
teacher sees "this key is all or nothing" rather than assuming steps exist.

The rules, tried in this order
------------------------------
  step-marks     a worked solution with the board's code against each step --
                 "M1 <method> ... A1 <the result>" -- where the codes add up
                 to the question's marks and each one has a step beneath it.
  any-of         "Any two of the following" over a list: each listed point is
                 worth marks/N and the scheme records `anyOf: N`, because only
                 N of them are awarded.
  marks-each     "1 mark each", "Award 1 mark for each point", "1X2=2": the
                 per-point value is printed, and it must fit the list exactly.
  marks-printed  "1+1", "2+1=3" printed for the list, or a mark printed after
                 each point of it ("(i) Definition of refraction 1 (ii)
                 Statement of the two laws 2"): the board printed the split
                 itself, and the values are assigned to the items in order.
  marks-for      "Award 1 mark for a valid explanation of 'daunting'. Award 1
                 mark for a valid explanation of 'venture'.": the board names
                 its points and what each is worth. Read only as a set that
                 accounts for the marks -- one award on its own is a level of
                 response or a tolerance, not a division of the answer.
  list-equal     a numbered or lettered list with no mark annotation at all --
                 "(a) ... (b) ... (c) ..." against a 3-mark question. The
                 source shows the points; dividing the marks equally over them
                 is the only allocation consistent with what it shows. Not
                 over a case study: CBSE prints its parts as 1, 1 and 2 in the
                 paper's own general instructions, so equal is known to be
                 wrong before the answer is read.

Every rule must ACCOUNT FOR THE MARKS. A split is accepted only when the marks
it reads add up to exactly the question's marks -- under `any N`, when N points
of the printed per-point value do. An annotation we can read but cannot
reconcile with the list (say "2+1=3" over three items: which two are the 1?)
refuses the split outright rather than falling back on an equal division. A
wrong split is worse than no split: it tells a teacher that a mark belongs to a
point the board did not award it for.

Every rule must also ACCOUNT FOR THE ANSWER. Text in front of the list is a
lead-in ("Hint—", "Two reasons ... are- 1+1=2") and carries no mark, but only
while it reads as one: a lead-in that ENDS A SENTENCE is a statement of its
own, and splitting the list behind it would put the question's marks on part
of its answer. That is measured, not assumed -- reading all 74 `list-equal`
splits against their source found two of the three keys with such a lead-in
wrong, and the rule now refuses all three
(tests/fixtures/marking_split_handcheck.json).

Marks are whole numbers here, never halves
------------------------------------------
`MarkingPoint.marks` is an `int` in frontend/lib/domain/entities/question.dart,
and one record the phone cannot decode fails the whole corpus load before
runApp(). So a scheme that prints "½" against each step is NOT split: half a
mark cannot be carried, and `int(0.5)` is 0 -- the "[0m]" key that Task 121
refused to serve.

The question's own text is not its answer
-----------------------------------------
Two bleeds are stripped here, because both are the question printed inside its
own key:

  * `strip_stem_echo` -- the key opens with the stem. Measured on the served
    bank: 38 records share 20+ characters with the head of their stem, of
    which 30 are cut, because only a run that ends on a sentence boundary is.
    The 8 left alone are the ones where cutting would be wrong or would leave
    nothing: 3 answers RESTATE the premise in their own words ("Helium atom
    has 2 electrons in its outermost shell...") and diverge mid-sentence, 2
    are nothing BUT the question restated, and 3 diverge on a spelling ("bonne
    ordre" answered "bon ordre") before any sentence ends.
  * `strip_restated_questions` -- CBE's mark scheme "restates the question,
    then an Answer | Guidance table" (corpus/cbse_cbe.py). In a multi-part item
    every later part's question text lands in the middle of the key: 59 of the
    436 served CBE records read as questions with an answer between them, and
    none of the 1,640 parsed items does after this -- including the two that
    number their sub-questions "3 Explain ..." rather than "1 (c)". It runs in
    the parser, before the answer's 40-word cut, so 143 served CBE keys also
    reach further into their item than they used to -- the words the cut used
    to spend on the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Rule names, as they are recorded in `answerScheme.metadata.markSplit`.
ANY_OF = "any-of"
MARKS_EACH = "marks-each"
MARKS_PRINTED = "marks-printed"
MARKS_FOR = "marks-for"
STEP_MARKS = "step-marks"
LIST_EQUAL = "list-equal"

# The audit measured the stem bleed at "20+ characters of the stem at the START
# of the model answer", and that is the threshold kept here. _MIN_KEEP stops a
# cut that would leave no answer behind: an answer that is ONLY its question is
# a different fault, and `bank_merge.exclusion_reason` already refuses it
# (answer-is-question-text).
_MIN_ECHO = 20
_MIN_KEEP = 15
# The shortest run that can be cut off the head of an answer once a 20+
# character echo has been measured -- the echo's last sentence may be shorter
# than the echo ("Mettez au négatif:").
_MIN_CUT = 8

# A lead-in ("Hint—", "Various modes of weed control are", "Two examples to
# show "Play influences social and emotional 1X2=2 development"- Child learns
# to-") carries no mark and is not a value point. Longer than this and the text
# before the first item is answer content that no item accounts for, so the
# answer is not split at all. 120 is the shortest bound that keeps the SQP
# schemes whose lead-in is the question restated with its mark annotation.
_MAX_LEAD_IN = 120

# Length is not the only tell. A lead-in that ENDS A SENTENCE is a statement of
# its own, not an introduction to the list -- and no value point accounts for
# it. Three of the 126 splits the rules made before this one had such a
# lead-in (five further keys reach it, and were already refused further down),
# and two of the three were wrong for exactly that reason:
# cbse:sqp:ClassXII_2025_26:Sociology:24,
# where the board prints THREE bullets against 2 marks and the parser lost the
# first bullet's marker ("Workers get exhausted earlier than otherwise. * ... *
# ..."), so splitting the two that kept theirs put a mark on two of the board's
# three points and none on the third; and exemplar:q:9:mathematics:3:3.3:8,
# whose key ran on into its neighbours' answers ("C, D, E, G 10. (7, 0), (0,
# -7) 11. (i) ... (ii) ... (iii) ..."), where the list at the end belongs to
# question 11 and question 8's own answer sits in the lead-in. A lead-in that
# runs INTO its list ("... because they help in (a) ... (b) ...") does not end
# a sentence and still introduces it.
_STATEMENT_LEAD_IN = re.compile(r"[.!?][\"'”’)\]]*\s*$")

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# "Any two of the following", "any 3", "Any other, Any two" (the last one wins:
# CBSE's schemes close with the instruction that counts).
_ANY_OF = re.compile(
    r"\bany\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\b", re.I)

# "Any one role of each": an "any one" that is an instruction inside every
# item of the list ("give one role of EACH of a, b and c"), not an instruction
# over the list.
_ANY_ONE_EACH = re.compile(r"\bany\s+(?:one|1)\b[^.]{0,40}?\beach\b", re.I)

# "1 mark each", "½ mark each", "Award 1 mark for each point", "One mark each
# is to be awarded", "1 mark for every correct answer", "Each correct answer
# merits 1 mark".
_MARKS_EACH = re.compile(
    r"(\d{1,2}|½|half|one|two|three)\s*marks?\s+(?:is\s+to\s+be\s+awarded\s+)?"
    r"(?:for\s+|to\s+)?(?:each|every)\b", re.I)

# "Award 2 marks for a clear explanation", "For 2 marks, there must be some
# awareness of the contradiction": a level-of-response rubric, where the
# question's WHOLE marks are awarded for the quality of one answer. Its bullets
# are strands of that one answer, not value points -- cbe:q:English9MH13 and
# cbe:q:English9JV32 were split 1+1 against a guidance that never says so.
# "up to a maximum of 2 marks" is not this: it caps a per-point award.
_WHOLE_MARK_AWARD = re.compile(
    r"\b(?:award|awarded|for)\s+(\d{1,2})\s+marks?\b(?!\s+(?:for\s+|to\s+)?(?:each|every))", re.I)

# "Award 1 mark for a valid explanation of 'daunting'", "2 marks for the
# diagram": the board naming a value point and what it is worth. Read only as
# a SET -- two or more of them that account for the question's marks between
# them. One on its own is a level-of-response rubric ("Award 2 marks for a
# clear explanation") when it is worth the whole question, and a tolerance
# ("Allow 2 marks for the correct answer only") when it is not; neither
# divides the answer. `_WHOLE_MARK_AWARD` above still refuses any text where
# one award is worth every mark, which no set of two or more can be.
_MARKS_FOR = re.compile(
    r"(?<!\w)(\d{1,2}|one|two|three|four|five|½|half)\s*marks?\s+for\s+"
    r"(?!each\b|every\b)", re.I)
# How much of the sentence after "N marks for" is the component's name.
_MAX_COMPONENT = 120

# "M1", "A 1", "B2": the step codes a mark scheme prints down the side of a
# worked solution -- M for the method, A for the accurate result it reaches, B
# for a result that stands on its own -- with the marks that step earns. This
# is how CBSE's CBE item banks mark a calculation: "M1 4 x 2(12+8) OR
# equivalent OR 160 seen / M1 their 160 x 5 / A1 Rs 800" (cbe:q:Maths8DB4, 3
# marks), each step ending in the result it reaches. Read only when the codes
# account for the question's marks exactly and every one of them has a step
# written beneath it: "M1 A1 - to find correct inner radius and use correct
# formula" (cbe:q:Maths9GB8) prints two marks of four and one step for both.
# The code is uppercase because "28.2191 m2" is square metres, not a 2-mark
# step, and is not read inside a bracket because "=SUM(A1:A5)" is a cell.
_STEP_CODE = re.compile(r"(?<![A-Za-z0-9(\[])([MAB])\s?([1-9])(?![0-9])")

# The mark printed after a value point rather than once for the list -- "(i)
# Definition of refraction 1 (ii) Statement of the two laws 2". A bare number
# at the end of an item is only a mark when the item SAYS something without
# it: every Exemplar maths key whose items are answers ("a) 8 b) 13",
# "(i) 49 (ii) 28") ends each item in a digit that is the answer itself.
_TRAILING_MARK = re.compile(r"\s(\d{1,2})\s*$")
_MIN_ITEM_WORDS = 2

# "CASE STUDY - III", "case based": a question whose parts the paper's own
# general instructions weight unequally. Every CBSE sample paper that carries
# case studies says so on page 1 -- "Each case study comprises of 3 case-based
# questions, where 2 VSA type questions are of 1 mark each and 1 SA type
# question is of 2 marks" (Applied-Maths-SQP.pdf, 2022-23) -- and the per-part
# marks are printed in a margin column that extraction drops. So the parts of
# a case study are 1, 1 and 2, never equal, and dividing its marks equally is
# the one allocation known in advance to be wrong.
_CASE_STUDY = re.compile(r"\bcase[\s\-]?(?:stud(?:y|ies)|based)\b", re.I)

_MERITS_EACH = re.compile(
    r"\beach\b[^.]{0,40}?\b(?:merits|carries|gets|worth)\s*(\d{1,2}|½|half)\s*marks?", re.I)
# "1X2=2", "2x1=2", "½ x 4=2": one factor is the per-point value and the other
# the number of points, and which is which is not knowable from the product
# alone -- so this is read only for the TOTAL it states, and the per-point
# value comes from the list or from `any N`.
_PRODUCT = re.compile(r"(\d{1,2}|½)\s*[x×X]\s*(\d{1,2}|½)\s*=\s*(\d{1,2})")

# "1+1", "2 + 2 + 1", "1+1=2". A run of at least two addends.
_ADDENDS = re.compile(r"(?<![\d.=])(\d{1,2}|½)(?:\s*\+\s*(?:\d{1,2}|½))+(?:\s*=\s*(\d{1,2}))?")

# A key holding both branches of an internal choice: "(OR)", " OR ". Its items
# come from two different answers, so they cannot be one scheme's value points.
_INTERNAL_CHOICE = re.compile(r"(?:^|\s)\(?\s*OR\s*\)?(?:\s|$)")

# "1 (b)", "1 (c)(ii)": a CBE group item's sub-question label, still in the key
# after `strip_restated_questions` has taken the question text with it. The key
# then covers SEVERAL sub-questions and the record's marks are the group's
# total -- how that total divides between the sub-questions is printed against
# each one in the source, not in this text. cbe:q:Science9TS3 (8 marks) reads
# "1. Filtration and evaporation (2) or 2. Make aqueous solution ... (1)": two
# items, and 4 marks each is not what the board awarded.
_GROUP_PART = re.compile(r"\d\s*\(\s*[a-h]\s*\)")
# Any sub-part label, "(b)". The served bank's extraction repair (bank_merge,
# _LEAKED_PART_MARK, 2026-10-01) takes the "1 " off every label but the first,
# so a repaired group stem reads "1 (a) ... (b) ... (c)": one numbered label,
# then plain ones.
_SUB_PART = re.compile(r"\(\s*[a-h]\s*\)")

# "(1)", "(1 mark)", "(2 marks)": the per-point value CBE prints beside a value
# point. Read only to be reconciled -- a set of printed marks that does not fit
# the list refuses the split.
_PARENTHESISED_MARKS = re.compile(r"\(\s*(\d{1,2})\s*(?:marks?)?\s*\)", re.I)

# The trailing instruction of a CBSE list is not a value point of its own.
_TRAILING_INSTRUCTION = re.compile(
    r"\s*(?:Any other[^.]*|Any\s+(?:one|two|three|four|five|six|\d{1,2})\b[^.]*)\.?\s*$", re.I)

# CBE: "1 (b) <the question, restated> Answer Guidance <the answer>".
_PART_LABEL = r"(\d{1,2}\s*\(\s*[a-h]\s*\)(?:\s*\(\s*(?:i{1,3}|iv|v|vi)\s*\))?)"
_RESTATED = re.compile(_PART_LABEL + r"\s*.{0,400}?\bAnswer\s+Guidance\b\s*", re.I | re.S)
# Two of the 1,640 items flatten the same table with the answer BETWEEN the
# column headings -- "1 (b) Calculate the area of the given figure. Answer 7
# cm2 Guidance 28 cm2" (Maths6HK7). A lone "Answer" ends a restatement only
# when a "Guidance" follows close behind, and only after the adjacent-heading
# pass above has run: an answer that says "Answer: 11 (A)" would otherwise end
# the restatement early and leave the real heading standing.
_RESTATED_SPLIT_HEADING = re.compile(
    _PART_LABEL + r"\s*.{0,400}?\bAnswer\b\s*(?=.{0,200}?\bGuidance\b)", re.I | re.S)
# Two of the 1,640 items number their sub-questions with a bare "3 Explain
# what is meant by 'crowned by adversity' (lines 18-19)." rather than "1 (b)"
# -- English9NS1 and English8SM1. A bare number is not a label anywhere else
# in a key, so it is read as one only where the CBE layout puts one: after a
# sentence ends, before a capital letter, with the question's own sentence and
# then the Answer|Guidance heading behind it. Nothing else in a key reaches a
# heading, because the labelled pass above has already taken every heading
# that belongs to a labelled restatement.
_RESTATED_BARE = re.compile(
    r"(?:(?<=[.?!])|(?<=\n))\s*(\d{1,2})\s+(?=[A-Z])"
    r"(?:(?!\bAnswer\b).){5,300}?[.?]\s*\bAnswer\s+Guidance\b\s*", re.S)

# Sentence boundaries a stem and its echo can both end on.
_TERMINATORS = ".:?!)"

# The label that opens the first item of a list -- "a.", "(i)", "1)". A stem
# echo is never cut past one: the label belongs to the key's own structure.
_ECHO_LABEL = re.compile(r"(?<!\S)\(?(?:[a-h]|i{1,3}|iv|vi?|\d{1,2})\s*[.)]\s")

_BULLETS = "•●▪‣⁃"

# A value point has to SAY something. "(a), (b) and (c). The number on the
# right is greater." (exemplar:q:6:mathematics:3:-:71) is one sentence that
# names the parts it answers, and reading it as a list makes two value points
# out of "(a)," and "(b) and".
_CONNECTORS = {"and", "or", "but", "also", "then", "&"}


@dataclass(frozen=True)
class ValuePoint:
    description: str
    marks: int


@dataclass(frozen=True)
class Split:
    """The value points of one answer, and how they were arrived at."""

    points: tuple[ValuePoint, ...]
    rule: str = ""
    any_of: int | None = None
    marks: int = 0

    @property
    def all_or_nothing(self) -> bool:
        """One point worth every mark: the teacher can only give 0 or all."""
        return len(self.points) <= 1 and self.marks > 1


def split_answer(answer: str, marks: int, stem: str = "") -> Split:
    """The value points `answer` prints, for a question worth `marks`.

    `stem` is the question this key answers, when the caller has it: a CBE
    stem that prints two or more sub-parts ("1 (a) ... 1 (b) ... 1 (c) ...")
    carries the GROUP's marks, while its key holds only the first part's
    answer. cbe:q:SCIENCE8JS41c is 6 marks over three sub-parts and its key
    lists the three structural differences of part (a), which the source marks
    "Any two valid structural differences, 1 mark for each": 2 marks a bullet
    is not what the board awards.

    Returns a single point holding the whole answer when the source shows no
    split -- which is the honest reading of 96% of the served multi-mark bank,
    not a failure.
    """
    text = " ".join(str(answer or "").split())
    marks = int(marks or 0)
    if not text:
        return Split(points=(), marks=marks)
    if marks <= 1 or _GROUP_PART.search(text):
        return _one_point(text, marks)
    group_stem = str(stem or "")
    if _GROUP_PART.search(group_stem) and len(_SUB_PART.findall(group_stem)) > 1:
        return _one_point(text, marks)

    if _half_mark_split(text, marks):
        return _one_point(text, marks)
    if any(int(n) == marks for n in _WHOLE_MARK_AWARD.findall(text)):
        return _one_point(text, marks)

    # A worked solution's step codes are read BEFORE the internal-choice
    # guard, because "M1 4 x 2(12+8) OR equivalent OR 160 seen" is a maths
    # scheme saying which forms of one step to accept, not two answers in one
    # key. Both branches of a real internal choice carry the question's whole
    # marks, so their codes add up to twice it and the split is refused there.
    stepped = _step_marks(text, marks)
    if stepped is not None:
        return stepped
    if _INTERNAL_CHOICE.search(text):
        return _one_point(text, marks)

    listing = _list_items(text)
    lead_in, items, instruction = listing.lead_in, listing.items, listing.instruction
    if not items or len(lead_in) > _MAX_LEAD_IN or _STATEMENT_LEAD_IN.search(lead_in):
        # No list to divide, or text in front of it that the list does not
        # account for. A mark scheme can still name its value points ("Award 1
        # mark for a valid explanation of 'daunting'") where the answer beside
        # them flattened into prose.
        return _marks_for(text, marks) or _one_point(text, marks)

    # The instruction that governs a list is printed WITH the list -- in the
    # lead-in ("Two reasons ... are-") or after the last item ("Any other, Any
    # two") -- never inside one of its points. cbse:sqp:ClassX_2023_24:
    # Painting:14 lists six value points at "1 mark" each and says "Symbolic
    # meaning of any two" inside the fourth: reading that as the scheme's
    # instruction made every one of the six worth 3 marks.
    any_n = _any_of_count(f"{lead_in} {instruction}")
    if any_n is not None and any_n > len(items):
        # More awarded than listed: the list is not the scheme's alternatives.
        return _one_point(text, marks)
    if any_n == 1 and not _ANY_ONE_EACH.search(f"{lead_in} {instruction}"):
        # "Any one of the following" over a list: the board awards the
        # question's WHOLE marks to whichever one the student gives, so an
        # equal division tells a teacher that the rest of the answer is
        # missing when it is not -- and which single point carries them all
        # is not something a scheme can record. The key stays whole.
        # "Any one role of EACH of the following" is the other instruction:
        # one role per listed part, so the parts still divide the marks
        # (cbse:sqp:ClassXII_2022_23:Home Science:29).
        return _one_point(text, marks)
    if any_n is not None and any_n > 1 and any_n < len(items):
        # "Any two of the following" lists more ways than it awards. Each
        # listed way is worth the same, and only N of them are ever given.
        if marks % any_n:
            return _one_point(text, marks)
        per = marks // any_n
        return Split(points=tuple(ValuePoint(i, per) for i in items),
                     rule=ANY_OF, any_of=any_n, marks=marks)

    per_point = _marks_each(text)
    if per_point is not None:
        if per_point >= 1 and per_point * len(items) == marks:
            return Split(points=tuple(ValuePoint(i, per_point) for i in items),
                         rule=MARKS_EACH, marks=marks)
        return _one_point(text, marks)

    # A mark printed beside a point ("(1)", "(2 marks)") -- unless the list's
    # own labels ARE parenthesised numbers: "(1) Iron + Air + Water → Iron
    # oxide (2) Copper sulphate + Iron → ..." (exemplar:q:7:science:6:-:17) is
    # two word equations, not a 1-mark point and a 2-mark one.
    printed = ([] if listing.style == _PAREN_NUMBER
               else [int(m) for m in _PARENTHESISED_MARKS.findall(text)])
    if printed:
        # CBE prints each value point's mark beside it. Where they fit the
        # list, they ARE the split; where they do not, the board allocated
        # marks we cannot place, and an equal division would contradict them.
        if len(printed) == len(items) and sum(printed) == marks and all(p >= 1 for p in printed):
            return Split(points=tuple(ValuePoint(i, p) for i, p in zip(items, printed)),
                         rule=MARKS_PRINTED, marks=marks)
        return _one_point(text, marks)

    # The mark printed after each point instead of once for the list.
    after_items = _printed_after_items(items, marks)
    if after_items is not None:
        return after_items

    named = _marks_for(text, marks, items)
    if named is not None:
        return named

    addends = _addends(text, marks)
    if addends is not None:
        if len(addends) == len(items) and all(a >= 1 for a in addends):
            return Split(points=tuple(ValuePoint(i, a) for i, a in zip(items, addends)),
                         rule=MARKS_PRINTED, marks=marks)
        return _one_point(text, marks)

    if marks % len(items) == 0:
        if _CASE_STUDY.search(text) or _CASE_STUDY.search(str(stem or "")):
            # A case study's parts are 1, 1 and 2 by the paper's own general
            # instructions, so an equal division is wrong before we read it.
            return _one_point(text, marks)
        per = marks // len(items)
        return Split(points=tuple(ValuePoint(i, per) for i in items),
                     rule=LIST_EQUAL, marks=marks)
    return _one_point(text, marks)


def value_points(answer: str, marks: int, record_id: str,
                 stem: str = "") -> tuple[list[dict], dict]:
    """The bank's marking-point shape for `answer`, and what the scheme records.

    The second return value goes into `answerScheme.metadata`: `markSplit`
    names the rule (empty when none fired), `allOrNothing` is written only when
    it is true, and `anyOf` only when the source said so. A reader of the bank
    can then tell a key that was never split from one that could not be.
    """
    split = split_answer(answer, marks, stem)
    # Last gate before the bank sees it: a split whose points are the marking
    # note rather than the answer is refused whole, whichever rule produced it.
    if len(split.points) > 1 and any(_is_annotation(p.description) for p in split.points):
        split = _one_point(" ".join(str(answer or "").split()), marks)
    points = [{
        "id": f"{record_id}:mp{i}",
        "description": p.description,
        "marks": p.marks,
        "keyword": "",
        "isRequired": split.any_of is None,
        "synonyms": [],
    } for i, p in enumerate(split.points, 1)]
    extras: dict = {"markSplit": split.rule}
    if split.any_of is not None:
        extras["anyOf"] = split.any_of
    if split.all_or_nothing:
        extras["allOrNothing"] = True
    return points, extras


def awardable_marks(scheme) -> int:
    """The marks a teacher can actually award against this scheme.

    Every scheme's points must add up to its question's marks -- except under
    `any N`, where the scheme lists MORE points than it awards and the total a
    student can earn is N points of the listed value. `evaluate.py` already
    caps an award at the question's marks; this says what the cap should be.

    Accepts a `Split` or a stored `answerScheme` dict, so the builders and a
    check over the served file can ask the same question.
    """
    if isinstance(scheme, Split):
        marks = [p.marks for p in scheme.points]
        any_of = scheme.any_of
    else:
        marks = [int((p or {}).get("marks") or 0)
                 for p in (scheme.get("markingPoints") or [])]
        any_of = (scheme.get("metadata") or {}).get("anyOf")
    if not marks:
        return 0
    if any_of:
        return int(any_of) * marks[0]
    return sum(marks)


# --------------------------------------------------------------------------- #
# the question's own text, out of its key
# --------------------------------------------------------------------------- #

def strip_stem_echo(stem: str, answer: str) -> str:
    """`answer` without the run of its own question it opens with.

    Only a run that ends where a sentence does is cut -- after ".", ":", "?",
    "!" or ")" -- and only when 20+ characters match and something is left to
    read. An answer that restates the premise in its own words shares an
    opening with the stem but diverges mid-sentence, and is returned untouched.

    Spacing is ignored while the run is measured, because the scheme's copy of
    the question is typeset again rather than copied: "Complétez avec une
    préposition :" answers "Complétez avec une préposition:"
    (cbse:sqp:ClassXII_2024_25:French:10) and "y/en/ etc" answers "y/en/etc"
    (French:4C). Compared character by character the run ends at that space,
    short of the colon that closes the instruction, and the whole question
    stays in the key.
    """
    text = " ".join(str(answer or "").split())
    head = " ".join(str(stem or "").split())
    if not text or not head:
        return text
    dense = [(i, c.lower()) for i, c in enumerate(text) if not c.isspace()]
    stem_dense = [c.lower() for c in head if not c.isspace()]
    run = 0
    while run < min(len(dense), len(stem_dense)) and dense[run][1] == stem_dense[run]:
        run += 1
    shared = dense[run - 1][0] + 1 if run else 0
    if shared < _MIN_ECHO:
        return text
    # The echo stops where the answer's own list begins. A key that answers
    # "a. Ma femme a mis des pâtes ____ cuire" repeats the whole sub-question
    # around the word it fills in, so the run reaches past "a." -- and cutting
    # there takes the label with it, leaving four labelled items where the
    # scheme printed five (cbse:sqp:ClassXII_2024_25:French:10). A label at
    # the very start is the STEM's part label rather than the key's own list
    # (cbse:sqp:ClassXII_2022_23:Computer Science:23 answers "(a) ... (i) SMTP
    # (ii) PPP" with "(i) SMTP: Simple Mail Transfer Protocol"), so it caps
    # nothing.
    label = _ECHO_LABEL.search(text[:shared])
    if label and label.start() > 0:
        shared = label.start()
    # The run has to be 20+ characters to count as an echo at all, but the
    # sentence it ends on can be shorter: "Mettez au négatif:" is the whole of
    # the question that cbse:sqp:ClassXII_2024_25:French:7 repeats before
    # answering it, and 18 characters. What is cut still has to read as
    # something -- two words -- so a key opening "Yes." keeps it.
    cut = 0
    for i in range(shared, _MIN_CUT - 1, -1):
        if (text[i - 1] in _TERMINATORS and (i == len(text) or text[i].isspace())
                and len(re.findall(r"[^\W\d_]{2,}", text[:i])) >= 2):
            cut = i
            break
    if not cut or len(text) - cut < _MIN_KEEP:
        return text
    return text[cut:].strip()


def strip_restated_questions(answer: str) -> str:
    """A CBE key without the sub-questions its mark scheme restates.

    The CBE layout is "<part label> <the question, restated> Answer Guidance
    <the answer>" for every part after the first. The label is kept -- it says
    which part the answer that follows belongs to -- and everything between it
    and the Answer|Guidance heading goes, because by that layout it is the
    question, not the key.

    The bare-number pass runs last, over what the labelled passes left: by
    then any remaining Answer|Guidance heading belongs to a restatement that
    numbered its sub-question "3" rather than "1 (c)".
    """
    text = str(answer or "")
    if "Answer" not in text:
        return text
    # Line breaks are left alone: `cbse_cbe._ANSWER_ROW` reads the MCQ answer
    # row off the start of a LINE, so flattening the scheme text here would
    # make it swallow the whole tail.
    return _RESTATED_BARE.sub(
        r"\1 ", _RESTATED_SPLIT_HEADING.sub(r"\1 ", _RESTATED.sub(r"\1 ", text)))


# --------------------------------------------------------------------------- #
# reading what the source printed
# --------------------------------------------------------------------------- #

def _one_point(text: str, marks: int) -> Split:
    return Split(points=(ValuePoint(text, marks),), marks=marks)


def _find_label(pattern: re.Pattern, text: str, at: int):
    """The next occurrence of a list label that is not inside a bracket.

    "a)" is also how "(x + a)" ends. exemplar:q:8:mathematics:7:-:88 answers
    eighteen factorisations, "i) 6b (a + 2c) ii) – y (x + a) iii) x (ax2 - bx +
    c) ...", and reading the a, b and c of its expressions as a three-item
    list split a 3-mark question into three points of arithmetic.
    """
    while at <= len(text):
        m = pattern.search(text, at)
        if not m:
            return None
        if not _inside_brackets(text, m.start()):
            return m
        at = m.end()
    return None


def _inside_brackets(text: str, pos: int) -> bool:
    """Is `pos` inside a "(...)"? Counting brackets over the whole text cannot
    tell: every "i)" label closes a bracket that was never opened."""
    for ch in reversed(text[:pos]):
        if ch == "(":
            return True
        if ch == ")":
            return False
    return False


# A value point a teacher cannot mark against: the marking INSTRUCTION,
# chopped up, with no answer in it. cbse:sqp:ClassXII_2023_24:Informatics
# Practices:19 split into "correct explanation) (1 mark for correct example)"
# and "correct example)" -- the annotation was the only text the key held, so
# every rule that reads annotations found structure and none of it was answer.
# One lump a teacher can read beats two fragments they cannot.
_ANNOTATION_ONLY = re.compile(
    r"^[\s)\]]*(?:\(?\s*(?:½|\d+(?:[.,]\d+)?)\s*(?:/\s*\d+\s*)?marks?\b"
    r"|marks?\s+(?:for|each|awarded)\b|correct\s+(?:explanation|example|answer)\b)",
    re.I)


def _is_annotation(item: str) -> bool:
    """The point is the board's marking note rather than the answer to mark."""
    text = item.strip()
    if not text:
        return True
    if _ANNOTATION_ONLY.match(text):
        return True
    # "…) (1 mark for correct example)": a fragment that opens with the tail of
    # a bracket it never opened is a cut annotation, not a sentence.
    return text.startswith(")") or text.startswith("]")


def _says_something(item: str) -> bool:
    """An item's text, beneath its label, is more than punctuation and "and"."""
    words = [w for w in re.findall(r"[^\W_]+", item.lower()) if w not in _CONNECTORS]
    return bool(words)


def _any_of_count(text: str) -> int | None:
    """N from "any N", the LAST one printed ("Any other, Any two").

    "any one" is not an any-of instruction for a scheme: it names alternative
    wordings of ONE value point ("Sweat glands/salivary glands (any one)"), or
    a choice inside a sub-part, and reading it as the scheme's own instruction
    made every listed point worth the question's whole marks.
    """
    found = _ANY_OF.findall(text)
    if not found:
        return None
    token = found[-1].lower()
    return _WORD_NUMBERS.get(token) or (int(token) if token.isdigit() else None)


def _half_mark_split(text: str, marks: int) -> bool:
    """The board printed THIS question's marks as half a mark a point.

    cbse:sqp:ClassX_2022_23:Home Science:24 lists four ways of creating
    variety in meals under "½ x 4 = 2" and closes "Any other, Any two": the
    board's per-point value is half a mark, which a `MarkingPoint` cannot
    carry, so the key stays whole rather than claiming one mark a point. A
    half-mark product for some OTHER total ("½ x 4= 2" beside the four steps
    of part b of a 3-mark question) says nothing about this question's split
    and is ignored.
    """
    for m in _PRODUCT.finditer(text):
        if int(m.group(3)) == marks and "½" in (m.group(1), m.group(2)):
            return True
    return False


def _marks_each(text: str) -> int | None:
    """The per-point value the scheme prints, or None when it prints none.

    A half mark comes back as 0: it is printed, so the answer is not free of
    structure, but it cannot be carried (the phone's `marks` is an int), and
    the caller refuses the split rather than rounding it away.
    """
    m = _MARKS_EACH.search(text) or _MERITS_EACH.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    if token in ("½", "half"):
        return 0
    return _WORD_NUMBERS.get(token) or int(token)


def _step_marks(text: str, marks: int) -> Split | None:
    """The steps of a worked solution, each carrying the mark the board coded.

    A CBE mark scheme writes the working down the page with a code against
    every step -- "M1 4 x 2(12+8) OR equivalent OR 160 seen / M1 their 160 x 5
    / A1 Rs 800" (cbe:q:Maths8DB4) -- and the digit in the code is that step's
    marks. A student who reaches 160 and stops has earned one of the three.

    Accepted only when the codes account for the question's marks exactly and
    every code has a step written beneath it. "M1 A1 - to find correct inner
    radius and use correct formula" (cbe:q:Maths9GB8, 4 marks) fails both: two
    marks of four, and one step for two codes.
    """
    codes = list(_STEP_CODE.finditer(text))
    if len(codes) < 2:
        return None
    values = [int(m.group(2)) for m in codes]
    if sum(values) != marks:
        return None
    if len(text[:codes[0].start()].strip()) > _MAX_LEAD_IN:
        return None
    bounds = [m.start() for m in codes] + [len(text)]
    steps = []
    for i, code in enumerate(codes):
        if not _says_something(text[code.end():bounds[i + 1]]):
            return None
        steps.append(text[bounds[i]:bounds[i + 1]].strip())
    return Split(points=tuple(ValuePoint(s, v) for s, v in zip(steps, values)),
                 rule=STEP_MARKS, marks=marks)


def _marks_for(text: str, marks: int, items: tuple[str, ...] = ()) -> Split | None:
    """The value points the scheme names, with what it says each is worth.

    "Award 1 mark for a valid explanation of 'daunting'. Award 1 mark for a
    valid explanation of 'venture'." (cbe:q:English9AM33, 2 marks) is the
    board printing both of its points and both values; the answer beside them
    is a two-row table that flattens into one line, so nothing in the answer
    reads as a list.

    Read only as a SET that accounts for the question's marks. Where the list
    has exactly one item per award and every award is the same, the ITEMS are
    the descriptions -- they carry the answer, and the awards only say what
    each is worth. Otherwise the components the board named are.
    """
    found = list(_MARKS_FOR.finditer(text))
    if len(found) < 2:
        return None
    values = [_mark_value(m.group(1)) for m in found]
    if any(v < 1 for v in values) or sum(values) != marks:
        return None
    if items and len(items) == len(values) and len(set(values)) == 1:
        return Split(points=tuple(ValuePoint(i, values[0]) for i in items),
                     rule=MARKS_FOR, marks=marks)
    components = []
    for m in found:
        rest = text[m.end():m.end() + _MAX_COMPONENT]
        end = re.search(r"[.;]\s|[.;]$", rest)
        component = (rest[:end.start()] if end else rest).strip(" .;,")
        if not _says_something(component):
            return None
        components.append(component)
    if len(set(components)) != len(components):
        return None
    return Split(points=tuple(ValuePoint(c, v) for c, v in zip(components, values)),
                 rule=MARKS_FOR, marks=marks)


def _printed_after_items(items: tuple[str, ...], marks: int) -> Split | None:
    """The mark printed after each point of a list, rather than once for it.

    "(i) Definition of refraction 1 (ii) Statement of the two laws 2" against
    a 3-mark question. A bare number at the end of an item is a mark only when
    the item says something without it and the numbers account for the marks:
    every Exemplar maths key whose items ARE numbers -- "a) 8 b) 13"
    (exemplar:q:8:mathematics:3:-:97) -- ends each item in its answer.
    """
    tails = [_TRAILING_MARK.search(i) for i in items]
    if not all(tails):
        return None
    values = [int(t.group(1)) for t in tails]
    if any(v < 1 for v in values) or sum(values) != marks:
        return None
    points = []
    for item, tail in zip(items, tails):
        said = item[:tail.start()].strip()
        if len(re.findall(r"[^\W\d_]{2,}", said)) < _MIN_ITEM_WORDS:
            return None
        points.append(said)
    return Split(points=tuple(ValuePoint(p, v) for p, v in zip(points, values)),
                 rule=MARKS_PRINTED, marks=marks)


def _mark_value(token: str) -> int:
    """A printed mark as an int; a half mark comes back 0, which no rule
    accepts -- `MarkingPoint.marks` is an int on the phone."""
    token = token.lower()
    if token in ("½", "half"):
        return 0
    return _WORD_NUMBERS.get(token) or (int(token) if token.isdigit() else 0)


def _addends(text: str, marks: int) -> list[int] | None:
    """The addends of the split the scheme printed ("1+1", "2 + 2 + 1").

    Read only when they add up to the question's marks, and when a stated
    total ("=3") agrees: "= 8 + 10 = 18" inside a worked solution is
    arithmetic, not a marking split, and adds up to something else.
    """
    best: list[int] | None = None
    for m in _ADDENDS.finditer(text):
        run = m.group(0).split("=")[0]
        if "½" in run:
            return [0]          # printed, not carryable -- refuse the split
        parts = [int(p.strip()) for p in run.split("+") if p.strip().isdigit()]
        stated = m.group(2)
        if len(parts) < 2 or sum(parts) != marks:
            continue
        if stated is not None and int(stated) != marks:
            continue
        if best is None or len(parts) > len(best):
            best = parts
    if best is not None:
        return best
    # A product ("1X2=2") that states this question's total says the answer
    # carries a printed split even though it does not say which factor is the
    # per-point value; the list decides that.
    for m in _PRODUCT.finditer(text):
        if int(m.group(3)) == marks:
            return None
    return None


@dataclass(frozen=True)
class _Listing:
    """A labelled list found in an answer."""

    lead_in: str = ""
    items: tuple[str, ...] = ()
    instruction: str = ""
    style: str = ""


_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
_LETTERS = list("abcdefgh")
_NUMBERS = [str(n) for n in range(1, 11)]
_PAREN_NUMBER = "paren-number"

# Labels are matched CASE-SENSITIVELY, and the letter and roman styles are
# lowercase. CBSE's newer papers print an internal choice as "(A) ... (B) ...",
# in the stem and in the key, and each branch is worth the question's WHOLE
# marks: cbse:sqp:ClassXII_2025_26:Kathakali:9 ("(A) Name the kathakali
# Veshams. (OR) (B) Write a short note ...") was split into two 1-mark points,
# which halves both answers. Sub-parts of one question are lowercase
# ("(a) ... (b) ..."), as are an Exemplar answer's parts.
_STYLES = (
    ("paren-roman", _ROMAN, r"\(\s*{}\s*\)"),
    ("paren-letter", _LETTERS, r"\(\s*{}\s*\)"),
    (_PAREN_NUMBER, _NUMBERS, r"\(\s*{}\s*\)"),
    # A "1." or "a." label opens its item, so nothing but whitespace stands
    # before it. Without that, a coordinate ("= A(2,1)and(x2, y2)") reads as a
    # list -- cbe:q:Maths10RK8 was split into "1)and(x2, y2) A1 Point P = ..."
    # and "2) A1 P(3,-2) lies on line", with the "1+2" of a section formula
    # read as the marks.
    ("number-dot", _NUMBERS, r"(?<![^\s]){}\s*[.)](?!\d)\s*"),
    ("letter-dot", _LETTERS, r"(?<![^\s]){}\s*[.)]\s*"),
)


def _list_items(text: str) -> _Listing:
    """The lead-in before the list, its items with their labels, and the
    instruction printed after the last one ("Any other, Any two").

    A style is accepted only when its labels run in order from the first --
    (a) then (b) then (c). "(a), (c) and (e) — are physical changes. (b) and
    (d) are chemical changes" (exemplar:q:10:science:1:-:30) is not a list of
    five points; it is two sentences, and the out-of-order labels say so.

    The OUTERMOST list wins -- the one that starts earliest. A Home Science
    scheme (cbse:sqp:ClassX_2022_23:Home Science:27) answers "a. ... 1+2=3 ...
    b. ... 1.Blue is added 2.The water is stirred ..."; the four inner steps
    are how part b earns its 2 marks, and the split the board printed is over
    a. and b.

    A list whose first label STARTS AGAIN after the list began is two lists,
    which means one key holding the answers to two questions
    (cbse:sqp:ClassX_2023_24:Home Science:24, "... Any two <second question>
    ... Any two"). Its items are not one scheme's value points, so it is
    refused.
    """
    best: _Listing | None = None
    best_at = len(text)
    for name, labels, pattern in _STYLES:
        spans: list[tuple[int, int]] = []
        at = 0
        patterns = [re.compile(pattern.format(re.escape(la))) for la in labels]
        for label_pattern in patterns:
            m = _find_label(label_pattern, text, at)
            if not m:
                break
            spans.append((m.start(), m.end()))
            at = m.end()
        if len(spans) < 2:
            continue
        if patterns[0].search(text, spans[0][1]):
            continue
        # The label the list stopped before, printed somewhere else in the
        # text: the labels are not in order, so they are not this list's.
        # exemplar:q:8:science:2:-:8 prints its answers in two columns --
        # "(a) antibodies (c) Anthrax (b) tuberculosis (d) fermentation" --
        # and reading (a) then (b) as the whole list puts two answers in each
        # of two value points.
        if len(spans) < len(patterns) and patterns[len(spans)].search(text):
            continue
        bounds = [s for s, _e in spans] + [len(text)]
        items = [text[bounds[i]:bounds[i + 1]].strip() for i in range(len(spans))]
        items = [i for i in items if i]
        if len(items) != len(spans):
            continue
        if not all(_says_something(text[e:b]) for (_s, e), b in zip(spans, bounds[1:])):
            continue
        last = items[-1]
        items[-1] = _TRAILING_INSTRUCTION.sub("", last).strip()
        if not items[-1]:
            continue
        if spans[0][0] < best_at or (spans[0][0] == best_at and len(items) > len(best.items)):
            best = _Listing(text[:spans[0][0]].strip(), tuple(items),
                            last[len(items[-1]):], name)
            best_at = spans[0][0]
    if best:
        return best
    # A bulleted list carries no order to check, so the text before the first
    # bullet is the lead-in and every bullet after it is an item.
    chunks = re.split(f"[{_BULLETS}]", text)
    lead_in, items = chunks[0].strip(), [c.strip() for c in chunks[1:] if c.strip()]
    if len(items) >= 2:
        last = items[-1]
        items[-1] = _TRAILING_INSTRUCTION.sub("", last).strip()
        if items[-1]:
            return _Listing(lead_in, tuple(items), last[len(items[-1]):], "bullet")
    return _Listing()
