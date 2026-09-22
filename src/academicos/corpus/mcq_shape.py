"""Reading an MCQ from extracted text: its shape, its options and its key.

Pure functions over strings, shared by the bank composer (`bank_merge`) and,
from Task 15, the scheme verifier. Nothing here decides what is served; it
reads what a stem and a key say, and refuses to read what they do not say
unambiguously. The measured cases behind each rule are in the docstrings;
the gates that act on them, and their counts, are in `bank_merge`.
"""
from __future__ import annotations

import re
import unicodedata

# A scheme that joined the wrong text often carries the next question's stem.
INSTRUCTION = re.compile(
    r"^(?:read the (?:following|given|passage|extract|data|information)|fill in the blank|"
    r"find the odd one|select the (?:correct|appropriate)|choose the (?:correct|appropriate))",
    re.I)
_AWARD = re.compile(r"\s*\bAward\b.*$", re.I | re.S)

_PART_OPTION = re.compile(r"^\s*\(?([A-Da-d])\)?\s*[.)–—:\-]\s*(.*)$", re.S)
_LEADING_LETTER = re.compile(r"^\s*\(?([A-Da-d])\)?\s*(?:[.)–—:\-]|$)")
_LABEL = re.compile(r"(?<![A-Za-z])([A-D])\s*[.)–—:\-]")
_INLINE_OPTION = re.compile(r"\(([A-Da-d])\)\s*")
_UPPER_LABEL = re.compile(r"\(([A-D])\)")
# The labels of each option style, in order. Options are keyed A-D whatever
# they were printed as.
LETTERS = ("a", "b", "c", "d")
_ROMAN = ("i", "ii", "iii", "iv")
_NUMERALS = ("1", "2", "3", "4")
_ROMAN_LABEL = "iv|iii|ii|i"


def _style(label: str, punct: str) -> re.Pattern:
    # A label is a lone letter, numeral or roman numeral followed by a space:
    # the "e." of "i.e.", the "d." ending "and.", the "4." of "14." or "2.5"
    # are not one.
    return re.compile(rf"(?<![\w(.])({label}){punct}(?=\s)")


# The other ways SQP papers print options inline, each with its labels: "a) 1
# b) 3 c) 4 d) 2" (SQP Kathakali XII 2025-26 Q5), "a. Chappu b. Toppi ...",
# "A. FTP B. SFTP ...", "a] Rajasthan b] Gujarat" (SQP Bharatanatyam XII
# 2022-23 Q7), "i. Squirrel ii. Tiger iii. Swine iv. Sheep" (SQP Painting X
# 2023-24 Q6), "1. Kriti 2. Tana Varnam ..." (SQP Carnatic Music (Vocal) X
# 2024-25 Q5). One case and one punctuation mark per style, so a stem is read
# in the style it was printed in. "1." is also how a paper numbers its
# questions, so several questions fused into one stem read as numbered labels
# that repeat: they do not parse as options, but they are the MCQ shape, and
# the gate refuses them (SQP Kathak XII 2024-25 Q4 and its kind).
_OPTION_STYLES: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = (
    *((_style(letters, punct), LETTERS)
      for letters in ("[a-d]", "[A-D]") for punct in (r"\)", r"\.")),
    (_style("[a-d]", r"\]"), LETTERS),
    (re.compile(rf"(?<!\w)\(({_ROMAN_LABEL})\)\s*"), _ROMAN),
    *((_style(_ROMAN_LABEL, punct), _ROMAN) for punct in (r"\.", r"\)")),
    *((_style("[1-4]", punct), _NUMERALS) for punct in (r"\.", r"\)")),
)
_ALL_OPTION_STYLES = ((_INLINE_OPTION, LETTERS), *_OPTION_STYLES)
# "A)(0, 0)", "C )": a capital label as extraction spaced it, for the shape only.
_LOOSE_UPPER_LABEL = re.compile(r"(?<![\w(.])([A-D])\s?\)")
# A key written in a roman or numeric option's label: "(iii)", "iv. S.tail()",
# "3.", "2. Statement I & II both are false".
_KEY_LABEL = {labels: re.compile(rf"^\s*\(?({'|'.join(sorted(labels, key=len, reverse=True))})"
                                 r"\)?\s*(?:[.)\]:–—\-]|$)")
              for labels in (_ROMAN, _NUMERALS)}
# "(1)", "(1 mark)" after an answer is the mark column, not part of the answer.
_MARK_NOTE = re.compile(r"(?:\s+\(\s*(?:\d+|½)\s*(?:marks?)?\s*\))+\s*$", re.I)
# "3" alone may be the number of option (3) in a numbered list, not option text.
_BARE_NUMERAL = re.compile(r"^\(?[1-4]\)?\.?$")
_DASHES = str.maketrans(dict.fromkeys("−–—‐‑", "-"))
# "1 (a) ...", "1(a) ...": the first part of a numbered group, not an MCQ in
# its own right -- cbe:q:Science9DT3 carried its group's 8 marks this way.
SUB_PART = re.compile(r"^\s*(?:\d+\s*)?\(\s*[a-z]\s*\)\s*\S")
# CBE prints each part of a numbered group under the group's number: "1 (a)",
# "1(b)". The same number before two different letters is a group's parts;
# "(a)8 (b)6 (c)12" (SQP Kathak XII 2022-23 Q7) is options with numeric text.
_GROUP_PART = re.compile(r"(?<![\w.])(\d+)\s*\(([a-z])\)")
MIN_OPTIONS = 3
_MAX_OPTION_TOKENS = 40
_TOKEN = re.compile(r"\w+|[^\w\s,.;:'\"‘’“”]")


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def tokens(text: str) -> list[str]:
    """Words and symbols, keeping everything `norm` throws away.

    `norm` reads "-48" and "48" alike, "2(x + y)" and "2(x – y)" alike, and
    in Indic text drops every vowel sign (they are not \\w). That is fine for
    finding a candidate and wrong for deciding one: in 15 of the served CBE
    MCQs a wrong option differs from the right one only by a sign, an operator
    or a bracket (measured 2026-09-21). Here signs, brackets and matras are
    tokens; commas, full stops and quotes are not, so "(2x+3), (3x-2)" still
    reads as "(2x+3) (3x-2)".
    """
    t = unicodedata.normalize("NFKC", _MARK_NOTE.sub("", text or ""))
    return _TOKEN.findall(t.casefold().translate(_DASHES).replace("&", " and "))


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return 0 < n <= len(hay) and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _named_option(words: list[str], options: dict[str, str]) -> str | None:
    """The one option these words name, or None when they name none or several.

    An exact reading wins. Failing that the words may quote part of an option,
    or carry one with something trailing ("... II is false SECTION-B") -- but
    only one option may fit.
    """
    toks = {k: tokens(v) for k, v in options.items()}
    exact = [k for k, t in toks.items() if t == words]
    if exact:
        return exact[0] if len(exact) == 1 else None
    near = [k for k, t in toks.items() if t and (_contains(words, t) or _contains(t, words))]
    return near[0] if len(near) == 1 else None


def answerable(options: dict[str, str]) -> bool:
    """Every option has text, no two read alike, and none runs on.

    Two options that read alike mean extraction lost what told them apart --
    cbe:q:Maths10AS4 prints 34, 34, 35, 35 with the key on the second 34 -- so
    the printed item is defective even where the key happens to be distinct.
    An option read from the stem ends where the stem ends, so when the stem
    ran on into the next section the last option carries it: SQP Home Science
    X 2024-25 Q14's option D is a whole case-study passage. The longest option
    in a served MCQ that did not run on is 18 tokens (measured 2026-09-22).
    """
    read = [tuple(tokens(v)) for v in options.values()]
    return (all(read) and len(set(read)) == len(read)
            and max(len(t) for t in read) <= _MAX_OPTION_TOKENS)


def options_from_parts(parts: list[dict]) -> dict[str, str] | None:
    """{'A': 'avoid', ...} from CBE/SQP option parts, or None if they are not A-D in order."""
    opts: dict[str, str] = {}
    for part in parts:
        m = _PART_OPTION.match(str(part.get("text") or ""))
        if not m:
            return None
        letter = m.group(1).upper()
        if letter in opts:
            return None
        # Text that opens with the NEXT option's label is two options fused in
        # extraction ("C. D. 7", cbe:q:Maths9IM4). Only the next label counts:
        # assertion-reason options legitimately open "(A) is correct but ...".
        if re.match(rf"\(?{chr(ord(letter) + 1)}[.)]\s", m.group(2)):
            return None
        opts[letter] = m.group(2).strip()
    if len(opts) < 2 or list(opts) != list("ABCD"[: len(opts)]):
        return None
    return opts


def _read_options(stem: str, matches: list[re.Match], minimum: int,
                  labels: tuple[str, ...]) -> dict[str, str] | None:
    found = [m.group(1).lower() for m in matches]
    if len(found) < minimum or found != list(labels[: len(found)]):
        return None
    return {"ABCD"[i]: stem[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(stem)]
            .strip(" .;") for i, m in enumerate(matches)}


def stem_options(stem: str, other_styles: bool = True
                  ) -> tuple[dict[str, str] | None, int, tuple[str, ...] | None]:
    """The options written inline in the stem, where the first one starts, and
    the labels they were printed with.

    (A)..(D) -- the board and most SQP shape -- is read first, as before; the
    other styles only when it does not parse, and only with three labels or
    more, so a stray "a)", "B." or "ii." in prose is never an option list.
    """
    matches = list(_INLINE_OPTION.finditer(stem))
    options = _read_options(stem, matches, 2, LETTERS)
    if options is not None:
        return options, matches[0].start(), LETTERS
    for style, labels in _OPTION_STYLES if other_styles else ():
        matches = list(style.finditer(stem))
        options = _read_options(stem, matches, MIN_OPTIONS, labels)
        if options is not None:
            return options, matches[0].start(), labels
    return None, -1, None


def options_from_stem(stem: str, other_styles: bool = True) -> dict[str, str] | None:
    """Options already written inline: (A)..(D), or any of `_OPTION_STYLES`."""
    return stem_options(stem, other_styles)[0]


def has_option_labels(stem: str) -> bool:
    """The stem is labelled like an MCQ, whether or not the labels parse.

    All four capital labels (A)-(D), in any order and any number of times:
    `options_from_stem` accepts only A, B, C, D once each in order, so two
    shapes read as "no options" and escaped the MCQ gate as short_answer:
    several questions fused into one stem (A-D repeated -- SQP Mathematics
    (Standard) X 2025-26 Q18 carries a tangent MCQ and the assertion-reason
    directions after it), and assertion-reason items, whose "Assertion (A)"
    puts an A before the options. 21 served 1-mark SQP rows had this shape at
    062a939, and at least 9 of their keys answered a different question.

    Or the first three labels, consecutively, in any one style -- (a), a),
    a., A), A., a], (i), i., i), 1., 1) -- since the capital (A) test never
    saw the others: served 1-mark SQP rows printed "a) 1 b) 3 c) 4 d) 2" or
    "i. Squirrel ii. Tiger iii. Swine iv. Sheep" and stayed short_answer with
    keys nobody checked (see `bank_merge`'s docstring for the counts).

    Or all four labels of one style in any order, as for (A)-(D): extraction
    that reads a two-column option grid by column prints "c) ... d) ... a) b)"
    (SQP Applied Mathematics XII 2022-23 Q14) or "b. ... a. ... c. ... d."
    (SQP Tangkhul XII 2023-24 Q7). And "A) (0, 0) B) (-4, 0) C ) (-5, 0) D)"
    (SQP Mathematics (Standard) X 2024-25 Q15), a space before the bracket or
    none after it, is the same shape: it does not parse, and is refused.
    """
    if {"A", "B", "C", "D"} <= set(_UPPER_LABEL.findall(stem)):
        return True
    if set("ABCD") <= set(_LOOSE_UPPER_LABEL.findall(stem)):
        return True
    for style, labels in _ALL_OPTION_STYLES:
        found = [m.group(1).lower() for m in style.finditer(stem)]
        if (set(labels) <= set(found)
                or any(found[k:k + 3] == list(labels[:3]) for k in range(len(found)))):
            return True
    return False


def holds_group(stem: str) -> bool:
    """Two or more parts of one numbered group: "1 (a) ... 1 (b) ..."."""
    letters: dict[str, set[str]] = {}
    for m in _GROUP_PART.finditer(stem):
        letters.setdefault(m.group(1), set()).add(m.group(2))
    return any(len(v) >= 2 for v in letters.values())


# A capital A-D standing alone in option text: a statement's label ("A & B"), a
# matching list's row ("A - I"), an assertion ("A is true but R is false").
# "Ahead" or "Bach" is a word, not a label.
_LETTER_IN_TEXT = re.compile(r"(?<![A-Za-z])[A-D](?![A-Za-z])")


def names_option_letter(options: dict[str, str]) -> bool:
    """Some option's text names a letter the options could be printed with.

    Numbered or roman options printed as (A)-(D) then read "(C) B & C": a
    student cannot tell the option from the statement, and a scorer that
    collects every A-D letter in an answer (evaluate.py, the phone's
    answer_evaluation.dart) reads "(C) B & C" as B and C.
    """
    return any(_LETTER_IN_TEXT.search(v) for v in options.values())


def correct_letter(answers: list[str], options: dict[str, str]) -> str | None:
    """The correct option, only when the scheme states it unambiguously.

    Refuses rather than guesses. Two failure shapes are common in the SQP
    schemes and both would yield a confident wrong key if read loosely:
    a matching-question key ("A - III; B - IV; ..."), and question text that
    the scheme parser joined in place of the answer ("modernity A. 1 & 3 B. ...",
    "Read the following data and select ..."). A wrong key is worse than none:
    a teacher trusts it and marks students wrong.

    A letter is not enough on its own either. When the scheme writes words after
    it, those words must name the same option. Taking the letter alone shipped
    keys the scheme itself contradicts -- "A. 243" where 243 is option C,
    "C. Melt" where Melt is option D -- and letters for items that were never
    MCQs ("(a) Na2CO3 (1) (b) ..." is a two-part answer, not option A).
    """
    for raw in answers:
        text = _AWARD.sub("", raw or "").strip()
        if not text:
            continue
        if len(set(_LABEL.findall(text))) >= 2 or INSTRUCTION.match(text):
            return None
        m = _LEADING_LETTER.match(text)
        if m:
            letter = m.group(1).upper()
            if letter not in options:
                return None
            words = tokens(text[m.end():])
            if words and _named_option(words, options) != letter:
                return None
            return letter
        if _BARE_NUMERAL.match(text):
            return None
        hits = [k for k, v in options.items() if norm(v) and norm(v) == norm(text)]
        if len(hits) == 1 and tokens(options[hits[0]]) == tokens(text):
            return hits[0]
        return None
    return None


def key_letter(answers: list[str], options: dict[str, str],
                labels: tuple[str, ...]) -> str | None:
    """`correct_letter` for options printed with roman or numeric labels.

    The scheme keys them by the printed label -- "(iii)", "3.", "iv. S.tail()"
    -- so the label is read as its letter and the words after it must still
    name that option. Refused: a letter ("C" names no numbered option), and a
    bare label that is also an option's text ("3" among "1. 7 2. 8 3. 3").
    As in `correct_letter`, the first non-empty answer decides.
    """
    pattern = _KEY_LABEL.get(labels)
    if pattern is None:
        return correct_letter(answers, options)
    for raw in answers:
        text = _AWARD.sub("", raw or "").strip()
        if not text:
            continue
        m = pattern.match(text)
        if m is None:
            return None if _LEADING_LETTER.match(text) else correct_letter([text], options)
        rest = text[m.end():]
        if not rest.strip() and any(tokens(v) == [m.group(1)] for v in options.values()):
            return None
        return correct_letter([f"{'ABCD'[labels.index(m.group(1))]}. {rest}"], options)
    return None


# A part label a stem may legitimately open with, lowercase: "i. Explain ...",
# "a) Give reasons ...", "a, b, c kala d ..." (SQP Tangkhul X 2023-24 Q9), and
# "a Trouvez la question." (SQP French X 2022-23 Q6-8, the full stop lost).
PART_LABEL_OPENING = re.compile(r"^\(?(?:[a-h]|[ivx]{1,4})\)?(?:[.),:]|\s+(?=[A-Z]))")
# A label that is not the first of its list: "ii.", "b)", "(b)", "B.". A
# capital followed by another initial is a name ("B. R. Ambedkar"), not a label.
LATER_PART_OPENING = re.compile(
    r"^\(?(?:(?:viii|vii|vi|iv|iii|ii|ix|x|v|[b-h])[.)\]]|[B-H][.)](?!\s*[A-Z]\.))\s")
