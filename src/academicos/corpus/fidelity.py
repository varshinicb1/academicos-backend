"""What the source extractions broke, repaired where the repair is certain and
refused where it is not (the bank rebuild of 2026-09-28).

The audits of 2026-09-27 and 2026-09-28 read every question the 20 standard
papers print and found 16 papers printing at least one that could not be
answered as printed (D31). Most of those came from the CBSE competency-based
item banks ("CBE"), whose PDF extraction:

  * printed each stacked fraction twice, numerator and denominator run
    together -- "Rohan painted 512512 of a wall" for 5/12 (D26, D74);
  * dropped the reciprocal from x + 1/x, leaving "(x + x) = Rs. 7", and 22/7
    from "Use pi = 22/7", leaving "Use pi = 7" (D26, D74);
  * set every letter of an equation in Mathematical Italic (U+1D400 block) and
    flattened exponents onto the line -- "12x2 + 11x - 15" (D21, D54);
  * kept the source's mark column at the end of the stem ("... places.   4)")
    and its "(Total 12 marks)" line, which contradict the marks the paper
    prints (D26, D76);
  * ran each answer key on into the next item's header ("44cm. Maths8PD4 This
    assessment item is designed to assess ...", D36) and printed Cambridge
    mark codes, M1/A1/B1, in the key (D20);
  * kept questions whose figure, graph or table never arrived ("The given bar
    graph represents ...", "Fig. 1 shows ...", D31).

The NCERT Exemplar class 10 Triangles chapter was set in a symbol font whose
angle, triangle, parallel and perpendicular signs extracted as C0 control
characters, which print as empty boxes (D45); and the SQP class 10 and 12
Mathematics papers set their equations in the same Mathematical Italic with
their fractions and matrices scrambled (D54).

`repair_source` makes the certain repairs and returns, as a reason, what it
found that cannot be repaired. `figure_missing` is the extended reading of a
stem that points at a visual it does not carry, for every source. Each rule is
counted by `scripts/merge_question_banks.py --check`.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any

CBE_SOURCE = "cbse_question_bank"
SQP_SOURCE = "cbse_sample_paper"

# The Mathematical Alphanumeric Symbols block, and the one italic letter
# Unicode keeps outside it (h, U+210E PLANCK CONSTANT).
_MATH_ALNUM = "\U0001D400-\U0001D7FFℎ"
MATH_ALNUM = re.compile(f"[{_MATH_ALNUM}]")

# ---- refusals -------------------------------------------------------------- #

# A fraction extracted twice: "512512" (5/12), "1313" (1/3), "56+2956+29"
# (5/6 + 2/9). Read on CBE only: in the other sources a repeated number is
# usually the number ("5050", "256256, 678678" in an Exemplar question about
# such numbers). A year ("2020 to 2040") is not a fraction.
_DOUBLED = re.compile(r"(?<![\d.,])(\d{2,4}|\d{1,3}\s?[+\-−×÷]\s?\d{1,3})\1(?![\d])")
_YEAR = re.compile(r"(?:19|20)\d\d")

# What the Mathematical Italic extraction lost, read on the italic text:
#   "(x + x) = Rs. 7", "x2 + x2 = 34"   x + 1/x with the reciprocal gone;
#   "(a + b) x (b + a)"                 (a + 1/b)(b + 1/a), the same;
#   "collecting th of the water"        "1/10th" with the numeral gone;
#   "(x 2 - 100)"                       x/2 flattened;
#   "sin3t + cos3t sint + cost"         a fraction's bar gone: numerator, space,
#                                       denominator.
_ITALIC_TOKEN = rf"[{_MATH_ALNUM}]\d?"
_LOST_STRUCTURE = re.compile("|".join((
    rf"({_ITALIC_TOKEN})\s*[+\-−]\s*\1(?![{_MATH_ALNUM}\w])",
    rf"\(\s*([{_MATH_ALNUM}])\s*\+\s*([{_MATH_ALNUM}])\s*\)\s*[×x·]?\s*\(\s*\3\s*\+\s*\2\s*\)",
    rf"(?<![\w{_MATH_ALNUM}])\U0001D461ℎ(?![\w{_MATH_ALNUM}])",
    rf"(?<![\w{_MATH_ALNUM}])[{_MATH_ALNUM}]\s\d(?![\d.,\w{_MATH_ALNUM}])",
    rf"[{_MATH_ALNUM}]\d?[{_MATH_ALNUM}]\s*[+\-−]\s*[{_MATH_ALNUM}]+\d?[{_MATH_ALNUM}]\s+[{_MATH_ALNUM}]",
)))
# "Use pi = 7" is 22/7 with the fraction gone; in any font.
_PI_AS_7 = re.compile(r"[π\U0001D70B]\s*=\s*7(?![\d.])")
# "breadth 4y2" beside options printed "4y³": an exponent flattened in a stem
# that prints the others. ASCII here; the italic ones are repaired.
_ASCII_EXPONENT = re.compile(r"(?<![A-Za-z])\d*[a-z][2-4]\b")
_SUPERSCRIPT = re.compile("[²³⁴]")
# A key for one part of a question of several: "B. Filtration" for a 4-mark,
# two-part question; "A. Milk" for three marks (D77).
_ONE_PART_KEY = re.compile(r"^\s*[A-D]\.\s")
_ONE_PART_KEY_MAX = 40
# Part labels run together ahead of the text they label: "1 (a) 1 (b) Two
# students ...", "Use the bar graph ... 1 (a) 1 (b) Which is ...". Which text
# belongs to which part is lost (13 served CBE items, all Mathematics; one was
# filling a "case based" Section E, audit D75).
_DETACHED_PARTS = re.compile(r"\([a-h]\)\s*1?\s?\([a-h]\)")

# ---- repairs --------------------------------------------------------------- #

# The item bank's header, printed before the question on some rows: "Item
# identity AO1 marks AO2 marks C/N/E* Content Reference(s) Marks Maths8PW4 N
# 8G1d ... Source information: book/journal, author, publisher, website link
# etc." The question follows it, or follows "Item purpose ... Question 4" /
# "... Questions" (a capital Q: the purpose itself says "The question
# assesses"). A header with no question after it is refused as a header.
_ITEM_HEADER = re.compile(
    r"^(?:This assessment item is designed\b[^.]*\.\s*)?"
    r"Item identity\b.*?(?:website link etc\.?|Item purpose\b.*?\bQuestions?\b(?:\s+\d+\b)?)\s*",
    re.S)
# The header's reference line, left between two parts of a question:
# "... Show your working.  Subject Class Question reference/Filename Maths
# Maths9DP8 1 (b) Calculate ...".
_REFERENCE_LINE = re.compile(r"\s*Subject Class Question reference/Filename\s+\S+\s+\S+")
# The source's mark column after the stem ("... places.   4)"), and after the
# key, and its per-question total.
_MARK_COLUMN = re.compile(r"\s{2,}\d{1,2}\)\s*$")
_TOTAL_MARKS = re.compile(r"\s*\(Total\s+(?:\d+\s+marks?|marks?\s+\d+)\)", re.I)
# Where a key runs into the next item: its id ("Maths8PD4") or its header.
_ITEM_ID = re.compile(r"\b(?:Maths?|Science|SCIENCE|English|Eng)\d{1,2}[A-Z]{1,4}\d{1,2}[a-z]?\b")
_NEXT_HEADER = re.compile(r"This assessment item is designed", re.I)
# Cambridge mark codes in a Mathematics key: M = method, A = accuracy, B =
# independent of method. Mathematics only: "B1" in a Science key is a vitamin.
_MARK_CODE = re.compile(r"\b([MAB])\s?(\d)\b")
_MARK_WORD = {"M": "method", "A": "answer"}
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
# The item bank's page footer.
_CALCULATOR_FOOTER = re.compile(r"\s*\*\s*C\s*=\s*Calculator required")
# Option labels printed with nothing after them: "... and why? (A) (B)".
_EMPTY_OPTIONS = re.compile(r"(?:\s*\([A-D]\))+\s*$")
# The Exemplar symbol font's C0 characters. \x01 is the angle sign before a
# vertex ("\x01A = 30°") and the parallel sign after a line's name ("PQ\x01
# RS"); \x02 sits after the triangle sign and is dropped.
_ANGLE = re.compile(r"(?<![A-Z])\x01\s?(?=[A-Z])")
_CONTROL = str.maketrans({"\x01": "∥", "\x02": "", "\x03": "⊥", "\x04": "Δ"})
CONTROL_CHARS = re.compile("[\x01-\x04]")


def restore_control_symbols(text: str) -> str:
    """`text` with the Exemplar symbol font's control characters restored
    (D45): 30 in one class 10 Triangles paper's text layer, printed as boxes."""
    if not CONTROL_CHARS.search(text):
        return text
    return _ANGLE.sub("∠", text).translate(_CONTROL)


def _plain_math(text: str, *, exponents: bool = True) -> str:
    """Mathematical Italic as the letters it styles, an exponent flattened
    after a letter raised again ("12x2 + 11x" -> "12x² + 11x", "cm3" -> "cm³",
    "sin3θ" -> "sin³θ"), and "36o" read as 36°. Measured on the CBE stems: every
    digit written straight after an italic letter was an exponent. Not in the
    keys (`exponents=False`): there it is as often a subscript -- "m1 : m2",
    "(x1, y1)" -- and an italic X is a multiplication sign ("1X5")."""
    if not MATH_ALNUM.search(text):
        return text
    text = re.sub(r"(?<=\d)\U0001D45C", "°", text)
    if exponents:
        # Not a decimal ("x2.5"): a full stop or comma after it ends the term.
        text = re.sub(rf"(?<=[{_MATH_ALNUM}])\d+(?!\d|[.,]\d)",
                      lambda m: m.group(0).translate(_SUPERSCRIPT_DIGITS), text)
    return MATH_ALNUM.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)


def _readable_codes(text: str) -> str:
    def word(m: re.Match) -> str:
        n = int(m.group(2))
        what = _MARK_WORD.get(m.group(1))
        marks = f"{n} mark" if n == 1 else f"{n} marks"
        return f"[{marks}: {what}]" if what else f"[{marks}]"
    return _MARK_CODE.sub(word, text)


def _cut_at_next_item(text: str, own: str) -> str:
    """`text` up to where it runs into another item: that item's id, or the
    item bank's header sentence."""
    cut = len(text)
    header = _NEXT_HEADER.search(text)
    if header:
        cut = header.start()
    for m in _ITEM_ID.finditer(text, 0, cut):
        if m.group(0).lower() != own.lower():
            cut = m.start()
            break
    return text if cut == len(text) else text[:cut].rstrip(" ;,–-")


def _walk(value: Any, fn) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _walk(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, fn) for v in value]
    return value


def scrambled_math(rec: dict) -> bool:
    """An SQP Mathematics stem set in Mathematical Italic: its fractions,
    matrices and exponents are scrambled ("If theta = 30o then the value of
    3tan theta is 1 3 A)1 B) C ) (D) not defined", D54). Read on the stem as
    served, after the merge has cut what ran on into it."""
    return (rec.get("source") == SQP_SOURCE and "math" in str(rec.get("subject") or "").lower()
            and bool(MATH_ALNUM.search(str(rec.get("stem") or ""))))


def _key_texts(rec: dict) -> list[str]:
    scheme = rec.get("answerScheme") or {}
    return [str(scheme.get("modelAnswer") or "")] + [
        str(mp.get("description") or "") for mp in scheme.get("markingPoints") or []
        if isinstance(mp, dict)]


def _reason(rec: dict) -> str | None:
    """What a CBE record lost that no repair can put back, read on the text as
    extracted -- the italic, before `_plain_math` makes it readable and hides
    the signature."""
    stem = str(rec.get("stem") or "")
    keys = _key_texts(rec)
    for text in (stem, *keys):
        for m in _DOUBLED.finditer(text):
            if not (len(m.group(0)) == 4 and _YEAR.fullmatch(m.group(0))):
                return "fraction-doubled"
    if _PI_AS_7.search(stem) or _LOST_STRUCTURE.search(stem):
        return "symbol-loss"
    if _DETACHED_PARTS.search(stem):
        return "parts-detached"
    # The options may be in the parts: cbe:q:Maths8BS2 prints "4y³" there.
    options = " ".join(str(p.get("text") or "") for p in rec.get("parts") or [] if isinstance(p, dict))
    if (str(rec.get("subject") or "").startswith("Math") and _SUPERSCRIPT.search(f"{stem} {options}")
            and _ASCII_EXPONENT.search(stem)):
        return "symbol-loss"
    marks = rec.get("marks") if isinstance(rec.get("marks"), (int, float)) else 1
    first = keys[0].strip()
    if (rec.get("type") != "mcq" and marks >= 2 and _ONE_PART_KEY.match(first)
            and len(first) < _ONE_PART_KEY_MAX):
        return "key-covers-one-part"
    return None


def repair_source(rec: dict) -> tuple[dict, list[str], str | None]:
    """A copy of a CBE, SQP or Exemplar record with the certain repairs made,
    the names of those made, and why it still cannot be served (or None).

    Runs before the merge's gates, so every gate reads the repaired text; the
    reason is read on the text as extracted, because the repairs hide what it
    looks for."""
    r = copy.deepcopy(rec)
    made: list[str] = []
    source = r.get("source")

    def each_text(fn, name: str, *, stem: bool = True, key: bool = True) -> None:
        changed = False
        if stem:
            new = fn(str(r.get("stem") or ""))
            changed = new != r.get("stem")
            r["stem"] = new
        scheme = r.get("answerScheme")
        if key and isinstance(scheme, dict):
            scheme = dict(scheme)
            new = fn(str(scheme.get("modelAnswer") or ""))
            changed = changed or new != (scheme.get("modelAnswer") or "")
            scheme["modelAnswer"] = new
            points = []
            for mp in scheme.get("markingPoints") or []:
                if isinstance(mp, dict):
                    new = fn(str(mp.get("description") or ""))
                    changed = changed or new != (mp.get("description") or "")
                    mp = {**mp, "description": new}
                points.append(mp)
            scheme["markingPoints"] = points
            r["answerScheme"] = scheme
        if changed:
            made.append(name)

    # Every string, as the symbol-font repair does: an Exemplar MCQ's options
    # live in its parts, and a key restored without them no longer matches one.
    restored = _walk(r, restore_control_symbols)
    if restored != r:
        r = restored
        made.append("control-symbols-restored")
    each_text(lambda t: _EMPTY_OPTIONS.sub("", t) if r.get("type") != "mcq" else t,
              "empty-option-labels", key=False)

    if source != CBE_SOURCE:
        return r, made, None

    own = str(r.get("id") or "").rsplit(":", 1)[-1]
    reason = _reason(r)
    each_text(lambda t: _REFERENCE_LINE.sub("", _ITEM_HEADER.sub("", t)), "item-header-removed",
              key=False)
    each_text(lambda t: _TOTAL_MARKS.sub("", t).strip(), "total-marks-removed")
    each_text(lambda t: _MARK_COLUMN.sub("", t).rstrip(), "mark-column-removed")
    each_text(lambda t: _cut_at_next_item(t, own), "key-cut-at-next-item", stem=False)
    each_text(_plain_math, "math-italic-plain", key=False)
    each_text(lambda t: _plain_math(t, exponents=False), "math-italic-plain", stem=False)
    parts = r.get("parts") or []
    fixed_parts = [{**p, "text": _plain_math(str(p.get("text") or ""))} if isinstance(p, dict) else p
                   for p in parts]
    if fixed_parts != parts:
        r["parts"] = fixed_parts
        if "math-italic-plain" not in made:
            made.append("math-italic-plain")
    if str(r.get("subject") or "").startswith("Math"):
        each_text(_readable_codes, "mark-codes-readable", stem=False)
    meta = r.get("metadata")
    if isinstance(meta, dict) and meta.get("contentReference"):
        # A topic label, listed by the MCP server: the page footer ran into it
        # ("... plane. *C = Calculator required, N = Calculator not allowed,
        # E = Either"), and on some rows the next item's header too.
        ref = _CALCULATOR_FOOTER.split(str(meta["contentReference"]))[0]
        ref = _cut_at_next_item(ref, own).strip()
        if ref != meta["contentReference"]:
            r["metadata"] = {**meta, "contentReference": ref}
            made.append("content-reference-cleaned")
    return r, list(dict.fromkeys(made)), reason


# The source's question number ("1 (a) Calculate ... 1 (b) ...") is left as it
# printed, deliberately. Removing it was tried on 2026-09-28 and undone:
#   * before the gates, it hid what `holds_group` and the MCQ-part gate read,
#     and a 1-mark item holding a 3-mark two-part question was served
#     (cbe:q:Maths6MG4);
#   * after them, it changed what the hand-checked value-point splitter
#     (corpus/marking_split.py) reads, and three keys split wrongly --
#     cbe:q:SCIENCE8JS41c's 6 marks spread over part (a)'s three bullets.
# A cosmetic change to text that other rules read is a change to those rules.


# ---- the visual a stem points at ------------------------------------------- #

# Read in addition to bank_merge._FIGURE_REF, on every source. Each alternative
# is a CBE shape that pattern did not read; none matched an Exemplar or board
# stem that is answerable as printed (measured 2026-09-28):
#   "Fig. 1 shows ...", "In figure 1 given above"   a numbered figure;
#   "The given bar graph represents", "Use the bar graph to answer", "The
#   following histogram shows"                      a graph named as shown;
#   "A food web is shown.", "Two lenses are shown." a visual announced;
#   "In the above triangle", "the above bar graph", "tabulated above",
#   "Diagram not drawn to scale";
#   "A student plots the current-voltage graphs"   a graph the student made.
_FIGURE_MORE = re.compile("|".join((
    r"\bfig(?:ure)?\.?\s*\d+(?:\.\d+)?\b",
    r"\b(?:the|use the)\s+(?:given\s+|above\s+|following\s+)?(?:bar\s+|double\s+bar\s+|line\s+)?"
    r"(?:graph|histogram|pictograph)s?\s+(?:shows?|represents?|given|below|above|to\s+answer)",
    r"\b(?:is|are)\s+shown\s*\.",
    r"\babove\s+(?:bar\s+|line\s+)?(?:triangle|figure|diagram|circle|quadrilateral|graph|chart|picture|image)\b",
    r"\btabulated\s+(?:above|below)\b",
    r"\bnot\s+drawn\s+to\s+scale\b",
    r"\b(?:student|he|she|they)\s+plot(?:s|ted)\b",
)), re.I)


def figure_missing(stem: str) -> bool:
    return bool(_FIGURE_MORE.search(stem))
