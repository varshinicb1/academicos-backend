"""Compose the served question bank from its three sources.

The served bank -- academicos-data/syllabus/questions.json and its byte-identical
APK copy, frontend/assets/corpus/questions.json -- is what every surface reads.
It is composed from:

  board  records already in it with source == "cbse_board_paper", and those
         an earlier run withheld (academicos-data/syllabus/board_withheld.json)
  cbe    academicos-data/corpus/cbse-cbe/questions.json   (classes 6-10)
  sqp    academicos-data/corpus/cbse-sqp/questions.json   (classes 10, 12)

A board record is served only with a marking scheme the relink verified
(no-verified-key), and only if a student can answer it as printed: it goes
through the same `exclusion_reason` as CBE and SQP, after `repair_board` has
made the repairs that are certain -- page furniture stripped, lowercase or
mark-prefixed options reprinted "(A) ... (D)" inline, the four standard
assertion-reason options appended, the correct option always the verified
one. A hand check of 30 served stems (audit 3.6, 2026-09-21) found 18 that
could not be answered as printed, and the web path's own filters (moved here
from pool.py) never ran on the served file. Every board record not served,
whatever the reason, is kept WHOLE in board_withheld.json (`hold`), which every
run adds to and none overwrites, and read back as a board source on the next
run: the served file is both this module's board input and its output, so a
record left out of it would otherwise leave the pipeline for good, and a
later, better relink or repair could never bring it back.
CBE and SQP records are normalised to the phone's contract and then gated.

Why the contract is the phone's and not the API's
--------------------------------------------------
`QuestionSchema` accepts all 720 CBE records; the phone's `Question.fromJson`
rejects 259 of them (parts with no `textLatex`) and all 1,834 SQP records (no
`competencyIds`, no `bloomLevel`). One rejected row fails the whole corpus load,
before runApp(), so the app never leaves its splash screen. The two contracts
have drifted, so this enforces the stricter one. `contract_errors` mirrors
frontend/lib/domain/entities/question.g.dart; a Dart test that decodes the real
asset is the final word.

Why a value is checked, not just its presence
---------------------------------------------
The phone's enum converters do not fail on an unknown value -- they default.
An unknown `source` becomes `ncert_textbook`, which would print a CBSE sample
paper question as an NCERT textbook one. So values are checked against the sets
the converters accept.

What is excluded, and why
-------------------------
Nothing is dropped silently; every exclusion is returned with its reason.

  figure-unavailable     no surface can resolve a `cbe-figure:` asset; for a
                         board record, a stem that points at a figure, map,
                         graph or table and carries no asset id
  passage-unavailable    a comprehension question with no passage to read
  header-as-stem         the "question" is a page header
  stem-too-short         no question text to speak of
  no-answer              rule Q1: no answer key, no question
  answer-is-question-text  the scheme joined the wrong text -- often another
                         question's stem
  answer-bleed           an answer far longer than its marks allow, or one that
                         runs on past a section heading: the scheme ran on
                         into the following questions
  figure-referenced      the stem points at a figure, graph, map or a table
                         "above" that was not extracted with it
  garbled-script         legacy-font extraction produced text nobody can read
                         -- on a board stem, anywhere in it (`_legacy_font`)
  symbol-loss            a math symbol the extraction dropped or misread:
                         "(Use = 3·14)", "(a ¹ b)", "c c b b a a = = ."
  private-use-glyph      a symbol-font character nothing can name (corpus/
                         symbol_font.py): the ones the Adobe Symbol encoding
                         covers are restored (symbol-font-restored), and a
                         record still holding one prints a hole where a symbol
                         belongs
  options-missing        an objective stem -- typed MCQ, ending in a colon, or
                         a 1-mark "which of the following" -- with no options
  case-study-without-question  a case study, or a stem announcing a text,
                         that asks nothing
  page-furniture         a board stem that is nothing once its page furniture
                         ("# 14| P a g e", "666 -11 6 of") is stripped
  mcq-marks-implausible  an MCQ carrying more than 2 marks: a sub-part holding
                         its whole group's marks, or not a single MCQ at all
  mcq-part-of-group      an MCQ whose stem opens "1 (a)": one part of a group
  mcq-options-unresolved the options cannot be read as A-D, fewer than three of
                         them, the stem and the parts disagree about them, or a
                         student could not choose between them (empty, or two
                         read alike)
  mcq-answer-unresolved  the correct option cannot be recovered with certainty;
                         for a board record, the relink verified no letter
                         for its options, or one they do not print
  option-labels-collide  numbered or roman options whose text names A-D -- "1.
                         A & B 2. A & C" over statements A-D -- which would read
                         "(C) B & C" once printed with the letters the key uses
  stem-spans-sections    a section heading inside the stem: several questions,
                         across a section break, extracted as one
  stem-empty-after-strip nothing, or only a header, is left of the stem once
                         its page furniture is stripped
  stem-fragment          the stem starts mid-sentence or with a later part's
                         label ("ii.", "(b)"), or an MCQ's or 1-mark item's
                         stem opens with its options: the question text went
                         elsewhere; or a board stem kept its numbers and
                         labels and lost its words ("21 cm 120° 1 (A) ...")
  stem-runs-on           the next question of the same paper is a stem-fragment:
                         this stem lost its ending to it or took its opening,
                         and its key is likely another question's
  stem-holds-group       a 1-mark item whose stem prints its group's numbered
                         parts, "1 (a) ... 1 (b) ...": several questions under
                         one part's mark, keyed with one part's answer
  duplicate-id           an id already served
  duplicate-stem         already served, under a board-paper id where possible;
                         exactly, or near-identically (`NEAR_DUPLICATE_MIN`)
  key-rejected           a CBE, SQP or Exemplar record whose own key the
                         relink's verifier rejects (question_bank.
                         builder_key_reason): an assertion-reason letter for a
                         stem that prints no options, a key in another
                         script. The relink runs again after the merge; served,
                         the record would lose its key there and stay served
                         with none (17 at 8e92ed9)
  no-verified-key        a board record whose scheme the relink did not verify
                         (not cbse_marking_scheme, or no relink stamp -- see
                         `has_verified_key`): correctness first (user
                         decision, 2026-09-22) -- rule Q1 applied to board
                         records. The relink (enrich_question_bank.py)
                         decides what is verified; a label from before it
                         fails closed. The record is kept whole in
                         board_withheld.json.

An item is an MCQ by its shape, not only by its declared type: the SQP source
types every row `short_answer`, and 92 served 1-mark SQP rows carry four
inline (A)-(D) options. Those are retyped `mcq` (marked `typeInferred`) and go
through the same gate as a declared MCQ. The shape is the label set -- all of
(A), (B), (C), (D) present -- not a clean parse of them, so a stem whose labels
repeat or come out of order is still an MCQ, and is refused as one. Three
consecutive labels in any other style -- a), a., A), A., (a) -- are the shape
too: at c45acef 141 served 1-mark SQP rows printed them and were served as
short_answer with keys nobody checked. Gated as MCQs, 80 get a certain letter
and 61 are refused (34 mcq-answer-unresolved, 15 mcq-options-unresolved, 12
stem-fragment); one of the 80, SQP Carnatic Music (Percussion) X 2022-23 Q3,
then goes as stem-runs-on. A CBE "1 (a) 1 (b) 1 (c)" group is not this shape:
its labels are parts, and a 1-mark item holding them is stem-holds-group.

Those styles were letters only. At fa8b78b 90 more served 1-mark SQP rows
printed roman, numbered or bracketed options -- "i. Squirrel ii. Tiger ...",
"1. Kriti 2. Tana Varnam ...", "a] Rajasthan b] Gujarat" -- keyed by a bare
label ("(iii)", "3.", "b"), and 6 printed letters out of order or spaced
("c) ... d) ... a) b)", "C )"). Gated as MCQs, with the key's label read as
its letter and the options printed as (A)-(D), 38 of the 90 get a certain
letter (16 roman, 21 numbered, 1 bracketed) and 52 are refused (28
mcq-options-unresolved -- several questions fused under numbered labels,
column-scrambled options, or options that read alike once case is folded --
23 mcq-answer-unresolved, 1 stem-fragment); all 6 disordered ones are
mcq-options-unresolved. No served 1-mark non-MCQ SQP row carries three
labels of one style any more.

Relabelled, those roman and numbered options can collide with their own
text. At 7b3aae4, 5 served rows printed options that name A-D: statements A-D
(SQP Carnatic Music (Melodic Instrument) X 2023-24 Q6, "1. A & B 2. A & C
..."), a matching list's rows A-D (SQP Hindustani Music (Vocal) XII 2023-24
Q6), and 3 assertion-reason items ("i. Both A and R are true ..."). The paper
used numbers to keep the option apart from the statement; printed as (A)-(D)
they read "(C) B & C". Keeping 1-4 would not help: `QuestionPart` could carry
the label, but a served MCQ keeps its options inline and is scored by
`answerScheme.metadata.correctOption`, and both scorers (evaluate.py and the
phone's answer_evaluation.dart) read only the letters A-D in an answer. So
these rows are option-labels-collide: the 5, and SQP Painting XII 2023-24 Q4
(until then stem-runs-on), whose option iv runs on into the next question's
"Assertion (A)".

How a stem's options and a key are read is `mcq_shape`'s; what is served, and
why not, is this module's.

Known limitation
----------------
A non-MCQ key is checked for its shape (empty, instruction text, too long,
running into another section), never for its content. A key taken from a
different question that is short, and does not open with an instruction or
cross a section heading, is still served. No gate here reads whether the key
answers its stem. TODO(Task 15): the scheme verifier checks a key against its
stem; do not build a second one here.
"""
from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any

from ..assessment.question_bank import (CONFLICT_HELD, OfficialAnswer,
                                         build_official_scheme, builder_key_reason)
from .mcq_shape import (
    INSTRUCTION, LATER_PART_OPENING, LETTERS, MIN_OPTIONS, PART_LABEL_OPENING, SUB_PART,
    answerable, has_option_labels, holds_group, key_letter, names_option_letter, norm,
    options_from_parts, stem_options, tokens,
)
from .symbol_font import PRIVATE_USE, private_use_glyph, restore_symbol_font

BOARD_SOURCE = "cbse_board_paper"
EXEMPLAR_SOURCE = "ncert_exemplar"
NO_VERIFIED_KEY = "no-verified-key"
# A CBE, SQP or Exemplar key the relink would remove (question_bank.builder_key_reason).
KEY_REJECTED = "key-rejected"
# A key whose value points are all worth 0 marks: the paper prints every point
# as "[0m]", so it tells a teacher nothing about how to award the marks. The
# re-audit of 2026-09-22 found 38% of the then-live bank like this.
SCHEME_MARKS_UNALLOCATED = "scheme-marks-unallocated"

# The values each phone converter accepts (frontend/lib/domain/entities/enums.dart).
BLOOM_LEVELS = frozenset({"remember", "understand", "apply", "analyze", "evaluate", "create"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
QUESTION_TYPES = frozenset({
    "mcq", "very_short_answer", "short_answer", "long_answer", "very_long_answer",
    "case_study", "assertion_reason", "map_based", "diagram_based", "graph_based",
    "table_based", "competency_based"})
SOURCES = frozenset({
    "cbse_board_paper", "cbse_sample_paper", "cbse_question_bank", "ncert_exemplar",
    "ncert_textbook", "school_database", "teacher_created", "ai_generated",
    "competency_framework"})
ANSWER_TYPES = frozenset({
    "singleChoice", "multipleChoice", "textShort", "textLong", "numeric", "diagram",
    "map", "graph", "table"})


class ContractError(Exception):
    """A normalised record still breaks the phone's contract -- a normaliser bug."""

    def __init__(self, record_id: str, errors: list[str]) -> None:
        super().__init__(f"{record_id}: {'; '.join(errors)}")
        self.record_id = record_id
        self.errors = errors


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _is_iso(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _check(obj: dict, where: str, spec: dict[str, str], errs: list[str]) -> None:
    for key, kind in spec.items():
        present = key in obj and obj[key] is not None
        value = obj.get(key)
        optional = kind.endswith("?")
        kind = kind.rstrip("?")
        if not present:
            if not optional:
                errs.append(f"{where}.{key} missing")
            continue
        ok = {
            "str": isinstance(value, str),
            "num": _is_num(value),
            "bool": isinstance(value, bool),
            "strlist": _is_str_list(value),
            "list": isinstance(value, list),
            "dict": isinstance(value, dict),
            "iso": _is_iso(value),
        }[kind]
        if not ok:
            errs.append(f"{where}.{key} is not {kind}")


_QUESTION = {
    "id": "str", "questionBankId": "str", "subject": "str", "grade": "num",
    "chapterIds": "strlist", "competencyIds": "strlist", "bloomLevel": "str",
    "difficulty": "str", "type": "str", "stem": "str", "stemLatex": "str",
    "parts": "list", "answerScheme": "dict", "estimatedTimeMinutes": "num",
    "marks": "num", "language": "str", "source": "str", "qualityScore": "num",
    "tags": "strlist", "createdAt": "iso", "updatedAt": "iso", "metadata": "dict?",
    "diagramAssetId": "str?", "mapAssetId": "str?", "graphAssetId": "str?",
    "tableAssetId": "str?",
}
_PART = {
    "id": "str", "partNumber": "num", "text": "str", "textLatex": "str", "marks": "num",
    "answerType": "str", "options": "strlist?", "correctOption": "str?",
    "expectedAnswer": "str?", "expectedAnswerLatex": "str?", "keywords": "strlist?",
    "alternativeAnswers": "strlist?",
}
_SCHEME = {
    "totalMarks": "num", "markingPoints": "list", "rubricLevels": "list",
    "commonErrors": "strlist", "alternativeAnswers": "strlist", "modelAnswer": "str",
    "modelAnswerLatex": "str", "hasPartialCredit": "bool?", "metadata": "dict?",
}
_POINT = {"id": "str", "description": "str", "marks": "num", "keyword": "str",
          "isRequired": "bool", "synonyms": "strlist?"}
_RUBRIC = {"level": "num", "label": "str", "minMarks": "num", "maxMarks": "num",
           "description": "str"}
_ENUMS = {"bloomLevel": BLOOM_LEVELS, "difficulty": DIFFICULTIES,
          "type": QUESTION_TYPES, "source": SOURCES}


def contract_errors(rec: dict) -> list[str]:
    """Every way this record would fail, or mis-decode on, the phone."""
    errs: list[str] = []
    _check(rec, "question", _QUESTION, errs)
    for key, allowed in _ENUMS.items():
        if isinstance(rec.get(key), str) and rec[key] not in allowed:
            errs.append(f"question.{key}={rec[key]!r} is not one of the phone's values")
    for i, part in enumerate(rec.get("parts") or []):
        if not isinstance(part, dict):
            errs.append(f"parts[{i}] is not an object")
            continue
        _check(part, f"parts[{i}]", _PART, errs)
        if isinstance(part.get("answerType"), str) and part["answerType"] not in ANSWER_TYPES:
            errs.append(f"parts[{i}].answerType={part['answerType']!r} is not one of the phone's values")
    scheme = rec.get("answerScheme")
    if isinstance(scheme, dict):
        _check(scheme, "answerScheme", _SCHEME, errs)
        for i, mp in enumerate(scheme.get("markingPoints") or []):
            _check(mp if isinstance(mp, dict) else {}, f"markingPoints[{i}]", _POINT, errs)
        for i, rl in enumerate(scheme.get("rubricLevels") or []):
            _check(rl if isinstance(rl, dict) else {}, f"rubricLevels[{i}]", _RUBRIC, errs)
    return errs


# --------------------------------------------------------------------------- #
# stem gates shared with the web registry path
# --------------------------------------------------------------------------- #
# Moved here from assessment/pool.py (Task 16), unchanged: pool's registry path
# and its baked-bank path call these, and so does the board gate below, so the
# web paper and questions.json are held to one implementation of each.

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


_OPTION_ORDER = ("(a)", "(b)", "(c)", "(d)")
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
    extraction destroyed the earlier options (see pool's `has_broken_fraction_options`)
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
    is ASCII/Latin-1 and slips past pool's `_is_mostly_non_latin` — but it is unreadable
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


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

_HEADER = re.compile(
    r"^(?:PART\s*[-–]|Page\b|Class\s*[-–]|Subject\s*[-–]|Time allowed|"
    r"Maximum marks|General Instructions|Draw neat figures|Use of calculators|"
    r"Internal choice is provided)", re.I)
# A stem that points at a visual it does not carry. Visuals are matched in any
# position ("given figure", "figure below"); a table only when it sat ABOVE the
# stem -- a "following table" is extracted inline, as text, after it. 50 served
# CBE/SQP stems matched this with needsFigure false (measured 2026-09-22).
_VISUAL = r"(?:figure|fig\.|graph|diagram|picture|image|map|chart|photograph)"
_FIGURE_REF = re.compile(
    rf"\b(?:{_VISUAL}\s+(?:(?:given|shown)\s+)?(?:above|below)"
    rf"|(?:above|below|given|following|adjoining|adjacent)\s+{_VISUAL}"
    rf"|{_VISUAL}\s+(?:given|shown)\b"
    r"|table\s+(?:(?:given|shown)\s+)?above|above\s+table)", re.I)
_PASSAGE_REF = re.compile(r"\bread the (?:following )?(?:passage|extract|poem|text)\b", re.I)
# The 2025-26 SQPs repeat this note in every page footer; 46 served stems
# carried it, 22 of them MCQs where it ran into the last option.
_FOOTER = re.compile(
    r"\s*\*?\s*Please note that the assessment scheme of the Academic Session \d{4}-\d{2}"
    r" will continue in the current session i\.?\s*e\.?,? \d{4}-\d{2}\.?\s*\*?"
    r"(?:\s*Page\s+(?:\d+\s+)?of\s+\d+)?", re.I)
# The next section's heading, run on after the last question of a section
# ("(d) Agriculture SECTION B Question numbers 18-23 are SA type questions.").
# The heading may carry one parenthesised title -- "SECTION C (SHORT ANSWER
# QUESTIONS)", "SECTION D (Literature)" -- but an option or item label after it,
# "(A) avoid", "(ii) ...", is content, not a title.
# Capitals may drop the dash ("SECTION B"); mixed case keeps it ("(Section – D)")
# so that "Section 39 of the Act" or "Section A of the paper" in prose is not one.
_HEADING = r"\(?(?:SECTION\s*[-–]?|Section\s*[-–])\s*[A-F]\b\)?"
# Only a heading AFTER text: at the start of the stem the `[^()]*` tail would
# take the question with it. "Section – C Page of 7 The following HTML
# statements ..." (SQP Computer Applications X 2024-25 Q19) was served with
# stem '' that way. A leading heading is `_LEADING_SECTION`'s.
_TRAILING_SECTION = re.compile(
    rf"\s+{_HEADING}[^()]*(?:\((?!(?:[A-Da-d]|[ivxIVX]+)\))[^()]*\)[^()]*)?$")
# In a key, only a heading and its title are furniture: "SECTION C (SHORT ANSWER
# QUESTION)", "SECTION C POLITICAL SCIENCE (20 marks)". Prose after a heading --
# "3. SECTION-B Brief explanation and types of Kan, Gamak ..." -- is the next
# section's answers, and is left in for `_INNER_SECTION` to refuse; stripped
# the looser way, that key would read "3.".
_TITLE = r"(?:\s*(?:\((?!(?:[A-Da-d]|[ivxIVX]+)\))[^()]*\)|[A-Z]{2,}\b))*"
# The section's own instructions may follow its title ("Section – B Each question
# carries 2-mark weightage", SQP Applied Mathematics XII 2022-23 Q20); they are
# furniture too. Anything else after a heading is not.
_NOTE_OPENING = (r"[\[(]?\s*(?:Each (?:question|[Cc]ase [Ss]tudy)|Question numbers?|"
                 r"This section|consists of|comprises|All questions)\b")
_SECTION_NOTE = rf"(?:\s*{_NOTE_OPENING}.*)?"
# What is left of "SECTION B This section consists of 6 questions of 2 marks
# each." once `_LEADING_SECTION` takes the heading: the instructions, and no
# question.
_SECTION_NOTE_STEM = re.compile(rf"^{_NOTE_OPENING}")
_TRAILING_HEADING_ONLY = re.compile(rf"(?:^|\s+){_HEADING}{_TITLE}{_SECTION_NOTE}\s*$",
                                    re.S)
# A heading that opens the text goes with its title and nothing else.
_LEADING_SECTION = re.compile(rf"^{_HEADING}{_TITLE}\s*")
# A heading still inside the text once the trailing one is gone: the text runs
# on past its own question into another section.
_INNER_SECTION = re.compile(rf"(?:^|\s){_HEADING}")
# A page number left anywhere in the text: "Page of 10", "Page 3 of 12", or a
# bare "Page of" where the number was lost.
_PAGE = re.compile(r"\s*\bPage\s+(?:\d+\s+)?of(?:\s+\d+\b)?")
_MIN_STEM = 15
_GARBLE_RUNS = 3


def _answer_text(rec: dict) -> str:
    scheme = rec.get("answerScheme") or {}
    points = " ".join(str(p.get("description") or "") for p in scheme.get("markingPoints") or [])
    return f"{scheme.get('modelAnswer') or ''} {points}".strip()


def _first_answer(rec: dict) -> str:
    scheme = rec.get("answerScheme") or {}
    if (scheme.get("modelAnswer") or "").strip():
        return scheme["modelAnswer"].strip()
    points = scheme.get("markingPoints") or []
    return str(points[0].get("description") or "").strip() if points else ""


def _vowel_sign_runs(text: str) -> int:
    """Places where two dependent vowel signs sit side by side.

    In any Indic script a vowel sign (matra) attaches to a consonant, so two in a
    row never occur in correctly spelled text. Legacy-font extraction produces
    them constantly: "নিিো".

    Two cases that would otherwise be false positives, both handled:
      * only characters NAMED as vowel signs count. Visarga and anusvara are
        spacing/nonspacing marks too, and "दुःख" (Hindi,
        "sorrow") correctly puts visarga straight after a vowel sign.
      * the text is NFC-composed first. Bengali "ো" decomposes to two
        vowel signs (U+09C7 U+09BE); decomposed-but-correct text must not count.
    """
    runs = 0
    prev = False
    for ch in unicodedata.normalize("NFC", text):
        is_sign = "VOWEL SIGN" in unicodedata.name(ch, "")
        if is_sign and prev:
            runs += 1
        prev = is_sign
    return runs


# The script a language paper is written in, as the first word of the Unicode
# character name. Manipuri is set in Bengali script or in Meetei Mayek.
_SUBJECT_SCRIPTS: dict[str, frozenset[str]] = {
    "malayalam": frozenset({"MALAYALAM"}), "odia": frozenset({"ORIYA"}),
    "hindi": frozenset({"DEVANAGARI"}), "sanskrit": frozenset({"DEVANAGARI"}),
    "marathi": frozenset({"DEVANAGARI"}), "nepali": frozenset({"DEVANAGARI"}),
    "assamese": frozenset({"BENGALI"}), "bengali": frozenset({"BENGALI"}),
    "manipuri": frozenset({"BENGALI", "MEETEI"}), "tamil": frozenset({"TAMIL"}),
    "telugu": frozenset({"TELUGU"}), "telugutelangana": frozenset({"TELUGU"}),
    "kannada": frozenset({"KANNADA"}), "gujarati": frozenset({"GUJARATI"}),
    "punjabi": frozenset({"GURMUKHI"}), "urdu": frozenset({"ARABIC"}),
    "persian": frozenset({"ARABIC"}), "arabic": frozenset({"ARABIC"}),
    "tibetan": frozenset({"TIBETAN"}), "bhutia": frozenset({"TIBETAN"}),
    "lepcha": frozenset({"LEPCHA"}), "limboo": frozenset({"LIMBU"}),
}
# Music and dance papers may quote a composition in its own script.
_ARTS = re.compile(r"music|dance|kathak|odissi|bharatanatyam|k[uh]+chipudi|mohiniyattam|painting",
                   re.I)
# Below this share of letters in its own script, a language paper was extracted
# through a legacy font. Correct papers measured >= 0.39 (Persian, whose
# instructions are English); every legacy-font one measured <= 0.01.
_OWN_SCRIPT_MIN = 0.2


def _is_indic(ch: str) -> bool:
    return "\u0900" <= ch <= "\u0dff"      # Devanagari through Sinhala


def _script_garbled(rec: dict, text: str) -> bool:
    """Text whose script cannot be right for its subject.

    Two failures that `_vowel_sign_runs` never sees, because neither produces
    Indic vowel signs at all (measured on the served CBE/SQP rows, 2026-09-22):

      * a language paper extracted through a legacy font reads as Latin-1:
        Malayalam "X∂ncn°p∂ ]Øv tNmZyßfn¬", Odia "Êz bûeZ @bò~û^". All 10
        served Malayalam rows, and the Odia, Lepcha, Limboo and Manipuri ones,
        have almost no letters in their own script.
      * a maths font misread as Indic code points: 29 served Mathematics rows
        carry Oriya/Tamil/Telugu characters, many of them unassigned, where
        the paper printed x, y and fractions.

    Music and dance papers are exempt: they are written in English and may
    quote a composition in its own script.
    """
    subject = str(rec.get("subject") or "").strip()
    if _ARTS.search(subject):
        # Before the language table: "Manipuri Dance" is an English paper.
        return False
    first = re.split(r"[\s(]", subject.lower(), maxsplit=1)[0]
    scripts = _SUBJECT_SCRIPTS.get(first)
    if scripts is not None:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False
        own = sum(unicodedata.name(ch, "").split(" ", 1)[0] in scripts for ch in letters)
        return own / len(letters) < _OWN_SCRIPT_MIN
    return any(_is_indic(ch) for ch in text)


_ASSET_KEYS = ("diagramAssetId", "mapAssetId", "graphAssetId", "tableAssetId")
# Board stems name their visual in more ways than `_FIGURE_REF` and pool's
# `needs_missing_figure` list: "connected ... as shown in figure" (Physics XII
# 55/1/2 Q17), "Find the area of the shaded region". 0 of the 3,286 served
# board records carries an asset id (audit 3.6, 2026-09-21).
_BOARD_FIGURE = re.compile(
    r"\bshown in (?:the\s+)?(?:adjoining\s+|given\s+|following\s+)?"
    r"(?:figure|fig\.?|diagram|graph|map|picture)\b|\bshaded region\b", re.I)

# Signatures of a math symbol the extraction dropped or read as another
# character, each measured on the served board records (2026-09-22, 3,286
# records; ids are the paper hash and question number):
#   "(Use = 3·14)", "(Use 3 = 1·732)"  pi or the root sign gone (16 stems carry
#                   "(Use ...)"; "(Use p = 3·14)" keeps the Symbol font's pi as
#                   p, which a student reads, and is not matched)
#   "(a ¹ b)"       the Symbol font's not-equal read as a superscript one
#                   (f01c7b6f:5, c0791b53:14, c0791b53:3)
#   "= = ." / "tan A = ;"  a fraction's right side gone (f319c93a:19, f319c93a:34,
#                   fc5d1667:19, b2736b37:23, b165af61:20)
#   "c c b b a a"   a1/a2 = b1/b2 = c1/c2 flattened into letter pairs, and
#                   "q q" for theta over theta (f319c93a:19, 6ab66ce4:28)
#   "D ABC ~ D PQR" the triangle sign read as D; "Ð A" the angle sign
#                   (f319c93a:34, 0916a372:17, dedcd4e0:22)
#   "contains m³ of air"  the quantity before the unit gone (4f618224:30)
_SYMBOL_LOSS = re.compile("|".join((
    r"\(Use\s+(?:=|-?\d+\s*=)",
    r"\S\s¹\s\S",
    r"=\s*(?:[.;,]|=)(?=\s|$)",
    r"\b([a-z])\s\1\b.*?\b([a-z])\s\2\b",
    r"\bD\s?[A-Z]{3}\s*~|~\s*D\s?[A-Z]{3}\b|\b[Ii]n a D\s?[A-Z]{3}\b|Ð\s?[A-Z]\b",
    r"\bcontains\s+(?:mm|cm|km|m)[²³]",
)))

# A stem that ends in a colon promises what follows it -- options, items to
# classify, a list -- and one that did not arrive leaves nothing to answer:
# "Classify the following items ... Companies Act, 2013 :" (Accountancy XII
# 67/1/2 Q31, verified). A mark value may follow the colon.
_ENDS_WITH_COLON = re.compile(r":\s*(?:\d{1,2}\s*)?$")

# A stem that announces a text and questions on it. Not "case study" alone: a
# section's instructions say it ("Section E consists of 3 case study based
# questions", run on after SQP Mathematics (Standard) X 2024-25 Q35).
_ANNOUNCES_QUESTIONS = re.compile(
    r"\b(?:read the (?:following |given )?(?:passage|extract|source|case|text|poem|paragraph)s?"
    r"|answer the (?:following )?questions?)\b", re.I)
# What a question says. Read only on the text after the announcement, and only
# its last `_CASE_TAIL` characters: a case study's questions follow its text.
_ASKS = re.compile(
    r"\?|\b(?:find|calculate|compute|solve|explain|state|name|write|give|identify|describe|"
    r"determine|show|prove|list|mention|define|compare|evaluate|analy[sz]e|justify|suggest|"
    r"how|why|what|which|who|whom|whose|when|where|draw|examine|assess|discuss|choose|"
    r"select|complete|fill|express|obtain|derive|estimate|predict|infer|highlight|elaborate|"
    r"distinguish|differentiate|classify|interpret|comment|enumerate|outline|illustrate|"
    r"verify|convert|represent|construct|plot|rewrite|frame|summari[sz]e|translate)\b"
    # A blank to fill, an assertion to judge, or a numbered sub-question --
    # "(ii)", "(35.1)" -- is a question too; the text a case study quotes is not
    # numbered that way, its paragraphs are "(1)", "(2)".
    r"|_{3,}|\bAssertion\b|\((?:i{1,3}|iv|v|vi{1,3}|\d+\.\d+)\)", re.I)
_CASE_TAIL = 250

# A board stem needs two words to be a question. The Hindi-medium copy of a
# bilingual paper arrives with its words dropped and its numbers, units and
# point labels kept: "21 cm 120° 1 (A) 231 cm² ..." (Mathematics X 430/2/2
# Q17, verified), ", DE || BC AD = 2.8 cm ..." (430/3/3 Q16), "ABCD AB, BC,
# CD DA P, Q, R S AOB + COD = 180 .". So a word here has two letters or more
# and is not all capitals (a point or line label), a unit or a roman numeral.
# Measured on the 3,286 served board stems (2026-09-22): 278 have fewer than
# two, and read as those do; "Prime factorisation of 424 is : (A) ..." has
# four, where counting only words of three letters or more gave it two.
_WORD = re.compile(r"[^\W\d_]{2,}")
_NOT_A_WORD = re.compile(r"[ivx]+|[IVX]+|c?m|mm|km|kg|m?[lL]")
_MIN_WORDS = 2


def _has_asset(rec: dict) -> bool:
    return any(str(rec.get(k) or "").strip() for k in _ASSET_KEYS)


def _words_lost(stem: str) -> bool:
    """Fewer than `_MIN_WORDS` words before the options, or in the whole stem
    when it has none -- or opens with its labels: "(a) Explain ... (b)
    Suggest ..." is a question in parts, not options."""
    options, at, _ = stem_options(stem, other_styles=False)
    head = stem[:at] if options is not None and at > 0 else stem
    words = [w for w in _WORD.findall(head) if not w.isupper() and not _NOT_A_WORD.fullmatch(w)]
    return len(words) < _MIN_WORDS


def _legacy_font(text: str) -> bool:
    """Legacy-font Devanagari anywhere in an English stem.

    `looks_mangled` reads the first 600 characters, and a board paper prints
    the Hindi copy AFTER the English: Physics XII 55/4/1 Q34 reads cleanly for
    800 characters, then "(J) {ÛY«wd AmKyU© 6 10⁻⁷ C-m H$m H$moB© ...". So
    every 300-character window is read."""
    return any(looks_mangled(text[i:i + 300]) for i in range(0, max(len(text), 1), 300)
               if len(text[i:i + 300].strip()) >= 60)


def _options_missing(rec: dict, stem: str) -> bool:
    """An objective stem with no options a student could choose from.

    Objective: typed MCQ, ending in a colon, or a 1-mark item worded as an
    MCQ ("which of the following", "choose the correct option"). A record
    with parts carries its content there and is not read from its stem. A
    stem with the whole label set printed has options, whether or not they
    parse: cbe:q:Science10GK2 "1 (a) Which statement ... (A) ... (D) ..."
    labels its sub-question too, and whether such options can be read is the
    MCQ gates' to say, not this one's."""
    if rec.get("parts") or stem_options(stem)[0] is not None or has_option_labels(stem):
        return False
    marks = rec.get("marks") if _is_num(rec.get("marks")) else 1
    return (rec.get("type") == "mcq" or bool(_ENDS_WITH_COLON.search(stem))
            or (marks == 1 and bool(_MCQ_LEAD_IN.search(stem))))


def _case_without_question(rec: dict, stem: str) -> bool:
    """A case study, or a stem announcing a text, that asks nothing.

    "Read the following passage carefully : 1 Floods are not new to India ...
    often there is very little time" (English X c53f84b1 Q1) stops inside the
    passage; "Rainbow is an arch of colours that is visible in the sky after
    rain" (Mathematics X 94cb306f Q36) is the case's first line and nothing
    else. A CBE/SQP item that carries its passage apart, in metadata, is its
    question and is not read here.

    After an announcement only the last `_CASE_TAIL` characters are read: a
    passage asks questions of its own ("Who doesn't love to sled and build
    snowmen ?", English X 7e81f931 Q1). With none, the whole stem is: SQP
    Geography XII 2022-23 Q17, typed case_study, asks "Which of the following
    ...?" and ends in its options. A paper in its own script is not read: its
    questions are not in English (SQP Persian XII 2025-26 Q5)."""
    if str((rec.get("metadata") or {}).get("passage") or "").strip():
        return False
    subject = str(rec.get("subject") or "").strip().lower()
    if re.split(r"[\s(]", subject, maxsplit=1)[0] in _SUBJECT_SCRIPTS:
        return False
    announced = list(_ANNOUNCES_QUESTIONS.finditer(stem))
    if rec.get("type") != "case_study" and not announced:
        return False
    if stem_options(stem)[0] is not None or has_option_labels(stem):
        return False    # an MCQ asks by its options
    if not announced:
        return not _ASKS.search(stem)
    return not _ASKS.search(stem[announced[-1].end():][-_CASE_TAIL:])


def exclusion_reason(rec: dict) -> str | None:
    """Why this record cannot be served, or None if it can.

    Every source goes through both halves: board records after `repair_board`;
    CBE and SQP records raw, `source_reason` before `normalise` and
    `unanswerable_reason` after it, so that `normalise`'s more specific
    reasons -- stem-fragment, the MCQ gates -- still name what is wrong with
    them (SQP Computer Applications X 2024-25 Q20 is a stem-fragment; it
    also ends "answer the following questions: 3")."""
    return source_reason(rec) or unanswerable_reason(rec)


def unanswerable_reason(rec: dict) -> str | None:
    """Why a student could not answer this stem as printed (Task 16): a symbol
    the extraction lost, options that never arrived, a case study that asks
    nothing."""
    stem = str(rec.get("stem") or "").strip()
    if _SYMBOL_LOSS.search(stem):
        return "symbol-loss"
    if _options_missing(rec, stem):
        return "options-missing"
    if _case_without_question(rec, stem):
        return "case-study-without-question"
    return None


def source_reason(rec: dict) -> str | None:
    """The gates read before a CBE/SQP record is normalised. A few read board
    records only, and say why."""
    meta = rec.get("metadata") or {}
    stem = str(rec.get("stem") or "").strip()
    passage = str(meta.get("passage") or "").strip()
    board = rec.get("source") == BOARD_SOURCE
    if meta.get("needsFigure"):
        return "figure-unavailable"
    if rec.get("source") == "cbse_question_bank" and rec.get("subject") == "English" and not passage:
        # Every CBE English item is read against a text; without it none can be answered.
        return "passage-unavailable"
    if not passage and _PASSAGE_REF.search(stem) and len(stem) < 400:
        return "passage-unavailable"
    if _HEADER.match(stem):
        return "header-as-stem"
    if len(stem) < _MIN_STEM:
        return "stem-too-short"
    if _FIGURE_REF.search(stem) and not _has_asset(rec):
        return "figure-unavailable" if board else "figure-referenced"
    if board and (needs_missing_figure(stem) or _BOARD_FIGURE.search(stem)) and not _has_asset(rec):
        # Board only: pool's list also names a "following table", which CBE and
        # SQP extract inline, as text (test_a_following_table_is_inline_and_kept);
        # the web path has refused board stems by this list since it existed.
        return "figure-unavailable"
    answer = _answer_text(rec)
    if (_vowel_sign_runs(stem) >= _GARBLE_RUNS or _vowel_sign_runs(answer) >= _GARBLE_RUNS
            or _script_garbled(rec, f"{stem} {answer}")):
        return "garbled-script"
    if board and not _ARTS.search(str(rec.get("subject") or "")) and _legacy_font(stem):
        # Board only: CBE and SQP language papers are in their own scripts, and
        # `_script_garbled` already reads them.
        return "garbled-script"
    if board and _words_lost(stem):
        # Board only: CBE and SQP stems are single-language and their fragments
        # are `_stem_after_strip_reason`'s.
        return "stem-fragment"
    if not answer:
        return "no-answer"
    if INSTRUCTION.match(_first_answer(rec)):
        return "answer-is-question-text"
    marks = rec.get("marks") if _is_num(rec.get("marks")) else 1
    if len(answer) > max(600, 450 * max(int(marks), 1)):
        return "answer-bleed"
    return None


# --------------------------------------------------------------------------- #
# MCQs
# --------------------------------------------------------------------------- #

# CBE authors some 2-mark MCQs (71 in the raw bank); none worth more than 2 was
# a single MCQ when read (measured 2026-09-22).
_MCQ_MAX_MARKS = 2
# The declared types an SQP row carries when it is really an MCQ.
_INFERABLE_TYPES = frozenset({"short_answer", "very_short_answer"})


def _objective_scheme(rec: dict, options: dict[str, str], letter: str) -> dict:
    """The same objective scheme a board MCQ carries, via the same builder."""
    old = rec.get("answerScheme") or {}
    marks = int(rec["marks"])
    scheme = build_official_scheme(
        OfficialAnswer(q_no=0, correct_option=letter, answer_text=options[letter], marks=marks),
        marks=marks, question_id=rec["id"],
        code=str(old.get("sourcePaperCode") or ""),
        document_id=str(old.get("sourceDocumentId") or ""),
        options=options,
    )
    out = scheme.model_dump(mode="json", by_alias=True)
    # The builder names every objective key a CBSE marking scheme. An NCERT
    # Exemplar key is NCERT's answer from the book (ncert_exemplar_answer), and
    # relabelled it would read as CBSE's own; every CBE and SQP scheme already
    # says cbse_marking_scheme, so for them this changes nothing.
    out["provenance"] = str(old.get("provenance") or out["provenance"])
    return out


def _listed_options(parts: list[dict]) -> dict[str, str] | None:
    """{'A': '4', ...} from the one part an NCERT Exemplar MCQ carries, whose
    `options` is a list (corpus/ncert_exemplar.py): 1,583 Exemplar MCQs, all
    four-option. More than four is not read as A-D."""
    if len(parts) != 1:
        return None
    options = parts[0].get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        return None
    texts = [str(o).strip() for o in options]
    if not all(texts):
        return None
    return {"ABCD"[i]: t for i, t in enumerate(texts)}


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

def _strip_furniture(text: str) -> str:
    """Drop the page footer and a trailing section heading the SQP extraction
    left in the text. Keys carry them as well as stems: "... R is not the
    correct explanation of A. SECTION C (SHORT ANSWER QUESTION)". A heading
    that opens the text goes without the question after it; what is left may
    be nothing, which `_stem_after_strip_reason` refuses."""
    text = _PAGE.sub(" ", _FOOTER.sub(" ", text)).strip()
    text = _TRAILING_SECTION.sub("", text).strip()
    return _LEADING_SECTION.sub("", text).strip()


def _stem_after_strip_reason(stem: str) -> str | None:
    """Why a stem, once its furniture is gone, is no question at all.

    `exclusion_reason` reads the raw stem, so nothing looked at what the strip
    left: SQP Computer Applications X 2024-25 Q19 was served with stem ''. The
    same `_MIN_STEM` applies after the strip as before it -- the shortest
    served stem that is a question is 18 characters ("Define a compound.",
    cbe:q:Science9PS2), and at c45acef the only served stem under 15 was the
    empty one.

    A stem that opens with a lowercase word that is not a part label starts
    mid-sentence: the question's opening went to the previous row. 54 served
    CBE/SQP stems opened lowercase at c45acef. The 25 that open with a part
    label ("i. Explain", "a) Give", "a Trouvez") are questions; the other 29
    are not -- "advanced personal communication system in India." (SQP
    Geography XII 2022-23 Q23), "f the lines 3x+2ky ..." (cbe:q:Maths10RK2,
    its "I" lost), "house. Which protocol ... should she prefer" (SQP Computer
    Applications X 2024-25 Q9, one of 7 from that paper), and 8 rows of SQP
    Geography XII 2025-26, each opening with the tail of the question before.
    A question does not need a "?" to be one, nor does a fragment lack one
    ("above 65 years, what challenge is most likely to arise?"), so the
    opening word is the signature, not a question cue.

    A stem that is a heading and its section's instructions ("SECTION B This
    section consists of 6 questions of 2 marks each.") keeps the instructions
    once the heading goes; they are furniture too, and no question.

    A stem that opens with a later part's label -- "ii.", "b)", "(b)", "B." --
    has lost its opening part the same way: SQP Carnatic Music (Percussion) X
    2022-23 Q13 was served as "ii. Explain the following term: Tattu, Meetu and
    Gumki", its part i at the end of Q12's stem and its key covering both. It
    was the only served stem of this shape at fa8b78b.
    """
    if len(stem) < _MIN_STEM or _HEADER.match(stem) or _SECTION_NOTE_STEM.match(stem):
        return "stem-empty-after-strip"
    if LATER_PART_OPENING.match(stem):
        return "stem-fragment"
    if re.match(r"[a-z]", stem) and not PART_LABEL_OPENING.match(stem):
        return "stem-fragment"
    return None


def _strip_answer_furniture(text: str) -> str:
    """As `_strip_furniture`, but a trailing heading goes only with its title."""
    text = _PAGE.sub(" ", _FOOTER.sub(" ", text)).strip()
    return _TRAILING_HEADING_ONLY.sub("", text).strip()


def _normalise_scheme(scheme: dict, marks: int, record_id: str) -> dict:
    s = dict(scheme)
    s["totalMarks"] = int(s.get("totalMarks") or marks)
    points = []
    for i, mp in enumerate(s.get("markingPoints") or []):
        points.append({
            **mp,
            "id": str(mp.get("id") or f"{record_id}:mp{i + 1}"),
            "description": _strip_answer_furniture(str(mp.get("description") or "")),
            "marks": int(mp.get("marks") or 0),
            "keyword": str(mp.get("keyword") or ""),
            "isRequired": bool(mp.get("isRequired", False)),
            "synonyms": [str(x) for x in (mp.get("synonyms") or [])],
        })
    s["markingPoints"] = points
    s["rubricLevels"] = [r for r in (s.get("rubricLevels") or []) if isinstance(r, dict)]
    s["commonErrors"] = [str(x) for x in (s.get("commonErrors") or [])]
    s["alternativeAnswers"] = [str(x) for x in (s.get("alternativeAnswers") or [])]
    s["modelAnswer"] = _strip_answer_furniture(str(s.get("modelAnswer") or ""))
    s["modelAnswerLatex"] = str(s.get("modelAnswerLatex") or "")
    s["hasPartialCredit"] = bool(s.get("hasPartialCredit", len(points) > 1))
    s["metadata"] = dict(s.get("metadata") or {})
    return s


def _normalise_part(part: dict, index: int, record_id: str) -> dict:
    p = dict(part)
    p["id"] = str(p.get("id") or f"{record_id}:p{index + 1}")
    p["partNumber"] = int(p.get("partNumber") or index + 1)
    p["text"] = str(p.get("text") or "")
    p["textLatex"] = str(p.get("textLatex") or "")
    p["marks"] = int(p.get("marks") or 0)
    if p.get("answerType") not in ANSWER_TYPES:
        p["answerType"] = "textShort"
    p["alternativeAnswers"] = [str(x) for x in (p.get("alternativeAnswers") or [])]
    return p


def normalise(rec: dict) -> tuple[dict, str | None]:
    """A new record that satisfies the phone's contract, or a reason it cannot.

    Values that had to be supplied are marked `metadata.<field>Inferred`, so an
    inferred Bloom level is never readable as one CBSE authored.
    """
    # Imported here, not at the top: mapping imports pool, and pool imports
    # this module for its stem gates.
    from ..assessment.mapping import _resolve_bloom, _resolve_difficulty, _resolve_type

    r = copy.deepcopy(rec)
    meta = dict(r.get("metadata") or {})
    scheme = r.get("answerScheme") or {}
    marks = r.get("marks") if _is_num(r.get("marks")) else scheme.get("totalMarks")
    marks = int(marks) if _is_num(marks) and marks > 0 else 1
    r["marks"] = marks

    if r.get("bloomLevel") not in BLOOM_LEVELS:
        r["bloomLevel"] = _resolve_bloom(None)
        meta["bloomInferred"] = True
    if r.get("type") not in QUESTION_TYPES:
        r["type"] = _resolve_type(r.get("type"), marks)
        meta["typeInferred"] = True
    if r.get("difficulty") not in DIFFICULTIES:
        r["difficulty"] = _resolve_difficulty(marks, r["bloomLevel"])
        meta["difficultyInferred"] = True

    for key in ("chapterIds", "competencyIds", "tags"):
        r[key] = [str(x) for x in (r.get(key) or [])]
    # Page furniture the SQP extraction left in the stem, where it would print.
    r["stem"] = _strip_furniture(str(r.get("stem") or ""))
    r["stemLatex"] = str(r.get("stemLatex") or "")
    r["language"] = str(r.get("language") or "en")
    r["questionBankId"] = str(r.get("questionBankId") or r.get("source") or "unknown")
    if not _is_num(r.get("estimatedTimeMinutes")):
        r["estimatedTimeMinutes"] = max(1, marks * 2)
    if not _is_num(r.get("qualityScore")):
        r["qualityScore"] = 0.7
    for key in ("createdAt", "updatedAt"):
        if not _is_iso(r.get(key)):
            r[key] = "2026-01-01T00:00:00+00:00"
    r["parts"] = [_normalise_part(p, i, r["id"]) for i, p in enumerate(r.get("parts") or [])]
    r["answerScheme"] = _normalise_scheme(scheme, marks, r["id"])

    reason = _stem_after_strip_reason(r["stem"])
    if reason is not None:
        r["metadata"] = meta
        return r, reason
    # Read before a passage is put in front of the stem: a stem that opens
    # with its own options has lost its question -- whether or not it also has
    # option parts: SQP Carnatic Music (Vocal) X 2025-26 Q5's stem is its
    # parts' options printed again, "A. Koteeshwara Iyer B. G. N. ...".
    opens_with_options = stem_options(r["stem"])[1] == 0

    if _INNER_SECTION.search(r["stem"]):
        # One stem holding several questions across a section break -- SQP
        # Political Science XII 2023-24 Q4 runs on through "SECTION-E (24 MARKS)
        # 27 ..." and its key answers a different question.
        r["metadata"] = meta
        return r, "stem-spans-sections"

    if marks == 1 and r["type"] != "mcq" and holds_group(r["stem"]):
        # cbe:q:Maths9RS7 "1 (a) 1 (b) 1 (c) The weights of new born babies
        # ...": three questions under 1 mark, keyed with part (a)'s answer
        # running on into "1 (b) ...". Its (a)-(c) are parts, not options;
        # typed an MCQ by their shape it was refused as an MCQ it never was.
        r["metadata"] = meta
        return r, "stem-holds-group"

    passage = str(meta.get("passage") or "").strip()
    if passage:
        r["stem"] = f"Read the passage below and answer the question that follows.\n\n{passage}\n\n{r['stem']}"
        meta["passageInline"] = True

    in_parts = options_from_parts(r["parts"]) or _listed_options(r["parts"])
    # With option parts, the parts are the options, and only an (A)-(D) stem is
    # held against them. Read in the other styles, the stem's labels are as
    # often a matching list's rows ("Match List I with List II ... a. Kan swar
    # i. Krishna Rao ...", SQP Hindustani Music (Vocal) XII 2025-26 Q6) or the
    # parts' own text before extraction cut it at a line end (SQP Economics XII
    # 2024-25 Q8); compared with the parts, 18 served MCQs were refused.
    in_stem, options_at, labels = stem_options(r["stem"], other_styles=in_parts is None)
    if (r["type"] in _INFERABLE_TYPES and marks == 1
            and (has_option_labels(r["stem"])
                 or any(o is not None and len(o) == 4 for o in (in_stem, in_parts)))):
        # An MCQ by its shape: A-D options on a 1-mark item. Left as
        # short_answer it skipped this gate entirely and was served with a key
        # nobody checked, filling short-answer slots with 1-mark MCQs. The
        # shape is the label set, not a clean parse: a stem whose labels do
        # not parse is still an MCQ, and one the gate then refuses.
        r["type"] = "mcq"
        meta["typeInferred"] = True

    if opens_with_options and (r["type"] == "mcq" or marks == 1):
        # "a. Chappu b. Toppi c. Choru d. Varu" (SQP Carnatic Music
        # (Percussion) X 2022-23 Q4, key "B"), "i. Tribal Paintings ii.
        # Company Paintings ..." (SQP Painting XII 2023-24 Q3, key "(iv)"):
        # options with no question. A multi-mark stem that opens "i. ... ii.
        # ... iii. ..." is a question in parts; none was served at fa8b78b.
        r["metadata"] = meta
        return r, "stem-fragment"

    if r["type"] == "mcq":
        r["metadata"] = meta
        reason = _mcq_reason(r, in_stem, in_parts)
        if reason is not None:
            return r, reason
        options = in_parts or in_stem
        # The key is read as the scheme wrote it, before its furniture is
        # stripped for display: "(b) SECTION-B" (SQP Kathak 2022-23 Q8) is a
        # key that ran into the next heading and is refused; stripped, it
        # would read as a clean "(b)".
        letter = key_letter(
            [str(scheme.get("modelAnswer") or "")]
            + [str(mp.get("description") or "") for mp in scheme.get("markingPoints") or []
               if isinstance(mp, dict)],
            options, LETTERS if in_parts else labels or LETTERS)
        if letter is None:
            return r, "mcq-answer-unresolved"
        if in_stem is None:
            r["stem"] = r["stem"] + " " + " ".join(f"({k}) {v}" for k, v in options.items())
        elif labels != LETTERS and not in_parts:
            # Printed "i. ... ii. ..." or "1. ... 2. ...", keyed as a letter:
            # the phone reads a student's answer as a letter, so the options
            # are printed with the letters the key uses -- unless the options
            # themselves name those letters. SQP Carnatic Music (Melodic
            # Instrument) X 2023-24 Q6 numbered "1. A & B 2. A & C" because
            # its statements are A-D; relabelled, it read "(C) B & C". The
            # 1-4 labels cannot be kept instead: both scorers read only A-D.
            if names_option_letter(in_stem):
                return r, "option-labels-collide"
            r["stem"] =(r["stem"][:options_at].rstrip() + " "
                         + " ".join(f"({k}) {v}" for k, v in in_stem.items()))
        r["parts"] = []            # options live inline, as on every board MCQ
        r["answerScheme"] = _normalise_scheme(_objective_scheme(r, options, letter), marks, r["id"])
    else:
        shown = r["answerScheme"]
        texts = [shown["modelAnswer"]] + [mp["description"] for mp in shown["markingPoints"]]
        r["metadata"] = meta
        if not any(t.strip() for t in texts):
            # The whole key was a heading: "SECTION-B" (SQP Music papers, Q8).
            return r, "no-answer"
        if any(_INNER_SECTION.search(t) for t in texts):
            # "3. SECTION-B Brief explanation of Kan, Gamak ..." (SQP Hindustani
            # Music (Vocal) XII 2023-24 Q8): the key ran on into the next section.
            return r, "answer-bleed"

    r["metadata"] = meta
    return r, None


def _mcq_reason(r: dict, in_stem: dict[str, str] | None,
                in_parts: dict[str, str] | None) -> str | None:
    """Why this MCQ cannot be served as one, before its key is even read."""
    if r["marks"] > _MCQ_MAX_MARKS:
        return "mcq-marks-implausible"
    if SUB_PART.match(r["stem"]):
        return "mcq-part-of-group"
    if in_stem is not None and in_parts is not None:
        # Both parse: the parts are the options. If the stem's labels say
        # something else they are not options at all -- the column labels of a
        # matching table (SQP History XII 2025-26 Q19, whose option D then ran
        # on into a page footer) -- and the item cannot be printed as it stands.
        same = [tokens(v) for v in in_stem.values()] == [tokens(v) for v in in_parts.values()]
        if not same:
            return "mcq-options-unresolved"
    options = in_parts or in_stem
    if options is None or len(options) < MIN_OPTIONS or not answerable(options):
        return "mcq-options-unresolved"
    return None


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# board records: repaired where the repair is certain
# --------------------------------------------------------------------------- #

# Page furniture a board paper's extraction left in a stem, each measured on
# the served board records (2026-09-22): "# 14| P a g e" (92 stems), "# 4 of"
# (26), "666 -11 6 of" (38), paper codes "/CD1BA/22" and "/21/BBCA2" (18),
# "P.T.O.", "Page 28 of 32"; control characters where spaces were
# ("Read\x01the\x01following", a493a2f3 Q36).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]+")
_BOARD_FURNITURE = (
    re.compile(r"\s*#\s*\d{1,3}\s*\|\s*P\s*a\s*g\s*e\b"),
    re.compile(r"\s*\|\s*P\s*a\s*g\s*e\b"),
    re.compile(r"\s*\bP\.\s?T\.\s?O\.?"),
    re.compile(r"\s*#\s*\d{1,3}\s+of\b"),
    re.compile(r"\s*\b666\s+\d{0,4}-\d{1,2}\s+\d{1,3}\s+of\b"),
    re.compile(r"\s*/[A-Z0-9]{4,6}/\d{2}\b"),
    re.compile(r"\s*/\d{2}/[A-Z0-9]{4,6}\b"),
    _PAGE,
)
# At the very end only: the paper's series after the last word ("Human
# Placental Lactogen -11", Biology XII 57/1/1 Q2; "reactance 11-"), a bare
# "/22", and a page count "... 23 of". "-11" after a word only: "(B) -11" and
# "x = -11" are values.
_BOARD_TRAILING = (
    re.compile(r"(?<=[A-Za-z.?])\s+-\d{2}\s*$"),
    re.compile(r"\s+\d{1,2}-\s*$"),
    re.compile(r"\s+/\d{2}\s*$"),
    re.compile(r"\s+\d{1,3}\s+of\s*$"),
)


def strip_board_furniture(text: str) -> str:
    """A board stem without the page furniture its extraction left in it.

    Returns the text unchanged, byte for byte, when there is none."""
    out = _CONTROL.sub(" ", text)
    for pattern in _BOARD_FURNITURE:
        out = pattern.sub(" ", out)
    before = None
    while before != out:
        before = out
        for pattern in _BOARD_TRAILING:
            out = pattern.sub("", out)
    out = " ".join(out.split())
    return text if out == " ".join(text.split()) else out


# CBSE's four assertion-reason options, word for word as the Home Science X
# 2024-25 SQP prints them (cbse:sqp:ClassX_2024_25:Home Science:17). No served
# board stem prints them (167 of 3,286 are assertion-reason, all without), and
# no code printed them either.
ASSERTION_REASON_OPTIONS = {
    "A": "Both A and R are true and R is the correct explanation of A.",
    "B": "Both A and R are true but R is not the correct explanation of A.",
    "C": "A is true but R is false.",
    "D": "A is false but R is true.",
}
_ASSERTION_REASON = re.compile(r"\bAssertion\s*\(A\).*\bReason\s*\(R\)", re.S)
# The mark printed between a question and its options: "... is called a 1 (A)
# secant" (Mathematics X 430/2/2 Q12), "... detritus ? 1 (A) ..." (Biology XII
# 57/5/2 Q9). After a word or a question mark only: "x = 1 (A) ..." is a value.
_MARK_BEFORE_OPTIONS = re.compile(r"(?<=[A-Za-z?:])\s+[12]$")


# The rupee sign, extracted as "<" from every commerce board paper's font:
# "Salary @ < 15,000 per quarter" (Accountancy XII 4d332115f14c6aeeaa34b438
# Q25), "Debtors -< 40,000" (be7fd19527d3a0bcac687817 Q22), "a purchase
# consideration of < 40,00,000. < 20,00,000 were paid" (d4ab284479dbd2fbc3a07c7f
# Q26), "Amount (<)" in a table's head. Measured on the 3,286 served board
# records (2026-09-22): 414 "<" before a number in Accountancy, Business
# Studies and Economics, and not one of them a less-than -- 14 of the 19
# Accountancy XII records served after relinking printed amounts this way,
# which on a paper read as "less than". Only before a number (or a blank to
# fill, "< ______ crore"), and not after a number or a one-letter variable:
# "Receipts < Payments" (SQP Economics XII 2024-25 Q10) is a less-than, as
# "x < 5" would be. Commerce papers only: Mathematics X prints "withdraw
# < 2,000" the same way, but also "f'(x) < 0".
_COMMERCE_SUBJECTS = frozenset({"accountancy", "business studies", "economics"})
_RUPEE_READ_AS_LT = re.compile(
    r"<(?=\s?(?:\d|_{3,}))|(?<=\()<(?=\))"
    # A Balance Sheet's head, "Liabilities Amount < Assets Amount <" (Accountancy
    # XII 4d332115f14c6aeeaa34b438 Q25, be7fd19527d3a0bcac687817 Q22).
    r"|(?<=\bLiabilities Amount )<(?= Assets Amount <)|(?<=\bLiabilities Amount < Assets Amount )<")
# The amount itself lost, only its sign left: "It had a credit balance of
# < Pass necessary journal entries" (Accountancy XII d4ab284479dbd2fbc3a07c7f
# Q26), "The company had a balance of < on the same date" (4d332115f14c6aeeaa34b438
# Q26), "agreed to pay him < share of goodwill", "Chetan brought < Profit and
# Loss". After "of", "him" or "brought" a "<" compares nothing.
_RUPEE_AMOUNT_LOST = re.compile(r"\b(?:of|him|brought)\s?<\s?(?=[^\W\d_])")
_COMPARED = re.compile(r"(?:\d|(?<![^\W\d_])[^\W\d_])\s?$")


def restore_rupee(text: str) -> str:
    """A commerce stem with its rupee signs, where extraction printed "<"."""
    def one(m: re.Match) -> str:
        return m.group(0) if _COMPARED.search(text, 0, m.start()) else "₹"
    return _RUPEE_READ_AS_LT.sub(one, text)


# Everything a student reads or a scorer scores: the stem, the options (inline
# in the stem, or in the parts), the answer key, and the relink's copy of the
# options in answerScheme.metadata. A CBE passage is put in front of the stem
# by `normalise`, so it prints too. This is the GATE's scope, not the repair's
# (`repair_symbol_font`): a question is refused because a student would see a
# hole, not because a provenance field carries one.
_SYMBOL_PRINTED_FIELDS = ("stem", "stemLatex", "parts", "answerScheme")
PRIVATE_USE_GLYPH = "private-use-glyph"


def printed_scheme(scheme: Any) -> Any:
    """The answer scheme as a scorer reads it: everything but the relink's
    provenance copy of the keys a conflict withheld (`CONFLICT_HELD`).

    `answerScheme` as a whole is a printed field -- the model answer and the
    marking points are what a scorer scores -- but one nested key inside it is
    not. When two official keys disagree the relink serves neither and keeps
    both on the record (`question_bank._quarantine`, `CONFLICT_HELD`), as a
    copy of what was removed AS it was removed: `_restore_scheme_symbols`
    deliberately leaves it raw so a person settling the conflict sees the code
    point the extractor produced. Reading it here would make the merge refuse
    a whole question over a field nobody reads, and rewrite the copy it is the
    point of to keep verbatim. 46 of the 15,605 rows in answer_keys.db carry a
    private-use character the table cannot name, and a conflicting key is
    copied verbatim from such a row.
    """
    if not isinstance(scheme, dict):
        return scheme
    meta = scheme.get("metadata")
    if not isinstance(meta, dict) or CONFLICT_HELD not in meta:
        return scheme
    return {**scheme, "metadata": {k: v for k, v in meta.items() if k != CONFLICT_HELD}}


def _strings(value: Any) -> Iterator[str]:
    """Every string inside a record's field, however deeply nested."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def repair_symbol_font(rec: dict) -> tuple[dict, bool, str | None]:
    """A copy of `rec` with its symbol-font code points restored, whether any
    were, and the first private-use character the table cannot name in a field
    that prints.

    The repair runs over EVERY string in the record, not only the printed ones.
    An option and its key must agree -- cbe:q:Science9PS1 keys "W / ρ g", and a
    stem repaired without its scheme would be scored against the unrepaired
    text -- and the field a printed one has to agree with is not always printed
    itself: 780 served SQP records carry a top-level `metadata.options` map
    (a different field from the `answerScheme.metadata.options` the scheme
    holds) and 313 carry `metadata.contentReference`. Repairing the stem and
    leaving those is the same stem/key disagreement, one field further out.

    The one field the walk skips is the relink's copy of a withheld conflicting
    key (`CONFLICT_HELD`), exactly as `question_bank._restore_scheme_symbols`
    skips it: it is provenance, a copy of what was removed as it was removed,
    and repairing it would destroy the thing it records.

    The GATE is the narrow one (`_SYMBOL_PRINTED_FIELDS` minus that same
    provenance key, plus the passage): a private-use character the table cannot
    name costs the record its place only where a student or a scorer would read
    the hole.
    """
    restored = False

    def fix(value: Any) -> Any:
        nonlocal restored
        if isinstance(value, str):
            out = restore_symbol_font(value)
            restored = restored or out != value
            return out
        if isinstance(value, dict):
            return {k: (v if k == CONFLICT_HELD else fix(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v) for v in value]
        return value

    r = fix(dict(rec))
    printed = {k: r.get(k) for k in _SYMBOL_PRINTED_FIELDS}
    printed["answerScheme"] = printed_scheme(printed["answerScheme"])
    printed["passage"] = (r.get("metadata") or {}).get("passage")
    left = next((glyph for text in _strings(printed)
                 if (glyph := private_use_glyph(text)) is not None), None)
    return r, restored, left


def _board_stem_broken(stem: str) -> bool:
    return len(stem) < _MIN_STEM or bool(_HEADER.match(stem)) or _words_lost(stem)


def repair_board(rec: dict) -> tuple[dict, str | None, list[str], str | None]:
    """A copy of a verified board record repaired where the repair is certain,
    the reason it cannot be served if a repair shows one, the repairs made, and
    the first private-use character the table cannot name in a field that
    prints.

    The glyph is RETURNED rather than made a reason here, so that `compose` can
    file it last, after `exclusion_reason` -- as the CBE/SQP path files it
    after `source_reason`. Filed first it claims records that are out of scope
    for a more informative reason anyway, and `_merge_excluded.json` is what
    decides which defects get worked next.

    Options stay inline, "(A) ... (B) ... (C) ... (D) ...", as on every served
    MCQ: the web PDF reads them only from the stem (pdf.split_stem_and_options)
    and the phone's inline branch prints the stem as it is. The correct option
    is the one the relink VERIFIED (answerScheme.metadata.correctOption, Tasks
    15/151/152), never a label read here: an objective item without it is
    mcq-answer-unresolved.
    """
    r = copy.deepcopy(rec)
    repairs: list[str] = []
    # First, so every gate below reads the symbols the paper printed rather
    # than a font's code points.
    r, symbols_restored, glyph = repair_symbol_font(r)
    if symbols_restored:
        repairs.append("symbol-font-restored")
    stem = str(r.get("stem") or "")
    cleaned = strip_board_furniture(stem)
    if cleaned != stem:
        repairs.append("page-furniture-stripped")
        if _board_stem_broken(cleaned):
            return r, "page-furniture", repairs, glyph
        stem = cleaned
    if str(r.get("subject") or "").strip().lower() in _COMMERCE_SUBJECTS:
        restored = restore_rupee(stem)
        if restored != stem:
            repairs.append("rupee-sign-restored")
            stem = restored
        if _RUPEE_AMOUNT_LOST.search(stem):
            return r, "symbol-loss", repairs, glyph
    scheme = r.get("answerScheme") or {}
    meta = scheme.setdefault("metadata", {})
    letter = str(meta.get("correctOption") or "").strip().upper() or None
    options, at, _ = stem_options(stem, other_styles=False)
    # Objective as `normalise` infers it: typed MCQ, or a 1-mark short answer.
    # A 1-mark case study numbering its questions "(i) ... (ii) ... (iii)" is
    # a question in parts, not options.
    objective = (r.get("type") == "mcq"
                 or (r.get("marks") == 1 and r.get("type") in _INFERABLE_TYPES))
    # stem_options never reads an A-R stem that prints its options: the "(A)" of
    # "Assertion (A)" comes first, so the labels run A, A, B, C, D. has_option_labels
    # sees them, and such a stem keeps its own four rather than gaining four more.
    if options is None and has_option_labels(stem) and objective:
        # Labels printed that do not parse: as for CBE and SQP in `normalise`,
        # the MCQ cannot be printed as one. Two MCQs fused into one stem
        # (Mathematics XII a52469d5f185d77194f532dd Q9 runs on into Q10, each
        # with its (A)-(D)) print, through pdf.split_stem_and_options, the
        # SECOND question's options under a letter verified for the first --
        # the shape whose keys answered a different question (mcq_shape.
        # has_option_labels). "A) (0, 0) B) ... C ) ..." is refused too. Only
        # an A-R stem is read here: its four printed options, and a verified
        # letter for them.
        from ..assessment.pdf import split_stem_and_options  # reportlab, only when needed

        if not _ASSERTION_REASON.search(stem) or len(split_stem_and_options(stem)[1]) != 4:
            return r, "mcq-options-unresolved", repairs, glyph
        if letter not in ASSERTION_REASON_OPTIONS:
            return r, "mcq-answer-unresolved", repairs, glyph
    elif options is None and _ASSERTION_REASON.search(stem) and not has_option_labels(stem):
        if letter not in ASSERTION_REASON_OPTIONS:
            return r, "mcq-answer-unresolved", repairs, glyph
        stem = f"{stem.rstrip()} " + " ".join(
            f"({k}) {v}" for k, v in ASSERTION_REASON_OPTIONS.items())
        meta["options"] = dict(ASSERTION_REASON_OPTIONS)
        repairs.append("assertion-reason-options")
    elif options is not None and objective and letter is None:
        # An objective item that prints options but has no verified letter has
        # no answer key to print, whatever its marks (5 served board MCQs are
        # worth 2) or option count.
        return r, "mcq-answer-unresolved", repairs, glyph
    elif options is not None and (letter is not None or (r.get("marks") == 1 and len(options) == 4)):
        # A lettered list on a multi-mark item with no verified letter is its
        # parts -- "(a) Explain ... (b) Suggest ..." -- and is left alone.
        if len(options) < MIN_OPTIONS or not answerable(options):
            return r, "mcq-options-unresolved", repairs, glyph
        if letter not in options:
            return r, "mcq-answer-unresolved", repairs, glyph
        # Reprinted only to fix something -- lowercase labels, or a mark before
        # them -- so the other options keep their text as the paper printed it.
        head = stem[:at].rstrip()
        bare = _MARK_BEFORE_OPTIONS.sub("", head)
        if bare != head or stem[at + 1].islower():
            stem = f"{bare} " + " ".join(f"({k}) {v}" for k, v in options.items())
            repairs.append("options-inline")
        if "options" in meta:
            # The relink's copy ends as the stem did: "D": "Human Placental Lactogen -11".
            meta["options"] = dict(options)
    r["stem"] = stem
    return r, None, repairs, glyph


# --------------------------------------------------------------------------- #
# near-duplicates
# --------------------------------------------------------------------------- #

# Two stems are one question when their words and numbers overlap this much,
# and one is the other with words only added or dropped. Measured on the
# served board records with the CBE and SQP sources (2026-09-22), pairs within
# one subject and grade: at 0.9 and above the pairs are one question printed
# twice -- "Which of the following is not a quadratic equation ?" in two papers
# of one series, extracted with different tails; SQP Tangkhul XII Q27 in
# 2024-25 and 2025-26, differing in "Page 6 of" -- and at 0.75-0.8 they are
# different questions: SQP Bharatanatyam XII Q8 asks for Krishna's bow in
# 2024-25 and Shiva's in 2025-26, with the same options and a different key.
# Numbers count as words, or "2x+3=7" and "2x+5=7" would be one question.
#
# Overlap alone reads words as a set, and misses which word stands where: "a
# bag contains 5 red, 8 white, 4 green and 7 black balls ... probability that
# it is not green" and "... not black" share every word, and scored 1.0. So a
# word swapped for another, or moved, makes two questions whatever the
# overlap. Word pairs (bigrams) were tried and do not separate them: that pair
# scores exactly 0.90 on bigrams, and a longer stem with one word swapped
# scores higher, while real duplicates with a run-on tail (SQP Bharatanatyam
# XII 2022-23 Q8 and 2023-24 Q7, "... Section B") fall to 0.83. Of the 83
# pairs at word-set overlap >= 0.9 in those sources, every one whose stems
# differ only by words added or dropped (tails, page furniture, "also") is
# still one question.
NEAR_DUPLICATE_MIN = 0.9
# Below this many distinct words a stem is too short to call a duplicate by
# overlap; exact duplicates are still `norm`'s.
_NEAR_MIN_TOKENS = 6
_NEAR_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "is", "are", "to", "and", "or", "for", "on", "at",
    "it", "its", "this", "that", "with", "from", "by", "as", "be"})


_POLARITY = frozenset({"not", "no", "never", "except", "incorrect", "false", "cannot",
                       "none", "neither", "nor"})


def _near_words(text: str) -> list[str]:
    return [t for t in re.findall(r"[^\W_]+", text.lower()) if t not in _NEAR_STOPWORDS]


def _words_only_added_or_dropped(a: list[str], b: list[str]) -> bool:
    """One stem is the other with words added or dropped: none swapped for
    another word, none moved to another place."""
    added: set[str] = set()
    dropped: set[str] = set()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "replace":
            return False
        if op == "equal":
            continue
        dropped.update(a[i1:i2])
        added.update(b[j1:j2])
    # "is correct" and "is not correct" ask for opposite answers: a negation
    # added or dropped makes another question (as answer keys agree only with
    # the same polarity).
    if (added | dropped) & _POLARITY:
        return False
    return not (added & dropped)


def near_duplicate_score(a: str, b: str) -> float:
    """How far two stems read as one question: 0.0 to 1.0.

    The overlap of their words (Jaccard), or 0.0 when a word of one was swapped
    for another or moved: then they ask different things."""
    wa, wb = _near_words(a), _near_words(b)
    return _near_score(wa, frozenset(wa), wb, frozenset(wb))


def _near_score(wa: list[str], ta: frozenset[str], wb: list[str], tb: frozenset[str]) -> float:
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / len(ta | tb)
    if overlap >= NEAR_DUPLICATE_MIN and not _words_only_added_or_dropped(wa, wb):
        return 0.0
    return overlap


class _NearDuplicates:
    """Stems already served, by subject and grade, compared by `near_duplicate_score`.

    Only sets of comparable size are compared: two sets at Jaccard >= 0.9 have
    sizes within 0.9 of each other."""

    def __init__(self) -> None:
        self._seen: dict[tuple, list[tuple[list[str], frozenset[str]]]] = {}

    def is_duplicate(self, key: tuple, stem: str) -> bool:
        words = _near_words(stem)
        tokens = frozenset(words)
        if len(tokens) < _NEAR_MIN_TOKENS:
            return False
        for other_words, other in self._seen.get(key, ()):
            small, large = sorted((len(tokens), len(other)))
            if small < NEAR_DUPLICATE_MIN * large:
                continue
            if _near_score(words, tokens, other_words, other) >= NEAR_DUPLICATE_MIN:
                return True
        return False

    def add(self, key: tuple, stem: str) -> None:
        words = _near_words(stem)
        tokens = frozenset(words)
        if len(tokens) >= _NEAR_MIN_TOKENS:
            self._seen.setdefault(key, []).append((words, tokens))


@dataclass
class ComposeResult:
    questions: list[dict]
    excluded: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    # Board records excluded, whole: `excluded` keeps 200 characters of a stem,
    # and these must survive to be relinked, or repaired, and served again.
    withheld: list[dict] = field(default_factory=list)
    # Repairs made to the records served: page-furniture-stripped,
    # rupee-sign-restored, options-inline, assertion-reason-options (board
    # records only), and symbol-font-restored (every source).
    repairs: dict[str, int] = field(default_factory=dict)


def unallocated_marks_reason(rec: dict) -> str | None:
    """`SCHEME_MARKS_UNALLOCATED` when the record's key allocates nothing.

    A key that lists its value points but gives every one 0 marks reads, in
    the printed answer key, as "[0m]" against each point: PRD rule Q1's "no
    answer key, no question" applied to what the key actually says. A key
    that allocates SOME marks is kept as the paper printed it -- only "all
    points at zero, on a question worth marks" is refused."""
    scheme = rec.get("answerScheme") or {}
    points = scheme.get("markingPoints") or []
    if not points or not (rec.get("marks") or 0):
        return None
    if any((p.get("marks") or 0) > 0 for p in points):
        return None
    return SCHEME_MARKS_UNALLOCATED


def _excluded(rec: dict, reason: str) -> dict:
    return {"id": rec.get("id"), "source": rec.get("source"), "reason": reason,
            "subject": rec.get("subject"), "grade": rec.get("grade"),
            "stem": str(rec.get("stem") or "")[:200]}


def _runs_on(rows: list[dict]) -> set[str]:
    """Ids whose next question in the same paper is a stem-fragment.

    A fragment's opening went somewhere, and in these papers it went to the
    question before: SQP Computer Applications X 2024-25 Q19 ends "Re-write
    the correct", Q20 opens "statements with underlined corrections", and
    Q19's key ("SMS MMS ...") answers a third question. Q19 read as a whole
    question, so it was served. The next question is the one numbered one on
    (`metadata.questionKey`), whatever else excludes it; CBE items carry no
    question number and are not paired.
    """
    fragments: set[tuple[str, int]] = set()
    numbered: list[tuple[str, int, str]] = []
    for rec in rows:
        key = str((rec.get("metadata") or {}).get("questionKey") or "")
        if not key.isdigit():
            continue
        paper = str(rec.get("questionBankId") or "")
        numbered.append((paper, int(key), str(rec.get("id"))))
        if normalise(rec)[1] == "stem-fragment":
            fragments.add((paper, int(key)))
    return {rid for paper, q, rid in numbered if (paper, q + 1) in fragments}


def has_verified_key(rec: dict) -> bool:
    """A board record carries a scheme the relink verified.

    The label alone is not enough. Since Task 15 `question_bank.
    relink_answer_schemes` writes cbse_marking_scheme on a board record only
    for a row the verifier accepted, but a bank not relinked since still
    carries the older labels -- 2,930 board records at 43da0f4, against 89
    the relink verifies, and hand audits found about 30% of those labels
    detectably wrong. Only the relink's `build_official_scheme` stamps the
    official row it verified (metadata.schemeCode and schemeQNo), so the
    gate requires the stamp too: a stale label fails closed whatever order
    the pipeline ran in."""
    scheme = rec.get("answerScheme") or {}
    has_content = bool(scheme.get("markingPoints") or str(scheme.get("modelAnswer") or "").strip())
    meta = scheme.get("metadata") or {}
    stamped = bool(str(meta.get("schemeCode") or "").strip()) and meta.get("schemeQNo") is not None
    return has_content and scheme.get("provenance") == "cbse_marking_scheme" and stamped


def hold(kept: list[dict], withheld: list[dict]) -> list[dict]:
    """The withheld board records after this run: `kept` with `withheld`
    merged in by id -- a newer copy replaces its entry in place, a new id is
    appended, and nothing is ever dropped. A record served again stays here
    too; `compose` prefers the served copy."""
    out = list(kept)
    at = {str(r.get("id")): i for i, r in enumerate(out)}
    for rec in withheld:
        rid = str(rec.get("id"))
        if rid in at:
            out[at[rid]] = rec
        else:
            at[rid] = len(out)
            out.append(rec)
    return out


WITHHELD_NAME = "board_withheld.json"
WITHHELD_ABOUT = (
    "Board records the merge withheld -- no-verified-key, or not answerable as "
    "printed (see _merge_excluded.json for each one's reason) -- kept WHOLE and "
    "unrepaired so a later relink or repair can bring them back and the next "
    "merge serve them again. "
    "merge_question_banks.py and enrich_question_bank.py both read this file "
    "back; a run adds a record to it or replaces one by id, and none drops one.")


def load_withheld(path: Path) -> list[dict]:
    """The withheld board records, or none when no run has withheld any."""
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("questions") or [])


def write_withheld(path: Path, records: list[dict]) -> None:
    """Replace the file with `records`; callers pass `hold(...)`, which never drops one."""
    body = json.dumps({"about": WITHHELD_ABOUT, "count": len(records), "questions": records},
                      ensure_ascii=False, indent=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def compose(served: list[dict], cbe: list[dict], sqp: list[dict],
            held: list[dict] = (), exemplar: list[dict] = ()) -> ComposeResult:
    """Board records from `served` and then `held` (the withheld ones an
    earlier run kept, where `served` has no record of that id), then CBE,
    then SQP, then NCERT Exemplar.

    Exemplar records go through exactly the gates CBE and SQP records do; they
    come last so that where an Exemplar item asks what a CBSE paper already
    asks, the CBSE copy is the one served.

    Non-board rows already in `served` are ignored and rebuilt from their source
    files, so running the merge twice gives the same bank. A normalised record
    that still breaks the contract raises ContractError and nothing is written.

    A verified board record is repaired (`repair_board`) and then gated like
    every other record (`exclusion_reason`, then exact and near duplicates).
    Every board record not served -- no-verified-key or any other reason -- is
    returned whole, unrepaired, in `withheld`.

    Near-duplicates keep the record seen first, and the order is the rule
    "keep the one with a verified scheme": verified board records come first,
    and nothing else here carries the relink's verified stamp.
    """
    out: list[dict] = []
    excluded: list[dict] = []
    withheld: list[dict] = []
    repairs: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_stems: set[tuple] = set()
    near = _NearDuplicates()
    counts = {"board": 0, "cbe": 0, "sqp": 0, "exemplar": 0}

    board = [r for r in served if r.get("source") == BOARD_SOURCE]
    in_served = {r.get("id") for r in board}
    board += [r for r in held if r.get("source") == BOARD_SOURCE and r.get("id") not in in_served]
    for rec in board:
        if not has_verified_key(rec):
            excluded.append(_excluded(rec, NO_VERIFIED_KEY))
            withheld.append(rec)
            continue
        fixed, reason, made, glyph = repair_board(rec)
        reason = reason or exclusion_reason(fixed) or unallocated_marks_reason(fixed)
        # Last, as on the CBE/SQP path below: a record already out of scope for
        # a more informative reason keeps that reason.
        if reason is None and glyph is not None:
            reason = PRIVATE_USE_GLYPH
        group = (rec.get("subject"), rec.get("grade"))
        key = (*group, norm(fixed["stem"]))
        if reason is None and (key in seen_stems or near.is_duplicate(group, fixed["stem"])):
            reason = "duplicate-stem"
        if reason:
            excluded.append(_excluded(rec, reason))
            withheld.append(rec)
            continue
        errors = contract_errors(fixed)
        if errors:
            raise ContractError(str(rec.get("id")), errors)
        out.append(fixed)
        seen_ids.add(fixed["id"])
        # Both spellings: a CBE/SQP copy of the stem as the paper printed it is
        # the same question as the repaired one.
        seen_stems.update({key, (*group, norm(rec.get("stem", "")))})
        near.add(group, fixed["stem"])
        counts["board"] += 1
        for name in made:
            repairs[name] = repairs.get(name, 0) + 1

    for label, rows in (("cbe", cbe), ("sqp", sqp), ("exemplar", exemplar)):
        runs_on = _runs_on(rows)
        for raw in rows:
            if raw.get("id") in seen_ids:
                excluded.append(_excluded(raw, "duplicate-id"))
                continue
            # Before the gates, as for a board record: the symbols are what the
            # paper printed, so every gate below reads the repaired text.
            raw, symbols_restored, glyph = repair_symbol_font(raw)
            # The glyph gate runs AFTER source_reason, not before it. Placed
            # first it claimed records that were already out of scope for a
            # more informative reason: of the 23 it excluded at 64dd21a, 12
            # were answer-bleed (9), figure-referenced (2) or garbled-script
            # (1) as well, and the --check tally read "17 SQP + 6 CBE lost to
            # private-use-glyph" -- about twice this change's real cost, with
            # a dozen answer-bleed records hidden behind the wrong label.
            reason = source_reason(raw)
            if reason:
                excluded.append(_excluded(raw, reason))
                continue
            rec, reason = normalise(raw)
            if reason is None and raw.get("id") in runs_on:
                reason = "stem-runs-on"
            # Read on the normalised record: its stem is stripped of the
            # furniture -- a next section's heading and passage, run on after
            # SQP Home Science X 2024-25 Q14's options -- and its options are
            # inline.
            reason = reason or unanswerable_reason(rec)
            # Last, on the normalised record and its normalised scheme: the
            # relink that runs after the merge applies the same rule, so what
            # is served here keeps its key there.
            if reason is None and builder_key_reason(rec) is not None:
                reason = KEY_REJECTED
            reason = reason or unallocated_marks_reason(rec)
            # Last, as on the board path: a record already out of scope for a
            # more informative reason keeps that reason. Filed after
            # `source_reason` alone, the glyph claimed three SQP records that
            # also span two sections.
            if reason is None and glyph is not None:
                reason = PRIVATE_USE_GLYPH
            if reason:
                excluded.append(_excluded(raw, reason))
                continue
            group = (rec.get("subject"), rec.get("grade"))
            stem = str(raw.get("stem") or "")
            key = (*group, norm(stem))
            if key in seen_stems or near.is_duplicate(group, stem):
                excluded.append(_excluded(raw, "duplicate-stem"))
                continue
            errors = contract_errors(rec)
            if errors:
                raise ContractError(str(rec.get("id")), errors)
            out.append(rec)
            seen_ids.add(rec["id"])
            seen_stems.add(key)
            near.add(group, stem)
            counts[label] += 1
            if symbols_restored:
                repairs["symbol-font-restored"] = repairs.get("symbol-font-restored", 0) + 1

    return ComposeResult(questions=out, excluded=excluded, counts=counts, withheld=withheld,
                         repairs=repairs)


def _chapter_key(name: Any) -> str:
    """A chapter name as it compares: "Statistics & Probability" and
    "Statistics and Probability" are one chapter, as are "Number systems" and
    "Number Systems"."""
    return " ".join(re.findall(r"[a-z0-9]+", str(name or "").lower().replace("&", " and ")))


# CBSE's learning-ladder content codes, as the CBE item banks print them:
# "10A2b" and "6N1e" (Mathematics), "9.1.6" (Science), "ENG9.5" (English).
# The same shape tests/test_catalog_chapters.py asserts no chapter list holds.
# Only an id of this shape has a strand to fall back on: `coarse_code` was
# written for these codes and degenerates for anything else --
# coarse_code("algebra"), coarse_code("acids-bases-salts") are both "A" -- so
# two unrelated slug-shaped ids would share one strand pool, and a pool that
# names exactly one chapter resolves every id in it to that chapter. Questions
# filed under a chapter they do not belong to, with no signal, is the wrong
# answer shipped silently. Every one of the 104 distinct unknown ids in the
# pre-resolution bank is code-shaped today; one renamed syllabus id or one
# taxonomy id leaking into `chapterIds` is all it would take.
CONTENT_CODE = re.compile(r"(?i)^(?:eng)?\d{1,2}(?:[a-z]\d[a-z]?|(?:\.\d+)+)$")


def _strand_chapters(records: list[dict], chapters: dict[tuple[str, int], dict[str, str]],
                     by_name: dict[tuple[str, int], dict[str, str]],
                     ) -> dict[tuple[tuple[str, int], str], str]:
    """(subject, grade, CBE strand) -> the syllabus chapter its records name.

    CBSE issues a content code per strand ("10A2b", "10A4a" are both the 10A
    strand) and prints the strand's name on the records that carry a topic:
    every 10A record that has one says "Algebra". So a coded record with no
    name of its own takes the name its strand's siblings carry. A strand whose
    records name two different chapters is left out -- it names neither. An id
    that is not a content code (`CONTENT_CODE`) has no strand and is skipped.
    """
    from ..assessment.topic_mapper import coarse_code

    found: dict[tuple[tuple[str, int], str], set[str]] = {}
    for rec in records:
        group = (rec.get("subject"), rec.get("grade"))
        known = chapters.get(group)
        if not known:
            continue
        for cid in rec.get("chapterIds") or []:
            if cid in known or not CONTENT_CODE.match(str(cid)):
                continue
            for name in (rec.get("topic"), (rec.get("tags") or [None])[0]):
                chapter = by_name[group].get(_chapter_key(name)) if name else None
                if chapter:
                    found.setdefault((group, coarse_code(cid)), set()).add(chapter)
                    break
    return {key: next(iter(names)) for key, names in found.items() if len(names) == 1}


def resolve_chapter_ids(records: list[dict], chapters: dict[tuple[str, int], dict[str, str]],
                        taxonomy_names: dict[str, str] | None = None,
                        ) -> tuple[list[dict], dict[str, int]]:
    """The records with every chapter id a teacher can be offered, and how many
    ids were resolved and how many dropped.

    `chapters` is {(subject, grade): {chapter id: name}} from the CBSE syllabus
    files; `taxonomy_names` is {taxonomy chapter id: name} from the NCERT
    taxonomy trees, for the record's `taxonomyChapterId` (Task 703's tagger).

    A record whose chapter id the syllabus does not know carries a raw CBE
    content code -- "10A2b", "9.1.6" -- which the chapter tagger could not
    place. Both chapter pickers name a chapter by what its questions carry
    (the web's catalog route titles the id itself, "10A2B"; the phone takes
    the first tag), so such a code is printed to a teacher as a chapter, and
    the same chapter is then listed several times: Mathematics 10 listed
    "Trigonometry" three times and 17 chapters against the syllabus's 7
    (measured at f90f42c).

    So each unknown id is resolved to the syllabus chapter its own names give
    -- CBSE's topic, the record's first tag, the taxonomy chapter, then, for a
    content-coded id only, its strand's name (`_strand_chapters`) -- and
    dropped when none of them names one. Dropped, the question keeps every
    other field and is still served and searchable; it is only not offered
    under a chapter, which is what a raw code was doing anyway. A subject with
    no syllabus file (Home Science, the
    SQP arts subjects) has nothing to check against and is left alone.
    """
    from ..assessment.topic_mapper import coarse_code

    taxonomy_names = taxonomy_names or {}
    by_name = {group: {_chapter_key(name): cid for cid, name in ids.items()}
               for group, ids in chapters.items()}
    strands = _strand_chapters(records, chapters, by_name)
    counts = {"resolved": 0, "dropped": 0}
    out: list[dict] = []
    for rec in records:
        group = (rec.get("subject"), rec.get("grade"))
        known = chapters.get(group)
        if not known or not rec.get("chapterIds"):
            out.append(rec)
            continue
        resolved: list[str] = []
        for cid in rec["chapterIds"]:
            if cid in known:
                chapter = cid
            else:
                names = [rec.get("topic"), (rec.get("tags") or [None])[0],
                         taxonomy_names.get(rec.get("taxonomyChapterId"))]
                strand = (strands.get((group, coarse_code(cid)))
                          if CONTENT_CODE.match(str(cid)) else None)
                chapter = next(
                    (by_name[group][_chapter_key(n)] for n in names
                     if n and _chapter_key(n) in by_name[group]),
                    strand)
                counts["resolved" if chapter else "dropped"] += 1
            if chapter and chapter not in resolved:
                resolved.append(chapter)
        if resolved == list(rec["chapterIds"]):
            out.append(rec)
            continue
        rec = dict(rec)
        rec["chapterIds"] = resolved
        out.append(rec)
    return out, counts


def apply_chapter_names(records: list[dict],
                        names: dict[tuple[str, int, str], str]) -> list[dict]:
    """Put each non-board record's syllabus chapter name first in its tags.

    The phone bundles no syllabus, so its chapter picker names a chapter by the
    first tag of its first question. CBE records carried a topic there, which
    labelled several different chapters "Algebra". Board records already lead
    with the chapter name and are left alone. A chapter id the syllabus does not
    know -- a raw CBE content code like "9N1e" -- keeps its tags unchanged.
    """
    out = []
    for rec in records:
        chapter = (rec.get("chapterIds") or [None])[0]
        name = names.get((rec.get("subject"), rec.get("grade"), chapter))
        if rec.get("source") == BOARD_SOURCE or not name:
            out.append(rec)
            continue
        rec = dict(rec)
        rec["tags"] = [name] + [t for t in rec.get("tags") or [] if t != name]
        out.append(rec)
    return out


def unresolved_chapters(records: list[dict],
                        names: dict[tuple[str, int, str], str]) -> int:
    """How many non-board records have a chapter id the syllabus does not know.

    apply_chapter_names leaves these records' tags alone, so the phone still
    names their chapter by a topic. The root cause is the chapter tagger keeping
    a raw CBE content code; merge_question_banks.py prints this count on every
    run so the gap shows instead of shipping silently.
    """
    return sum(
        1 for rec in records
        if rec.get("source") != BOARD_SOURCE
        and rec.get("chapterIds")
        and (rec.get("subject"), rec.get("grade"), rec["chapterIds"][0]) not in names)


def untagged_chapters(records: list[dict]) -> int:
    """How many non-board records carry no chapter id at all.

    Kept apart from unresolved_chapters: an untagged record (mostly SQP subjects
    with no syllabus file -- Home Science, Odissi, Carnatic Music, Sociology) is
    a different gap from a raw code the syllabus cannot name. Counted together
    they read as 840 unknown ids on the real sources when only ~149 were.
    """
    return sum(1 for rec in records
               if rec.get("source") != BOARD_SOURCE and not rec.get("chapterIds"))


def shared_chapter_labels(records: list[dict]) -> dict[tuple[str, int], dict[str, list[str]]]:
    """(subject, grade) -> {label: chapter ids} wherever one label names several chapters.

    Mirrors the phone's chapter picker (corpus_repository.dart, chapters()): it
    groups questions by every chapter id they carry and names each group by the
    first tag of its first question, falling back to the id. Two chapters with
    one label look like one chapter listed twice.
    """
    first_label: dict[tuple, str] = {}
    for rec in records:
        tags = rec.get("tags") or []
        for cid in rec.get("chapterIds") or []:
            key = (rec.get("subject"), rec.get("grade"), cid)
            first_label.setdefault(key, tags[0] if tags else cid)
    by_label: dict[tuple[str, int], dict[str, list[str]]] = {}
    for (subject, grade, cid), label in first_label.items():
        by_label.setdefault((subject, grade), {}).setdefault(label, []).append(cid)
    return {
        sg: {label: sorted(cids) for label, cids in labels.items() if len(cids) > 1}
        for sg, labels in by_label.items()
        if any(len(cids) > 1 for cids in labels.values())
    }
