"""Compose the served question bank from its three sources.

The served bank -- academicos-data/syllabus/questions.json and its byte-identical
APK copy, frontend/assets/corpus/questions.json -- is what every surface reads.
It is composed from:

  board  records already in it with source == "cbse_board_paper", and those
         an earlier run withheld (academicos-data/syllabus/board_withheld.json)
  cbe    academicos-data/corpus/cbse-cbe/questions.json   (classes 6-10)
  sqp    academicos-data/corpus/cbse-sqp/questions.json   (classes 10, 12)

Board records pass through untouched: they already decode on the phone and
render on the web, and this module must not be the thing that breaks them.
One gate applies to them, no-verified-key: a board record is served only with
a marking scheme the relink verified. A board record the gate withholds is kept
WHOLE in board_withheld.json (`hold`), which every run adds to and none
overwrites, and read back as a board source on the next run: the served file is
both this module's board input and its output, so a record left out of it would
otherwise leave the pipeline for good, and a later, better relink could never
re-attach its key.
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

  figure-unavailable     no surface can resolve a `cbe-figure:` asset
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
  mcq-marks-implausible  an MCQ carrying more than 2 marks: a sub-part holding
                         its whole group's marks, or not a single MCQ at all
  mcq-part-of-group      an MCQ whose stem opens "1 (a)": one part of a group
  mcq-options-unresolved the options cannot be read as A-D, fewer than three of
                         them, the stem and the parts disagree about them, or a
                         student could not choose between them (empty, or two
                         read alike)
  mcq-answer-unresolved  the correct option cannot be recovered with certainty
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
                         elsewhere
  stem-runs-on           the next question of the same paper is a stem-fragment:
                         this stem lost its ending to it or took its opening,
                         and its key is likely another question's
  stem-holds-group       a 1-mark item whose stem prints its group's numbered
                         parts, "1 (a) ... 1 (b) ...": several questions under
                         one part's mark, keyed with one part's answer
  duplicate-id           an id already served
  duplicate-stem         already served, under a board-paper id where possible
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..assessment.mapping import _resolve_bloom, _resolve_difficulty, _resolve_type
from ..assessment.question_bank import OfficialAnswer, build_official_scheme
from .mcq_shape import (
    INSTRUCTION, LATER_PART_OPENING, LETTERS, MIN_OPTIONS, PART_LABEL_OPENING, SUB_PART,
    answerable, has_option_labels, holds_group, key_letter, names_option_letter, norm,
    options_from_parts, stem_options, tokens,
)

BOARD_SOURCE = "cbse_board_paper"
NO_VERIFIED_KEY = "no-verified-key"

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


def exclusion_reason(rec: dict) -> str | None:
    """Why this CBE/SQP record cannot be served, or None if it can."""
    meta = rec.get("metadata") or {}
    stem = str(rec.get("stem") or "").strip()
    passage = str(meta.get("passage") or "").strip()
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
    if _FIGURE_REF.search(stem):
        return "figure-referenced"
    answer = _answer_text(rec)
    if (_vowel_sign_runs(stem) >= _GARBLE_RUNS or _vowel_sign_runs(answer) >= _GARBLE_RUNS
            or _script_garbled(rec, f"{stem} {answer}")):
        return "garbled-script"
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
    return scheme.model_dump(mode="json", by_alias=True)


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

    in_parts = options_from_parts(r["parts"])
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

@dataclass
class ComposeResult:
    questions: list[dict]
    excluded: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    # Board records excluded as no-verified-key, whole: `excluded` keeps 200
    # characters of a stem, and these must survive to be relinked again.
    withheld: list[dict] = field(default_factory=list)


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
    "Board records the merge withheld as no-verified-key, kept WHOLE so a later "
    "relink can re-attach a verified key and the next merge serve them again. "
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
            held: list[dict] = ()) -> ComposeResult:
    """Board records from `served` and then `held` (the withheld ones an
    earlier run kept, where `served` has no record of that id), then CBE,
    then SQP.

    Non-board rows already in `served` are ignored and rebuilt from their source
    files, so running the merge twice gives the same bank. A normalised record
    that still breaks the contract raises ContractError and nothing is written.
    """
    out: list[dict] = []
    excluded: list[dict] = []
    withheld: list[dict] = []
    seen_ids: set[str] = set()
    seen_stems: set[tuple] = set()
    counts = {"board": 0, "cbe": 0, "sqp": 0}

    board = [r for r in served if r.get("source") == BOARD_SOURCE]
    in_served = {r.get("id") for r in board}
    board += [r for r in held if r.get("source") == BOARD_SOURCE and r.get("id") not in in_served]
    for rec in board:
        if not has_verified_key(rec):
            excluded.append(_excluded(rec, NO_VERIFIED_KEY))
            withheld.append(rec)
            continue
        errors = contract_errors(rec)
        if errors:
            raise ContractError(str(rec.get("id")), errors)
        out.append(rec)
        seen_ids.add(rec["id"])
        seen_stems.add((rec.get("subject"), rec.get("grade"), norm(rec.get("stem", ""))))
        counts["board"] += 1

    for label, rows in (("cbe", cbe), ("sqp", sqp)):
        runs_on = _runs_on(rows)
        for raw in rows:
            if raw.get("id") in seen_ids:
                excluded.append(_excluded(raw, "duplicate-id"))
                continue
            reason = exclusion_reason(raw)
            if reason:
                excluded.append(_excluded(raw, reason))
                continue
            rec, reason = normalise(raw)
            if reason is None and raw.get("id") in runs_on:
                reason = "stem-runs-on"
            if reason:
                excluded.append(_excluded(raw, reason))
                continue
            key = (rec.get("subject"), rec.get("grade"), norm(raw.get("stem", "")))
            if key in seen_stems:
                excluded.append(_excluded(raw, "duplicate-stem"))
                continue
            errors = contract_errors(rec)
            if errors:
                raise ContractError(str(rec.get("id")), errors)
            out.append(rec)
            seen_ids.add(rec["id"])
            seen_stems.add(key)
            counts[label] += 1

    return ComposeResult(questions=out, excluded=excluded, counts=counts, withheld=withheld)


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
