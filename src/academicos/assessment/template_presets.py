"""Read-only paper-template presets: classes 1-10, their main subjects, four exam types.

A teacher should never start from a blank builder. Every (class, subject,
exam type) here is a complete, exactly-summing template; "use as starting
point" copies one into the teacher's own templates (see
paper_template_routes.duplicate), and the copy is what they edit.

Where each pattern comes from -- said in each preset's `source` so a teacher
(or a principal checking the paper) can verify it:

* **Class 10, annual, Mathematics, Science, Social Science** -- `cbse_board`,
  read (Task 904, PyMuPDF) from the CBSE Class X 2024-25 sample question
  papers and marking schemes in `academicos-data/corpus/cbse-sqp/
  ClassX_2024_25/` (`MathsStandard-SQP.pdf`, `Science-SQP.pdf`,
  `SocialScience-SQP.pdf` and their `-MS.pdf`; the PDFs are git-ignored, so
  they are in the main checkout, not in every worktree). Section names,
  counts and marks come from each SQP's General Instructions on p. 1, and
  the internal choice from counting the ORs question by question:
    - Mathematics: OR in 2 questions of B, 2 of C, 2 of D (instruction 8).
    - Science: "Attempt either option A or B" on Q23 and Q25 (B, pp. 5-6),
      Q28 (C, p. 8), Q34-36 (D, pp. 10-13).
    - Social Science: OR on Q22 (B, p. 7), Q26 (C, p. 8), Q30-33 (D,
      pp. 8-9); none in E (case based) or F (map).
  Section E's choice in Mathematics and Science is inside a case unit's
  2-mark sub-part ("Attempt either subpart B or C"): part of the question's
  own text, which a template cannot add, and the source line says so.
  `tests/test_paper_templates.py::test_the_sqp_pdfs_say_what_the_presets_say`
  re-reads all of this from the PDFs where they are present.
  The 2025-26 SQPs (`ClassX_2025_26/`, p. 1 of each) state that "the
  assessment scheme of the Academic Session 2024-25 will continue"; Science
  and Social Science regroup the same questions by discipline (Biology /
  Chemistry / Physics; four 20-mark sections with the map item split into
  Q9 and Q19), so the 2024-25 A-E/F design is kept.
* **Class 9, every subject** -- `suggested`. CBSE sets no class IX paper; the
  annual preset applies the class X board design, which is what most schools
  do, and is labelled as such.
* **English and Hindi, every class** -- `suggested`. The board papers are
  passage-based (a 10-mark reading passage is one "question" with sub-parts),
  which the bank does not hold as such, so the presets print the sub-parts
  as questions and keep each section's weight. English keeps the 2025 board
  paper's section weights at question level, read from `2025/X/
  184_English_Language_and_Literature.zip` -> `2-1-1_English Language and
  Literature.pdf`: Reading 20 (p. 2), Grammar "any ten of twelve" + two
  5-mark writing tasks = 20 (pp. 8-12), Literature 40 = two extracts,
  "any four of five" and "any two of three" 3-mark answers, two 6-mark long
  answers (pp. 13-19). Hindi (class 9-10, half-yearly and annual) follows
  the Hindi Course A (002) SQP 2024-25, `ClassX_2024_25/HindiCourseA-SQP.pdf`,
  read with PyMuPDF (the 2025 board PDFs' Devanagari does not
  extract, the SQP's does): 15 questions in four khands (p. 1); Ka 14 --
  two 7-mark unseen passages, each three 1-mark and two 2-mark sub-questions
  (pp. 1-4); Kha 16 -- four grammar questions, any 4 of 5 one-mark parts
  each (pp. 4-5); Ga 30 -- two 5 x 1 extract MCQ sets, two "any 3 of 4"
  2-mark sets, "any 2 of 3" 4-mark supplementary-reader answers (pp. 5-9);
  Gha 20 -- a 6-mark paragraph on one of three topics, a 5-mark letter and
  a 5-mark CV / e-mail and a 4-mark advertisement / message, each of the
  last three with an OR (pp. 9-10). Course B (085, `HindiCourseB-SQP.pdf`,
  16 questions) weights Ga 28 and Gha 22; the source line says so.
* **Class 6-8, every subject, and every unit test / periodic test** --
  `suggested`: CBSE prescribes no paper design for these. They are common
  school patterns, labelled as a suggestion, not a CBSE rule.
* **Half-yearly, class 9-10** -- `suggested`: the board design applied to the
  half-yearly, which is what most CBSE schools do but not a CBSE rule.
* **Mathematics 6-8, Science 9-10 unit and periodic tests, Science 6 term
  papers** -- `suggested`, written for what the class's question bank holds
  (`_BANK_SHAPED`): the common patterns above ask for 2-mark, 5-mark or
  4-mark case-based questions those banks have almost none of, where CBSE
  requires none of them. Each one's source line says why.

Difficulty mixes are suggestions in every case: CBSE does not publish a
per-section difficulty split.

Long-answer, case-based and map sections carry the question kind they
promise (`_types_for`), so a source item cannot fill a long-answer heading
and a map section is a gap, not any 5-mark question, when the bank has no map
question at that mark value.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Sequence

from .schemas import DifficultyDistribution, GeneratedPaper, GeneratedSectionSchema, \
    PaperTemplate, TemplateHeader, TemplateSection

MAIN_SUBJECTS: tuple[str, ...] = ("Mathematics", "Science", "Social Science", "English", "Hindi")
PRESET_EXAM_TYPES: tuple[str, ...] = ("unit_test", "periodic", "half_yearly", "annual")

_EXAM_NAMES = {
    "unit_test": "Unit Test",
    "periodic": "Periodic Test",
    "half_yearly": "Half-Yearly Examination",
    "annual": "Annual Examination",
}
_DURATION = {"unit_test": 40, "periodic": 90, "half_yearly": 180, "annual": 180}
# The total each exam type is meant to carry. Passed to the model as the
# template's total (not summed from the rows), so a row table that drifts
# from it fails `PaperTemplateDraft`'s exact-totals check at import.
_TOTAL = {"unit_test": 20, "periodic": 40, "half_yearly": 80, "annual": 80}

# Suggested difficulty mix by mark value: short items lean easy, long ones hard.
_MIX = {
    1: DifficultyDistribution(easy=0.5, medium=0.4, hard=0.1),
    2: DifficultyDistribution(easy=0.4, medium=0.5, hard=0.1),
    3: DifficultyDistribution(easy=0.3, medium=0.5, hard=0.2),
    4: DifficultyDistribution(easy=0.2, medium=0.5, hard=0.3),
    5: DifficultyDistribution(easy=0.2, medium=0.4, hard=0.4),
    6: DifficultyDistribution(easy=0.2, medium=0.4, hard=0.4),
}

# (title, questions, marks each, attempt or None, competency share or None
#  [, questions carrying an OR alternative -- the CBSE internal choice])
_Row = tuple  # 5 or 6 fields, see above

_BOARD: dict[str, list[_Row]] = {
    "Mathematics": [
        ("Section A - Objective (MCQ / assertion-reason)", 20, 1, None, None),
        ("Section B - Very short answer", 5, 2, None, None, 2),
        ("Section C - Short answer", 6, 3, None, None, 2),
        ("Section D - Long answer", 4, 5, None, None, 2),
        ("Section E - Case-study based", 3, 4, None, 1.0),
    ],
    "Science": [
        ("Section A - Objective (MCQ / assertion-reason)", 20, 1, None, None),
        ("Section B - Very short answer", 6, 2, None, None, 2),
        ("Section C - Short answer", 7, 3, None, None, 1),
        ("Section D - Long answer", 3, 5, None, None, 3),
        ("Section E - Source / case based", 3, 4, None, 1.0),
    ],
    "Social Science": [
        ("Section A - Objective (MCQ)", 20, 1, None, None),
        ("Section B - Very short answer", 4, 2, None, None, 1),
        ("Section C - Short answer", 5, 3, None, None, 1),
        ("Section D - Long answer", 4, 5, None, None, 4),
        # The SQP's own heading (p. 9): "CASE-BASED QUESTIONS (3X4=12)".
        ("Section E - Case based", 3, 4, None, 1.0),
        ("Section F - Map skill", 1, 5, None, None),
    ],
}

_QP = "cbse-dataset/question-papers/question-paper/2025/X/"
_SQP = "academicos-data/corpus/cbse-sqp/"


def _sqp_source(subject: str, name: str, design: str, choice: str) -> str:
    return (
        f"CBSE Class X Sample Question Paper 2024-25, {subject}, General Instructions, "
        f"{_SQP}ClassX_2024_25/{name}-SQP.pdf, p. 1: {design} = 80. Internal choice: "
        f"{choice} Section headings checked against the marking scheme "
        f"{_SQP}ClassX_2024_25/{name}-MS.pdf. The 2025-26 SQP "
        f"({_SQP}ClassX_2025_26/{name}-SQP.pdf, p. 1) keeps the 2024-25 scheme.")


_BOARD_SOURCE = {
    "Mathematics": _sqp_source(
        "Mathematics (Standard)", "MathsStandard",
        "A 18 MCQ + 2 assertion-reason x1, B 5x2, C 6x3, D 4x5, E 3 case studies x4",
        "an OR in 2 questions each of B, C and D (instruction 8), printed as OR "
        "alternatives. The choice in every case study's 2-mark sub-part is part of "
        "the question's own text; this preset does not add one."),
    "Science": _sqp_source(
        "Science", "Science",
        "A 16 MCQ + 4 assertion-reason x1, B 6x2, C 7x3, D 3x5, E 3 source / case "
        "based units x4",
        "an OR on 2 questions of B (Q23, Q25, pp. 5-6), 1 of C (Q28, p. 8) and all 3 "
        "of D (Q34-36, pp. 10-13), printed as OR alternatives. The choice in each "
        "Section E unit's sub-part (pp. 14-16) is part of the question's own text; "
        "this preset does not add one."),
    "Social Science": _sqp_source(
        "Social Science", "SocialScience",
        "A 20 MCQ x1, B 4x2, C 5x3, D 4x5, E 3 case based x4, F one map question of 5 "
        "(History 2 + Geography 3)",
        "an OR on 1 question of B (Q22, p. 7), 1 of C (Q26, p. 8) and all 4 of D "
        "(Q30-33, pp. 8-9); none in E or F. Printed as OR alternatives."),
}

# Language papers at question level, keeping the SQP's weights (see module doc).
_LANGUAGE_80: list[_Row] = [
    ("Section A - Reading comprehension", 20, 1, None, 1.0),
    ("Section B - Grammar (attempt any 10)", 12, 1, 10, None),
    ("Section B - Writing", 2, 5, None, None),
    ("Section C - Literature: extract based", 10, 1, None, None),
    ("Section C - Literature: short answer (attempt any 4)", 5, 3, 4, None),
    # Q9 of the 2025 board paper: "any two of the following three" at 3 marks
    # (p. 17). The earlier 3 x 2-mark row matched no paper.
    ("Section C - Literature: short answer II (attempt any 2)", 3, 3, 2, None),
    ("Section C - Literature: long answer (attempt any 2)", 3, 6, 2, None),
]

# Hindi Course A (002), CBSE Class X SQP 2024-25 at question level (see
# module doc and `_HINDI_SOURCE`): Ka 6x1 + 4x2 = 14, Kha 16 of 20 x1 = 16,
# Ga 10x1 + 6 of 8 x2 + 2 of 3 x4 = 30, Gha 1 of 3 x6 + 2x5 + 1x4 = 20.
_HINDI_80: list[_Row] = [
    ("Section A (Khand Ka) - Reading: unseen passages, objective", 6, 1, None, 1.0),
    ("Section A (Khand Ka) - Reading: unseen passages, short answer", 4, 2, None, 1.0),
    ("Section B (Khand Kha) - Grammar (attempt any 16)", 20, 1, 16, None),
    ("Section C (Khand Ga) - Textbook: extract based MCQ", 10, 1, None, None),
    ("Section C (Khand Ga) - Textbook: short answer (attempt any 6)", 8, 2, 6, None),
    ("Section C (Khand Ga) - Supplementary reader (attempt any 2)", 3, 4, 2, None),
    ("Section D (Khand Gha) - Writing: paragraph (attempt any 1)", 3, 6, 1, None),
    ("Section D (Khand Gha) - Writing: letter, CV / e-mail", 2, 5, None, None, 2),
    ("Section D (Khand Gha) - Writing: advertisement / message", 1, 4, None, None, 1),
]
_HINDI_SOURCE = (
    f"the CBSE Class X Hindi Course A (002) Sample Question Paper 2024-25, "
    f"{_SQP}ClassX_2024_25/HindiCourseA-SQP.pdf, at question level: p. 1 General "
    "Instructions, 15 questions in four khands, 80 marks; Khand Ka Q1-2, two unseen "
    "passages of 7 marks, each three 1-mark and two 2-mark sub-questions (pp. 1-4); "
    "Khand Kha Q3-6, grammar, any 4 of 5 one-mark sub-questions each (pp. 4-5); "
    "Khand Ga Q7 and Q9 five 1-mark extract MCQs each, Q8 and Q10 any 3 of 4 two-mark "
    "answers, Q11 any 2 of 3 four-mark supplementary-reader answers (pp. 5-9); Khand "
    "Gha Q12 a 6-mark paragraph on one of three topics, Q13 letter and Q14 CV / "
    "e-mail of 5 marks and Q15 advertisement / message of 4, each with an OR "
    "(pp. 9-10). The passages are one question with sub-parts in the paper; here "
    "each sub-part prints as a question. Hindi Course B (085) differs "
    f"({_SQP}ClassX_2024_25/HindiCourseB-SQP.pdf: 16 questions, Khand Ga 28 and "
    "Gha 22) -- edit the sections for it.")

# Class 6-8 school pattern for the term exams, maths / science / social science.
_JUNIOR_80: list[_Row] = [
    ("Section A - Objective", 16, 1, None, None),
    ("Section B - Very short answer", 8, 2, None, None),
    ("Section C - Short answer", 8, 3, None, None),
    ("Section D - Long answer", 4, 5, None, None),
    ("Section E - Case based", 1, 4, None, 1.0),
]
_JUNIOR_LANGUAGE_80: list[_Row] = [
    ("Section A - Reading comprehension", 15, 1, None, 1.0),
    ("Section B - Grammar", 15, 1, None, None),
    ("Section C - Writing", 2, 5, None, None),
    ("Section D - Literature: extract based", 10, 1, None, None),
    ("Section D - Literature: short answer", 6, 3, None, None),
    ("Section D - Literature: long answer", 2, 6, None, None),
]

_PERIODIC_40: list[_Row] = [
    ("Section A - Objective", 10, 1, None, None),
    ("Section B - Very short answer", 4, 2, None, None),
    ("Section C - Short answer", 4, 3, None, None),
    ("Section D - Long answer", 2, 5, None, None),
]
_PERIODIC_LANGUAGE_40: list[_Row] = [
    ("Section A - Reading comprehension", 10, 1, None, 1.0),
    ("Section B - Grammar", 10, 1, None, None),
    ("Section C - Writing", 1, 5, None, None),
    ("Section D - Literature: extract based", 5, 1, None, None),
    ("Section D - Literature: long answer", 2, 5, None, None),
]

_UNIT_20: list[_Row] = [
    ("Section A - Objective", 6, 1, None, None),
    ("Section B - Very short answer", 4, 2, None, None),
    ("Section C - Short answer", 2, 3, None, None),
]

# Suggested patterns for the classes whose question bank cannot print the
# shapes above, where CBSE requires none of them (unit and periodic tests,
# class 6-8 term papers). Measured through the builder's own calls on the
# served bank (2026-09-30), 18 of the 40 class 6-10 Mathematics/Science
# whole-syllabus presets reached their marks (audit D85/D18). The rest were
# short for three shapes the bank has almost none of, not for a lack of
# questions: 2-mark questions in Mathematics 6-8 and Science 9-10 (2-3 and
# 1-2 keyed; the NCERT Exemplar items these banks are built from come at 1,
# 3 and 5 marks), 5-mark long answers in Mathematics 6-8 (1-3 keyed), and a
# 4-mark case-based question in Science 6 (none; its case-based questions
# carry 6). These patterns keep each paper's total and ask only for what the
# class's bank holds. Class 9-10 half-yearly and annual papers keep the
# board design: there the missing 2-mark and case-based questions are the
# bank's gap, and the builder names them.
_UNIT_20_1_3: list[_Row] = [
    ("Section A - Objective", 8, 1, None, None),
    ("Section B - Short answer", 4, 3, None, None),
]
_PERIODIC_40_1_3: list[_Row] = [
    ("Section A - Objective", 16, 1, None, None),
    ("Section B - Short answer", 8, 3, None, None),
]
_PERIODIC_40_1_3_5: list[_Row] = [
    ("Section A - Objective", 10, 1, None, None),
    ("Section B - Short answer", 5, 3, None, None),
    ("Section C - Long answer", 3, 5, None, None),
]
_JUNIOR_80_1_3: list[_Row] = [
    ("Section A - Objective", 16, 1, None, None),
    ("Section B - Short answer", 20, 3, None, None),
    ("Section C - Case based", 1, 4, None, 1.0),
]
_JUNIOR_80_CASE_6: list[_Row] = [
    ("Section A - Objective", 16, 1, None, None),
    ("Section B - Very short answer", 7, 2, None, None),
    ("Section C - Short answer", 8, 3, None, None),
    ("Section D - Long answer", 4, 5, None, None),
    ("Section E - Case based", 1, 6, None, 1.0),
]
_NO_2_MARK = ("its question bank holds almost no 2-mark questions (the NCERT Exemplar "
              "items it is built from come at 1, 3 and 5 marks)")
_NO_2_OR_5_MARK = ("its question bank holds almost no 2-mark or 5-mark questions (the NCERT "
                   "Exemplar items it is built from are nearly all 1 and 3 marks)")
# (subject, class) -> exam type -> (rows, why the class's pattern differs).
_BANK_SHAPED: dict[tuple[str, int], dict[str, tuple[list[_Row], str]]] = {
    **{("Mathematics", g): {
        "unit_test": (_UNIT_20_1_3, _NO_2_OR_5_MARK),
        "periodic": (_PERIODIC_40_1_3, _NO_2_OR_5_MARK),
        "half_yearly": (_JUNIOR_80_1_3, _NO_2_OR_5_MARK),
        "annual": (_JUNIOR_80_1_3, _NO_2_OR_5_MARK),
    } for g in (6, 7, 8)},
    **{("Science", g): {
        "unit_test": (_UNIT_20_1_3, _NO_2_MARK),
        "periodic": (_PERIODIC_40_1_3_5, _NO_2_MARK),
    } for g in (9, 10)},
    ("Science", 6): {
        "half_yearly": (_JUNIOR_80_CASE_6, "its question bank has no 4-mark case-based "
                                          "question; its case-based questions carry 6 marks"),
        "annual": (_JUNIOR_80_CASE_6, "its question bank has no 4-mark case-based "
                                     "question; its case-based questions carry 6 marks"),
    },
}
_UNIT_LANGUAGE_20: list[_Row] = [
    ("Section A - Reading comprehension", 5, 1, None, 1.0),
    ("Section B - Grammar", 5, 1, None, None),
    ("Section C - Writing", 1, 5, None, None),
    ("Section D - Literature", 5, 1, None, None),
]

# Classes 1-5 (requirements v3): CBSE prescribes no paper design for primary
# classes, so these are suggested school patterns -- shorter papers, no answer
# above 5 marks, and EVS from class 3 (the NCERT books' own range). Classes 1-5
# share one pattern per exam type; a school edits it for its younger classes.
PRIMARY_SUBJECTS: dict[int, tuple[str, ...]] = {
    1: ("English", "Hindi", "Mathematics"), 2: ("English", "Hindi", "Mathematics"),
    3: ("English", "Hindi", "Mathematics", "EVS"), 4: ("English", "Hindi", "Mathematics", "EVS"),
    5: ("English", "Hindi", "Mathematics", "EVS"),
}
_PRIMARY_TOTAL = {"unit_test": 20, "periodic": 25, "half_yearly": 50, "annual": 50}
_PRIMARY_DURATION = {"unit_test": 40, "periodic": 60, "half_yearly": 120, "annual": 120}
_PRIMARY: dict[str, list[_Row]] = {
    "unit_test": [
        ("Section A - Objective (MCQ, fill in the blanks, true or false)", 10, 1, None, None),
        ("Section B - Short answer", 5, 2, None, None),
    ],
    "periodic": [
        ("Section A - Objective (MCQ, fill in the blanks, true or false)", 10, 1, None, None),
        ("Section B - Very short answer", 5, 1, None, None),
        ("Section C - Short answer", 5, 2, None, None),
    ],
    "half_yearly": [
        ("Section A - Objective (MCQ, fill in the blanks, true or false)", 15, 1, None, None),
        ("Section B - Very short answer", 10, 1, None, None),
        ("Section C - Short answer", 8, 2, None, None),
        ("Section D - Answer in a few sentences", 3, 3, None, None),
    ],
}
_PRIMARY["annual"] = _PRIMARY["half_yearly"]
_PRIMARY_LANGUAGE: dict[str, list[_Row]] = {
    "unit_test": [
        ("Section A - Reading comprehension", 5, 1, None, 1.0),
        ("Section B - Grammar and vocabulary", 5, 1, None, None),
        ("Section C - Writing", 2, 5, None, None),
    ],
    "periodic": [
        ("Section A - Reading comprehension", 5, 1, None, 1.0),
        ("Section B - Grammar and vocabulary", 10, 1, None, None),
        ("Section C - Writing", 2, 5, None, None),
    ],
    "half_yearly": [
        ("Section A - Reading comprehension", 10, 1, None, 1.0),
        ("Section B - Grammar and vocabulary", 15, 1, None, None),
        ("Section C - Writing", 2, 5, None, None),
        ("Section D - Literature", 5, 1, None, None),
        ("Section E - Literature, short answer", 5, 2, None, None),
    ],
}
_PRIMARY_LANGUAGE["annual"] = _PRIMARY_LANGUAGE["half_yearly"]

_SUGGESTED_SOURCE = (
    "Suggested school pattern, not a CBSE rule: CBSE prescribes no paper design "
    "for {what}. Edit it to match your school.")


def _pattern(grade: int, subject: str, exam_type: str) -> tuple[list[_Row], str, str]:
    """(rows, pattern_status, source) for one preset."""
    language = subject in ("English", "Hindi")
    if grade <= 5:
        return ((_PRIMARY_LANGUAGE if language else _PRIMARY)[exam_type], "suggested",
                _SUGGESTED_SOURCE.format(what=f"class {grade}"))
    shaped = _BANK_SHAPED.get((subject, grade), {}).get(exam_type)
    if shaped is not None:
        rows, why = shaped
        what = {"unit_test": "unit tests", "periodic": "periodic tests"}.get(
            exam_type, f"class {grade} term examinations")
        return rows, "suggested", (
            f"Suggested school pattern, not a CBSE rule: CBSE prescribes no paper design for "
            f"{what}. Written for Class {grade} {subject}, because {why}. Edit it to match "
            "your school.")
    if exam_type == "unit_test":
        return (_UNIT_LANGUAGE_20 if language else _UNIT_20, "suggested",
                _SUGGESTED_SOURCE.format(what="unit tests"))
    if exam_type == "periodic":
        return (_PERIODIC_LANGUAGE_40 if language else _PERIODIC_40, "suggested",
                _SUGGESTED_SOURCE.format(what="periodic tests"))
    if grade >= 9 and not language:
        rows = _BOARD[subject]
        if exam_type == "annual" and grade == 10:
            return rows, "cbse_board", _BOARD_SOURCE[subject]
        if exam_type == "annual":
            # No CBSE paper exists for class IX to check a preset against.
            return rows, "suggested", (
                "Suggested: CBSE sets no class IX paper; this is the class X board "
                "design, which most schools use for class IX. Class X design: "
                + _BOARD_SOURCE[subject])
        return rows, "suggested", (
            "Suggested: the CBSE board design applied to the half-yearly, as most "
            "CBSE schools do; not a CBSE rule. Board design: " + _BOARD_SOURCE[subject])
    if language and grade >= 9:
        if subject == "English":
            return _LANGUAGE_80, "suggested", (
                "Suggested: keeps the CBSE Class X English board paper 2025's section "
                "weights at question level (Reading 20, Grammar + Writing 20, "
                f"Literature 40), read from {_QP}184_English_Language_and_Literature.zip "
                "-> 2-1-1_English Language and Literature.pdf, pp. 2-19. The paper "
                "itself is passage-based and not reproduced exactly.")
        if exam_type == "annual" and grade == 10:
            return _HINDI_80, "suggested", "Suggested: follows " + _HINDI_SOURCE
        what = ("CBSE sets no class IX paper; this is the class X design"
                if grade == 9 else "the class X board design applied to the half-yearly")
        return _HINDI_80, "suggested", (
            f"Suggested: {what}, which most schools use; not a CBSE rule. Class X "
            "design: " + _HINDI_SOURCE)
    return (_JUNIOR_LANGUAGE_80 if language else _JUNIOR_80), "suggested", \
        _SUGGESTED_SOURCE.format(what=f"class {grade} term examinations")


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-")


def instructions_for(sections: list[TemplateSection],
                     printed: Optional[Sequence[tuple[int, int]]] = None) -> str:
    """The General Instructions for `sections`, as a preset writes them.

    With `printed` -- (questions, OR alternatives) per section, in order,
    read off the paper actually generated -- the lines describe that paper
    instead of the template: a section the bank left empty is not listed,
    a short one states the questions it holds, and a section says only the
    ORs it really carries. Written once from `choice_count`, the class 10
    Social Science preset printed "Section D ... 4 of them offer an internal
    choice (OR)" on a paper whose Section D had none (the real bank pairs 0
    of 4, 2026-09-22), and these lines replace the PDF's default list."""
    if printed is None:
        rows = [(s, s.question_count, s.choice_count) for s in sections]
    else:
        rows = [(s, n, c) for s, (n, c) in zip(sections, printed) if n > 0]
    return _instructions([(s.title, count, s.marks_each, min(s.attempts, count), choices)
                          for s, count, choices in rows])


_FIRST_LINE = "This question paper has {n} section{s}."
_LAST_LINE = "All questions are compulsory unless a choice is stated."


def _instructions(rows: Sequence[tuple[str, int, Optional[int], int, int]]) -> str:
    """The generated lines, one per (title, questions, marks each or None
    when they differ, attempts, OR alternatives) section row."""
    lines = [_FIRST_LINE.format(n=len(rows), s="" if len(rows) == 1 else "s")]
    for title, count, marks_each, attempts, choices in rows:
        attempt = f" Attempt any {attempts}." if attempts < count else ""
        choice = (f" {choices} of them offer an internal choice (OR): attempt only "
                  "one of the two." if choices else "")
        each = (f" of {marks_each} mark(s) each" if marks_each is not None
                else ", marks as indicated")
        lines.append(f"{title}: {count} question(s){each}.{attempt}{choice}")
    lines.append(_LAST_LINE)
    return "\n".join(lines)


_ATTEMPT_SUFFIX = re.compile(r"\s*\(attempt any \d+ of \d+\)$")


def section_title(name: str) -> str:
    """A printed section's name without the "(attempt any N of M)" the
    builder adds when a section prints more questions than are answered."""
    return _ATTEMPT_SUFFIX.sub("", name)


def section_attempts(section: GeneratedSectionSchema) -> int:
    """How many of a printed section's questions are answered: all of them,
    unless its marks say fewer ("attempt any 10 of 12" is worth 10 marks)."""
    marks = {q.marks for q in section.questions}
    count = len(section.questions)
    if len(marks) != 1:
        return count
    each = marks.pop()
    return min(count, section.total_marks // each) if each else count


def instructions_for_paper(paper: GeneratedPaper) -> str:
    """The generated General Instructions for `paper` as it stands now --
    its own sections, each with the questions and OR alternatives it
    prints -- in the words `instructions_for` uses.

    Stamped once at generation, the lines went stale with the paper's first
    edit: remove Q38 and one of Section B's two ORs on a class 10 Maths
    half-yearly paper and the PDF and Word file still said "Section E ... 3
    question(s)" and "Section B ... 2 of them offer an internal choice (OR)"
    (v3 audit N-2-6). Every place that prints or stores the instructions
    calls this on the paper it has."""
    rows = []
    for s in paper.sections:
        if not s.questions:
            continue
        marks = {q.marks for q in s.questions}
        rows.append((section_title(s.name), len(s.questions),
                     marks.pop() if len(marks) == 1 else None, section_attempts(s),
                     sum(1 for q in s.questions if q.internal_choice_question_id)))
    return _instructions(rows)


_GENERATED_LINE = re.compile(
    r".+: \d+ question\(s\)(?: of \d+ mark\(s\) each|, marks as indicated)\."
    r"(?: Attempt any \d+\.)?"
    r"(?: \d+ of them offer an internal choice \(OR\): attempt only one of the two\.)?")


def reads_as_generated(text: str) -> bool:
    """Whether `text` is, line for line, instructions this module generated
    -- for a paper stored before it said so itself
    (`PaperMetadataSchema.instructions_generated` None)."""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return (len(lines) >= 2
            and re.fullmatch(r"This question paper has \d+ sections?\.", lines[0]) is not None
            and lines[-1] == _LAST_LINE
            and all(_GENERATED_LINE.fullmatch(x) for x in lines[1:-1]))


def _types_for(title: str) -> list[str]:
    """The question kind a heading promises, where the engine can check it.

    Without this every section was matched by marks alone, and the class 10
    Social Science annual preset reported itself complete while printing
    "Read the given source" items under "Long answer" and outline-map items
    under "Case based". The engine reads map and case / source from the
    question text (paper_templates._content_fits), not from the bank's
    mark-derived type label, so these filters do not empty the real long
    answers (labelled case_study at 5 marks) or case studies (labelled
    long_answer at 4). Other headings (objective, short answer, grammar...)
    stay untyped: nothing in the bank could check them, and the engine says
    so for the ones that name a kind."""
    low = title.lower()
    if "map" in low:
        return ["map_based"]
    if "case" in low:
        return ["case_study"]
    if "long answer" in low:
        return ["long_answer", "very_long_answer"]
    return []


def _build(grade: int, subject: str, exam_type: str) -> PaperTemplate:
    rows, status, source = _pattern(grade, subject, exam_type)
    sections = [
        TemplateSection(
            id=f"s{i + 1}", title=title, question_count=count, marks_each=marks,
            attempt_count=attempt, difficulty_mix=_MIX.get(marks),
            competency_share=competency, question_types=_types_for(title),
            choice_count=choices[0] if choices else 0,
        )
        for i, (title, count, marks, attempt, competency, *choices) in enumerate(rows)
    ]
    exam_name = _EXAM_NAMES[exam_type]
    return PaperTemplate(
        id=f"preset:cbse:{grade}:{_slug(subject)}:{exam_type}",
        school_id="",
        name=f"Class {grade} {subject} - {exam_name}",
        grade=grade,
        subject=subject,
        exam_type=exam_type,
        total_marks=(_PRIMARY_TOTAL if grade <= 5 else _TOTAL)[exam_type],
        duration_minutes=(_PRIMARY_DURATION if grade <= 5 else _DURATION)[exam_type],
        instructions=instructions_for(sections),
        instructions_generated=True,
        header=TemplateHeader(exam_name=exam_name),
        sections=sections,
        is_preset=True,
        pattern_status=status,
        source=source,
    )


@lru_cache(maxsize=1)
def _all() -> tuple[PaperTemplate, ...]:
    return tuple(
        _build(grade, subject, exam_type)
        for grade in range(1, 11)
        for subject in (PRIMARY_SUBJECTS[grade] if grade <= 5 else MAIN_SUBJECTS)
        for exam_type in PRESET_EXAM_TYPES
    )


def presets(*, grade: int | None = None, subject: str | None = None) -> list[PaperTemplate]:
    """Copies, so a caller mutating one can never change the next caller's preset."""
    return [
        p.model_copy(deep=True) for p in _all()
        if (grade is None or p.grade == grade)
        and (subject is None or p.subject.lower() == subject.lower())
    ]


def get_preset(preset_id: str) -> PaperTemplate | None:
    for p in _all():
        if p.id == preset_id:
            return p.model_copy(deep=True)
    return None


def is_preset_id(template_id: str) -> bool:
    return template_id.startswith("preset:")
