"""What it takes for a generated question paper to be called verified.

A paper is made from a template (a CBSE preset for the class, subject and exam), and the checks
here hold the finished paper to that template and to what a teacher can use:

  structure   every section prints exactly the template's number of questions at its marks, and
              the sections add up to the template's total
  keys        every printed question, and every OR alternative, has a marking scheme or answer
  stems       real text: not empty or a fragment, no replacement characters, no unresolved
              reference to a figure, table or passage the paper does not carry
  choices     an MCQ prints four options the printer can split out of its stem
  language    a Hindi paper is in Devanagari, an English one is not
  uniqueness  no question twice in a paper
  kind        in an English or Hindi paper for class 9-10, every question is of the kind its
              section is headed with (reading, grammar, extract, writing, literature), by the
              `paperSection` tag the bank holds for it

`verify` takes the paper as the API returns it (camelCase JSON) and the template, and returns the
problems in plain sentences; none is a verified paper. Nothing here reads the bank or the network:
scripts/verify_papers.py makes the papers and calls this.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..corpus import bank_merge
from . import paper_templates
from . import pdf as pdf_mod

MIN_STEM_CHARS = 12


def _devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def _all_questions(paper: dict) -> Iterable[tuple[str, dict]]:
    for section in paper.get("sections") or []:
        for q in section.get("questions") or []:
            yield section.get("label") or section.get("sectionId") or "?", q


def verify(paper: dict, template: Any, *, language: str = "en",
           tags: Mapping[str, str] | None = None) -> list[str]:
    """Every reason `paper` is not a verified paper for `template`; empty means verified.

    `tags` maps a question id to the `paperSection` the bank holds for it. A paper of a tagged
    kind (class 9-10 English or Hindi) is checked against it when given; the paper itself does not
    carry the tag."""
    problems: list[str] = []
    sections = paper.get("sections") or []
    expected = list(template.sections)

    if paper.get("warnings"):
        problems.append("the paper came with warnings: " + "; ".join(str(w) for w in paper["warnings"][:3]))

    # --- structure
    if len(sections) != len(expected):
        problems.append(f"the paper has {len(sections)} sections; the template has {len(expected)}")
    for got, want in zip(sections, expected):
        label = got.get("label") or want.title
        n = len(got.get("questions") or [])
        if n != want.question_count:
            problems.append(f"{label}: {n} questions printed, the template asks for {want.question_count}")
        wrong = [q.get("marks") for q in got.get("questions") or [] if q.get("marks") != want.marks_each]
        if wrong:
            problems.append(f"{label}: questions worth {sorted(set(wrong))} marks, the template says {want.marks_each} each")
        printed = sum(int(q.get("marks") or 0) for q in got.get("questions") or [])
        if got.get("totalMarks") != printed:
            problems.append(f"{label}: totalMarks says {got.get('totalMarks')} but its questions add to {printed}")
    total = sum(int(s.get("totalMarks") or 0) for s in sections)
    meta_total = (paper.get("metadata") or {}).get("totalMarks")
    if total != template.total_marks or meta_total != template.total_marks:
        problems.append(f"the paper is {total} marks (metadata {meta_total}); the template is {template.total_marks}")

    # --- numbering, stems, choices, language, uniqueness
    numbers = [q.get("displayNumber") for _, q in _all_questions(paper)]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append("the question numbers are not 1, 2, 3, ... in order")
    seen_ids: set[str] = set()
    seen_stems: set[str] = set()
    key = paper.get("answerKey") or {}
    for label, q in _all_questions(paper):
        qid = q.get("questionId") or "?"
        stem = (q.get("stem") or "").strip()
        where = f"{label} Q{q.get('displayNumber')} ({qid})"
        if len(stem) < MIN_STEM_CHARS:
            problems.append(f"{where}: the stem is empty or a fragment")
            continue
        if "�" in stem:
            problems.append(f"{where}: the stem holds a replacement character (broken extraction)")
        # The one gate for a bare stem (bank_merge.raw_extraction_reason): a figure the paper does not
        # print, options the extraction destroyed, a stem that stops mid-sentence.
        reason = bank_merge.raw_extraction_reason(stem)
        # "truncated" is a heuristic ("which of the following" with no (A)-(D)): a stem that ends in a
        # full stop or a question mark is a complete question, as the multi-select "Which of the
        # following points lie on y-axis? A (1, 1), B (1, 0) ... I (3, 3)." is.
        if reason == "truncated" and stem.rstrip().endswith((".", "?", "।", "!")):
            reason = None
        if reason:
            problems.append(f"{where}: the stem needs something the paper does not print ({reason})")
        if qid in seen_ids:
            problems.append(f"{where}: question {qid} is printed twice")
        seen_ids.add(qid)
        norm = re.sub(r"\W+", " ", stem.lower()).strip()
        if norm in seen_stems:
            problems.append(f"{where}: the same stem is printed twice")
        seen_stems.add(norm)
        if q.get("type") in ("mcq", "assertion_reason"):
            _, options = pdf_mod.split_stem_and_options(stem)
            if len(options) != 4:
                problems.append(f"{where}: an MCQ prints {len(options)} options, not 4")
        if language == "hi" and _devanagari_share(stem) < 0.4:
            problems.append(f"{where}: a Hindi paper's stem is not Devanagari")
        if language == "en" and _devanagari_share(stem) > 0.2:
            problems.append(f"{where}: an English paper's stem is in Devanagari")
        number = str(q.get("displayNumber"))
        if not str(key.get(number) or "").strip():
            problems.append(f"{where}: no answer or marking scheme in the key")
    if tags is not None and template.grade in paper_templates._TAGGED_GRADES             and template.subject.lower().startswith(("english", "hindi")):
        for got, want in zip(sections, expected):
            kinds = paper_templates.section_kinds(want)
            if kinds is None:
                continue
            for q in got.get("questions") or []:
                tag = tags.get(q.get("questionId") or "")
                if tag not in kinds:
                    problems.append(f"{got.get('label') or want.title} Q{q.get('displayNumber')} "
                                    f"({q.get('questionId')}): a {tag or 'untagged'} question in a "
                                    f"section that takes {' / '.join(sorted(kinds))}")
    if paper.get("competencyTargetMet") is False:
        problems.append("the competency share is under the board's target")
    return problems
