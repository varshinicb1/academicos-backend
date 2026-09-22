"""Swap or pick one question on a generated paper (Task 903).

The builder shows the paper it just generated; a teacher replaces a question
in one tap ("swap": the next best question for that slot) or puts a question
of their own there ("pick"). The routes are in paper_template_routes.py;
this module decides what may go in a slot and rewrites the paper.

**Model: the saved paper.** Generation already saves every paper, and the
builder opens the paper it just made, so a swap edits that row (and its set
rows) in place rather than a separate unsaved draft: the export, mail and
evaluation routes then see exactly what the teacher sees, with nothing to
"save" and nothing to lose. A paper past principal approval is locked, as it
is for every other edit (routes._require_editable).

**A slot** is a printed question number ("7") or its OR alternative
("7_OR") -- the answer key's own keys, so the client never translates.

**What a swap may put in a slot**, in the order the constraints narrow
(each one is counted, so "no alternative" says which constraint left
nothing):

  same marks and kind as the slot's section (`paper_templates._fits`, the
  generator's own rule) -> verified answer key (rule Q1) -> competency-based
  for an all-competency section -> not on this paper, compulsory or OR ->
  not swapped out of this paper before (a swap never brings a question
  back; a pick can) -> inside the paper's chapters (outside only when the
  paper was generated with fill-from-outside-scope) -> not a near-duplicate
  of any other question on the paper (`pool.near_duplicate`).

Of those, a question on the teacher's recent papers is used only when
nothing else is left, and the note says so. **Recent** = the same teacher's
`RECENT_PAPERS` most recently created other papers for the same class and
subject: about a term of unit and periodic tests, so a class does not meet
a question it answered last month. Ranking then follows the generator:
same difficulty as the question replaced (the section keeps the mix it
was generated with), then quality score, then id -- deterministic.

A pick is the teacher's choice and is refused only for what would make the
paper wrong: no verified answer key (Q1), marks that do not fit the slot
(the section's total would no longer match the template), a question
already on the paper, or a near-duplicate of one. A pick outside the
chapters or of another kind is taken, and the note says so.

Speed: the class+subject bank is mapped to the wire shape and its answer
keys checked once per pool (`bank_for`), not per request. Measured numbers
are in tests/test_paper_swap_pick.py's timing test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from . import grades
from .mapping import to_question_schema
from .paper import answer_key_entry, generated_question, or_answer_key_entry, render_text
from .paper_templates import ScopeFilter, _all_competency, _content_fits, _fits, \
    _is_competency, build_scope_filter, has_verified_key
from .pool import QuestionPool, get_pool, near_duplicate
from .schemas import Camel, GeneratedPaper, GeneratedQuestionSchema, GeneratedSectionSchema, \
    QuestionSchema, TemplateScope, TemplateSection

RECENT_PAPERS = 5


class EditError(Exception):
    """A refused edit: the HTTP status and the detail to send."""

    def __init__(self, status: int, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ---- the bank, mapped once per pool

@dataclass(frozen=True)
class Bank:
    questions: tuple[QuestionSchema, ...]
    by_id: dict[str, QuestionSchema]
    keyed: frozenset[str]


_BANKS: dict[tuple[str, str], tuple[QuestionPool, Bank]] = {}


def _build_bank(pool: QuestionPool, subject: str, roman: str) -> Bank:
    qs = tuple(to_question_schema(c) for c in pool.filter(subject=subject, grade=roman))
    return Bank(qs, {q.id: q for q in qs}, frozenset(q.id for q in qs if has_verified_key(q)))


def bank_for(cfg, subject: str, grade) -> Bank:
    """The class+subject bank as QuestionSchema, with its keyed ids.

    Rebuilt only when the pool object itself changes (`pool.get_pool` builds
    it once per process). Mapping the 408 class 10 Science records and
    checking their keys took 22 ms of every request (2026-09-22). Callers
    must not mutate these objects: copy before attaching anything."""
    roman = grades.to_roman(grade)
    pool = get_pool(cfg, subject=subject, grade=roman)
    key = (subject, roman)                 # get_pool's own key
    hit = _BANKS.get(key)
    if hit is None or hit[0] is not pool:
        hit = (pool, _build_bank(pool, subject, roman))
        _BANKS[key] = hit
    return hit[1]


# ---- slots and what is printed

_SLOT_RE = re.compile(r"^(\d+)(_or)?$", re.I)


@dataclass(frozen=True)
class Printed:
    """One question on the paper: compulsory or an OR alternative."""
    key: str           # "7" or "7_OR"
    id: str
    stem: str

    @property
    def label(self) -> str:
        n, _, alt = self.key.partition("_")
        return f"Q{n}" + (" (OR)" if alt else "")


@dataclass(frozen=True)
class Slot:
    key: str
    alternative: bool
    section: GeneratedSectionSchema
    question: GeneratedQuestionSchema

    @property
    def question_id(self) -> str:
        return (self.question.internal_choice_question_id if self.alternative
                else self.question.question_id)

    @property
    def label(self) -> str:
        return Printed(self.key, "", "").label


def printed_questions(paper: GeneratedPaper) -> list[Printed]:
    out: list[Printed] = []
    for section in paper.sections:
        for q in section.questions:
            out.append(Printed(str(q.display_number), q.question_id, q.stem))
            if q.internal_choice_question_id:
                out.append(Printed(f"{q.display_number}_OR", q.internal_choice_question_id,
                                   q.internal_choice_text or ""))
    return out


def find_slot(paper: GeneratedPaper, key: str) -> Slot:
    m = _SLOT_RE.match(key.strip())
    if m:
        number, alternative = int(m.group(1)), bool(m.group(2))
        for section in paper.sections:
            for q in section.questions:
                if q.display_number != number:
                    continue
                if alternative and not q.internal_choice_question_id:
                    break
                return Slot(f"{number}_OR" if alternative else str(number), alternative,
                            section, q)
    raise EditError(404, f"this paper has no question {key!r} (use a question number, "
                         "or <number>_OR for its OR alternative)")


# ---- the slot's rule

@dataclass
class SlotRule:
    section: TemplateSection
    scope: ScopeFilter
    allow_outside: bool


def rule_for(slot: Slot, paper_template: dict, chapter_ids: list[str], bank: Bank,
             subtopic_question_ids: Optional[Callable[[list[str]], list[str]]] = None,
             ) -> SlotRule:
    """The section rule and scope the slot was generated under.

    A template paper records them (`paperTemplate.sections` / `.scope`, from
    generate-from-template). A paper without them -- quick-generate, or a
    template paper generated before they were recorded -- gets the slot's own
    marks, the replaced question's type, and the assessment's chapters."""
    section = None
    for s in paper_template.get("sections") or []:
        if s.get("id") == slot.section.section_id:
            section = TemplateSection.model_validate(s)
            break
    if section is None:
        current = bank.by_id.get(slot.question_id)
        qtype = current.type if current is not None else (slot.question.type or None)
        section = TemplateSection(
            id=slot.section.section_id, title=slot.section.name or "Section",
            question_count=max(1, len(slot.section.questions)),
            marks_each=slot.question.marks, question_types=[qtype] if qtype else [])
    raw_scope = paper_template.get("scope")
    scope = (TemplateScope.model_validate(raw_scope) if raw_scope is not None
             else TemplateScope(chapter_ids=list(chapter_ids)))
    return SlotRule(section=section,
                    scope=build_scope_filter(scope, list(bank.questions), subtopic_question_ids),
                    allow_outside=bool(paper_template.get("fillFromOutsideScope")))


def _repeats(q: QuestionSchema, printed: list[Printed]) -> Optional[Printed]:
    for p in printed:
        if p.id != q.id and near_duplicate(q.stem, p.stem):
            return p
    return None


# ---- swap

class SwapCounts(Camel):
    """Why the bank has as many alternatives for a slot as it has. Every
    count is of questions the step before let through."""
    fitting: int = 0             # same marks and kind as the slot's section, and
                                 # its question's type when the section names none
    without_key: int = 0         # ... but no verified answer key (Q1)
    not_competency: int = 0      # ... but the section is competency-based only
    on_paper: int = 0            # ... but already on this paper
    swapped_out_earlier: int = 0  # ... but swapped out of this paper before
    outside_scope: int = 0       # ... but outside the paper's chapters
    near_duplicate: int = 0      # ... but repeats a question on the paper
    on_recent_papers: int = 0    # usable, but on one of the teacher's recent papers
    alternatives: int = 0        # usable, recent ones included


@dataclass
class Choice:
    question: QuestionSchema
    notes: list[str]
    counts: Optional[SwapCounts] = None


def choose_swap(slot: Slot, rule: SlotRule, bank: Bank, printed: list[Printed],
                removed: set[str], recent: set[str]) -> Choice:
    others = [p for p in printed if p.key != slot.key]
    on_paper = {p.id for p in printed}
    target = bank.by_id.get(slot.question_id)
    difficulty = target.difficulty if target is not None else slot.question.difficulty
    section = rule.section
    # Same type as the question it replaces. A section that names types
    # already holds the swap to them (`_fits`); one that names none (the
    # class 10 Science annual preset's Sections A-C) lets any type of its
    # marks in at generation, and the served bank's 1-mark keyed questions
    # include 3 long_answer and 1 mcq beside 164 very short answers -- a
    # swap must not turn a very short answer into one of those.
    same_type = None
    if not section.question_types:
        same_type = target.type if target is not None else (slot.question.type or None)
        if same_type:
            section = section.model_copy(update={"question_types": [same_type]})
    all_cbq = _all_competency(section)
    counts = SwapCounts()
    usable: list[tuple[bool, QuestionSchema]] = []
    for q in bank.questions:
        if not _fits(q, rule.section) or (same_type and q.type != same_type):
            continue
        counts.fitting += 1
        inside = rule.scope.in_scope(q)
        if q.id not in bank.keyed:
            counts.without_key += 1
        elif all_cbq and not _is_competency(q):
            counts.not_competency += 1
        elif q.id in on_paper:
            counts.on_paper += 1
        elif q.id in removed:
            counts.swapped_out_earlier += 1
        elif not inside and not rule.allow_outside:
            counts.outside_scope += 1
        elif _repeats(q, others) is not None:
            counts.near_duplicate += 1
        else:
            usable.append((inside, q))
            counts.on_recent_papers += q.id in recent
    counts.alternatives = len(usable)
    if not usable:
        raise EditError(409, {
            "message": f"no alternative in this scope for {slot.label}: "
                       + _no_alternative_reason(section, counts),
            "counts": counts.model_dump(by_alias=True)})
    inside, best = min(usable, key=lambda iq: (
        not iq[0], iq[1].id in recent, iq[1].difficulty != difficulty,
        -iq[1].quality_score, iq[1].id))
    notes = []
    if best.id in recent:
        notes.append(f"{slot.label}: {best.id} is on one of your recent papers (your last "
                     f"{RECENT_PAPERS} for this class and subject) -- no other question "
                     "in the paper's chapters fits this slot.")
    if not inside:
        notes.append(f"{slot.label}: {best.id} is from outside the paper's chapters.")
    return Choice(best, notes, counts)


def _no_alternative_reason(section: TemplateSection, c: SwapCounts) -> str:
    kind = f"{section.marks_each}-mark"
    if section.question_types:
        kind += " " + "/".join(t.replace("_", " ") for t in section.question_types)
    parts = [f"the bank has {c.fitting} {kind} question(s) of this class and subject"]
    for n, text in ((c.without_key, "have no verified answer key"),
                    (c.not_competency, "are not competency-based"),
                    (c.on_paper, "are already on this paper"),
                    (c.swapped_out_earlier, "were swapped out of this paper earlier "
                                            "(pick one by id to bring it back)"),
                    (c.outside_scope, "are outside the paper's chapters"),
                    (c.near_duplicate, "repeat a question already on the paper")):
        if n:
            parts.append(f"{n} {text}")
    return "; ".join(parts) + "."


# ---- pick

def check_pick(slot: Slot, rule: SlotRule, bank: Bank, question_id: str,
               printed: list[Printed], recent: set[str], *, grade: int, subject: str,
               ) -> Choice:
    q = bank.by_id.get(question_id)
    if q is None:
        raise EditError(404, f"{question_id} is not in the Class {grade} {subject} bank")
    if question_id not in bank.keyed:
        raise EditError(422, f"{question_id} has no verified answer key, so it cannot be "
                             "printed (rule Q1: no answer key, no question).")
    marks = slot.question.marks
    if q.marks != marks:
        raise EditError(422, f"{slot.label} is a {marks}-mark slot and {question_id} carries "
                             f"{q.marks} marks: choose a {marks}-mark question, or change "
                             "the section's marks in the template.")
    for p in printed:
        if p.id == question_id:
            raise EditError(409, f"{question_id} is already on the paper as {p.label}.")
    other = _repeats(q, [p for p in printed if p.key != slot.key])
    if other is not None:
        raise EditError(409, f"{question_id} is a near-duplicate of {other.label} "
                             f"({other.id}) on this paper.")
    notes = []
    section = rule.section
    if section.question_types and not _content_fits(q, section.question_types):
        notes.append(f"{slot.label}: {question_id} is not a "
                     f"{'/'.join(t.replace('_', ' ') for t in section.question_types)} "
                     "question, which this section asks for.")
    if _all_competency(section) and not _is_competency(q):
        notes.append(f"{slot.label}: {question_id} is not competency-based, which this "
                     "section asks for.")
    if not rule.scope.in_scope(q):
        notes.append(f"{slot.label}: {question_id} is from outside the paper's chapters.")
    if question_id in recent:
        notes.append(f"{slot.label}: {question_id} is on one of your recent papers (your "
                     f"last {RECENT_PAPERS} for this class and subject).")
    return Choice(q, notes)


# ---- rewriting the paper

def replace_question(paper: GeneratedPaper, old_id: str, new: QuestionSchema) -> GeneratedPaper:
    """`old_id` replaced by `new` wherever it prints -- compulsory or OR, in
    the paper and in each of its sets (a set rotates the questions and swaps
    primary and OR, so the same question sits at a different number there)
    -- with its answer-key entry and the paper's text rebuilt to match."""
    answer_key = dict(paper.answer_key)
    sections = []
    for section in paper.sections:
        questions = []
        for gq in section.questions:
            n = gq.display_number
            if gq.question_id == old_id:
                gq = generated_question(new, n, choice_stem=gq.internal_choice_text,
                                        choice_id=gq.internal_choice_question_id)
                answer_key[str(n)] = answer_key_entry(new)
            elif gq.internal_choice_question_id == old_id:
                gq = gq.model_copy(update={"internal_choice_text": new.stem,
                                           "internal_choice_question_id": new.id})
                answer_key[f"{n}_OR"] = or_answer_key_entry(new.answer_scheme.model_answer)
            questions.append(gq)
        sections.append(section.model_copy(update={"questions": questions}))
    edited = paper.model_copy(update={
        "sections": sections, "answer_key": answer_key,
        "sets": [replace_question(s, old_id, new) for s in paper.sets]})
    edited.formatted_content = render_text(edited)
    return edited
