"""What counts as a competency-based question: one rule, read only from what a
source printed.

CBSE asks that at least half of a paper be competency-based. Whether a
question is one is a fact about the question's source, so this rule reads
only three things a source put there, and takes the first that holds:

  * ``cbeItemBank`` -- the question comes from one of CBSE's Competency-Based
    Education item banks (https://cbseacademic.nic.in/cbe/assessment.html;
    page 1 of every Maths/Science item-bank PDF reads "Competency-based
    education for CBSE"). The importer writes ``questionBankId`` as
    ``"cbe:" + <source pdf>`` (corpus/cbse_cbe.py).
  * ``competencyType`` -- the record's type is one of CBSE's competency
    section kinds (case study, assertion-reason, competency-based) and our
    code did not supply it. A supplied type carries ``metadata.typeInferred``:
    bank_merge sets it when it retypes an item ``mcq``, and mapping.py on
    every record it builds from extraction, where a 5-mark question becomes a
    ``case_study`` by marks alone and a stem that says "passage" becomes one
    by keyword.
  * ``assertionReasonLabels`` -- the stem carries both of CBSE's printed
    labels, "Assertion (A)" and "Reason (R)", in either order (an SQP item
    typed ``mcq`` is still an assertion-reason item).

What the rule never reads, and why (the served bank, composedAt 2026-09-23):

  * the Bloom level -- "understand" on 5,429 of 5,429 records: the bank's
    default, not a judgement;
  * the metadata flags ``is_competency`` / ``cbq`` -- nothing writes them
    (0 of 5,429 records carry either);
  * stem phrases such as "read the following" or "case study" -- they match
    text our merge prepends ("Read the passage below ..."), Exemplar's "Read
    the following statements", and an SQP question whose extraction ran on
    into the next section's heading ("Section E consists of 3 case study
    based questions");
  * marks.

`competency_ceiling` applies the rule to a paper's sections: before any
question is chosen, an estimate of the most competency-based questions they
can print from a bank. selection.optimize raises it to what the paper then
printed (see the function for why an estimate cannot be exact).

This module imports nothing from the academicos package. selection,
paper_templates and syllabus.bank_health all need the rule, and a leaf module
lets each import it without the paper_templates/selection import cycle or
pulling pool.py into bank_health.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

# questionBankId prefix of every item read from a CBSE CBE item-bank PDF.
CBE_ITEM_BANK_PREFIX = "cbe:"

# CBSE's competency-based question kinds, as QuestionSchema types.
COMPETENCY_TYPES = frozenset({"case_study", "assertion_reason", "competency_based"})

# CBSE: at least half of a paper competency-based.
CBSE_COMPETENCY_TARGET = 0.50

# The one sentence every surface shows for "what did you count?".
COMPETENCY_RULE_TEXT = (
    "Counted as competency-based: questions from CBSE's competency-based item banks, "
    "questions their source prints as case-based, assertion-reason or competency-based, "
    "and questions printed with CBSE's Assertion (A) and Reason (R) labels. "
    "A Bloom level, the marks or other wording in a question are never counted.")

_ASSERTION_LABEL = re.compile(r"\bAssertion\s*\(A\)")
_REASON_LABEL = re.compile(r"\bReason\s*\(R\)")


def _field(q: Any, camel: str, snake: str) -> Any:
    """`q`'s field, from a camelCase (or snake_case) record or a QuestionSchema."""
    if isinstance(q, Mapping):
        value = q.get(camel)
        return q.get(snake) if value is None else value
    return getattr(q, snake, None)


def from_cbe_item_bank(q: Any) -> bool:
    """Is `q` from one of CBSE's CBE item banks (``questionBankId`` ``"cbe:..."``)?

    One test of the prefix for the rule below and for selection's tier
    signals (the importer's Bloom level and difficulty are its defaults, not
    judgements). `q` is a QuestionSchema or a camelCase record.
    """
    bank_id = _field(q, "questionBankId", "question_bank_id")
    return isinstance(bank_id, str) and bank_id.startswith(CBE_ITEM_BANK_PREFIX)


def competency_signal(q: Any) -> str | None:
    """Why `q` is competency-based -- ``"cbeItemBank"``, ``"competencyType"`` or
    ``"assertionReasonLabels"`` -- or None when nothing its source printed says so.

    `q` is a QuestionSchema or a camelCase record (the served JSON shape); both
    give the same answer.
    """
    if from_cbe_item_bank(q):
        return "cbeItemBank"
    metadata = _field(q, "metadata", "metadata") or {}
    if _field(q, "type", "type") in COMPETENCY_TYPES and not metadata.get("typeInferred"):
        return "competencyType"
    stem = _field(q, "stem", "stem") or ""
    if _ASSERTION_LABEL.search(stem) and _REASON_LABEL.search(stem):
        return "assertionReasonLabels"
    return None


def is_competency_question(q: Any) -> bool:
    """Is `q` competency-based? See `competency_signal` for the one rule."""
    return competency_signal(q) is not None


def _id(q: Any) -> Any:
    return _field(q, "id", "id")


def _unique(qs: Iterable[Any], exclude: set) -> list[Any]:
    """`qs` once each, in order, leaving out the ids in `exclude`."""
    return list({_id(q): q for q in qs if _id(q) not in exclude}.values())


def competency_ceiling(sections: Iterable[tuple[int, Iterable[Any], Iterable[Any]]],
                       repeats: Callable[[str, str], bool] | None = None) -> int:
    """The most competency-based questions a paper's sections can print, as
    far as can be known before any question is chosen.

    Each section is ``(count, in_scope_fits, fallback_fits)``: how many
    questions it asks for, the questions in the chosen scope that fit it, and
    the ones outside the scope that fit it. A question is a QuestionSchema or
    a camelCase record (as for `competency_signal`), told apart by its id,
    and counted once however often the lists name it: a fallback fit the
    section already counted from its own scope is not borrowed for it again.

    The sections are walked in order over one bank, as selection.optimize
    fills them, and a question placed in one is not counted in another. Each
    section holds up to its count of unused competency-based in-scope fits;
    the rest of its count comes from its other unused in-scope fits, and only
    what those leave short is borrowed from the fallback (optimize borrows
    only for a shortfall), competency-based ones first.

    `repeats(stem_a, stem_b)`, when given, says two stems are the same
    question; selection.optimize passes pool.near_duplicate, which this leaf
    does not import. optimize never prints a question that restates one
    already on the paper, so a fit that restates a question counted before it
    neither counts nor fills its section: the section is that much shorter,
    and borrows for it.

    Still an estimate, not a bound: optimize also prints OR partners, which
    can empty a later section's own fits, and ranks by score, which can take
    the plain question a later section needed. So optimize raises the
    ceiling it reports to the competency-based questions it printed.
    """
    taken: set = set()
    counted: list[str] = []   # stems of the questions counted so far
    ceiling = 0

    def fill(qs: list[Any], n: int) -> list[Any]:
        got: list[Any] = []
        for q in qs:
            if len(got) >= n:
                break
            if repeats is not None:
                stem = _field(q, "stem", "stem") or ""
                if any(repeats(stem, seen) for seen in counted):
                    continue
                counted.append(stem)
            got.append(q)
        return got

    for count, in_scope, fallback in sections:
        free = _unique(in_scope, taken)
        own = fill([q for q in free if is_competency_question(q)], count)
        rest = fill([q for q in free if not is_competency_question(q)], count - len(own))
        short = count - len(own) - len(rest)
        outside = _unique(fallback, taken | {_id(q) for q in free})
        borrowed = fill([q for q in outside if is_competency_question(q)], short)
        taken.update(_id(q) for q in (*own, *rest, *borrowed))
        ceiling += len(own) + len(borrowed)
    return ceiling


def whole_percent(part: int, whole: int) -> int:
    """`part` of `whole` as a whole-number percentage, rounded half up: 1 of 8
    is 13%, where round() would give 12."""
    return (200 * part + whole) // (2 * whole) if whole else 0


def share_summary(competency_based: int, printed: int, set_label: str | None = None) -> str:
    """**The** way a paper's competency share is written, on every surface:
    "Competency-based: 11 of 37 questions (30%)", or for a set other than the
    paper itself "Set B competency-based: 9 of 37 questions (24%)".

    A below-target paper used to state set A's share twice, rounded two ways
    ("11 of 37 (30%)" and "Set A has 29.7%", audit D79): the builder's check,
    the per-set lines and quick-generate each wrote their own. Callers add
    what follows ("Below CBSE's 50%." and why)."""
    noun = "question" if printed == 1 else "questions"
    head = f"Set {set_label} competency-based" if set_label else "Competency-based"
    return f"{head}: {competency_based} of {printed} {noun} ({whole_percent(competency_based, printed)}%)"


def below_target(target: float) -> str:
    return f"Below CBSE's {target * 100:g}%."
