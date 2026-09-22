"""Filling a teacher's paper template from the bank: availability, then selection.

One engine does both, on purpose. The availability check the builder shows
BEFORE generating and the paper generation itself walk the same candidate
list, in the same section order, with the same claiming rule -- so a section
reported as "4 of 4 available" cannot come back short, and a gap the teacher
was shown is exactly the gap the paper has.

Eligibility, in the order it narrows (each step is reported, so a shortfall
names the constraint that caused it rather than a bare zero):

  bank (class + subject) -> verified answer key (PRD rule Q1: no answer key,
  no question) -> marks match the section -> the section's question kind
  (read from the question's content where the bank's type label cannot be
  trusted, see `_content_fits`) -> competency-based, for a section that is
  all competency questions -> inside the template's scope (chapters,
  subtopics, topics)

The availability report is not an estimate beside the generator: it is
`plan`, the generator's own selection, run without saving. Each section's
`filled`, `notes` (difficulty and competency shortfalls, borrowed questions)
and gaps are read off the questions actually picked, so the builder shows
before generating every note the paper comes back with. Measured on the
real class 10 bank (2026-09-22) before this: 16 of the 20 presets (all but
Hindi, which has no keyed question) carried notes the check never showed,
among them Science "Section E - Source / case based" at 3 of 3 available
with 1 competency question.

A shortfall's reason names the constraint that left questions out, largest
first, and `fixes` lists the changes that would close it (widen the scope,
change the marks, relax the kind or the competency rule, ask for fewer).

A section with internal choice (`choice_count`) pairs that many of its
questions with an OR alternative, drawn from the same eligible pool and
claimed like any pick, so availability counts the alternatives the paper
will print and a later section cannot reuse them. An OR the bank cannot
pair is a note and a fix, not a gap: the paper still carries its marks.

A heading that names a kind nothing here can check (grammar, reading,
extract, writing, literature -- the bank does not label questions that way)
is reported as "section content not checked" instead of passing silently.

A section with a shortfall is filled from outside the scope only when the
caller asks (`fill_from_outside_scope`), and every borrowed question is
named in the notes. Unlike `selection.optimize`, nothing is borrowed silently:
the selector's own fallback exists for the one-click path, where there is no
builder to show the gap first.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional

from .pool import near_duplicate
from .schemas import Camel, DifficultyDistribution, PaperTemplateDraft, QuestionSchema, \
    QuestionType, TemplateScope, TemplateSection

# Where the board-level topic taxonomy will live (being built in another
# stream). Its absence is reported, never treated as "every id is invalid".
TAXONOMY_DIR = Path(__file__).resolve().parents[3] / "academicos-data" / "syllabus" / "taxonomy"


# Keys under which the taxonomy files may name an entry. The taxonomy's final
# shape is another stream's decision; reading every id-like key at any depth
# keeps this check correct for a nested chapters -> topics -> subtopics file
# and for a flat list alike.
_TAXONOMY_ID_KEYS = ("id", "topicId", "topic_id", "subtopicId", "subtopic_id")

UNCHECKED_NOTE = ("Topic and subtopic ids were accepted without checking: the topic "
                  "taxonomy (academicos-data/syllabus/taxonomy/) is not on this server yet.")


def taxonomy_ids() -> Optional[set[str]]:
    """Every id the taxonomy files define, or None when there is no taxonomy.
    Re-read only when a file changes (keyed on names and mtimes), so a
    taxonomy dropped in by the other stream is picked up without a restart."""
    root = TAXONOMY_DIR
    if not root.is_dir():
        return None
    files = tuple(sorted((str(f), f.stat().st_mtime_ns) for f in root.rglob("*.json")))
    return set(_read_taxonomy_ids(files))


@lru_cache(maxsize=4)
def _read_taxonomy_ids(files: tuple[tuple[str, int], ...]) -> frozenset[str]:
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key in _TAXONOMY_ID_KEYS:
                if isinstance(node.get(key), str):
                    found.add(node[key])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for path, _ in files:
        try:
            walk(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue  # one unreadable file must not reject every template
    return frozenset(found)


def check_scope_ids(scope: TemplateScope) -> tuple[list[str], list[str]]:
    """(unknown ids, notes) for the scope's topic and subtopic ids.

    With a taxonomy present, an id it does not define is unknown -- the
    caller answers 422 naming it, since a mistyped id would otherwise filter
    silently to nothing. Without one, the ids are accepted as given and the
    note says they were not checked. Chapter ids are not checked here: they
    are the bank's own ids, already validated by the chapter picker."""
    asked = [*scope.topic_ids, *scope.subtopic_ids]
    if not asked:
        return [], []
    known = taxonomy_ids()
    if known is None:
        return [], [UNCHECKED_NOTE]
    if not known:
        return [], ["Topic and subtopic ids were accepted without checking: the topic "
                    "taxonomy on this server defines no ids."]
    return [i for i in asked if i not in known], []


FixKind = Literal["widen_scope", "fill_from_outside_scope", "change_marks", "relax_type",
                  "relax_competency", "reduce_count", "reduce_choices"]


class SectionFix(Camel):
    """One change the teacher can make to a short section, with what it buys.

    A reason alone ("Only 3 are available") left the teacher guessing which
    of five settings to change; each fix names one setting and how many more
    questions changing it makes available, so the builder can offer it as a
    single tap: every kind carries the value to apply, never only a
    sentence to parse."""
    kind: FixKind
    label: str
    # How many more questions the change makes available to this section.
    available: int = 0
    # change_marks: the mark value to switch the section to.
    marks_each: Optional[int] = None
    # reduce_count: the count the bank can fill (0 = drop the section).
    question_count: Optional[int] = None
    # reduce_choices: the number of ORs the bank can pair.
    choice_count: Optional[int] = None
    # relax_type: the section's question types after relaxing -- its own
    # types plus the one kind admitted, to set as `questionTypes` as is.
    question_types: Optional[list[QuestionType]] = None


class SectionAvailability(Camel):
    section_id: str
    title: str
    marks_each: int
    needed: int
    available: int
    shortfall: int
    # What the generated paper will hold for this section: the in-scope
    # questions it takes plus any borrowed from outside the scope. The same
    # number generation prints -- both come from one `plan` run.
    filled: int = 0
    # Taken from outside the chosen chapters (only when the caller allows it).
    borrowed: int = 0
    # Eligible (keyed, right marks and type) but outside the chosen scope.
    outside_scope_available: int = 0
    # In scope with the right marks and type, but no verified answer key.
    without_key: int = 0
    # Internal choice: ORs the section asks for, and ORs the paper will print.
    choices_needed: int = 0
    choices_filled: int = 0
    reason: str = ""
    # Everything generation will report about this section -- difficulty and
    # competency shortfalls, borrowed questions, headings nothing could check.
    # Identical to the notes the generated paper comes back with.
    notes: list[str] = []
    fixes: list[SectionFix] = []


class AvailabilityReport(Camel):
    template_id: str
    grade: int
    subject: str
    bank_questions: int
    keyed_questions: int
    complete: bool
    sections: list[SectionAvailability]
    scope_notes: list[str] = []


def has_verified_key(q: QuestionSchema) -> bool:
    """Rule Q1, the same content check the question-bank API applies
    (`QuestionBank.has_answer_key`): a scheme with actual text in it, never
    just a provenance label. Reused rather than copied so the two surfaces
    can never disagree about which questions are answerable."""
    from .qbank_routes import QuestionBank
    return QuestionBank.has_answer_key(q.model_dump(by_alias=True, mode="json"))


@dataclass
class ScopeFilter:
    """`in_scope(q)` plus the notes explaining what the filter could not do."""
    in_scope: Callable[[QuestionSchema], bool]
    notes: list[str] = field(default_factory=list)


def build_scope_filter(scope: TemplateScope, candidates: list[QuestionSchema],
                       subtopic_question_ids: Optional[Callable[[list[str]], list[str]]] = None,
                       ) -> ScopeFilter:
    notes: list[str] = []
    chapters = set(scope.chapter_ids)

    topics = set(scope.topic_ids)
    unknown, id_notes = check_scope_ids(scope)
    notes.extend(id_notes)
    if unknown:
        # Only a template saved before the taxonomy arrived gets here (saving
        # refuses unknown ids); report it rather than refuse to plan the paper.
        notes.append("Not in the topic taxonomy (edit the template's scope): "
                     + ", ".join(unknown) + ".")
    if topics and not any(_topic_ids(q) for q in candidates):
        # No question carries topic tags yet. Filtering on them would empty
        # every section and blame the bank for a tagging gap; say so instead.
        notes.append("Topic filter not applied: no question in the bank is tagged "
                      "with topics yet, so the paper uses the chosen chapters.")
        topics = set()

    subtopic_ids: Optional[set[str]] = None
    if scope.subtopic_ids:
        tagged = set(subtopic_question_ids(scope.subtopic_ids)) if subtopic_question_ids else set()
        if tagged:
            subtopic_ids = tagged
        else:
            notes.append("Subtopic filter not applied: no question is tagged to the "
                          "chosen subtopics yet, so the paper uses the chosen chapters.")

    def in_scope(q: QuestionSchema) -> bool:
        if chapters and not chapters.intersection(q.chapter_ids):
            return False
        if subtopic_ids is not None and q.id not in subtopic_ids:
            return False
        if topics and not topics.intersection(_topic_ids(q)):
            return False
        return True

    return ScopeFilter(in_scope=in_scope, notes=notes)


def _topic_ids(q: QuestionSchema) -> list[str]:
    ids = q.metadata.get("topicIds") or q.metadata.get("topic_ids") or []
    single = q.metadata.get("topicId") or q.metadata.get("topic_id")
    return [*ids, single] if single else list(ids)


# Evidence in the question's own text of what kind of question it is. The
# served bank's type labels cannot say this: when extraction found no type,
# mapping._resolve_type derives one from marks (3-4 -> long_answer, 5+ ->
# case_study). Measured on the class 10 bank (2026-09-22): all 76 Mathematics
# and 30 Science 5-mark questions are labelled case_study, though they are the
# papers' long answers; every 4-mark question, the real case studies included,
# is labelled long_answer; Social Science stores its outline-map items as 4
# marks (long_answer) and 20 "Read the given source" items as 5 (case_study).
# So the label alone would put a source item under "Long answer" and a map
# item under "Case based" -- or, used as a filter, report the real long
# answers and case studies as missing.
# Not "Map Skill": in the Social Science bank that phrase is the next
# section's heading run into the end of a source item's stem (an extraction
# artifact), and it made "Read the given source ... Loans from Cooperatives"
# a map question.
_MAP_RE = re.compile(
    r"\b(?:outline|political|physical)\s+(?:outline\s+)?map\b|\bon the (?:given )?"
    r"(?:outline )?map\b|\bmap of (?:india|the world)\b", re.I)
_CASE_RE = re.compile(
    r"\bread the (?:given |following |above )?(?:source|passage|case|extract|text|"
    r"paragraph|information)\b|\bcase[- ]?(?:study|based)\b|\bsource[- ]based\b|"
    r"\bbased on the (?:above |given |following )?(?:source|passage|case|extract|"
    r"information|text|paragraph)\b", re.I)
# CBSE numbers a source-based question's sub-parts "(36.1) ... (36.2)"; long
# answers use "(a) ... OR (b)". In the class 10 Social Science bank 36 of the
# 48 5-mark items carry this numbering -- many are fragments that lost their
# "Read the given source" opening -- and no Mathematics or Science item does.
_SUBPART_RE = re.compile(r"\((\d{1,2})\.1\).*\(\1\.2\)", re.S)
# Labels the bank assigns from marks, so between them marks decide, not label.
_LONG_FORM = frozenset({"long_answer", "very_long_answer", "case_study"})


def is_map_question(q: QuestionSchema) -> bool:
    # Tags are not used: in the bank they are inferred chapter names, and the
    # "Map Work" tag sits on a 1-mark History quotation question too.
    if q.type == "map_based" or q.map_asset_id:
        return True
    return bool(_MAP_RE.search(q.stem))


def is_case_question(q: QuestionSchema) -> bool:
    """Source / passage / case based, by what the stem says. Only positive
    evidence counts: many real case studies open with a plain narrative
    ("In a coffee shop, coffee is served in two types of cups..."), so the
    absence of these words is not evidence of a long answer."""
    return bool(_CASE_RE.search(q.stem) or _SUBPART_RE.search(q.stem))


def _is_competency(q: QuestionSchema) -> bool:
    from .selection import is_competency_question
    return is_competency_question(q) or is_case_question(q)


def _content_fits(q: QuestionSchema, types: list[str]) -> bool:
    """Does the question belong under a section asking for `types`?

    * A map question belongs only in a map section, and a map section takes
      only a map question -- never "any 5-mark question".
    * Between the long-form labels (long / very long answer, case study) the
      label is mark-derived, so the text decides: a question that reads as
      source / case based goes only to a case-study section; one that does
      not is taken by either, and the section's marks place it (in the CBSE
      board papers the 4-mark questions are the case studies).
    * Every other label is matched as it stands.
    """
    wanted = set(types)
    if is_map_question(q):
        return "map_based" in wanted
    wanted.discard("map_based")
    if q.type in _LONG_FORM and wanted & _LONG_FORM:
        if is_case_question(q):
            return "case_study" in wanted
        return True
    return q.type in wanted


def _fits(q: QuestionSchema, section: TemplateSection) -> bool:
    if q.marks != section.marks_each:
        return False
    return not section.question_types or _content_fits(q, section.question_types)


def _all_competency(section: TemplateSection) -> bool:
    return (section.competency_share or 0.0) >= 1.0


# Kinds a heading can name, and whether the section's constraints check it.
# Only map and case / source are checkable (see `_content_fits`); the bank
# carries no label for grammar, reading, extract, writing or literature.
_HEADING_KINDS: tuple[tuple[str, "re.Pattern[str]", Callable[[TemplateSection], bool]], ...] = (
    ("map", re.compile(r"\bmaps?\b", re.I),
     lambda s: "map_based" in s.question_types),
    ("case / source", re.compile(r"\bcase\b|\bsource\b", re.I),
     lambda s: "case_study" in s.question_types or _all_competency(s)),
    ("reading", re.compile(r"\breading\b|\bcomprehension\b|\bpassage\b", re.I),
     lambda s: False),
    ("grammar", re.compile(r"\bgrammar\b", re.I), lambda s: False),
    ("extract", re.compile(r"\bextracts?\b", re.I), lambda s: False),
    ("writing", re.compile(r"\bwriting\b", re.I), lambda s: False),
    ("literature", re.compile(r"\bliterature\b", re.I), lambda s: False),
)


def _content_notes(section: TemplateSection) -> list[str]:
    unchecked = [name for name, pattern, checked in _HEADING_KINDS
                 if pattern.search(section.title) and not checked(section)]
    if not unchecked:
        return []
    by = "marks and question type" if section.question_types else "marks"
    return [f"{section.title}: section content not checked -- the heading says "
            f"{', '.join(unchecked)}, which nothing in the bank labels, so the "
            f"section is filled by {by} alone. Read its questions before printing."]


@dataclass
class SectionPlan:
    section: TemplateSection
    picked: list[QuestionSchema]
    borrowed: list[QuestionSchema]
    availability: SectionAvailability
    notes: list[str] = field(default_factory=list)
    # Primary question id -> its OR alternative (internal choice).
    alternatives: dict[str, QuestionSchema] = field(default_factory=dict)


@dataclass
class _LeftOut:
    """Keyed, unclaimed, in-scope questions a section did not count, by the
    constraint that refused them -- what a shortfall reason has to name.

    Measured on the class 10 Social Science bank (2026-09-22): Section D was
    one long answer short while 45 5-mark source items sat unused, refused
    as source based; Section E (4 marks) had no case question because those
    same source items carry 5 marks. The old reason named neither."""
    # At the section's marks, refused for what they are, by the question
    # type that would admit them ("case_study": n -- they read as source /
    # case based). Never map questions, and nothing for a map section: a
    # map question only ever fills a map section, and nothing else can
    # answer one, so "also take them" would be a fix that breaks the paper.
    # Measured on the real bank before this: Section F (map) offered "the
    # 45 5-mark question(s) that read as source / case based", Section E
    # (case based) "the 3 4-mark question(s) that are map questions".
    wrong_kind: dict[str, int] = field(default_factory=dict)
    # Would fit this section at another mark value (marks: n).
    other_marks: dict[int, int] = field(default_factory=dict)


def _admitting_type(q: QuestionSchema) -> str:
    """The question type a section would add to take `q` (map questions
    never reach here). A long-form item that reads as source / case based
    is admitted by `case_study` whatever its mark-derived label says (see
    `_content_fits`); any other by its own label."""
    if q.type in _LONG_FORM and is_case_question(q):
        return "case_study"
    return q.type


def _kind_of(admit: str) -> str:
    if admit == "case_study":
        return "read as source / case based"
    return f"are labelled {admit.replace('_', ' ')}"


def _left_out(section: TemplateSection, pool: list[QuestionSchema]) -> _LeftOut:
    """Only for a section whose kind the engine checks (typed, or all
    competency): for an untyped "Writing" section, "127 questions exist at 1
    mark" is true and useless.

    Other mark values are counted only where the kind is read from the
    question's text (map, case / source) or is competency: a "long answer"
    label is derived from marks, so on the class 10 Social Science bank it
    offered "88 long-answer questions at 3 marks" for Section D -- the short
    answers of Section C."""
    out = _LeftOut()
    typed = bool(section.question_types)
    if not typed and not _all_competency(section):
        return out
    by_text = bool({"map_based", "case_study"} & set(section.question_types))
    marks_matter = by_text or _all_competency(section)
    map_section = "map_based" in section.question_types
    for q in pool:
        fits_kind = not typed or _content_fits(q, section.question_types)
        if q.marks == section.marks_each:
            if not fits_kind and not map_section and not is_map_question(q):
                admit = _admitting_type(q)
                out.wrong_kind[admit] = out.wrong_kind.get(admit, 0) + 1
        elif marks_matter and fits_kind and (not _all_competency(section) or _is_competency(q)):
            if by_text and not (is_map_question(q) or _is_competency(q)):
                continue  # at another mark value, only content evidence says "same kind"
            out.other_marks[q.marks] = out.other_marks.get(q.marks, 0) + 1
    return out


def plan(template: PaperTemplateDraft, template_id: str, candidates: list[QuestionSchema],
         scope: ScopeFilter, *, fill_from_outside_scope: bool = False,
         stale: frozenset[str] = frozenset(),
         ) -> tuple[AvailabilityReport, list[SectionPlan]]:
    """Walks the sections in order, each claiming its questions so a later
    section with the same mark value only counts what is left.

    Two passes: every section's compulsory questions first, then the OR
    alternatives from what all of them left. An OR is optional and a
    compulsory question is not, so an OR must never be what empties a later
    section. Pairing inside one pass did exactly that: Section A (2 x 3
    marks, 2 ORs) and Section B (2 x 3 marks) on a bank of four 3-mark
    questions printed A with two ORs and B empty -- 6 compulsory marks
    given up for two optional ones, and the report called a bank that could
    fill the paper incomplete.

    The availability report IS this selection, run dry: every section's
    `filled`, `notes` and gaps are read off the questions actually picked,
    so the check before generating and the paper after cannot disagree.

    `candidates` is the whole class+subject bank; this function applies
    every other constraint itself so it can report each one.

    `stale`: questions to print only when nothing else fits ("Make another
    like this" passes the last paper's). They rank after every fresh
    question and are never excluded, so a section is filled exactly as far
    as it would be without them -- a teacher asked for a new paper, not a
    shorter one -- and each reused question is named in a note.
    """
    keyed = [q for q in candidates if has_verified_key(q)]
    keyed_ids = {q.id for q in keyed}
    claimed: set[str] = set()
    # Every question printed so far, compulsory or OR: a later pick must not
    # repeat one of them in other words (`pool.near_duplicate`).
    printed: list[QuestionSchema] = []
    plans: list[SectionPlan] = []
    firsts: list[_FirstPass] = []

    for section in template.sections:
        fits_all = [q for q in candidates if _fits(q, section)]
        fits_keyed = [q for q in fits_all if q.id in keyed_ids and q.id not in claimed]
        eligible = [q for q in fits_keyed if scope.in_scope(q)]
        outside = [q for q in fits_keyed if not scope.in_scope(q)]
        # A section that is all competency questions (the case-based
        # sections) and has not one competency question to draw on is a gap,
        # not a section of whatever else carries those marks. A partial
        # match is picked competency-first and reported by `_mix_notes`.
        not_competency = 0
        if _all_competency(section):
            if eligible and not any(_is_competency(q) for q in eligible):
                not_competency, eligible = len(eligible), []
            if outside and not any(_is_competency(q) for q in outside):
                outside = []
        without_key = sum(1 for q in fits_all if q.id not in keyed_ids and scope.in_scope(q))
        used_elsewhere = sum(1 for q in fits_all if q.id in claimed and scope.in_scope(q))
        left_out = _left_out(section, [q for q in keyed if q.id not in claimed
                                       and scope.in_scope(q)])

        needed = section.question_count
        picked = _pick(eligible, section, needed, avoid=printed, stale=stale)
        printed.extend(picked)
        borrowed: list[QuestionSchema] = []
        notes: list[str] = []
        if len(picked) < needed and fill_from_outside_scope and outside:
            borrowed = _pick(outside, section, needed - len(picked), avoid=printed,
                             stale=stale)
            printed.extend(borrowed)
            notes.append(
                f"{section.title}: {len(borrowed)} question(s) taken from outside the "
                f"chosen chapters to fill the section ({', '.join(q.id for q in borrowed)}).")
        claimed.update(q.id for q in picked + borrowed)
        firsts.append(_FirstPass(section, fits_all, eligible, outside, picked, borrowed,
                                 notes, left_out, without_key, used_elsewhere,
                                 not_competency))

    # Second pass: the ORs, section by section, from what no section prints
    # as a compulsory question.
    primaries = [{q.id for q in f.picked + f.borrowed} for f in firsts]
    for i, f in enumerate(firsts):
        section, picked, borrowed, notes = f.section, f.picked, f.borrowed, f.notes
        eligible, outside, left_out = f.eligible, f.outside, f.left_out
        fits_all, without_key = f.fits_all, f.without_key
        used_elsewhere, not_competency = f.used_elsewhere, f.not_competency
        pool = eligible + (outside if fill_from_outside_scope else [])
        later = set().union(*primaries[i + 1:]) if i + 1 < len(firsts) else set()
        printed_later = sum(1 for q in pool if q.id in later)
        alternatives = _pair_choices(
            section, picked + borrowed,
            [q for q in eligible if q.id not in claimed],
            [q for q in outside if q.id not in claimed] if fill_from_outside_scope else [],
            avoid=printed, stale=stale)
        claimed.update(q.id for q in alternatives.values())
        printed.extend(alternatives.values())
        mix_notes, mix_fixes = _mix_notes(section, picked + borrowed, left_out)
        notes.extend(mix_notes)
        choice_notes, choice_fixes = _choice_notes(section, picked + borrowed, alternatives,
                                                   eligible_ids={q.id for q in eligible},
                                                   printed_later=printed_later)
        notes.extend(choice_notes)
        notes.extend(_content_notes(section))
        reused = [q.id for q in [*picked, *borrowed, *alternatives.values()] if q.id in stale]
        if reused:
            notes.append(
                f"{section.title}: {len(reused)} question(s) repeat your last paper "
                f"({', '.join(reused)}) -- the bank has no other question in these "
                "chapters that fits the section's marks, kind and difficulty mix.")

        filled = len(picked) + len(borrowed)
        needed = section.question_count
        shortfall = needed - filled
        reason, fixes = "", [*mix_fixes, *choice_fixes]
        if shortfall:
            reason, gap_fixes = _diagnose(
                template, section, available=len(eligible), filled=filled,
                borrowed=len(borrowed), bank=len(candidates), keyed=len(keyed),
                fits=len(fits_all), outside=len(outside), without_key=without_key,
                used_elsewhere=used_elsewhere, not_competency=not_competency,
                left_out=left_out)
            # Merged, not replaced: the mix and choice fixes stand whether or
            # not the section is short. `_diagnose` offers the same
            # change-marks fix a competency shortfall does, so one fix per
            # kind and value is kept.
            fixes = _merge_fixes(gap_fixes, fixes)
        availability = SectionAvailability(
            section_id=section.id, title=section.title, marks_each=section.marks_each,
            needed=needed, available=len(eligible), shortfall=shortfall, filled=filled,
            borrowed=len(borrowed), outside_scope_available=len(outside),
            without_key=without_key, choices_needed=min(section.choice_count, filled),
            choices_filled=len(alternatives), reason=reason, notes=notes, fixes=fixes,
        )
        plans.append(SectionPlan(section, picked, borrowed, availability, notes,
                                 alternatives))

    report = AvailabilityReport(
        template_id=template_id, grade=template.grade, subject=template.subject,
        bank_questions=len(candidates), keyed_questions=len(keyed),
        complete=all(p.availability.shortfall == 0 for p in plans),
        sections=[p.availability for p in plans], scope_notes=scope.notes,
    )
    return report, plans


@dataclass
class _FirstPass:
    """A section after its compulsory questions are picked, waiting for the
    second pass to pair its ORs (see `plan`)."""
    section: TemplateSection
    fits_all: list[QuestionSchema]
    eligible: list[QuestionSchema]
    outside: list[QuestionSchema]
    picked: list[QuestionSchema]
    borrowed: list[QuestionSchema]
    notes: list[str]
    left_out: _LeftOut
    without_key: int
    used_elsewhere: int
    not_competency: int


def _merge_fixes(first: list[SectionFix], then: list[SectionFix]) -> list[SectionFix]:
    out: list[SectionFix] = []
    seen: set[tuple] = set()
    for f in [*first, *then]:
        key = (f.kind, f.marks_each, f.question_count, f.choice_count,
               tuple(f.question_types or ()))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _pair_choices(section: TemplateSection, printed: list[QuestionSchema],
                  pool: list[QuestionSchema], outside: list[QuestionSchema], *,
                  avoid: Optional[list[QuestionSchema]] = None,
                  stale: frozenset[str] = frozenset(),
                  ) -> dict[str, QuestionSchema]:
    """An OR alternative for `choice_count` of the printed questions.

    The alternative comes from the questions this section may take (same
    marks, kind and scope), so "Q23 ... OR ..." is two questions of one
    kind. The CBSE papers pair within a chapter where they can, so a
    same-chapter alternative is preferred, and for a competency-based
    question a competency-based one. The last questions of the section carry
    the ORs, as in the SQPs (Science Section D: all three; Social Science
    Section D: all four). Outside-scope questions are used only when the
    caller allowed filling from outside the scope, and are named in a note.
    An alternative that repeats a question already on the paper (`avoid`,
    and the ORs paired here) in other words is never used."""
    want = min(section.choice_count, len(printed))
    if want <= 0:
        return {}
    on_paper = list(avoid or [])
    left = _order(pool, stale) + _order(outside, stale)
    pairs: dict[str, QuestionSchema] = {}
    for primary in printed[len(printed) - want:]:
        chapters = set(primary.chapter_ids)
        cbq = _is_competency(primary)
        # First best by (fresh, same chapter, same competency-ness), ties to
        # the better-ordered question -- `left` is already best first. Only
        # the candidates actually reached are compared with the paper.
        ranked = sorted(range(len(left)), key=lambda i: (
            left[i].id in stale, not chapters & set(left[i].chapter_ids),
            _is_competency(left[i]) != cbq, i))
        alt = next((left[i] for i in ranked if not clashes(left[i], on_paper)), None)
        if alt is None:
            break
        pairs[primary.id] = alt
        left.remove(alt)
        on_paper.append(alt)
    return pairs


def _choice_notes(section: TemplateSection, printed: list[QuestionSchema],
                  alternatives: dict[str, QuestionSchema], *, eligible_ids: set[str],
                  printed_later: int = 0) -> tuple[list[str], list[SectionFix]]:
    """`printed_later`: questions this section could have paired that later
    sections print as compulsory -- they win, and the note names them, so
    "the bank has no more" is not said of a bank that has them."""
    notes: list[str] = []
    fixes: list[SectionFix] = []
    want = min(section.choice_count, len(printed))
    borrowed = [a.id for a in alternatives.values() if a.id not in eligible_ids]
    if borrowed:
        notes.append(f"{section.title}: {len(borrowed)} OR alternative(s) taken from outside "
                     f"the chosen chapters ({', '.join(borrowed)}).")
    if len(alternatives) < want:
        why = (f"the {printed_later} other {section.marks_each}-mark question(s) this "
               "section could pair are printed as compulsory questions in a later section"
               if printed_later else
               f"the bank has no more {section.marks_each}-mark questions this section takes")
        notes.append(
            f"{section.title}: {len(alternatives)} of {want} internal choice (OR) "
            f"alternative(s) could be paired -- {why}, so the rest print without an OR.")
        fixes.append(SectionFix(
            kind="reduce_choices", choice_count=len(alternatives),
            label=(f"Offer {len(alternatives)} OR choice(s) instead of {want}."
                   if alternatives else "Print this section without OR choices.")))
    return notes, fixes


def _order(pool: list[QuestionSchema],
           stale: frozenset[str] = frozenset()) -> list[QuestionSchema]:
    """Best first, deterministic: the same bank and template give the same
    paper. `stale` questions (see `plan`) go after every other one."""
    return sorted(pool, key=lambda q: (q.id in stale, -q.quality_score, q.id))


def clashes(q: QuestionSchema, printed: list[QuestionSchema]) -> Optional[QuestionSchema]:
    """The question on the paper that `q` repeats in other words, if any.

    Near-identical stems never share a paper (Task 903): the served bank
    holds the same MCQ from two board papers under two ids, which the pool's
    own dedupe lets through (see `pool.near_duplicate`)."""
    for other in printed:
        if other.id != q.id and near_duplicate(q.stem, other.stem):
            return other
    return None


def _pick(pool: list[QuestionSchema], section: TemplateSection, n: int, *,
          avoid: Optional[list[QuestionSchema]] = None,
          stale: frozenset[str] = frozenset()) -> list[QuestionSchema]:
    """Up to `n` questions honouring the section's difficulty mix and
    competency share as far as the pool allows. When a difficulty runs out,
    the slot goes to the nearest other difficulty rather than staying empty:
    a slightly easier paper is a better outcome than a shorter one, and the
    mismatch is reported by `_mix_notes`. A question repeating one already
    printed (`avoid`) or picked here is skipped (`clashes`)."""
    if n <= 0 or not pool:
        return []

    printed = list(avoid or [])
    remaining = _order(pool, stale)
    targets = _difficulty_targets(section.difficulty_mix, n)
    cbq_target = round((section.competency_share or 0.0) * n)
    picked: list[QuestionSchema] = []
    repeats: set[str] = set()      # clash once, clash for good: `printed` only grows

    def usable(q: QuestionSchema) -> bool:
        if q.id in repeats:
            return False
        if clashes(q, printed):
            repeats.add(q.id)
            return False
        return True

    def take(from_list: list[QuestionSchema]) -> Optional[QuestionSchema]:
        want_cbq = sum(1 for q in picked if _is_competency(q)) < cbq_target
        if want_cbq:
            for q in from_list:
                if _is_competency(q) and usable(q):
                    return q
        return next((q for q in from_list if usable(q)), None)

    for difficulty, count in targets:
        for _ in range(count):
            q = take([c for c in remaining if difficulty is None or c.difficulty == difficulty])
            if q is None:
                break
            picked.append(q)
            printed.append(q)
            remaining.remove(q)
    while len(picked) < n and remaining:
        q = take(remaining)
        if q is None:
            break
        picked.append(q)
        printed.append(q)
        remaining.remove(q)
    return picked


def _difficulty_targets(mix: Optional[DifficultyDistribution],
                        n: int) -> list[tuple[Optional[str], int]]:
    """Largest-remainder split of `n` over the mix, so the counts always sum to n."""
    if mix is None:
        return [(None, n)]
    shares = [("easy", mix.easy), ("medium", mix.medium), ("hard", mix.hard)]
    raw = [(d, share * n) for d, share in shares]
    counts = {d: int(v) for d, v in raw}
    left = n - sum(counts.values())
    for d, v in sorted(raw, key=lambda dv: dv[1] - int(dv[1]), reverse=True)[:left]:
        counts[d] += 1
    return [(d, counts[d]) for d, _ in shares]


def _kind_noun(section: TemplateSection) -> str:
    types = set(section.question_types)
    if "map_based" in types:
        return "map"
    if "case_study" in types:
        return "case / competency-based"
    if types & _LONG_FORM:
        return "long-answer"
    if _all_competency(section):
        return "competency-based"
    return "matching"


def _best_other_marks(left_out: _LeftOut) -> Optional[tuple[int, int]]:
    """(marks, count) of the mark value holding the most questions this
    section would take -- the one "change the marks" fix worth offering."""
    if not left_out.other_marks:
        return None
    marks, n = max(left_out.other_marks.items(), key=lambda mn: (mn[1], -mn[0]))
    return marks, n


def _change_marks_fix(section: TemplateSection, left_out: _LeftOut) -> Optional[SectionFix]:
    best = _best_other_marks(left_out)
    if best is None:
        return None
    marks, n = best
    return SectionFix(
        kind="change_marks", marks_each=marks, available=n,
        label=(f"Make it a {marks}-mark section: {n} {_kind_noun(section)} question(s) "
               f"are in the bank at {marks} marks (then rebalance the paper's total)."))


def _mix_notes(section: TemplateSection, picked: list[QuestionSchema],
               left_out: _LeftOut) -> tuple[list[str], list[SectionFix]]:
    notes: list[str] = []
    fixes: list[SectionFix] = []
    if section.difficulty_mix is not None and picked:
        wanted = dict(_difficulty_targets(section.difficulty_mix, len(picked)))
        got = {d: sum(1 for q in picked if q.difficulty == d) for d in wanted}
        if got != wanted:
            notes.append(
                f"{section.title}: difficulty is easy/medium/hard "
                f"{got['easy']}/{got['medium']}/{got['hard']}, the template asked for "
                f"{wanted['easy']}/{wanted['medium']}/{wanted['hard']} -- the bank has "
                "too few at the requested level.")
    if section.competency_share and picked:
        target = round(section.competency_share * len(picked))
        got_cbq = sum(1 for q in picked if _is_competency(q))
        if got_cbq < target:
            note = (f"{section.title}: {got_cbq} competency-based question(s), "
                    f"the template asked for {target}.")
            # For an all-competency section `left_out.other_marks` counts
            # competency questions only, so this names where they are.
            fix = _change_marks_fix(section, left_out) if _all_competency(section) else None
            if fix is not None:
                note += (f" {fix.available} competency-based question(s) are in the bank "
                         f"at {fix.marks_each} marks.")
                fixes.append(fix)
            notes.append(note)
    return notes, fixes


def _diagnose(template: PaperTemplateDraft, section: TemplateSection, *, available: int,
              filled: int, borrowed: int, bank: int, keyed: int, fits: int, outside: int,
              without_key: int, used_elsewhere: int, not_competency: int,
              left_out: _LeftOut) -> tuple[str, list[SectionFix]]:
    """The shortfall, the constraints that caused it -- largest first, so the
    binding one leads -- and the changes that would close it."""
    who = f"Class {template.grade} {template.subject}"
    marks = section.marks_each
    kind = f"{marks}-mark"
    if section.question_types:
        kind += " " + "/".join(t.replace("_", " ") for t in section.question_types)
    needed = section.question_count
    shortfall = needed - filled
    if bank == 0:
        return f"The bank has no {who} questions yet.", []
    if keyed == 0:
        return (f"The bank has {bank} {who} questions but none with a verified "
                "answer key yet, so none can be printed."), []

    if fits == 0:
        lead = f"The bank has no {kind} {who} questions."
    elif not_competency:
        lead = (f"None of the {not_competency} {kind} {who} question(s) is "
                "competency-based, and this section asks for competency-based "
                "questions only.")
    elif borrowed:
        lead = (f"Only {filled} {kind} question(s) are available, {borrowed} of them from "
                f"outside the chosen chapters; {shortfall} short.")
    else:
        lead = f"Only {available} {kind} question(s) are available; {shortfall} short."

    because: list[tuple[int, str]] = []
    for admit, n in left_out.wrong_kind.items():
        because.append((n, f"{n} more {marks}-mark question(s) were left out because they "
                           f"{_kind_of(admit)}, which this section does not take."))
    change = _change_marks_fix(section, left_out)
    if change is not None:
        because.append((change.available, (
            f"{change.available} {_kind_noun(section)} question(s) are in the bank at "
            f"{change.marks_each} marks, but this section asks for {marks}-mark ones -- "
            f"change the section to {change.marks_each} marks to use them.")))
    if outside and not borrowed:
        because.append((outside, f"{outside} more are in other chapters -- widen the "
                                 "chapters or allow filling from outside them."))
    if without_key:
        because.append((without_key, f"{without_key} more have no verified answer key yet."))
    if used_elsewhere:
        because.append((used_elsewhere, f"{used_elsewhere} are already used by earlier "
                                        "sections."))
    because.sort(key=lambda nt: -nt[0])          # stable: ties keep the order above
    reason = " ".join([lead] + [text for _, text in because])

    fixes: list[SectionFix] = []
    if outside and not borrowed:
        fixes.append(SectionFix(kind="widen_scope", available=outside,
                                label=f"Add the chapters that hold {outside} more such question(s)."))
        fixes.append(SectionFix(kind="fill_from_outside_scope", available=outside,
                                label="Fill the gap from outside the chosen chapters "
                                      "(the borrowed questions are listed)."))
    if change is not None:
        fixes.append(change)
    # One fix per kind, each carrying the types to set, so applying it is a
    # tap, not a sentence to parse. Map questions are never offered (see
    # `_LeftOut.wrong_kind`).
    for admit, n in sorted(left_out.wrong_kind.items(), key=lambda wn: -wn[1]):
        fixes.append(SectionFix(
            kind="relax_type", available=n,
            question_types=[*section.question_types, admit],
            label=f"Also take the {n} {marks}-mark question(s) that {_kind_of(admit)}."))
    if not_competency:
        fixes.append(SectionFix(kind="relax_competency", available=not_competency,
                                label=f"Drop the competency-only rule and take the "
                                      f"{not_competency} ordinary {marks}-mark question(s)."))
    fixes.append(SectionFix(
        kind="reduce_count", question_count=filled,
        label=(f"Ask for {filled} question(s) instead of {needed} (then rebalance the "
               "paper's total)." if filled else
               "Remove this section (then rebalance the paper's total).")))
    return reason, fixes
