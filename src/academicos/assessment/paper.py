"""Paper Generator: assembles selected questions into the school's section
format (Section A/B/C/D/E...), producing the structured GeneratedPaper the
Flutter review screen renders and the PDF exporter formats.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .wording import counted
from .competency import CBSE_COMPETENCY_TARGET, below_target, share_summary
from .duration import format_duration
from .schemas import (
    Blueprint,
    GeneratedPaper,
    GeneratedQuestionSchema,
    GeneratedSectionSchema,
    PaperMetadataSchema,
    QuestionSchema,
)
from .templates import default_sections


_SECTION_LETTER = re.compile(r"^(section)\s+[A-Z]\b", re.I)


def lettered(name: str, label: str) -> str:
    """`name` with a leading "Section X" saying the letter the paper prints.

    A template's section titles carry their own letter ("Section C - Short
    answer"), while the paper letters the sections it prints in order, so a
    section the bank left empty moved every later heading up a letter and
    the name kept the old one: "SECTION B" over "(Section C - Short answer
    ...)" on 4 of 131 gap-paper headings (audit D44), and the instructions
    naming a Section C the paper did not print. A name with no leading
    section letter is returned as it is."""
    return _SECTION_LETTER.sub(lambda m: f"{m.group(1)} {label}", name, count=1)


def reletter(sections: list[GeneratedSectionSchema]) -> list[GeneratedSectionSchema]:
    """Sections lettered A, B, C... in order, their names to match, when
    they carry that lettering with a letter missing -- what removing a
    section's last question leaves. Sections labelled any other way ("I",
    "II", a custom blueprint's own labels) are returned as they are: there
    is no order to restore."""
    letters = [s.label for s in sections]
    if not (all(len(x) == 1 and "A" <= x <= "Z" for x in letters)
            and letters == sorted(set(letters))):
        return sections
    return [s.model_copy(update={"label": chr(ord("A") + i),
                                 "name": lettered(s.name, chr(ord("A") + i))})
            for i, s in enumerate(sections)]


def _key_note(scheme, marks: int) -> str:
    """What the printed key must say about the value points above it.

    Two things a marker cannot see from the points alone:

      * `anyOf` -- the scheme lists more value points than it awards ("Any
        other, Any two"), so a student who gives two of the five listed gets
        full marks.
      * one point worth every mark -- the source printed no split, so the key
        really is all or nothing. Saying so is the point: a marking sheet that
        shows "- <the whole answer> [5m]" and nothing else invites a marker to
        assume steps exist and invent them.

    Not said of an option key. An MCQ worth 2 marks has one point because the
    student picked the right letter or did not, and "unless you set your own
    value points" is advice a marker cannot take on it -- there is nothing to
    divide. One served record is an objective scheme worth more than one mark
    (cbe:q:Maths9SM1).
    """
    meta = getattr(scheme, "metadata", None) or {}
    points = getattr(scheme, "marking_points", None) or []
    any_of = meta.get("anyOf")
    if any_of and points:
        return counted(f"\n(Any {int(any_of)} of the above value points, "
                       f"{points[0].marks} mark(s) each.)")
    if meta.get("objective"):
        return ""
    if len(points) == 1 and (marks or 0) > 1:
        return "\n" + ALL_OR_NOTHING_NOTE
    return ""


ALL_OR_NOTHING_NOTE = ("(The source prints no split for this answer: it is all or nothing "
                       "unless you set your own value points.)")


def generate_paper(*, paper_id: str, assessment_id: str, assessment_title: str,
                    subject: str, grade: int, blueprint: Blueprint,
                    selected_questions: list[QuestionSchema],
                    set_label: str | None = None) -> GeneratedPaper:
    sections_bp = blueprint.sections or default_sections(blueprint.total_marks)
    by_marks: dict[int, list[QuestionSchema]] = {}
    for q in selected_questions:
        by_marks.setdefault(q.marks, []).append(q)

    gen_sections: list[GeneratedSectionSchema] = []
    display_no = 1
    answer_key: dict[str, str] = {}

    for section in sections_bp:
        bucket = by_marks.get(section.marks_per_question, [])
        take = bucket[: section.question_count]
        del bucket[: section.question_count]

        gen_questions: list[GeneratedQuestionSchema] = []
        for q in take:
            choice_id = q.metadata.get("internal_choice_id")
            gen_questions.append(generated_question(
                q, display_no, choice_stem=q.metadata.get("internal_choice_stem"),
                choice_id=choice_id))
            answer_key[str(display_no)] = answer_key_entry(q)
            if choice_id:
                answer_key[f"{display_no}_OR"] = or_answer_key_entry(
                    q.metadata.get("internal_choice_scheme"))
            display_no += 1

        # A section with a choice ("attempt any 10 of 12") is worth the marks
        # a student can score, which the blueprint states; the printed
        # questions' sum would overstate it and make the sections add up to
        # more than Maximum Marks. min() keeps an under-filled section (and
        # the CBSE OR-choice sections, whose total is count x marks) at what
        # is actually printed.
        printed = sum(q.marks for q in gen_questions)
        gen_sections.append(GeneratedSectionSchema(
            section_id=section.id,
            label=section.label,
            name=section.name,
            questions=gen_questions,
            total_marks=min(printed, section.total_marks) if section.has_internal_choice
            else printed,
        ))

    # Maximum Marks is what the paper holds, never more than was asked for.
    # A bank too thin for a section leaves it short: the live release printed
    # "Maximum Marks: 80" on an English 10 paper holding 20 (re-audit,
    # 2026-09-22), and a student cannot score marks that were never printed.
    # min() keeps a template whose sections add up to more than its stated
    # total at that total.
    total_marks = min(blueprint.total_marks, sum(s.total_marks for s in gen_sections))
    formatted = _render_text(assessment_title, subject, grade, blueprint.duration_minutes,
                             total_marks, gen_sections, set_label)

    return GeneratedPaper(
        id=paper_id,
        assessment_id=assessment_id,
        sections=gen_sections,
        formatted_content=formatted,
        answer_key=answer_key,
        metadata=PaperMetadataSchema(
            assessment_title=assessment_title,
            subject=subject,
            grade=grade,
            total_marks=total_marks,
            duration_minutes=blueprint.duration_minutes,
            generated_at=datetime.now(timezone.utc),
            # The report card names a paper by it; it stayed None, so a
            # Half-yearly printed as "Test" (QA P-21).
            exam_type=blueprint.exam_type,
        ),
        set_label=set_label,
    )


def generate_paper_sets(*, paper_id: str, assessment_id: str, assessment_title: str,
                        subject: str, grade: int, blueprint: Blueprint,
                        selected_questions: list[QuestionSchema],
                        set_count: int = 1,
                        rotation_groups: list[list[QuestionSchema]] | None = None,
                        alternatives_pool: list[QuestionSchema] | None = None,
                        set_questions: list[list[QuestionSchema]] | None = None,
                        ) -> GeneratedPaper:
    """Generate parallel equivalent question paper sets (Sets A, B, C...) with strictly invariant difficulty.

    With `set_questions` -- one list per set, set A's first (it is
    `selected_questions`), each in section order and filling `blueprint`'s
    sections exactly as set A does -- every set prints its own list: the
    caller chose each set's questions by the sections' own rules (the
    builder, `paper_template_routes._parallel_sets`). `set_overlap` says how
    many printed questions, OR alternatives included, each shares with an
    earlier set.

    With `alternatives_pool`, set B onwards print different questions of the
    same marks and type (selection.alternative_sets), and `set_overlap` says
    how many each shares with an earlier set. That is the first choice: real
    alternatives beat any reordering.

    Without a pool there is nothing to draw alternatives from, and later sets
    are set A rotated within a group. By default a group is every question of
    one mark value, which is right when each mark value is one section. A
    teacher's template can have several sections at the same mark (reading,
    grammar, extract), each with its own type and difficulty rules; rotating
    across all of them moved questions into the wrong section in set B.
    `rotation_groups` -- one list per section, in section order, together
    exactly `selected_questions` -- keeps each question in its own section:
    `generate_paper` takes same-mark sections in order, so concatenating the
    rotated groups in order refills each section from its own group."""
    if set_count <= 1:
        return generate_paper(
            paper_id=paper_id,
            assessment_id=assessment_id,
            assessment_title=assessment_title,
            subject=subject,
            grade=grade,
            blueprint=blueprint,
            selected_questions=selected_questions,
            set_label=None,
        )

    labels = ["A", "B", "C", "D", "E"][:set_count]
    if len(labels) < set_count:
        labels = [f"Set {i + 1}" for i in range(set_count)]

    paper_sets: list[GeneratedPaper] = []
    alternatives: list[list[QuestionSchema]] | None = None
    overlaps: list[int] = []
    if set_questions is not None:
        if len(set_questions) != set_count:
            raise ValueError(f"{len(set_questions)} question lists for {set_count} sets")
        alternatives, overlaps = set_questions, _shared_with_earlier(set_questions)
    elif alternatives_pool is not None:
        from .selection import alternative_sets
        from .templates import default_sections
        sections = blueprint.sections or default_sections(blueprint.total_marks)
        alternatives, overlaps = alternative_sets(selected_questions, alternatives_pool, set_count,
                                                  sections)

    for idx, label in enumerate(labels):
        variant_questions: list[QuestionSchema] = []
        groups: list[list[QuestionSchema]]
        if alternatives is not None:
            # Each set already holds its own questions: nothing to rotate.
            groups = []
            variant_questions = [q.model_copy(deep=True) for q in alternatives[idx]]
        elif rotation_groups is not None:
            groups = rotation_groups
        else:
            by_marks: dict[int, list[QuestionSchema]] = {}
            for q in selected_questions:
                by_marks.setdefault(q.marks, []).append(q)
            groups = list(by_marks.values())

        for q_list in groups:
            curr_list = [q.model_copy(deep=True) for q in q_list]
            if idx > 0:
                shift = idx % max(1, len(curr_list))
                if shift == 0 and len(curr_list) > 1:
                    shift = 1
                curr_list = curr_list[shift:] + curr_list[:shift]

                # Swap primary and OR questions on alternate sets for sections with internal choice
                if idx % 2 == 1:
                    curr_list = [_alternative_first(q) for q in curr_list]

            variant_questions.extend(curr_list)

        set_paper_id = f"{paper_id}_set_{label}" if idx > 0 else paper_id
        set_paper = generate_paper(
            paper_id=set_paper_id,
            assessment_id=assessment_id,
            assessment_title=assessment_title,
            subject=subject,
            grade=grade,
            blueprint=blueprint,
            selected_questions=variant_questions,
            set_label=label,
        )
        paper_sets.append(set_paper)

    primary_paper = paper_sets[0].model_copy(deep=True)
    primary_paper.set_label = labels[0]
    primary_paper.sets = [s.model_copy(deep=True, update={"sets": []}) for s in paper_sets]
    if alternatives is not None:
        primary_paper.set_overlap = dict(zip(labels, overlaps))
    return primary_paper


def set_repeats(paper: GeneratedPaper) -> dict[str, list[str]]:
    """Per set of `paper` (A first), the printed slots -- "Q7", "Q12 (OR)"
    -- whose question an earlier set also prints."""
    seen: set[str] = set()
    out: dict[str, list[str]] = {}
    for s in paper.sets:
        slots: list[str] = []
        ids: set[str] = set()
        for section in s.sections:
            for q in section.questions:
                if q.question_id in seen:
                    slots.append(f"Q{q.display_number}")
                ids.add(q.question_id)
                alt = q.internal_choice_question_id
                if alt:
                    if alt in seen:
                        slots.append(f"Q{q.display_number} (OR)")
                    ids.add(alt)
        out[s.set_label or ""] = slots
        seen |= ids
    return out


_SET_REPEAT_LINE = re.compile(r"^Set [A-Z] repeats \d+ question")


def set_repeat_warnings(paper: GeneratedPaper) -> list[str]:
    """One line per set that prints a question an earlier set prints: how
    many and where. A later set repeats one only where the bank has nothing
    else its section's rules and the paper's chapters allow."""
    return [counted(f"Set {label} repeats {len(slots)} question(s) from an earlier set "
            f"({', '.join(slots)}): no other question in the bank fits "
            f"{'that slot' if len(slots) == 1 else 'those slots'} under the section's rules "
            "and the paper's chapters.")
            for label, slots in set_repeats(paper).items() if slots]


def restate_set_repeats(paper: GeneratedPaper) -> GeneratedPaper:
    """`set_overlap` and its warning lines recounted from the sets as they
    now stand, after an edit, on a paper that counts them (a non-empty
    `set_overlap`). A swap can take a repeat off a set, or a pick put one on."""
    if not paper.set_overlap or not paper.sets:
        return paper
    paper.set_overlap = {label: len(slots) for label, slots in set_repeats(paper).items()}
    paper.warnings = [w for w in paper.warnings if not _SET_REPEAT_LINE.match(w)] \
        + set_repeat_warnings(paper)
    return paper


def _shared_with_earlier(sets: list[list[QuestionSchema]]) -> list[int]:
    """Per set, its printed questions (OR alternatives included) that an
    earlier set also prints -- `GeneratedPaper.set_overlap`'s count."""
    seen: set[str] = set()
    out: list[int] = []
    for qs in sets:
        ids = [q.id for q in qs] + [q.metadata["internal_choice_id"] for q in qs
                                    if q.metadata.get("internal_choice_id")]
        out.append(sum(1 for i in ids if i in seen))
        seen.update(ids)
    return out


def _alternative_first(q: QuestionSchema) -> QuestionSchema:
    """`q`'s OR alternative as the printed question, with `q` as its OR --
    what sets B and D print for a question with internal choice.

    The alternative is rebuilt from the full record the pairing stored
    (metadata "internal_choice_question": selection.optimize and
    selection._attach_choice on the quick/custom path,
    paper_template_routes._attach_choices on the template path), so the set
    prints the alternative's own id, type, bank id, marks, marking points and
    competency flag. Moving only its stem, id and model answer onto `q`'s
    record printed an Exemplar short answer as the CBE long answer it was
    paired with, flagged competency-based, and keyed it with the primary's
    value points (`answer_key_entry` reads the record's marking points).

    Without the stored record (a primary paired before it was kept) the
    stem, id and model answer are all there is to swap."""
    from .selection import _CHOICE_KEYS

    meta = q.metadata
    if not (meta.get("internal_choice_id") and meta.get("internal_choice_stem")):
        return q
    record = meta.get("internal_choice_question")
    if record:
        alt = QuestionSchema.model_validate(record)
        primary = q.model_copy(deep=True)
        for key in _CHOICE_KEYS:
            primary.metadata.pop(key, None)
            alt.metadata.pop(key, None)
        alt.metadata["internal_choice_id"] = primary.id
        alt.metadata["internal_choice_stem"] = primary.stem
        alt.metadata["internal_choice_scheme"] = primary.answer_scheme.model_answer
        alt.metadata["internal_choice_question"] = primary.model_dump(by_alias=False)
        return alt
    old_stem, old_id, old_scheme = q.stem, q.id, q.answer_scheme.model_answer
    q.stem = meta["internal_choice_stem"]
    q.id = meta["internal_choice_id"]
    q.answer_scheme.model_answer = meta.get("internal_choice_scheme", "")
    meta["internal_choice_stem"] = old_stem
    meta["internal_choice_id"] = old_id
    meta["internal_choice_scheme"] = old_scheme
    return q


def generated_question(q: QuestionSchema, display_number: int, *,
                       choice_stem: str | None = None,
                       choice_id: str | None = None) -> GeneratedQuestionSchema:
    """One printed question. Shared by generation and by a swap or pick
    (paper_edit.py), so a question put on the paper later prints exactly as
    one chosen at generation would."""
    from .selection import is_competency_question

    return GeneratedQuestionSchema(
        question_id=q.id,
        display_number=display_number,
        stem=q.stem,
        marks=q.marks,
        bloom_level=q.bloom_level,
        difficulty=q.difficulty,
        internal_choice_text=choice_stem,
        internal_choice_question_id=choice_id,
        is_competency=is_competency_question(q),
        type=q.type,
    )


def answer_key_entry(q: QuestionSchema) -> str:
    """The answer-key text for a printed question: model answer, its marking
    points, and what the source said about splitting them.

    The note belongs to THIS helper, not to its callers: a question put on a
    paper by a swap or a pick prints through here too, so a 5-mark answer
    that says "all or nothing" when generated cannot lose that sentence
    because a teacher swapped it in."""
    model_ans = q.answer_scheme.model_answer or "(model answer pending)"
    points = q.answer_scheme.marking_points or []
    if len(points) == 1:
        # One value point that carries the answer ("Correct option C: three
        # decimal places" under "three decimal places") printed the answer
        # twice on every such row of the key (E2E run, 2026-10-01): say it once,
        # the longer wording, with its marks.
        norm = lambda s: " ".join((s or "").split()).casefold()  # noqa: E731
        a, d = norm(q.answer_scheme.model_answer), norm(points[0].description)
        if a and d and (a in d or d in a):
            once = points[0].description if len(d) >= len(a) else q.answer_scheme.model_answer
            return (f"{once} [{points[0].marks}m]" + _key_note(q.answer_scheme, q.marks)).strip()
    if q.answer_scheme.marking_points:
        pts = [f"• {mp.description} [{mp.marks}m]" for mp in q.answer_scheme.marking_points]
        model_ans = f"{model_ans}\n" + "\n".join(pts)
    return (model_ans + _key_note(q.answer_scheme, q.marks)).strip()


def or_answer_key_entry(model_answer: str | None) -> str:
    """The answer-key text for an OR alternative (its `<n>_OR` entry)."""
    return model_answer or "(model answer pending)"


def competency_counts(paper: GeneratedPaper) -> tuple[int, int]:
    """(competency-based, printed) over the questions `paper` prints: its
    compulsory questions, each flagged by the one rule when it was put on the
    paper (`generated_question`). OR alternatives are not counted."""
    flags = [q.is_competency for s in paper.sections for q in s.questions]
    return sum(flags), len(flags)


def competency_share(paper: GeneratedPaper) -> float:
    """The competency-based share of the questions `paper` prints
    (`competency_counts`)."""
    c, n = competency_counts(paper)
    return round(c / n, 3) if n else 0.0


def stamp_competency(paper: GeneratedPaper, target: float | None = None) -> None:
    """`competency_share` and whether it meets `target` (CBSE's 50% when
    None), on `paper` alone -- not its sets."""
    target = CBSE_COMPETENCY_TARGET if target is None else target
    paper.competency_share = competency_share(paper)
    paper.competency_target_met = paper.competency_share >= target


def report_competency(paper: GeneratedPaper, target: float | None, *,
                      stated: bool = False) -> list[str]:
    """The CBQ share of the printed questions and whether it meets `target`
    (CBSE: at least 50%), stamped on the paper and each of its sets. Missed in
    17/25 subject/grade pairs at the 2026-09-21 audit, with nothing on the
    paper to say so.

    Set B/C print other questions than set A, so each set gets its own share:
    reporting set A's for all three said 0.45 where B and C held 0.25 and 0.15.
    Returns a warning per set that misses the target.

    Every generation path calls this one function -- quick-generate,
    /papers/generate, generate-from-ids and generate-from-template -- so a
    template paper carries the share the builder's availability check showed
    (`paper_templates.competency_check`, same denominator).

    Returns the paper's own line when it misses the target, then one per
    other set that does. Set A is the paper itself (it carries the paper's
    id), so it gets no line of its own: it had one, rounded another way than
    the paper's ("11 of 37 questions (30%)" beside "Set A has 29.7%", audit
    D79). `stated` is for a caller that has already written the paper's line
    -- the builder's check, which also says why, or selection's -- so it is
    not written twice. Every line is `competency.share_summary`'s.
    """
    target = CBSE_COMPETENCY_TARGET if target is None else target
    warnings: list[str] = []
    for p in [paper, *paper.sets]:
        stamp_competency(p, target)
        if p.competency_target_met:
            continue
        c, n = competency_counts(p)
        if p is paper:
            if not stated:
                warnings.append(f"{share_summary(c, n)}. {below_target(target)}")
        elif p.id != paper.id:
            warnings.append(f"{share_summary(c, n, p.set_label)}. {below_target(target)}")
    return warnings


def render_text(paper: GeneratedPaper) -> str:
    """`formatted_content` for a paper as it now stands: after a swap or pick
    the stored text must say what the sections say."""
    m = paper.metadata
    return _render_text(m.assessment_title, m.subject, m.grade, m.duration_minutes,
                        m.total_marks, paper.sections, paper.set_label)


def _render_text(title: str, subject: str, grade: int, duration_minutes: int,
                 total_marks: int, sections: list[GeneratedSectionSchema],
                 set_label: str | None = None) -> str:
    header_title = f"{title} (SET {set_label})" if set_label else title
    lines = [
        header_title, f"Subject: {subject}    Grade: {grade}",
        f"Time Allowed: {format_duration(duration_minutes)}    Maximum Marks: {total_marks}",
        "General Instructions: This paper is machine-generated by AcademicOS and pending teacher review.",
        "",
    ]
    for section in sections:
        lines.append(f"SECTION {section.label} — {section.name} ({section.total_marks} marks)")
        for gq in section.questions:
            lines.append(f"{gq.display_number}. {gq.stem}  [{gq.marks}]")
            if gq.internal_choice_text:
                lines.append(f"   [OR] {gq.internal_choice_text}")
        lines.append("")
    return "\n".join(lines)
