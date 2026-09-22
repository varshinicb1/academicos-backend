"""Paper Generator: assembles selected questions into the school's section
format (Section A/B/C/D/E...), producing the structured GeneratedPaper the
Flutter review screen renders and the PDF exporter formats.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .schemas import (
    Blueprint,
    GeneratedPaper,
    GeneratedQuestionSchema,
    GeneratedSectionSchema,
    PaperMetadataSchema,
    QuestionSchema,
)
from .templates import default_sections


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
        ),
        set_label=set_label,
    )


def generate_paper_sets(*, paper_id: str, assessment_id: str, assessment_title: str,
                        subject: str, grade: int, blueprint: Blueprint,
                        selected_questions: list[QuestionSchema],
                        set_count: int = 1,
                        rotation_groups: list[list[QuestionSchema]] | None = None,
                        ) -> GeneratedPaper:
    """Generate parallel equivalent question paper sets (Sets A, B, C...) with strictly invariant difficulty.

    Each later set rotates questions within a group. By default a group is
    every question of one mark value, which is right when each mark value is
    one section. A teacher's template can have several sections at the same
    mark (reading, grammar, extract), each with its own type and difficulty
    rules; rotating across all of them moved questions into the wrong
    section in set B. `rotation_groups` -- one list per section, in section
    order, together exactly `selected_questions` -- keeps each question in
    its own section: `generate_paper` takes same-mark sections in order, so
    concatenating the rotated groups in order refills each section from its
    own group."""
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

    for idx, label in enumerate(labels):
        variant_questions: list[QuestionSchema] = []
        if rotation_groups is not None:
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
                    for q in curr_list:
                        if q.metadata.get("internal_choice_id") and q.metadata.get("internal_choice_stem"):
                            old_stem = q.stem
                            old_id = q.id
                            old_scheme = q.answer_scheme.model_answer
                            q.stem = q.metadata["internal_choice_stem"]
                            q.id = q.metadata["internal_choice_id"]
                            q.answer_scheme.model_answer = q.metadata.get("internal_choice_scheme", "")
                            q.metadata["internal_choice_stem"] = old_stem
                            q.metadata["internal_choice_id"] = old_id
                            q.metadata["internal_choice_scheme"] = old_scheme

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
    return primary_paper


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
    """The answer-key text for a printed question: model answer, then its
    marking points."""
    model_ans = q.answer_scheme.model_answer or "(model answer pending)"
    if q.answer_scheme.marking_points:
        pts = [f"• {mp.description} [{mp.marks}m]" for mp in q.answer_scheme.marking_points]
        model_ans = f"{model_ans}\n" + "\n".join(pts)
    return model_ans.strip()


def or_answer_key_entry(model_answer: str | None) -> str:
    """The answer-key text for an OR alternative (its `<n>_OR` entry)."""
    return model_answer or "(model answer pending)"


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
        f"Time Allowed: {duration_minutes} minutes    Maximum Marks: {total_marks}",
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
