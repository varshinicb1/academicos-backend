"""Seeds the real, official CBSE curriculum data for any grade 6-12
(syllabus/cbse_syllabus.py, academicos-data/syllabus/*.json --
hand-verified against the official 2025-26 CBSE curriculum PDFs) into the
curriculum/ data model as the first real Board -> AcademicYear -> Grade ->
Subject -> Book -> Unit -> Chapter rows.

(The module name predates the any-grade path; Class 10 is one grade it
seeds, not the only one.)

The source CBSE documents don't break down below chapter level, so nothing
here invents Topic/Subtopic content. What it does do, since 2026-09-22, is
apply the committed decomposition TEMPLATES (decomposition_templates.py) to
the chapters it just created, as pending proposals a principal approves --
real content from a real source, never written straight into this school's
approved curriculum. A chapter whose subject/grade has no template stays
chapter-only and is reported in the result (SubjectTemplateReport.note),
not quietly left looking finished.

Seeds any grade 6-12, not only 10: every grade's syllabus JSON has existed
in academicos-data/syllabus/ all along, but until 2026-09-22 only Class 10
had a route, so PRD section 0 decision 4 ("all grades 6-12") was
unreachable through the product.

Idempotent: safe to run more than once against the same store -- reuses
existing AcademicYear/Grade/Subject/Book rows by their natural key, and
Unit/Chapter rows by canonical_id (which has a UNIQUE constraint), rather
than erroring or creating duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

from ..syllabus.cbse_syllabus import _FILENAME_BY_SUBJECT, SyllabusUnit, _slug, load_syllabus
from . import decomposition_templates as templates
from .store import CurriculumStore

CBSE_BOARD_CODE = "CBSE"
GRADE_10 = 10
# Requirements v3: classes 1-12. Classes 1-5 have no CBSE marks table; their
# syllabus is the NCERT book's chapters (cbse_syllabus.book_chapters). Outside
# 1-12 the route answers 422 rather than creating an empty grade row.
MIN_GRADE = 1
MAX_GRADE = 12


@dataclass
class SubjectTemplateReport:
    """Per subject: how far below chapter level this seed actually got, and --
    when it got nowhere -- the measured reason. A grade with no template source
    stays chapter-only and is REPORTED; nothing is invented to fill it."""
    subject: str
    chapters: int
    chapters_with_topics: int
    chapters_without_topics: int
    topics_proposed: int
    subtopics_proposed: int
    provenance: list[str]
    note: Optional[str] = None


@dataclass
class SeedResult:
    board_id: str
    academic_year_id: str
    grade_id: str
    grade_number: int
    subjects_seeded: int
    units_seeded: int
    chapters_seeded: int
    subjects_skipped: list[str]   # subject names with no real syllabus JSON
    topic_templates: list[SubjectTemplateReport] = dataclass_field(default_factory=list)

    @property
    def chapters_with_topics(self) -> int:
        return sum(t.chapters_with_topics for t in self.topic_templates)

    @property
    def chapters_without_topics(self) -> int:
        return sum(t.chapters_without_topics for t in self.topic_templates)

    @property
    def topics_proposed(self) -> int:
        return sum(t.topics_proposed for t in self.topic_templates)

    @property
    def subtopics_proposed(self) -> int:
        return sum(t.subtopics_proposed for t in self.topic_templates)


def _canonical_prefix(book_id: str) -> str:
    # Book-scoped, not (board, grade, subject)-scoped: §6 makes Book the
    # direct parent of Chapters because two schools can select genuinely
    # different books for the same subject with different chapter
    # breakdowns. A global (board, grade, subject) prefix looked appealing
    # (dedupe identical CBSE content across schools) but a real test
    # caught the actual bug it causes: a second school's seed run found
    # the first school's units "already existing" by canonical_id and
    # reused them -- silently leaving the second school's own Book with
    # zero chapters, since none pointed at its book_id. Each book owns
    # its own real rows now, even when the content (names, marks) happens
    # to be identical to another school's book for the same official
    # curriculum.
    return book_id


def seed_cbse_grade(store: CurriculumStore, *, school_id: str,
                    academic_year_label: str, start_date: str, end_date: str,
                    grade_number: int,
                    apply_templates: bool = True) -> SeedResult:
    if not MIN_GRADE <= grade_number <= MAX_GRADE:
        raise ValueError(
            f"grade {grade_number} is outside {MIN_GRADE}-{MAX_GRADE}; there is no "
            "syllabus data for it")
    board = store.get_board_by_code(CBSE_BOARD_CODE) or store.create_board(name="CBSE", code=CBSE_BOARD_CODE)

    year = (store.get_academic_year_by_label(school_id, academic_year_label)
           or store.create_academic_year(school_id=school_id, label=academic_year_label,
                                          start_date=start_date, end_date=end_date))

    grade = (store.get_grade_by_number(year.id, grade_number)
            or store.create_grade(academic_year_id=year.id, number=grade_number))

    subjects_seeded = 0
    units_seeded = 0
    chapters_seeded = 0
    subjects_skipped: list[str] = []
    topic_templates: list[SubjectTemplateReport] = []

    # Every subject this grade SHOULD have, not just the files that happen to
    # be on disk. The difference is item 3 of the task: while this list came
    # from `get_available_subjects_for_grade` (which is built by scanning the
    # syllabus directory), "expected" and "present" were the same set by
    # construction and `subjects_skipped` could never be non-empty -- measured
    # 2026-09-23, seeding grade 11 reported nothing skipped although
    # History_11/Geography_11/Political_Science_11 do not exist.
    from ..syllabus.cbse_syllabus import get_expected_subjects_for_grade
    expected_subjects = get_expected_subjects_for_grade(grade_number)
    if not expected_subjects and grade_number == 10:
        expected_subjects = list(_FILENAME_BY_SUBJECT.keys())

    for subject_name in expected_subjects:
        doc = load_syllabus(subject_name, grade_number)
        if doc is None:
            subjects_skipped.append(subject_name)
            continue

        subject = (store.get_subject_by_name(grade.id, subject_name)
                  or store.create_subject(grade_id=grade.id, name=subject_name))

        book_title = f"CBSE Class {grade_number} {subject_name} — official curriculum ({doc.source})"
        book = (store.get_book_by_title(subject.id, book_title)
               or store.create_book(subject_id=subject.id, board_id=board.id, title=book_title,
                                     status="ready"))
        subjects_seeded += 1

        prefix = _canonical_prefix(book.id)
        units = list(doc.units)
        if not units and doc.chapters:
            # No CBSE marks table (classes 1-5): the book's chapters, in book
            # order, under one unit that claims no marks.
            units = [SyllabusUnit(unit_no="1", name=f"{subject_name} chapters", marks=0, chapters=doc.chapters)]
        for unit_seq, u in enumerate(units):
            unit_canonical_id = f"{prefix}:unit:{u.unit_no}"
            unit = store.get_unit_by_canonical_id(unit_canonical_id)
            if unit is None:
                unit = store.create_unit(canonical_id=unit_canonical_id, book_id=book.id,
                                         unit_no=u.unit_no, name=u.name,
                                         marks=u.marks if doc.units else None, seq=unit_seq)
                units_seeded += 1

            chapter_specs = (
                [(c.id, c.name) for c in u.chapters] if u.chapters
                else [(_slug(u.name), u.name)]
            )
            for chapter_seq, (chapter_key, chapter_name) in enumerate(chapter_specs):
                chapter_canonical_id = f"{prefix}:chapter:{chapter_key}"
                if store.get_chapter_by_canonical_id(chapter_canonical_id) is None:
                    store.create_chapter(canonical_id=chapter_canonical_id, unit_id=unit.id,
                                         name=chapter_name, seq=chapter_seq)
                    chapters_seeded += 1

        if apply_templates:
            applied = templates.apply_template(
                store, school_id=school_id, book_id=book.id, subject=subject_name,
                grade=grade_number, chapters=store.chapters_for_book(book.id))
            topic_templates.append(SubjectTemplateReport(
                subject=subject_name, chapters=applied.chapters,
                chapters_with_topics=applied.chapters_with_topics,
                chapters_without_topics=applied.chapters_without_topics,
                topics_proposed=applied.topics_proposed,
                subtopics_proposed=applied.subtopics_proposed,
                provenance=applied.provenance, note=applied.note))

    return SeedResult(
        board_id=board.id, academic_year_id=year.id, grade_id=grade.id,
        grade_number=grade_number,
        subjects_seeded=subjects_seeded, units_seeded=units_seeded,
        chapters_seeded=chapters_seeded, subjects_skipped=subjects_skipped,
        topic_templates=topic_templates,
    )


def seed_cbse_class_10(store: CurriculumStore, *, school_id: str,
                       academic_year_label: str, start_date: str, end_date: str,
                       apply_templates: bool = True) -> SeedResult:
    """Kept as its own entry point because /seed/cbse10 and its tests predate
    the any-grade path; it is a thin delegation now, not a second seeder."""
    return seed_cbse_grade(store, school_id=school_id, academic_year_label=academic_year_label,
                           start_date=start_date, end_date=end_date, grade_number=GRADE_10,
                           apply_templates=apply_templates)


def seed_cbse_all_grades(store: CurriculumStore, *, school_id: str,
                         academic_year_label: str, start_date: str, end_date: str,
                         grades: list[int] | None = None,
                         apply_templates: bool = True) -> list[SeedResult]:
    """Seeds CBSE/NCERT curriculum for all main subjects across grades (default 1-12)."""
    target_grades = grades or list(range(MIN_GRADE, MAX_GRADE + 1))
    return [
        seed_cbse_grade(store, school_id=school_id, academic_year_label=academic_year_label,
                        start_date=start_date, end_date=end_date, grade_number=g,
                        apply_templates=apply_templates)
        for g in target_grades
    ]

