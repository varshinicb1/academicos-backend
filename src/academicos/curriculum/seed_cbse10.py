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

import json
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Optional

from ..syllabus import cbse_syllabus
from ..syllabus.cbse_syllabus import (_FILENAME_BY_SUBJECT, SyllabusChapter, SyllabusDocument, SyllabusUnit,
                                      _slug, book_chapters, load_syllabus, taxonomy_chapters)
from . import decomposition_templates as templates
from .store import CurriculumStore

CBSE_BOARD_CODE = "CBSE"
GRADE_10 = 10
# Requirements v3: classes 1-12. Classes 1-5 have no CBSE marks table; their
# syllabus is the NCERT book's chapters (cbse_syllabus.book_chapters). Outside
# 1-12 the route answers 422 rather than creating an empty grade row.
MIN_GRADE = 1
MAX_GRADE = 12

# Audit D118. Mathematics 6-10 and Science 7-9 list their CBSE units (with
# the marks) and, separately, the book's own chapters -- not which chapter
# is in which unit. Each unit used to be seeded as its own only chapter, so
# those classes had chapters no textbook prints and no topic source could
# key onto. The book's chapters are seeded now, under the unit the CBSE
# course structure puts them in where it says so: an explicit map below, or
# a chapter named exactly as a unit. Where the course structure does not
# place every chapter (the NCF-SE books of classes 6-9, whose units are not
# a CBSE chapter list), the whole book goes under one unit carrying the
# subject's marks.
#
# Class X Mathematics: the CBSE 2025-26 course structure
# (Maths_Sec_2025-26.pdf), Units I-VII.
_CHAPTER_UNITS: dict[tuple[str, int], dict[str, str]] = {
    ("Mathematics", 10): {
        "mathematics-10/real-numbers": "I",
        "mathematics-10/polynomials": "II",
        "mathematics-10/pair-of-linear-equations-in-two-variables": "II",
        "mathematics-10/quadratic-equations": "II",
        "mathematics-10/arithmetic-progressions": "II",
        "mathematics-10/coordinate-geometry": "III",
        "mathematics-10/triangles": "IV",
        "mathematics-10/circles": "IV",
        "mathematics-10/introduction-to-trigonometry": "V",
        "mathematics-10/some-applications-of-trigonometry": "V",
        "mathematics-10/areas-related-to-circles": "VI",
        "mathematics-10/surface-areas-and-volumes": "VI",
        "mathematics-10/statistics": "VII",
        "mathematics-10/probability": "VII",
    },
}
# The one unit a book's chapters go under when the units cannot hold them
# (the same number and naming as the classes 1-5 holder unit).
BOOK_UNIT_NO = "1"


def _name_key(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def book_chapters_for(subject: str, grade: int, doc: SyllabusDocument) -> list[SyllabusChapter]:
    """The book's own chapters in book order: the whole book as its contents
    page prints it (ncert_books.json: every class 1-5 book, and Hindi 6-9,
    whose contents trees are missing or garbled), else the taxonomy file (the
    textbook's contents pages), else the syllabus file's top-level list."""
    listed = book_chapters(subject, grade)
    if listed:
        return [SyllabusChapter(id=c["id"], name=c["name"]) for c in listed]
    contents = taxonomy_chapters(subject, grade)
    if contents:
        return [SyllabusChapter(id=cid, name=name) for cid, name in contents.items()]
    return list(doc.chapters)


def units_holding_book_chapters(subject: str, grade: int, doc: SyllabusDocument,
                                chapters: list[SyllabusChapter]) -> list[SyllabusUnit]:
    """The units to seed for a syllabus whose units list no chapters, with
    the book's chapters placed in them (D118). Every chapter goes under its
    CBSE unit when the explicit map or an exact unit name places it; if any
    chapter is left over, the whole book goes under one unit instead, so
    no chapter is put in a unit by a guess."""
    explicit = _CHAPTER_UNITS.get((subject, grade), {})
    by_no = {u.unit_no: u for u in doc.units}
    by_name = {_name_key(u.name): u for u in doc.units}
    placed: dict[str, list[SyllabusChapter]] = {u.unit_no: [] for u in doc.units}
    for c in chapters:
        unit = by_no.get(explicit.get(c.id, "")) or by_name.get(_name_key(c.name))
        if unit is None:
            marks = sum(u.marks for u in doc.units) or doc.total_marks
            return [SyllabusUnit(unit_no=BOOK_UNIT_NO, name=f"{subject} chapters", marks=marks,
                                 chapters=tuple(chapters))]
        placed[unit.unit_no].append(c)
    return [SyllabusUnit(unit_no=u.unit_no, name=u.name, marks=u.marks, chapters=tuple(placed[u.unit_no]))
            for u in doc.units]


# The unit of a language syllabus that is the textbook ("Poorvi Textbook",
# "Literature", "Pathyapustak Vasant"); its other units are skills (reading,
# grammar, writing) taught from no book chapter.
_TEXTBOOK_UNIT = re.compile(r"textbook|literature|pathyapustak", re.IGNORECASE)
LANGUAGES = ("English", "Hindi")


def _book_title(subject: str, grade: int) -> Optional[str]:
    listed = book_chapters(subject, grade)
    if listed:
        return listed[0].get("bookTitle")
    path = cbse_syllabus._DATA_DIR / "taxonomy" / f"{subject.strip().replace(' ', '_')}_{grade}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("book") if path.exists() else None


def units_with_the_current_book(subject: str, grade: int, doc: SyllabusDocument,
                                book: list[SyllabusChapter]) -> Optional[list[SyllabusUnit]]:
    """The units to seed when the syllabus file's chapters are not the book a
    school teaches from now, or None when they are.

    Measured 2026-10-04: Social Science 7-9 seeded 0 chapters of the current
    book (Our Pasts, not Exploring Society), Social Science 6 missed 4 of its
    14, and English and Hindi 6-9 seeded only skill headings ("Reading
    Comprehension", "Vasant Bhag 2 (Kavita evam Kahani)"), no lesson at all.

    A language keeps its skill units and its textbook unit holds the book's
    lessons, named after the book; any other subject takes the book's
    chapters the way D118 does (units_holding_book_chapters)."""
    listed = [c for u in doc.units for c in u.chapters]
    if not book or not listed:
        return None
    if {_name_key(c.name) for c in listed} >= {_name_key(c.name) for c in book}:
        return None
    if subject in LANGUAGES:
        holders = [u for u in doc.units if _TEXTBOOK_UNIT.search(u.name)]
        if len(holders) != 1:
            return None
        title = _book_title(subject, grade)
        return [SyllabusUnit(unit_no=u.unit_no, marks=u.marks,
                             name=(f"Textbook: {title}" if title else u.name) if u is holders[0] else u.name,
                             chapters=tuple(book) if u is holders[0] else u.chapters)
                for u in doc.units]
    return units_holding_book_chapters(subject, grade, doc, book)


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
        elif doc.chapters and not any(u.chapters for u in units):
            # D118: the book's own chapters, under their CBSE units where the
            # course structure places them (units_holding_book_chapters).
            units = units_holding_book_chapters(subject_name, grade_number, doc,
                                                book_chapters_for(subject_name, grade_number, doc))
        else:
            units = units_with_the_current_book(subject_name, grade_number, doc,
                                                book_chapters_for(subject_name, grade_number, doc)) or units
        for unit_seq, u in enumerate(units):
            unit_canonical_id = f"{prefix}:unit:{u.unit_no}"
            unit = store.get_unit_by_canonical_id(unit_canonical_id)
            if unit is None:
                # After every unit the book already has: a re-seed that adds
                # the book's holder unit (D118) puts it after the old ones.
                unit = store.create_unit(canonical_id=unit_canonical_id, book_id=book.id,
                                         unit_no=u.unit_no, name=u.name,
                                         marks=u.marks if doc.units else None,
                                         seq=max(unit_seq, len(store.units_for_book(book.id))))
                units_seeded += 1

            chapter_specs = (
                [(c.id, c.name) for c in u.chapters] if u.chapters
                else [] if doc.chapters
                else [(_slug(u.name), u.name)]
            )
            # A re-seed of a school seeded before D118 adds the book's
            # chapters after the unit-named chapter it already has, never
            # deleting or renaming it (it may carry approved topics and taught
            # lessons).
            wanted = {f"{prefix}:chapter:{key}" for key, _ in chapter_specs}
            earlier = sum(1 for c in store.chapters_for_unit(unit.id) if c.canonical_id not in wanted)
            for chapter_seq, (chapter_key, chapter_name) in enumerate(chapter_specs):
                chapter_canonical_id = f"{prefix}:chapter:{chapter_key}"
                if store.get_chapter_by_canonical_id(chapter_canonical_id) is None:
                    store.create_chapter(canonical_id=chapter_canonical_id, unit_id=unit.id,
                                         name=chapter_name, seq=earlier + chapter_seq)
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

