"""Seeds the real, official CBSE Class X curriculum data
(syllabus/cbse_syllabus.py, academicos-data/syllabus/*.json --
hand-verified against the official 2025-26 CBSE curriculum PDFs) into the
curriculum/ data model as the first real Board -> AcademicYear -> Grade ->
Subject -> Book -> Unit -> Chapter rows.

This does NOT invent Topic/Subtopic data -- the source CBSE documents
don't break down below chapter level (see docs/ACADEMIC_DATA_MODEL.md
section 6, an open decision on how that content should be populated).
Chapters seeded here have zero topics/subtopics until that follow-up work
lands; they are real, correctly-sourced Chapter rows in the meantime, not
placeholders.

Idempotent: safe to run more than once against the same store -- reuses
existing AcademicYear/Grade/Subject/Book rows by their natural key, and
Unit/Chapter rows by canonical_id (which has a UNIQUE constraint), rather
than erroring or creating duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..syllabus.cbse_syllabus import _FILENAME_BY_SUBJECT, _slug, load_syllabus
from .store import CurriculumStore

CBSE_BOARD_CODE = "CBSE"
GRADE_10 = 10


@dataclass
class SeedResult:
    board_id: str
    academic_year_id: str
    grade_id: str
    subjects_seeded: int
    units_seeded: int
    chapters_seeded: int
    subjects_skipped: list[str]   # subject names with no real syllabus JSON


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


def seed_cbse_class_10(store: CurriculumStore, *, school_id: str,
                       academic_year_label: str, start_date: str, end_date: str) -> SeedResult:
    board = store.get_board_by_code(CBSE_BOARD_CODE) or store.create_board(name="CBSE", code=CBSE_BOARD_CODE)

    year = (store.get_academic_year_by_label(school_id, academic_year_label)
           or store.create_academic_year(school_id=school_id, label=academic_year_label,
                                          start_date=start_date, end_date=end_date))

    grade = (store.get_grade_by_number(year.id, GRADE_10)
            or store.create_grade(academic_year_id=year.id, number=GRADE_10))

    subjects_seeded = 0
    units_seeded = 0
    chapters_seeded = 0
    subjects_skipped: list[str] = []

    for subject_name in _FILENAME_BY_SUBJECT:
        doc = load_syllabus(subject_name, GRADE_10)
        if doc is None:
            subjects_skipped.append(subject_name)
            continue

        subject = (store.get_subject_by_name(grade.id, subject_name)
                  or store.create_subject(grade_id=grade.id, name=subject_name))

        book_title = f"CBSE Class X {subject_name} — official curriculum ({doc.source})"
        book = (store.get_book_by_title(subject.id, book_title)
               or store.create_book(subject_id=subject.id, board_id=board.id, title=book_title,
                                     status="ready"))
        subjects_seeded += 1

        prefix = _canonical_prefix(book.id)
        for unit_seq, u in enumerate(doc.units):
            unit_canonical_id = f"{prefix}:unit:{u.unit_no}"
            unit = store.get_unit_by_canonical_id(unit_canonical_id)
            if unit is None:
                unit = store.create_unit(canonical_id=unit_canonical_id, book_id=book.id,
                                         unit_no=u.unit_no, name=u.name, marks=u.marks, seq=unit_seq)
                units_seeded += 1

            # Real CBSE units with an explicit chapter list (e.g. Science);
            # units with none (e.g. Mathematics, English) stand in for
            # their own single chapter -- same fallback
            # SyllabusDocument.all_chapters() already documents and uses,
            # replicated here so a Unit with no sub-chapters still gets a
            # real, schedulable Chapter row instead of being unreachable
            # from the curriculum hierarchy.
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

    return SeedResult(
        board_id=board.id, academic_year_id=year.id, grade_id=grade.id,
        subjects_seeded=subjects_seeded, units_seeded=units_seeded,
        chapters_seeded=chapters_seeded, subjects_skipped=subjects_skipped,
    )
