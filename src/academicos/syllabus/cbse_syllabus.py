"""CBSE syllabus by class: units, chapters and official marks-weightage, and
for classes 1-5 the NCERT book's own chapters.

Data source: academicos-data/syllabus/*.json, hand-verified against the
official 2025-26 CBSE curriculum PDFs (cbseacademic.nic.in/curriculum_2026.html)
-- see each JSON's "source" field for the exact document and section (each
subject's PDF covers both Class IX and Class X; the Class X table was
extracted specifically, since the two years have different unit/chapter/marks
breakdowns).

CBSE gives marks-weightage per unit, not time/hours -- there is no official
"how many periods should this take" figure (the curriculum's own Section 3.3
explicitly leaves timetable design to individual schools). Anything in this
package that derives a suggested *time* budget (see timetable.py) is doing so
as a documented proportional estimate from marks-weightage, not quoting an
official number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "academicos-data" / "syllabus"


@dataclass(frozen=True)
class SyllabusChapter:
    id: str
    name: str


@dataclass(frozen=True)
class SyllabusUnit:
    unit_no: str
    name: str
    marks: int
    chapters: tuple[SyllabusChapter, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SyllabusDocument:
    subject: str
    grade: int
    total_marks: int
    source: str
    units: tuple[SyllabusUnit, ...]
    # The textbook's own chapters, for a class whose book does not divide into
    # the CBSE unit table this file records marks against. Read from the
    # file's top-level "chapters", and carrying the taxonomy's ids
    # ("mathematics-6/number-play") so that the id a question is tagged with
    # and the id a teacher picks are one vocabulary. Marks stay on the units:
    # CBSE weights a unit, and which chapter sits in which unit is not
    # something this file claims unless the file says so.
    chapters: tuple[SyllabusChapter, ...] = field(default_factory=tuple)

    def all_chapters(self) -> list[tuple[SyllabusUnit | None, SyllabusChapter]]:
        """Flattened (unit, chapter) pairs -- for subjects whose CBSE table
        has no sub-chapter list (e.g. Mathematics, English), each unit stands
        in for its own single chapter, since that's the finest breakdown
        CBSE's own document gives.

        Where the file lists the book's chapters at top level, those are the
        chapters, and a unit with no chapter list of its own does NOT also
        stand in for one: doing both would offer the same class two chapter
        lists from two different editions of its book.
        """
        out: list[tuple[SyllabusUnit | None, SyllabusChapter]] = []
        for c in self.chapters:
            out.append((None, c))
        for u in self.units:
            if u.chapters:
                for c in u.chapters:
                    out.append((u, c))
            elif not self.chapters:
                out.append((u, SyllabusChapter(id=_slug(u.name), name=u.name)))
        return out


def _slug(text: str) -> str:
    return "-".join(text.lower().replace("&", "and").split())[:60]


_FILENAME_BY_SUBJECT = {
    "Mathematics": "Mathematics_10.json",
    "Science": "Science_10.json",
    "Social Science": "Social_Science_10.json",
    "English": "English_10.json",
    "Hindi": "Hindi_10.json",
}


def _resolve_syllabus_file(subject: str, grade: int) -> Path | None:
    slug = subject.strip().replace(" ", "_")
    candidates = [
        f"{slug}_{grade}.json",
        f"{subject}_{grade}.json",
    ]
    if grade == 10:
        legacy = _FILENAME_BY_SUBJECT.get(subject)
        if legacy:
            candidates.append(legacy)
    for c in candidates:
        p = _DATA_DIR / c
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def _ncert_books() -> dict:
    path = _DATA_DIR / "ncert_books.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("classes", {})


def book_chapters(subject: str, grade: int) -> list[dict]:
    """The chapters of the NCERT book(s) this class studies now, in book order:
    [{"id", "name", "book", "bookTitle", "number", "topics": [{"id", "title"}]}].

    Written by scripts/build_textbook_bank.py from the chapters it read, with
    the id a syllabus file gives the chapter where one names it, and the
    taxonomy's form ("mathematics-6/number-play") where none does. These are
    the current books (NCERT's 2026-27 catalogue), so a class whose syllabus
    file still lists an older edition (class 9), or lists CBSE assessment units
    rather than chapters (the languages), still offers the chapters its book
    prints. An older edition's slugs are not in here and stay unfiled.
    """
    return list(_ncert_books().get(f"{subject}|{grade}", {}).get("chapters", []))


@lru_cache(maxsize=None)
def load_syllabus(subject: str, grade: int) -> SyllabusDocument | None:
    path = _resolve_syllabus_file(subject, grade)
    if path is None or not path.exists():
        # No CBSE syllabus file (classes 1-5 have no CBSE marks table): the
        # book's own chapters are the syllabus, with no units and no marks.
        chapters = book_chapters(subject, grade)
        if not chapters:
            return None
        # A fixed wording: the seed names the class's book from `source`, so a
        # source that grew as chapters arrived made a re-seed create a second book.
        return SyllabusDocument(
            subject=subject, grade=grade, total_marks=0,
            source=f"NCERT Class {grade} {subject}: the book's own chapters (no CBSE marks table)",
            units=(), chapters=tuple(SyllabusChapter(id=c["id"], name=c["name"]) for c in chapters))
    data = json.loads(path.read_text(encoding="utf-8"))
    units = tuple(
        SyllabusUnit(
            unit_no=u["unit_no"], name=u["name"], marks=u["marks"],
            chapters=tuple(SyllabusChapter(id=c["id"], name=c["name"]) for c in u.get("chapters", [])),
        )
        for u in data["units"]
    )
    return SyllabusDocument(
        subject=data["subject"], grade=data["grade"], total_marks=data["total_marks"],
        source=data["source"], units=units,
        chapters=tuple(SyllabusChapter(id=c["id"], name=c["name"])
                       for c in data.get("chapters", [])),
    )


@lru_cache(maxsize=None)
def taxonomy_chapters(subject: str, grade: int) -> dict[str, str]:
    """{taxonomy chapter id: the name the book prints it under}, or {}.

    The taxonomy trees are the textbooks' own contents pages and are what the
    tagger writes `taxonomyChapterId` from. A caller that has a tag and needs
    the syllabus chapter it belongs to matches on this name: where the
    syllabus file and the taxonomy read the same edition (Science 6, Science
    10) the names are the same words, and where they do not, no name matches
    and the caller is told so rather than guessing.
    """
    path = _DATA_DIR / "taxonomy" / f"{subject.strip().replace(' ', '_')}_{grade}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["id"]: c["name"] for c in data.get("chapters", [])}


def list_available_syllabi() -> list[tuple[str, int]]:
    """Returns sorted list of (subject, grade) tuples available on disk."""
    out: list[tuple[str, int]] = []
    if not _DATA_DIR.exists():
        return out
    for f in _DATA_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "subject" in data and "grade" in data:
                out.append((data["subject"], int(data["grade"])))
        except Exception:
            continue
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def get_available_subjects_for_grade(grade: int) -> list[str]:
    """Subjects with a syllabus for the requested grade: a CBSE file on disk,
    or (classes 1-5) the chapters of the class's NCERT book."""
    books = {key.split("|")[0] for key in _ncert_books() if key.endswith(f"|{grade}")}
    return sorted({s for s, g in list_available_syllabi() if g == grade} | books)


# What a CBSE school should be able to set up for a grade -- DECLARED here, not
# derived from the files on disk.
#
# The distinction is the whole of PRD item 3 ("a grade with no syllabus file for
# a subject seeds the ones that do exist and REPORTS the ones that don't").
# Deriving the list from `list_available_syllabi()` makes "expected" and
# "present" the same set by construction, so nothing is ever reported missing
# and a genuinely absent subject just isn't there: measured 2026-09-23, seeding
# grade 11 returned subjects_skipped == [] while History_11.json,
# Geography_11.json and Political_Science_11.json do not exist and their
# grade-12 equivalents do.
#
# Grades 6-10 are the CBSE secondary core. Grades 11-12 list the electives this
# repo carries syllabus data for in either year; a school does not teach all of
# them, but a missing FILE for one is still a data gap worth naming rather than
# a curriculum choice this module can make on a school's behalf.
_SECONDARY_CORE = ("English", "Hindi", "Mathematics", "Science", "Social Science")
_SENIOR_SECONDARY = (
    "Accountancy", "Biology", "Business Studies", "Chemistry", "Economics",
    "English", "Geography", "Hindi", "History", "Mathematics", "Physics",
    "Political Science",
)
# Classes 1-5 (requirements v3): the subjects NCERT publishes a book for --
# EVS from class 3, when "The World Around Us" begins.
_PRIMARY = ("English", "Hindi", "Mathematics")
EXPECTED_SUBJECTS_BY_GRADE: dict[int, tuple[str, ...]] = {
    1: _PRIMARY, 2: _PRIMARY, 3: _PRIMARY + ("EVS",), 4: _PRIMARY + ("EVS",), 5: _PRIMARY + ("EVS",),
    **{g: _SECONDARY_CORE for g in range(6, 11)},
    **{g: _SENIOR_SECONDARY for g in (11, 12)},
}


def get_expected_subjects_for_grade(grade: int) -> list[str]:
    """Every subject a seed for this grade should try, whether or not a file
    exists: the declared list, plus anything on disk that isn't in it (a new
    subject file should be seeded, not ignored because the list is older)."""
    return sorted(set(EXPECTED_SUBJECTS_BY_GRADE.get(grade, ()))
                  | set(get_available_subjects_for_grade(grade)))


def get_missing_subjects_for_grade(grade: int) -> list[str]:
    """Expected subjects with no syllabus file for this grade."""
    available = set(get_available_subjects_for_grade(grade))
    return sorted(s for s in EXPECTED_SUBJECTS_BY_GRADE.get(grade, ())
                  if s not in available)

