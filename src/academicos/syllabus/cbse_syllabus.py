"""Real CBSE Class X syllabus: units, chapters, and official marks-weightage.

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

    def all_chapters(self) -> list[tuple[SyllabusUnit, SyllabusChapter]]:
        """Flattened (unit, chapter) pairs -- for subjects whose CBSE table
        has no sub-chapter list (e.g. Mathematics, English), each unit stands
        in for its own single chapter, since that's the finest breakdown
        CBSE's own document gives."""
        out: list[tuple[SyllabusUnit, SyllabusChapter]] = []
        for u in self.units:
            if u.chapters:
                for c in u.chapters:
                    out.append((u, c))
            else:
                out.append((u, SyllabusChapter(id=_slug(u.name), name=u.name)))
        return out


def _slug(text: str) -> str:
    return "-".join(text.lower().replace("&", "and").split())[:60]


_FILENAME_BY_SUBJECT = {
    "Mathematics": "Mathematics_10.json",
    "Science": "Science_10.json",
    "Social Science": "Social_Science_10.json",
    "English": "English_10.json",
}


@lru_cache(maxsize=None)
def load_syllabus(subject: str, grade: int) -> SyllabusDocument | None:
    if grade != 10:
        return None  # only Class X is populated for now
    fname = _FILENAME_BY_SUBJECT.get(subject)
    if fname is None:
        return None
    path = _DATA_DIR / fname
    if not path.exists():
        return None
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
    )
