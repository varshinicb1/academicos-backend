"""Which chapter of the CURRENT book an NCERT Exemplar chapter belongs to.

An Exemplar question was printed under a numbered chapter of its own book, and
``ncert_exemplar.to_bank_record`` keeps that as ``metadata.exemplarChapter``.
The served taxonomy is keyed by the books schools teach now, which for classes
6-9 are the NEP books: they renamed and reorganised the old chapters
("Integers" -> "The Other Side of Zero"), so the Exemplar chapter's taxonomy id
cannot be looked up by name, and it must not be guessed by word overlap.

``academicos-data/syllabus/taxonomy/_exemplar_chapter_map.json`` is that
correspondence, decided once against the current books' own contents pages and
section headings and committed with the reason for every row
(``scripts/build_exemplar_chapter_map.py``). 57 of its 144 rows name a chapter;
the other 87 are null, because the old chapter is split across two chapters of
the current book, was dropped by the NEP rationalisation, moved to another
class, or keeps more than a tenth of its questions on content that class teaches
elsewhere. A null row leaves the record to the tagger, which may still leave it
blank -- a blank shows in the coverage report, a wrong chapter does not.

This module only reads the map. ``syllabus.tagger.apply_tags`` is what writes it
onto a record, with ``tagMethod = "exemplar-chapter-map"`` and chapter
confidence 1.0: it is the book's own filing, not a model's opinion, and nothing
downstream should weigh it as one.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MAP_PATH = REPO / "academicos-data" / "syllabus" / "taxonomy" / "_exemplar_chapter_map.json"

# What `apply_tags` writes as `tagMethod` on a record the map placed. Not a
# version of the tagger's own method: those tags came from the question's words
# and carry a fitted confidence, these came from the book's table of contents.
EXEMPLAR_MAP_METHOD = "exemplar-chapter-map"

_CACHE: dict[Path, dict[tuple[str, int, int, str], str | None]] = {}


def load_chapter_map(path: Path | None = None) -> dict[tuple[str, int, int, str], str | None]:
    """(subject, grade, exemplar number, exemplar title) -> chapter id or None.

    A key with value None is a decided row -- the current book has no single
    counterpart chapter -- and a MISSING key is an Exemplar chapter the map has
    never seen, which ``tests/test_ncert_exemplar.py`` refuses to let happen.
    """
    path = Path(path) if path else MAP_PATH
    if path not in _CACHE:
        doc = json.loads(path.read_text(encoding="utf-8"))
        _CACHE[path] = {
            (row["subject"], int(row["grade"]), int(row["exemplarNumber"]),
             row["exemplarTitle"]): row["taxonomyChapterId"]
            for row in doc["rows"]
        }
    return _CACHE[path]


def chapter_for(rec: dict, path: Path | None = None) -> str | None:
    """The taxonomy chapter the current book prints this record's Exemplar
    chapter as, or None for a record with no Exemplar chapter, a null row, or a
    chapter the map does not hold."""
    chapter = (rec.get("metadata") or {}).get("exemplarChapter")
    if not chapter or not chapter.get("title") or chapter.get("number") is None:
        return None
    try:
        key = (rec.get("subject"), int(rec.get("grade")), int(chapter["number"]),
               chapter["title"])
    except (TypeError, ValueError):
        return None
    return load_chapter_map(path).get(key)
