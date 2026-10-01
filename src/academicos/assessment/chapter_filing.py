"""Which chapter a question is filed under: one answer for the chapter a
teacher is shown and the chapter a paper draws from.

The builder's chapter picker lists chapters, with counts, from
GET /catalog/{subject}/{grade}/chapters, which files each question by its
taxonomy tag first (`pillar_routes._chapter_of`). Until 2026-09-27 every filter
that takes a teacher's chosen chapters -- the template scope, quick-generate,
the question search, the parallel sets -- matched the record's own
`chapterIds` instead. Audit D24 (2026-09-26): across classes 6-10 Mathematics
and Science, 2,297 questions were filed under named chapters and choosing
those chapters reached 1,266; Mathematics 6 "Lines and Angles" showed 16
questions and generating a paper from it returned 400.

Now both read this rule. `pillar_routes._chapter_of` still keeps a copy (its
file has another owner); tests/test_chapter_filing.py holds the two equal on
every served class 6-10 question, so neither can change alone. The question
bank API and the MCP server read it too, through `qbank_engine.QuestionBank`
(audit D39); tests/test_qbank_chapter_filing.py holds them to the picker.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

# The chapter view's bucket for a question no chapter claims.
UNMAPPED = "unmapped"


def name_key(name: str) -> str:
    """A chapter name compared as the book prints it, ignoring case,
    punctuation and the dash and apostrophe variants the sources mix
    ("Light – Reflection and Refraction" == "Light - Reflection and
    Refraction")."""
    name = (name.replace("–", "-").replace("—", "-")
                .replace("’", "'").replace("…", ""))
    # Letters, digits and combining marks of any script: an ASCII-only key made
    # every Devanagari name "", so all Hindi chapters were one chapter and any
    # unknown tag found a "twin" among them (review of #54).
    return "".join(ch for ch in unicodedata.normalize("NFC", name).lower() if unicodedata.category(ch)[0] in "LMN")


@dataclass(frozen=True)
class ChapterFiling:
    """The chapters of one class and subject, as the teacher's view lists them.

    `known` is the syllabus's chapter ids -- the book the class studies.
    `by_name` maps a chapter's name to its syllabus id; `book_names` maps a
    taxonomy chapter id to the name the book prints, so a tag under a
    different id for the same chapter finds its syllabus twin.
    """
    known: frozenset[str]
    by_name: dict[str, str] = field(default_factory=dict)
    book_names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_class(cls, subject: str, grade: int) -> "ChapterFiling":
        from ..syllabus.cbse_syllabus import book_chapters, load_syllabus, taxonomy_chapters
        syllabus = load_syllabus(subject, grade)
        chapters = syllabus.all_chapters() if syllabus is not None else []
        # The current NCERT book's chapters are chapters the class studies too
        # (`book_chapters`): known, and named as the book prints them.
        book = book_chapters(subject, grade)
        by_name = {name_key(c["name"]): c["id"] for c in book}
        by_name.update({name_key(c.name): c.id for _unit, c in chapters})
        by_name.pop("", None)          # a name with no letters names no chapter
        return cls(known=frozenset(c.id for _unit, c in chapters) | {c["id"] for c in book},
                   by_name=by_name,
                   book_names={**{c["id"]: c["name"] for c in book}, **taxonomy_chapters(subject, grade)})

    def chapter_of(self, taxonomy_chapter_id: Optional[str],
                   chapter_ids: Iterable[str]) -> str:
        """The one chapter a question is filed under.

        The record carries up to two chapter claims and they are not the same
        kind. `taxonomyChapterId` is the chapter of the NCERT book the tagger
        read (or, for an Exemplar record, the chapter the book printed it in);
        `chapterIds` is whatever the source bank supplied, which for six of the
        ten class 6-10 Mathematics/Science pairs names a chapter of the pre-2024
        book. So the tag is preferred, and `chapterIds` is read only when the
        syllabus still knows the id. Anything left is UNMAPPED: a question with
        no chapter is shown as having none, never given one it did not earn.
        """
        known = self.known
        tid = taxonomy_chapter_id
        if tid:
            if not known or tid in known:
                return tid
            twin = self.by_name.get(name_key(self.book_names.get(tid, ""))) if self.book_names.get(tid) else None
            if twin:
                return twin
        for cid in chapter_ids:
            if not known or cid in known:
                return cid
        return UNMAPPED

    def selects(self, chosen: Iterable[str], filed_in_class: Iterable[str],
                ) -> Callable[[Optional[str], Iterable[str]], bool]:
        """A test for "is this question in the chosen chapters?".

        A question is in them when the view files it under one of them. A
        chosen id the view cannot show -- a pre-2024 slug a template saved
        before the view changed -- keeps matching the records that carry it,
        as it always did, rather than emptying the paper. `filed_in_class` is
        every chapter the class's questions are filed under, so "can show" is
        what the view would list: the syllabus's chapters and those.
        """
        wanted = set(chosen)
        legacy = wanted - set(self.known) - set(filed_in_class) - {UNMAPPED}

        def test(taxonomy_chapter_id: Optional[str], chapter_ids: Iterable[str]) -> bool:
            ids = list(chapter_ids)
            if self.chapter_of(taxonomy_chapter_id, ids) in wanted:
                return True
            return bool(legacy) and not legacy.isdisjoint(ids)

        return test
