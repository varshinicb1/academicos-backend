"""Reusable Chapter -> Topic -> Subtopic templates, applied to a newly seeded
school as PENDING proposals.

Why this exists, measured on a fresh school 2026-09-22: `POST /seed/cbse10`
produced 5 subjects, 23 units and 58 chapters, and then
`GET /chapters/{ch}/topics` returned `[]` for every one of them. Everything
below chapter level -- micro-scheduling, the assessment designer's subtopic
filter, coverage -- reads Subtopic rows, so none of it worked for any school
except school_1, whose decomposition was approved by hand months earlier. The
syllabus JSONs stop at chapters, and extraction.py deliberately has no offline
fallback (inventing a chapter's internals with no source is not something a
deterministic heuristic can do honestly), so without a Sarvam key a new school
could never get past chapters at all.

A template is the honest deterministic path that was missing: content that
already exists somewhere real, re-keyed so it can be offered to a new school.
The two committed sources (see scripts/export_decomposition_templates.py) are
school_1's approved Class 10 decomposition and the NCERT textbooks' own printed
section headings, the latter read out of the PDFs by
`academicos.syllabus.textbook_headings` -- in the repo and re-runnable, so
every committed template can be rebuilt from its source.

Applied as proposals, never as rows. school_1's principal approved those topics
for school_1; writing them into another school's `topics` table would be this
system claiming an approval nobody gave. So a template goes in through exactly
the states extraction.py already defines -- a CurriculumExtractionRun whose
`prompt_version` is the template's provenance and whose proposals are "pending"
-- and `extraction.approve_run` is still the only thing that can materialize
them. The one difference from an LLM run is what approval records: source_type
"imported", model_used null.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .models import Chapter
from .store import CurriculumStore

_TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "academicos-data" / "syllabus" / "decomposition"
_TAXONOMY_DIR = _TEMPLATE_DIR.parent / "taxonomy"

# A run created from a template carries its provenance in prompt_version, which
# is exactly what a principal reviewing the proposal needs to see: whether this
# came from another school's approved work or from a textbook's own headings.
PROVENANCE_PREFIX = "template:"
# The textbook's own contents (academicos-data/syllabus/taxonomy), for the
# chapters seeded from the book itself (audit D118).
CONTENTS_PROVENANCE = PROVENANCE_PREFIX + "ncert-textbook-contents"
# Classes 1-5 are seeded from the book too, but their topic proposals are a
# separate decision; this source is used from class 6 up.
CONTENTS_FROM_GRADE = 6


@dataclass(frozen=True)
class TemplateSubtopic:
    name: str
    # The textbook's own section number ("2.2.1") for a heading-derived
    # template. None both for a template exported from an approved
    # decomposition and for one from a book that prints headings but does not
    # number them (the Class 6-8 Social Science titles). Kept because, where it
    # exists, it is the thing a principal can check the proposal against.
    heading_number: Optional[str] = None


@dataclass(frozen=True)
class TemplateTopic:
    name: str
    heading_number: Optional[str] = None
    subtopics: tuple[TemplateSubtopic, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TemplateChapter:
    chapter_slug: str
    chapter_name: str
    provenance: str
    topics: tuple[TemplateTopic, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DecompositionTemplate:
    subject: str
    grade: int
    chapters: dict[str, TemplateChapter]   # keyed by syllabus chapter slug


@lru_cache(maxsize=None)
def load_template(subject: str, grade: int) -> Optional[DecompositionTemplate]:
    """The committed template for a subject/grade, or None when neither source
    covered it (Class 11-12, Hindi, and the Class 7-9 subjects whose syllabus
    JSON predates the NEP textbook edition -- see the decomposition
    `_manifest.json` for the per-subject reason)."""
    path = _TEMPLATE_DIR / f"{subject.strip().replace(' ', '_')}_{grade}.json"
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    chapters = {
        c["chapter_slug"]: TemplateChapter(
            chapter_slug=c["chapter_slug"], chapter_name=c["chapter_name"],
            provenance=c["provenance"],
            topics=tuple(
                TemplateTopic(
                    name=t["name"], heading_number=t.get("heading_number"),
                    subtopics=tuple(
                        TemplateSubtopic(name=s["name"], heading_number=s.get("heading_number"))
                        for s in t.get("subtopics", [])
                    ),
                )
                for t in c.get("topics", [])
            ),
        )
        for c in body.get("chapters", [])
    }
    return DecompositionTemplate(subject=body["subject"], grade=int(body["grade"]),
                                 chapters=chapters)


@lru_cache(maxsize=None)
def contents_template(subject: str, grade: int) -> Optional[DecompositionTemplate]:
    """The textbook's contents pages (taxonomy/<Subject>_<grade>.json) as a
    template, keyed by the taxonomy chapter id -- the key seed_cbse10 writes
    for a chapter it took from the book (D118). Its sections are the topics
    and their printed sub-sections the subtopics. A section the book does
    not divide is proposed as one subtopic of its own name: it is the
    smallest piece the book prints, and a topic with no subtopic can never
    be timed or dated, so its chapter would drop out of every plan. A
    chapter whose contents list no sections gets no proposal. None below
    class 6 or with no taxonomy file."""
    if grade < CONTENTS_FROM_GRADE:
        return None
    path = _TAXONOMY_DIR / f"{subject.strip().replace(' ', '_')}_{grade}.json"
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    chapters = {}
    for c in body.get("chapters", []):
        topics = []
        for t in c.get("topics", []):
            number = t.get("number")
            subtopics = tuple(TemplateSubtopic(name=s["name"]) for s in t.get("subtopics") or ())
            topics.append(TemplateTopic(name=t["name"], heading_number=number,
                                        subtopics=subtopics or (TemplateSubtopic(name=t["name"],
                                                                                 heading_number=number),)))
        chapters[c["id"]] = TemplateChapter(chapter_slug=c["id"], chapter_name=c["name"],
                                            provenance=CONTENTS_PROVENANCE, topics=tuple(topics))
    return DecompositionTemplate(subject=body.get("subject", subject), grade=grade, chapters=chapters)


@lru_cache(maxsize=1)
def manifest() -> dict:
    """The export's own report of what each subject/grade got and why it didn't
    -- surfaced by the seed route so a principal is told "Class 7 is chapter-only
    because ..." instead of being left with a setup that looks complete."""
    path = _TEMPLATE_DIR / "_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def why_uncovered(subject: str, grade: int) -> Optional[str]:
    for row in manifest().get("by_subject_grade", []):
        if row.get("subject") == subject and int(row.get("grade", -1)) == grade:
            return row.get("why_uncovered")
    return None


@dataclass
class TemplateApplication:
    """What applying a subject's template actually did, per subject -- reported
    all the way up to the seed route. `chapters_without_topics` plus `note` is
    the honest half: a subject with no source says so."""
    subject: str
    grade: int
    chapters: int
    chapters_with_topics: int
    chapters_without_topics: int
    topics_proposed: int
    subtopics_proposed: int
    provenance: list[str] = field(default_factory=list)
    note: Optional[str] = None


def chapter_slug(chapter: Chapter) -> Optional[str]:
    """`book_xxxx:chapter:<syllabus slug>` -> `<syllabus slug>`, the key a
    template is written against. A chapter created by the TOC-ingest path is
    slugged from its own title and simply won't match -- returning it anyway is
    fine, no template will hold that key."""
    marker = ":chapter:"
    i = chapter.canonical_id.find(marker)
    return chapter.canonical_id[i + len(marker):] if i >= 0 else None


def apply_template(store: CurriculumStore, *, school_id: str, book_id: str,
                   subject: str, grade: int,
                   chapters: list[Chapter]) -> TemplateApplication:
    """Proposes the template's topics/subtopics for every chapter of one book
    that doesn't already have them.

    Idempotent two ways, because seeding is documented as safe to re-run: a
    chapter that already has real (approved) topics is left alone, and so is one
    that already carries a template run -- otherwise a second seed would hand
    the principal a second identical pile of pending proposals to review.

    A chapter no committed template covers falls back to the textbook's own
    contents (contents_template), which covers the chapters seeded from the
    book (D118).
    """
    tpl = load_template(subject, grade)
    contents = contents_template(subject, grade)
    with_topics = topics = subtopics = 0
    provenance: list[str] = []
    no_source: list[str] = []

    for chapter in chapters:
        slug = chapter_slug(chapter) or ""
        entry = ((tpl.chapters.get(slug) if tpl else None)
                 or (contents.chapters.get(slug) if contents else None))
        if entry is None or not entry.topics:
            no_source.append(chapter.name)
            continue
        if store.topics_for_chapter(chapter.id):
            continue
        if any(r.prompt_version.startswith(PROVENANCE_PREFIX)
               for r in store.extraction_runs_for_chapter(chapter.id)):
            continue

        run = store.create_extraction_run(
            school_id=school_id, book_id=book_id, chapter_id=chapter.id,
            # No source_hash: the grounding is the named template, not a blob of
            # text this run hashed. No model: nothing generated this.
            source_hash=None, model=None, prompt_version=entry.provenance)
        for t_seq, t in enumerate(entry.topics):
            topic_proposal = store.add_extraction_proposal(
                run_id=run.id, entity_type="topic", proposed_name=t.name,
                proposed_description=_heading_note(t.heading_number),
                proposed_parent=None, sequence=t_seq,
                # 1.0 is not "certainly right", it is "this was copied
                # verbatim from a real source, nothing was inferred" -- the
                # review is still the quality gate (extraction.py).
                confidence=1.0)
            topics += 1
            for s_seq, s in enumerate(t.subtopics):
                store.add_extraction_proposal(
                    run_id=run.id, entity_type="subtopic", proposed_name=s.name,
                    proposed_description=_heading_note(s.heading_number),
                    proposed_parent=topic_proposal.id, sequence=s_seq, confidence=1.0)
                subtopics += 1
        with_topics += 1
        if entry.provenance not in provenance:
            provenance.append(entry.provenance)

    note = None
    if contents is not None and any((chapter_slug(c) or "") in contents.chapters for c in chapters):
        # Chapters seeded from the book: the manifest's reasons were written
        # before its contents were a source, so they no longer describe them.
        if no_source:
            note = (f"{len(no_source)} of {len(chapters)} chapters have no sections in the textbook's "
                    f"contents to propose: {', '.join(no_source)} -- add their topics by hand")
    elif with_topics < len(chapters):
        note = why_uncovered(subject, grade) or (
            "no decomposition template covers these chapters -- use the LLM "
            "extraction route, or add topics by hand")
    return TemplateApplication(
        subject=subject, grade=grade, chapters=len(chapters),
        chapters_with_topics=with_topics,
        chapters_without_topics=len(chapters) - with_topics,
        topics_proposed=topics, subtopics_proposed=subtopics,
        provenance=provenance, note=note)


def _heading_note(heading_number: Optional[str]) -> Optional[str]:
    return f"NCERT section {heading_number}" if heading_number else None


@dataclass
class ChapterApproval:
    chapter_id: str
    runs_approved: int
    topics_created: int
    subtopics_created: int
    topic_ids: list[str] = field(default_factory=list)
    subtopic_ids: list[str] = field(default_factory=list)


def approve_chapter_templates(store: CurriculumStore, chapter_id: str, *,
                              approved_by: str) -> ChapterApproval:
    """"Approve all for this chapter" -- §29's review step for the case where
    the principal has nothing to correct, which is the common one for a template
    copied verbatim from a source they trust. Reviewing 58 chapters one proposal
    at a time is what stops a school finishing setup at all.

    Deliberately narrow: template runs only. An LLM run's proposals are a draft
    of something no human has ever checked and still go through the per-proposal
    review surface (`POST /extraction-runs/{id}/approve`).
    """
    from . import extraction as extraction_mod

    runs = [r for r in store.extraction_runs_for_chapter(chapter_id)
            if r.prompt_version.startswith(PROVENANCE_PREFIX)
            and r.status in ("pending", "reviewed")]
    out = ChapterApproval(chapter_id=chapter_id, runs_approved=0,
                          topics_created=0, subtopics_created=0)
    for run in runs:
        result = extraction_mod.approve_run(store, run.id, approved_by=approved_by)
        out.runs_approved += 1
        out.topics_created += result.topics_created
        out.subtopics_created += result.subtopics_created
        out.topic_ids.extend(result.topic_ids)
        out.subtopic_ids.extend(result.subtopic_ids)
    return out
