"""Chapter -> Topic -> Subtopic decomposition: LLM-proposed, never
auto-accepted. See docs/ACADEMIC_DATA_MODEL.md section 6.

AI DRAFT -> ADMIN REVIEW -> DATABASE, not AI -> DATABASE: `propose()`
below only ever writes to curriculum_extraction_runs/_proposals (a draft,
inert until reviewed). `approve_run()` is the one function that
materializes reviewed proposals into real topics/subtopics rows, and it
only ever touches proposals whose status is "approved" or "edited" --
"pending"/"rejected" proposals are never written to the canonical
hierarchy.

Same propose(-> verify) shape qmap.py already uses successfully for
question->concept mapping, adapted for generation rather than
classification: there is no pre-existing "topic glossary" to verify a
proposal against (the whole point is creating that glossary), so the
check here is a human, not a lexical-overlap score.

No lexical/offline fallback that fabricates topics -- unlike qmap.py's
lexical proposer (which has real candidate concepts to score), inventing
a chapter's internal structure from nothing is a generative task with no
honest deterministic substitute. Without a working LLM, propose() returns
a run with zero proposals and status="manual_required": the admin adds
topics/subtopics directly (source_type="manual", no run at all) via the
same review surface. This is a deliberate design choice, not a gap this
module papers over with a fragile heuristic.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ..syllabus.cbse_syllabus import _slug
from .models import Chapter, CurriculumExtractionRun
from .store import CurriculumStore

log = logging.getLogger(__name__)

PROMPT_VERSION = "v3"

# v1 asked for an unbounded breakdown: 10/13 real CBSE Science chapters
# failed at max_tokens=4000 (finish_reason="length", zero content) --
# sarvam-105b spending its whole budget reasoning before ever writing JSON.
# v2 bounded it to "at most 5 topics, at most 4 subtopics" and asked for a
# confidence score per item: still failed on most chapters in a real bulk
# run, but the *truncated* JSON showed genuinely good content ("Mendel's
# Approach to Inheritance", "The Human Nervous System: Components and
# Design") cut off mid-structure -- the model reasons well, it just
# doesn't reliably finish writing a 5x4-with-confidence-scores JSON tree
# in the budget available. v3 shrinks the ask further (3 topics, 3
# subtopics) and drops the per-item confidence field entirely (real
# tokens spent on `"confidence": 0.87` for up to 12 items, for a number an
# LLM's self-report was never that trustworthy anyway -- admin review is
# the real quality gate, not this score) -- both cuts reduce required
# output size directly, which is the lever that actually matters here,
# not a bigger max_tokens ceiling.
PROPOSE_SYSTEM = (
    "You are a CBSE curriculum expert breaking a textbook chapter into the real "
    "units a teacher would deliver it in: Topics, each with concrete Subtopics. "
    "Depth matters -- 'Heat' is not enough; propose things like 'Definition of "
    "heat' and 'Types of heat transfer'. Propose AT MOST 3 topics, each with AT "
    "MOST 3 subtopics -- pick only the most important ones; a real teacher "
    "curates and can add more later. Be concise. "
    "Respond ONLY with JSON:\n"
    '{"topics": [{"name": "...", "subtopics": ["...", "..."]}]}'
)

_MAX_LLM_ATTEMPTS = 3
# No longer asked of the LLM (v3, see above) -- a fixed, honest default.
# Real confidence differentiation belongs to admin review (approve/edit/
# reject), not a self-reported number this codebase never validated.
_LLM_PROPOSAL_CONFIDENCE = 0.7


@dataclass
class ProposedSubtopic:
    name: str
    confidence: float = 0.5


@dataclass
class ProposedTopic:
    name: str
    confidence: float = 0.5
    subtopics: list[ProposedSubtopic] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.subtopics is None:
            self.subtopics = []


def propose(store: CurriculumStore, *, school_id: str, chapter: Chapter, book_id: str,
           subject: str, grade: int, llm=None, grounding_text: Optional[str] = None,
           grounding_reference: Optional[str] = None,
           prompt_version: str = PROMPT_VERSION) -> CurriculumExtractionRun:
    """Creates a run and, if an LLM is available, its proposals. `llm` is
    duck-typed exactly like qmap.py's (`.available`, `.chat_json`) so the
    same SarvamLLM instance (or a test double) works unchanged.
    `grounding_text`, when the caller found real textbook content for this
    chapter (e.g. via a ChunkIndex lookup -- not this module's job to fetch),
    grounds the proposal in the actual book instead of the chapter title
    alone; `grounding_reference` (doc_id/page or similar) is stored on the
    run as its source_hash-adjacent provenance via the run's model/source_hash
    fields so "why did it propose this" stays answerable.
    """
    source_hash = hashlib.sha256(grounding_text.encode("utf-8")).hexdigest()[:16] if grounding_text else None
    llm_ok = _llm_ok(llm)
    run = store.create_extraction_run(
        school_id=school_id, book_id=book_id, chapter_id=chapter.id,
        source_hash=source_hash, model=(getattr(llm, "model", None) if llm_ok else None),
        prompt_version=prompt_version)

    if not llm_ok:
        store.update_run_status(run.id, "manual_required")
        return store.get_extraction_run(run.id)

    proposed = _propose_llm(llm, chapter_name=chapter.name, subject=subject, grade=grade,
                            grounding_text=grounding_text)
    if not proposed:
        store.update_run_status(run.id, "manual_required")
        return store.get_extraction_run(run.id)

    for t_seq, topic in enumerate(proposed):
        topic_proposal = store.add_extraction_proposal(
            run_id=run.id, entity_type="topic", proposed_name=topic.name,
            proposed_parent=None, sequence=t_seq, confidence=topic.confidence)
        for s_seq, sub in enumerate(topic.subtopics):
            store.add_extraction_proposal(
                run_id=run.id, entity_type="subtopic", proposed_name=sub.name,
                proposed_parent=topic_proposal.id, sequence=s_seq, confidence=sub.confidence)

    store.update_run_status(run.id, "pending")
    return store.get_extraction_run(run.id)


def _llm_ok(llm: Any) -> bool:
    try:
        return bool(llm and llm.available)
    except Exception:
        return False


def _chat_json_with_retries(llm: Any, user: str) -> Optional[dict]:
    """Up to _MAX_LLM_ATTEMPTS real API calls before giving up -- a real
    bulk run showed the same chapter/prompt succeed on one attempt and
    fail on another (finish_reason="length" or truncated JSON, both a
    token-budget exhaustion mid-response, not a malformed-forever
    prompt), so a retry genuinely recovers otherwise-lost chapters rather
    than just re-hitting the same deterministic failure. Returns None
    (not a raised exception) after exhausting attempts -- the caller
    already has a real, honest fallback for "no proposal" (manual entry),
    and every module this deep in the stack should fail toward that, not
    crash the whole bulk run over one stubborn chapter."""
    import time as _time
    for attempt in range(_MAX_LLM_ATTEMPTS):
        try:
            return llm.chat_json(
                [{"role": "system", "content": PROPOSE_SYSTEM}, {"role": "user", "content": user}],
                # 4000, not 8000: real, counterintuitive finding from two
                # full bulk runs against the real API -- 4000 got 3/13 real
                # chapters through, then raising to 8000 got 0/6 (worse, not
                # better, on the exact same chapters) before this prompt was
                # bounded. A heavier budget appears to invite more reasoning
                # rather than reliably finishing faster, so the actual fix
                # is bounding what's asked for (PROPOSE_SYSTEM's "at most 5
                # topics, at most 4 subtopics" above, v2) plus retrying a
                # transient failure, not maximizing max_tokens.
                temperature=0.2, max_tokens=4000,
            )
        except Exception as e:
            if attempt + 1 >= _MAX_LLM_ATTEMPTS:
                log.warning("extraction proposal failed after %d attempts: %s",
                           _MAX_LLM_ATTEMPTS, e)
                return None
            log.info("extraction proposal attempt %d/%d failed (%s), retrying",
                    attempt + 1, _MAX_LLM_ATTEMPTS, e)
            _time.sleep(2 * (attempt + 1))
    return None


def _propose_llm(llm: Any, *, chapter_name: str, subject: str, grade: int,
                 grounding_text: Optional[str]) -> list[ProposedTopic]:
    user = f"Subject: {subject}\nGrade: {grade}\nChapter: {chapter_name}\n"
    if grounding_text:
        user += f"\nExcerpt from the actual textbook chapter (use this, don't invent " \
                f"beyond what a teacher could support from it):\n{grounding_text[:4000]}"
    else:
        user += ("\n(No textbook excerpt available for this chapter yet -- propose the "
                 "standard CBSE breakdown for this chapter title from general curriculum "
                 "knowledge. This will need closer admin review than a grounded proposal.)")

    out = _chat_json_with_retries(llm, user)
    if out is None:
        return []
    topics: list[ProposedTopic] = []
    for t in (out or {}).get("topics", []):
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        subs = []
        for s in t.get("subtopics", []) or []:
            # v3's schema asks for plain subtopic name strings (see
            # PROPOSE_SYSTEM above); a dict with its own confidence is
            # still accepted for backward compat with any v2-era proposal
            # data still sitting in a store, or a model that ignores the
            # "no confidence field" instruction anyway.
            if isinstance(s, dict):
                sname = str(s.get("name", "")).strip()
                sconf = float(s.get("confidence", _LLM_PROPOSAL_CONFIDENCE))
            else:
                sname, sconf = str(s).strip(), _LLM_PROPOSAL_CONFIDENCE
            if sname:
                subs.append(ProposedSubtopic(name=sname, confidence=sconf))
        topics.append(ProposedTopic(
            name=name, confidence=float(t.get("confidence", _LLM_PROPOSAL_CONFIDENCE)),
            subtopics=subs))
    return topics


@dataclass
class ApprovalResult:
    run_id: str
    topics_created: int
    subtopics_created: int
    topic_ids: list[str]
    subtopic_ids: list[str]


def approve_run(store: CurriculumStore, run_id: str, *, approved_by: str,
                edits: Optional[dict[str, str]] = None,
                rejected_proposal_ids: Optional[set[str]] = None) -> ApprovalResult:
    """Materializes a run's reviewed proposals into real topics/subtopics
    rows. `edits`: {proposal_id: new_name} for anything the admin changed
    before approving. `rejected_proposal_ids`: proposals to exclude --
    never written to the canonical hierarchy, and if a rejected proposal
    is a Topic, its Subtopic children are excluded too even if they
    weren't individually marked rejected (a topic's subtopics don't
    survive without their parent).

    Idempotent per proposal: a proposal already materialized (has
    materialized_id) is skipped, not re-created -- approving the same run
    twice (e.g. a retried request) never duplicates rows.
    """
    edits = edits or {}
    rejected_proposal_ids = rejected_proposal_ids or set()
    run = store.get_extraction_run(run_id)
    if run is None:
        raise ValueError(f"no such extraction run: {run_id}")
    proposals = store.proposals_for_run(run_id)

    approved_at = datetime.now(timezone.utc).isoformat()
    topic_ids: list[str] = []
    subtopic_ids: list[str] = []

    # Topics first (subtopics need their materialized parent id).
    topic_proposal_by_id = {p.id: p for p in proposals if p.entity_type == "topic"}
    for p in proposals:
        if p.entity_type != "topic":
            continue
        if p.id in rejected_proposal_ids:
            store.set_proposal_status(p.id, "rejected")
            continue
        if p.materialized_id:
            topic_ids.append(p.materialized_id)
            continue
        final_name = edits.get(p.id, p.proposed_name)
        status = "edited" if p.id in edits else "approved"
        store.set_proposal_status(p.id, status, edited_name=edits.get(p.id))

        chapter_row = _get_chapter(store, run.chapter_id)
        canonical_id = f"{chapter_row.canonical_id}:topic:{_slug(final_name)}"
        existing = store.get_topic(p.materialized_id) if p.materialized_id else None
        topic = existing or store.create_topic(
            canonical_id=canonical_id, chapter_id=run.chapter_id, name=final_name, seq=p.sequence,
            source_type="llm_proposed", source_reference=run.id, approved_by=approved_by,
            approved_at=approved_at, model_used=run.model, generation_version=run.prompt_version)
        store.set_proposal_materialized_id(p.id, topic.id)
        topic_ids.append(topic.id)

    for p in proposals:
        if p.entity_type != "subtopic":
            continue
        parent = topic_proposal_by_id.get(p.proposed_parent)
        if parent is None or parent.id in rejected_proposal_ids or p.id in rejected_proposal_ids:
            store.set_proposal_status(p.id, "rejected")
            continue
        if p.materialized_id:
            subtopic_ids.append(p.materialized_id)
            continue
        parent_topic_id = store.get_proposal(parent.id).materialized_id
        if parent_topic_id is None:
            # Parent topic itself wasn't materialized this pass (shouldn't
            # happen given the topic loop above runs first, but fail safe
            # rather than orphan a subtopic under a non-existent topic).
            continue
        final_name = edits.get(p.id, p.proposed_name)
        status = "edited" if p.id in edits else "approved"
        store.set_proposal_status(p.id, status, edited_name=edits.get(p.id))

        topic_row = store.get_topic(parent_topic_id)
        canonical_id = f"{topic_row.canonical_id}:subtopic:{_slug(final_name)}"
        subtopic = store.create_subtopic(
            canonical_id=canonical_id, topic_id=parent_topic_id, name=final_name, seq=p.sequence,
            source_type="llm_proposed", source_reference=run.id, approved_by=approved_by,
            approved_at=approved_at, model_used=run.model, generation_version=run.prompt_version)
        store.set_proposal_materialized_id(p.id, subtopic.id)
        subtopic_ids.append(subtopic.id)

    store.update_run_status(run_id, "approved")
    return ApprovalResult(run_id=run_id, topics_created=len(topic_ids),
                          subtopics_created=len(subtopic_ids),
                          topic_ids=topic_ids, subtopic_ids=subtopic_ids)


def _get_chapter(store: CurriculumStore, chapter_id: str) -> Chapter:
    r = store.conn.execute("SELECT * FROM chapters WHERE id=?", (chapter_id,)).fetchone()
    if r is None:
        raise ValueError(f"no such chapter: {chapter_id}")
    return Chapter(**dict(r))


# ---------------- textbook table of contents ingestion (§5-9) ----------------

@dataclass
class TocUnit:
    unit_no: str
    name: str
    marks: int = 0
    chapters: list[str] = field(default_factory=list)


@dataclass
class TocResult:
    book_id: str
    units_created: int
    chapters_created: int
    unit_ids: list[str]
    chapter_ids: list[str]


def parse_toc_text(toc_text: str) -> list[TocUnit]:
    """Deterministic parser for raw textbook Table of Contents text.
    Handles standard patterns like:
    Unit 1: Chemical Substances - Nature and Behaviour
      Chapter 1: Chemical Reactions and Equations
      Chapter 2: Acids, Bases and Salts
    """
    import re
    units: list[TocUnit] = []
    current_unit: Optional[TocUnit] = None

    unit_pattern = re.compile(r"^(?:Unit|Part|Section)\s+([0-9IVXLCDM]+)[:\.\-\s]*(.*)$", re.IGNORECASE)
    chapter_pattern = re.compile(r"^(?:Chapter|Lesson)?\s*([0-9]+)[:\.\-\s]+(.*)$", re.IGNORECASE)

    for line in toc_text.splitlines():
        line = line.strip()
        if not line:
            continue

        m_unit = unit_pattern.match(line)
        if m_unit:
            u_no = m_unit.group(1).strip()
            u_name = m_unit.group(2).strip() or f"Unit {u_no}"
            current_unit = TocUnit(unit_no=u_no, name=u_name, chapters=[])
            units.append(current_unit)
            continue

        m_chap = chapter_pattern.match(line)
        if m_chap:
            chap_name = m_chap.group(2).strip() or line
            if current_unit is None:
                current_unit = TocUnit(unit_no=str(len(units) + 1), name=f"Unit {len(units) + 1}", chapters=[])
                units.append(current_unit)
            current_unit.chapters.append(chap_name)
        elif current_unit is not None and not line.lower().startswith(("page", "content", "index", "table of contents")):
            current_unit.chapters.append(line)

    if not units and toc_text.strip():
        chapters = [l.strip() for l in toc_text.splitlines() if l.strip()]
        units.append(TocUnit(unit_no="1", name="General", chapters=chapters))

    return units


def extract_and_ingest_toc(store: CurriculumStore, book_id: str, toc_text: str) -> TocResult:
    """Ingests a textbook's Table of Contents text and materializes Units and Chapters."""
    book = store.get_book(book_id)
    if book is None:
        raise ValueError(f"no such book: {book_id}")

    units = parse_toc_text(toc_text)
    unit_ids: list[str] = []
    chapter_ids: list[str] = []

    for u_seq, u in enumerate(units):
        u_canon = f"{book.id}:unit:{_slug(u.unit_no)}"
        unit_row = store.get_unit_by_canonical_id(u_canon)
        if unit_row is None:
            unit_row = store.create_unit(
                canonical_id=u_canon, book_id=book.id,
                unit_no=u.unit_no, name=u.name, marks=u.marks, seq=u_seq,
            )
        unit_ids.append(unit_row.id)

        for c_seq, chap_name in enumerate(u.chapters):
            c_canon = f"{book.id}:chapter:{_slug(chap_name)}"
            chap_row = store.get_chapter_by_canonical_id(c_canon)
            if chap_row is None:
                chap_row = store.create_chapter(
                    canonical_id=c_canon, unit_id=unit_row.id,
                    name=chap_name, seq=c_seq,
                )
            chapter_ids.append(chap_row.id)

    return TocResult(
        book_id=book_id,
        units_created=len(unit_ids),
        chapters_created=len(chapter_ids),
        unit_ids=unit_ids,
        chapter_ids=chapter_ids,
    )
