"""The question bank, as a school sees it: coverage class by class, subject by
subject, chapter by chapter and topic by topic, and a review queue for the
answers no board published (qbank_engine.CHECKED_PROVENANCE).

The bank is shared by every school. A school's reviewer -- its principal, or a
teacher it made a `qbank_review` admin -- approves a checked answer (for that
school it then reads `teacher_verified`) or rejects the question (that school's
papers, homework and practice never draw it again, `rejected_for`). One
school's decision never changes what another school is served.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field

from ..assessment.auth_routes import require_admin, require_staff
from ..assessment.users import User
from ..curriculum.schemas import Camel, CamelRequest

router = APIRouter(prefix="/api/v1/qbank")

QUESTION_REVIEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS question_reviews (
    school_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    note TEXT,
    reviewer TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (school_id, question_id)
);
"""


class QuestionReviewsMixin:
    """Added to OperationsStore."""

    def review_decisions(self, school_id: str) -> dict[str, dict]:
        return {r["question_id"]: r for r in self._fetchall(
            "SELECT * FROM question_reviews WHERE school_id=?", (school_id,))}

    def decide_question(self, school_id: str, question_id: str, decision: str, note: Optional[str],
                        reviewer: str) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._exec("INSERT INTO question_reviews (school_id, question_id, decision, note, reviewer, decided_at) "
                   "VALUES (?,?,?,?,?,?) ON CONFLICT(school_id, question_id) DO UPDATE SET "
                   "decision=excluded.decision, note=excluded.note, reviewer=excluded.reviewer, "
                   "decided_at=excluded.decided_at", (school_id, question_id, decision, note, reviewer, now))
        self._commit()
        return self._fetchone("SELECT * FROM question_reviews WHERE school_id=? AND question_id=?",
                              (school_id, question_id))


def rejected_for(school_id: Optional[str]) -> frozenset[str]:
    """The questions a school's reviewer rejected; none when there is no store."""
    if not school_id:
        return frozenset()
    try:
        from .routes import _cfg, store
        if _cfg is None:
            return frozenset()
        return frozenset(q for q, r in store().review_decisions(school_id).items() if r["decision"] == "rejected")
    except Exception:  # noqa: BLE001 - a review lookup must never stop a paper being made
        return frozenset()


def usable(items, school_id: Optional[str]) -> list:
    """`items` (anything with an `.id`) less the questions `school_id`'s
    reviewer rejected, in order. Every place that draws questions for a school
    calls this with the caller's own school: the pool and the mapped banks are
    shared by every school and cached, so the rejection is applied at the draw,
    never stored in a shared list (review of #54: an ambient per-request value
    set in a sync dependency never reached the endpoint, and a cache filtered
    for one school would have served every school)."""
    items = list(items)
    rejected = rejected_for(school_id)
    return [q for q in items if q.id not in rejected] if rejected else items


def bank_usable(bank, school_id: Optional[str]):
    """A `paper_edit.Bank` less the school's rejected questions: the cached
    bank itself is never changed."""
    rejected = rejected_for(school_id)
    if not rejected:
        return bank
    qs = tuple(q for q in bank.questions if q.id not in rejected)
    return type(bank)(qs, {q.id: q for q in qs}, frozenset(bank.keyed - rejected))


def _bank():
    from ..assessment import qbank_routes
    if qbank_routes._bank is None:
        raise HTTPException(503, "the question bank is not loaded")
    return qbank_routes._bank


def _topic(rec: dict) -> str:
    meta = rec.get("metadata") or {}
    return str(meta.get("topicTitle") or rec.get("topic") or "").strip() or "(no topic yet)"


# ---------------------------------------------------------------- coverage


class TopicCount(Camel):
    topic: str
    questions: int
    published: int
    checked: int


class ChapterCount(Camel):
    chapter_id: str
    chapter_name: str
    questions: int
    published: int
    checked: int
    by_type: dict[str, int]
    by_difficulty: dict[str, int]
    topics: list[TopicCount]


class SubjectCoverage(Camel):
    grade: int
    subject: str
    questions: int
    published: int
    checked: int
    generated: int
    chapters: list[ChapterCount]
    thin_topics: list[str] = Field(default_factory=list)      # "Chapter: topic" with fewer than 3 questions


class CoverageMap(Camel):
    grades: list[int]
    subjects: list[SubjectCoverage]
    pending_review: int


@router.get("/coverage-map", response_model=CoverageMap)
def coverage_map(grade: Optional[int] = Query(None, ge=1, le=12), subject: Optional[str] = None,
                 current: User = Depends(require_staff)) -> CoverageMap:
    """What the bank holds for each class and subject, chapter by chapter and
    topic by topic, split by how far each answer can be trusted: published
    (CBSE's or NCERT's own) or checked (grounded in the textbook, two agreeing
    solves). Staff read it; `pendingReview` counts the checked answers this
    school has not reviewed yet."""
    from ..assessment.qbank_engine import key_tier
    from .routes import store
    bank = _bank()
    decided = store().review_decisions(current.school_id)
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    grades: set[int] = set()
    pending = 0
    for rec in bank.records:
        g = int(rec.get("grade") or 0)
        grades.add(g)
        if grade is not None and g != grade:
            continue
        if subject and str(rec.get("subject") or "").lower() != subject.lower():
            continue
        groups[(g, str(rec.get("subject") or ""))].append(rec)
        if key_tier(rec) == "checked" and str(rec.get("id")) not in decided:
            pending += 1
    out = []
    for (g, subj), recs in sorted(groups.items()):
        chapters: dict[str, list[dict]] = defaultdict(list)
        names: dict[str, str] = {}
        for r in recs:
            cid = bank.chapter_of(r) or "unmapped"
            chapters[cid].append(r)
            name = (r.get("metadata") or {}).get("chapterName")
            if name:
                names.setdefault(cid, str(name))
        rows, thin = [], []
        for cid, rs in sorted(chapters.items(), key=lambda kv: kv[0]):
            tiers = Counter(key_tier(r) for r in rs)
            topics: dict[str, list[dict]] = defaultdict(list)
            for r in rs:
                topics[_topic(r)].append(r)
            trows = []
            for t, trs in sorted(topics.items()):
                tt = Counter(key_tier(r) for r in trs)
                trows.append(TopicCount(topic=t, questions=len(trs), published=tt["published"], checked=tt["checked"]))
                if len(trs) < 3 and t != "(no topic yet)":
                    thin.append(f"{names.get(cid, cid)}: {t}")
            rows.append(ChapterCount(
                chapter_id=cid, chapter_name=names.get(cid, cid.replace("-", " ").title()), questions=len(rs),
                published=tiers["published"], checked=tiers["checked"],
                by_type=dict(Counter(str(r.get("type")) for r in rs)),
                by_difficulty=dict(Counter(str(r.get("difficulty")) for r in rs)), topics=trows))
        all_tiers = Counter(key_tier(r) for r in recs)
        out.append(SubjectCoverage(grade=g, subject=subj, questions=len(recs), published=all_tiers["published"],
                                   checked=all_tiers["checked"],
                                   generated=sum(1 for r in recs if r.get("source") == "ai_generated"),
                                   chapters=rows, thin_topics=thin[:50]))
    return CoverageMap(grades=sorted(grades), subjects=out, pending_review=pending)


# ---------------------------------------------------------------- review queue


class ReviewItem(Camel):
    id: str
    grade: int
    subject: str
    chapter: str
    topic: str
    type: str
    marks: int
    difficulty: Optional[str] = None
    source: str
    provenance: str
    stem: str
    model_answer: str
    marking_points: list[dict]
    evidence: list[dict]
    decision: Optional[str] = None
    note: Optional[str] = None


class ReviewPage(Camel):
    items: list[ReviewItem]
    total: int
    counts: dict[str, int]


class DecisionRequest(CamelRequest):
    decision: Literal["approved", "rejected"]
    note: Optional[str] = Field(default=None, max_length=1000)


def _item(bank, rec: dict, decided: dict) -> ReviewItem:
    s = rec.get("answerScheme") or {}
    d = decided.get(str(rec.get("id"))) or {}
    return ReviewItem(
        id=str(rec.get("id")), grade=int(rec.get("grade") or 0), subject=str(rec.get("subject") or ""),
        chapter=str((rec.get("metadata") or {}).get("chapterName") or bank.chapter_of(rec) or ""),
        topic=_topic(rec), type=str(rec.get("type") or ""), marks=int(rec.get("marks") or 0),
        difficulty=rec.get("difficulty"), source=str(rec.get("source") or ""),
        provenance=str(s.get("provenance") or ""), stem=str(rec.get("stem") or ""),
        model_answer=str(s.get("modelAnswer") or ""),
        marking_points=[{"description": p.get("description"), "marks": p.get("marks")}
                        for p in s.get("markingPoints") or []],
        evidence=list((s.get("metadata") or {}).get("evidence") or []),
        decision=d.get("decision"), note=d.get("note"))


@router.get("/review-queue", response_model=ReviewPage)
def review_queue(grade: Optional[int] = Query(None, ge=1, le=12), subject: Optional[str] = None,
                 chapter: Optional[str] = None,
                 status: Literal["pending", "approved", "rejected", "all"] = "pending",
                 limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                 reviewer: User = Depends(require_admin("qbank_review"))) -> ReviewPage:
    """The checked answers (grounded in the textbook, two agreeing solves) for
    this school's reviewer to approve or reject, with the textbook sentences
    each answer rests on."""
    from ..assessment.qbank_engine import key_tier
    from .routes import store
    bank = _bank()
    decided = store().review_decisions(reviewer.school_id)
    rows = []
    for rec in bank.records:
        if key_tier(rec) != "checked":
            continue
        if grade is not None and int(rec.get("grade") or 0) != grade:
            continue
        if subject and str(rec.get("subject") or "").lower() != subject.lower():
            continue
        if chapter and bank.chapter_of(rec) != chapter:
            continue
        rows.append(rec)
    counts = Counter((decided.get(str(r.get("id"))) or {}).get("decision") or "pending" for r in rows)
    if status != "all":
        rows = [r for r in rows if ((decided.get(str(r.get("id"))) or {}).get("decision") or "pending") == status]
    rows.sort(key=lambda r: (int(r.get("grade") or 0), str(r.get("subject")), bank.chapter_of(r) or "",
                             str(r.get("id"))))
    return ReviewPage(items=[_item(bank, r, decided) for r in rows[offset:offset + limit]], total=len(rows),
                      counts={k: counts.get(k, 0) for k in ("pending", "approved", "rejected")})


@router.post("/review/{question_id}", response_model=ReviewItem)
def decide(question_id: str, req: DecisionRequest,
           reviewer: User = Depends(require_admin("qbank_review"))) -> ReviewItem:
    """Approve a checked answer for this school (it then reads teacher_verified)
    or reject the question (this school's papers, homework and practice never
    draw it again). Recorded in the audit log."""
    from ..assessment.audit_log import get_audit_log
    from ..curriculum import routes as cr
    from .routes import store
    bank = _bank()
    rec = bank.get(question_id)
    if rec is None:
        raise HTTPException(404, "no such question in the bank")
    store().decide_question(reviewer.school_id, question_id, req.decision, req.note, reviewer.id)
    get_audit_log(cr._cfg.data_root).append("question_reviewed", actor=reviewer.id, details={
        "schoolId": reviewer.school_id, "questionId": question_id, "decision": req.decision})
    return _item(bank, rec, store().review_decisions(reviewer.school_id))
