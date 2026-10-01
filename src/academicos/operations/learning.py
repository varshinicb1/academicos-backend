"""Learning progress (SA-3): per subject, chapter and topic, from homework,
tests and practice, with what to revise next and the term week by week.

Every marked answer is one row of evidence: the student, where it came from,
the question's subject, class, chapter and topic (the bank's taxonomy tags),
and the fraction of its marks the student got. Evidence arrives through the
knowledge store's listener (assessment/knowledge.py), so homework, finalized
answer sheets and practice all land here without each flow knowing about
this module. A re-marked sheet or homework replaces its rows and keeps the
time it was first sat, as the knowledge store does.

Mastery here is deliberately simple and explainable to a student and a
parent: a recency-weighted share of marks, pulled toward "unknown" when
there is little evidence. Nothing is compared with classmates (SA-3: no
leaderboards, no streak pressure).
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Iterable, Optional

LEARNING_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    grade INTEGER NOT NULL,
    chapter_id TEXT NOT NULL,
    topic_id TEXT,
    question_id TEXT NOT NULL,
    outcome REAL NOT NULL,
    occurred_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_student ON learning_evidence(student_id, subject, grade);
CREATE INDEX IF NOT EXISTS idx_learning_source ON learning_evidence(student_id, source_id);
"""

# An answer counts half as much after this many days: a chapter struggled
# with in June and practised well in August reads as mostly recovered.
HALF_LIFE_DAYS = 30.0
# Weight of the "unknown" prior (0.5). One right answer reads as 75%, not
# 100%; by ten answers the prior hardly matters.
PRIOR_WEIGHT = 1.0
MIN_EVIDENCE = 3            # fewer answers than this is "too early to say"
MASTERED = 0.85
DEVELOPING = 0.6
REVISE_LIMIT = 5
WEEKS = 16

KINDS = ("homework", "test", "practice")


def source_kind(source: Optional[str]) -> tuple[str, str]:
    """(kind, source id) for a knowledge-store source tag. Practice is
    recorded untagged; each practice submission is its own source."""
    if not source:
        return "practice", f"practice:{uuid.uuid4().hex[:12]}"
    if source.startswith("homework:"):
        return "homework", source
    if source.startswith(("sheet:", "marks:")):
        return "test", source
    return "other", source


def _chapter_and_topic(q: Any) -> tuple[str, Optional[str]]:
    chapter = q.taxonomy_chapter_id or (q.chapter_ids[0] if q.chapter_ids else None) or "unmapped"
    topic = next((t for t in q.topic_ids if t.startswith(chapter + "/")), None) if q.topic_ids else None
    return chapter, topic


@lru_cache(maxsize=64)
def _taxonomy(subject: str, grade: int) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """(chapter id -> name, chapter id -> book order, topic id -> name) from
    the bank's taxonomy for a class; empty where there is none."""
    from ..syllabus import cbse_syllabus
    path = cbse_syllabus._DATA_DIR / "taxonomy" / f"{subject.strip().replace(' ', '_')}_{grade}.json"
    if not path.exists():
        return {}, {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    chapters, order, topics = {}, {}, {}
    for i, c in enumerate(data.get("chapters", [])):
        chapters[c["id"]] = c["name"]
        order[c["id"]] = c.get("number") or i + 1
        for t in c.get("topics", []):
            topics[t["id"]] = t["name"]
    return chapters, order, topics


def _chapter_name(cid: str, names: dict[str, str]) -> str:
    if cid in names:
        return names[cid]
    if cid == "unmapped":
        return "Not yet sorted into a chapter"
    from ..assessment.chapters import chapter_name
    return chapter_name(cid) or cid.rsplit("/", 1)[-1].replace("-", " ").title()


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def score(rows: Iterable[dict], now: datetime) -> dict[str, Any]:
    """Mastery, share of marks, evidence count, last seen and status for a
    set of evidence rows."""
    rows = list(rows)
    if not rows:
        return {"mastery": None, "accuracy": None, "evidence_count": 0, "last_seen": None, "status": "not_started"}
    wsum = osum = 0.0
    for r in rows:
        age = max(0.0, (now - _parse(r["occurred_at"])).total_seconds() / 86400)
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        wsum += w
        osum += w * r["outcome"]
    mastery = (osum + 0.5 * PRIOR_WEIGHT) / (wsum + PRIOR_WEIGHT)
    accuracy = sum(r["outcome"] for r in rows) / len(rows)
    n = len(rows)
    status = ("early" if n < MIN_EVIDENCE else "mastered" if mastery >= MASTERED
              else "developing" if mastery >= DEVELOPING else "needs_work")
    return {"mastery": round(mastery, 4), "accuracy": round(accuracy, 4), "evidence_count": n,
            "last_seen": max(r["occurred_at"] for r in rows), "status": status}


class LearningMixin:
    """Added to OperationsStore; uses its _fetchall/_commit and connection."""

    def record_learning(self, student_id: str, results: list[tuple[Any, Any]], source: Optional[str]) -> int:
        """Record marked answers as evidence. A tagged source (a sheet, a
        homework) replaces its earlier rows and keeps when it was first sat."""
        kind, source_id = source_kind(source)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for q, ev in results:
            chapter, topic = _chapter_and_topic(q)
            outcome = (float(ev.awarded_marks) / ev.max_marks) if ev.max_marks else 0.0
            rows.append((student_id, kind, source_id, q.subject, int(q.grade), chapter, topic, q.id,
                         round(max(0.0, min(1.0, outcome)), 4)))
        with self._conn_lock:
            first = self.conn.execute(
                "SELECT MIN(occurred_at) FROM learning_evidence WHERE student_id=? AND source_id=?",
                (student_id, source_id)).fetchone()[0]
            self.conn.execute("DELETE FROM learning_evidence WHERE student_id=? AND source_id=?",
                              (student_id, source_id))
            self.conn.executemany(
                "INSERT INTO learning_evidence (student_id, source_kind, source_id, subject, grade, chapter_id,"
                " topic_id, question_id, outcome, occurred_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [r + (first or now,) for r in rows])
            self._commit()
        return len(rows)

    def learning_evidence(self, student_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM learning_evidence WHERE student_id=? ORDER BY occurred_at",
                              (student_id,))

    def learning_progress(self, student_id: str, *, now: Optional[datetime] = None,
                          term_start: Optional[date] = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        rows = self.learning_evidence(student_id)
        by_subject: dict[tuple[str, int], list[dict]] = {}
        for r in rows:
            by_subject.setdefault((r["subject"], r["grade"]), []).append(r)

        subjects, candidates = [], []
        for (subject, grade), srows in sorted(by_subject.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            names, order, topic_names = _taxonomy(subject, grade)
            by_chapter: dict[str, list[dict]] = {}
            for r in srows:
                by_chapter.setdefault(r["chapter_id"], []).append(r)
            chapter_ids = sorted(set(names) | set(by_chapter),
                                 key=lambda c: (c == "unmapped", order.get(c, 10_000), c))
            chapters = []
            for cid in chapter_ids:
                crows = by_chapter.get(cid, [])
                topics = []
                by_topic: dict[str, list[dict]] = {}
                for r in crows:
                    if r["topic_id"]:
                        by_topic.setdefault(r["topic_id"], []).append(r)
                for tid, trows in sorted(by_topic.items()):
                    t = {"topic_id": tid, "topic_name": topic_names.get(tid) or tid.rsplit("/", 1)[-1]
                         .replace("-", " ").capitalize(), **score(trows, now)}
                    topics.append(t)
                c = {"chapter_id": cid, "chapter_name": _chapter_name(cid, names), **score(crows, now),
                     "topics": topics}
                chapters.append(c)
                if cid == "unmapped" or not crows:
                    continue
                # Revise at the finest level the evidence has: a topic when
                # the answers were tagged to one, else the chapter.
                weak_topics = [t for t in topics if t["status"] in ("needs_work", "developing")]
                for t in weak_topics:
                    candidates.append({"subject": subject, "grade": grade, "chapter_id": cid,
                                       "chapter_name": c["chapter_name"], "topic_id": t["topic_id"],
                                       "topic_name": t["topic_name"], **_revise(t, now)})
                if not weak_topics and c["status"] in ("needs_work", "developing"):
                    candidates.append({"subject": subject, "grade": grade, "chapter_id": cid,
                                       "chapter_name": c["chapter_name"], "topic_id": None, "topic_name": None,
                                       **_revise(c, now)})
            subjects.append({"subject": subject, "grade": grade, **score(srows, now), "chapters": chapters})

        candidates.sort(key=lambda x: (x["mastery"], x["last_seen"]))
        return {
            "student_id": student_id,
            "as_of": now.isoformat(timespec="seconds"),
            "subjects": subjects,
            "revise_next": candidates[:REVISE_LIMIT],
            "weekly": _weekly(rows, now, term_start),
            "sources": {k: len({r["source_id"] for r in rows if r["source_kind"] == k}) for k in KINDS},
        }


def _revise(unit: dict, now: datetime) -> dict:
    days = (now - _parse(unit["last_seen"])).days
    when = "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"
    return {"mastery": unit["mastery"], "last_seen": unit["last_seen"],
            "reason": (f"{round(unit['accuracy'] * 100)}% of marks across {unit['evidence_count']} "
                       f"answer{'s' if unit['evidence_count'] != 1 else ''}; last practised {when}")}


def _weekly(rows: list[dict], now: datetime, term_start: Optional[date]) -> list[dict]:
    """Share of marks per week (Monday start), from the term's start or the
    last WEEKS weeks, whichever is later. A week with no answers is listed
    with none, so a gap reads as a gap and not as a drop."""
    today = now.date()
    this_monday = today - timedelta(days=today.weekday())
    first = this_monday - timedelta(weeks=WEEKS - 1)
    if term_start is not None:
        first = max(first, term_start - timedelta(days=term_start.weekday()))
    buckets: dict[date, list[float]] = {}
    for r in rows:
        d = _parse(r["occurred_at"]).date()
        monday = d - timedelta(days=d.weekday())
        if monday >= first:
            buckets.setdefault(monday, []).append(r["outcome"])
    out, week = [], first
    while week <= this_monday:
        vals = buckets.get(week, [])
        out.append({"week_start": week.isoformat(), "answered": len(vals),
                    "average": round(sum(vals) / len(vals), 4) if vals else None})
        week += timedelta(weeks=1)
    return out
