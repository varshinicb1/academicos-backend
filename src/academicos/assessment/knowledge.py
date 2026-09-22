"""Pillar 3 — Learning Intelligence.

Turns evaluated answers into knowledge, not marks:

    Student -> Question -> Concept -> Learning Outcome -> Bloom -> Mastery

Each evaluated answer becomes an `Interaction` on the student's `LearnerModel`
(the algorithms library's existing learner state), with partial credit carried
through as a fractional outcome so a 3/5 answer is not recorded as simply
"wrong". Mastery is then computed by `KnowledgeMastery`, which already blends
accuracy, recency-weighted confidence, retention decay and higher-Bloom
application.

State is persisted per student so the loop survives restarts: one
`learner_models` row per student (student_id + JSONB payload), resolved
through postgres_kv.durable_table like every other JSONB store -- Cloud SQL
on GCP, Supabase on Render, local JSON files when neither is configured
(local dev). Losing it matters more than most stores: it is a student's
accumulated mastery history, not a single in-flight session.

2026-09-22: this store used to call Supabase's PostgREST API itself, reading
the Supabase env vars directly. When the GCP deploy dropped those secrets
(no school data outside GCP), it silently fell back to local JSON under
/app, which Cloud Run discards on every redeploy -- every student's history
would have vanished at each deploy with no error anywhere. Going through
durable_table puts it in Cloud SQL there; the rows already in Supabase are
copied by scripts/migrate_supabase_to_cloudsql.py (`learner_models` entry).

A live backend failure is not caught here: it raises SupabaseUnavailable
(PostgresUnavailable on Cloud SQL), which api/main.py answers with a 503.
Falling back to local JSON on failure would split one student's history
across two places.

Migration-order guard: that copy is a manual step, and deployed before it
the service did not fail -- every student read as a brand-new learner. So on
Cloud SQL an EMPTY learner_models table is treated as "not migrated" and
answers 503 (LearnerModelsNotMigrated) on read and write, until either a row
exists or ACOS_LEARNER_MODELS_ALLOW_EMPTY=1 says this is a new school with
no history. Writes are blocked too: one new row would satisfy the check and
hide that everyone else's history is missing. Checked once per process;
once satisfied it costs nothing.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..algorithms.learner_model import ConceptState, Interaction, LearnerModel
from ..algorithms.mastery import KnowledgeMastery
from .chapters import chapter_name
from .evaluate import Evaluation
from .postgres_kv import PostgresTable, durable_table
from .supabase_kv import SupabaseUnavailable
from .schemas import QuestionSchema

MASTERY_TARGET = 0.85   # the remediation loop drives toward this
WEAK_THRESHOLD = 0.55


@dataclass
class ConceptMasteryView:
    concept_id: str
    concept_name: str
    mastery: float
    accuracy: float
    retention: float
    confidence: float
    evidence_count: int
    confident: bool
    status: str
    misconceptions: list[str] = field(default_factory=list)

    @property
    def is_weak(self) -> bool:
        return self.mastery < WEAK_THRESHOLD


def sheet_source(assessment_id: str, student_id: str) -> str:
    """The tag on a finalized sheet's interactions (see record_sheet)."""
    return f"sheet:{assessment_id}:{student_id}"


def _ts_key(ts: str) -> float:
    """Order ISO timestamps by instant, not by string (offsets may differ)."""
    return datetime.fromisoformat(ts).timestamp()


def _status(m: float, confident: bool) -> str:
    if not confident:
        return "learning"
    if m >= MASTERY_TARGET:
        return "mastered"
    if m >= 0.7:
        return "proficient"
    if m >= WEAK_THRESHOLD:
        return "developing"
    return "needsReview"


class LearnerModelsNotMigrated(SupabaseUnavailable):
    """Cloud SQL's learner_models is empty and nobody said that is expected.
    A SupabaseUnavailable subclass so api/main.py answers 503 with this text."""

    def __init__(self) -> None:
        detail = ("learner_models in Cloud SQL is empty, so every student's mastery "
                  "history would read as a brand-new learner. Copy it with "
                  "scripts/migrate_supabase_to_cloudsql.py --only learner_models, or, "
                  "for a new school with no history, set "
                  "ACOS_LEARNER_MODELS_ALLOW_EMPTY=1")
        Exception.__init__(self, detail)  # bypass SupabaseUnavailable's wording
        self.table = "learner_models"
        self.op = "migration check"
        self.status = None
        self.body = detail


def _is_cloud_sql(table: Any) -> bool:
    """Only Cloud SQL is a migration destination (Render's Supabase table
    always held the rows). A function so tests can mark a fake as Cloud SQL."""
    return isinstance(table, PostgresTable)


class KnowledgeStore:
    """Per-student learner models -- in the `learner_models` durable table
    when one is configured, local JSON files otherwise. See module docstring."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._scorer = KnowledgeMastery()
        self._table = durable_table("learner_models")
        self._migration_checked = False
        self._migration_lock = threading.Lock()

    @property
    def _remote(self) -> bool:
        return self._table.enabled

    def _require_migrated(self) -> None:
        """See the module docstring's migration-order guard."""
        if self._migration_checked:
            return
        with self._migration_lock:
            if self._migration_checked:
                return
            if (not _is_cloud_sql(self._table)
                    or os.environ.get("ACOS_LEARNER_MODELS_ALLOW_EMPTY", "").strip() == "1"
                    or self._table.select(limit=1)):
                self._migration_checked = True
                return
        raise LearnerModelsNotMigrated()

    def _path(self, student_id: str) -> Path:
        # Injective sanitization: disallowed chars are percent-encoded rather
        # than dropped, so distinct student ids ("stu.dent" vs "student" vs
        # "stu%2Edent") never collide on the same file. Literal '%' is itself
        # encoded ('%25'), keeping the mapping one-to-one. Clean alnum/-/_ ids
        # are unchanged, so existing files for those students keep working.
        safe = "".join(
            c if (c.isalnum() or c in "-_") else f"%{ord(c):02x}"
            for c in student_id)
        return self.root / f"{safe}.json"

    def _read_raw(self, student_id: str) -> Optional[dict[str, Any]]:
        if self._remote:
            self._require_migrated()
            rows = self._table.select(student_id=student_id, limit=1)
            return rows[0]["payload"] if rows else None
        p = self._path(student_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def load(self, student_id: str) -> LearnerModel:
        model = LearnerModel(learner_id=student_id)
        raw = self._read_raw(student_id)
        if raw is None:
            return model
        for cid, cs in raw.get("concepts", {}).items():
            state = ConceptState(
                concept_id=cid, attempts=cs.get("attempts", 0),
                correct=cs.get("correct", 0), last_seen=cs.get("last_seen"),
            )
            state.history = [Interaction(**i) for i in cs.get("history", [])]
            model.concepts[cid] = state
        model.created_at = raw.get("created_at", model.created_at)
        model.updated_at = raw.get("updated_at", model.updated_at)
        return model

    def save(self, model: LearnerModel) -> None:
        payload = {
            "learner_id": model.learner_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "concepts": {
                cid: {
                    "attempts": s.attempts, "correct": s.correct, "last_seen": s.last_seen,
                    "history": [i.__dict__ for i in s.history],
                }
                for cid, s in model.concepts.items()
            },
        }
        if self._remote:
            self._require_migrated()
            self._table.upsert({"student_id": model.learner_id, "payload": payload},
                               on_conflict="student_id")
        else:
            self._path(model.learner_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _interactions(results: list[tuple[QuestionSchema, Evaluation]],
                      source: Optional[str]) -> list[Interaction]:
        now = datetime.now(timezone.utc).isoformat()
        out: list[Interaction] = []
        for question, ev in results:
            # The teacher's award is the ground truth, whatever the scanner
            # classified the verdict as: a teacher correcting a "blank" to
            # full marks must not have that correction dropped to 0.0. A
            # genuinely blank (awarded 0) still lands at 0.0.
            outcome = (ev.awarded_marks / ev.max_marks) if ev.max_marks else 0.0
            concepts = question.chapter_ids or ["unmapped"]
            for cid in concepts:
                out.append(Interaction(
                    concept_id=cid, kind="answer", outcome=round(outcome, 4),
                    bloom=question.bloom_level, ts=now,
                    difficulty={"easy": 0.3, "medium": 0.6, "hard": 0.9}.get(
                        question.difficulty, 0.5),
                    source=source,
                ))
        return out

    def record_evaluations(self, student_id: str,
                           results: list[tuple[QuestionSchema, Evaluation]]) -> LearnerModel:
        """Fold a marked paper into the student's knowledge state. Appends:
        calling it twice counts the paper twice. A finalized sheet goes
        through record_sheet() instead."""
        model = self.load(student_id)
        for i in self._interactions(results, None):
            model.observe(i)
        self.save(model)
        return model

    # -- a finalized sheet: replaceable as a unit ---------------------------
    #
    # A finalized sheet can be corrected with a reason and finalized again
    # (grade_lock.py). Before 2026-09-22 each finalize appended the whole sheet
    # through record_evaluations(), so "correct, then press Finalize again" put
    # every answer into mastery twice. The sheet's interactions now carry a
    # source tag, and a re-finalize replaces them: the stored model always
    # holds exactly one copy of the sheet, with its latest marks. Replacing,
    # rather than applying a delta, is what keeps it exact: mastery is scored
    # from the history itself (KnowledgeMastery), so a delta interaction would
    # still be a second piece of evidence.

    @staticmethod
    def _signature(interactions: list[Interaction]) -> list[tuple]:
        """What a sheet contributed, ignoring when it was recorded."""
        return sorted((i.concept_id, i.kind, i.outcome, i.bloom, i.difficulty)
                      for i in interactions)

    def _tagged(self, model: LearnerModel, source: str) -> list[Interaction]:
        return [i for s in model.concepts.values() for i in s.history if i.source == source]

    def sheet_recorded(self, student_id: str, source: str) -> bool:
        """Whether this sheet's answers are in the student's model."""
        return bool(self._tagged(self.load(student_id), source))

    def sheet_matches(self, student_id: str, source: str,
                      results: list[tuple[QuestionSchema, Evaluation]]) -> bool:
        """Whether the model already holds exactly these marks for the sheet,
        so recording it again would change nothing."""
        tagged = self._tagged(self.load(student_id), source)
        return bool(tagged) and (self._signature(tagged)
                                 == self._signature(self._interactions(results, source)))

    def record_sheet(self, student_id: str,
                     results: list[tuple[QuestionSchema, Evaluation]],
                     source: str) -> bool:
        """Record a finalized sheet, replacing any earlier copy tagged with the
        same source. Returns False, and writes nothing, when the stored copy
        already has these marks. Interactions without this tag (practice,
        other sheets, sheets finalized before tagging existed) are untouched."""
        if not source:
            raise ValueError("record_sheet needs a source tag")
        model = self.load(student_id)
        new = self._interactions(results, source)
        old = self._tagged(model, source)
        if old and self._signature(old) == self._signature(new):
            return False
        if old:
            # A re-finalize corrects evidence from an earlier sitting; it is
            # not a new sitting. KnowledgeMastery weights confidence by each
            # answer's age and decays retention from last_seen, so stamping
            # the replacements with "now" would make a sheet sat 40 days ago
            # score as if sat today (retention from its 0.10 floor back to
            # 1.0 at the default 14-day tau) because one mark was corrected.
            # Keep the sheet's original time.
            sat = min((i.ts for i in old), key=_ts_key)
            for i in new:
                i.ts = sat
        for cid in list(model.concepts):
            state = model.concepts[cid]
            removed = [i for i in state.history if i.source == source]
            if not removed:
                continue
            state.history = [i for i in state.history if i.source != source]
            state.attempts = max(0, state.attempts - len(removed))
            state.correct = max(0, state.correct - sum(
                1 for i in removed
                if i.kind == "answer" and i.outcome is not None and i.outcome >= 0.5))
            if not state.history and state.attempts == 0:
                del model.concepts[cid]
        for i in new:
            model.observe(i)
        # observe() sets last_seen to the interaction it just appended, and the
        # replacements carry the sheet's original (older) time, so the last
        # interaction written is not the newest: practice done after the sheet
        # must stay the latest evidence. last_seen is the newest ts in history.
        for cid in {i.concept_id for i in old} | {i.concept_id for i in new}:
            state = model.concepts.get(cid)
            if state is not None:
                state.last_seen = (max((i.ts for i in state.history), key=_ts_key)
                                   if state.history else None)
        self.save(model)
        return True

    def mastery(self, student_id: str,
                misconception_map: dict[str, list[str]] | None = None,
                ) -> list[ConceptMasteryView]:
        model = self.load(student_id)
        misconception_map = misconception_map or {}
        out: list[ConceptMasteryView] = []
        for cid in sorted(model.concepts):
            r = self._scorer.score(model, cid)
            out.append(ConceptMasteryView(
                concept_id=cid,
                concept_name=chapter_name(cid) or cid.replace("-", " ").title(),
                mastery=r.mastery, accuracy=r.accuracy, retention=r.retention,
                confidence=r.confidence, evidence_count=r.evidence_count,
                confident=r.confident, status=_status(r.mastery, r.confident),
                misconceptions=misconception_map.get(cid, []),
            ))
        return out

    def weak_concepts(self, student_id: str, limit: int = 5) -> list[ConceptMasteryView]:
        views = [v for v in self.mastery(student_id) if v.is_weak]
        views.sort(key=lambda v: v.mastery)
        return views[:limit]

    def overall(self, student_id: str) -> float:
        views = self.mastery(student_id)
        return round(sum(v.mastery for v in views) / len(views), 4) if views else 0.0
