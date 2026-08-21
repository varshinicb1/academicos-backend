"""Pillar 3 — Learning Intelligence.

Turns evaluated answers into knowledge, not marks:

    Student -> Question -> Concept -> Learning Outcome -> Bloom -> Mastery

Each evaluated answer becomes an `Interaction` on the student's `LearnerModel`
(the algorithms library's existing learner state), with partial credit carried
through as a fractional outcome so a 3/5 answer is not recorded as simply
"wrong". Mastery is then computed by `KnowledgeMastery`, which already blends
accuracy, recency-weighted confidence, retention decay and higher-Bloom
application.

State is persisted per student so the loop survives restarts. When
SUPABASE_KNOWLEDGE_URL/SUPABASE_KNOWLEDGE_ANON_KEY are set, that persistence
is a Postgres row (via Supabase's PostgREST API) rather than a local JSON
file -- this is the one store in the app where losing data to a Render
redeploy (not just an idle restart) actually matters, since it's a student's
accumulated mastery history, not a single in-flight session. Falls back to
local JSON files when those env vars are absent (local dev).

The anon key used here is never shipped to the Flutter client -- it lives
only as a Render env var on this backend -- so despite the name it functions
as a private server-side credential, not a public one; RLS on the
`learner_models` table is scoped accordingly (see the migration that created
it).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from ..algorithms.learner_model import ConceptState, Interaction, LearnerModel
from ..algorithms.mastery import KnowledgeMastery
from .chapters import chapter_name
from .evaluate import Evaluation
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


class KnowledgeStore:
    """Per-student learner models -- Postgres-backed when Supabase is
    configured, local JSON files otherwise. See module docstring."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._scorer = KnowledgeMastery()
        self._supabase_url = (os.environ.get("SUPABASE_KNOWLEDGE_URL", "").strip().rstrip("/")
                              or None)
        self._supabase_key = os.environ.get("SUPABASE_KNOWLEDGE_ANON_KEY", "").strip() or None

    @property
    def _remote(self) -> bool:
        return bool(self._supabase_url and self._supabase_key)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Content-Type": "application/json",
        }

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
            r = requests.get(
                f"{self._supabase_url}/rest/v1/learner_models",
                params={"student_id": f"eq.{student_id}", "select": "payload"},
                headers=self._headers(), timeout=10,
            )
            r.raise_for_status()
            rows = r.json()
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
            r = requests.post(
                f"{self._supabase_url}/rest/v1/learner_models",
                json={"student_id": model.learner_id, "payload": payload},
                headers={**self._headers(), "Prefer": "resolution=merge-duplicates"},
                timeout=10,
            )
            r.raise_for_status()
        else:
            self._path(model.learner_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_evaluations(self, student_id: str,
                           results: list[tuple[QuestionSchema, Evaluation]]) -> LearnerModel:
        """Fold a marked paper into the student's knowledge state."""
        model = self.load(student_id)
        now = datetime.now(timezone.utc).isoformat()
        for question, ev in results:
            # The teacher's award is the ground truth, whatever the scanner
            # classified the verdict as: a teacher correcting a "blank" to
            # full marks must not have that correction dropped to 0.0. A
            # genuinely blank (awarded 0) still lands at 0.0.
            outcome = (ev.awarded_marks / ev.max_marks) if ev.max_marks else 0.0
            concepts = question.chapter_ids or ["unmapped"]
            for cid in concepts:
                model.observe(Interaction(
                    concept_id=cid, kind="answer", outcome=round(outcome, 4),
                    bloom=question.bloom_level, ts=now,
                    difficulty={"easy": 0.3, "medium": 0.6, "hard": 0.9}.get(
                        question.difficulty, 0.5),
                ))
        self.save(model)
        return model

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
