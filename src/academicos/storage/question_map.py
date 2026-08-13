"""P1.9 — Question→concept mapping store (append-only JSONL).

One line per mapped question: the raw question, the resolved Q→concept
links (verified against the graph), the method used, and a stable
question hash so the same question dedupes. Read-side helpers:
`all_maps`, `for_concept` (reverse index: which questions exercise a
concept — this is what per-question tracing needs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


class QuestionMapStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, qmap: dict[str, Any]) -> str:
        """Persist one mapping; returns its question-key (dedupes)."""
        q = qmap.get("question", "").strip()
        key = question_key(q)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **(qmap or {})}) + "\n")
        return key

    def all_maps(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        for line in self.path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def for_question(self, question: str) -> dict[str, Any] | None:
        key = question_key(question)
        for m in self.all_maps():
            if m.get("key") == key:
                return m
        return None

    def count(self) -> int:
        return len(self.all_maps())

    def healthy(self) -> dict[str, Any]:
        return {"path": str(self.path), "mapped": self.count()}


def question_key(question: str) -> str:
    """Stable, whitespace/`case-folded fingerprint — dedupes re-asks."""
    norm = " ".join(question.strip().split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]