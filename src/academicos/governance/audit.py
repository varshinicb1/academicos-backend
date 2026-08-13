"""Governance: provenance ledger + audit trail for every extraction/write."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..models.common import Provenance


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, subject: str, payload: dict[str, Any] | None = None,
               provenance: Optional[Provenance] = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "subject": subject,
            "payload": payload or {},
        }
        if provenance:
            entry["provenance"] = provenance.model_dump(mode="json")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]


class Verifier:
    """Rule-of-evidence checks before facts enter the verified graph."""

    REQUIRED = ("document_id", "page")

    @staticmethod
    def check_evidence(provenance: Provenance | None) -> tuple[bool, str]:
        if provenance is None:
            return False, "no provenance"
        if not provenance.source.document_id:
            return False, "missing document_id"
        if provenance.confidence is None or provenance.confidence.score < 0.5:
            return False, "confidence below 0.5"
        return True, "ok"

    @staticmethod
    def validate_text_provenance(snippet: str, source_text: str) -> tuple[bool, float]:
        """Verify a quoted snippet actually appears in the source text."""
        if not snippet or len(snippet) < 8:
            return False, 0.0
        norm_s = " ".join(snippet.lower().split())
        norm_t = " ".join(source_text.lower().split())
        return norm_s in norm_t, 1.0 if norm_s in norm_t else 0.0
