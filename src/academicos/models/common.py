"""Cross-cutting models: provenance, confidence, source references.

Every extracted fact carries a Provenance record so outputs are traceable
to (document, page, region, method, confidence, version, timestamp).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .enums import ExtractionMethod


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BoundingBox(BaseModel):
    """Page coordinate space: [0,1] relative units (y down)."""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0


class SourceSpan(BaseModel):
    """Pointer to the exact source location of a fact."""
    document_id: str
    page: int = Field(ge=1)
    block_idx: Optional[int] = None
    paragraph: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    span_text: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class Confidence(BaseModel):
    method: ExtractionMethod
    score: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None
    sub_scores: dict[str, float] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Evidence-first: every fact remembers where it came from."""
    source: SourceSpan
    method: ExtractionMethod
    confidence: Confidence
    extracted_at: str = Field(default_factory=utcnow)
    extractor_version: str = "academicos-0.1.0"
    human_verified: bool = False
    human_correction: Optional[str] = None


class Evidence(BaseModel):
    """A bundle of provenance used by retrieval and generation."""
    items: list[Provenance] = Field(default_factory=list)

    def add(self, prov: Provenance) -> None:
        self.items.append(prov)

    def top(self, k: int = 5) -> list[Provenance]:
        return sorted(self.items, key=lambda p: p.confidence.score, reverse=True)[:k]
