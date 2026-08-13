"""Document-layer models: Document, Page, Block, and the parse result envelope."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import BoundingBox, Provenance
from .enums import BlockKind, DocType, PageKind


class Block(BaseModel):
    kind: BlockKind
    text: str = ""
    bbox: Optional[BoundingBox] = None
    order: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list["Block"] = Field(default_factory=list)


class Page(BaseModel):
    page_no: int = Field(ge=1)
    kind: PageKind = PageKind.TEXT
    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    image_path: Optional[str] = None
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    """Output of one parser provider for one document."""
    document_id: str
    method: str
    pages: list[Page] = Field(default_factory=list)
    markdown: Optional[str] = None          # VLM long-horizon markdown when available
    structured: Optional[dict[str, Any]] = None
    language: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    provider_meta: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """A document with all parse artifacts available (ensemble fusion)."""
    document_id: str
    doc_type: DocType
    title: str = ""
    pages: list[Page] = Field(default_factory=list)
    full_text: str = ""
    page_map: dict[int, str] = Field(default_factory=dict)  # page_no -> text
    provenance: list[Provenance] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
