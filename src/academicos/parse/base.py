"""Parser provider protocol: each parser produces a ParseResult for a document."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models.document import ParseResult


class ParserProvider(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, document_path: Path, document_id: str, max_pages: int = 500) -> ParseResult:
        """Parse one document into structured pages/blocks."""
        ...
