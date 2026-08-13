"""Chunking: heading-anchored hierarchical chunks from parsed pages.

Each chunk keeps (heading_path, text, page, char span) so retrieval can cite
chapter > topic > section and reconstruction into canonical objects stays easy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.document import ParsedDocument

@dataclass
class Chunk:
    id: str
    document_id: str
    page: int
    heading_path: list[str] = field(default_factory=list)
    text: str = ""
    char_start: int = 0
    char_end: int = 0
    kind: str = "text"

    @property
    def heading(self) -> str:
        return " > ".join(h for h in self.heading_path if h)


_HEADING_RE = re.compile(r"^\s*(unit|chapter|lesson|module|section|topic|part)\s*[\dIVXl]+", re.I)
_MAX_CHARS = 4000


def chunk_document(doc: ParsedDocument, max_chars: int = _MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_path: list[str] = []
    char_pos = 0

    for page in doc.pages:
        buf: list[str] = []
        buf_start = char_pos
        current_heading = list(heading_path)

        def flush() -> None:
            nonlocal buf, buf_start
            if buf:
                chunks.append(_make_chunk(doc, page, current_heading, buf, buf_start, char_pos, max_chars))
                buf = []
                buf_start = char_pos

        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue
            if block.kind.value == "heading" or _HEADING_RE.match(text):
                flush()
                if len(heading_path) < 3 and len(text) < 120:
                    heading_path.append(text[:80])
                    current_heading = list(heading_path)
            else:
                if not buf:
                    buf_start = char_pos
                buf.append(text)
                char_pos += len(text) + 1
                if sum(len(b) for b in buf) >= max_chars:
                    flush()
        flush()
    return chunks


def _make_chunk(doc: ParsedDocument, page, heading_path, buf, start, end, max_chars) -> Chunk:
    text = "\n".join(buf)
    return Chunk(
        id=f"{doc.document_id}:p{page.page_no}:{start}",
        document_id=doc.document_id,
        page=page.page_no,
        heading_path=list(heading_path),
        text=text,
        char_start=start,
        char_end=end,
    )
