"""PDF-native parser: PyMuPDF text extraction with block layout and reading order.

Fast path for born-digital CBSE PDFs. Classifies pages as text/scan/mixed and
attaches bbox for every block so downstream layout/table modules can work in
relative page coordinates.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from ..models.document import Page, ParseResult, Block
from ..models.enums import BlockKind, PageKind
from .base import ParserProvider

log = logging.getLogger(__name__)

_NON_TEXT_THRESHOLD = 0.85
_BLANK_THRESHOLD = 20

# Some CBSE math/physics papers draw the minus sign in an equation as a thin
# filled rectangle instead of setting a "-" glyph — confirmed by inspecting
# the raw character stream: the gap where a minus belongs has no char record
# at all, just an oversized space between the surrounding words' bboxes.
# `page.get_text("text")` only reads the character stream, so it silently
# drops these, turning "2y - 5 = 0" into "2y  5 = 0" — a different equation.
_DASH_RECT_WIDTH = (2.0, 16.0)
_DASH_RECT_HEIGHT = 2.5


def _drawn_dash_rects(page) -> list[tuple[float, float, float, float]]:
    out = []
    for dr in page.get_drawings():
        r = dr.get("rect")
        if r is None or dr.get("type") not in ("f", "fs"):
            continue
        w, h = r.x1 - r.x0, r.y1 - r.y0
        if _DASH_RECT_WIDTH[0] <= w <= _DASH_RECT_WIDTH[1] and h <= _DASH_RECT_HEIGHT:
            out.append((r.x0, r.y0, r.x1, r.y1))
    return out


def _restore_vector_dashes(dashes: list[tuple[float, float, float, float]],
                           words: list, text: str) -> str:
    """Splices a "-" back into `text` for each drawn-rectangle dash found.

    For every candidate rectangle, finds the word immediately to its left and
    right on the same text baseline (via `get_text("words")`, which carries
    per-word bboxes `get_text("text")` does not) and inserts a dash between
    them wherever they still appear adjacent, separated only by whitespace,
    in the plain-text output. `count=1` keeps this to the first match so a
    word pair that legitimately repeats elsewhere on the page is untouched.

    `dashes` and `words` are computed once per page by the caller — both are
    page-wide PyMuPDF queries, expensive enough that recomputing them for
    every block on the page (this runs once per block too) turned a ~1.6s/file
    parse into ~20s/file.
    """
    if not dashes:
        return text
    for dx0, dy0, dx1, dy1 in dashes:
        cy = (dy0 + dy1) / 2
        row = sorted((w for w in words if w[1] - 2 <= cy <= w[3] + 2), key=lambda w: w[0])
        before = next((w for w in reversed(row) if w[2] <= dx0 + 1), None)
        after = next((w for w in row if w[0] >= dx1 - 1), None)
        if not before or not after or not before[4] or not after[4]:
            continue
        pattern = re.compile(re.escape(before[4]) + r"(\s+)" + re.escape(after[4]))
        text = pattern.sub(lambda m, b=before[4], a=after[4]: f"{b} -{a}", text, count=1)
    return text


class PdfTextParser(ParserProvider):
    name = "pdf_native"

    def parse(self, document_path: Path, document_id: str, max_pages: int = 500) -> ParseResult:
        import fitz  # pymupdf

        res = ParseResult(document_id=document_id, method=self.name)
        with fitz.open(document_path) as doc:
            for i in range(min(doc.page_count, max_pages)):
                page = doc[i]
                # Both are page-wide queries — compute once and reuse for the
                # page text and every block's text below, rather than paying
                # for them again per block.
                dashes = _drawn_dash_rects(page)
                words = page.get_text("words") if dashes else []
                raster = page.get_text("dict")

                text = page.get_text("text").strip()
                text = _restore_vector_dashes(dashes, words, text)
                blocks: list[Block] = []
                kind = PageKind.TEXT

                if len(text) < _BLANK_THRESHOLD:
                    kind = PageKind.BLANK
                else:
                    char_count = sum(len(b[4]) for b in page.get_text("blocks"))
                    drawn_area = page.rect.width * page.rect.height

                    # rough scan heuristic: if most spans come from rawdict as image blocks
                    images = page.get_images(full=True)
                    text_chars = char_count
                    if images and text_chars == 0:
                        kind = PageKind.SCAN
                    elif images and text_chars < 40:
                        kind = PageKind.MIXED

                    order = 0
                    for b in page.get_text("blocks"):
                        x0, y0, x1, y1, btext, bno, btype = b[:7]
                        if not btext.strip():
                            continue
                        # extract_questions() reads Block.text (not Page.text)
                        # to build each question's source_text, so the same
                        # vector-dash recovery has to run per block too.
                        btext = _restore_vector_dashes(dashes, words, btext)
                        blocks.append(Block(
                            kind=self._kind(btype, btext, order, blocks),
                            text=btext.strip(),
                            bbox={"x0": x0 / page.rect.width, "y0": y0 / page.rect.height,
                                  "x1": x1 / page.rect.width, "y1": y1 / page.rect.height},
                            order=order,
                            metadata={"fontsize_estimate": self._fontsize(raster, bno)},
                        ))
                        order += 1

                page_model = Page(
                    page_no=i + 1,
                    kind=kind,
                    text=text,
                    blocks=blocks,
                    quality=1.0 if kind != PageKind.SCAN else 0.3,
                )
                res.pages.append(page_model)

        res.confidence = min(1.0, 0.9 if any(p.kind != PageKind.SCAN for p in res.pages) else 0.4)
        if not res.pages:
            res.warnings.append("no pages parsed")
        return res

    @staticmethod
    def _kind(btype: int, text: str, order: int, prev: list[Block]) -> BlockKind:
        if btype == 1:
            return BlockKind.IMAGE_ONLY
        low = text.strip().lower()
        if order == 0 and (len(low) < 120 and (low.endswith((")", "]", "»")) or " " not in low.strip())):
            return BlockKind.HEADING
        return BlockKind.PARAGRAPH

    @staticmethod
    def _fontsize(raster: dict, bno: int) -> float:
        try:
            for b in raster.get("blocks", []):
                if b.get("number") == bno or b.get("bbox", [0])[0] == bno:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            return round(s.get("size", 0), 1)
        except Exception:
            pass
        return 0.0
