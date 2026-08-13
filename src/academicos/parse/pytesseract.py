"""Tesseract OCR fallback for scanned pages (requires optional extra `ocr`)."""
from __future__ import annotations

from pathlib import Path

from ..models.document import Page, ParseResult
from ..models.enums import PageKind
from .base import ParserProvider


class TesseractParser(ParserProvider):
    name = "tesseract"

    def parse(self, document_path: Path, document_id: str, max_pages: int = 500) -> ParseResult:
        try:
            import pytesseract
            from PIL import Image
            import fitz
        except ImportError as e:
            res = ParseResult(document_id=document_id, method=self.name, confidence=0.0)
            res.warnings.append(f"ocr extras missing: {e}")
            return res

        res = ParseResult(document_id=document_id, method=self.name)
        with fitz.open(document_path) as doc:
            for i in range(min(doc.page_count, max_pages)):
                pix = doc[i].get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img).strip()
                res.pages.append(Page(page_no=i + 1, kind=PageKind.SCAN, text=text,
                                      quality=0.75 if text else 0.1))
        res.confidence = 0.7 if res.pages else 0.0
        return res
