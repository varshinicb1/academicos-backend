"""Ensemble fusion: pick the best parse per page across providers.

Strategy (research-informed):
  * born-digital text pages -> pdf_native preferred (fast, exact).
  * scanned pages (pdf_native quality < threshold) -> VLM/OCR parse preferred.
  * confidence = max over providers, penalized when providers disagree.
Records per-page provenance of the winning provider.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..models.common import Confidence, Provenance, SourceSpan
from ..models.document import Page, ParseResult, ParsedDocument
from ..models.enums import DocType, ExtractionMethod
from .base import ParserProvider
from .pdftext import PdfTextParser
from .unlimited import UnlimitedOcrParser
from .pytesseract import TesseractParser

log = logging.getLogger(__name__)


class EnsembleParser:
    def __init__(self, priority: list[str] | None = None,
                 vlm_mode: str = "service", vlm_url: str = "http://127.0.0.1:8910",
                 vlm_device: str = "cuda"):
        self.providers: dict[str, ParserProvider] = {
            "pdf_native": PdfTextParser(),
            "unlimited_ocr": UnlimitedOcrParser(mode=vlm_mode, url=vlm_url, device=vlm_device),
            "tesseract": TesseractParser(),
        }
        self.priority = priority or ["pdf_native", "unlimited_ocr", "tesseract"]

    def parse(self, path: Path, document_id: str, doc_type: DocType,
              max_pages: int = 500) -> ParsedDocument:
        results: dict[str, ParseResult] = {}
        for name in self.priority:
            provider = self.providers.get(name)
            if not provider:
                continue
            try:
                r = provider.parse(path, document_id, max_pages=max_pages)
                if r.pages:
                    results[name] = r
                    log.debug("provider %s: %d pages, conf %.2f", name, len(r.pages), r.confidence)
            except Exception as e:  # never let one provider kill the pipeline
                log.warning("provider %s failed: %s", name, e)

        if not results:
            raise RuntimeError(f"no parser produced output for {path}")

        best_name = max(results, key=lambda n: self._page_score(results[n]))
        best = results[best_name]

        pages: list[Page] = []
        n_pages = max(len(r.pages) for r in results.values())
        for i in range(n_pages):
            chosen = best.pages[i] if i < len(best.pages) else Page(page_no=i + 1)
            if chosen.quality < 0.5 and "unlimited_ocr" in results and i < len(results["unlimited_ocr"].pages):
                alt = results["unlimited_ocr"].pages[i]
                if alt.quality > chosen.quality:
                    chosen = alt
                    chosen.metadata["provider"] = "unlimited_ocr"
            pages.append(chosen)

        full_text = "\n\n".join(p.text for p in pages)
        return ParsedDocument(
            document_id=document_id,
            doc_type=doc_type,
            title=path.stem,
            pages=pages,
            full_text=full_text,
            page_map={p.page_no: p.text for p in pages},
            provenance=[Provenance(
                source=SourceSpan(document_id=document_id, page=1),
                method=ExtractionMethod(results[best_name].method),
                confidence=Confidence(method=ExtractionMethod(results[best_name].method),
                                      score=results[best_name].confidence),
            )],
            quality_score=results[best_name].confidence,
            metadata={"providers": list(results), "winner": best_name},
        )

    # A page carrying less than this many characters has no usable text on it.
    MIN_CHARS_PER_PAGE = 80

    @classmethod
    def _page_score(cls, r: ParseResult) -> float:
        """Rank a provider by how much *text* it actually recovered.

        Providers self-report `confidence`/`quality` without checking whether
        they extracted anything: on a scanned PDF, pdf_native returns full pages
        with empty strings and still scored ~0.9, so it beat every OCR provider
        and the document silently yielded zero questions. Text yield is the
        ground truth here, so it gates the score.
        """
        if not r.pages:
            return 0.0
        avg_quality = sum(p.quality for p in r.pages) / len(r.pages)
        pages_with_text = sum(1 for p in r.pages if len(p.text.strip()) >= cls.MIN_CHARS_PER_PAGE)
        text_yield = pages_with_text / len(r.pages)
        if text_yield == 0.0:
            return 0.0        # produced nothing readable — never let this win
        return r.confidence * avg_quality * text_yield

    @classmethod
    def has_usable_text(cls, doc: ParsedDocument) -> bool:
        """Did the winning parse actually recover text? Callers use this to
        decide whether a document needs an OCR pass before extraction."""
        if not doc.pages:
            return False
        good = sum(1 for p in doc.pages if len(p.text.strip()) >= cls.MIN_CHARS_PER_PAGE)
        return good / len(doc.pages) >= 0.3
