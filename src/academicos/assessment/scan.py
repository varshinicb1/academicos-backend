"""Pillar 2 — answer sheet capture and question segmentation.

Takes a scanned answer booklet (PDF of page images, e.g. the evaluated CBSE
scripts in `cbse-model-answers/`), renders pages, OCRs each one through Sarvam
Vision, then splits the transcript into per-question answers.

Segmentation is text-based rather than the colored-separator-line approach:
students already write the question number before each answer, and that marker
survives OCR far more reliably than a drawn line survives a phone camera. When
no marker is found the page is attributed to the previous question so text is
never silently dropped.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import vision

log = logging.getLogger(__name__)

# "12.", "12)", "Q12", "Ans 12", "Answer 12." at the start of a line. The
# leading `\.?` tolerates a stray period OCR sometimes inserts before the
# real marker (observed on a real CBSE booklet: ".13)" for "13)").
_Q_MARKER = re.compile(
    r"^\s*\.?\s*(?:(?:ans(?:wer)?|q(?:uestion)?)[\s.:\-]*)?(\d{1,2})\s*[.):\-]\s*",
    re.I | re.M,
)

# Sarvam Vision renders some tabular MCQ answer blocks as markdown/HTML tables
# (e.g. "<td>Ans1.</td><td>(A) No Solution</td>") instead of plain
# "1. (A) No Solution" lines. _Q_MARKER requires the marker at the start of a
# line, but inside a table every line starts with a tag, so every answer was
# silently discarded as "no answer found" -- confirmed against a real CBSE
# Class X Math Standard handwritten booklet, all 20 real answers lost. Strip
# tags before segmenting; each table row already renders on its own line, so
# once the tags are gone "Ans1." naturally lands at the start of its line.
_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass
class AnswerSegment:
    question_no: int
    text: str
    page_numbers: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_blank(self) -> bool:
        return len(self.text.strip()) <= 3


@dataclass
class ScannedSheet:
    sheet_id: str
    student_id: str
    student_name: str
    roll_number: str
    assessment_id: str
    pages: list[vision.PageOCR] = field(default_factory=list)
    segments: list[AnswerSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Fraction of pages that yielded usable text."""
        if not self.pages:
            return 0.0
        good = sum(1 for p in self.pages if p.text.strip() and not p.hallucinated)
        return round(good / len(self.pages), 3)


def render_pdf_pages(pdf_path: Path, out_dir: Path, *, dpi: int = 140,
                     max_pages: int | None = None,
                     first_page: int = 0) -> list[Path]:
    """Rasterise a scanned booklet to page PNGs."""
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    doc = fitz.open(str(pdf_path))
    last = doc.page_count if max_pages is None else min(doc.page_count, first_page + max_pages)
    for i in range(first_page, last):
        target = out_dir / f"page_{i:03d}.png"
        doc[i].get_pixmap(dpi=dpi).save(str(target))
        paths.append(target)
    doc.close()
    return paths


def _split_nested_marker(qno: int, body: str) -> tuple[int, str]:
    """A matched marker can itself be OCR noise swallowing the real marker
    right after it -- confirmed on a real CBSE booklet where Sarvam misread
    handwritten "Q." as "9." (so "Q.5)" and "Q.7)" both matched as a bogus
    "9)", eating both answers into question 9's slot), and separately
    prefixed a spurious leading digit before a real two-digit number
    ("8.14)" for what should be "14)", landing question 14's answer inside
    question 8's segment). A genuine single-answer body never legitimately
    starts with its own "<digit><separator>" -- students don't write MCQ
    answers that way -- so if the body we just captured itself begins with
    a clean, distinct marker, that inner one is the real question number.
    """
    nested = _Q_MARKER.match(body)
    if nested and int(nested.group(1)) != qno:
        return int(nested.group(1)), body[nested.end():].strip()
    return qno, body


def segment_answers(pages: list[vision.PageOCR]) -> list[AnswerSegment]:
    """Split OCR'd pages into per-question answers on question-number markers."""
    segments: dict[int, AnswerSegment] = {}
    current: int | None = None
    orphan: list[str] = []

    for page in pages:
        text = _HTML_TAG.sub("", page.text or "")
        if not text.strip():
            continue
        matches = list(_Q_MARKER.finditer(text))
        if not matches:
            if current is None:
                orphan.append(text.strip())
            else:
                seg = segments[current]
                seg.text = (seg.text + "\n" + text.strip()).strip()
                if page.page_no not in seg.page_numbers:
                    seg.page_numbers.append(page.page_no)
            continue

        # Text before the first marker continues the previous question.
        head = text[: matches[0].start()].strip()
        if head and current is not None:
            seg = segments[current]
            seg.text = (seg.text + "\n" + head).strip()
            if page.page_no not in seg.page_numbers:
                seg.page_numbers.append(page.page_no)

        for idx, m in enumerate(matches):
            qno = int(m.group(1))
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[m.end():end].strip()
            qno, body = _split_nested_marker(qno, body)
            seg = segments.get(qno)
            if seg is None:
                seg = AnswerSegment(question_no=qno, text=body, page_numbers=[page.page_no])
                segments[qno] = seg
            else:
                seg.text = (seg.text + "\n" + body).strip()
                if page.page_no not in seg.page_numbers:
                    seg.page_numbers.append(page.page_no)
            current = qno

    out = [segments[k] for k in sorted(segments)]
    if orphan and out:
        out[0].warnings.append("text appeared before any question number and was discarded")
    for seg in out:
        for page in pages:
            if page.page_no in seg.page_numbers:
                seg.warnings.extend(page.warnings)
    return out


def scan_sheet(pdf_path: Path, *, sheet_id: str, assessment_id: str,
               student_id: str, student_name: str, roll_number: str,
               workdir: Path, max_pages: int = 6, first_page: int = 0,
               language: str = "en-IN") -> ScannedSheet:
    """Full capture pipeline for one answer booklet."""
    sheet = ScannedSheet(sheet_id=sheet_id, student_id=student_id,
                         student_name=student_name, roll_number=roll_number,
                         assessment_id=assessment_id)
    page_paths = render_pdf_pages(pdf_path, workdir / "pages", max_pages=max_pages,
                                  first_page=first_page)
    for i, p in enumerate(page_paths):
        page_no = first_page + i + 1
        try:
            sheet.pages.append(vision.ocr_page(p, page_no, workdir=workdir / "ocr",
                                               language=language))
        except Exception as e:
            log.warning("OCR failed on page %s: %s", page_no, e)
            sheet.pages.append(vision.PageOCR(page_no=page_no, text="",
                                              warnings=[f"OCR failed: {e}"]))
    sheet.segments = segment_answers(sheet.pages)
    if sheet.quality_score < 0.6:
        sheet.warnings.append(
            f"low scan quality ({sheet.quality_score:.0%} of pages readable) — "
            "re-scan recommended before evaluation")
    return sheet
