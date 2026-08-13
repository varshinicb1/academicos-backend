"""Academic object extraction: questions, marking points, blueprints, syllabus items.

Deterministic heuristics over parsed page blocks. The VLM/LLM extraction path
(agent-based) is layered on top and validated against these baselines.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models.academic import (
    AssessmentBlueprint,
    MarkingPoint,
    Question,
    QuestionPart,
    QuestionPaper,
)
from ..models.common import Evidence, Provenance, SourceSpan
from ..models.document import ParsedDocument
from ..models.enums import BloomLevel, DocType, ExtractionMethod

_Q_NO = re.compile(r"^\s*(\d{1,3})\s*[.)\]]\s*")
_PART = re.compile(r"^\s*\(([a-z])\)\s*")
_MARKS = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*(?:marks?)?\s*\)", re.I)
_BLOOM = {
    "remember": BloomLevel.REMEMBER, "recall": BloomLevel.REMEMBER,
    "understand": BloomLevel.UNDERSTAND, "explain": BloomLevel.UNDERSTAND,
    "apply": BloomLevel.APPLY, "calculate": BloomLevel.APPLY,
    "analyze": BloomLevel.ANALYZE, "compare": BloomLevel.ANALYZE,
    "evaluate": BloomLevel.EVALUATE, "justify": BloomLevel.EVALUATE,
    "create": BloomLevel.CREATE, "design": BloomLevel.CREATE,
}


def extract_questions(doc: ParsedDocument, doc_type: DocType) -> list[Question]:
    """Extract questions from question/sample paper pages."""
    out: list[Question] = []
    current: Optional[Question] = None
    part_queue: list[QuestionPart] = []

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.parts = part_queue
            out.append(current)
        current = None
        part_queue.clear()

    for page in doc.pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue
            m = _Q_NO.match(text)
            if m:
                flush()
                marks = _MARKS.search(text)
                current = Question(
                    canonical_id=_canonical_qid(doc.document_id, m.group(1)),
                    question_paper_id=doc.document_id,
                    q_no=m.group(1),
                    marks=float(marks.group(1)) if marks else None,
                    question_type=_guess_type(text),
                    cognitive=_guess_bloom(text),
                    title=text[:120],
                )
                current.source_text = text
                continue
            pm = _PART.match(text)
            if pm and current is not None:
                part_queue.append(QuestionPart(
                    canonical_id=_canonical_qid(doc.document_id, f"{current.q_no}{pm.group(1)}"),
                    question_id=current.canonical_id,
                    part=pm.group(1),
                    text=text,
                    title=f"Q{current.q_no}({pm.group(1)})",
                ))
                continue
            if current is not None:
                current.source_text = (current.source_text or "") + " " + text
    flush()
    return out


def _canonical_qid(doc_id: str, q_no: str) -> str:
    return f"cbse:q:{doc_id}:{q_no}"


def _guess_type(text: str) -> str:
    low = text.lower()
    if "multiple choice" in low or "mcq" in low or "choose" in low:
        return "mcq"
    if "case study" in low or "passage" in low:
        return "case_study"
    if "long" in low or "essay" in low:
        return "long_answer"
    if "short" in low:
        return "short_answer"
    return "unknown"


def _guess_bloom(text: str) -> BloomLevel | None:
    low = text.lower()
    for key, level in _BLOOM.items():
        if key in low:
            return level
    return None


_MARK_POINT = re.compile(r"^\s*([ivx]+[.)-]|\(\s*\d+\s*\)|\.\s*|[-*•])\s*", re.I)


def extract_marking_points(doc: ParsedDocument) -> list[MarkingPoint]:
    """Extract marking points from marking-scheme/model-answer documents."""
    out: list[MarkingPoint] = []
    idx = 1
    for page in doc.pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue
            m = _MARKS.search(text)
            if m:
                marks = float(m.group(1))
                out.append(MarkingPoint(
                    canonical_id=f"cbse:mp:{doc.document_id}:{idx}",
                    answer_scheme_id=doc.document_id,
                    point_no=idx,
                    marks=marks,
                    detail=text[:400],
                    title=f"MP{idx}",
                ))
                idx += 1
    return out


def extract_blueprint(doc: ParsedDocument) -> list[AssessmentBlueprint]:
    """Extract assessment blueprints (design of question paper tables)."""
    out: list[AssessmentBlueprint] = []
    for page in doc.pages:
        text = page.text.lower()
        if "design of question paper" in text or ("blueprint" in text and "marks" in text):
            out.append(AssessmentBlueprint(
                canonical_id=f"cbse:bp:{doc.document_id}:{page.page_no}",
                title=f"Blueprint page {page.page_no}",
            ))
    return out
