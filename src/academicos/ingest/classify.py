"""Document classifier: decide DocType from filename hints + first-page signals.

Two-layer approach (heuristics first, cheap; VLM/LLM refinement later) so the
pipeline works even with zero heavy dependencies installed.
"""
from __future__ import annotations

import re

from ..models.enums import DocType
from .discovery import SourceFile

_FN_RULES: list[tuple[re.Pattern, DocType]] = [
    (re.compile(r"marking|scheme|marking\s*scheme|answer\s*key", re.I), DocType.MARKING_SCHEME),
    (re.compile(r"question\s*-?\s*paper|qp[-_\s]|questionpaper", re.I), DocType.QUESTION_PAPER),
    (re.compile(r"model\s*answer|modelanswer", re.I), DocType.MODEL_ANSWER),
    (re.compile(r"sample\s*paper|sop|sqp|samplequestion", re.I), DocType.SAMPLE_PAPER),
    (re.compile(r"curriculum", re.I), DocType.CURRICULUM),
    (re.compile(r"syllabus", re.I), DocType.SYLLABUS),
    (re.compile(r"question\s*bank", re.I), DocType.QUESTION_BANK),
    (re.compile(r"circular", re.I), DocType.CIRCULAR),
    (re.compile(r"bye\s*law|byelaw", re.I), DocType.BYE_LAW),
    (re.compile(r"datesheet|date\s*sheet|time\s*table", re.I), DocType.DATESHEET),
    (re.compile(r"notification|press\s*release", re.I), DocType.NOTIFICATION),
    (re.compile(r"guideline|guidelines", re.I), DocType.GUIDELINE),
    (re.compile(r"reference\s*material|reference-material", re.I), DocType.REFERENCE_MATERIAL),
]

_MARKING_KEYWORDS = ("marking scheme", "marking-scheme", "answer key", "marks awarded", "expected answer")
_QP_KEYWORDS = ("time allowed", "maximum marks", "general instructions", "section a", "section b",
                "this question paper", "compartment", "improvement")


def classify_from_name(sf: SourceFile) -> DocType:
    name = sf.path.stem
    for pat, dtype in _FN_RULES:
        if pat.search(name):
            return dtype
    return DocType.UNKNOWN


def classify_from_text(sf: SourceFile, first_pages_text: str) -> DocType:
    low = first_pages_text.lower()
    name_type = classify_from_name(sf)
    if name_type is not DocType.UNKNOWN:
        return name_type
    marks = sum(1 for k in _MARKING_KEYWORDS if k in low)
    qp = sum(1 for k in _QP_KEYWORDS if k in low)
    if "marking scheme" in low and qp >= 2:
        return DocType.MARKING_SCHEME
    if qp >= 2:
        return DocType.QUESTION_PAPER
    if "model answer" in low or "model answers" in low:
        return DocType.MODEL_ANSWER
    if "curriculum" in low and ("learning outcomes" in low or "course content" in low):
        return DocType.CURRICULUM
    return DocType.UNKNOWN
