"""Layout analysis: heading detection, reading order, block reclassification.

Operates on Page.blocks; upgrades PARAGRAPH blocks to HEADING/QUESTION/etc.
based on font size deltas and text patterns. Pure heuristic — cheap and
deterministic; VLM refinement is a later-phase upgrade path.
"""
from __future__ import annotations

import re

from ..models.document import Page
from ..models.enums import BlockKind

_HEADING_PAT = re.compile(r"^\s*(unit|chapter|lesson|module|section|topic|part)[ .\-:\t]*[\dIVX]+", re.I)
_NUM_PAT = re.compile(r"^\s*(\d{1,3})[.|\-|\)|\)\s]\s")
_SUB_NUM_PAT = re.compile(r"^\s*([a-z][.)\-])\s")
_Q_MARKS_PAT = re.compile(r"\(\s*\d+\s*(?:marks?|×\s*\d+)?\s*\)", re.I)


def analyze_page_layout(page: Page) -> Page:
    sizes: list[float] = []
    for b in page.blocks:
        fs = b.metadata.get("fontsize_estimate", 0.0)
        if fs > 0:
            sizes.append(fs)
    max_size = max(sizes) if sizes else 0.0

    prev_heading = False
    for b in page.blocks:
        text = b.text.strip()
        fs = b.metadata.get("fontsize_estimate", 0.0)

        if b.kind in (BlockKind.HEADING, BlockKind.QUESTION):
            prev_heading = True
            continue

        if _HEADING_PAT.match(text) or (fs and max_size and fs >= max_size * 0.85 and len(text) < 140):
            b.kind = BlockKind.HEADING
            prev_heading = True
            continue
        if _NUM_PAT.match(text) and _Q_MARKS_PAT.search(text):
            b.kind = BlockKind.QUESTION
            prev_heading = True
            continue
        if _NUM_PAT.match(text) and not prev_heading and len(text) < 160:
            b.kind = BlockKind.LIST if not _Q_MARKS_PAT.search(text) else BlockKind.QUESTION
            prev_heading = False
            continue
        if _SUB_NUM_PAT.match(text) and len(text) < 200:
            b.kind = BlockKind.LIST
            prev_heading = False
            continue
        prev_heading = False
    return page
