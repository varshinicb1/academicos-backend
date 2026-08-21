"""Table detection + reconstruction from PDF raw dict (TATR-style heuristics).

Phase 1: text-grid reconstruction from span coordinates (works on born-digital
CBSE tables). Phase 2 (roadmap): TATR/DETR layout model for scanned tables,
with MLLM validation per XLLM-2025 ensemble finding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Cell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1

@dataclass
class Table:
    page_no: int
    cells: list[Cell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def to_markdown(self) -> str:
        grid: list[list[str]] = [[""] * self.n_cols for _ in range(self.n_rows)]
        for c in self.cells:
            grid[c.row][c.col] = c.text
        lines = []
        for r in grid:
            lines.append("| " + " | ".join(x.replace("|", "\\|") for x in r) + " |")
        if grid:
            lines.insert(1, "|" + "---|" * len(grid[0]))
        return "\n".join(lines)


def detect_tables(page_raw: dict[str, Any], page_no: int) -> list[Table]:
    """Detect tables via text alignment heuristics on a PyMuPDF rawdict."""
    lines: list[tuple[float, float, float, str]] = []
    for block in page_raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            y = line["bbox"][1]
            for s in spans:
                text = s.get("text", "").strip()
                if text:
                    lines.append((y, s["bbox"][0], s["bbox"][2], text))

    if len(lines) < 2:
        return []

    # group lines into visual rows by y proximity
    lines.sort(key=lambda t: (round(t[0], 0), t[1]))
    rows: list[list[tuple[float, float, str]]] = []
    cur_y = None
    for y, x0, x1, text in lines:
        if cur_y is None or abs(y - cur_y) > 3:
            rows.append([])
            cur_y = y
        rows[-1].append((x0, x1, text))

    tables: list[Table] = []
    for r in rows:
        if len(r) >= 3 and _column_aligned(r):
            tbl = Table(page_no=page_no)
            for i, (x0, x1, text) in enumerate(r):
                tbl.cells.append(Cell(row=0, col=i, text=text))
            tbl.n_rows = 1
            tbl.n_cols = len(r)
            tables.append(tbl)
    return tables


def _column_aligned(row: list[tuple[float, float, str]]) -> bool:
    xs = [x0 for x0, _, _ in row]
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    if not gaps:
        return False
    min_gap = min(gaps)
    return min_gap > 8 and all(g >= min_gap * 0.6 for g in gaps)


_MARKS_RE = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*(?:marks?|×\s*\d+)?\s*\)?")


def extract_table_marks(table: Table) -> dict[str, float]:
    """Best-effort column/value extraction for weightage tables."""
    out: dict[str, float] = {}
    for c in table.cells:
        m = _MARKS_RE.search(c.text)
        if m:
            out[f"r{c.row}c{c.col}"] = float(m.group(1))
    return out
