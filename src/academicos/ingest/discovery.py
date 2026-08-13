"""Discovery: crawl the corpus root, classify files by shape, dedupe by checksum."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..models.enums import DocType
from .checksum import file_checksum


@dataclass
class SourceFile:
    path: Path
    rel_key: str
    size_bytes: int
    sha256: str = ""
    sha512: str = ""
    doc_type: DocType = DocType.UNKNOWN
    subject: str | None = None
    grade: str | None = None
    academic_year: str | None = None
    series: str | None = None
    title: str = ""
    hints: list[str] = field(default_factory=list)
    dedupe_of: str | None = None        # canonical source_id this file duplicates

    @property
    def source_id(self) -> str:
        # canonical: sha512 first 24 hex chars
        return f"src:{self.sha512[:24]}" if self.sha512 else f"src:{self.sha256[:24]}"

    @property
    def extension(self) -> str:
        return self.path.suffix.lower().lstrip(".")


class Discovery:
    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root

    def walk(self, max_files: int | None = None) -> list[SourceFile]:
        out: list[SourceFile] = []
        for p in sorted(self.corpus_root.rglob("*")):
            if not p.is_file():
                continue
            if self._ignored(p):
                continue
            out.append(self._make(p))
            if max_files and len(out) >= max_files:
                break
        return out

    @staticmethod
    def _ignored(p: Path) -> bool:
        name = p.name
        if p.suffix.lower() not in (".pdf", ".zip", ".docx", ".doc", ".jpg", ".png", ".jpeg", ".html", ".txt", ".xlsx"):
            return True
        if name.startswith((".", "~$")):
            return True
        if len(name) < 5:
            return True
        return False

    def _make(self, p: Path) -> SourceFile:
        sha256, sha512 = file_checksum(p)
        sf = SourceFile(
            path=p,
            rel_key=p.relative_to(self.corpus_root).as_posix(),
            size_bytes=p.stat().st_size,
            sha256=sha256,
            sha512=sha512,
            title=p.stem,
        )
        sf.hints = self._hints(p)
        return sf

    @staticmethod
    def _hints(p: Path) -> list[str]:
        hints = []
        s = p.name.lower()
        for word, hint in (
            ("marking", "marking_scheme"),
            ("scheme", "marking_scheme"),
            ("question", "question_paper"),
            ("sample", "sample_paper"),
            ("curriculum", "curriculum"),
            ("syllabus", "syllabus"),
            ("circular", "circular"),
            ("byelaw", "bye_law"),
            ("model", "model_answer"),
            ("bank", "question_bank"),
            ("compartment", "compartment"),
            ("improvement", "improvement"),
            ("supplementary", "supplementary"),
        ):
            if word in s:
                hints.append(hint)
        return hints

    @staticmethod
    def classify_hints(hints: list[str]) -> DocType:
        for h in hints:
            try:
                return DocType(h)
            except ValueError:
                continue
        return DocType.UNKNOWN

    def dedupe(self, files: list[SourceFile]) -> list[SourceFile]:
        seen: dict[str, SourceFile] = {}
        out = []
        for f in files:
            key = f.sha256
            if key in seen:
                f.dedupe_of = seen[key].source_id
                out.append(f)  # keep a marker so the ledger records the duplicate
                continue
            seen[key] = f
            out.append(f)
        return out


_YEAR_RE = re.compile(r"(20[12][0-9])")
_GRADE_RE = re.compile(r"\b(xii|x|10th|12th|12|class\s*-?\s*(x|xii))\b", re.I)


def infer_metadata(sf: SourceFile) -> None:
    m = _YEAR_RE.search(sf.path.name)
    if m:
        sf.academic_year = m.group(1)
    if sf.doc_type is DocType.UNKNOWN:
        sf.doc_type = Discovery.classify_hints(sf.hints)
