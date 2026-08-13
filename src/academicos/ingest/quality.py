"""Quality gates: reject corrupt/error-page/duplicate files early.

The CBSE crawl produced a known class of false positives: 1208-byte HTML 500
error pages and 624-byte 404 pages downloaded with a .pdf extension. This gate
catches those before parsing.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .discovery import SourceFile

_MIN_REASONABLE_PDF = 2048
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"
_HTML_ERR = re.compile(rb"(?i)<html|<title>|error|not found|500|404")


def validate_file(sf: SourceFile, min_pdf_bytes: int = _MIN_REASONABLE_PDF) -> tuple[bool, list[str]]:
    """Return (ok, problems). A file failing this gate is not ingested."""
    problems: list[str] = []
    size = sf.size_bytes
    ext = sf.extension

    if size < 16:
        problems.append("trivially small file")
    if ext == "pdf" and size < min_pdf_bytes:
        problems.append(f"pdf under {min_pdf_bytes} bytes")
    if ext in ("pdf", "zip") and size < 4096:
        head = _read_head(sf.path, 512)
        if _HTML_ERR.search(head):
            problems.append("HTML error page (500/404) with pdf/zip extension")

    magic = _read_head(sf.path, 4)
    if ext == "pdf" and magic and magic != _PDF_MAGIC:
        problems.append("missing %PDF magic")
    if ext == "zip" and magic and magic != _ZIP_MAGIC:
        problems.append("missing PK zip magic")
    if ext == "zip":
        try:
            with zipfile.ZipFile(sf.path) as zf:
                bad = zf.testzip()
                if bad:
                    problems.append(f"corrupt zip member: {bad}")
        except (zipfile.BadZipFile, OSError) as e:
            problems.append(f"zip open failed: {e}")

    return not problems, problems


def _read_head(path: Path, n: int) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)
