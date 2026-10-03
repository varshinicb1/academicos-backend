"""A Hindi textbook chapter whose PDF text layer is not Hindi, read by OCR instead.

NCERT's class 10 Hindi books (Kshitij, Sparsh, Kritika, Sanchayan) were typeset in a legacy
font (Walkman-Chanakya): the PDF's text layer holds Latin look-alikes ("rqe gkS vfr cM+Hkkxh"
for "तुम हौ अति बड़भागी"), 0% Devanagari, so nothing can be set from it. Sarvam Vision reads
the page images as proper Unicode (96% Devanagari on Kshitij chapter 1).

`scripts/ocr_hindi_chapter.py` runs a chapter once, a page at a time, and keeps the cleaned text
in a cache outside the repository (the books are NCERT's, not ours to commit). This module is the
cleaner and the reader of that cache; `gen_cbse_chapter.py` uses the cache in place of the text
layer only when the text layer is not Hindi.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Where the OCR text is kept: outside the repository.
DEFAULT_CACHE = Path(r"D:\acos-wt\_ocwork\ocr")

DEVANAGARI_SHARE_MIN = 0.4

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# What the OCR says about a picture rather than printing the book's own words: a generated
# description of the image, or a note that it found a code. None of it is in the book.
_ABOUT_A_PICTURE = re.compile(
    r"^\W*(?:यह|इस|इन|ये)\s*(?:छवि|चित्र|तस्वीर|फ़ोटो|फोटो)|^\W*(?:छवि|चित्र)\s*में|^\W*this image|^\W*the image",
    re.I)
_FURNITURE = re.compile(r"^\W*reprint\s*20\d\d\W*(?:\d\d)?\W*$|^\W*-{3,}\W*$", re.I)


def devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0900" <= c <= "\u097f") / len(letters)


def clean(markdown: str) -> list[str]:
    """The book's own lines from one page of Sarvam's markdown.

    Images, their generated descriptions, the reprint line and page rules are dropped; heading
    marks and emphasis are not text. A blank line ends a paragraph, so a description that
    runs to several lines goes whole.
    """
    text = _IMAGE.sub("", markdown or "")
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in paragraph.splitlines() if l.strip()]
        if not lines:
            continue
        first = lines[0].lstrip("#*_> ").strip()
        if _ABOUT_A_PICTURE.match(first) or _ABOUT_A_PICTURE.match(" ".join(lines)):
            continue
        for line in lines:
            if _FURNITURE.match(line):
                continue
            line = re.sub(r"^[#>\s]+", "", line)
            line = re.sub(r"[*_]{1,3}", "", line).strip()
            if line:
                out.append(line)
    return out


def cache_path(book_code: str, number: int, cache: Optional[Path] = None) -> Path:
    return (cache or DEFAULT_CACHE) / f"{book_code}-{number:02d}.json"


def load(book_code: str, number: int, cache: Optional[Path] = None) -> Optional[list[tuple[int, str]]]:
    """(page, line) for a cached chapter, or None when it has not been read.

    `page` is the 1-based position in the chapter's PDF: the OCR cannot see the number the
    book prints, and the text layer that could say it is the one that is unusable."""
    path = cache_path(book_code, number, cache)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(int(p["page"]), line) for p in data.get("pages", []) for line in p.get("lines", [])]
