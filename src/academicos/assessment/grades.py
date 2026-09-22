"""The one place a school grade is parsed.

Before this module the web paper path converted grades three incompatible ways,
and each hid the others' bugs:

  * `routes._int_grade_to_roman` knew only 8-12 and defaulted to "X", so a class
    6 or 7 request silently became class 10 -- a class 7 teacher got a class 10
    board paper printed "Class: 7".
  * the baked-bank loader labelled items 6-9 as "6".."9", while `_grade_matches`
    knew only "10", so class 8 and 9 requests ("VIII", "IX") matched nothing.
  * `mapping.grade_to_int` defaulted to 10 on anything it did not know, so a
    class 9 question reported itself as class 10.

Every caller now goes through here. A grade that cannot be read is refused
(`None` / `ValueError`), never defaulted: a wrong grade is worse than an error,
because nothing downstream can tell it was wrong.
"""
from __future__ import annotations

import re

_ROMAN = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
_BY_ROMAN = {r: i for i, r in enumerate(_ROMAN) if r}
_PREFIX = re.compile(r"^(?:CLASS|GRADE|STD)\s*")
_ORDINAL = re.compile(r"(?<=\d)(?:ST|ND|RD|TH)$")


def to_int(grade: object) -> int | None:
    """1-12 from an int, a digit string, a roman numeral, "7th" or "Class 7"."""
    if isinstance(grade, bool):
        return None
    if isinstance(grade, int):
        return grade if 1 <= grade <= 12 else None
    if not isinstance(grade, str):
        return None
    s = _ORDINAL.sub("", _PREFIX.sub("", grade.strip().upper())).strip()
    if s.isascii() and s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return _BY_ROMAN.get(s)


def to_roman(grade: object) -> str:
    """The roman numeral a pool is keyed by. Raises on an unreadable grade."""
    n = to_int(grade)
    if n is None:
        raise ValueError(f"not a school grade (1-12): {grade!r}")
    return _ROMAN[n]


def matches(a: object, b: object) -> bool:
    """True when both sides name the same grade, whatever form each is in."""
    na = to_int(a)
    return na is not None and na == to_int(b)
