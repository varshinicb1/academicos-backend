"""Counted nouns in the sentences the builder and the papers show.

The availability report, the paper warnings and the swap reasons were
written as "2 question(s)", "1 OR alternative(s)", "1 mark(s) each": QA
P-33 (2026-10-01) found them on the builder's short-paper box and the
printed answer key. `counted` settles each "N ... noun(s)" by its own count,
so the many sentences that build them need not each carry a plural rule.
"""
from __future__ import annotations

import re

# A count, then up to four words ("3-mark", "competency-based", "OR",
# "Social Science"), then the noun with "(s)". A number followed by "-" is
# part of a word ("3-mark"), never the count.
_COUNTED = re.compile(
    r"\b(\d+)(?![\d.-])((?: [\w/'-]+){0,4}?) "
    r"(question|alternative|chapter|mark|section|point|set|student)\(s\)")


def counted(text: str) -> str:
    """`text` with each "N ... noun(s)" as "N ... noun" or "N ... nouns"."""
    def fix(m: re.Match) -> str:
        noun = m.group(3)
        return f"{m.group(1)}{m.group(2)} {noun if int(m.group(1)) == 1 else noun + 's'}"
    return _COUNTED.sub(fix, text)
