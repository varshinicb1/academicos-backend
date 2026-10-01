"""A paper's time allowed, written the way a paper prints it.

Every header printed `{minutes // 60} hours {minutes % 60:02d} minutes`, so a
40-minute unit test read "Time Allowed: 0 hours 40 minutes" (v3 audit, 14 of
14 unit tests) and a 60-minute paper "1 hours 00 minutes". The PDF, the Word
file and the paper's stored text all call this one function.
"""
from __future__ import annotations


def _unit(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def format_duration(minutes: int) -> str:
    """"40 minutes", "1 hour", "1 hour 30 minutes", "3 hours"."""
    minutes = max(0, int(minutes))
    hours, rest = divmod(minutes, 60)
    if not hours:
        return _unit(rest, "minute")
    if not rest:
        return _unit(hours, "hour")
    return f"{_unit(hours, 'hour')} {_unit(rest, 'minute')}"
