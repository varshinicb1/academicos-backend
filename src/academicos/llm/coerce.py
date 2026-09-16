"""Strict coercion of model output into application types.

Model output is untrusted input. OWASP's LLM Top 10 puts this at LLM10 (Improper
Output Handling): "treat model output as untrusted input at every destination".
The rule is easy to state and easy to skip, and the failure is quiet -- a bad
value silently becomes a plausible wrong one.

What went wrong before this module existed
------------------------------------------
`critic.decide_retrieve` did `bool(out.get("retrieve", True))`. Python's `bool`
on a non-empty string is `True`, so a model that answered `{"retrieve": "false"}`
-- a perfectly reasonable thing for a language model to emit -- was read as
`True` and retrieval ran anyway. The direction of the decision inverted, with no
error and nothing in the log. That behaviour was even pinned by a test
(`test_decide_retrieve_string_value_is_truthy`) which documented it as intended.

The same class of bug sat in `critic.score`, where `int(out.get("utility", 3))`
raised on `"utility": "high"`, and the surrounding `except Exception` turned a
parsing problem into silently neutral scores.

Two rules follow, and every function here obeys them:

1. **Never use Python truthiness on a value that came from outside.** `bool`
   is only correct for values that are already `bool`.
2. **Record the coercion failure.** Falling back to a default is fine and often
   right; falling back *silently* is what makes the bug unfindable. Callers pass
   a `problems` list and surface it.
"""
from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")

_TRUE_LITERALS = frozenset({"true", "yes", "y", "1", "on", "t"})
_FALSE_LITERALS = frozenset({"false", "no", "n", "0", "off", "f", "none", "null", ""})

Problems = list[str] | None


def _note(problems: Problems, message: str) -> None:
    if problems is not None:
        problems.append(message)


def as_bool(value: Any, default: bool, *, problems: Problems = None, field: str = "value") -> bool:
    """Coerce `value` to `bool` without ever consulting Python truthiness.

    Accepts real booleans, numbers (`0` is False) and the string literals a
    model realistically emits for a boolean. Anything unrecognised returns
    `default` and records a problem -- it does not guess.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Deliberately not `bool(value)` on arbitrary objects: only numbers get
        # arithmetic truth. NaN is falsy here via `!= 0`.
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_LITERALS:
            return True
        if text in _FALSE_LITERALS:
            return False
    _note(problems, f"{field}: expected a boolean, got {value!r}; using {default}")
    return default


def as_int(
    value: Any,
    default: int,
    *,
    lo: int | None = None,
    hi: int | None = None,
    problems: Problems = None,
    field: str = "value",
) -> int:
    """Coerce to `int`, optionally clamped to `[lo, hi]`.

    Floats are truncated only when they are integral (`4.0` -> `4`); `4.7` is a
    different thing that the model did not mean, so it is rejected rather than
    silently rounded. Strings must parse as a whole integer -- `"3 things"` is
    not `3`.
    """
    result: int | None = None
    if isinstance(value, bool):
        # bool is an int subclass; treat it as the type confusion it is.
        result = None
    elif isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if value.is_integer():
            result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            result = int(text)
        except ValueError:
            result = None

    if result is None:
        _note(problems, f"{field}: expected an integer, got {value!r}; using {default}")
        result = default

    if lo is not None and result < lo:
        _note(problems, f"{field}: {result} below minimum {lo}; clamped")
        result = lo
    if hi is not None and result > hi:
        _note(problems, f"{field}: {result} above maximum {hi}; clamped")
        result = hi
    return result


def as_float(
    value: Any,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    problems: Problems = None,
    field: str = "value",
) -> float:
    """Coerce to `float`. `NaN` and infinities are rejected, not propagated.

    A NaN that reaches a score comparison makes every comparison false, which
    reads downstream as "no path was best" rather than as a parse failure.
    """
    result: float | None = None
    if isinstance(value, bool):
        result = None
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            result = None

    if result is None or result != result or result in (float("inf"), float("-inf")):
        _note(problems, f"{field}: expected a number, got {value!r}; using {default}")
        result = default

    if lo is not None and result < lo:
        _note(problems, f"{field}: {result} below minimum {lo}; clamped")
        result = lo
    if hi is not None and result > hi:
        _note(problems, f"{field}: {result} above maximum {hi}; clamped")
        result = hi
    return result


def as_choice(
    value: Any,
    allowed: Iterable[str],
    default: T,
    *,
    problems: Problems = None,
    field: str = "value",
) -> str | T:
    """Coerce to one of `allowed`, case-insensitively. Anything else is refused.

    An allowlist, not a mapping-with-fallback: OWASP's guidance for tool
    arguments is exactly this ("allowlist for tools"). A model that invents
    `"mostly supported"` must not be silently bucketed into `"partial"`.
    """
    options = tuple(allowed)
    if isinstance(value, str):
        text = value.strip().lower()
        for option in options:
            if option.lower() == text:
                return option
    _note(problems, f"{field}: {value!r} is not one of {options}; using {default!r}")
    return default


def as_text(
    value: Any,
    default: str = "",
    *,
    max_chars: int | None = None,
    problems: Problems = None,
    field: str = "value",
) -> str:
    """Coerce to `str`, optionally truncating.

    Truncation is reported: a model answer that got cut is materially different
    from one that was short, and downstream callers deserve to know.
    """
    if isinstance(value, str):
        text = value
    elif value is None:
        _note(problems, f"{field}: expected text, got null; using default")
        return default
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        _note(problems, f"{field}: expected text, got {type(value).__name__}; using default")
        return default

    if max_chars is not None and len(text) > max_chars:
        _note(problems, f"{field}: truncated from {len(text)} to {max_chars} chars")
        return text[:max_chars]
    return text
