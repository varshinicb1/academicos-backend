"""A hard per-request ceiling on outbound LLM calls.

OWASP's LLM Top 10 (2026) ranks **Unbounded Consumption** sixth, up four places
on the previous edition, and frames it plainly: the failure is "the absence of
adequate controls over how resources are consumed", and the impact is a
denial-of-wallet -- the attacker's cost is one HTTP request, yours is the
inference bill. Their survey found 12 of 14 AI-surface applications had a
confirmed denial-of-wallet path.

Where this codebase stood
-------------------------
One `POST /v1/agent/doubt` with the reflection critic enabled performed **four**
provider calls: one `decide_retrieve`, then one `score` per candidate evidence
path (capped at three). That is bounded today, and it is bounded by an accident
of the current implementation -- a `hits[:3]` slice in `orchestrator.py` -- not
by anything that would stop a future edit from turning it into "retry until the
critic is confident" and quietly making the ceiling unbounded.

Why a context variable rather than a parameter
----------------------------------------------
The call that must be limited is `SarvamLLM.chat`, four frames below the route
handler. Threading a budget through every intermediate signature (`solve` ->
`_solve_with_critic` -> `critic.score` -> `llm.chat_json` -> `llm.chat`) adds a
parameter to five functions that do not otherwise care, and the next person to
add a call path can forget it. A context variable makes the ceiling ambient:
whoever enters `with llm_budget(n):` has capped every provider call beneath them,
and no intermediate function has to cooperate.

FastAPI runs synchronous endpoints in a worker thread, and `chat` runs in that
same thread, so a value set in the handler is visible to the call. It is *not*
visible across threads the handler might spawn -- no current path does this, and
a budget that silently fails to apply would be worse than none, so
`test_llm_budget.py` pins the behaviour that matters (that a budget does bind,
and that exceeding it raises) rather than assuming it.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Iterator

# Default ceiling for one inbound request. Comfortably above the deepest current
# path (1 retrieval decision + 3 path scores + 1 question mapping = 5) and low
# enough that a runaway loop stops after single-digit spend rather than after
# the request times out.
DEFAULT_MAX_LLM_CALLS = 12


class LLMBudgetExceeded(RuntimeError):
    """Raised when one inbound request tries to make more provider calls than allowed.

    This is a deliberate hard stop, not a retryable condition. The caller is
    asking for more model work than the request was budgeted for, which means
    the control flow is wrong -- so it surfaces loudly instead of degrading.

    Carries plain values, not a reference to the `LLMBudget`. Two reasons, and
    the first one is not theoretical: the previous version took the budget and
    read `budget.used` in this constructor, which re-acquired the same
    non-reentrant lock `consume()` was still holding and deadlocked the request
    thread. An exception that holds a live, mutable object is also just a bad
    idea -- by the time anything inspects it, the counters have moved and the
    message it prints no longer describes what went wrong.
    """

    def __init__(self, *, label: str, used: int, max_calls: int,
                 last_operation: str | None = None) -> None:
        super().__init__(
            f"LLM call budget exhausted for {label}: "
            f"{used} of {max_calls} calls used "
            f"(last operation: {last_operation or 'unknown'})"
        )
        self.label = label
        self.used = used
        self.max_calls = max_calls
        self.last_operation = last_operation


class LLMBudget:
    """A counter of provider calls, scoped to one logical operation.

    Thread-safe because a budget may legitimately be shared by concurrent work
    inside one request (for example parallel evidence scoring). Contention is
    negligible -- the critical section is a comparison and an increment -- and
    correctness under concurrency is not optional for a ceiling.
    """

    __slots__ = ("max_calls", "label", "_used", "_last_operation", "_lock")

    def __init__(self, max_calls: int = DEFAULT_MAX_LLM_CALLS, *, label: str = "request") -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.label = label
        self._used = 0
        self._last_operation: str | None = None
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self._used)

    @property
    def last_operation(self) -> str | None:
        with self._lock:
            return self._last_operation

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._used >= self.max_calls

    def consume(self, operation: str = "llm") -> None:
        """Claim one call. Raises `LLMBudgetExceeded` when there are none left.

        Checked, incremented and reported under one lock so two concurrent
        callers cannot both see the last slot as free. The exception is built
        from values read *inside* the critical section and holds none of them by
        reference -- see `LLMBudgetExceeded`'s docstring for why that distinction
        is load-bearing rather than stylistic.
        """
        with self._lock:
            if self._used >= self.max_calls:
                raise LLMBudgetExceeded(
                    label=self.label,
                    used=self._used,
                    max_calls=self.max_calls,
                    last_operation=self._last_operation,
                )
            self._used += 1
            self._last_operation = operation

    def snapshot(self) -> dict:
        """Serialisable view, for a response body or a log line."""
        with self._lock:
            return {
                "label": self.label,
                "max_calls": self.max_calls,
                "used": self._used,
                "remaining": max(0, self.max_calls - self._used),
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LLMBudget({self.label!r}, {self.used}/{self.max_calls})"


_active_budget: contextvars.ContextVar[LLMBudget | None] = contextvars.ContextVar(
    "academicos_llm_budget", default=None
)


def current_budget() -> LLMBudget | None:
    """The budget in force for the calling context, if any.

    `None` means "unbudgeted" -- allowed for CLI and test paths that are not
    serving a request. Request-serving code should always be inside
    `llm_budget(...)`, and `test_llm_budget.py` asserts that the routes are.
    """
    return _active_budget.get()


def consume_budget(operation: str = "llm") -> None:
    """Spend one call against the active budget; a no-op when unbudgeted."""
    budget = _active_budget.get()
    if budget is not None:
        budget.consume(operation)


@contextmanager
def llm_budget(
    max_calls: int = DEFAULT_MAX_LLM_CALLS, *, label: str = "request"
) -> Iterator[LLMBudget]:
    """Bound every provider call made inside this block.

    Nesting replaces rather than adds: an inner, tighter budget is a deliberate
    choice by the inner scope, and it resets on exit.
    """
    budget = LLMBudget(max_calls, label=label)
    token = _active_budget.set(budget)
    try:
        yield budget
    finally:
        _active_budget.reset(token)
