"""Retry, backoff and jitter policy for outbound provider calls.

Scope: this module is policy, not I/O. It decides *whether* and *how long* to
wait; the caller owns the loop and the socket. Keeping it pure is what makes the
timing testable without sleeping -- a retry policy that can only be verified by
waiting four times is a policy nobody verifies.

Why the shape is what it is
---------------------------
1. **Full jitter, not plain exponential backoff.** Capped exponential backoff
   without jitter is the classic own-goal: every client that failed at the same
   moment wakes at the same moment, so the dependency is hit by a synchronised
   wave (a "retry storm") exactly when it is least able to absorb one. The AWS
   Builders' Library measures no-jitter / equal-jitter / full-jitter /
   decorrelated-jitter and finds full jitter -- a uniform draw in `[0, capped]`
   -- gives the best behaviour under typical load. We use full jitter.

2. **Retry only what can change.** A 400, 401, 403, 404 or 422 will fail
   identically on retry; retrying it spends the provider's time to get the same
   answer. Only genuinely transient conditions are retryable.

3. **Bound every wait, including `Retry-After`.** An upstream is free to send
   `Retry-After: 86400`. Honouring that literally parks a worker thread for a
   day. We honour it, then cap it.

4. **Bounded attempts.** Retries are "selfish": each one spends more of the
   dependency's time. Beyond a handful, returns fall to zero while load keeps
   rising.

Not implemented here, and why: a *retry budget* (Google SRE's ~10% cap on
retries as a fraction of normal traffic) is the right tool when many clients
retry into one backend. In this codebase the retrying caller is a single
request-serving process talking to one provider, so per-request attempt caps
bound the amplification, and a budget would add state without changing an
outcome.
"""
from __future__ import annotations

import calendar
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping

# 408 Request Timeout      -- the request may not have been processed
# 425 Too Early            -- transient by definition
# 429 Too Many Requests    -- explicitly "try again later"
# 500 Internal Server Error-- often transient in hosted inference
# 502 Bad Gateway          -- edge/infra, transient
# 503 Service Unavailable  -- overloaded or restarting
# 504 Gateway Timeout      -- upstream was slow, not necessarily wedged
#
# Deliberately absent: 400, 401, 403, 404, 409, 422. These are deterministic for
# a fixed request and retrying them only burns the provider's quota.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status: int) -> bool:
    """True when an HTTP status is worth another attempt."""
    return status in RETRYABLE_STATUS


def parse_retry_after(
    headers: Mapping[str, str] | None, *, now: float | None = None
) -> float | None:
    """Read `Retry-After` as a number of seconds.

    RFC 9110 allows two forms and servers really do use both: a delta in seconds
    (`Retry-After: 30`) and an HTTP-date (`Retry-After: Wed, 21 Oct 2026 07:28:00
    GMT`). Returning `None` means "no usable instruction", which the caller
    treats as "fall back to the computed backoff" -- never as "retry now".
    """
    if not headers:
        return None
    # Case-insensitive by scanning keys rather than probing two spellings.
    # `requests` headers are case-insensitive, but callers pass plain dicts
    # (tests, and any non-requests transport), and "RETRY-AFTER" must work
    # there too -- a look-up that only knows two casings silently misses.
    raw: str | None = None
    for key in headers:
        if isinstance(key, str) and key.lower() == "retry-after":
            raw = headers[key]
            break
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None

    if when.tzinfo is None:
        # An HTTP-date is defined to be GMT. A naive value is read as UTC rather
        # than local time, so this does not shift with the host's timezone.
        target = float(calendar.timegm(when.timetuple()))
    else:
        target = when.timestamp()
    reference = time.time() if now is None else now
    return max(0.0, target - reference)


@dataclass(frozen=True)
class RetryPolicy:
    """Attempt budget and delay curve for one outbound call.

    Defaults are for a user-facing path against a hosted inference provider:
    four attempts is where the returns stop, and 20s is a ceiling a request can
    afford to wait without the caller's own timeout becoming the real limit.
    """

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("delays must be non-negative")

    @property
    def retries(self) -> int:
        """How many *additional* attempts are allowed after the first."""
        return self.max_attempts - 1

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after_s: float | None = None,
        rand: Callable[[], float] = random.random,
    ) -> float:
        """Seconds to wait before the attempt *after* `attempt` (0-based).

        `attempt=0` is the delay that follows the first failure. A caller that
        reaches its final attempt must not call this at all -- see
        `should_retry`, which encodes that so callers cannot forget it.

        A server-supplied `Retry-After` takes precedence over our own curve,
        because the server knows its own recovery time and we do not. It is
        still clamped to `max_delay_s`: an upstream may legitimately ask for a
        wait longer than this request is worth holding a thread for.
        """
        if retry_after_s is not None:
            return min(max(retry_after_s, 0.0), self.max_delay_s)
        if rand is None:  # pragma: no cover - defensive
            return min(self.base_delay_s * (2.0**attempt), self.max_delay_s)
        capped = min(self.base_delay_s * (2.0**attempt), self.max_delay_s)
        # Full jitter: uniform in [0, capped]. See the module docstring.
        return rand() * capped

    def should_retry(self, attempt: int, *, retryable: bool) -> bool:
        """Whether another attempt is both permitted and worthwhile.

        Both halves matter. `retryable` is the caller's classification of *why*
        the attempt failed; a non-retryable reason (a 400, a malformed payload)
        makes further attempts pure waste regardless of the remaining budget.
        """
        if not retryable:
            return False
        return attempt + 1 < self.max_attempts


DEFAULT_POLICY = RetryPolicy()


def backoff_sequence(
    policy: RetryPolicy = DEFAULT_POLICY,
    *,
    retry_after_s: float | None = None,
    rand: Callable[[], float] = random.random,
) -> list[float]:
    """The delays this policy would produce, for logging and for tests."""
    return [
        policy.delay_for(i, retry_after_s=retry_after_s, rand=rand)
        for i in range(policy.retries)
    ]
