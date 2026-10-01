"""In-memory rate limiter for sensitive endpoints (auth brute-force protection).

Zero external dependencies: uses Python stdlib collections.deque and threading.Lock
to maintain a sliding-window request log per client IP or key.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._records: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Check if request is permitted; raise HTTP 429 if window limit exceeded."""
        if os.environ.get("ACOS_DISABLE_RATE_LIMIT", "").strip().lower() in ("1", "true", "yes"):
            return

        # In automated pytest suites, bypass rate limiting unless explicitly testing it
        if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("ACOS_TEST_RATE_LIMIT"):
            return

        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            queue = self._records[key]
            # Evict timestamps older than the sliding window
            while queue and queue[0] <= cutoff:
                queue.popleft()

            if len(queue) >= self.max_requests:
                oldest = queue[0]
                retry_after = max(1, int(oldest - cutoff) + 1)
                logger.warning("Rate limit exceeded for key %s (max %d in %ds)", key, self.max_requests, self.window_seconds)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            queue.append(now)

    def _bypassed(self) -> bool:
        if os.environ.get("ACOS_DISABLE_RATE_LIMIT", "").strip().lower() in ("1", "true", "yes"):
            return True
        return "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("ACOS_TEST_RATE_LIMIT")

    def blocked(self, key: str) -> Optional[int]:
        """Seconds until `key` may try again, or None. Records nothing: for
        a limit on failures, checked before the attempt and counted after it
        fails (`record`)."""
        if self._bypassed():
            return None
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            queue = self._records[key]
            while queue and queue[0] <= cutoff:
                queue.popleft()
            if len(queue) >= self.max_requests:
                return max(1, int(queue[0] - cutoff) + 1)
        return None

    def record(self, key: str) -> None:
        """Count one event (a failed attempt) against `key`."""
        if self._bypassed():
            return
        with self._lock:
            self._records[key].append(time.monotonic())

    def reset(self) -> None:
        """Clear all tracked request timestamps (primarily for test isolation)."""
        with self._lock:
            self._records.clear()


# Per network (client IP): generous, because a school's whole class signs in
# or registers from one NAT address at once -- 20 sign-ins and 10
# registrations a minute per IP turned the 21st student in a room away
# (audit N-9-4; production check 2026-10-01). Brute force is stopped per
# account instead: a password guessed against one email is refused after
# `ACCOUNT_FAILURES` wrong tries in the window, from any number of addresses.
login_limiter = RateLimiter(max_requests=300, window_seconds=60.0)
register_limiter = RateLimiter(max_requests=120, window_seconds=60.0)
ACCOUNT_FAILURES = 10
account_failures = RateLimiter(max_requests=ACCOUNT_FAILURES, window_seconds=15 * 60.0)


def get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting standard reverse-proxy headers.

    Security fix 2026-09-17 (flagged by automated review): this used to take
    the FIRST (leftmost) value in X-Forwarded-For, which is exactly backwards
    -- the leftmost entry is whatever the ORIGINAL client claimed, so an
    attacker sending `X-Forwarded-For: 1.2.3.4` (or a fresh fake value on
    every request) reset their own rate-limit bucket key on demand,
    completely defeating the brute-force protection this file exists for.

    Both real deployment targets (Render and Cloud Run, see
    docs/PRODUCTS_AND_RELEASES.md) sit their app behind a managed edge proxy
    that APPENDS the true connecting peer's IP as the last hop rather than
    trusting/forwarding whatever the client sent -- so the rightmost value is
    the one the platform itself vouches for, not something a remote client
    can control. There's no fixed, publishable IP allowlist for either
    platform's edge (it's managed infrastructure, not a static IP range), so
    "trust only a known proxy IP" isn't practical here; taking the last hop
    is the standard mitigation when you can't pin the proxy's own address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def rate_limit_login(request: Request) -> None:
    """FastAPI dependency for login endpoint rate limiting."""
    client_ip = get_client_ip(request)
    login_limiter.check(f"login:{client_ip}")


def rate_limit_register(request: Request) -> None:
    """FastAPI dependency for register endpoint rate limiting."""
    client_ip = get_client_ip(request)
    register_limiter.check(f"register:{client_ip}")
