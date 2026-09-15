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

    def reset(self) -> None:
        """Clear all tracked request timestamps (primarily for test isolation)."""
        with self._lock:
            self._records.clear()


# Default limiters: 20 login attempts/min and 10 registrations/min per IP
login_limiter = RateLimiter(max_requests=20, window_seconds=60.0)
register_limiter = RateLimiter(max_requests=10, window_seconds=60.0)


def get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting standard reverse-proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_login(request: Request) -> None:
    """FastAPI dependency for login endpoint rate limiting."""
    client_ip = get_client_ip(request)
    login_limiter.check(f"login:{client_ip}")


def rate_limit_register(request: Request) -> None:
    """FastAPI dependency for register endpoint rate limiting."""
    client_ip = get_client_ip(request)
    register_limiter.check(f"register:{client_ip}")
