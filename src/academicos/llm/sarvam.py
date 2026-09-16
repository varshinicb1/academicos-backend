"""Minimal Sarvam LLM client (OpenAI-compatible chat completions).

Reads the API key from `SARVAM_API_KEY` (or ACOS_LLM_API_KEY / config).
Kept dependency-free (stdlib + requests) so the whole brain can call
sarvam-30b / sarvam-105b without SDKs.

Hardening, and what it is responding to
---------------------------------------
This client is the single place every model call leaves the process, which
makes it the right place for the controls that must not be forgettable. Each
addition below traces to a specific criticism of the previous version:

* **Retries now use full-jitter exponential backoff** (`llm/resilience.py`)
  instead of a fixed `5 * (attempt + 1)` sleep. The old curve was linear *and*
  unjittered, so every client that failed together also retried together -- the
  synchronised-wave problem the AWS Builders' Library describes. It also
  retried 502/504 not at all, despite those being the most common transient
  failures in front of a hosted model.
* **`Retry-After` is honoured, then capped.** Previously read with a bare
  `int()` in a `try/except ValueError`, so an HTTP-date form (which RFC 9110
  permits and providers do send) silently fell back to the computed wait, and a
  hostile or confused upstream could have asked us to sleep for a day.
* **A spent budget stops the request instead of the clock** (`llm/budget.py`).
  Attempt counting bounds one call; it does not bound how many calls one
  inbound request makes, which is the denial-of-wallet shape.
* **Every attempt is recorded** (`llm/telemetry.py`) with latency, attempt
  number, status and prompt size -- shape, never prompt content.
* **Prompt size is bounded before the request goes out**, so an oversized
  prompt fails immediately and locally rather than after a timeout and an
  inference bill.
* **The response shape is validated.** `data["choices"][0]["message"]` raises
  `KeyError`/`IndexError`/`TypeError` on an error envelope or a changed schema,
  which surfaced as an opaque crash four frames up. It is now one explicit
  error naming what arrived.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from .base import LLMProvider
from .budget import consume_budget
from .resilience import RetryPolicy, is_retryable_status, parse_retry_after
from .telemetry import TELEMETRY, LLMCallRecord, now_ms

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sarvam.ai"

# Supply a cap when the caller does not. Without one, a reasoning model can run
# to its own ceiling, which is both slow and expensive; the previous behaviour
# of omitting `max_tokens` entirely made the completion length someone else's
# decision.
DEFAULT_MAX_TOKENS = 2048

# Refuse absurd prompts before spending a request on them. 200k characters is
# far beyond any legitimate use here (the largest real prompt is a chapter of
# textbook text plus a marking scheme) and well inside every current model's
# context, so this only ever fires on a bug or an attack.
DEFAULT_MAX_PROMPT_CHARS = 200_000

# Hosted inference degrades in seconds, so the curve is shorter than a
# database's would be: ~0.5s, ~1s, ~2s on average with full jitter, and a 20s
# ceiling that keeps one request from outliving its own client timeout.
LLM_RETRY_POLICY = RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=20.0)


class LLMProviderError(RuntimeError):
    """A provider call failed after exhausting its retries, or returned something unusable.

    Subclasses `RuntimeError` so existing `except RuntimeError` handlers keep
    working -- the previous code raised bare `RuntimeError` from these paths.
    """


class SarvamLLM(LLMProvider):
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL,
                 model: str = "sarvam-105b", timeout: float = 120.0,
                 *, policy: RetryPolicy | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
                 sleep=time.sleep, telemetry=TELEMETRY):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.policy = policy or LLM_RETRY_POLICY
        self.max_tokens = max_tokens
        self.max_prompt_chars = max_prompt_chars
        # Injected so retry behaviour is testable without real waiting. A test
        # suite that sleeps for the backoff is a test suite that gets skipped.
        self._sleep = sleep
        self._telemetry = telemetry

    @property
    def available(self) -> bool:
        local = "127.0.0.1" in self.base_url or "localhost" in self.base_url
        return bool(self.api_key) or local

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int | None = None, operation: str = "chat") -> str:
        if not self.available:
            raise LLMProviderError("Sarvam LLM: no API key (set SARVAM_API_KEY)")

        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        if prompt_chars > self.max_prompt_chars:
            raise ValueError(
                f"Sarvam LLM: prompt is {prompt_chars} chars, over the "
                f"{self.max_prompt_chars} limit -- refusing to send"
            )

        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for attempt in range(self.policy.max_attempts):
            # Raises LLMBudgetExceeded when this request has spent its allowance.
            # Deliberately before the network call, so the ceiling is enforced
            # by the clock we control rather than by the provider's patience.
            consume_budget(operation)

            started = now_ms()
            response = None
            try:
                response = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                self._record(operation, attempt, ok=False, started=started,
                             prompt_chars=prompt_chars, error=type(exc).__name__)
                if not self.policy.should_retry(attempt, retryable=True):
                    raise LLMProviderError(
                        f"Sarvam LLM: {type(exc).__name__} after "
                        f"{attempt + 1} attempt(s) to {url}"
                    ) from exc
                self._sleep(self.policy.delay_for(attempt))
                continue

            status = response.status_code
            latency_ok = not is_retryable_status(status)

            if is_retryable_status(status):
                self._record(operation, attempt, ok=False, started=started,
                             prompt_chars=prompt_chars, status_code=status,
                             error=f"HTTP {status}")
                if not self.policy.should_retry(attempt, retryable=True):
                    raise LLMProviderError(
                        f"Sarvam LLM: HTTP {status} after {attempt + 1} attempt(s) "
                        f"to {url}"
                    )
                retry_after = parse_retry_after(response.headers)
                self._sleep(self.policy.delay_for(attempt, retry_after_s=retry_after))
                continue

            if status >= 400:
                # Deterministic for this request: a 400/401/403/404/422 will
                # fail identically however many times it is sent. Retrying is
                # pure waste of the provider's quota.
                body = (response.text or "")[:300]
                self._record(operation, attempt, ok=False, started=started,
                             prompt_chars=prompt_chars, status_code=status,
                             error=f"HTTP {status}")
                raise LLMProviderError(
                    f"Sarvam LLM: HTTP {status} (not retryable) from {url}: {body}"
                )

            content = self._read_content(response)
            self._record(operation, attempt, ok=latency_ok, started=started,
                         prompt_chars=prompt_chars, status_code=status,
                         completion_chars=len(content))
            return content

        # Every attempt was retryable and the budget ran out. `last_error` is
        # set on the connection/timeout path; the status path raises above, so
        # reaching here without it means the loop was exhausted by retryable
        # statuses whose final iteration already raised. Kept as a belt-and-
        # braces terminator so the function can never fall through to `None`.
        raise LLMProviderError(
            f"Sarvam LLM: request failed after {self.policy.max_attempts} attempts"
        ) from last_error

    def _read_content(self, response) -> str:
        """Pull the assistant text out of a response, or say what was wrong with it."""
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError(
                f"Sarvam LLM: response was not JSON ({exc}); "
                f"body starts {(response.text or '')[:200]!r}"
            ) from exc

        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                f"Sarvam LLM: unexpected response shape "
                f"(keys={sorted(data) if isinstance(data, dict) else type(data).__name__}): "
                f"{str(data)[:200]}"
            ) from exc

        if msg.get("content"):
            return msg["content"]

        # A reasoning-only reply is a real provider behaviour, not a bug, but it
        # is unusable as an answer and the caller must not receive an empty
        # string as though it were one.
        if msg.get("reasoning_content"):
            log.warning(
                "Sarvam LLM returned reasoning-only (finish=%s); increase max_tokens",
                choice.get("finish_reason"),
            )
        raise LLMProviderError(
            f"Sarvam LLM returned no content (finish={choice.get('finish_reason')})"
        )

    def _record(self, operation: str, attempt: int, *, ok: bool, started: float,
                prompt_chars: int, status_code: int | None = None,
                completion_chars: int = 0, error: str | None = None) -> None:
        if self._telemetry is None:
            return
        self._telemetry.record(LLMCallRecord(
            ts=time.time(),
            operation=operation,
            model=self.model,
            attempt=attempt,
            ok=ok,
            latency_ms=round(now_ms() - started, 1),
            status_code=status_code,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars,
            error=error,
        ))

    def chat_json(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        """Chat then parse the first JSON object/array out of the reply."""
        reply = self.chat(messages, **kw)
        return extract_json(reply)


def extract_json(text: str) -> Any:
    """Best-effort extraction of the first JSON value from an LLM reply.

    Falls back to salvaging a truncated JSON document (common with reasoning
    models that hit the completion cap): retry on the prefixes ending at the
    last few closing brackets.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty LLM reply")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"no JSON in LLM reply: {text[:120]!r}")
    depth = 0
    in_str = False
    esc = False
    ends: list[int] = []
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
            ends.append(i)
    for cut in ends[-4:]:
        try:
            return json.loads(text[start:cut + 1])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unterminated JSON in LLM reply: {text[:120]!r}")
