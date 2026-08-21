"""Minimal Sarvam LLM client (OpenAI-compatible chat completions).

Reads the API key from `SARVAM_API_KEY` (or ACOS_LLM_API_KEY / config).
Kept dependency-free (stdlib + requests) so the whole brain can call
sarvam-30b / sarvam-105b without SDKs.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .base import LLMProvider

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sarvam.ai"


class SarvamLLM(LLMProvider):
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL,
                 model: str = "sarvam-105b", timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        local = "127.0.0.1" in self.base_url or "localhost" in self.base_url
        return bool(self.api_key) or local

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int | None = None) -> str:
        if not self.available:
            raise RuntimeError("Sarvam LLM: no API key (set SARVAM_API_KEY)")
        import requests

        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        import time

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                r = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if r.status_code in (429, 500, 503):
                    wait = 5 * (attempt + 1)
                    if "retry-after" in r.headers:
                        try:
                            wait = max(wait, int(r.headers["retry-after"]))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        else:
            raise last_err or RuntimeError(f"Sarvam LLM: request failed after retries")
        data = r.json()
        msg = data["choices"][0]["message"]
        if msg.get("content"):
            return msg["content"]
        if msg.get("reasoning_content"):
            log.warning(
                "Sarvam LLM returned reasoning-only (finish=%s); increase max_tokens",
                data["choices"][0].get("finish_reason"),
            )
        raise ValueError(
            f"Sarvam LLM returned no content (finish={data['choices'][0].get('finish_reason')})"
        )

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
