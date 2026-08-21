"""Composio ActionProvider: executes one named Composio tool over HTTP.

Plain `requests` against Composio's REST API directly (no composio SDK
dependency) -- this is the exact call shape confirmed working live against
real Google Calendar in composio_calendar.py; kept unchanged here, just
moved behind the ActionProvider interface so any future Composio-backed
capability (Drive, Gmail, Sheets, ...) is a new caller of this one class,
not a new copy of the HTTP/error-handling logic.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

from ..agents.base import ToolResult
from .base import ActionProvider

log = logging.getLogger(__name__)

API_BASE = "https://backend.composio.dev/api/v3"
TIMEOUT_SEC = 15


class ComposioProvider(ActionProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("COMPOSIO_API_KEY", "").strip() or None

    def available(self) -> bool:
        return bool(self.api_key)

    def execute(self, tool_slug: str, arguments: dict[str, Any], *,
                user_id: Optional[str] = None,
                connected_account_id: Optional[str] = None) -> ToolResult:
        if not self.api_key:
            return ToolResult(tool=tool_slug, ok=False, error="Composio: no API key configured")
        if not connected_account_id or not user_id:
            return ToolResult(tool=tool_slug, ok=False,
                               error="Composio: connected_account_id and user_id are required")
        try:
            r = requests.post(
                f"{API_BASE}/tools/execute/{tool_slug}",
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                json={"connected_account_id": connected_account_id, "user_id": user_id,
                      "arguments": arguments},
                timeout=TIMEOUT_SEC,
            )
            r.raise_for_status()
            data = r.json()
            payload = data.get("data") if isinstance(data, dict) else None
            response_data = payload.get("response_data") if isinstance(payload, dict) else None
            if response_data is None:
                log.warning("Composio %s returned no response_data: %s", tool_slug, data)
                return ToolResult(tool=tool_slug, ok=False, error="no response_data in Composio reply",
                                   payload=data)
            return ToolResult(tool=tool_slug, ok=True, payload=response_data)
        except requests.RequestException as exc:
            log.warning("Composio %s failed: %s", tool_slug, exc)
            return ToolResult(tool=tool_slug, ok=False, error=str(exc))
