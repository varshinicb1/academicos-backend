"""External-action provider seam.

The Academic Brain (retrieval/graph/algorithms/assessment) must never import
Composio, Sarvam, or any other external SDK directly -- see
docs/provider-architecture.md. Every external-action integration (Google
Calendar today, Drive/Gmail/Sheets/etc. later) implements this one small
interface instead of being special-cased into caller code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..agents.base import ToolResult


class ActionProvider(ABC):
    """One external action-execution backend (e.g. Composio)."""

    @abstractmethod
    def available(self) -> bool:
        """Cheap, no-network check: is this provider configured at all?"""
        ...

    @abstractmethod
    def execute(self, tool_slug: str, arguments: dict[str, Any], *,
                user_id: Optional[str] = None) -> ToolResult:
        """Run one named external action. Must never raise -- any transport
        or provider-side failure comes back as ToolResult(ok=False, error=...)."""
        ...
