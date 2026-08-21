"""A Composio-backed external action, exposed through the same Tool ABC
every Academic Brain agent tool (SearchTool, ConceptLookupTool) already
implements -- see docs/provider-architecture.md. Not wired into
orchestrator.py's live tool list; this makes a future capability (e.g.
"save to Drive") a real, independently testable class an orchestrator can
register, without any special-casing of Composio in the orchestrator itself.
"""
from __future__ import annotations

from typing import Any, Optional

from ..integrations.base import ActionProvider
from .base import Tool, ToolResult


class ComposioActionTool(Tool):
    def __init__(self, provider: ActionProvider, tool_slug: str, *,
                 name: str, description: str):
        self.provider = provider
        self.tool_slug = tool_slug
        self.name = name
        self.description = description

    def run(self, *, user_id: Optional[str] = None,
            connected_account_id: Optional[str] = None,
            **arguments: Any) -> ToolResult:
        if not self.provider.available():
            return ToolResult(tool=self.name, ok=False,
                               error=f"{self.name}: provider not configured")
        return self.provider.execute(
            self.tool_slug, arguments, user_id=user_id,
            connected_account_id=connected_account_id,
        )
