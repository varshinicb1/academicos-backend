"""Graph models: canonical node/edge payloads and version envelope."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Provenance
from .enums import EdgeType, GraphStatus, NodeType


class Node(BaseModel):
    id: str
    type: NodeType
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: GraphStatus = GraphStatus.DRAFT
    version: int = 1
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    provenance: Optional[Provenance] = None
    created_at: str = ""
    updated_at: str = ""


class Edge(BaseModel):
    id: str
    source: str                     # node id
    target: str                     # node id
    type: EdgeType
    attributes: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    status: GraphStatus = GraphStatus.DRAFT
    version: int = 1
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    provenance: Optional[Provenance] = None
    created_at: str = ""
    updated_at: str = ""


class GraphSnapshot(BaseModel):
    """Versioned memory: a labeled graph state."""
    snapshot_id: str
    created_at: str
    parent_snapshot_id: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    notes: str = ""
