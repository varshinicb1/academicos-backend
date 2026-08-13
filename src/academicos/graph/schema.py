"""Graph schema: canonical id scheme, node/edge factories, validity rules."""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from ..models.common import utcnow
from ..models.enums import EdgeType, GraphStatus, NodeType
from ..models.graph import Edge, Node


def node_id(ntype: NodeType, *parts: str) -> str:
    joined = ":".join(p for p in parts if p)
    return f"cbse:{ntype.value.lower()}:{joined}"


def edge_id(etype: EdgeType, source: str, target: str) -> str:
    digest = hashlib.sha1(f"{source}|{target}|{etype.value}".encode()).hexdigest()[:16]
    return f"edge:{digest}"


def new_node(ntype: NodeType, label: str, id_: str | None = None,
             attributes: Optional[dict[str, Any]] = None, status: GraphStatus = GraphStatus.DRAFT,
             provenance=None) -> Node:
    now = utcnow()
    return Node(
        id=id_ or node_id(ntype, label),
        type=ntype,
        label=label,
        attributes=attributes or {},
        status=status,
        provenance=provenance,
        created_at=now,
        updated_at=now,
    )


def new_edge(etype: EdgeType, source: str, target: str, weight: float = 1.0,
             attributes: Optional[dict[str, Any]] = None, provenance=None) -> Edge:
    now = utcnow()
    return Edge(
        id=edge_id(etype, source, target),
        source=source,
        target=target,
        type=etype,
        weight=weight,
        attributes=attributes or {},
        provenance=provenance,
        created_at=now,
        updated_at=now,
    )


def retract(obj: Node | Edge) -> None:
    obj.status = GraphStatus.RETRACTED
    obj.valid_until = utcnow()
