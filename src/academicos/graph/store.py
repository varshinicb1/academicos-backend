"""SQLite graph store: versioned nodes/edges with snapshot semantics."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..models.enums import EdgeType, GraphStatus, NodeType
from ..models.graph import Edge, GraphSnapshot, Node

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  label TEXT NOT NULL,
  attributes TEXT,
  status TEXT NOT NULL DEFAULT 'draft',
  version INTEGER DEFAULT 1,
  valid_from TEXT,
  valid_until TEXT,
  provenance TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS edges (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL REFERENCES nodes(id),
  target TEXT NOT NULL REFERENCES nodes(id),
  type TEXT NOT NULL,
  attributes TEXT,
  weight REAL DEFAULT 1.0,
  status TEXT NOT NULL DEFAULT 'draft',
  version INTEGER DEFAULT 1,
  valid_from TEXT,
  valid_until TEXT,
  provenance TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  created_at TEXT,
  parent_snapshot_id TEXT,
  node_count INTEGER,
  edge_count INTEGER,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
"""


class GraphStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- nodes ----
    def upsert_node(self, node: Node) -> None:
        self.conn.execute(
            """INSERT INTO nodes (id, type, label, attributes, status, version,
                                  valid_from, valid_until, provenance, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 label=excluded.label,
                 attributes=excluded.attributes,
                 status=excluded.status,
                 valid_until=excluded.valid_until,
                 provenance=excluded.provenance,
                 updated_at=excluded.updated_at,
                 version=version+1""",
            (node.id, node.type.value, node.label, _dumps(node.attributes),
             node.status.value, node.version, node.valid_from, node.valid_until,
             _dumps(node.provenance), node.created_at, node.updated_at),
        )
        self.conn.commit()

    def get_node(self, nid: str) -> Optional[Node]:
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (nid,)).fetchone()
        return _node_from_row(row) if row else None

    def query_nodes(self, ntype: Optional[NodeType] = None, label_contains: Optional[str] = None,
                    status: GraphStatus = GraphStatus.DRAFT, limit: int = 100) -> list[Node]:
        sql = "SELECT * FROM nodes WHERE 1=1"
        args: list[Any] = []
        if ntype:
            sql += " AND type=?"
            args.append(ntype.value)
        if status:
            sql += " AND status=?"
            args.append(status.value)
        if label_contains:
            sql += " AND label LIKE ?"
            args.append(f"%{label_contains}%")
        sql += " LIMIT ?"
        args.append(limit)
        return [_node_from_row(r) for r in self.conn.execute(sql, args).fetchall()]

    # ---- edges ----
    def upsert_edge(self, edge: Edge) -> None:
        self.conn.execute(
            """INSERT INTO edges (id, source, target, type, attributes, weight, status,
                                  version, valid_from, valid_until, provenance, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 weight=excluded.weight, attributes=excluded.attributes,
                 status=excluded.status, valid_until=excluded.valid_until,
                 provenance=excluded.provenance, updated_at=excluded.updated_at,
                 version=version+1""",
            (edge.id, edge.source, edge.target, edge.type.value, _dumps(edge.attributes),
             edge.weight, edge.status.value, edge.version, edge.valid_from, edge.valid_until,
             _dumps(edge.provenance), edge.created_at, edge.updated_at),
        )
        self.conn.commit()

    def get_edge(self, eid: str) -> Optional[Edge]:
        row = self.conn.execute("SELECT * FROM edges WHERE id=?", (eid,)).fetchone()
        return _edge_from_row(row) if row else None

    def neighbors(self, nid: str, max_depth: int = 2, max_nodes: int = 200) -> list[dict[str, Any]]:
        """BFS traversal returning [{node, edge, depth}] (edges via in/out adjacency)."""
        out: list[dict[str, Any]] = []
        visited: set[str] = {nid}
        frontier = [nid]
        for depth in range(1, max_depth + 1):
            nxt: list[str] = []
            for cur in frontier:
                rows = self.conn.execute(
                    """SELECT * FROM edges WHERE (source=? OR target=?) AND status='draft'""",
                    (cur, cur)).fetchall()
                for r in rows:
                    edge = _edge_from_row(r)
                    other = r["target"] if r["source"] == cur else r["source"]
                    if other in visited:
                        continue
                    visited.add(other)
                    node = self.get_node(other)
                    if node:
                        out.append({"node": node, "edge": edge, "depth": depth})
                    nxt.append(other)
            frontier = nxt
            if not frontier or len(out) >= max_nodes:
                break
        return out

    def paths(self, start: str, end: str, max_depth: int = 4) -> list[list[Edge]]:
        """Simple BFS edge-path enumeration (multi-hop, capped)."""
        results: list[list[Edge]] = []
        from collections import deque
        q: deque[tuple[str, list[Edge]]] = deque([(start, [])])
        seen = {start}
        while q:
            cur, path = q.popleft()
            if len(path) > max_depth:
                continue
            for r in self.conn.execute(
                    """SELECT * FROM edges WHERE (source=? OR target=?) AND status='draft'""",
                    (cur, cur)).fetchall():
                e = _edge_from_row(r)
                other = r["target"] if r["source"] == cur else r["source"]
                if other == end:
                    results.append(path + [e])
                    continue
                if other in seen:
                    continue
                seen.add(other)
                q.append((other, path + [e]))
            if len(results) >= 50:
                break
        return results

    def count(self) -> tuple[int, int]:
        n = self.conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        e = self.conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        return n, e

    def snapshot(self, parent: Optional[str] = None, notes: str = "") -> GraphSnapshot:
        n, e = self.count()
        snap = GraphSnapshot(
            snapshot_id=f"snap:{self.conn.execute('SELECT COUNT(*) c FROM snapshots').fetchone()['c'] + 1}",
            created_at=_now(),
            parent_snapshot_id=parent,
            node_count=n,
            edge_count=e,
            notes=notes,
        )
        self.conn.execute(
            "INSERT INTO snapshots VALUES (?,?,?,?,?,?)",
            (snap.snapshot_id, snap.created_at, snap.parent_snapshot_id, n, e, notes),
        )
        self.conn.commit()
        return snap

    def close(self) -> None:
        self.conn.close()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _dumps(v: Any) -> Optional[str]:
    if v is None:
        return None
    if hasattr(v, "model_dump"):
        return json.dumps(v.model_dump(mode="json"), ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False, default=str)


def _loads(v: Optional[str]) -> Any:
    if not v:
        return None
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _node_from_row(r: sqlite3.Row) -> Node:
    return Node(
        id=r["id"], type=NodeType(r["type"]), label=r["label"],
        attributes=_loads(r["attributes"]) or {},
        status=GraphStatus(r["status"]), version=r["version"],
        valid_from=r["valid_from"], valid_until=r["valid_until"],
        provenance=_loads(r["provenance"]),
        created_at=r["created_at"] or "", updated_at=r["updated_at"] or "",
    )


def _edge_from_row(r: sqlite3.Row) -> Edge:
    return Edge(
        id=r["id"], source=r["source"], target=r["target"], type=EdgeType(r["type"]),
        attributes=_loads(r["attributes"]) or {}, weight=r["weight"],
        status=GraphStatus(r["status"]), version=r["version"],
        valid_from=r["valid_from"], valid_until=r["valid_until"],
        provenance=_loads(r["provenance"]),
        created_at=r["created_at"] or "", updated_at=r["updated_at"] or "",
    )
