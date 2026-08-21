"""Append-only audit log -- the server-side counterpart to the Dart offline
engine's LocalStore.appendAuditLog. Every real grading write and every real
paper export gets a timestamped, immutable record here: who did what, to
which assessment, when. This is the concrete control docs/compliance.md
calls "corruption resistance" and the log-retention half of CERT-In's
6-month requirement.

Local SQLite by default; same Supabase-when-configured pattern as every
other store in this codebase (see supabase_kv.py's module docstring for why:
Render's free tier wipes local disk on every restart, so a durable audit
trail needs the same real persistence everything else got).

Nothing in this module ever UPDATEs or DELETEs a row through normal
application code -- that's the point. If a real deployment ever needs to
prune old entries, that should be a deliberate, logged, admin-only action,
not something any route handler can do implicitly.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .supabase_kv import SupabaseTable

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
  id            TEXT PRIMARY KEY,
  timestamp     TEXT NOT NULL,
  action        TEXT NOT NULL,
  assessment_id TEXT,
  student_id    TEXT,
  actor         TEXT,
  details       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_assessment ON audit_log(assessment_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""


class AuditLog:
    def __init__(self, db_path: Path):
        self._remote = SupabaseTable("audit_log")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def append(self, action: str, *, assessment_id: str | None = None,
              student_id: str | None = None, actor: str | None = None,
              details: dict[str, Any] | None = None) -> str:
        entry_id = f"audit_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {})
        if self._remote.enabled:
            self._remote.upsert({
                "id": entry_id, "timestamp": timestamp, "action": action,
                "assessment_id": assessment_id, "student_id": student_id,
                "actor": actor, "details": details or {},
            }, on_conflict="id")
            return entry_id
        self.conn.execute(
            """INSERT INTO audit_log (id, timestamp, action, assessment_id, student_id, actor, details)
               VALUES (?,?,?,?,?,?,?)""",
            (entry_id, timestamp, action, assessment_id, student_id, actor, details_json),
        )
        self.conn.commit()
        return entry_id

    def for_assessment(self, assessment_id: str) -> list[dict[str, Any]]:
        if self._remote.enabled:
            return self._remote.select(assessment_id=assessment_id, order="timestamp.desc")
        rows = self.conn.execute(
            "SELECT * FROM audit_log WHERE assessment_id=? ORDER BY timestamp DESC",
            (assessment_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def for_action(self, action: str) -> list[dict[str, Any]]:
        if self._remote.enabled:
            return self._remote.select(action=action, order="timestamp.desc")
        rows = self.conn.execute(
            "SELECT * FROM audit_log WHERE action=? ORDER BY timestamp DESC",
            (action,)).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["details"] = json.loads(d.get("details") or "{}")
    return d


_INSTANCE: Optional[AuditLog] = None


def get_audit_log(data_root: Path) -> AuditLog:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AuditLog(data_root / "audit" / "audit_log.sqlite")
    return _INSTANCE
