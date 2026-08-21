"""Source registry: SQLite-backed ledger of every ingested source.

Tracks canonical ids, checksums, provenance chain, and retract/supersede state.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id      TEXT PRIMARY KEY,
  file_key       TEXT NOT NULL,          -- object-store key of the artifact
  sha256         TEXT,
  sha512         TEXT,
  size_bytes     INTEGER,
  source_url     TEXT,
  fetched_at     TEXT,
  kind           TEXT,                   -- 'file' | 'zip' | 'web' | 'scan'
  doc_type       TEXT,
  title          TEXT,
  subject        TEXT,
  grade          TEXT,
  academic_year  TEXT,
  series         TEXT,
  page_count     INTEGER,
  status         TEXT DEFAULT 'registered',  -- registered|parsed|extracted|failed|retracted
  quality_score  REAL DEFAULT 0,
  notes          TEXT,
  meta           TEXT,                   -- json
  registered_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(doc_type);
"""


class SourceRegistry:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def register(self, source_id: str, file_key: str, **fields: Any) -> None:
        cols = ["source_id", "file_key", "registered_at", "meta"]
        vals: dict[str, Any] = dict(fields)
        meta = vals.pop("meta", None)
        from datetime import datetime, timezone
        values = {
            "source_id": source_id,
            "file_key": file_key,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "meta": json.dumps(meta) if meta else None,
        }
        for k, v in vals.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            values[k] = v
        keys = list(values.keys())
        placeholders = ",".join("?" * len(keys))
        cols_sql = ",".join(keys)
        self.conn.execute(
            f"INSERT INTO sources ({cols_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(source_id) DO UPDATE SET file_key=excluded.file_key, status='registered'",
            list(values.values()),
        )
        self.conn.commit()

    def update(self, source_id: str, **fields: Any) -> None:
        sets = []
        vals: list[Any] = []
        for k, v in fields.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(source_id)
        self.conn.execute(f"UPDATE sources SET {', '.join(sets)} WHERE source_id=?", vals)
        self.conn.commit()

    def get(self, source_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
        return dict(row) if row else None

    def by_status(self, status: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM sources WHERE status=?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def all(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM sources ORDER BY registered_at").fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]

    def close(self) -> None:
        self.conn.close()
