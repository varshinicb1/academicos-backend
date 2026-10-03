"""What a person has done in the product's own guidance, kept for them.

The Guide's jobs, the first-time welcome and the setup wizard used to remember on the
device (a browser's storage), so a new browser, a new phone or a cleared cache started
again and nobody could say how far a school had got. This is the server's record, per
user: a key (`guide.done.setup`, `welcome`, ...) and a small JSON value, last write wins.

It holds a person's own state and nothing about anyone else, so erasure deletes it
(`forget.py`); the setup level of a whole school is computed from the school's own data
(`progress_routes.setup_status`), never stored here.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

PROGRESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_progress (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,59}$")
MAX_KEYS_PER_USER = 300
MAX_VALUE_BYTES = 2048


class ProgressRefused(ValueError):
    """The key or value is not one we keep; the message is for the person."""


class ProgressMixin:
    """Added to OperationsStore."""

    def progress_of(self, user_id: str) -> dict[str, dict[str, Any]]:
        return {r["key"]: {"value": json.loads(r["value"]), "updatedAt": r["updated_at"]}
                for r in self._fetchall("SELECT key, value, updated_at FROM user_progress WHERE user_id=?",
                                        (user_id,))}

    def set_progress(self, user_id: str, key: str, value: Any) -> dict[str, Any]:
        if not KEY_PATTERN.match(key):
            raise ProgressRefused("a progress key is lowercase letters, digits and . _ : - (60 at most)")
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_VALUE_BYTES:
            raise ProgressRefused(f"a progress value is at most {MAX_VALUE_BYTES} bytes")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn_lock:
            have = self.conn.execute("SELECT COUNT(*) FROM user_progress WHERE user_id=?", (user_id,)).fetchone()[0]
            exists = self.conn.execute("SELECT 1 FROM user_progress WHERE user_id=? AND key=?",
                                       (user_id, key)).fetchone()
            if not exists and have >= MAX_KEYS_PER_USER:
                raise ProgressRefused(f"at most {MAX_KEYS_PER_USER} progress entries per person")
            self.conn.execute(
                "INSERT INTO user_progress (user_id, key, value, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (user_id, key, encoded, now))
            self._commit()
        return {"value": value, "updatedAt": now}

    def forget_progress(self, user_id: str) -> int:
        with self._conn_lock:
            n = self.conn.execute("DELETE FROM user_progress WHERE user_id=?", (user_id,)).rowcount
            self._commit()
        return n
