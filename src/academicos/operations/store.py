"""The operations store: one SQLite file (WAL, 60 s busy timeout, one
serialized connection), restored from and snapshotted to the blob store like
CurriculumStore. Feature mixins add their tables and methods."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..storage.snapshot_sync import SnapshotSync
from .notifications import NOTIFICATIONS_SCHEMA, NotificationsMixin


class OperationsStore(NotificationsMixin):
    _SNAPSHOT_KEY = "operations.sqlite"
    _SNAPSHOT_DEBOUNCE_SECONDS = 20.0

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn_lock = threading.RLock()
        self._snapshots = SnapshotSync("operations-snapshots", self._SNAPSHOT_KEY, db_path, self._conn_lock,
                                       debounce_seconds=self._SNAPSHOT_DEBOUNCE_SECONDS,
                                       on_reload=self._bring_up_to_date, allow_empty_boot=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._exec("PRAGMA journal_mode=WAL")
        self._exec("PRAGMA busy_timeout=60000")
        self._bring_up_to_date()
        self._snapshots.commit_derived(self.conn)

    def _bring_up_to_date(self) -> None:
        with self._conn_lock:
            self.conn.executescript(self.schema())

    @classmethod
    def schema(cls) -> str:
        return NOTIFICATIONS_SCHEMA

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._conn_lock:
            self.conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self._conn_lock:
            r = self.conn.execute(sql, params).fetchone()
            return dict(r) if r is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn_lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def _commit(self) -> None:
        self._snapshots.commit(self.conn)

    def close(self) -> None:
        self._snapshots.close()
        self.conn.close()


_INSTANCES: dict[str, OperationsStore] = {}
_LOCK = threading.Lock()


def get_operations_store(data_root: Path) -> OperationsStore:
    key = str(Path(data_root) / "operations" / "operations.sqlite")
    with _LOCK:
        store = _INSTANCES.get(key)
        if store is None:
            store = OperationsStore(Path(key))
            store._snapshots.save_empty_boot(store.conn)
            _INSTANCES[key] = store
        return store
