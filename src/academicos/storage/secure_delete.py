"""Deletes that leave nothing behind in the SQLite files.

An ordinary DELETE only marks the row's space free: the bytes stay in the file
until SQLite reuses the page, and every snapshot of the file (the operations
and curriculum stores upload theirs to the blob store) carries them along.
`PRAGMA secure_delete` makes SQLite overwrite the freed content with zeros.
Some builds have it on by default (Debian's), others not (the SQLite Python
bundles on Windows), so erasure (assessment/erasure.py) turns it on for its
own deletes and puts the connection back as it was.

That is not enough in WAL mode, which every store here uses: the WAL file
keeps the earlier versions of each page, the deleted row included, until a
checkpoint copies the pages back and the file is reused. An end-to-end run
found a student's erased answers still in operations.sqlite-wal. So after
the deletes are committed, `truncate_wal` checkpoints every frame into the
database (whose pages now hold zeros) and truncates the WAL to nothing.

Call both while holding the store's connection lock, so no other statement
runs on the connection meanwhile.
"""
from __future__ import annotations

import contextlib
import logging
import sqlite3
from typing import Iterator

log = logging.getLogger(__name__)

# What `PRAGMA secure_delete` reads back (0, 1, 2) as the value that sets it
# again: 2 is FAST, which an integer does not set.
_SETTING = {0: "OFF", 1: "ON", 2: "FAST"}


@contextlib.contextmanager
def secure_delete(conn: sqlite3.Connection) -> Iterator[None]:
    before = conn.execute("PRAGMA secure_delete").fetchone()[0]
    conn.execute("PRAGMA secure_delete=ON")
    try:
        yield
    finally:
        conn.execute(f"PRAGMA secure_delete={_SETTING.get(int(before), 'OFF')}")


def truncate_wal(conn: sqlite3.Connection) -> bool:
    """Checkpoint the whole WAL into the database and empty it. Call after
    the commit. False when another connection's reader kept a frame in use
    (logged): the frames go at the next checkpoint instead."""
    busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if busy:
        log.warning("the WAL could not be emptied after an erasure: another connection is reading")
    return not busy
