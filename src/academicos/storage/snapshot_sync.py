"""SnapshotSync: keep one local SQLite file in sync with one remote blob.

CurriculumStore keeps 18 FK-linked tables in a local SQLite file and persists
the whole file as one object (see curriculum/store.py's module docstring for
why it is not wrapped per table). Everything that makes that durable lives
here, not in the curriculum domain class, because it is storage mechanics:

  - restore on start: a fresh container downloads the last snapshot before the
    connection opens, and refuses to start empty on anything but a genuine
    "no snapshot yet";
  - debounced upload, with a trailing timer so the debounce only delays;
  - one upload at a time, in commit order;
  - the generation-guarded single writer (storage/blobs.py refuses to replace
    a snapshot this instance did not load), and what happens after a refusal:
    conflict copy, reload, or wedge;
  - flush on shutdown (flush_all_snapshots, called from api/main.py).

The owner keeps its connection and its lock. It constructs a SnapshotSync
before opening the connection (the constructor is the restore) and routes
every commit through SnapshotSync.commit(conn). That is the whole interface,
so curriculum/store.py carries a constructor line and a one-line _commit
hook and nothing else -- another stream edits that file heavily.

The wording of errors and the empty-boot switch are the curriculum
snapshot's: it is the only snapshot there is. `CurriculumNotDurable` (blobs.py)
subclasses SupabaseUnavailable, so a route that meets it answers 503.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Optional

import requests

from .blobs import (
    BlobConflict,
    CurriculumNotDurable,
    durable_blob_store,
    is_not_found,
    record_conflict_copy,
    record_upload_failure,
)

logger = logging.getLogger(__name__)

_CONTENT_TYPE = "application/x-sqlite3"

# Every live sync, so the shutdown hook can flush them without knowing who
# owns them. Weak: a sync whose owner is gone has nothing left to flush.
_LIVE: "weakref.WeakSet[SnapshotSync]" = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


class SnapshotSync:
    def __init__(self, purpose: str, key: str, db_path: Path,
                 conn_lock: threading.RLock, *, debounce_seconds: float = 30.0):
        """Restores `key` from the `purpose` blob store into `db_path` when the
        file does not exist yet. Call BEFORE opening the connection.

        `conn_lock` is the owner's connection lock (an RLock: the owner's
        helpers take it per call and this class nests inside it). Everything
        here that touches the connection holds it."""
        self.remote = durable_blob_store(purpose)
        self.purpose = purpose
        self.key = key
        self.db_path = db_path
        self.debounce_seconds = debounce_seconds
        self._conn_lock = conn_lock
        self._conn: Optional[sqlite3.Connection] = None
        self._last_upload_at = 0.0
        # _commit_seq counts commits that changed rows; _snapshot_seq is the
        # count the last successful upload contained. Unequal means
        # "something is not in the remote snapshot yet".
        self._committed_changes = 0
        self._commit_seq = 0
        self._snapshot_seq = 0
        self._timer: Optional[threading.Timer] = None
        self._upload_lock = threading.Lock()
        # An allowed empty start (below) has nothing to upload until someone
        # edits a curriculum; save_empty_boot() publishes it instead.
        self._empty_boot = False
        self._wedged = False
        self._closed = False

        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.exists() and self.remote.enabled:
            self._restore()
        with _LIVE_LOCK:
            _LIVE.add(self)

    # ------------------------------------------------------------------
    # restore on start
    # ------------------------------------------------------------------

    def _restore(self) -> None:
        """A fresh container restoring last known state instead of starting
        empty. No snapshot yet (first-ever boot) is the normal case, not an
        error: the owner then creates its schema in an empty file.

        Any OTHER failure raises. It used to be logged at INFO and treated
        exactly like "none yet", so a transient outage at cold start served
        an empty curriculum as if it were the real one; an admin re-entering
        it then lost everything at the next deploy. Raising fails app
        startup (Cloud Run retries the container) or answers 503 at the
        route, and leaves no local file behind, so the next attempt restores
        properly."""
        try:
            data = self.remote.download(self.key)
        except requests.exceptions.RequestException as exc:
            if is_not_found(exc):
                # On GCS, "no object" is ALSO what a service migrated off
                # Supabase sees until someone copies the old snapshot across,
                # and starting empty there serves a blank curriculum as if it
                # were the school's. So an empty GCS boot must be asked for.
                if (getattr(self.remote, "backend", None) == "gcs"
                        and os.environ.get("ACOS_CURRICULUM_ALLOW_EMPTY_BOOT", "").strip() != "1"):
                    logger.error("No curriculum snapshot in GCS; refusing to start "
                                 "empty without ACOS_CURRICULUM_ALLOW_EMPTY_BOOT=1")
                    raise CurriculumNotDurable(
                        "no curriculum snapshot in GCS. If this school's curriculum "
                        "lived in Supabase Storage, copy it with "
                        "scripts/migrate_supabase_blobs_to_gcs.py; if this is a new "
                        "school with no curriculum yet, set "
                        "ACOS_CURRICULUM_ALLOW_EMPTY_BOOT=1 for the first boot") from exc
                logger.info("No curriculum snapshot uploaded yet -- starting with "
                            "a fresh local database")
                self._empty_boot = True
                return
            logger.error("Curriculum snapshot restore failed; refusing to start "
                         "with an empty curriculum", exc_info=True)
            raise CurriculumNotDurable(
                f"curriculum snapshot could not be restored ({exc}); not starting "
                f"empty in its place") from exc
        self.db_path.write_bytes(data)
        logger.info("Restored %s from its remote snapshot (%d bytes)", self.db_path, len(data))

    # ------------------------------------------------------------------
    # commit hook + debounced upload. The debounce limits how often the
    # whole file goes to GCS; it must never decide WHETHER a commit gets
    # there. It used to: a commit inside the window simply returned, with no
    # dirty flag and no trailing upload, so a burst of admin edits reached
    # GCS as its first edit only and the rest died with the container. Now a
    # commit inside the window arms one trailing timer for the end of it, a
    # failed upload re-arms it, and api/main.py's shutdown hook calls
    # flush_all_snapshots() inside Cloud Run's 10 s SIGTERM grace.
    # ------------------------------------------------------------------

    def commit(self, conn: sqlite3.Connection) -> None:
        """Commit `conn` and schedule the upload. The owner's only hook.

        The commit and the row-change count happen under one hold of the
        connection lock. Splitting them would let another thread execute
        rows in between, get counted by this commit before they are
        committed, and then never mark the snapshot pending on their own
        commit. The upload itself runs outside the lock: it takes the upload
        lock first, and a request thread holding the connection lock while
        waiting for the upload lock would deadlock against a timer upload
        waiting for the connection lock."""
        with self._conn_lock:
            conn.commit()
            self._conn = conn
            # One shared connection, so commit() commits every thread's
            # pending rows; total_changes then says whether any row changed.
            # A commit that changed nothing (schema no-ops at boot) must not
            # mark the snapshot pending: re-uploading an unchanged restored
            # database would bump the generation under an overlapping
            # instance and turn a rollout into a conflict for nothing.
            if conn.total_changes != self._committed_changes:
                self._committed_changes = conn.total_changes
                self._commit_seq += 1
        self._maybe_upload()

    def save_empty_boot(self, conn: sqlite3.Connection) -> None:
        """Publish the empty database this service started with, once.

        A schema-only start changes no row, so `commit` marks nothing pending
        and nothing is uploaded until someone edits a curriculum. The live GCP
        service booted that way on 2026-09-22 with
        ACOS_CURRICULUM_ALLOW_EMPTY_BOOT=1; the next deploy then refused to
        start ("no curriculum snapshot in GCS", revision 00003), because the
        acknowledgement is a first-boot variable, not a standing one. Saving
        the empty snapshot here keeps it a first-boot variable.

        Called by the store's owner (get_curriculum_store), not by __init__:
        constructing a store must not touch the remote."""
        if not self._empty_boot:
            return
        self._empty_boot = False
        with self._conn_lock:
            self._conn = conn
            self._commit_seq += 1
        self.flush()

    def _pending(self) -> bool:
        return self._commit_seq != self._snapshot_seq

    def _maybe_upload(self) -> None:
        if not self.remote.enabled:
            return
        if self._wedged:
            raise CurriculumNotDurable(
                "curriculum snapshot conflict: another instance wrote a newer "
                "snapshot and it could not be reloaded; this edit is on this "
                "instance's disk only and will be lost at restart")
        with self._conn_lock:
            if not self._pending():
                return
            wait = self.debounce_seconds - (time.monotonic() - self._last_upload_at)
            if wait > 0:
                self._arm_timer(wait)
                return
        self.flush(raise_on_conflict=True)

    def _arm_timer(self, delay: float) -> None:
        """One trailing upload at most; it picks up every commit before it
        fires. Caller holds the connection lock."""
        if self._timer is not None or self._closed:
            return
        timer = threading.Timer(max(delay, 0.0), self._trailing)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _trailing(self) -> None:
        with self._conn_lock:
            self._timer = None
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - a timer thread has nobody to raise to
            logger.exception("Trailing snapshot upload failed for %s", self.db_path)

    def flush(self, *, raise_on_conflict: bool = False) -> bool:
        """Upload now if any commit is not in the remote snapshot yet.

        True when nothing is left pending. On a plain upload failure the
        trailing timer is re-armed, so the edit is retried rather than
        dropped. On a conflict see _on_conflict; `raise_on_conflict` is set
        on the request path so the edit that could not be saved answers 503
        instead of a false success."""
        if not self.remote.enabled:
            return True
        with self._upload_lock:   # uploads in commit order, never two at once
            with self._conn_lock:
                if self._closed or self._conn is None or not self._pending():
                    return True
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                self._last_upload_at = time.monotonic()
                seq = self._commit_seq
                try:
                    # WAL mode means recent commits can still live only in the
                    # -wal sidecar file -- checkpoint first so db_path itself is
                    # a complete, self-contained snapshot, not a stale main file
                    # missing whatever hasn't been checkpointed back into it yet.
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    data = self.db_path.read_bytes()
                except OSError:
                    logger.warning("Could not read %s to snapshot", self.db_path,
                                   exc_info=True)
                    self._arm_timer(self.debounce_seconds)
                    return False
            try:
                self.remote.upload(self.key, data, _CONTENT_TYPE)
            except BlobConflict as exc:
                self._on_conflict()
                if raise_on_conflict:
                    if self._wedged:
                        raise CurriculumNotDurable(
                            "curriculum snapshot conflict: another instance saved a "
                            "newer curriculum and this instance could not preserve "
                            "or reload; this edit is not saved and curriculum writes "
                            "are paused until an operator intervenes") from exc
                    raise CurriculumNotDurable(
                        "curriculum snapshot conflict: another instance saved a newer "
                        "curriculum, so this edit was not saved. This instance has "
                        "reloaded the newer one (its unsaved edits are kept under "
                        "curriculum-snapshots/conflicts/ for an operator to merge); "
                        "retry the edit.") from exc
                return False
            except requests.exceptions.RequestException:
                record_upload_failure(self.purpose)
                logger.warning("Failed to upload the %s snapshot; retrying in %.0f s",
                               self.purpose, self.debounce_seconds, exc_info=True)
                with self._conn_lock:
                    self._arm_timer(self.debounce_seconds)
                return False
            with self._conn_lock:
                # Commits that landed during the upload are not in `data`:
                # they stay pending and get their own trailing upload.
                self._snapshot_seq = seq
                if self._pending():
                    self._arm_timer(self.debounce_seconds)
            return True

    # ------------------------------------------------------------------
    # single writer: after a refusal
    # ------------------------------------------------------------------

    def _on_conflict(self) -> None:
        """Another instance saved a newer snapshot (blobs.py refused to erase
        it). Keeping this instance's copy would leave it stale for good: its
        generation never refreshes, so every later upload 412s while every
        request reports success -- the "stuck and silent" state. So this
        instance adopts the newer snapshot.

        But what it has not uploaded includes commits that already answered
        200: the debounce deferred them, and a timer or shutdown flush is
        where they meet the conflict. Reloading used to discard them -- an
        edit the admin was told was saved then existed nowhere. So FIRST the
        whole losing file goes to `conflicts/<utc>-<instance>.sqlite` in the
        same store; only once that upload has succeeded is the newer snapshot
        reloaded into the live connection (sqlite3 backup API, no reopen).
        The download also refreshes the generation blobs.py holds, so the
        next upload is accepted.

        All of it runs under the connection lock: a commit landing between
        the copy and the reload would be in neither. A conflict is rare
        (rollout overlap), and a few seconds of queued requests is the price.

        If the copy cannot be saved, nothing is reloaded: the sync wedges
        (every later write answers 503) and the losing edits stay on this
        instance's disk, still readable, rather than being destroyed. A
        failed reload wedges it too."""
        with self._conn_lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                losing = self.db_path.read_bytes()
                copy_key = self._preserve_conflict_copy(losing)
            except Exception:  # noqa: BLE001 - unpreserved means "do not reload"
                self._wedged = True
                logger.error("Curriculum snapshot conflict, and this instance's "
                             "unsaved edits could not be preserved in GCS; NOT "
                             "reloading, so they stay in %s. Curriculum writes now "
                             "answer 503 until an operator recovers that file and "
                             "restarts", self.db_path, exc_info=True)
                return
            try:
                data = self.remote.download(self.key)
                self._load_snapshot_bytes(data)
                self._committed_changes = self._conn.total_changes
                self._snapshot_seq = self._commit_seq
            except Exception:  # noqa: BLE001 - any failure here means "wedged"
                self._wedged = True
                logger.error("Curriculum snapshot conflict and the newer snapshot could "
                             "not be reloaded; curriculum writes now answer 503 until "
                             "this instance restarts (this instance's edits are "
                             "preserved at %s/%s)", self.purpose, copy_key,
                             exc_info=True)
                return
        logger.error("Curriculum snapshot conflict: this instance's edits not yet in "
                     "GCS were preserved at %s/%s (%d bytes) and then replaced in the "
                     "live view by the newer snapshot (%d bytes). Merge them from that "
                     "copy.", self.purpose, copy_key, len(losing), len(data))

    def _preserve_conflict_copy(self, data: bytes) -> str:
        """Upload the losing file under a key no other writer can hold: UTC
        time, the Cloud Run revision, and a per-call random suffix. For the
        generation-guarded GCS store a never-seen key uploads with
        if_generation_match=0, so even a collision could not overwrite."""
        instance = os.environ.get("K_REVISION", "").strip() or "local"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        key = f"conflicts/{stamp}-{instance}-{uuid.uuid4().hex[:8]}.sqlite"
        self.remote.upload(key, data, _CONTENT_TYPE)
        record_conflict_copy(self.purpose, key)
        return key

    def _load_snapshot_bytes(self, data: bytes) -> None:
        """Replace the live database with `data` via the backup API. Caller
        holds the connection lock. The source is a temporary file of our own,
        but it is a connection like any other, so AGENTS.md section 3's
        pragmas apply to it too (tests/test_sqlite_pragmas.py)."""
        incoming = self.db_path.with_name(self.db_path.name + ".incoming")
        incoming.write_bytes(data)
        src = sqlite3.connect(str(incoming))
        try:
            src.execute("PRAGMA busy_timeout=60000")
            src.execute("PRAGMA journal_mode=WAL")
            src.backup(self._conn)
        finally:
            src.close()
            for leftover in (incoming, incoming.with_name(incoming.name + "-wal"),
                             incoming.with_name(incoming.name + "-shm")):
                leftover.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Best-effort final upload, then no timer may touch the connection
        the owner is about to close. Never raises."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - closing must not raise
            logger.warning("Final snapshot upload on close failed for %s",
                           self.db_path, exc_info=True)
        with self._conn_lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        with _LIVE_LOCK:
            _LIVE.discard(self)


def flush_all_snapshots() -> None:
    """Upload every live sync's pending snapshot now. Called from the app's
    shutdown hook: Cloud Run sends SIGTERM and allows 10 s, and a commit still
    inside the debounce window would otherwise die with the container."""
    with _LIVE_LOCK:
        syncs = list(_LIVE)
    for sync in syncs:
        try:
            sync.flush()
        except Exception:  # noqa: BLE001 - one store must not block the rest
            logger.exception("Shutdown snapshot flush failed for %s", sync.db_path)
