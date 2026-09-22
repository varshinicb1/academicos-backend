"""One factory for the app's blob stores, mirroring postgres_kv.durable_table.

Two call sites keep binary blobs that are school data: scan-session photos and
PDFs (assessment/mobile_scan.py, name "scan-media") and the curriculum store's
whole-file SQLite snapshot (storage/snapshot_sync.py, name "curriculum-snapshots").
Both were built against SupabaseStorage's three-member interface -- `enabled`,
`upload(key, data, content_type)`, `download(key) -> bytes` -- and
`durable_blob_store(name)` returns an object with exactly that interface, so
neither call site changes beyond its constructor line.

Precedence, in order:
  1. GCS, when ACOS_GCS_BUCKET is set. On Cloud Run the container disk is
     wiped on every redeploy and scale-to-zero, so this is the only backend
     under which a GCP deployment keeps these blobs at all.
  2. SupabaseStorage, when the Supabase credentials are set (the Render path).
  3. Otherwise the same SupabaseStorage object, disabled. That is the "local"
     backend: both call sites already treat `enabled is False` as "keep the
     file on local disk only", which is the right behaviour for development.
     Returning the disabled object rather than a new local class keeps the
     existing tests that poke `_url`/`_key` on it working unchanged.

Only the environment variable selects GCS -- not config/config.toml's
`gcs_bucket`. That TOML value names a developer bucket for the CLI corpus
sync (cli.py `sync`); reading it here would send every local run's scan
photos into it, or crash on machines without the `gcs` extra.

Objects live under `<ACOS_GCS_PREFIX or "academicos">/<name>/<key>`, one
prefix per purpose, so a bucket lifecycle or IAM rule can target scan media
without touching the snapshots.

Single writer for the curriculum snapshot
-----------------------------------------
The snapshot is ONE object holding the whole SQLite file. If two Cloud Run
instances each hold a local copy, each debounced upload replaces the other's
object wholesale: whatever the other instance wrote since it last restored is
gone, silently. deploy-gcp.yml pins the service to one instance for that
reason; this module is the second line of defence. A store created with
`guard_generations=True` remembers the GCS generation of every object it read
or wrote and uploads with `if_generation_match` set to it (0 -- "must not
exist" -- for an object it never saw). A snapshot someone else wrote in the
meantime therefore fails with HTTP 412, and the upload raises BlobConflict,
logged at ERROR, instead of erasing it. Refusing is the lesser loss: the
alternative destroys the other instance's writes with no trace at all.

Two instances overlap even with --max-instances=1: the limit is per revision,
so every rollout runs the old and new revision side by side for a while.

What happens after a refusal is the snapshot sync's job (storage/snapshot_sync.py
`SnapshotSync._on_conflict`): it first uploads its own losing file, whole, to
`conflicts/<utc>-<instance>.sqlite` in this store, then reloads the newer
snapshot -- which also refreshes the generation this module holds, so the
store is not wedged into 412-ing forever. The request that triggered it (if
any) answers 503; the conflict and the copy's key are recorded here for
/health/storage. A refusal therefore never destroys an edit: edits that were
debounced and already answered 200 are in the conflict copy, to be merged by
hand. It does take them out of the live view until then.

A 412 is not always another writer. With if_generation_match set,
google-cloud-storage retries the upload; when the first attempt landed but
its response was lost, the retry meets the generation it just created. Before
declaring a conflict, upload() re-reads the object's metadata and, if its
md5 is the md5 of the bytes just sent, adopts that generation as its own
write.

Status, process-wide: `blob_status()` is what /health/storage shows beside
blob_backend -- snapshot conflicts, failed snapshot uploads and failed
scan-media uploads, each with a count and the last time. A failed scan upload
is still best-effort for the teacher (mobile_scan.py), but on GCP it means
that artifact is on container disk only, so it has to be visible somewhere.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

from ..assessment.supabase_kv import SupabaseUnavailable
from ..config import credential_or_none

logger = logging.getLogger(__name__)

# The names the call sites use; the snapshot store is the only one that needs
# the single-writer precondition (scan keys are unique per session, and a
# re-captured page must overwrite its own earlier upload).
_GUARDED = frozenset({"curriculum-snapshots"})


class BlobUnavailable(requests.exceptions.RequestException):
    """A GCS read or write failed, or the object does not exist.

    A RequestException subclass on purpose, like SupabaseUnavailable: the call
    sites were written against SupabaseStorage, whose failures are requests
    errors, and storage/snapshot_sync.py catches exactly that type on both its
    restore and its snapshot path."""


class BlobNotFound(BlobUnavailable):
    """The object does not exist. Split from BlobUnavailable because the two
    mean opposite things at boot: "no snapshot yet" is a first boot and may
    start empty; "GCS did not answer" must not, or an outage at cold start
    serves an empty curriculum as if it were the real one."""


class BlobConflict(BlobUnavailable):
    """A guarded upload was refused: the object changed since this process
    last read or wrote it (GCS precondition failure, HTTP 412)."""


class CurriculumNotDurable(SupabaseUnavailable):
    """A curriculum write or boot could not be made durable, and saying
    "saved" would be a lie: the snapshot restore failed for a reason other
    than "none yet", or a snapshot upload was refused because another
    instance wrote a newer one.

    A SupabaseUnavailable subclass for the same reason PostgresUnavailable
    is one: api/main.py's existing handler turns it into a 503 with this
    message, so a route cannot forget to."""

    def __init__(self, detail: str):
        Exception.__init__(self, detail)  # bypass SupabaseUnavailable's wording
        self.table = "curriculum-snapshots"
        self.op = "snapshot"
        self.status = None
        self.body = detail


def is_not_found(exc: BaseException) -> bool:
    """True when a blob store's download failure means "no such object".

    BlobNotFound for GCS. SupabaseStorage raises the raw HTTPError: a plain
    404 on newer Storage deployments, a 400 whose body says not_found on
    older ones. Anything else -- a connection error, a timeout, a 5xx -- is
    an outage, not an absence."""
    if isinstance(exc, BlobNotFound):
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 404:
        return True
    if status == 400:
        body = (getattr(response, "text", "") or "").lower()
        return "not_found" in body or "not found" in body
    return False


# --- status for /health/storage ----------------------------------------------

_status_lock = threading.Lock()
_status: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(event: str, name: str) -> None:
    with _status_lock:
        entry = _status.setdefault(f"{name}:{event}", {"count": 0, "at": None})
        entry["count"] += 1
        entry["at"] = _now()


def record_conflict(name: str) -> None:
    """A guarded upload of `name` was refused because of a real concurrent
    writer (not our own retried request -- see upload())."""
    _record("conflict", name)


def record_upload_failure(name: str) -> None:
    """An upload of `name` failed for any reason other than a conflict: the
    bytes are on container disk only until a later upload succeeds."""
    _record("upload_failure", name)


def record_conflict_copy(name: str, key: str) -> None:
    """The losing side of a snapshot conflict was preserved under `key`
    (storage/snapshot_sync.py SnapshotSync._on_conflict) before it reloaded the
    newer snapshot. Health names the last one so an operator can find the
    edits the reload took out of the live view."""
    _record("conflict_copy", name)
    with _status_lock:
        _status[f"{name}:conflict_copy"]["key"] = key


def reset_status() -> None:
    """Test support: the counters are process-wide by design."""
    with _status_lock:
        _status.clear()


def blob_status() -> dict:
    """Flat, stable keys for /health/storage. Counts are since process start;
    `*_at` is the last occurrence (UTC ISO-8601) or None. Per instance, like
    /health/llm: a multi-instance view needs a metrics backend."""
    with _status_lock:
        def get(key: str) -> dict:
            return dict(_status.get(key, {"count": 0, "at": None}))
        conflict = get("curriculum-snapshots:conflict")
        snap_fail = get("curriculum-snapshots:upload_failure")
        scan_fail = get("scan-media:upload_failure")
        copy = get("curriculum-snapshots:conflict_copy")
    return {
        "curriculum_snapshot_conflict": conflict["count"] > 0,
        "curriculum_snapshot_conflict_count": conflict["count"],
        "curriculum_snapshot_conflict_at": conflict["at"],
        "curriculum_snapshot_upload_failures": snap_fail["count"],
        "curriculum_snapshot_upload_failure_at": snap_fail["at"],
        # Each is a whole losing SQLite file under
        # curriculum-snapshots/conflicts/; non-zero means edits await merging.
        "curriculum_snapshot_conflict_copies": copy["count"],
        "curriculum_snapshot_last_conflict_copy": copy.get("key"),
        "scan_media_upload_failures": scan_fail["count"],
        "scan_media_upload_failure_at": scan_fail["at"],
    }


def _gcs_bucket() -> Optional[str]:
    return credential_or_none("ACOS_GCS_BUCKET")


def _is_precondition_failure(exc: BaseException) -> bool:
    # google.api_core.exceptions.PreconditionFailed carries code 412; matched
    # by code rather than by import so this module works without the extra.
    return getattr(exc, "code", None) == 412


class GcsBlobStore:
    """SupabaseStorage's interface over storage/gcs.py's GcsStore."""

    backend = "gcs"
    enabled = True

    def __init__(self, name: str, *, bucket: str, prefix: str,
                 client=None, guard_generations: bool = False):
        from .gcs import GcsStore

        self.name = name
        self._store = GcsStore(bucket, prefix=f"{prefix.strip('/')}/{name}", client=client)
        self._guard = guard_generations
        # key -> generation this process last read (0 = saw it absent) or wrote.
        self._generations: dict[str, int] = {}
        # Guarded stores serialize read-modify-write of _generations; scan
        # media does not need it, and must not queue 40 concurrent requests'
        # photo uploads behind one lock.
        self._lock = threading.Lock() if guard_generations else contextlib.nullcontext()

    def upload(self, key: str, data: bytes, content_type: str) -> None:
        with self._lock:
            expected = self._generations.get(key, 0) if self._guard else None
            try:
                generation = self._store.put_bytes(
                    key, data, content_type, if_generation_match=expected)
            except Exception as exc:
                if self._guard and _is_precondition_failure(exc):
                    adopted = self._own_write_generation(key, data)
                    if adopted is not None:
                        # A retried request met the write its own first
                        # attempt made. Not a conflict: adopt it.
                        logger.warning(
                            "412 on %s/%s was this instance's own retried "
                            "upload (md5 matches); adopting generation %s",
                            self.name, key, adopted)
                        self._generations[key] = adopted
                        return
                    record_conflict(self.name)
                    logger.error(
                        "REFUSED to overwrite gs://%s/%s: it changed since this "
                        "instance loaded generation %s, so another instance wrote "
                        "it. Keeping the newer object. More than one instance is "
                        "running against the %s store (a rollout overlap, or "
                        "--max-instances above 1).",
                        self._store.bucket.name, self._store._key(key), expected,
                        self.name)
                    raise BlobConflict(
                        f"refused to overwrite {self.name}/{key}: changed since "
                        f"generation {expected}") from exc
                raise BlobUnavailable(f"GCS upload of {self.name}/{key} failed: {exc}") from exc
            self._generations[key] = generation

    def _own_write_generation(self, key: str, data: bytes) -> Optional[int]:
        """The object's generation if its content is exactly `data`, else None.

        md5 is what GCS records for every non-composite upload (base64 of the
        raw digest). If it is missing, compare the bytes themselves. A failed
        read here means we cannot prove the object is ours, so it stays a
        conflict: wrongly adopting would let this instance overwrite another's
        snapshot on its next write."""
        try:
            meta = self._store.stat(key)
            if meta is None:
                return None
            generation, md5_b64 = meta
            if md5_b64:
                mine = base64.b64encode(hashlib.md5(data).digest()).decode()
                return generation if md5_b64 == mine else None
            found = self._store.get_bytes(key)
            return found[1] if found is not None and found[0] == data else None
        except Exception:  # noqa: BLE001 - unprovable is treated as a conflict
            logger.warning("could not re-read %s/%s after a 412", self.name, key,
                           exc_info=True)
            return None

    def download(self, key: str) -> bytes:
        with self._lock:
            try:
                found = self._store.get_bytes(key)
            except Exception as exc:
                raise BlobUnavailable(f"GCS download of {self.name}/{key} failed: {exc}") from exc
            if found is None:
                # Seen absent: creating it later is safe, replacing is not.
                self._generations[key] = 0
                raise BlobNotFound(f"{self.name}/{key} does not exist")
            data, generation = found
            self._generations[key] = generation
            return data


def durable_blob_store(name: str, *, gcs_client=None):
    """The blob backend for one purpose ("scan-media", "curriculum-snapshots").

    See the module docstring for the precedence. `gcs_client` exists for
    tests; production builds the default client from the Cloud Run service
    account."""
    bucket = _gcs_bucket()
    if bucket:
        return GcsBlobStore(
            name, bucket=bucket,
            prefix=os.environ.get("ACOS_GCS_PREFIX", "").strip() or "academicos",
            client=gcs_client, guard_generations=name in _GUARDED)
    from ..assessment.supabase_kv import SupabaseStorage
    return SupabaseStorage(name)


def blob_backend() -> str:
    """"gcs" | "supabase" | "local": what durable_blob_store resolves to now,
    reported by /health/storage next to durability_backend."""
    if _gcs_bucket():
        return "gcs"
    from ..assessment.supabase_kv import _supabase_credentials
    url, key = _supabase_credentials()
    return "supabase" if (url and key) else "local"
