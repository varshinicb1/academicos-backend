"""Third-party API keys and per-key quotas for the question-bank API.

`docs/question-bank-api.md` section 4 sets the requirement and the constraint
that shapes it:

> Partner schools, internal apps: API keys (hashed at rest, scoped per school,
> revocable). ... **An API key must never grant access to anything a school's
> own users cannot see.** Scope keys to the curriculum and questions only.

Three consequences, all of which are implemented here:

  * **Hashed at rest.** The plaintext is returned exactly once, at creation.
    A database dump must not be a set of usable credentials.
  * **Scoped, and the scope list is a closed set.** A key cannot be minted with
    a permission this module does not know about, so a typo cannot become a
    grant.
  * **Revoked, never deleted.** A revoked key stays resolvable so an audit can
    answer "was this key ever issued, and when was it stopped". Deleting it
    would destroy exactly the evidence an incident review needs.

**No student data is reachable through this path, and that is enforced here
rather than documented.** `SCOPES` contains no student scope, and
`authenticate()` returns a principal that the question routes can only use for
content reads. `docs/question-bank-api.md:119` is explicit that student-linked
reads are never part of this API; the way to keep that true is to make the
permission unexpressible.

Quotas are counted in SQLite rather than in memory. The existing
`api/rate_limit.py` uses a per-process deque, which is right for login
brute-force on one instance and wrong for a quota: two Cloud Run instances would
each allow the full limit, and a redeploy would reset the counter to zero
mid-abuse.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# The complete permission vocabulary. Deliberately tiny, and deliberately
# contains nothing that can reach a student. Every scope here gates a route:
# `questions:read` the question and homework-set routes, `facets:read`
# `/v1/facets` and `/v1/coverage`.
SCOPES: frozenset[str] = frozenset({
    "questions:read",
    "facets:read",
})

# Refused by design, and named here so the refusal is visible rather than
# implied by absence. `question-bank-api.md:119`: "Student-linked reads
# (/knowledge/*, progress) -- Never part of this API."
FORBIDDEN_SCOPES: frozenset[str] = frozenset({
    "knowledge:read", "progress:read", "students:read", "students:write",
    "grading:read", "grading:write", "consent:read", "consent:write",
})

# Planned, but no route checks them yet, so a key minted with one was granted
# nothing while the web page said otherwise (audit N-67-10: the API keys page
# offered `papers:create` and ticked `curriculum:read` by default). Refused at
# mint the way the student scopes are, with a reason that says it is "not
# yet" rather than "never". A key minted before this change may still carry
# one; that is harmless, because nothing reads it.
NOT_YET_SCOPES: frozenset[str] = frozenset({"papers:create", "curriculum:read"})

DEFAULT_QUOTA_PER_MINUTE = 120
KEY_PREFIX = "acos_qb_"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
  id            TEXT PRIMARY KEY,
  key_hash      TEXT NOT NULL UNIQUE,
  prefix        TEXT NOT NULL,
  label         TEXT NOT NULL DEFAULT '',
  school_id     TEXT NOT NULL,
  scopes        TEXT NOT NULL,
  quota_per_min INTEGER NOT NULL,
  created_at    TEXT NOT NULL,
  created_by    TEXT NOT NULL DEFAULT '',
  last_used_at  TEXT,
  revoked_at    TEXT,
  grades        TEXT NOT NULL DEFAULT '',
  subjects      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ak_hash   ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_ak_school ON api_keys(school_id);

CREATE TABLE IF NOT EXISTS api_key_usage (
  key_id     TEXT NOT NULL,
  window_start TEXT NOT NULL,
  count      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, window_start)
);
CREATE INDEX IF NOT EXISTS idx_aku_window ON api_key_usage(window_start);
"""

# Columns added after the table first shipped, with the DEFAULT an old row
# reads as. `CREATE TABLE IF NOT EXISTS` never alters a table the snapshot
# restored, so a live store from before the column would otherwise fail its
# first INSERT. '' is "no limit", which is what every earlier key was.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("grades", "TEXT NOT NULL DEFAULT ''"),
    ("subjects", "TEXT NOT NULL DEFAULT ''"),
)


class ScopeError(ValueError):
    """A scope outside the vocabulary, or one that is forbidden outright."""


class QuotaExceeded(RuntimeError):
    """The key has spent its allowance for the current window."""

    def __init__(self, key: "ApiKey", retry_after: int) -> None:
        super().__init__(
            f"quota exceeded for {key.label or key.prefix}: "
            f"{key.quota_per_minute}/min")
        self.retry_after = max(1, retry_after)


@dataclass(frozen=True)
class ApiKey:
    id: str
    prefix: str          # the public, safe-to-log part
    label: str
    school_id: str
    scopes: frozenset[str]
    quota_per_minute: int
    created_at: str
    created_by: str
    last_used_at: str | None
    revoked_at: str | None
    # The classes and subjects the key may read (API-3: "scoped ... which
    # grades/subjects"). Empty means every one, which is what a key minted
    # before the limits existed was. Subjects keep the spelling they were
    # minted with; `subject_keys` is the lower-cased set they compare by.
    grades: frozenset[int] = frozenset()
    subjects: frozenset[str] = frozenset()

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def may(self, scope: str) -> bool:
        return self.active and scope in self.scopes

    @property
    def limited(self) -> bool:
        return bool(self.grades or self.subjects)

    @property
    def subject_keys(self) -> frozenset[str]:
        return frozenset(s.lower() for s in self.subjects)

    def admits_grade(self, grade: Any) -> bool:
        """A grade inside the key's limit. A limit that is set refuses an
        absent grade: a record nobody could place is not "in class 10"."""
        return not self.grades or grade in self.grades

    def admits_subject(self, subject: Any) -> bool:
        return not self.subjects or str(subject or "").lower() in self.subject_keys

    def to_dict(self) -> dict[str, Any]:
        """The public shape. `key_hash` is never included."""
        return {
            "id": self.id,
            "prefix": self.prefix,
            "label": self.label,
            "schoolId": self.school_id,
            "scopes": sorted(self.scopes),
            "grades": sorted(self.grades),
            "subjects": sorted(self.subjects),
            "quotaPerMinute": self.quota_per_minute,
            "createdAt": self.created_at,
            "createdBy": self.created_by,
            "lastUsedAt": self.last_used_at,
            "revokedAt": self.revoked_at,
            "active": self.active,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_key(plaintext: str) -> str:
    """SHA-256 of the key.

    Not a password hash, and deliberately so: these keys are machine-generated
    with 256 bits of entropy, so there is nothing to brute-force and a slow KDF
    would only add latency to every request. Passwords are the case that needs
    bcrypt; random tokens are not.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def validate_scopes(scopes: Iterable[str]) -> frozenset[str]:
    """Refuse anything outside the vocabulary, and anything forbidden outright."""
    wanted = frozenset(scopes or ())
    unknown = wanted - SCOPES - NOT_YET_SCOPES
    forbidden = wanted & FORBIDDEN_SCOPES
    not_yet = wanted & NOT_YET_SCOPES
    if forbidden:
        raise ScopeError(
            "these scopes are permanently refused for this API because they "
            f"would reach student data: {sorted(forbidden)}")
    if not_yet:
        raise ScopeError(
            f"these scopes are not available yet: {sorted(not_yet)}. No route "
            "checks them, so a key holding one would be granted nothing; the "
            f"scopes a key can hold today are {sorted(SCOPES)}")
    if unknown:
        raise ScopeError(
            f"unknown scope(s) {sorted(unknown)}; the vocabulary is "
            f"{sorted(SCOPES)}")
    if not wanted:
        raise ScopeError("a key must have at least one scope")
    return wanted


class ApiKeyStore:
    def __init__(self, db_path: Path | str, *, durable: bool = False):
        """`durable=True` (the API server) snapshots the file to the blob
        store, as the operations store does, so an issued key survives a
        restart (audit D38: the file lived on container disk only). Only
        issuing and revoking publish a snapshot; per-request counters are
        committed locally and ride along with the next one -- they are
        minute windows, worthless after five minutes."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._snapshots = None
        if durable:
            from ..storage.snapshot_sync import SnapshotSync
            self._snapshots = SnapshotSync("operations-snapshots", "api_keys.sqlite", self.db_path, self._lock,
                                           debounce_seconds=2.0, allow_empty_boot=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(api_keys)")}
        for name, decl in _ADDED_COLUMNS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE api_keys ADD COLUMN {name} {decl}")
        self.conn.commit()
        if self._snapshots is not None:
            self._snapshots.commit_derived(self.conn)
            self._snapshots.save_empty_boot(self.conn)

    def _publish(self) -> None:
        if self._snapshots is not None:
            self._snapshots.commit(self.conn)
        else:
            self.conn.commit()

    # -- issuing ----------------------------------------------------------- #

    def create(
        self,
        *,
        school_id: str,
        scopes: Iterable[str],
        label: str = "",
        created_by: str = "",
        quota_per_minute: int = DEFAULT_QUOTA_PER_MINUTE,
        grades: Iterable[int] = (),
        subjects: Iterable[str] = (),
    ) -> tuple[str, ApiKey]:
        """Mint a key. Returns `(plaintext, record)`; the plaintext is shown once.

        `grades` and `subjects` limit what the key can read; empty is every
        class and subject. Stored as JSON rather than comma-joined like
        `scopes`, because a subject's name is free text and may hold a comma.
        """
        granted = validate_scopes(scopes)
        if quota_per_minute < 1:
            raise ValueError("quota_per_minute must be >= 1")
        only_grades = sorted({int(g) for g in grades})
        if any(not 1 <= g <= 12 for g in only_grades):
            raise ValueError(f"grades must be between 1 and 12; got {only_grades}")
        only_subjects = sorted({s.strip() for s in subjects if s and s.strip()})

        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        key_id = f"key_{uuid.uuid4().hex[:16]}"
        prefix = plaintext[: len(KEY_PREFIX) + 6]
        self.conn.execute(
            "INSERT INTO api_keys (id, key_hash, prefix, label, school_id, "
            "scopes, quota_per_min, created_at, created_by, grades, subjects) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (key_id, hash_key(plaintext), prefix, label, school_id,
             ",".join(sorted(granted)), quota_per_minute, _now().isoformat(),
             created_by, json.dumps(only_grades) if only_grades else "",
             json.dumps(only_subjects) if only_subjects else ""),
        )
        self._publish()
        return plaintext, self.get(key_id)  # type: ignore[return-value]

    # -- reading ----------------------------------------------------------- #

    def get(self, key_id: str) -> ApiKey | None:
        row = self.conn.execute(
            "SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return _to_key(row) if row else None

    def list_for_school(self, school_id: str) -> list[ApiKey]:
        rows = self.conn.execute(
            "SELECT * FROM api_keys WHERE school_id=? ORDER BY created_at DESC",
            (school_id,)).fetchall()
        return [_to_key(r) for r in rows]

    def authenticate(self, plaintext: str) -> ApiKey | None:
        """Resolve a presented key. Returns None for unknown, revoked or blank.

        One query, no timing branch: the presented key is hashed and compared in
        SQL, so a wrong key costs the same as a right one.
        """
        if not plaintext or not plaintext.startswith(KEY_PREFIX):
            return None
        row = self.conn.execute(
            "SELECT * FROM api_keys WHERE key_hash=?", (hash_key(plaintext),)
        ).fetchone()
        if row is None:
            return None
        key = _to_key(row)
        if not key.active:
            log.warning("revoked api key presented: %s", key.prefix)
            return None
        self.conn.execute(
            "UPDATE api_keys SET last_used_at=? WHERE id=?",
            (_now().isoformat(), key.id),
        )
        self.conn.commit()
        # Re-read rather than returning the pre-update row: the caller uses this
        # record (to log a prefix, or to report last use), and handing back a
        # stale `last_used_at` makes the field look broken to every consumer.
        return self.get(key.id)

    # -- revoking ---------------------------------------------------------- #

    def revoke(self, key_id: str) -> ApiKey | None:
        """Revoke without deleting: an incident review needs the record."""
        key = self.get(key_id)
        if key is None or not key.active:
            return key
        self.conn.execute(
            "UPDATE api_keys SET revoked_at=? WHERE id=?",
            (_now().isoformat(), key_id),
        )
        self._publish()
        return self.get(key_id)

    # -- quota ------------------------------------------------------------- #

    def check_quota(self, key: ApiKey) -> None:
        """Count one request against the key's per-minute allowance.

        Counted in the database, in fixed one-minute windows, for two reasons:
        a per-process counter lets N instances each allow the full quota, and an
        in-memory counter resets to zero on redeploy -- which is exactly when an
        abuser would want it to.
        """
        now = _now()
        window = now.replace(second=0, microsecond=0)
        window_key = window.isoformat()
        row = self.conn.execute(
            "SELECT count FROM api_key_usage WHERE key_id=? AND window_start=?",
            (key.id, window_key)).fetchone()

        if row is not None and row["count"] >= key.quota_per_minute:
            self.conn.execute(
                "DELETE FROM api_key_usage WHERE window_start < ?",
                ((window - timedelta(minutes=5)).isoformat(),))
            self.conn.commit()
            # The seconds left in THIS window, rounded up so a client that
            # waits exactly that long lands in the next one. This was
            # `60 - window.second`, and `window` has its seconds zeroed two
            # lines up, so every 429 said 60 (audit N-67-10): a client honouring
            # it waited a full minute when the window reopened in two seconds.
            left = (window + timedelta(minutes=1) - now).total_seconds()
            raise QuotaExceeded(key, retry_after=math.ceil(left))

        self.conn.execute(
            "INSERT INTO api_key_usage (key_id, window_start, count) VALUES (?,?,1) "
            "ON CONFLICT(key_id, window_start) DO UPDATE SET count = count + 1",
            (key.id, window_key),
        )
        self.conn.commit()

    def usage(self, key_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM api_key_usage WHERE key_id=? ORDER BY window_start DESC LIMIT 60",
            (key_id,)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._snapshots is not None:
            self._snapshots.close()
        self.conn.close()


def _to_key(row: sqlite3.Row) -> ApiKey:
    raw = str(row["scopes"] or "")
    return ApiKey(
        id=row["id"],
        prefix=row["prefix"],
        label=row["label"] or "",
        school_id=row["school_id"],
        scopes=frozenset(s for s in raw.split(",") if s),
        quota_per_minute=int(row["quota_per_min"]),
        created_at=row["created_at"],
        created_by=row["created_by"] or "",
        last_used_at=row["last_used_at"],
        revoked_at=row["revoked_at"],
        grades=frozenset(int(g) for g in json.loads(row["grades"] or "[]")),
        subjects=frozenset(str(s) for s in json.loads(row["subjects"] or "[]")),
    )
