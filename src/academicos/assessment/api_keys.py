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
import logging
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# The complete permission vocabulary. Deliberately tiny, and deliberately
# contains nothing that can reach a student.
SCOPES: frozenset[str] = frozenset({
    "questions:read",
    "curriculum:read",
    "facets:read",
    "papers:create",
})

# Refused by design, and named here so the refusal is visible rather than
# implied by absence. `question-bank-api.md:119`: "Student-linked reads
# (/knowledge/*, progress) -- Never part of this API."
FORBIDDEN_SCOPES: frozenset[str] = frozenset({
    "knowledge:read", "progress:read", "students:read", "students:write",
    "grading:read", "grading:write", "consent:read", "consent:write",
})

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
  revoked_at    TEXT
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

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def may(self, scope: str) -> bool:
        return self.active and scope in self.scopes

    def to_dict(self) -> dict[str, Any]:
        """The public shape. `key_hash` is never included."""
        return {
            "id": self.id,
            "prefix": self.prefix,
            "label": self.label,
            "schoolId": self.school_id,
            "scopes": sorted(self.scopes),
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
    unknown = wanted - SCOPES
    forbidden = wanted & FORBIDDEN_SCOPES
    if forbidden:
        raise ScopeError(
            "these scopes are permanently refused for this API because they "
            f"would reach student data: {sorted(forbidden)}")
    if unknown:
        raise ScopeError(
            f"unknown scope(s) {sorted(unknown)}; the vocabulary is "
            f"{sorted(SCOPES)}")
    if not wanted:
        raise ScopeError("a key must have at least one scope")
    return wanted


class ApiKeyStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
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
    ) -> tuple[str, ApiKey]:
        """Mint a key. Returns `(plaintext, record)`; the plaintext is shown once."""
        granted = validate_scopes(scopes)
        if quota_per_minute < 1:
            raise ValueError("quota_per_minute must be >= 1")

        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        key_id = f"key_{uuid.uuid4().hex[:16]}"
        prefix = plaintext[: len(KEY_PREFIX) + 6]
        self.conn.execute(
            "INSERT INTO api_keys (id, key_hash, prefix, label, school_id, "
            "scopes, quota_per_min, created_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (key_id, hash_key(plaintext), prefix, label, school_id,
             ",".join(sorted(granted)), quota_per_minute, _now().isoformat(),
             created_by),
        )
        self.conn.commit()
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
        self.conn.commit()
        return self.get(key_id)

    # -- quota ------------------------------------------------------------- #

    def check_quota(self, key: ApiKey) -> None:
        """Count one request against the key's per-minute allowance.

        Counted in the database, in fixed one-minute windows, for two reasons:
        a per-process counter lets N instances each allow the full quota, and an
        in-memory counter resets to zero on redeploy -- which is exactly when an
        abuser would want it to.
        """
        window = _now().replace(second=0, microsecond=0)
        window_key = window.isoformat()
        row = self.conn.execute(
            "SELECT count FROM api_key_usage WHERE key_id=? AND window_start=?",
            (key.id, window_key)).fetchone()

        if row is not None and row["count"] >= key.quota_per_minute:
            self.conn.execute(
                "DELETE FROM api_key_usage WHERE window_start < ?",
                ((window - timedelta(minutes=5)).isoformat(),))
            self.conn.commit()
            raise QuotaExceeded(key, retry_after=60 - window.second)

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
    )
