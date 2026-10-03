"""Owners over several schools (ROLE-4): one owner account linked to several
schools, each school's data kept separate.

A school's principal issues an owner link: a single-use code, shown once and
kept here only as its SHA-256. The owner redeems it, and the link row then IS
the membership: it names the owner, and it holds until the principal revokes
it. A revoke is one UPDATE, and auth_routes.get_current_user asks this store
on every request whether the session's chosen school is still linked, so the
owner loses that school on its next call.

An owner session chooses one school at a time (`owner_sessions`, keyed by
the SHA-256 of the session token, never the token). While it has, every route
sees the owner as that school's principal and scopes by that school alone.

Kept in the operations store (one SQLite file, WAL, 60 s busy timeout,
snapshotted like the admin grants): the links are relational to nothing
outside this file, and Cloud SQL would need a schema change applied by hand
before the first deploy that used them.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

OWNERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS owner_links (
    id TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    code_hash TEXT NOT NULL UNIQUE,
    code_hint TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    owner_id TEXT,
    redeemed_at TEXT,
    revoked_at TEXT,
    revoked_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_owner_links_school ON owner_links(school_id);
CREATE INDEX IF NOT EXISTS idx_owner_links_owner ON owner_links(owner_id);
CREATE TABLE IF NOT EXISTS owner_sessions (
    token_hash TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    school_id TEXT NOT NULL,
    chosen_at TEXT NOT NULL
);
"""

# The code says what it is, so the web's one "invite code" box can send an
# owner's code to the owner route. 18 random bytes: 144 bits.
CODE_PREFIX = "own_"
DEFAULT_DAYS = 14
MAX_DAYS = 90
# A session lives 30 days (users.py); its chosen school is forgotten after.
_SESSION_DAYS = 31


class OwnerLinkRefused(Exception):
    """The code cannot be redeemed; the message says why, in plain words."""


class AlreadyLinked(Exception):
    """The owner already holds an active link to this school."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def link_status(row: dict, now: Optional[datetime] = None) -> str:
    """`linked` | `revoked` | `expired` | `open`. A redeemed link stays
    `linked` past its expiry: the expiry bounds the code, not the access."""
    if row.get("revoked_at"):
        return "revoked"
    if row.get("redeemed_at"):
        return "linked"
    if datetime.fromisoformat(row["expires_at"]) <= (now or _now()):
        return "expired"
    return "open"


class OwnersMixin:
    """Added to OperationsStore."""

    # ---------------- the principal's side ----------------

    def create_owner_link(self, *, school_id: str, created_by: str, label: str = "",
                          expires_in_days: Optional[int] = None) -> tuple[str, dict]:
        """(code, row). The code is returned once and only its hash kept."""
        days = DEFAULT_DAYS if expires_in_days is None else expires_in_days
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_DAYS:
            raise ValueError(f"expiresInDays must be a whole number from 1 to {MAX_DAYS}")
        code = CODE_PREFIX + secrets.token_urlsafe(18)
        now = _now()
        link_id = f"olk_{uuid.uuid4().hex[:12]}"
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO owner_links (id, school_id, code_hash, code_hint, label, created_by, created_at,"
                " expires_at) VALUES (?,?,?,?,?,?,?,?)",
                (link_id, school_id, hash_code(code), code[:len(CODE_PREFIX) + 4], label.strip(), created_by,
                 _iso(now), _iso(now + timedelta(days=days))))
            self._commit()
        return code, self.get_owner_link(link_id)

    def get_owner_link(self, link_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM owner_links WHERE id=?", (link_id,))

    def owner_links_for_school(self, school_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM owner_links WHERE school_id=? ORDER BY created_at DESC, id",
                              (school_id,))

    def revoke_owner_link(self, link_id: str, *, revoked_by: str) -> Optional[dict]:
        """Withdraw an unused code, or end a redeemed link's access. The
        owner's sessions stop acting in the school at once."""
        with self._conn_lock:
            self.conn.execute("UPDATE owner_links SET revoked_at=?, revoked_by=? WHERE id=? AND revoked_at IS NULL",
                              (_iso(_now()), revoked_by, link_id))
            row = self.conn.execute("SELECT * FROM owner_links WHERE id=?", (link_id,)).fetchone()
            if row is not None and row["owner_id"]:
                self.conn.execute("DELETE FROM owner_sessions WHERE owner_id=? AND school_id=?",
                                  (row["owner_id"], row["school_id"]))
            self._commit()
        return dict(row) if row is not None else None

    # ---------------- the owner's side ----------------

    def find_owner_link(self, code: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM owner_links WHERE code_hash=?", (hash_code(code),))

    def claim_owner_link(self, code: str, owner_id: str) -> dict:
        """Redeem `code` for `owner_id`, or raise OwnerLinkRefused /
        AlreadyLinked. Single use is enforced by the UPDATE's condition, not
        by the status check before it, so two redemptions racing on one code
        have one winner."""
        row = self.find_owner_link(code)
        if row is None:
            raise OwnerLinkRefused("owner link code not recognised; check it with the school's principal")
        status = link_status(row)
        if status != "open":
            raise OwnerLinkRefused({
                "linked": "this owner link code has already been used; ask the principal for a new one",
                "revoked": "this owner link code was withdrawn; ask the principal for a new one",
                "expired": "this owner link code has expired; ask the principal for a new one",
            }[status])
        if row["school_id"] in self.linked_school_ids(owner_id):
            raise AlreadyLinked(row["school_id"])
        with self._conn_lock:
            cur = self.conn.execute(
                "UPDATE owner_links SET owner_id=?, redeemed_at=? "
                "WHERE id=? AND owner_id IS NULL AND redeemed_at IS NULL AND revoked_at IS NULL AND expires_at > ?",
                (owner_id, _iso(_now()), row["id"], _iso(_now())))
            self._commit()
        if cur.rowcount != 1:
            raise OwnerLinkRefused("this owner link code has just been used or withdrawn; "
                                   "ask the principal for a new one")
        return self.get_owner_link(row["id"])

    def release_owner_link(self, link_id: str, owner_id: str) -> None:
        """Undo a claim whose account could not be created, so a storage
        error does not burn the principal's code."""
        with self._conn_lock:
            self.conn.execute("UPDATE owner_links SET owner_id=NULL, redeemed_at=NULL WHERE id=? AND owner_id=?",
                              (link_id, owner_id))
            self._commit()

    def owner_memberships(self, owner_id: str) -> list[dict]:
        """The owner's active links, one per school, oldest first."""
        return self._fetchall("SELECT * FROM owner_links WHERE owner_id=? AND redeemed_at IS NOT NULL"
                              " AND revoked_at IS NULL ORDER BY redeemed_at, id", (owner_id,))

    def linked_school_ids(self, owner_id: str) -> list[str]:
        return [r["school_id"] for r in self.owner_memberships(owner_id)]

    def is_linked(self, owner_id: str, school_id: str) -> bool:
        return self._fetchone("SELECT 1 AS x FROM owner_links WHERE owner_id=? AND school_id=?"
                              " AND redeemed_at IS NOT NULL AND revoked_at IS NULL", (owner_id, school_id)) is not None

    # ---------------- which school a session acts in ----------------

    def set_owner_session_school(self, token: str, owner_id: str, school_id: Optional[str]) -> None:
        """Choose `school_id` for this session (None: no school). The caller
        has checked the link; owner_session_school re-checks it each time."""
        th = hash_token(token)
        with self._conn_lock:
            self.conn.execute("DELETE FROM owner_sessions WHERE token_hash=? OR chosen_at < ?",
                              (th, _iso(_now() - timedelta(days=_SESSION_DAYS))))
            if school_id:
                self.conn.execute("INSERT INTO owner_sessions (token_hash, owner_id, school_id, chosen_at)"
                                  " VALUES (?,?,?,?)", (th, owner_id, school_id, _iso(_now())))
            self._commit()

    def owner_session_school(self, token: str, owner_id: str) -> Optional[str]:
        """The school this session chose, if it is still linked to the owner.
        One keyed read and one membership read, on each owner request."""
        row = self._fetchone("SELECT * FROM owner_sessions WHERE token_hash=?", (hash_token(token),))
        if row is None or row["owner_id"] != owner_id:
            return None
        return row["school_id"] if self.is_linked(owner_id, row["school_id"]) else None
