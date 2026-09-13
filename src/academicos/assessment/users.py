"""Teacher/principal identity — the real gap behind two things found this
session: `reviewerId` left permanently blank in every review/finalize call
(there was no authenticated actor to put there), and the principal-approval
*lock* (`Assessment.status == principalApproved`) existing with no real
*workflow* to reach it, since nothing could authenticate as a principal.

No new dependency added for this: password hashing uses stdlib
`hashlib.pbkdf2_hmac` (OWASP's 2023-recommended minimum of 200,000 rounds for
PBKDF2-SHA256, the same primitive Django's default hasher uses), and sessions
are opaque server-side tokens (`secrets.token_urlsafe`) in their own table,
not a JWT — no signing-key management, and revocation is just deleting a row.
This project has otherwise deliberately kept its dependency footprint small
(see pyproject.toml) and neither bcrypt nor a JWT library was already there.

Bootstrap rule for who becomes principal (fixed 2026-09-11, was a real
privilege-escalation bug): registration used to grant `role="principal"` to
whichever request happened to be the *first* to register for a given
`schoolId` -- an unauthenticated race, not an authorization check. Anyone
who beat the real school administrator to `/auth/register` with that
school's id became its principal. Now: a caller only becomes principal by
supplying a `principalKey` that matches `Config.principal_bootstrap_key`
(server-side secret, `ACOS_PRINCIPAL_BOOTSTRAP_KEY` in the gitignored
`config/secrets.env`, compared with `secrets.compare_digest`). No key
configured -> nobody can self-register as principal, ever; that's the
secure default. This still doesn't answer "how does a real school's admin
receive that key" (out-of-band: the operator who provisions a school's
account hands it to them) -- an intentionally minimal fix scoped to closing
the escalation, not a full admin-console/invite-email system.

Same SQLite-local / Supabase-when-configured persistence pattern as every
other store here (see supabase_kv.py's module docstring: Render's free tier
wipes local disk on restart, so anything meant to survive a redeploy needs
Postgres, not just a local file) — critical for this specific store, since a
users table that doesn't survive a restart means every teacher account
(and every session token) silently vanishes on the next Render cold start.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .supabase_kv import SupabaseTable

_PBKDF2_ROUNDS = 200_000
_SESSION_LIFETIME = timedelta(days=30)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id            TEXT PRIMARY KEY,
  school_id     TEXT NOT NULL,
  name          TEXT NOT NULL,
  email         TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  password_salt TEXT NOT NULL,
  role          TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_school ON users(school_id);

CREATE TABLE IF NOT EXISTS sessions (
  token      TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


@dataclass
class User:
    id: str
    school_id: str
    name: str
    email: str
    role: str  # "teacher" | "principal" | "student"
    created_at: str


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS).hex()


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class UserStore:
    def __init__(self, db_path: Path, principal_bootstrap_key: str = ""):
        self._remote = SupabaseTable("users")
        self._remote_sessions = SupabaseTable("sessions")
        self._principal_key = principal_bootstrap_key
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- registration / login ----------------

    def register(self, *, school_id: str, name: str, email: str, password: str,
                principal_key: Optional[str] = None, requested_role: str = "teacher") -> User:
        email = email.strip().lower()
        if self.get_by_email(email) is not None:
            raise EmailAlreadyRegistered(email)
        if requested_role not in ("teacher", "student"):
            raise ValueError(f"invalid requested_role: {requested_role!r} (must be 'teacher' or 'student')")

        # "principal" is never something a caller can request directly --
        # it's granted only by a valid principal_key, same as before this
        # role field existed. requested_role only chooses between the two
        # roles a caller is actually allowed to self-select.
        role = "principal" if self._valid_principal_key(principal_key) else requested_role
        salt = secrets.token_bytes(16)
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()
        row = {
            "id": user_id, "school_id": school_id, "name": name, "email": email,
            "password_hash": _hash_password(password, salt), "password_salt": salt.hex(),
            "role": role, "created_at": created_at,
        }
        if self._remote.enabled:
            self._remote.upsert(row, on_conflict="id")
        else:
            self.conn.execute(
                """INSERT INTO users (id, school_id, name, email, password_hash, password_salt,
                                       role, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (user_id, school_id, name, email, row["password_hash"], row["password_salt"],
                 role, created_at),
            )
            self.conn.commit()
        return User(id=user_id, school_id=school_id, name=name, email=email, role=role,
                    created_at=created_at)

    def authenticate(self, *, email: str, password: str) -> User:
        email = email.strip().lower()
        raw = self._raw_by_email(email)
        if raw is None:
            raise InvalidCredentials(email)
        salt = bytes.fromhex(raw["password_salt"])
        if not secrets.compare_digest(_hash_password(password, salt), raw["password_hash"]):
            raise InvalidCredentials(email)
        return _row_to_user(raw)

    # ---------------- sessions ----------------

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = (now + _SESSION_LIFETIME).isoformat()
        row = {"token": token, "user_id": user_id, "created_at": now.isoformat(),
               "expires_at": expires_at}
        if self._remote_sessions.enabled:
            self._remote_sessions.upsert(row, on_conflict="token")
        else:
            self.conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                (token, user_id, row["created_at"], expires_at),
            )
            self.conn.commit()
        return token

    def user_for_session(self, token: str) -> Optional[User]:
        if self._remote_sessions.enabled:
            rows = self._remote_sessions.select(token=token)
            session_row = rows[0] if rows else None
        else:
            r = self.conn.execute(
                "SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
            session_row = dict(r) if r else None
        if session_row is None:
            return None
        if datetime.fromisoformat(session_row["expires_at"]) < datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return self.get(session_row["user_id"])

    def delete_session(self, token: str) -> None:
        if self._remote_sessions.enabled:
            self._remote_sessions.delete(token=token)
        else:
            self.conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            self.conn.commit()

    # ---------------- lookups ----------------

    def get(self, user_id: str) -> Optional[User]:
        if self._remote.enabled:
            rows = self._remote.select(id=user_id)
            return _row_to_user(rows[0]) if rows else None
        r = self.conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return _row_to_user(dict(r)) if r else None

    def get_by_email(self, email: str) -> Optional[User]:
        raw = self._raw_by_email(email.strip().lower())
        return _row_to_user(raw) if raw else None

    def users_for_school(self, school_id: str, *, role: Optional[str] = None) -> list[User]:
        """The real roster a principal picks from when assigning a teacher
        to a book or enrolling a student in a class -- without this, those
        actions require already knowing a raw user id, which no real UI
        can ask a principal to type in by hand."""
        if self._remote.enabled:
            rows = self._remote.select(school_id=school_id, **({"role": role} if role else {}))
            return [_row_to_user(r) for r in rows]
        if role:
            rows = self.conn.execute(
                "SELECT * FROM users WHERE school_id=? AND role=? ORDER BY name",
                (school_id, role)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM users WHERE school_id=? ORDER BY name", (school_id,)).fetchall()
        return [_row_to_user(dict(r)) for r in rows]

    def _raw_by_email(self, email: str) -> Optional[dict]:
        if self._remote.enabled:
            rows = self._remote.select(email=email)
            return rows[0] if rows else None
        r = self.conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(r) if r else None

    def _valid_principal_key(self, supplied: Optional[str]) -> bool:
        """True only when a bootstrap key is actually configured AND the
        caller supplied the exact match. Constant-time compare so a wrong
        guess can't be timed to learn the real key; empty/`None` on either
        side always fails closed (never grants principal by default)."""
        if not self._principal_key or not supplied:
            return False
        return secrets.compare_digest(supplied, self._principal_key)


def _row_to_user(row: dict) -> User:
    return User(id=row["id"], school_id=row["school_id"], name=row["name"], email=row["email"],
                role=row["role"], created_at=row["created_at"])


_INSTANCE: Optional[UserStore] = None


def get_user_store(data_root: Path, principal_bootstrap_key: str = "") -> UserStore:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UserStore(data_root / "users" / "users.sqlite",
                              principal_bootstrap_key=principal_bootstrap_key)
    return _INSTANCE
