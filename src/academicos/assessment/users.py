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

Who may join which school (fixed 2026-09-21, scorecard item 8.4): membership
is granted by an INVITE, never asserted by the caller. Until then
`/auth/register` took `schoolId` and `role` from the request body and stored
them unchecked. Two independent audit probes registered
`attacker@evil.example` into school_A as a teacher, and that account then got
200 on /assessments/{id}, /knowledge/{studentId}, /insights/class/{aid},
every student's answers under /evaluations/assessment/{aid}/question/q_1, and
/consent/students/{sid}. Every `school_id` check in the product rests on this
one claim, so none of them meant anything. Now:

  * a principal creates an invite for THEIR OWN school (`create_invite`):
    a `secrets.token_urlsafe(16)` code (128 bits), a role (`teacher` or
    `student`), optionally one email, and an expiry (14 days by default).
  * `register(invite_code=...)` takes the new account's school and role from
    the invite. A `school_id`/`role` sent alongside must agree with it
    (`RegistrationConflict`, 400), so an old client cannot believe it joined
    one school while landing in another.
  * an invite is single use. The claim is one conditional UPDATE
    (`... WHERE used_at = '' AND revoked_at = ''`), so a race on one code has
    exactly one winner on SQLite, Supabase and Cloud SQL alike.

The principal bootstrap key (`ACOS_PRINCIPAL_BOOTSTRAP_KEY`, server-side,
gitignored `config/secrets.env`, compared with `secrets.compare_digest`) is an
OPERATOR secret. Its history: until 2026-09-11 the first registrant for a
`schoolId` became its principal, an unauthenticated race. The key replaced
that, but it was global, so its holder could become principal of ANY school,
including one that already had a principal (audit item 8.4 again). Now it can
still do exactly one thing: create the FIRST principal of a school that has
none. That principal then invites everyone else. No key configured means
nobody can register as principal, which is the secure default.

Accounts created before 2026-09-21 are untouched: their stored school and
role stay as they are. The code cannot tell an account that self-asserted
its school from a legitimate one, so an operator has to review each school's
roster (`GET /auth/users`) once.

Demo accounts (`teacher@school.com` / `principal@school.com`, password
`password123`, school `demo_school`) are seeded only when
`ACOS_SEED_DEMO_USERS=1`. Before 2026-09-21 they were seeded whenever the
LOCAL users table was empty outside pytest. On Render the durable store is
Supabase and the local table is empty after every cold start, so a principal
account with a published password was re-created in production on each
restart.

Same SQLite-local / Supabase-when-configured persistence pattern as every
other store here (see supabase_kv.py's module docstring: Render's free tier
wipes local disk on restart, so anything meant to survive a redeploy needs
Postgres, not just a local file) — critical for this specific store, since a
users table that doesn't survive a restart means every teacher account
(and every session token) silently vanishes on the next Render cold start.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from .supabase_kv import SupabaseUnavailable
from .postgres_kv import durable_table
from ..config import demo_accounts_enabled

logger = logging.getLogger(__name__)

_PBKDF2_ROUNDS = 200_000
_SESSION_LIFETIME = timedelta(days=30)

# An invite is a credential: whoever holds the code gets an account at that
# school. 14 days covers handing a code to a teacher across a school week or
# two; 90 days is the cap, so an invite cannot turn into a standing key.
INVITE_ROLES = ("teacher", "student")
_INVITE_DEFAULT_DAYS = 14
_INVITE_MAX_DAYS = 90

_DEMO_SEED_ENV = "ACOS_SEED_DEMO_USERS"

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

-- '' (not NULL) means "not yet" for email/used_by/used_at/revoked_at. The
-- remote tables filter on equality only, and equality never matches NULL in
-- PostgREST or Postgres. With '' the single-use claim is one conditional
-- UPDATE (`used_at = '' AND revoked_at = ''`) on every backend.
CREATE TABLE IF NOT EXISTS invites (
  code       TEXT PRIMARY KEY,
  school_id  TEXT NOT NULL,
  role       TEXT NOT NULL,
  email      TEXT NOT NULL DEFAULT '',
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used_by    TEXT NOT NULL DEFAULT '',
  used_at    TEXT NOT NULL DEFAULT '',
  revoked_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_invites_school ON invites(school_id);
"""


@dataclass
class User:
    id: str
    school_id: str
    name: str
    email: str
    role: str  # "teacher" | "principal" | "student"
    created_at: str


@dataclass
class Invite:
    """A principal's single-use permission for one person to join their
    school with one role. `email == ""` means any email may use it. `used_*`
    and `revoked_at` are `""` until they happen (see SCHEMA)."""
    code: str
    school_id: str
    role: str
    email: str
    created_by: str
    created_at: str
    expires_at: str
    used_by: str = ""
    used_at: str = ""
    revoked_at: str = ""

    def status(self, now: Optional[datetime] = None) -> str:
        """`used` | `revoked` | `expired` | `open`. Used wins over revoked:
        a claim and a revoke cannot both succeed (both are conditional on
        the other not having happened), so at most one is ever set."""
        if self.used_at:
            return "used"
        if self.revoked_at:
            return "revoked"
        if datetime.fromisoformat(self.expires_at) <= (now or datetime.now(timezone.utc)):
            return "expired"
        return "open"


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS).hex()


def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:12]}"


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class RegistrationRefused(Exception):
    """The caller has no right to an account: no invite and no key, an invite
    that is unknown, used, revoked, expired or bound to another email, or a
    principal key that is wrong or whose school already has a principal. The
    route answers 403 with this message."""


class RegistrationConflict(ValueError):
    """The request contradicts itself or its invite: a `school_id`/`role`
    that disagrees with the invite, an invite code AND a principal key, or a
    principal key without a school. The route answers 400. Nothing is created
    and no invite is consumed."""


class InviteNotFound(LookupError):
    """No such invite at the caller's school. Another school's code is
    reported the same way, so a principal learns nothing about other
    schools' invites."""


class InviteAlreadyUsed(Exception):
    """Revoking an invite that already created an account does nothing, so
    it is refused rather than reported as done."""


# What a refused registrant is told, in plain words. Specific on purpose: the
# person holding the code needs to know whether to retype it or ask for a new
# one, and the codes are 128-bit random, so confirming that one existed tells
# a guesser nothing.
_INVITE_UNKNOWN = "invite code not recognised; check it with your principal"
_INVITE_REFUSALS = {
    "used": "this invite code has already been used; ask your principal for a new one",
    "revoked": "this invite code was withdrawn; ask your principal for a new one",
    "expired": "this invite code has expired; ask your principal for a new one",
}


class UserStore:
    def __init__(self, db_path: Path, principal_bootstrap_key: str = ""):
        self._remote = durable_table("users")
        self._remote_sessions = durable_table("sessions")
        self._remote_invites = durable_table("invites")
        self._principal_key = principal_bootstrap_key
        self._conn_lock = threading.Lock()
        # Serialises "does this school have a principal yet?" with the insert
        # that answers it, within this process. Two processes can still race
        # it, but both would have to hold the operator's key.
        self._bootstrap_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._seed_default_users_if_empty()

    def _seed_default_users_if_empty(self) -> None:
        """Demo accounts, only on explicit request (`ACOS_SEED_DEMO_USERS=1`).

        This used to run whenever the local table was empty and the process
        was not under pytest. On Render that was every cold start (the users
        live in Supabase; the local file starts empty), so
        `principal@school.com` / `password123` was re-created as a working
        principal account in production each time. The opt-in replaces the
        pytest check as well: nothing seeds unless someone asked for it."""
        if os.environ.get(_DEMO_SEED_ENV, "").strip().lower() not in ("1", "true", "yes"):
            return
        # Never on the GCP release, even when requested: login falls back to
        # this SQLite file when Cloud SQL has no row, so a seeded
        # principal@school.com/password123 would be a known principal password
        # on a school's service (and the production guard refuses the flag).
        if not demo_accounts_enabled():
            return
        logger.warning(
            "%s is set: seeding demo accounts with a published password into "
            "demo_school. Never set this on a deployment real schools use.",
            _DEMO_SEED_ENV)
        with self._conn_lock:
            count = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count > 0:
                return
            default_accounts = [
                ("user_teacher_demo", "demo_school", "Demo Teacher", "teacher@school.com", "password123", "teacher"),
                ("user_principal_demo", "demo_school", "Demo Principal", "principal@school.com", "password123", "principal"),
            ]
            now = datetime.now(timezone.utc).isoformat()
            for uid, school_id, name, email, pwd, role in default_accounts:
                salt = secrets.token_bytes(16)
                pwd_hash = _hash_password(pwd, salt)
                self.conn.execute(
                    """INSERT OR IGNORE INTO users (id, school_id, name, email, password_hash, password_salt, role, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (uid, school_id, name, email, pwd_hash, salt.hex(), role, now)
                )
            self.conn.commit()

    # ---------------- registration / login ----------------

    def register(self, *, name: str, email: str, password: str,
                 invite_code: Optional[str] = None, principal_key: Optional[str] = None,
                 school_id: Optional[str] = None, role: Optional[str] = None) -> User:
        """Create an account. There are exactly two ways in:

        * `invite_code`: the normal path. School and role come from the
          invite. `school_id`/`role`, if sent, must agree with it.
        * `principal_key` + `school_id`: the operator's path for a school's
          FIRST principal, refused once that school has one.

        Anything else is `RegistrationRefused`. The caller never picks its
        own school or role. Blank strings count as "not sent", because an
        empty field claims nothing.
        """
        email = email.strip().lower()
        invite_code = (invite_code or "").strip()
        school_id = (school_id or "").strip() or None
        role = (role or "").strip() or None
        if invite_code and principal_key:
            raise RegistrationConflict(
                "send an invite code or a principal key, not both")
        if invite_code:
            return self._register_with_invite(
                code=invite_code, name=name, email=email, password=password,
                school_id=school_id, role=role)
        if principal_key:
            return self._register_first_principal(
                principal_key=principal_key, school_id=school_id, role=role,
                name=name, email=email, password=password)
        raise RegistrationRefused(
            "registration needs an invite code from your school's principal")

    def _register_with_invite(self, *, code: str, name: str, email: str, password: str,
                              school_id: Optional[str], role: Optional[str]) -> User:
        found = self._find_invite(code)
        if found is None:
            raise RegistrationRefused(_INVITE_UNKNOWN)
        invite, backend = found
        status = invite.status()
        if status != "open":
            raise RegistrationRefused(_INVITE_REFUSALS[status])
        if school_id is not None and school_id != invite.school_id:
            raise RegistrationConflict(
                "this invite code is for a different school than the schoolId sent; "
                "send the invite code on its own")
        if role is not None and role != invite.role:
            raise RegistrationConflict(
                f"this invite code is for a {invite.role} account, not {role!r}")
        if invite.email and invite.email != email:
            raise RegistrationRefused(
                "this invite code was issued for a different email address")
        if self.get_by_email(email) is not None:
            raise EmailAlreadyRegistered(email)

        # Claim first, then create. The claim is the atomic step (see
        # _claim_invite); creating the account afterwards means a lost race
        # never leaves an account behind. If the create fails, the claim is
        # released so a storage error does not burn the principal's invite.
        user_id = _new_user_id()
        self._claim_invite(invite.code, user_id, backend)
        try:
            return self._insert_user(user_id=user_id, school_id=invite.school_id,
                                     name=name, email=email, password=password,
                                     role=invite.role)
        except BaseException:
            self._release_invite(invite.code, user_id, backend)
            raise

    def _register_first_principal(self, *, principal_key: str, school_id: Optional[str],
                                  role: Optional[str], name: str, email: str,
                                  password: str) -> User:
        # The key is checked before anything else, so a caller without it
        # learns nothing: not whether a school has a principal, not whether an
        # email is registered.
        if not self._valid_principal_key(principal_key):
            raise RegistrationRefused("principal key not accepted")
        if school_id is None:
            raise RegistrationConflict("a principal key needs the schoolId it is for")
        if role is not None and role != "principal":
            raise RegistrationConflict(
                "a principal key registers the school's principal; do not send another role")
        with self._bootstrap_lock:
            if self._school_has_principal(school_id):
                raise RegistrationRefused(
                    "this school already has a principal; ask them for an invite")
            if self.get_by_email(email) is not None:
                raise EmailAlreadyRegistered(email)
            return self._insert_user(user_id=_new_user_id(), school_id=school_id,
                                     name=name, email=email, password=password,
                                     role="principal")

    def _school_has_principal(self, school_id: str) -> bool:
        """Deliberately not `users_for_school`: that falls back to local
        SQLite when the remote read fails, and during an outage the local
        file does not know the school's principal, so the bootstrap would
        mint a second one. Here a failed remote read propagates (503):
        fail closed."""
        if self._remote.enabled and self._remote.select(school_id=school_id, role="principal"):
            return True
        with self._conn_lock:
            row = self.conn.execute(
                "SELECT 1 FROM users WHERE school_id=? AND role='principal' LIMIT 1",
                (school_id,)).fetchone()
        return row is not None

    def _insert_user(self, *, user_id: str, school_id: str, name: str, email: str,
                     password: str, role: str) -> User:
        salt = secrets.token_bytes(16)
        created_at = datetime.now(timezone.utc).isoformat()
        row = {
            "id": user_id, "school_id": school_id, "name": name, "email": email,
            "password_hash": _hash_password(password, salt), "password_salt": salt.hex(),
            "role": role, "created_at": created_at,
        }
        # No local fallback when the remote store is on. The old path caught
        # the upsert failure and saved the account to the local file, which
        # Render wipes at the next cold start; nothing raised, so the invite
        # claim was never released and the person lost account and code
        # together. SupabaseUnavailable also wraps a 409, so a unique-email
        # violation on the remote table was swallowed and the same email was
        # saved twice. Now an outage propagates (the caller releases the
        # invite, api/main.py answers 503) and a unique violation is what
        # it is.
        if self._remote.enabled:
            try:
                self._remote.upsert(row, on_conflict="id")
            except (SupabaseUnavailable, requests.exceptions.RequestException) as exc:
                if _is_unique_violation(exc):
                    raise EmailAlreadyRegistered(email) from exc
                raise
        else:
            with self._conn_lock:
                try:
                    self.conn.execute(
                        """INSERT INTO users (id, school_id, name, email, password_hash, password_salt,
                                               role, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                        (user_id, school_id, name, email, row["password_hash"], row["password_salt"],
                         role, created_at),
                    )
                    self.conn.commit()
                except sqlite3.IntegrityError as exc:
                    # Two invites, one email, registered at the same moment:
                    # both passed get_by_email, the unique index stops the
                    # second. Report it as what it is rather than a 500.
                    self.conn.rollback()
                    raise EmailAlreadyRegistered(email) from exc
        return User(id=user_id, school_id=school_id, name=name, email=email, role=role,
                    created_at=created_at)

    # ---------------- invites ----------------

    def create_invite(self, *, school_id: str, role: str, created_by: str,
                      email: Optional[str] = None,
                      expires_in_days: Optional[int] = None) -> Invite:
        """A single-use invite to `school_id`. The caller (the route) passes
        the principal's own school, never one from a request body."""
        if not school_id:
            raise ValueError("an invite needs a school")
        if role not in INVITE_ROLES:
            raise ValueError(
                f"invite role must be one of {', '.join(INVITE_ROLES)}, not {role!r}; "
                "a school's principal comes from the operator, not an invite")
        days = _INVITE_DEFAULT_DAYS if expires_in_days is None else expires_in_days
        if isinstance(days, bool) or not isinstance(days, int) \
                or not 1 <= days <= _INVITE_MAX_DAYS:
            raise ValueError(f"expiresInDays must be a whole number from 1 to {_INVITE_MAX_DAYS}")
        bound_email = (email or "").strip().lower()
        if bound_email and "@" not in bound_email:
            raise ValueError(f"invite email is not an email address: {email!r}")

        now = datetime.now(timezone.utc)
        invite = Invite(
            code=secrets.token_urlsafe(16), school_id=school_id, role=role,
            email=bound_email, created_by=created_by, created_at=now.isoformat(),
            expires_at=(now + timedelta(days=days)).isoformat(),
        )
        row = asdict(invite)
        if self._remote_invites.enabled:
            # No local fallback when a remote store is configured, unlike
            # users and sessions. An invite is a credential that has to
            # outlive the process for days, and on Render the local disk is
            # wiped at every cold start: a code saved only there was handed
            # to the principal as working and then answered "not recognised"
            # after the next restart. The same happened on EVERY call when
            # the `invites` table had not been created yet (PostgREST 404).
            # A failure here propagates; the app's handler answers 503 and
            # the principal can retry.
            self._remote_invites.upsert(row, on_conflict="code")
        else:
            with self._conn_lock:
                self.conn.execute(
                    f"INSERT INTO invites ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                    tuple(row.values()))
                self.conn.commit()
        return invite

    def invites_for_school(self, school_id: str) -> list[Invite]:
        """Every invite the school has issued, used or not, newest first.
        Local rows are included too (a store that ran without a remote
        before one was configured keeps its invites there).

        A failed remote read raises (503) instead of listing local rows
        only: during an outage that list is empty or partial, and a
        principal reading it as complete would issue duplicates of codes
        that are still live."""
        merged: dict[str, Invite] = {}
        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT * FROM invites WHERE school_id=?", (school_id,)).fetchall()
        for r in rows:
            merged[r["code"]] = _row_to_invite(dict(r))
        if self._remote_invites.enabled:
            for r in self._remote_invites.select(school_id=school_id):
                merged[r["code"]] = _row_to_invite(r)
        return sorted(merged.values(), key=lambda i: i.created_at, reverse=True)

    def revoke_invite(self, *, code: str, school_id: str) -> Invite:
        found = self._find_invite(code)
        if found is None or found[0].school_id != school_id:
            raise InviteNotFound(code)
        invite, backend = found
        if invite.used_at:
            raise InviteAlreadyUsed(code)
        if invite.revoked_at:
            return invite
        # Conditional on "not used", so a revoke and a registration racing on
        # the same code cannot both win (the claim is conditional on "not
        # revoked" in the same way).
        revoked_at = datetime.now(timezone.utc).isoformat()
        if backend == "remote":
            self._remote_invites.update({"revoked_at": revoked_at},
                                        code=code, used_at="", revoked_at="")
        else:
            with self._conn_lock:
                self.conn.execute(
                    "UPDATE invites SET revoked_at=? WHERE code=? AND used_at='' AND revoked_at=''",
                    (revoked_at, code))
                self.conn.commit()
        after = self._find_invite(code)
        if after is None:
            raise InviteNotFound(code)
        if after[0].used_at:
            raise InviteAlreadyUsed(code)
        return after[0]

    def _find_invite(self, code: str) -> Optional[tuple[Invite, str]]:
        """The invite and which backend holds it ("remote" | "local"); the
        claim and the revoke must write to the same place they read from.

        When the remote read fails, only a positive local hit is returned.
        No local match is NOT "unknown": the local file does not hold
        remotely stored invites, so answering None here told a user with a
        valid code "invite code not recognised" (403) and a principal "no
        such invite at your school" (404) when the truth was that storage
        was down. The remote error is re-raised instead (503)."""
        remote_error: Optional[BaseException] = None
        if self._remote_invites.enabled:
            try:
                rows = self._remote_invites.select(code=code)
                if rows:
                    return _row_to_invite(rows[0]), "remote"
            except (SupabaseUnavailable, requests.exceptions.RequestException) as exc:
                remote_error = exc
        with self._conn_lock:
            r = self.conn.execute("SELECT * FROM invites WHERE code=?", (code,)).fetchone()
        if r:
            return _row_to_invite(dict(r)), "local"
        if remote_error is not None:
            raise remote_error
        return None

    def _claim_invite(self, code: str, user_id: str, backend: str) -> None:
        """Mark the invite used by `user_id`, or raise RegistrationRefused if
        someone else got there first.

        Single use is enforced HERE, by the storage layer, not by the status
        check the caller already made: two registrations racing on one code
        both pass that check. The claim is one UPDATE conditional on
        `used_at = '' AND revoked_at = ''`, and only one such UPDATE can
        match. SQLite reports it through rowcount. The remote API returns no
        rowcount, so the row is read back and the claim won only if
        `used_by` is ours. A remote failure here propagates (503) rather
        than falling back to a local copy that does not hold the invite."""
        used_at = datetime.now(timezone.utc).isoformat()
        if backend == "remote":
            self._remote_invites.update({"used_by": user_id, "used_at": used_at},
                                        code=code, used_at="", revoked_at="")
            rows = self._remote_invites.select(code=code)
            won = bool(rows) and rows[0].get("used_by") == user_id
        else:
            with self._conn_lock:
                cur = self.conn.execute(
                    "UPDATE invites SET used_by=?, used_at=? "
                    "WHERE code=? AND used_at='' AND revoked_at=''",
                    (user_id, used_at, code))
                self.conn.commit()
                won = cur.rowcount == 1
        if not won:
            raise RegistrationRefused(
                "this invite code has just been used or withdrawn; ask your principal for a new one")

    def _release_invite(self, code: str, user_id: str, backend: str) -> None:
        """Undo OUR claim (only where `used_by` is still us) after the account
        insert failed. If even this fails, the invite stays used. That fails
        closed, and the principal can issue another."""
        try:
            if backend == "remote":
                self._remote_invites.update({"used_by": "", "used_at": ""},
                                            code=code, used_by=user_id)
            else:
                with self._conn_lock:
                    self.conn.execute(
                        "UPDATE invites SET used_by='', used_at='' WHERE code=? AND used_by=?",
                        (code, user_id))
                    self.conn.commit()
        except Exception:  # noqa: BLE001 - logged; the original error is re-raised by the caller
            # Never log the code itself: it is a credential until it expires.
            logger.warning("could not release invite %s... after a failed registration; "
                           "it stays used", code[:4], exc_info=True)

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
        saved_remotely = False
        if self._remote_sessions.enabled:
            try:
                self._remote_sessions.upsert(row, on_conflict="token")
                saved_remotely = True
            except (SupabaseUnavailable, requests.exceptions.RequestException):
                logger.warning("Supabase unavailable for create_session, falling back to local SQLite", exc_info=True)
        if not saved_remotely:
            with self._conn_lock:
                self.conn.execute(
                    "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                    (token, user_id, row["created_at"], expires_at),
                )
                self.conn.commit()
        return token

    def user_for_session(self, token: str) -> Optional[User]:
        session_row = None
        if self._remote_sessions.enabled:
            try:
                rows = self._remote_sessions.select(token=token)
                session_row = rows[0] if rows else None
            except (SupabaseUnavailable, requests.exceptions.RequestException):
                logger.warning("Supabase unavailable for user_for_session, falling back to local SQLite", exc_info=True)
        if session_row is None:
            with self._conn_lock:
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
            try:
                self._remote_sessions.delete(token=token)
            except (SupabaseUnavailable, requests.exceptions.RequestException) as exc:
                # Was `pass`. The local delete below still runs, so logout
                # succeeds from the caller's point of view -- but a failed
                # REMOTE delete means the session can outlive the logout on the
                # durable store. Security-relevant, so it is logged.
                logger.warning(
                    "remote session delete failed during logout; the session may "
                    "outlive it on the remote store (%s: %s)",
                    type(exc).__name__, exc)
        with self._conn_lock:
            self.conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            self.conn.commit()

    # ---------------- lookups ----------------

    def get(self, user_id: str) -> Optional[User]:
        if self._remote.enabled:
            try:
                rows = self._remote.select(id=user_id)
                if rows:
                    return _row_to_user(rows[0])
            except (SupabaseUnavailable, requests.exceptions.RequestException):
                logger.warning("Supabase unavailable for get(user_id), falling back to local SQLite", exc_info=True)
        with self._conn_lock:
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
            try:
                rows = self._remote.select(school_id=school_id, **({"role": role} if role else {}))
                if rows:
                    return [_row_to_user(r) for r in rows]
            except (SupabaseUnavailable, requests.exceptions.RequestException):
                logger.warning("Supabase unavailable for users_for_school, falling back to local SQLite", exc_info=True)
        with self._conn_lock:
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
            try:
                rows = self._remote.select(email=email)
                if rows:
                    return rows[0]
            except (SupabaseUnavailable, requests.exceptions.RequestException):
                logger.warning("Supabase unavailable for _raw_by_email, falling back to local SQLite", exc_info=True)
        with self._conn_lock:
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


def _is_unique_violation(exc: BaseException) -> bool:
    """A remote write refused by a unique index, as opposed to an outage.
    PostgREST answers 409 for SQLSTATE 23505; Cloud SQL (PostgresUnavailable)
    has no HTTP status, so the wrapped psycopg error's sqlstate is read."""
    if getattr(exc, "status", None) == 409:
        return True
    cause = exc.__cause__
    return getattr(cause, "sqlstate", None) == "23505"


def _row_to_user(row: dict) -> User:
    return User(id=row["id"], school_id=row["school_id"], name=row["name"], email=row["email"],
                role=row["role"], created_at=row["created_at"])


def _row_to_invite(row: dict) -> Invite:
    # `or ""`: a remote row written by hand (or by a schema without the
    # DEFAULT '') may carry NULL. It lists as "not yet", but the claim's
    # `used_at = ''` filter does not match NULL, so such a row is refused at
    # registration rather than used twice.
    return Invite(
        code=row["code"], school_id=row["school_id"], role=row["role"],
        email=row.get("email") or "", created_by=row.get("created_by") or "",
        created_at=row.get("created_at") or "", expires_at=row["expires_at"],
        used_by=row.get("used_by") or "", used_at=row.get("used_at") or "",
        revoked_at=row.get("revoked_at") or "",
    )


_INSTANCE: Optional[dict] = {}


def get_user_store(data_root: Path, principal_bootstrap_key: str = "") -> UserStore:
    # Keyed on the RESOLVED PATH, not a bare global.
    #
    # This was `if _INSTANCE is None`, so the FIRST data_root ever passed won
    # for the life of the process and every later call with a different root got
    # that first instance. In production only one root is used, which is why it
    # survived unspotted -- but anywhere a process touches more than one root
    # (tests, a CLI run beside a server, tooling) the store silently reads and
    # writes the WRONG database. For the audit log that means a compliance trail
    # filed against another data root.
    # The bootstrap key is part of the key as well: it is applied at
    # construction, so two calls for the same file with different keys are two
    # different stores and the first must not answer for the second.
    # `_INSTANCE` is a PATH-KEYED cache, not one store. Tests reset it with
    # `monkeypatch.setattr(..., "_INSTANCE", None)`, which is kept working,
    # but the reset is no longer load-bearing: two data roots now get two
    # stores by construction. See git history for the single-global version.
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = {}                      # a caller cleared the cache
    key = (str(Path(data_root) / "users" / "users.sqlite"),
           principal_bootstrap_key or "")
    instance = _INSTANCE.get(key)
    if instance is None:
        instance = UserStore(Path(key[0]),
                             principal_bootstrap_key=principal_bootstrap_key)
        _INSTANCE[key] = instance
    return instance
