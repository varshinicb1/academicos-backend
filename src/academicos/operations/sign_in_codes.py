"""Sign in with a one-time code by email (ROLE-2) -- for students and
parents, who should not need a password to open their homework or their
child's progress.

A code is six digits, lives ten minutes, allows five tries, and is stored
only as a hash. Asking for a code answers the same whether or not the email
has an account, so the route cannot be used to find out who is enrolled;
only student and parent accounts are sent one (staff use their password).
Requests are rate-limited per client and per email. Nothing is sent until
the school's email is set up (SMTP), and the route says so.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from ..api.rate_limit import RateLimiter, get_client_ip
from ..assessment.schemas import Camel

router = APIRouter(prefix="/api/v1/auth/code")

CODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sign_in_codes (
    email TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""
LIFETIME = timedelta(minutes=10)
MAX_ATTEMPTS = 5
ROLES = ("student", "parent")

_request_limiter = RateLimiter(max_requests=5, window_seconds=600)
_verify_limiter = RateLimiter(max_requests=20, window_seconds=600)


def _hash(email: str, code: str) -> str:
    return hashlib.sha256(f"{email}:{code}".encode("utf-8")).hexdigest()


class CodesMixin:
    """Added to OperationsStore."""

    def issue_code(self, email: str) -> str:
        code = f"{secrets.randbelow(10**6):06d}"
        now = datetime.now(timezone.utc)
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO sign_in_codes (email, code_hash, expires_at, attempts, created_at) VALUES (?,?,?,0,?)"
                " ON CONFLICT(email) DO UPDATE SET code_hash=excluded.code_hash, expires_at=excluded.expires_at,"
                " attempts=0, created_at=excluded.created_at",
                (email, _hash(email, code), (now + LIFETIME).isoformat(), now.isoformat()))
            self._commit()
        return code

    def check_code(self, email: str, code: str) -> bool:
        """True once, for the right unexpired code; every try counts."""
        row = self._fetchone("SELECT * FROM sign_in_codes WHERE email=?", (email,))
        if row is None:
            return False
        expired = datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc)
        ok = not expired and hmac.compare_digest(row["code_hash"], _hash(email, code))
        with self._conn_lock:
            if ok or expired or row["attempts"] + 1 >= MAX_ATTEMPTS:
                self.conn.execute("DELETE FROM sign_in_codes WHERE email=?", (email,))
            else:
                self.conn.execute("UPDATE sign_in_codes SET attempts=attempts+1 WHERE email=?", (email,))
            self._commit()
        return ok


class CodeRequest(Camel):
    email: str = Field(min_length=3, max_length=200)


class CodeVerify(Camel):
    email: str = Field(min_length=3, max_length=200)
    code: str = Field(min_length=6, max_length=6)


def _limit(limiter: RateLimiter, request: Request, email: str) -> None:
    limiter.check(f"ip:{get_client_ip(request)}")
    limiter.check(f"email:{email}")


def _store():
    from .routes import _cfg, store
    if _cfg is None:
        raise HTTPException(503, "sign-in by code is not initialised")
    return store()


@router.post("/request")
def request_code(req: CodeRequest, request: Request) -> dict:
    """Email a sign-in code to a student or parent account. The answer is
    the same whether or not the email has one."""
    from ..assessment import auth_routes, mailer
    from .notifications import send_email
    s = _store()
    email = req.email.strip().lower()
    _limit(_request_limiter, request, email)
    if not any(b.get("configured") for b in mailer.available_backends().values()):
        raise HTTPException(503, "sign-in by code needs the school's email to be set up; use your password for now")
    user = auth_routes._require().get_by_email(email)
    if user is not None and user.role in ROLES:
        code = s.issue_code(email)
        send_email(email, "Your AcademicOS sign-in code",
                   f"Your sign-in code is {code}. It works once, for 10 minutes.\n\n"
                   "If you did not ask for it, ignore this email.")
    return {"detail": "If this email belongs to a student or parent account, a code is on its way. "
                      "It works for 10 minutes."}


@router.post("/verify")
def verify_code(req: CodeVerify, request: Request):
    """Sign in with the emailed code: the same answer as a password sign-in."""
    from ..assessment import auth_routes
    s = _store()
    email = req.email.strip().lower()
    _limit(_verify_limiter, request, email)
    user = auth_routes._require().get_by_email(email)
    if user is None or user.role not in ROLES or not s.check_code(email, req.code.strip()):
        raise HTTPException(401, "that code is not right or has expired; ask for a new one")
    token = auth_routes._require().create_session(user.id)
    return auth_routes.AuthResponse(user=auth_routes._to_response(user), token=token)
