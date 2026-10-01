"""Sign in with Google (INT-1): for anyone who already has an AcademicOS
account under that Google address. Registration stays by invite -- Google
proves who is at the keyboard, not that the school enrolled them.

The browser gets an ID token from Google Identity Services and posts it here.
It is checked against Google's published signing keys (RS256), for this
deployment's client id (`ACOS_GOOGLE_CLIENT_ID`, a public value), an issuer
of accounts.google.com, an expiry, and a verified email. Nothing else from
Google is kept. With no client id configured the route says so, and
`/auth/providers` tells the sign-in page not to show the button.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from ..api.rate_limit import RateLimiter, get_client_ip
from ..assessment.schemas import Camel
from ..assessment.users import AccountClosed

router = APIRouter(prefix="/api/v1/auth")

GOOGLE_CERTS = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = ("accounts.google.com", "https://accounts.google.com")

_limiter = RateLimiter(max_requests=20, window_seconds=600)
_jwks: Any = None


def client_id() -> Optional[str]:
    return (os.environ.get("ACOS_GOOGLE_CLIENT_ID") or "").strip() or None


def _signing_key(token: str):
    """Google's key for this token, from its published set (cached by
    PyJWKClient). Tests replace this."""
    global _jwks
    import jwt
    if _jwks is None:
        _jwks = jwt.PyJWKClient(GOOGLE_CERTS, cache_keys=True, lifespan=3600)
    return _jwks.get_signing_key_from_jwt(token).key


def verify(token: str, audience: str) -> dict:
    """The token's claims when Google signed it for us and it is current;
    ValueError, with the reason, otherwise."""
    import jwt
    try:
        claims = jwt.decode(token, _signing_key(token), algorithms=["RS256"], audience=audience,
                            options={"require": ["exp", "iat", "iss", "aud", "sub"]}, leeway=30)
    except jwt.PyJWTError as e:
        raise ValueError(f"the Google sign-in could not be checked ({e})") from None
    if claims.get("iss") not in ISSUERS:
        raise ValueError("the sign-in did not come from Google")
    if not claims.get("email") or claims.get("email_verified") is not True:
        raise ValueError("Google has not verified this account's email address")
    return claims


class GoogleSignIn(Camel):
    credential: str = Field(min_length=20, max_length=4096)


@router.get("/providers")
def providers() -> dict:
    """Which ways in this deployment offers, for the sign-in page: Google
    (with the public client id the button needs) and emailed codes."""
    from ..assessment import mailer
    return {"google": {"clientId": client_id()} if client_id() else None,
            "emailCode": any(b.get("configured") for b in mailer.available_backends().values())}


@router.post("/google")
def sign_in_with_google(req: GoogleSignIn, request: Request):
    """Sign in with a Google ID token: the same answer as a password sign-in."""
    from ..assessment import auth_routes
    audience = client_id()
    if audience is None:
        raise HTTPException(503, "Google sign-in is not set up for this school; use your password")
    _limiter.check(f"ip:{get_client_ip(request)}")
    try:
        claims = verify(req.credential, audience)
    except ValueError as e:
        raise HTTPException(401, str(e))
    users = auth_routes._require()
    user = users.get_by_email(claims["email"].strip().lower())
    if user is None:
        raise HTTPException(401, f"No AcademicOS account uses {claims['email']}. Ask your school for an "
                                 "invite, or sign in with the email your school has.")
    try:
        token = users.create_session(user.id)
    except AccountClosed:
        raise HTTPException(403, auth_routes.CLOSED_DETAIL)
    return auth_routes.AuthResponse(user=auth_routes._to_response(user), token=token)
