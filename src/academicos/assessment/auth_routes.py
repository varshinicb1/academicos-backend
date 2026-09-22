"""Teacher/principal authentication. See users.py's module docstring for the
password-hashing/session-token design choices, the invite rule (a new
account's school and role come from a principal-issued invite, never from the
request) and what the principal bootstrap key can still do (create a school's
first principal, nothing more).

`get_current_user` / `get_current_user_optional` are the dependencies other
routers use to identify the caller: `Depends(get_current_user)` for anything
that must have a real identity (the principal-approve action below),
`Depends(get_current_user_optional)` for existing routes that accepted a
free-text reviewerId/actor before real auth existed -- they now prefer the
authenticated identity when present and fall back to the client-supplied
value otherwise, so an older/offline client that has never logged in still
works exactly as before.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..api.rate_limit import rate_limit_login, rate_limit_register
from ..config import Config
from .schemas import Camel
from .users import (
    EmailAlreadyRegistered, InvalidCredentials, Invite, InviteAlreadyUsed, InviteNotFound,
    RegistrationRefused, User, UserStore, get_user_store,
)

router = APIRouter(prefix="/api/v1/auth")

_cfg: Optional[Config] = None
_users: Optional[UserStore] = None


def init(config: Config) -> None:
    global _cfg, _users
    _cfg = config
    _users = get_user_store(config.data_root, config.principal_bootstrap_key)


def _require() -> UserStore:
    if _users is None:
        raise HTTPException(503, "auth module not initialized")
    return _users


class RegisterRequest(Camel):
    name: str
    email: str
    password: str
    # The normal way in: a single-use code a principal issued. The account's
    # school and role are the invite's.
    invite_code: Optional[str] = None
    # With principal_key: the school whose FIRST principal this is. With an
    # invite it is optional, and if sent it must match the invite (400).
    school_id: Optional[str] = None
    principal_key: Optional[str] = None
    # Never chosen by the caller. Accepted only when it agrees with the
    # invite ("principal" on the key path), so an old client that still sends
    # it is told when it disagrees instead of silently landing elsewhere.
    role: Optional[str] = None


class LoginRequest(Camel):
    email: str
    password: str


class UserResponse(Camel):
    id: str
    school_id: str
    name: str
    email: str
    role: str


class AuthResponse(Camel):
    user: UserResponse
    token: str


def _to_response(user: User) -> UserResponse:
    return UserResponse(id=user.id, school_id=user.school_id, name=user.name,
                         email=user.email, role=user.role)


@router.post("/register", response_model=AuthResponse, dependencies=[Depends(rate_limit_register)])
def register(req: RegisterRequest) -> AuthResponse:
    """Needs an invite code, or (for a school's first principal only) the
    operator's principal key with a schoolId. 403 without either, or when
    the invite is unknown/used/revoked/expired/for another email, or the
    school already has a principal. 400 when schoolId/role disagree with the
    invite. Until 2026-09-21 any caller could name any school here."""
    if not req.password or len(req.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    store = _require()
    try:
        user = store.register(name=req.name, email=req.email, password=req.password,
                               invite_code=req.invite_code, principal_key=req.principal_key,
                               school_id=req.school_id, role=req.role)
    except RegistrationRefused as e:
        raise HTTPException(403, str(e))
    except EmailAlreadyRegistered:
        raise HTTPException(409, "an account with this email already exists")
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = store.create_session(user.id)
    return AuthResponse(user=_to_response(user), token=token)


@router.post("/login", response_model=AuthResponse, dependencies=[Depends(rate_limit_login)])
def login(req: LoginRequest) -> AuthResponse:
    store = _require()
    try:
        user = store.authenticate(email=req.email, password=req.password)
    except InvalidCredentials:
        raise HTTPException(401, "incorrect email or password")
    token = store.create_session(user.id)
    return AuthResponse(user=_to_response(user), token=token)


@router.post("/logout")
def logout(authorization: str = Header(default="")) -> dict:
    token = _bearer_token(authorization)
    if token:
        _require().delete_session(token)
    return {"ok": True}


def _bearer_token(authorization: str) -> Optional[str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization[7:].strip() or None


# Declared as a real security scheme rather than a bare `Header(...)`.
# Functionally identical at runtime, but FastAPI now publishes
# `components.securitySchemes.bearerAuth` and marks all 89 routes that depend
# on get_current_user/require_principal as secured. Before this, the generated
# OpenAPI schema advertised `securitySchemes: None` and zero secured
# operations -- i.e. /docs told every integrator the entire API was public,
# including endpoints that 401 on the first call. `auto_error=False` keeps our
# own 401 wording instead of HTTPBearer's default 403.
_bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Session token from POST /api/v1/auth/login, sent as "
                "`Authorization: Bearer <token>`.",
)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> User:
    """Required-auth dependency: 401s if there's no valid session token."""
    token = creds.credentials if creds is not None else None
    if not token:
        raise HTTPException(401, "missing bearer token")
    user = _require().user_for_session(token)
    if user is None:
        raise HTTPException(401, "session expired or invalid; log in again")
    return user


def get_current_user_optional(authorization: str = Header(default="")) -> Optional[User]:
    """Best-effort identity for routes that pre-date auth and must keep
    working for a client that has never logged in -- returns None instead of
    401ing when there's no token or it's invalid/expired."""
    token = _bearer_token(authorization)
    if token is None or _users is None:
        return None
    return _users.user_for_session(token)


def require_principal(current: User = Depends(get_current_user)) -> User:
    if current.role != "principal":
        raise HTTPException(403, "this action requires the principal role")
    return current


# The roles that may do a teacher's work. A tuple of what is allowed, not a
# check for "student": a role added later (parent, guardian) is refused here
# until someone decides otherwise.
STAFF_ROLES = ("teacher", "principal")


def require_staff(current: User = Depends(get_current_user)) -> User:
    """Teacher or principal. Until 2026-09-21 the teacher routes depended on
    get_current_user alone, which proves identity and nothing else: a
    student-role token read every classmate's answer to a question and
    changed another student's marks. api/route_policy.py lists which routes
    carry this gate; tests/test_role_gates.py holds every route to that list."""
    if current.role not in STAFF_ROLES:
        raise HTTPException(403, "this action requires a teacher or principal account")
    return current


@router.get("/me", response_model=UserResponse)
def me(current: User = Depends(get_current_user)) -> UserResponse:
    return _to_response(current)


@router.get("/users", response_model=list[UserResponse])
def list_school_users(role: Optional[str] = None,
                      principal: User = Depends(require_principal)) -> list[UserResponse]:
    """The real roster an admin UI needs to assign a teacher to a book or
    enroll a student in a class -- without a name-searchable list, those
    actions would require already knowing a raw user id. Principal-gated
    and scoped to the caller's own school_id (never a query parameter),
    same posture as every other per-school administrative read in this
    codebase."""
    if role is not None and role not in ("teacher", "principal", "student"):
        raise HTTPException(400, f"invalid role filter: {role!r}")
    users = _require().users_for_school(principal.school_id, role=role)
    return [_to_response(u) for u in users]


# ---------------- invites (principal only, own school only) ----------------


class InviteCreateRequest(Camel):
    role: str  # "teacher" | "student"
    # Bind the invite to one person; any email may use it when omitted.
    email: Optional[str] = None
    # 1..90, default 14 (users.py explains the cap).
    expires_in_days: Optional[int] = None


class InviteResponse(Camel):
    code: str
    role: str
    email: Optional[str] = None
    expires_at: str
    created_at: str
    created_by: str
    status: str  # "open" | "used" | "revoked" | "expired"
    used_by: Optional[str] = None
    used_at: Optional[str] = None
    revoked_at: Optional[str] = None


def _invite_response(invite: Invite) -> InviteResponse:
    return InviteResponse(
        code=invite.code, role=invite.role, email=invite.email or None,
        expires_at=invite.expires_at, created_at=invite.created_at,
        created_by=invite.created_by, status=invite.status(),
        used_by=invite.used_by or None, used_at=invite.used_at or None,
        revoked_at=invite.revoked_at or None,
    )


@router.post("/invites", response_model=InviteResponse)
def create_invite(req: InviteCreateRequest,
                  principal: User = Depends(require_principal)) -> InviteResponse:
    """A single-use invite to the CALLER's school. There is no school field
    to send: a principal can only ever invite people into their own school."""
    try:
        invite = _require().create_invite(
            school_id=principal.school_id, role=req.role, created_by=principal.id,
            email=req.email, expires_in_days=req.expires_in_days)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _invite_response(invite)


@router.get("/invites", response_model=list[InviteResponse])
def list_invites(principal: User = Depends(require_principal)) -> list[InviteResponse]:
    """The caller's school's invites, used and unused, newest first. Each
    row says who created it and which account used it and when."""
    return [_invite_response(i) for i in _require().invites_for_school(principal.school_id)]


@router.delete("/invites/{code}", response_model=InviteResponse)
def revoke_invite(code: str, principal: User = Depends(require_principal)) -> InviteResponse:
    """Withdraw an unused invite. Another school's code is a 404, the same as
    an unknown one."""
    try:
        invite = _require().revoke_invite(code=code, school_id=principal.school_id)
    except InviteNotFound:
        raise HTTPException(404, "no such invite at your school")
    except InviteAlreadyUsed:
        raise HTTPException(409, "this invite has already been used and cannot be withdrawn")
    return _invite_response(invite)
