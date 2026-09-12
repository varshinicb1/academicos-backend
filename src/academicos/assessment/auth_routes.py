"""Teacher/principal authentication. See users.py's module docstring for the
password-hashing/session-token design choices and the "first registrant per
school becomes principal" bootstrap rule.

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

from ..config import Config
from .schemas import Camel
from .users import EmailAlreadyRegistered, InvalidCredentials, User, UserStore, get_user_store

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
    school_id: str
    name: str
    email: str
    password: str
    principal_key: Optional[str] = None
    # Self-selected role: "teacher" (default) or "student". Never
    # "principal" -- that's granted only by a valid principal_key,
    # unaffected by this field (see UserStore.register's docstring).
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


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    if not req.password or len(req.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    store = _require()
    try:
        user = store.register(school_id=req.school_id, name=req.name, email=req.email,
                               password=req.password, principal_key=req.principal_key,
                               requested_role=req.role or "teacher")
    except EmailAlreadyRegistered:
        raise HTTPException(409, "an account with this email already exists")
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = store.create_session(user.id)
    return AuthResponse(user=_to_response(user), token=token)


@router.post("/login", response_model=AuthResponse)
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


def get_current_user(authorization: str = Header(default="")) -> User:
    """Required-auth dependency: 401s if there's no valid session token."""
    token = _bearer_token(authorization)
    if token is None:
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
