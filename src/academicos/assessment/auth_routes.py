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

import logging
import os
import secrets
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

from ..api.rate_limit import account_failures, rate_limit_login, rate_limit_register
from ..config import Config
from .schemas import Camel
from .users import (
    OWNER_ROLE, AccountClosed, EmailAlreadyRegistered, InvalidCredentials, Invite, InviteAlreadyUsed,
    InviteNotFound, RegistrationRefused, User, UserStore, get_user_store, is_closed,
)

CLOSED_DETAIL = "your school has closed this account; ask the school office if this is a mistake"

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
    # ROLE-4: an owner account. With no school chosen its role is "owner" and
    # its schoolId ""; acting in one of its schools it is that school's
    # principal (role "principal") and this stays true, so the web can offer
    # the school switcher.
    is_owner: bool = False


class AuthResponse(Camel):
    user: UserResponse
    token: str


def _to_response(user: User) -> UserResponse:
    return UserResponse(id=user.id, school_id=user.school_id, name=user.name,
                        email=user.email, role=user.role,
                        is_owner=user.acting_owner or user.role == OWNER_ROLE)


class SchoolInviteRequest(Camel):
    school_id: str
    expires_in_days: Optional[int] = None


class SchoolInviteResponse(Camel):
    school_id: str
    code: str
    expires_at: str
    # Opens Create account with the code filled in; the school and the
    # principal role come from the invite.
    link: str


@router.post("/operator/school-invites", response_model=SchoolInviteResponse,
             dependencies=[Depends(rate_limit_login)])
def create_school_invite(req: SchoolInviteRequest,
                         operator_key: Optional[str] = Header(default=None, alias="X-Operator-Key")
                         ) -> SchoolInviteResponse:
    """The operator onboards a new school (2026-10-01): a single-use invite
    for its first principal, as a link. Needs the operator key in
    `X-Operator-Key` (scripts/new_school.py reads it from Secret Manager).
    403 without it, 409 once the school has a principal, 422 for a bad ID."""
    store = _require()
    if not store._valid_principal_key(operator_key):
        raise HTTPException(403, "operator key not accepted")
    try:
        invite = store.create_principal_invite(school_id=req.school_id,
                                               expires_in_days=req.expires_in_days)
    except RegistrationRefused as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    if _cfg is not None:
        from .audit_log import get_audit_log
        get_audit_log(_cfg.data_root).append(
            "school_invite_created", actor="operator",
            details={"schoolId": invite.school_id, "expiresAt": invite.expires_at})
    origin = (os.environ.get("ACOS_WEB_ORIGIN")
              or (os.environ.get("ACOS_CORS_ORIGINS") or "").split(",")[0].strip()
              or "https://vidyuthlabs.web.app").rstrip("/")
    return SchoolInviteResponse(school_id=invite.school_id, code=invite.code, expires_at=invite.expires_at,
                                link=f"{origin}/#/login?invite={quote(invite.code)}")


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
    for hook in list(_REGISTER_HOOKS):
        try:
            hook(user, req.invite_code)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("a register hook failed for %s", user.id, exc_info=True)
    token = store.create_session(user.id)
    return AuthResponse(user=_to_response(user), token=token)


# Run after an account is created: (user, invite code or None). The parent
# link (operations/guardian_routes.on_register) is one. A hook that
# fails is logged; the account stays, and the principal can link by hand.
_REGISTER_HOOKS: list = []


def add_register_hook(fn) -> None:
    if fn not in _REGISTER_HOOKS:
        _REGISTER_HOOKS.append(fn)


@router.post("/login", response_model=AuthResponse, dependencies=[Depends(rate_limit_login)])
def login(req: LoginRequest) -> AuthResponse:
    store = _require()
    # Brute force is stopped per account (rate_limit.py): after too many
    # wrong passwords for one email, from any number of addresses, that
    # email is refused for a while -- a right password included, so the
    # guesser cannot tell when they hit it.
    account = f"login-failures:{req.email.strip().lower()}"
    wait = account_failures.blocked(account)
    if wait is not None:
        raise HTTPException(429, f"too many wrong passwords for this account; try again in "
                                 f"{max(1, round(wait / 60))} minute(s)", headers={"Retry-After": str(wait)})
    try:
        user = store.authenticate(email=req.email, password=req.password)
        token = store.create_session(user.id)
    except InvalidCredentials:
        account_failures.record(account)
        raise HTTPException(401, "incorrect email or password")
    except AccountClosed:
        raise HTTPException(403, CLOSED_DETAIL)
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


# ---------------- owners over several schools (ROLE-4) ----------------
#
# An owner account has no school of its own. operations/owners.py records
# which schools it is linked to and which one each of its sessions has
# chosen; it plugs that in at start-up (set_owner_scope), the way the grant
# checker is plugged in, because the operations package imports this module.

OWNER_DETAIL = "this action requires an owner account"
OWNER_CHOOSE_DETAIL = ("choose one of your schools first: an owner acts in one school "
                       "at a time (POST /api/v1/owner/active-school)")
_owner_scope = None


def set_owner_scope(fn) -> None:
    """fn(owner: User, token: str) -> the school id the session has chosen,
    or None when it has chosen none or that school's link was revoked."""
    global _owner_scope
    _owner_scope = fn


def acting_view(owner: User, school_id: str) -> User:
    """The owner as the principal of one of its linked schools. Every route
    then scopes by `school_id` exactly as it does for that school's own
    principal, so no route needs to know owners exist."""
    return User(id=owner.id, school_id=school_id, name=owner.name, email=owner.email,
                role="principal", created_at=owner.created_at, acting_owner=True)


def _as_seen_by_routes(user: User, token: str) -> User:
    """An owner whose session has chosen a still-linked school becomes that
    school's principal view; anyone else is unchanged. The link is checked on
    every request, so a principal's revoke takes effect on the owner's very
    next call."""
    if user.role != OWNER_ROLE or _owner_scope is None:
        return user
    school_id = _owner_scope(user, token)
    return acting_view(user, school_id) if school_id else user


def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> User:
    """Required-auth dependency: 401s if there's no valid session token.

    A parent account may call only the routes in route_policy.PARENT_PATHS
    (their own inbox and settings, and their linked children's pages). Every
    other route answers 403 here, before its handler runs: handlers written
    before parents existed tell students from staff by `role == "student"`,
    and a parent must never be taken for staff by one of them.

    An owner (ROLE-4) is the principal of the school its session has chosen.
    With none chosen it may call only route_policy.OWNER_PATHS: its school_id
    is "", and no route should be asked what "no school" holds."""
    token = creds.credentials if creds is not None else None
    if not token:
        raise HTTPException(401, "missing bearer token")
    user = _require().user_for_session(token)
    if user is None:
        raise HTTPException(401, "session expired or invalid; log in again")
    user = _as_seen_by_routes(user, token)
    # For api/main.py's admin audit: who made the request, and for an owner
    # in which of its schools.
    request.scope["acos_actor"] = user.id
    if user.acting_owner:
        request.scope["acos_owner_school"] = user.school_id
    route_path = getattr(request.scope.get("route"), "path", None)
    if user.role == "parent":
        from ..api.route_policy import PARENT_PATHS
        if route_path not in PARENT_PATHS:
            raise HTTPException(403, "a parent account can see only their children's pages")
    if user.role == OWNER_ROLE:
        from ..api.route_policy import OWNER_PATHS
        if route_path not in OWNER_PATHS:
            raise HTTPException(403, OWNER_CHOOSE_DETAIL)
    return user


def get_current_user_optional(authorization: str = Header(default="")) -> Optional[User]:
    """Best-effort identity for routes that pre-date auth and must keep
    working for a client that has never logged in -- returns None instead of
    401ing when there's no token or it's invalid/expired."""
    token = _bearer_token(authorization)
    if token is None or _users is None:
        return None
    user = _users.user_for_session(token)
    return _as_seen_by_routes(user, token) if user is not None else None


def require_owner(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> User:
    """The owner account itself (ROLE-4), whichever school its session has
    chosen: its own routes (its schools, switching, the overview) act on the
    account, never on one school. 403 for every other account, a principal
    included."""
    token = creds.credentials if creds is not None else None
    if not token:
        raise HTTPException(401, "missing bearer token")
    user = _require().user_for_session(token)
    if user is None:
        raise HTTPException(401, "session expired or invalid; log in again")
    if user.role != OWNER_ROLE:
        raise HTTPException(403, OWNER_DETAIL)
    request.scope["acos_actor"] = user.id
    return user


def require_principal(current: User = Depends(get_current_user)) -> User:
    if current.role != "principal":
        raise HTTPException(403, "this action requires the principal role")
    return current


# ---------------- delegated admin (M1.4, ROLE-1) ----------------
#
# A principal-only action that a teacher may be granted. The principal
# always passes; a teacher passes with an active grant of that capability
# (operations/grants.py), whose scope some routes check further. The grant
# store lives in the operations package, which imports this module, so it
# plugs its check in at start-up (set_grant_checker) instead of importing.

ADMIN_CAPABILITIES = ("users", "calendar", "timetable", "leave", "exams", "qbank_review", "reports")
ADMIN_DETAIL = "this action needs the principal, or a teacher the principal has made an admin for it"
_grant_checker = None
_ADMIN_DEPS: dict = {}


_scope_reader = None
SCOPE_DETAIL = ("your admin grant covers only some classes, sections or subjects, "
                "and this action is outside them")


def set_grant_checker(fn, scopes=None) -> None:
    """fn(user_id, capability, **target) -> bool; scopes(user_id, capability)
    -> the scopes of the user's active grants for it."""
    global _grant_checker, _scope_reader
    _grant_checker = fn
    _scope_reader = scopes


def holds(user: User, capability: str, **target) -> bool:
    """Whether `user` may administer `capability` for `target` (the grade,
    section and subject the action touches). With no target, only the
    principal and an unlimited grant hold it (grants.scope_admits)."""
    if user.role == "principal":
        return True
    if user.role != "teacher" or _grant_checker is None:
        return False
    return bool(_grant_checker(user.id, capability, **target))


def holds_any(user: User, capability: str) -> bool:
    """Whether `user` holds the capability for at least part of the school."""
    if user.role == "principal":
        return True
    if user.role != "teacher" or _scope_reader is None:
        return False
    return bool(_scope_reader(user.id, capability))


def require_in_scope(user: User, capability: str, *, grade: Optional[int] = None,
                     section_id: Optional[str] = None, subject_id: Optional[str] = None) -> None:
    """For a route taken with require_admin(capability, scoped=True): the
    action's own target must be inside one of the caller's grants."""
    if not holds(user, capability, grade=grade, section_id=section_id, subject_id=subject_id):
        raise HTTPException(403, SCOPE_DETAIL)


def require_admin(capability: str, *, scoped: bool = False):
    """The dependency for a delegable principal route. One function object
    per (capability, scoped), so tests/test_role_gates.py can read the
    capability off a route's dependency chain.

    Unscoped (the default), the route acts on the whole school, and a grant
    limited to some classes does not reach it. `scoped=True` admits any
    holder of the capability, and the route must then check the section or
    class it acts on with require_in_scope."""
    if capability not in ADMIN_CAPABILITIES:
        raise ValueError(f"unknown admin capability {capability!r}")
    dep = _ADMIN_DEPS.get((capability, scoped))
    if dep is None:
        def dep(current: User = Depends(get_current_user)) -> User:
            if scoped and not holds_any(current, capability):
                raise HTTPException(403, ADMIN_DETAIL)
            if not scoped and not holds(current, capability):
                raise HTTPException(403, SCOPE_DETAIL if holds_any(current, capability) else ADMIN_DETAIL)
            return current
        dep.capability = capability
        dep.scoped = scoped
        dep.__name__ = f"require_admin_{capability}" + ("_scoped" if scoped else "")
        _ADMIN_DEPS[(capability, scoped)] = dep
    return dep


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
                      principal: User = Depends(require_admin("users"))) -> list[UserResponse]:
    """The real roster an admin UI needs to assign a teacher to a book or
    enroll a student in a class -- without a name-searchable list, those
    actions would require already knowing a raw user id. Principal-gated
    and scoped to the caller's own school_id (never a query parameter),
    same posture as every other per-school administrative read in this
    codebase."""
    if role is not None and role not in ("teacher", "principal", "student", "parent", "left"):
        raise HTTPException(400, f"invalid role filter: {role!r}")
    if role == "left":
        # The accounts the school has closed, each with the role it had.
        users = [u for u in _require().users_for_school(principal.school_id) if is_closed(u.role)]
    else:
        users = [u for u in _require().users_for_school(principal.school_id, role=role)
                 if not is_closed(u.role)]
    return [_to_response(u) for u in users]


def _school_account(user_id: str, admin: User) -> User:
    user = _require().get(user_id)
    if user is None or user.school_id != admin.school_id:
        raise HTTPException(404, "no such account in your school")
    return user


def _audit_account(action: str, admin: User, user: User, before: str) -> None:
    from .audit_log import get_audit_log
    if _cfg is not None:
        get_audit_log(_cfg.data_root).append(
            action, actor=admin.id,
            details={"userId": user.id, "name": user.name, "before": before, "after": user.role})


@router.post("/users/{user_id}/close", response_model=UserResponse)
def close_account(user_id: str, admin: User = Depends(require_admin("users"))) -> UserResponse:
    """A person has left the school: their account can no longer sign in,
    every session they hold ends, and their admin grants are revoked. Their
    papers, marks and history stay, attributed to them (N-67-8)."""
    user = _school_account(user_id, admin)
    if user.id == admin.id:
        raise HTTPException(422, "you cannot close your own account")
    if user.role == "principal":
        raise HTTPException(403, "the principal's account is changed by the operator, not in the app")
    before = user.role
    closed = _require().close_account(user.id)
    try:
        from ..operations import routes as ops
        for g in ops.store().active_grants(user.id):
            ops.store().revoke_grant(g["id"], revoked_by=admin.id)
    except Exception:  # noqa: BLE001 - a closed account holds no role a grant could apply to
        logging.getLogger(__name__).warning("could not revoke %s's grants on closing", user.id, exc_info=True)
    if before != closed.role:
        _audit_account("account_closed", admin, closed, before)
    return _to_response(closed)


@router.post("/users/{user_id}/reopen", response_model=UserResponse)
def reopen_account(user_id: str, admin: User = Depends(require_admin("users"))) -> UserResponse:
    """Undo a close: the account gets back the role it had. Grants are not
    restored; the principal gives them again if they are still wanted."""
    user = _school_account(user_id, admin)
    before = user.role
    reopened = _require().reopen_account(user.id)
    if before != reopened.role:
        _audit_account("account_reopened", admin, reopened, before)
    return _to_response(reopened)


# ---------------- passwords ----------------
#
# There was no way back in for a teacher, student or parent who forgot their
# password: no reset, no change, and the emailed code is off where email is
# not configured (QA S-05, 2026-10-01). The principal (or a teacher holding
# the "users" grant) resets a school account to a one-time password they
# read out; anyone signed in changes their own. The principal's own account
# is the operator's, as for closing it.

# Unambiguous when read aloud or copied from a screen: no 0/O, 1/l/I.
_TEMP_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


class PasswordResetResponse(Camel):
    user_id: str
    name: str
    temporary_password: str


class PasswordChangeRequest(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")
    current_password: str
    new_password: str


def _temporary_password() -> str:
    """Three groups of four, e.g. "k7mq-x2rw-9tpd": 60 bits, easy to dictate."""
    return "-".join("".join(secrets.choice(_TEMP_ALPHABET) for _ in range(4)) for _ in range(3))


@router.post("/users/{user_id}/password-reset", response_model=PasswordResetResponse)
def reset_password(user_id: str, admin: User = Depends(require_admin("users"))) -> PasswordResetResponse:
    """Set a one-time password for a school account and sign it out
    everywhere. The response is the only place the password appears; it is
    not stored in the clear or written to the audit log."""
    user = _school_account(user_id, admin)
    if user.id == admin.id:
        raise HTTPException(403, "change your own password under Settings, not here")
    if user.role == "principal":
        raise HTTPException(403, "the principal's account is changed by the operator, not in the app")
    # A delegated admin (a teacher with the users grant) resets students and
    # parents only: a staff member's temporary password would let them sign
    # in as that colleague, with the colleague's grants (review 2026-10-04).
    if admin.role != "principal" and user.role not in ("student", "parent"):
        raise HTTPException(403, "only the principal resets a staff member's password")
    if is_closed(user.role):
        raise HTTPException(409, "this account is closed; reopen it first")
    password = _temporary_password()
    store = _require()
    store.set_password(user.id, password)
    store.delete_sessions_for(user.id)
    if _cfg is not None:
        from .audit_log import get_audit_log
        get_audit_log(_cfg.data_root).append(
            "password_reset", actor=admin.id, details={"userId": user.id, "schoolId": admin.school_id})
    return PasswordResetResponse(user_id=user.id, name=user.name, temporary_password=password)


@router.post("/password")
def change_password(req: PasswordChangeRequest, current: User = Depends(get_current_user)) -> dict:
    """The signed-in person's own password. Other sessions end; this one
    stays signed in."""
    store = _require()
    # The current password guards a stolen session from taking the account:
    # wrong guesses are counted like sign-in's, per account.
    account = f"password-change-failures:{current.id}"
    wait = account_failures.blocked(account)
    if wait is not None:
        raise HTTPException(429, f"too many wrong passwords; try again in {max(1, round(wait / 60))} minute(s)",
                            headers={"Retry-After": str(wait)})
    if not store.check_password(current.id, req.current_password):
        account_failures.record(account)
        raise HTTPException(400, "the current password is not right")
    if len(req.new_password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    store.set_password(current.id, req.new_password)
    store.delete_sessions_for(current.id)
    if _cfg is not None:
        from .audit_log import get_audit_log
        get_audit_log(_cfg.data_root).append(
            "password_changed", actor=current.id, details={"schoolId": current.school_id})
    return {"ok": True, "token": store.create_session(current.id)}


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
                  principal: User = Depends(require_admin("users"))) -> InviteResponse:
    """A single-use invite to the CALLER's school. There is no school field
    to send: a principal can only ever invite people into their own school.
    A parent is invited for one student instead (POST
    /students/{id}/guardian-invites), so every parent account has a child."""
    if req.role == "parent":
        raise HTTPException(400, "invite a parent from their child's record: "
                                 "POST /api/v1/students/{studentId}/guardian-invites")
    try:
        invite = _require().create_invite(
            school_id=principal.school_id, role=req.role, created_by=principal.id,
            email=req.email, expires_in_days=req.expires_in_days)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _invite_response(invite)


@router.get("/invites", response_model=list[InviteResponse])
def list_invites(principal: User = Depends(require_admin("users"))) -> list[InviteResponse]:
    """The caller's school's invites, used and unused, newest first. Each
    row says who created it and which account used it and when."""
    return [_invite_response(i) for i in _require().invites_for_school(principal.school_id)]


@router.delete("/invites/{code}", response_model=InviteResponse)
def revoke_invite(code: str, principal: User = Depends(require_admin("users"))) -> InviteResponse:
    """Withdraw an unused invite. Another school's code is a 404, the same as
    an unknown one."""
    try:
        invite = _require().revoke_invite(code=code, school_id=principal.school_id)
    except InviteNotFound:
        raise HTTPException(404, "no such invite at your school")
    except InviteAlreadyUsed:
        raise HTTPException(409, "this invite has already been used and cannot be withdrawn")
    return _invite_response(invite)
