"""Error tracking (REQUIREMENTS NFR-7: "error tracking, structured logs, ...
alerting").

Until now an unhandled exception became a bare 500 and a line in the
container log, and an error in the web or phone app was never seen by anyone
but the person holding the device. Nothing grouped them, so no one could say
whether a failure was new, how often it happened, or since when.

Two sources, one shape:

- **The server.** An exception no handler turned into an answer is recorded
  and written to the log as a structured Cloud Error Reporting event
  (`@type` ReportedErrorEvent, with the service, its release, the route and
  the stack). On Cloud Run, Error Reporting groups those lines by itself and
  can alert on a new group -- the durable record and the alerting live there
  (docs/incident-response-runbook.md), with no new vendor and no new secret.
- **The apps.** The web and phone apps post what Flutter caught to
  `POST /api/v1/client-errors`; it is recorded and logged the same way under
  the app's own service name.

Each error is grouped by a fingerprint (the exception type and where it was
raised, or the app's message and top frame), and this process keeps the
counts, first and last seen, and one scrubbed sample per group, so an
operator can read the recent picture from `GET /api/v1/ops/errors` (behind
`ACOS_OPS_KEY`, like the cron route) and `/health/errors` gives the counts.

Nothing personal is kept: no request body, no query string, the route's
template instead of its path (so no ids), no user id, and email addresses
and long digit runs are masked in messages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .rate_limit import get_client_ip

logger = logging.getLogger(__name__)

SERVICE = "academicos-backend"
MAX_GROUPS = 200                 # distinct fingerprints kept in memory
CLIENT_PER_MINUTE = 30           # client reports accepted per source address per minute
# Bounded repeats: an unbounded `[\w.+-]+@` backtracks quadratically on a long
# run of word characters with no "@" -- 20,000 of them took 1.1 s, holding the
# GIL, from an unauthenticated POST (review of #50).
_EMAIL = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63})+")
_DIGITS = re.compile(r"\d{6,}")
_MARGIN = 256            # an address straddling the cut is still masked whole


def scrub(text: str, limit: int) -> str:
    """Mask what could name a person: email addresses and long digit runs
    (phone numbers, roll numbers, ids), and cut to `limit` characters. Only
    `limit` plus a margin is ever scanned, whatever length arrives."""
    text = (text or "")[:limit + _MARGIN]
    text = _EMAIL.sub("[email]", text)
    text = _DIGITS.sub("[number]", text)
    return text[:limit]


def _release() -> str:
    """The build, as /health reports it (config.build_identity)."""
    from ..config import build_identity
    return build_identity()["commit"][:40]


@dataclass
class ErrorGroup:
    fingerprint: str
    source: str                   # server | web | android | ios
    kind: str                     # the exception type, or "client"
    route: str
    message: str
    count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    release: str = ""
    schools: set[str] = field(default_factory=set)


class ErrorLog:
    """This process's grouped errors, newest last. Bounded: the least
    recently seen group goes when a new one would exceed MAX_GROUPS."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: "OrderedDict[str, ErrorGroup]" = OrderedDict()
        self.server_total = 0
        self.client_total = 0
        self.dropped_client = 0
        self._window: dict[str, list[float]] = {}

    def record(self, *, fingerprint: str, source: str, kind: str, route: str, message: str,
               school_id: Optional[str] = None) -> ErrorGroup:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            g = self._groups.pop(fingerprint, None)
            if g is None:
                g = ErrorGroup(fingerprint=fingerprint, source=source, kind=kind, route=route,
                               message=message, first_seen=now)
            g.count += 1
            g.last_seen = now
            g.release = _release()
            g.message = message
            if school_id:
                g.schools.add(school_id)
            self._groups[fingerprint] = g
            while len(self._groups) > MAX_GROUPS:
                self._groups.popitem(last=False)
            if source == "server":
                self.server_total += 1
            else:
                self.client_total += 1
            return g

    def allow_client(self, address: str) -> bool:
        """A simple per-address budget, so a crash loop on one device cannot
        flood the log."""
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._window.get(address, []) if now - t < 60]
            if len(recent) >= CLIENT_PER_MINUTE:
                self._window[address] = recent
                self.dropped_client += 1
                return False
            recent.append(now)
            self._window[address] = recent
            if len(self._window) > 5000:          # forget idle addresses
                self._window = {a: ts for a, ts in self._window.items() if ts and now - ts[-1] < 60}
            return True

    def groups(self) -> list[ErrorGroup]:
        with self._lock:
            return sorted(self._groups.values(), key=lambda g: g.last_seen, reverse=True)

    def reset(self) -> None:
        with self._lock:
            self._groups.clear()
            self.server_total = self.client_total = self.dropped_client = 0
            self._window.clear()


ERRORS = ErrorLog()


def _emit(event: dict[str, Any]) -> None:
    """One JSON line on stderr: Cloud Run sends it to Cloud Logging, where
    Error Reporting picks up ReportedErrorEvent lines. Never raises."""
    try:
        sys.stderr.write(json.dumps(event, default=str) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - reporting an error must not become one
        logger.exception("could not write the error event")


def _fingerprint(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:12]


def _raised_at(exc: BaseException) -> str:
    """Where in our own code the exception was raised (file:function), so two
    errors of one type from different places are two groups."""
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if "academicos" in f.filename.replace("\\", "/")]
    frame = (ours or frames or [None])[-1]
    if frame is None:
        return "unknown"
    name = frame.filename.replace("\\", "/").split("academicos/")[-1]
    return f"{name}:{frame.name}"


def record_server_error(request: Request, exc: BaseException) -> ErrorGroup:
    route = getattr(request.scope.get("route"), "path", None) or "unmatched"
    where = _raised_at(exc)
    kind = type(exc).__name__
    message = scrub(str(exc), 300)
    school = getattr(getattr(request.state, "user", None), "school_id", None)
    group = ERRORS.record(fingerprint=_fingerprint("server", kind, where), source="server", kind=kind,
                          route=f"{request.method} {route}", message=message, school_id=school)
    stack = scrub("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), 8000)
    _emit({
        "severity": "ERROR",
        "@type": "type.googleapis.com/google.devtools.clouderrorreporting.v1beta1.ReportedErrorEvent",
        "message": stack,
        "serviceContext": {"service": SERVICE, "version": _release()},
        "context": {"httpRequest": {"method": request.method, "url": route, "responseStatusCode": 500},
                    "reportLocation": {"filePath": where.split(":")[0], "functionName": where.split(":")[-1]}},
        "fingerprint": group.fingerprint,
    })
    return group


async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """The last handler: a 500 that says which report it was, never the
    exception's text (which may carry data the caller should not see)."""
    group = record_server_error(request, exc)
    return JSONResponse(status_code=500, content={
        "detail": "Something went wrong on our side. It has been recorded; try again, and if it keeps "
                  "happening tell the school's administrator this reference.",
        "errorRef": group.fingerprint,
    })


# ---------------------------------------------------------------- routes

router = APIRouter()


class _Req(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ClientErrorRequest(_Req):
    source: str = Field(pattern="^(web|android|ios)$")
    message: str = Field(min_length=1, max_length=2000)
    stack: Optional[str] = Field(default=None, max_length=20000)
    route: Optional[str] = Field(default=None, max_length=200)
    app_version: Optional[str] = Field(default=None, max_length=60)


class ClientErrorResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    recorded: bool
    error_ref: Optional[str] = None


def _optional_user(authorization: str = Header(default="")):
    from ..assessment.auth_routes import get_current_user_optional
    try:
        return get_current_user_optional(authorization)
    except HTTPException:
        return None


@router.post("/api/v1/client-errors", response_model=ClientErrorResponse, response_model_by_alias=True)
def report_client_error(req: ClientErrorRequest, request: Request,
                        current=Depends(_optional_user)) -> ClientErrorResponse:
    """What the web or phone app caught. Open to a signed-out app too (the
    sign-in page can fail), limited per address, and kept scrubbed: no user
    id, only the school."""
    # The caller, not Cloud Run's front end: request.client is the proxy, so
    # every user shared one bucket (review of #50). get_client_ip takes the
    # hop the platform appends, which a client cannot forge.
    address = get_client_ip(request)
    if not ERRORS.allow_client(address):
        return ClientErrorResponse(recorded=False)
    message = scrub(req.message, 500)
    stack = scrub(req.stack or "", 4000)
    top = next((line.strip() for line in stack.splitlines() if line.strip()), "")
    route = scrub(req.route or "", 200)
    group = ERRORS.record(fingerprint=_fingerprint(req.source, message[:120], top[:160]), source=req.source,
                          kind="client", route=route, message=message,
                          school_id=getattr(current, "school_id", None))
    _emit({
        "severity": "ERROR",
        "@type": "type.googleapis.com/google.devtools.clouderrorreporting.v1beta1.ReportedErrorEvent",
        "message": f"{message}\n{stack}".strip(),
        "serviceContext": {"service": f"academicos-{req.source}", "version": scrub(req.app_version or "unknown", 60)},
        "context": {"httpRequest": {"url": route}},
        "fingerprint": group.fingerprint,
    })
    return ClientErrorResponse(recorded=True, error_ref=group.fingerprint)


class ErrorGroupResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    fingerprint: str
    source: str
    kind: str
    route: str
    message: str
    count: int
    first_seen: str
    last_seen: str
    release: str
    schools: int


@router.get("/api/v1/ops/errors", response_model=list[ErrorGroupResponse], response_model_by_alias=True)
def list_errors(x_ops_key: Optional[str] = Header(default=None, alias="X-Ops-Key")) -> list[ErrorGroupResponse]:
    """This process's error groups, most recent first, for the operator.
    Needs the operator's key (env ACOS_OPS_KEY); 404 when none is
    configured, so the route does not exist until someone sets one."""
    import hmac
    expected = os.environ.get("ACOS_OPS_KEY", "")
    if not expected:
        raise HTTPException(404, "Not Found")
    if not x_ops_key or not hmac.compare_digest(x_ops_key, expected):
        raise HTTPException(401, "a valid X-Ops-Key is required")
    return [ErrorGroupResponse(fingerprint=g.fingerprint, source=g.source, kind=g.kind, route=g.route,
                               message=g.message, count=g.count, first_seen=g.first_seen,
                               last_seen=g.last_seen, release=g.release, schools=len(g.schools))
            for g in ERRORS.groups()]


@router.get("/health/errors")
def error_counts() -> dict[str, Any]:
    """Counts only -- never a message -- so it can be public like /health."""
    return {"serverErrors": ERRORS.server_total, "clientErrors": ERRORS.client_total,
            "clientReportsDropped": ERRORS.dropped_client, "groups": len(ERRORS.groups()),
            "release": _release()}


def install(app) -> None:
    """Called once from api/main.py."""
    app.add_exception_handler(Exception, _on_unhandled)
    app.include_router(router)
