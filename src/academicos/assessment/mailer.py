"""Email delivery for generated question papers and reports.

Two backends behind one interface:

* **Composio** — routes through the school's connected Gmail/Outlook account, so
  mail comes *from the school*, with delivery handled by a provider the school
  already trusts. Preferred when configured.
* **SMTP** — a direct fallback so the feature works without a third-party
  dependency (schools often have their own mail server).

Credentials are read from the environment, never from `config/config.toml`:
that file is committed to git and already leaks one API key. Put secrets in
`config/secrets.env` (gitignored) or the real environment.

Nothing here sends mail as a side effect of building a paper — delivery is an
explicit, separately authorised action.
"""
from __future__ import annotations

import base64
import time
import logging
import mimetypes
import os
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

COMPOSIO_BASE = "https://backend.composio.dev/api/v3"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MAX_ATTACHMENT_MB = 20


class MailError(RuntimeError):
    pass


@dataclass
class MailResult:
    ok: bool
    backend: str
    detail: str
    recipients: list[str] = field(default_factory=list)
    message_id: Optional[str] = None


@dataclass
class MailMessage:
    to: list[str]
    subject: str
    body: str
    attachments: list[Path] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.to:
            raise MailError("no recipients")
        for addr in self.to + self.cc:
            if not _EMAIL_RE.match(addr):
                raise MailError(f"not a valid email address: {addr!r}")
        if not self.subject.strip():
            raise MailError("subject is empty")
        total = 0
        for p in self.attachments:
            if not p.exists():
                raise MailError(f"attachment not found: {p}")
            total += p.stat().st_size
        if total > MAX_ATTACHMENT_MB * 1024 * 1024:
            raise MailError(
                f"attachments total {total / 1e6:.1f} MB, over the "
                f"{MAX_ATTACHMENT_MB} MB limit")


# ---------------------------------------------------------------- Composio

def composio_key() -> str:
    return os.environ.get("COMPOSIO_API_KEY", "").strip()


def _composio_request(method: str, path: str, **kw) -> Any:
    key = composio_key()
    if not key:
        raise MailError("COMPOSIO_API_KEY is not set")
    r = requests.request(method, f"{COMPOSIO_BASE}{path}",
                         headers={"x-api-key": key, "Content-Type": "application/json"},
                         timeout=30, **kw)
    if r.status_code == 401:
        raise MailError(
            "Composio rejected the API key. Check it is the full key from "
            "app.composio.dev (they are considerably longer than 23 characters).")
    if r.status_code >= 400:
        raise MailError(f"Composio {method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {}


# How long one answer about Composio is trusted. The public sign-in page asks
# (`/auth/providers` -> `emailCode`) on every load, and each answer is a call
# to Composio.
_STATUS_TTL_SECONDS = 300
_status_cache: dict[str, Any] = {"at": 0.0, "key": "", "value": None}


def _gmail_account() -> Optional[tuple[str, str]]:
    """(connected account id, user id) of an ACTIVE Gmail account connected
    in Composio, or None. Composio runs a tool only with both: the account
    id alone is refused with "User ID is required with connected account"
    (2026-10-04, the owner's first Gmail connection).
    Mail goes through Composio's GMAIL_SEND_EMAIL, so a key with only Google
    Calendar connected sends nothing (2026-10-04: the live key had five
    calendar accounts and no Gmail, and the sign-in page offered emailed codes
    that could never arrive)."""
    data = _composio_request("GET", "/connected_accounts?toolkit_slugs=gmail&statuses=ACTIVE&limit=10")
    for item in data.get("items") or []:
        toolkit = item.get("toolkit") or {}
        slug = toolkit.get("slug") if isinstance(toolkit, dict) else toolkit
        if str(slug).lower() == "gmail" and str(item.get("status", "")).upper() == "ACTIVE" and item.get("id"):
            return item["id"], str(item.get("user_id") or "")
    return None


def composio_status() -> dict:
    """Can Composio send mail right now? Only with its key AND a Gmail
    account connected. Surfaced in the UI instead of failing at send, and
    cached for `_STATUS_TTL_SECONDS`."""
    key = composio_key()
    if not key:
        return {"configured": False, "reason": "COMPOSIO_API_KEY not set"}
    now = time.monotonic()
    cached = _status_cache["value"]
    if cached is not None and _status_cache["key"] == key and now - _status_cache["at"] < _STATUS_TTL_SECONDS:
        return dict(cached)
    try:
        account = _gmail_account()
        value = ({"configured": True, "reason": "ok", "gmailAccountId": account[0], "gmailUserId": account[1]}
                 if account else
                 {"configured": False,
                  "reason": "Composio has no Gmail account connected; connect Gmail at app.composio.dev "
                            "(Google Calendar alone cannot send mail)"})
    except MailError as e:
        value = {"configured": False, "reason": str(e)}
    except requests.RequestException as e:
        value = {"configured": False, "reason": f"Composio could not be reached: {e}"}
    _status_cache.update(at=now, key=key, value=value)
    return dict(value)


def send_via_composio(msg: MailMessage, *, connected_account_id: str | None = None,
                      tool_slug: str = "GMAIL_SEND_EMAIL") -> MailResult:
    msg.validate()
    attachments = []
    for p in msg.attachments:
        mime, _ = mimetypes.guess_type(p.name)
        attachments.append({
            "name": p.name,
            "mimetype": mime or "application/octet-stream",
            "content": base64.b64encode(p.read_bytes()).decode("ascii"),
        })
    payload: dict[str, Any] = {
        "arguments": {
            "recipient_email": msg.to[0],
            "subject": msg.subject,
            "body": msg.body,
            "is_html": False,
        }
    }
    if len(msg.to) > 1:
        payload["arguments"]["extra_recipients"] = msg.to[1:]
    if msg.cc:
        payload["arguments"]["cc"] = msg.cc
    if attachments:
        payload["arguments"]["attachment"] = attachments[0]
    # Composio runs a tool for a connected account: the caller's, else the
    # Gmail account composio_status found.
    if not connected_account_id:
        status = composio_status()
        connected_account_id = status.get("gmailAccountId")
        if status.get("gmailUserId"):
            payload["user_id"] = status["gmailUserId"]
    if connected_account_id:
        payload["connected_account_id"] = connected_account_id

    data = _composio_request("POST", f"/tools/execute/{tool_slug}", json=payload)
    ok = bool(data.get("successful", data.get("successfull", True)))
    return MailResult(
        ok=ok, backend="composio",
        detail=str(data.get("error") or "sent"),
        recipients=msg.to,
        message_id=(data.get("data") or {}).get("id") if isinstance(data.get("data"), dict) else None,
    )


# -------------------------------------------------------------------- SMTP

def smtp_settings() -> dict[str, str]:
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "port": os.environ.get("SMTP_PORT", "587").strip(),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "from": os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "")).strip(),
    }


def smtp_status() -> dict:
    s = smtp_settings()
    missing = [k for k in ("host", "user", "password") if not s[k]]
    if missing:
        return {"configured": False, "reason": f"missing {', '.join('SMTP_' + m.upper() for m in missing)}"}
    return {"configured": True, "reason": "ok", "host": s["host"], "from": s["from"]}


def send_via_smtp(msg: MailMessage) -> MailResult:
    msg.validate()
    s = smtp_settings()
    if not s["host"]:
        raise MailError("SMTP_HOST is not set")

    email = EmailMessage()
    email["From"] = s["from"] or s["user"]
    email["To"] = ", ".join(msg.to)
    if msg.cc:
        email["Cc"] = ", ".join(msg.cc)
    email["Subject"] = msg.subject
    email.set_content(msg.body)
    for p in msg.attachments:
        mime, _ = mimetypes.guess_type(p.name)
        maintype, _, subtype = (mime or "application/octet-stream").partition("/")
        email.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype,
                             filename=p.name)

    port = int(s["port"] or 587)
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(s["host"], port, context=context, timeout=30) as srv:
            srv.login(s["user"], s["password"])
            srv.send_message(email)
    else:
        with smtplib.SMTP(s["host"], port, timeout=30) as srv:
            srv.starttls(context=context)
            srv.login(s["user"], s["password"])
            srv.send_message(email)
    return MailResult(ok=True, backend="smtp", detail="sent",
                      recipients=msg.to + msg.cc)


# ----------------------------------------------------------------- facade

def available_backends() -> dict[str, dict]:
    return {"composio": composio_status(), "smtp": smtp_status()}


def send(msg: MailMessage, *, prefer: str = "auto",
         connected_account_id: str | None = None) -> MailResult:
    """Send via the preferred backend, falling back when it is unavailable."""
    msg.validate()
    backends = available_backends()

    order: list[str]
    if prefer == "composio":
        order = ["composio"]
    elif prefer == "smtp":
        order = ["smtp"]
    else:
        order = [b for b in ("composio", "smtp") if backends[b]["configured"]]

    if not order:
        reasons = "; ".join(f"{k}: {v['reason']}" for k, v in backends.items())
        raise MailError(f"no email backend is configured ({reasons})")

    last: Optional[Exception] = None
    for backend in order:
        try:
            if backend == "composio":
                return send_via_composio(msg, connected_account_id=connected_account_id)
            return send_via_smtp(msg)
        except Exception as e:                      # try the next backend
            log.warning("%s send failed: %s", backend, e)
            last = e
    raise MailError(str(last))


def paper_email(paper_title: str, school: str, subject_name: str, grade: int,
                pdf: Path, answer_key: Path | None, recipients: list[str],
                note: str = "") -> MailMessage:
    """Compose the standard 'here is your question paper' email."""
    attachments = [pdf] + ([answer_key] if answer_key and answer_key.exists() else [])
    body = (
        f"Dear Colleague,\n\n"
        f"Please find attached the question paper for {subject_name}, Class {grade}"
        f" — {paper_title}.\n\n"
        f"{'Answer key and marking scheme are attached as well.' if answer_key else ''}\n"
        f"{note}\n\n"
        f"This paper was generated by AcademicOS from the CBSE question bank and "
        f"is pending teacher review before printing.\n\n"
        f"Regards,\n{school}\n"
    )
    return MailMessage(
        to=recipients,
        subject=f"{subject_name} Class {grade} — {paper_title}",
        body=body,
        attachments=attachments,
    )
