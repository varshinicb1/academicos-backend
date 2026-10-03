"""Webhooks for partners (API-6): "new questions in a chapter", pushed.

A partner holding a question-bank API key asks its school's principal to
register an HTTPS URL for that key (the web console's API keys page). When
the questions the key may read grow -- a bank load or reload that serves new
answer-keyed questions, or the school's reviewer taking back a rejection --
each such chapter becomes one `questions.added` event, POSTed to the URL as
JSON and signed.

What a delivery carries, and what it never can:

  * question ids, with the class, subject and chapter they were filed under,
    and why they were added. Nothing else: no question text (the partner
    fetches that through `/v1` with its key, which re-checks its scope), and
    no student, teacher or school record of any kind. `_EVENT_DATA_KEYS` is
    the whole vocabulary, and `_event` refuses anything outside it.
  * only what the key may read: its classes and subjects (API-3), the
    `questions:read` scope, and not a question its school's reviewer
    rejected. Checked when the event is made and again before each send.

Signing. Each webhook has its own secret (`whsec_...`), shown once when it
is registered. Every request carries

    X-AcademicOS-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256>

over `<t>.<raw body>`, the scheme Stripe uses, so the receiver can refuse a
forged body and a replayed old one. The secret has to be kept in a form the
server can sign with, so it is stored as it is (in this file and its
snapshot), unlike an API key, which is stored as a hash.

Where the request may go (SSRF). Only `https://`, no user:password in the
URL, and only to public addresses: a host that is, or resolves to, a
private, loopback, link-local, carrier-grade NAT, multicast, reserved or
cloud-metadata address is refused. The check runs at registration and again
at every send, and the send connects to the address it just checked (with
TLS verified against the URL's own host name), so a name that is re-pointed
at an internal address after registration (DNS rebinding) reaches nothing.
Redirects are not followed. Each send has a 5 s connect and 10 s read
timeout.

Delivery. A send is retried with backoff (1 min, 5 min, 30 min, 2 h, 6 h)
until a 2xx, six attempts in all, then the delivery is `failed`. Every
attempt is recorded (status, attempts, last HTTP status or error), and the
principal reads the log on the web. A webhook or key revoked before a
delivery goes out cancels it: nothing is sent for a revoked key.

Sends happen off the request that caused them, in a background thread, and
again on the operator's automations run (`POST /api/v1/automations/run`) and
the throttled run ordinary traffic triggers (operations/routes.py), which is
what picks up retries. The sender and the DNS resolver are parameters of
`WebhookService`, so tests drive the whole path with no network.

Storage: one SQLite file (WAL, 60 s busy timeout, one connection behind one
lock), snapshotted to the blob store like the API key store it sits beside.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import secrets
import socket
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit

from .api_keys import ApiKey, ApiKeyStore
from .qbank_engine import has_answer_key

log = logging.getLogger(__name__)

EVENTS: frozenset[str] = frozenset({"questions.added"})
SECRET_PREFIX = "whsec_"
SIGNATURE_HEADER = "X-AcademicOS-Signature"
EVENT_HEADER = "X-AcademicOS-Event"
DELIVERY_HEADER = "X-AcademicOS-Delivery"
# Seconds to wait after attempt N fails (N = 1..5); attempt 6 is the last.
BACKOFF_SECONDS: tuple[int, ...] = (60, 300, 1800, 7200, 21600)
MAX_ATTEMPTS = len(BACKOFF_SECONDS) + 1
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
# One event names at most this many ids; a bigger chapter addition is split.
MAX_IDS_PER_EVENT = 500
# A receiver should refuse a signature older than this (the docs' snippets do).
SIGNATURE_TOLERANCE_SECONDS = 300
MAX_WEBHOOKS_PER_KEY = 5

# The whole vocabulary of an event's `data`. Nothing about a person fits it.
_EVENT_DATA_KEYS = frozenset({"subject", "grade", "chapterId", "questionIds", "count", "reason"})

_BLOCKED_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal", "metadata.goog",
                            "instance-data", "instance-data.ec2.internal"})
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_METADATA = frozenset({ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("fd00:ec2::254"),
                       ipaddress.ip_address("100.100.100.200")})

Resolver = Callable[[str, int], list[str]]
# sender(url, address, body, headers, (connect, read) timeout) -> HTTP status
Sender = Callable[[str, str, bytes, dict[str, str], tuple[float, float]], int]


class UnsafeWebhookUrl(ValueError):
    """The URL may not be called: not HTTPS, or not a public address."""


class WebhookError(ValueError):
    """A registration the API refuses (the key, the events, the count)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# where a webhook may point
# --------------------------------------------------------------------------- #

def is_public_address(address: str) -> bool:
    """True only for a globally routable unicast address.

    An IPv6 address that carries an IPv4 one (mapped, 6to4, Teredo, NAT64) is
    judged by the IPv4 address it reaches, so `::ffff:127.0.0.1` is loopback.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        inner = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if inner is None and ip in _NAT64:
            inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if inner is not None:
            return is_public_address(str(inner))
    if ip in _METADATA:
        return False
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified or (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT)):
        return False
    return ip.is_global


def system_resolver(host: str, port: int) -> list[str]:
    """Every address `host` resolves to, IPv4 and IPv6."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


def check_url(url: str, resolve: Resolver = system_resolver) -> tuple[str, int, list[str]]:
    """(host, port, addresses) for a URL a webhook may call; UnsafeWebhookUrl
    otherwise. Every address the name resolves to must be public: a name with
    one public and one private address is refused, because the connection
    could be made to either."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port or 443
    except ValueError as exc:
        raise UnsafeWebhookUrl(f"not a URL: {exc}") from exc
    if parts.scheme != "https":
        raise UnsafeWebhookUrl("a webhook URL must start with https://")
    if parts.username or parts.password:
        raise UnsafeWebhookUrl("a webhook URL must not carry a user name or password")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeWebhookUrl("a webhook URL needs a host")
    if host in _BLOCKED_HOSTS or host.endswith(".localhost") or host.endswith(".internal"):
        raise UnsafeWebhookUrl(f"{host} is an internal name; a webhook must reach a public address")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            addresses = resolve(host, port)
        except (OSError, UnicodeError) as exc:
            raise UnsafeWebhookUrl(f"{host} does not resolve ({exc})") from exc
        if not addresses:
            raise UnsafeWebhookUrl(f"{host} does not resolve")
    bad = [a for a in addresses if not is_public_address(a)]
    if bad:
        raise UnsafeWebhookUrl(
            f"{host} is or resolves to {bad[0]}, a private, loopback, link-local or metadata address; "
            "a webhook must reach a public address")
    return host, port, addresses


# --------------------------------------------------------------------------- #
# signing
# --------------------------------------------------------------------------- #

def sign(secret: str, timestamp: int, body: bytes) -> str:
    """Hex HMAC-SHA256 of `<timestamp>.<body>` under `secret`."""
    return hmac.new(secret.encode("utf-8"), str(timestamp).encode("ascii") + b"." + body,
                    hashlib.sha256).hexdigest()


def signature_header(secret: str, body: bytes, timestamp: int) -> str:
    return f"t={timestamp},v1={sign(secret, timestamp, body)}"


def verify_signature(secret: str, header: str, body: bytes, *, now: Optional[int] = None,
                     tolerance: int = SIGNATURE_TOLERANCE_SECONDS) -> bool:
    """What a receiver does (docs/question-bank-api.md has the same in Python
    and JavaScript): parse `t` and `v1`, refuse a stale `t`, compare in
    constant time."""
    fields = dict(part.split("=", 1) for part in header.split(",") if "=" in part)
    try:
        ts = int(fields["t"])
    except (KeyError, ValueError):
        return False
    if abs((now if now is not None else int(time.time())) - ts) > tolerance:
        return False
    return hmac.compare_digest(sign(secret, ts, body).encode("ascii"),
                               fields.get("v1", "").encode("utf-8", "replace"))


def https_sender(url: str, address: str, body: bytes, headers: dict[str, str],
                 timeout: tuple[float, float]) -> int:
    """POST `body` to `url`, connecting to `address` (already checked public)
    and verifying TLS against the URL's own host. No redirects, no retries
    here (the service retries), nothing of the answer read but its status."""
    import urllib3
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or 443
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    try:
        import certifi
        ca_certs: Optional[str] = certifi.where()
    except ImportError:  # pragma: no cover - requests depends on certifi
        ca_certs = None
    pool = urllib3.HTTPSConnectionPool(
        host=address, port=port, server_hostname=host, assert_hostname=host, cert_reqs="CERT_REQUIRED",
        ca_certs=ca_certs, timeout=urllib3.Timeout(connect=timeout[0], read=timeout[1]), retries=False,
        maxsize=1)
    host_header = f"[{host}]" if ":" in host else host
    if port != 443:
        host_header += f":{port}"
    try:
        response = pool.urlopen("POST", path, body=body, headers={**headers, "Host": host_header},
                                redirect=False, retries=False, preload_content=False, assert_same_host=False)
        try:
            return int(response.status)
        finally:
            response.release_conn()
    finally:
        pool.close()


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS webhooks (
  id          TEXT PRIMARY KEY,
  key_id      TEXT NOT NULL,
  school_id   TEXT NOT NULL,
  url         TEXT NOT NULL,
  events      TEXT NOT NULL,
  secret      TEXT NOT NULL,
  secret_hint TEXT NOT NULL,
  created_by  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  revoked_at  TEXT,
  revoked_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_webhooks_key ON webhooks(key_id);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
  id               TEXT PRIMARY KEY,
  webhook_id       TEXT NOT NULL,
  event_id         TEXT NOT NULL,
  event_type       TEXT NOT NULL,
  payload          TEXT NOT NULL,
  status           TEXT NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  next_attempt_at  TEXT,
  last_status_code INTEGER,
  last_error       TEXT NOT NULL DEFAULT '',
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  delivered_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_wh_deliveries_due ON webhook_deliveries(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_wh_deliveries_hook ON webhook_deliveries(webhook_id, created_at);

-- The answer-keyed question ids the bank served at its last load, so the
-- next load can tell which are new. Empty means "never loaded": that load
-- records the set and announces nothing.
CREATE TABLE IF NOT EXISTS webhook_served (
  question_id TEXT PRIMARY KEY
);
"""


class WebhookStore:
    def __init__(self, db_path: Path | str, *, durable: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._snapshots = None
        if durable:
            from ..storage.snapshot_sync import SnapshotSync
            self._snapshots = SnapshotSync("operations-snapshots", "webhooks.sqlite", self.db_path, self._lock,
                                           debounce_seconds=2.0, allow_empty_boot=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=60000")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        if self._snapshots is not None:
            self._snapshots.commit_derived(self.conn)
            self._snapshots.save_empty_boot(self.conn)

    def _publish(self) -> None:
        if self._snapshots is not None:
            self._snapshots.commit(self.conn)
        else:
            self.conn.commit()

    def _one(self, sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
        with self._lock:
            r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r is not None else None

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # -- webhooks ---------------------------------------------------------- #

    def create(self, *, key_id: str, school_id: str, url: str, events: Iterable[str],
               created_by: str) -> tuple[str, dict[str, Any]]:
        secret = SECRET_PREFIX + secrets.token_urlsafe(32)
        wid = f"wh_{uuid.uuid4().hex[:16]}"
        with self._lock:
            self.conn.execute(
                "INSERT INTO webhooks (id, key_id, school_id, url, events, secret, secret_hint, created_by,"
                " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (wid, key_id, school_id, url, ",".join(sorted(set(events))), secret,
                 secret[:len(SECRET_PREFIX) + 4], created_by, _iso(_now())))
            self._publish()
        return secret, self.get(wid)  # type: ignore[return-value]

    def get(self, webhook_id: str) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM webhooks WHERE id=?", (webhook_id,))

    def for_key(self, key_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM webhooks WHERE key_id=? ORDER BY created_at DESC, id", (key_id,))

    def active(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM webhooks WHERE revoked_at IS NULL ORDER BY created_at, id")

    def revoke(self, webhook_id: str, *, revoked_by: str) -> Optional[dict[str, Any]]:
        """Stop the webhook and cancel what it has not yet delivered."""
        now = _iso(_now())
        with self._lock:
            self.conn.execute("UPDATE webhooks SET revoked_at=?, revoked_by=? WHERE id=? AND revoked_at IS NULL",
                              (now, revoked_by, webhook_id))
            self.conn.execute("UPDATE webhook_deliveries SET status='cancelled', last_error=?, updated_at=?,"
                              " next_attempt_at=NULL WHERE webhook_id=? AND status='pending'",
                              ("the webhook was revoked", now, webhook_id))
            self._publish()
        return self.get(webhook_id)

    # -- deliveries -------------------------------------------------------- #

    def enqueue(self, webhook_id: str, event: dict[str, Any], *, due_at: datetime) -> str:
        did = f"whd_{uuid.uuid4().hex[:16]}"
        now = _iso(_now())
        with self._lock:
            self.conn.execute(
                "INSERT INTO webhook_deliveries (id, webhook_id, event_id, event_type, payload, status, attempts,"
                " next_attempt_at, created_at, updated_at) VALUES (?,?,?,?,?,'pending',0,?,?,?)",
                (did, webhook_id, event["id"], event["type"], json.dumps(event, separators=(",", ":"),
                                                                         sort_keys=True), _iso(due_at), now, now))
            self._publish()
        return did

    def delivery(self, delivery_id: str) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM webhook_deliveries WHERE id=?", (delivery_id,))

    def due(self, now: datetime, limit: int = 100) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM webhook_deliveries WHERE status='pending' AND next_attempt_at <= ?"
                         " ORDER BY next_attempt_at, created_at LIMIT ?", (_iso(now), limit))

    def deliveries(self, webhook_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM webhook_deliveries WHERE webhook_id=? ORDER BY created_at DESC, id"
                         " LIMIT ?", (webhook_id, limit))

    def record(self, delivery_id: str, *, status: str, attempts: int, code: Optional[int], error: str,
               next_attempt_at: Optional[datetime]) -> None:
        now = _iso(_now())
        with self._lock:
            self.conn.execute(
                "UPDATE webhook_deliveries SET status=?, attempts=?, last_status_code=?, last_error=?,"
                " next_attempt_at=?, updated_at=?, delivered_at=CASE WHEN ?='delivered' THEN ? ELSE delivered_at END"
                " WHERE id=?",
                (status, attempts, code, error[:500], _iso(next_attempt_at) if next_attempt_at else None, now,
                 status, now, delivery_id))
            self._publish()

    # -- the served baseline ----------------------------------------------- #

    def served_ids(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self.conn.execute("SELECT question_id FROM webhook_served")}

    def set_served_ids(self, ids: Iterable[str]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM webhook_served")
            self.conn.executemany("INSERT INTO webhook_served (question_id) VALUES (?)", [(i,) for i in ids])
            self._publish()

    def close(self) -> None:
        if self._snapshots is not None:
            self._snapshots.close()
        self.conn.close()


# --------------------------------------------------------------------------- #
# events and delivery
# --------------------------------------------------------------------------- #

def _event(*, subject: str, grade: Any, chapter_id: str, question_ids: list[str], reason: str) -> dict[str, Any]:
    data = {"subject": subject, "grade": grade, "chapterId": chapter_id, "questionIds": question_ids,
            "count": len(question_ids), "reason": reason}
    assert set(data) <= _EVENT_DATA_KEYS
    return {"id": f"evt_{uuid.uuid4().hex[:20]}", "type": "questions.added", "createdAt": _iso(_now()),
            "data": data}


def _rejected_for(school_id: str) -> frozenset[str]:
    from ..operations.question_reviews import rejected_for
    return rejected_for(school_id)


class WebhookService:
    """Registration, the bank triggers, and delivery, over one store."""

    def __init__(self, store: WebhookStore, keys: ApiKeyStore, *, sender: Sender = https_sender,
                 resolver: Resolver = system_resolver, clock: Callable[[], datetime] = _now,
                 background: bool = True):
        self.store = store
        self.keys = keys
        self.sender = sender
        self.resolver = resolver
        self.clock = clock
        self.background = background
        self._sending = threading.Lock()

    # -- registration ------------------------------------------------------ #

    def register(self, key: ApiKey, *, url: str, events: Iterable[str], created_by: str) -> tuple[str, dict]:
        wanted = set(events or ())
        if not wanted:
            raise WebhookError("name at least one event; the events are " + ", ".join(sorted(EVENTS)))
        unknown = wanted - EVENTS
        if unknown:
            raise WebhookError(f"unknown event(s) {sorted(unknown)}; the events are " + ", ".join(sorted(EVENTS)))
        if not key.active:
            raise WebhookError("this key is revoked")
        if not key.may("questions:read"):
            raise WebhookError("questions.added names questions, so the key must hold the 'questions:read' scope")
        live = [w for w in self.store.for_key(key.id) if w["revoked_at"] is None]
        if len(live) >= MAX_WEBHOOKS_PER_KEY:
            raise WebhookError(f"a key can have at most {MAX_WEBHOOKS_PER_KEY} webhooks; revoke one first")
        check_url(url, self.resolver)
        return self.store.create(key_id=key.id, school_id=key.school_id, url=url.strip(), events=wanted,
                                 created_by=created_by)

    # -- the triggers ------------------------------------------------------ #

    def on_bank_loaded(self, bank: Any) -> int:
        """Called whenever the served bank is (re)loaded. Diffs the answer-keyed
        ids against the last load's, records the new set, and makes one event
        per chapter of new ids for each webhook whose key may read them. The
        first load ever announces nothing. Returns the events made."""
        served = {str(r.get("id")): r for r in bank.records if has_answer_key(r)}
        before = self.store.served_ids()
        if set(served) == before:
            return 0                     # an ordinary restart: nothing to write or announce
        self.store.set_served_ids(served)
        if not before:
            return 0
        added = [served[i] for i in sorted(set(served) - before)]
        return self._announce(bank, added, reason="bank_updated") if added else 0

    def on_review_decision(self, bank: Any, *, school_id: str, question_id: str,
                           before: Optional[str], after: str) -> int:
        """A school's reviewer decided on a question. The question rejoins what
        that school's keys are announced when a rejection is taken back. A
        first approval adds nothing (/v1 already serves a checked answer), so
        it makes no event."""
        if before != "rejected" or after == "rejected":
            return 0
        rec = bank.get(question_id)
        if rec is None or not has_answer_key(rec):
            return 0
        return self._announce(bank, [rec], reason="review_approved", school_id=school_id)

    def _announce(self, bank: Any, records: list[dict[str, Any]], *, reason: str,
                  school_id: Optional[str] = None) -> int:
        made = 0
        rejected: dict[str, frozenset[str]] = {}
        for hook in self.store.active():
            if school_id is not None and hook["school_id"] != school_id:
                continue
            if "questions.added" not in hook["events"].split(","):
                continue
            key = self.keys.get(hook["key_id"])
            if key is None or not key.may("questions:read"):
                continue
            if hook["school_id"] not in rejected:
                rejected[hook["school_id"]] = _rejected_for(hook["school_id"])
            groups: dict[tuple[str, Any, str], list[str]] = {}
            for rec in records:
                rid = str(rec.get("id"))
                if rid in rejected[hook["school_id"]]:
                    continue
                if not (key.admits_grade(rec.get("grade")) and key.admits_subject(rec.get("subject"))):
                    continue
                groups.setdefault((str(rec.get("subject") or ""), rec.get("grade"), bank.chapter_of(rec)),
                                  []).append(rid)
            for (subject, grade, chapter), ids in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]),
                                                                                         kv[0][2])):
                for i in range(0, len(ids), MAX_IDS_PER_EVENT):
                    self.store.enqueue(hook["id"], _event(subject=subject, grade=grade, chapter_id=chapter,
                                                          question_ids=ids[i:i + MAX_IDS_PER_EVENT],
                                                          reason=reason), due_at=self.clock())
                    made += 1
        if made:
            self.kick()
        return made

    # -- delivery ---------------------------------------------------------- #

    def kick(self) -> None:
        """Deliver what is due now, off the caller's thread."""
        if not self.background:
            return
        threading.Thread(target=self._deliver_quietly, name="webhook-delivery", daemon=True).start()

    def _deliver_quietly(self) -> None:
        try:
            self.deliver_due()
        except Exception:  # noqa: BLE001 - a delivery run must never take a thread down noisily
            log.warning("webhook delivery run failed", exc_info=True)

    def deliver_due(self) -> dict[str, int]:
        """Send every delivery that is due, once. Two runs never overlap: the
        second returns at once. Returns counts by outcome."""
        counts = {"delivered": 0, "retrying": 0, "failed": 0, "cancelled": 0}
        if not self._sending.acquire(blocking=False):
            return counts
        try:
            # Batches until nothing is due, so events made while this run was
            # sending go out in it too (their kick found the run busy). Each
            # outcome takes a delivery out of "due", so this ends.
            for _ in range(50):
                batch = self.store.due(self.clock())
                if not batch:
                    break
                for d in batch:
                    counts[self._deliver(d)] += 1
        finally:
            self._sending.release()
        return counts

    def _cancel(self, d: dict[str, Any], why: str) -> str:
        self.store.record(d["id"], status="cancelled", attempts=d["attempts"], code=d["last_status_code"],
                          error=why, next_attempt_at=None)
        return "cancelled"

    def _deliver(self, d: dict[str, Any]) -> str:
        hook = self.store.get(d["webhook_id"])
        if hook is None or hook["revoked_at"] is not None:
            return self._cancel(d, "the webhook was revoked")
        key = self.keys.get(hook["key_id"])
        if key is None or not key.active:
            return self._cancel(d, "the API key was revoked")
        event = json.loads(d["payload"])
        data = event.get("data") or {}
        if (not key.may("questions:read") or not key.admits_grade(data.get("grade"))
                or not key.admits_subject(data.get("subject"))):
            return self._cancel(d, "outside what the key may read")
        attempt = int(d["attempts"]) + 1
        body = d["payload"].encode("utf-8")
        code: Optional[int] = None
        error = ""
        try:
            _host, _port, addresses = check_url(hook["url"], self.resolver)
            # IPv4 first: Cloud Run's egress may have no IPv6 route.
            address = next((a for a in addresses if ":" not in a), addresses[0])
            timestamp = int(self.clock().timestamp())
            headers = {"Content-Type": "application/json", "User-Agent": "AcademicOS-Webhooks/1",
                       SIGNATURE_HEADER: signature_header(hook["secret"], body, timestamp),
                       EVENT_HEADER: d["event_type"], DELIVERY_HEADER: d["id"]}
            code = int(self.sender(hook["url"], address, body, headers, (CONNECT_TIMEOUT, READ_TIMEOUT)))
            if 200 <= code < 300:
                self.store.record(d["id"], status="delivered", attempts=attempt, code=code, error="",
                                  next_attempt_at=None)
                return "delivered"
            error = f"the receiver answered HTTP {code}"
        except UnsafeWebhookUrl as exc:
            error = f"refused: {exc}"
        except Exception as exc:  # noqa: BLE001 - every failure is recorded on the delivery and retried
            error = f"{type(exc).__name__}: {exc}"
        if attempt >= MAX_ATTEMPTS:
            self.store.record(d["id"], status="failed", attempts=attempt, code=code, error=error,
                              next_attempt_at=None)
            return "failed"
        self.store.record(d["id"], status="pending", attempts=attempt, code=code, error=error,
                          next_attempt_at=self.clock() + timedelta(seconds=BACKOFF_SECONDS[attempt - 1]))
        return "retrying"


# --------------------------------------------------------------------------- #
# the process's service
# --------------------------------------------------------------------------- #

_service: Optional[WebhookService] = None


def init(db_path: Path | str, keys: ApiKeyStore, *, durable: bool = True, **kwargs: Any) -> WebhookService:
    global _service
    _service = WebhookService(WebhookStore(db_path, durable=durable), keys, **kwargs)
    return _service


def service() -> Optional[WebhookService]:
    return _service


def bank_loaded_safely(bank: Any) -> int:
    """For the bank loader: a webhook problem must never stop the bank."""
    if _service is None or bank is None:
        return 0
    try:
        return _service.on_bank_loaded(bank)
    except Exception:  # noqa: BLE001
        log.warning("could not compute webhook events for the bank load", exc_info=True)
        return 0


def review_decided_safely(bank: Any, *, school_id: str, question_id: str, before: Optional[str],
                          after: str) -> int:
    """For the review queue: a webhook problem must never stop a decision."""
    if _service is None or bank is None:
        return 0
    try:
        return _service.on_review_decision(bank, school_id=school_id, question_id=question_id,
                                           before=before, after=after)
    except Exception:  # noqa: BLE001
        log.warning("could not make webhook events for a review decision", exc_info=True)
        return 0


def deliver_due_safely() -> dict[str, int]:
    """For the automations runs: retries go out even with no new events."""
    if _service is None:
        return {}
    try:
        return _service.deliver_due()
    except Exception:  # noqa: BLE001
        log.warning("webhook delivery run failed", exc_info=True)
        return {}
