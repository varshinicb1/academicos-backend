"""SMS and WhatsApp as notification channels (REQUIREMENTS INT-2, NTF-1 P1;
SA-6: a parent "receives notices by push/WhatsApp/SMS").

Until now a notification reached a person only in the app (inbox and push)
or by email. Many parents read neither; nearly all read WhatsApp and SMS.

- **Who.** A person gives their own Indian mobile number and says, per
  channel, whether they want SMS and whether they want WhatsApp -- nothing
  is sent without that consent, and it can be withdrawn at any time. Staff
  and parents only: a student is a minor, and messages about them go to
  their parent (DPDP Act 2023). Nobody else can read the number.
- **What.** Only kinds about the person's next action that matter off the
  app: a cover request, a leave decision, homework due, an exam or duty
  tomorrow, a holiday. And not a kind whose push the person has turned off.
  Quiet hours hold messages like push; someone on a daily digest gets only
  the kinds that cannot wait.
- **How.** WhatsApp through the WhatsApp Business Platform (Cloud API, one
  approved utility template with the notice's title and body as its two
  parameters); SMS through a DLT-registered gateway (MSG91's v5 flow API,
  one flow template with `##title##` and `##body##`). Each needs the owner's
  account, sender registration and template approval -- until then the
  channel records "skipped: not configured" and nothing leaves the server.
  Each delivery is recorded per channel (pending, deferred, sent, failed,
  skipped) and retried up to three times, like push and email.

Nothing here imports notifications.py or FastAPI at module level: that
module mixes this one in, and the routes are in messaging_routes.py.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

MESSAGING_SCHEMA = """
-- INT-2: a person's own mobile number, and their consent per channel.
CREATE TABLE IF NOT EXISTS contact_numbers (
  user_id     TEXT PRIMARY KEY,
  phone       TEXT NOT NULL,
  sms_ok      INTEGER NOT NULL DEFAULT 0,
  whatsapp_ok INTEGER NOT NULL DEFAULT 0,
  updated_at  TEXT NOT NULL
);
-- One row per notification and message channel it goes out on.
CREATE TABLE IF NOT EXISTS message_deliveries (
  notification_id TEXT NOT NULL,
  user_id         TEXT NOT NULL,
  channel         TEXT NOT NULL,
  status          TEXT NOT NULL,
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_error      TEXT,
  provider_id     TEXT,
  updated_at      TEXT NOT NULL,
  PRIMARY KEY (notification_id, channel)
);
CREATE INDEX IF NOT EXISTS idx_message_due ON message_deliveries(status);
"""

MESSAGE_CHANNELS = ("sms", "whatsapp")
# The kinds worth a message: each is about the person's next action and
# loses its point if it waits until they open the app.
MESSAGE_KINDS = frozenset({"substitution_proposed", "leave_decided", "homework_due", "exam_reminder",
                           "duty_reminder", "holiday_declared"})
MESSAGE_ROLES = ("teacher", "principal", "parent")
_MOBILE = re.compile(r"^[6-9]\d{9}$")
SMS_LIMIT = 300          # characters: two SMS parts at most


def normalize_mobile(raw: str) -> str:
    """An Indian mobile number as +91XXXXXXXXXX, from the ways people type
    it ("98765 43210", "+91-98765-43210", "098765 43210"). ValueError for
    anything else: this product serves Indian schools."""
    digits = re.sub(r"[\s\-().]", "", raw or "")
    if digits.startswith("+91"):
        digits = digits[3:]
    elif digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if not _MOBILE.match(digits):
        raise ValueError("give an Indian mobile number: 10 digits starting with 6, 7, 8 or 9")
    return "+91" + digits


# ---------------------------------------------------------------- senders

def _configured(channel: str) -> Optional[str]:
    """None when the channel is set up; otherwise what is missing."""
    if channel == "whatsapp":
        missing = [v for v in ("ACOS_WHATSAPP_TOKEN", "ACOS_WHATSAPP_PHONE_NUMBER_ID") if not os.environ.get(v)]
    else:
        missing = [v for v in ("ACOS_SMS_AUTHKEY", "ACOS_SMS_TEMPLATE_ID") if not os.environ.get(v)]
    return f"{channel} is not configured ({', '.join(missing)})" if missing else None


def send_whatsapp(phone: str, title: str, body: str, language: str) -> tuple[str, Optional[str], Optional[str]]:
    """(status, error, provider message id). The WhatsApp Cloud API template
    message: the approved template ACOS_WHATSAPP_TEMPLATE (default
    "academicos_notice") with the title and body as its body parameters."""
    missing = _configured("whatsapp")
    if missing:
        return "skipped", missing, None
    import requests
    version = os.environ.get("ACOS_WHATSAPP_API_VERSION", "v21.0")
    url = f"https://graph.facebook.com/{version}/{os.environ['ACOS_WHATSAPP_PHONE_NUMBER_ID']}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": phone.lstrip("+"),
        "type": "template",
        "template": {
            "name": os.environ.get("ACOS_WHATSAPP_TEMPLATE", "academicos_notice"),
            "language": {"code": "hi" if language == "hi" else "en"},
            "components": [{"type": "body", "parameters": [{"type": "text", "text": title[:60]},
                                                           {"type": "text", "text": body[:900]}]}],
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=10,
                          headers={"Authorization": f"Bearer {os.environ['ACOS_WHATSAPP_TOKEN']}"})
        if r.status_code >= 300:
            return "failed", f"{r.status_code} {r.text[:200]}", None
        ids = (r.json() or {}).get("messages") or [{}]
        return "sent", None, ids[0].get("id")
    except Exception as exc:  # noqa: BLE001 - recorded on the delivery and retried
        return "failed", str(exc)[:300], None


def send_sms(phone: str, title: str, body: str) -> tuple[str, Optional[str], Optional[str]]:
    """(status, error, provider request id). MSG91's v5 flow API with the
    DLT-registered flow ACOS_SMS_TEMPLATE_ID, whose text carries ##title##
    and ##body##; at most SMS_LIMIT characters in all."""
    missing = _configured("sms")
    if missing:
        return "skipped", missing, None
    import requests
    room = max(0, SMS_LIMIT - len(title) - 2)
    payload = {
        "template_id": os.environ["ACOS_SMS_TEMPLATE_ID"],
        "short_url": "0",
        "recipients": [{"mobiles": phone.lstrip("+"), "title": title[:80], "body": body[:room]}],
    }
    try:
        r = requests.post("https://control.msg91.com/api/v5/flow", json=payload, timeout=10,
                          headers={"authkey": os.environ["ACOS_SMS_AUTHKEY"], "accept": "application/json"})
        data = r.json() if r.content else {}
        if r.status_code >= 300 or (isinstance(data, dict) and data.get("type") == "error"):
            return "failed", f"{r.status_code} {str(data)[:200]}", None
        return "sent", None, (data.get("message") if isinstance(data, dict) else None)
    except Exception as exc:  # noqa: BLE001
        return "failed", str(exc)[:300], None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the store

class MessagingMixin:
    """Mixed into NotificationsMixin (operations store). ValueError is the
    route's 422."""

    def contact_for(self, user_id: str) -> Optional[dict[str, Any]]:
        return self._fetchone("SELECT * FROM contact_numbers WHERE user_id=?", (user_id,))

    def set_contact(self, user_id: str, *, phone: Optional[str], sms_ok: bool,
                    whatsapp_ok: bool) -> tuple[Optional[dict], Optional[dict]]:
        """Set, or with phone None remove, a person's number and consent.
        Returns (before, after). Removing the number withdraws both."""
        before = self.contact_for(user_id)
        if phone is None:
            self._exec("DELETE FROM contact_numbers WHERE user_id=?", (user_id,))
            # Anything still waiting to go to that number goes nowhere now.
            self._exec("UPDATE message_deliveries SET status='skipped', last_error='consent withdrawn', "
                       "updated_at=? WHERE user_id=? AND status IN ('pending','deferred','failed')",
                       (_now(), user_id))
            self._commit()
            return before, None
        number = normalize_mobile(phone)
        self._exec("INSERT INTO contact_numbers (user_id, phone, sms_ok, whatsapp_ok, updated_at) VALUES (?,?,?,?,?) "
                   "ON CONFLICT(user_id) DO UPDATE SET phone=excluded.phone, sms_ok=excluded.sms_ok, "
                   "whatsapp_ok=excluded.whatsapp_ok, updated_at=excluded.updated_at",
                   (user_id, number, int(sms_ok), int(whatsapp_ok), _now()))
        for ch, ok in (("sms", sms_ok), ("whatsapp", whatsapp_ok)):
            if not ok:
                self._exec("UPDATE message_deliveries SET status='skipped', last_error='consent withdrawn', "
                           "updated_at=? WHERE user_id=? AND channel=? AND status IN ('pending','deferred','failed')",
                           (_now(), user_id, ch))
        self._commit()
        return before, self.contact_for(user_id)

    def _queue_messages(self, notification_id: str, user_id: str, kind: str, *, quiet: bool,
                        on_digest: bool) -> None:
        """Called by notify() inside its lock: one delivery row per channel
        the person consented to, for a kind worth a message. Executes only;
        notify() commits."""
        if kind not in MESSAGE_KINDS:
            return
        from .notifications import DIGEST_EXEMPT
        if on_digest and kind not in DIGEST_EXEMPT:
            return
        contact = self.contact_for(user_id)
        if contact is None or not self.channel_enabled(user_id, kind, "push"):
            return
        for ch in MESSAGE_CHANNELS:
            if contact[f"{ch}_ok"]:
                self._exec("INSERT OR IGNORE INTO message_deliveries (notification_id, user_id, channel, status, "
                           "updated_at) VALUES (?,?,?,?,?)",
                           (notification_id, user_id, ch, "deferred" if quiet else "pending", _now()))

    def _deliver_messages(self, now: datetime) -> int:
        """Send due messages (pending, deferred past quiet hours, failed with
        attempts left). Returns how many went."""
        from .notifications import IST, MAX_ATTEMPTS, in_quiet_hours
        sent = 0
        rows = self._fetchall("SELECT d.*, n.title, n.body FROM message_deliveries d "
                              "JOIN notifications n ON n.id = d.notification_id "
                              "WHERE d.status IN ('pending','deferred','failed') AND d.attempts < ?", (MAX_ATTEMPTS,))
        for r in rows:
            s = self.settings_for(r["user_id"])
            if in_quiet_hours(now.astimezone(IST), s["quiet_start"], s["quiet_end"]):
                continue
            contact = self.contact_for(r["user_id"])
            if contact is None or not contact[f"{r['channel']}_ok"]:
                status, err, pid = "skipped", "consent withdrawn", None
            elif r["channel"] == "whatsapp":
                status, err, pid = send_whatsapp(contact["phone"], r["title"], r["body"], s["language"])
            else:
                status, err, pid = send_sms(contact["phone"], r["title"], r["body"])
            self._exec("UPDATE message_deliveries SET status=?, attempts=attempts+?, last_error=?, provider_id=?, "
                       "updated_at=? WHERE notification_id=? AND channel=?",
                       (status, 1 if status == "failed" else 0, err, pid, _now(), r["notification_id"], r["channel"]))
            sent += status == "sent"
        return sent

    def message_status(self, notification_id: str) -> dict[str, str]:
        return {r["channel"]: r["status"] for r in self._fetchall(
            "SELECT channel, status FROM message_deliveries WHERE notification_id=?", (notification_id,))}
