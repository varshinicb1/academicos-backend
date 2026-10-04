"""One notification service (REQUIREMENTS NTF-1, docs/notifications.md).

Every notification lands in the recipient's in-app inbox. Push (FCM) and
email go out only for the kinds the recipient allows, and never inside their
quiet hours (default 21:00-07:00 IST, SA-5): then they wait and go at the end
of the window. Each item records its delivery per channel (pending, deferred,
sent, failed, skipped) and failed sends are retried up to three times.

The policy is in the catalogue below: a kind exists only because it is about
the recipient's next action (docs/notifications.md rule 1). There is no
engagement kind, no metric kind, and nothing with an emoji.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Optional

from .messaging import MESSAGING_SCHEMA, MessagingMixin

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

NOTIFICATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  user_id      TEXT NOT NULL,
  kind         TEXT NOT NULL,
  title        TEXT NOT NULL,
  body         TEXT NOT NULL,
  link         TEXT,
  data_json    TEXT NOT NULL DEFAULT '{}',
  created_at   TEXT NOT NULL,
  read_at      TEXT,
  push_status  TEXT NOT NULL DEFAULT 'none',
  email_status TEXT NOT NULL DEFAULT 'none',
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_error   TEXT,
  dedupe_key   TEXT,
  UNIQUE(user_id, dedupe_key)
);
CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notif_due ON notifications(push_status, email_status);

CREATE TABLE IF NOT EXISTS notification_prefs (
  user_id TEXT NOT NULL,
  kind    TEXT NOT NULL,
  channel TEXT NOT NULL,
  enabled INTEGER NOT NULL,
  PRIMARY KEY (user_id, kind, channel)
);

CREATE TABLE IF NOT EXISTS notification_settings (
  user_id     TEXT PRIMARY KEY,
  quiet_start TEXT NOT NULL DEFAULT '21:00',
  quiet_end   TEXT NOT NULL DEFAULT '07:00',
  language    TEXT NOT NULL DEFAULT 'en'
);

-- SA-5: a person who wants one message a day instead of each as it comes.
-- Its own table, not a column, so a restored older snapshot needs no migration.
CREATE TABLE IF NOT EXISTS notification_digests (
  user_id      TEXT PRIMARY KEY,
  digest_at    TEXT NOT NULL,
  last_sent_on TEXT
);

CREATE TABLE IF NOT EXISTS device_tokens (
  token      TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL,
  platform   TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_seen  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_device_user ON device_tokens(user_id);

-- N-8-10: how many sends in a row the push provider refused because the
-- token is dead (unregistered or not valid). Its own table, not a column,
-- so a restored older snapshot needs no migration.
CREATE TABLE IF NOT EXISTS device_token_failures (
  token      TEXT PRIMARY KEY,
  failures   INTEGER NOT NULL,
  last_error TEXT
);
""" + MESSAGING_SCHEMA

CHANNELS = ("push", "email")
MAX_ATTEMPTS = 3
# Kinds that go at once even for someone on a daily digest: their point is
# that the person acts within the hour.
DIGEST_EXEMPT = frozenset({"substitution_proposed"})


@dataclass(frozen=True)
class Kind:
    key: str
    label: str
    push: bool             # default for push
    email: bool            # default for email
    en: tuple[str, str]    # (title, body) templates
    hi: tuple[str, str]


CATALOGUE: dict[str, Kind] = {k.key: k for k in (
    Kind("substitution_proposed", "A substitution for you to accept", True, False,
         ("Substitution: {section} period {period}", "{date}: {subject} for {section}, period {period}. Accept or decline."),
         ("स्थानापन्न: {section} पीरियड {period}", "{date}: {section} के लिए {subject}, पीरियड {period}। स्वीकार या अस्वीकार करें।")),
    Kind("substitution_confirmed", "Who is covering your periods", False, False,
         ("Your periods are covered", "{date}: {count} of your periods have a substitute."),
         ("आपके पीरियड व्यवस्थित हैं", "{date}: आपके {count} पीरियड के लिए स्थानापन्न तय है।")),
    Kind("leave_decided", "A decision on your leave", True, False,
         ("Leave {decision}", "Your leave from {start} to {end} was {decision}."),
         ("अवकाश {decision}", "{start} से {end} तक का आपका अवकाश {decision}।")),
    # Staff automation (2026-10-04): the principal hears of leave that waits
    # for them, and of leave the school's rule approved, with what cover lacks.
    Kind("leave_requested", "Leave waiting for your decision", True, True,
         ("Leave request: {teacher}", "{teacher} asks for leave from {start} to {end} ({days}). Approve or decline."),
         ("अवकाश अनुरोध: {teacher}", "{teacher} ने {start} से {end} तक ({days}) अवकाश माँगा है। स्वीकार या अस्वीकार करें।")),
    Kind("leave_auto_approved", "Leave approved by the school's rule", True, False,
         ("Leave approved: {teacher}", "{teacher}'s leave from {start} to {end} was approved under your rule. {cover}"),
         ("अवकाश स्वीकृत: {teacher}", "{teacher} का {start} से {end} तक का अवकाश नियम से स्वीकृत हुआ। {cover_hi}")),
    Kind("check_in_reminder", "A reminder to check in", True, False,
         ("Check in for today", "You have not checked in yet. Open AcademicOS and tap I'm here."),
         ("आज की उपस्थिति दर्ज करें", "आपने अभी उपस्थिति दर्ज नहीं की है। AcademicOS खोलें और 'मैं यहाँ हूँ' दबाएँ।")),
    Kind("staff_not_checked_in", "Teachers who have not checked in", True, False,
         ("{count} not checked in", "{names}. Mark them, or arrange cover for their periods."),
         ("{count} ने उपस्थिति नहीं दी", "{names}। उपस्थिति दर्ज करें या उनके पीरियड की व्यवस्था करें।")),
    Kind("timetable_published", "A new or changed timetable", True, False,
         ("Timetable updated", "The timetable has changed. Check your week."),
         ("समय-सारिणी बदली", "समय-सारिणी बदली है। अपना सप्ताह देखें।")),
    Kind("lesson_rescheduled", "A lesson moved on your calendar", True, False,
         ("Lesson moved", "{subject} for {section} moved to {date}."),
         ("पाठ स्थानांतरित", "{section} का {subject} {date} पर स्थानांतरित।")),
    Kind("homework_assigned", "New homework", True, False,
         ("New homework: {subject}", "{title}. Due {due}."),
         ("नया गृहकार्य: {subject}", "{title}। अंतिम तिथि {due}।")),
    Kind("homework_due", "Homework due soon", True, False,
         ("Homework due {due}", "{title} is due {due} and is not submitted yet."),
         ("गृहकार्य {due} तक", "{title} {due} तक जमा करना है।")),
    Kind("homework_graded", "Homework marked", True, False,
         ("Homework marked: {subject}", "{title}: {marks}."),
         ("गृहकार्य जाँचा गया: {subject}", "{title}: {marks}।")),
    # TA-7: a teacher had no notice that work was handed in, or that a period
    # was about to start (v3 audit).
    Kind("homework_submitted", "Homework handed in", False, False,
         ("Handed in: {title}", "{student} handed in {title} ({count} of {total} so far)."),
         ("जमा हुआ: {title}", "{student} ने {title} जमा किया (अब तक {total} में से {count})।")),
    Kind("period_starting", "A period about to start", True, False,
         ("Period {period} at {start}", "{section} {subject}{room}."),
         ("पीरियड {period} {start} पर", "{section} {subject}{room}।")),
    # SA-5: students heard of homework results but not of a test's.
    Kind("test_marks", "Marks for a test", True, False,
         ("Marks: {title}", "{subject}: {marks}."),
         ("अंक: {title}", "{subject}: {marks}।")),
    Kind("homework_overdue", "Homework with students missing", True, False,
         ("{count} not submitted: {title}", "{title} was due {due}; {count} students have not submitted."),
         ("{count} ने जमा नहीं किया: {title}", "{title} {due} तक था; {count} विद्यार्थियों ने जमा नहीं किया।")),
    Kind("marking_due", "Work waiting for your marks", True, False,
         ("{count} waiting for your marks", "{count} homework submissions have waited over {days} days; the oldest is {title}."),
         ("{count} आपके अंकों की प्रतीक्षा में", "{count} गृहकार्य {days} दिन से अधिक से जाँच की प्रतीक्षा में; सबसे पुराना {title}।")),
    Kind("daily_digest", "Tomorrow's periods and duties", True, False,
         ("Tomorrow: {periods} periods, {duties} duties", "First: {first}."),
         ("कल: {periods} पीरियड, {duties} ड्यूटी", "पहला: {first}।")),
    Kind("weekly_digest", "The week in review", False, True,
         ("Week of {week}", "{substitutions} substitutions ({unfilled} unfilled), {lost} periods lost, "
                            "{homework} homework set."),
         ("{week} का सप्ताह", "{substitutions} स्थानापन्न ({unfilled} खाली), {lost} पीरियड छूटे, "
                              "{homework} गृहकार्य।")),
    Kind("paper_review_requested", "A paper to review", True, True,
         ("Paper to review: {title}", "{author} asks you to review {title}."),
         ("समीक्षा हेतु प्रश्नपत्र: {title}", "{author} ने {title} की समीक्षा का अनुरोध किया है।")),
    Kind("paper_review_decided", "A review of your paper", True, False,
         ("Review of {title}", "{reviewer} {decision}. {comment}"),
         ("{title} की समीक्षा", "{reviewer}: {decision}। {comment}")),
    Kind("exam_scheduled", "An exam datesheet", True, False,
         ("Datesheet: {exam}", "{exam} starts {start}. The datesheet is in the app."),
         ("डेटशीट: {exam}", "{exam} {start} से शुरू। डेटशीट ऐप में है।")),
    Kind("exam_changed", "A change to a published datesheet", True, False,
         ("Datesheet changed: {exam}", "{changes}. Check the datesheet in the app."),
         ("डेटशीट बदली: {exam}", "{changes}। डेटशीट ऐप में देखें।")),
    Kind("invigilation_duty", "Your invigilation duties", True, True,
         ("Invigilation: {exam}", "You have {count} invigilation duties in {exam}."),
         ("निरीक्षण ड्यूटी: {exam}", "{exam} में आपकी {count} निरीक्षण ड्यूटी हैं।")),
    Kind("exam_reminder", "An exam tomorrow", True, False,
         ("Exam tomorrow: {subject}", "{exam}: {subject}, {start}-{end}."),
         ("कल परीक्षा: {subject}", "{exam}: {subject}, {start}-{end}।")),
    Kind("duty_reminder", "Invigilation tomorrow", True, False,
         ("Invigilation tomorrow: {count}", "{exam}: the first is {first}."),
         ("कल निरीक्षण: {count}", "{exam}: पहली {first}।")),
    Kind("holiday_declared", "A holiday declared, and lessons moved", True, False,
         ("Holiday: {label}", "{dates}: no school. {moved}"),
         ("अवकाश: {label}", "{dates}: विद्यालय बंद। {moved_hi}")),
    Kind("term_report_ready", "The term report at the end of a term", False, True,
         ("Term report: {term}", "{term} ends {end}. Its report is ready to open or download."),
         ("सत्र रिपोर्ट: {term}", "{term} {end} को समाप्त। इसकी रिपोर्ट खोलने या डाउनलोड करने के लिए तैयार है।")),
    Kind("principal_alert", "Something needs the principal", False, True,
         ("{title}", "{body}"), ("{title}", "{body}")),
)}


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render(kind: str, params: dict[str, Any], language: str = "en") -> tuple[str, str]:
    """A parameter built from words, not data ("tomorrow", an alert's
    sentence) is sent twice: `key` in English and `key_hi` in Hindi. A Hindi
    reader gets `key_hi` wherever the template says `{key}` (N-8-4)."""
    k = CATALOGUE[kind]
    title, body = k.hi if language == "hi" else k.en
    p = _SafeDict({key: str(v) for key, v in params.items()})
    if language == "hi":
        p.update({key[:-3]: str(v) for key, v in params.items() if key.endswith("_hi")})
    return title.format_map(p), body.format_map(p)


def _parse_hhmm(v: str) -> time:
    h, m = v.split(":")
    return time(int(h), int(m))


def in_quiet_hours(now_ist: datetime, start: str, end: str) -> bool:
    t, s, e = now_ist.time(), _parse_hhmm(start), _parse_hhmm(end)
    return (s <= t < e) if s < e else (t >= s or t < e)


# ---------------- channel senders ----------------

def token_is_dead(status_code: int, text: str) -> bool:
    """FCM's answer for a token that will never work again: UNREGISTERED (the
    app was uninstalled or the token rotated), or a 400 naming the
    registration token as not valid. Anything else -- a 5xx, a quota, an auth
    failure, a wrong project (a bare 404, SENDER_ID_MISMATCH) -- is transient
    or a configuration fault, says nothing about the token, and must never
    get every device deleted."""
    t = (text or "").upper()
    return "UNREGISTERED" in t or (status_code == 400 and "REGISTRATION TOKEN" in t)


def send_push(tokens: list[str], title: str, body: str, data: dict[str, str]) -> tuple:
    """FCM HTTP v1 through the service account's credentials. ('skipped',
    reason) when push is not set up; ('failed', error) when FCM refused.
    A third element lists the tokens FCM called dead (token_is_dead); callers
    that unpack two values may ignore it."""
    if not tokens:
        return "skipped", "no device registered", []
    project = os.environ.get("ACOS_FCM_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return "skipped", "push is not configured (ACOS_FCM_PROJECT)", []
    try:
        import google.auth
        import google.auth.transport.requests
        import requests
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/firebase.messaging"])
        creds.refresh(google.auth.transport.requests.Request())
        errors, dead = [], []
        for token in tokens:
            r = requests.post(f"https://fcm.googleapis.com/v1/projects/{project}/messages:send",
                              headers={"Authorization": f"Bearer {creds.token}"},
                              json={"message": {"token": token, "notification": {"title": title, "body": body},
                                                "data": {k: str(v) for k, v in data.items()}}}, timeout=10)
            if r.status_code >= 300:
                errors.append(f"{r.status_code} {r.text[:120]}")
                if token_is_dead(r.status_code, r.text):
                    dead.append(token)
        if errors and len(errors) == len(tokens):
            return "failed", "; ".join(errors)[:300], dead
        return "sent", None, dead
    except Exception as exc:  # noqa: BLE001 - recorded on the item, retried
        return "failed", str(exc)[:300], []


def send_email(address: Optional[str], title: str, body: str) -> tuple[str, Optional[str]]:
    if not address:
        return "skipped", "no email address"
    try:
        from ..assessment import mailer
        if not any(b.get("configured") for b in mailer.available_backends().values()):
            return "skipped", "email is not configured"
        result = mailer.send(mailer.MailMessage(to=[address], subject=title, body=body))
        return ("sent", None) if result.ok else ("failed", result.detail[:300])
    except Exception as exc:  # noqa: BLE001
        return "failed", str(exc)[:300]


@dataclass
class Notification:
    id: str
    school_id: str
    user_id: str
    kind: str
    title: str
    body: str
    link: Optional[str]
    data: dict[str, Any]
    created_at: str
    read_at: Optional[str]
    push_status: str
    email_status: str
    attempts: int
    last_error: Optional[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotificationsMixin(MessagingMixin):
    # Tests and the synchronous paths set this; production hands delivery to a
    # thread so a request never waits on FCM or SMTP.
    deliver_inline = False
    email_for: Optional[Callable[[str], Optional[str]]] = None
    # Guards only the creation of each store's own delivery lock.
    _delivery_guard = threading.Lock()

    def _delivery_lock(self) -> threading.Lock:
        lock = self.__dict__.get("_delivering")
        if lock is None:
            with NotificationsMixin._delivery_guard:
                lock = self.__dict__.setdefault("_delivering", threading.Lock())
        return lock

    def _notif(self, r: dict) -> Notification:
        r = dict(r)
        r["data"] = json.loads(r.pop("data_json") or "{}")
        r.pop("dedupe_key", None)
        return Notification(**r)

    # ---------------- settings ----------------

    def settings_for(self, user_id: str) -> dict[str, Any]:
        r = self._fetchone("SELECT * FROM notification_settings WHERE user_id=?", (user_id,))
        out = dict(r or {"user_id": user_id, "quiet_start": "21:00", "quiet_end": "07:00", "language": "en"})
        d = self._fetchone("SELECT digest_at FROM notification_digests WHERE user_id=?", (user_id,))
        out["digest_at"] = d["digest_at"] if d else None
        return out

    def set_digest(self, user_id: str, digest_at: Optional[str]) -> None:
        """One push and one email a day at `digest_at` (HH:MM, IST) instead
        of each as it comes; None turns it off. The inbox is unchanged."""
        if digest_at is None:
            self._exec("DELETE FROM notification_digests WHERE user_id=?", (user_id,))
        else:
            _parse_hhmm(digest_at)
            self._exec("INSERT INTO notification_digests (user_id, digest_at) VALUES (?,?) "
                       "ON CONFLICT(user_id) DO UPDATE SET digest_at=excluded.digest_at", (user_id, digest_at))
        self._commit()

    def set_settings(self, user_id: str, *, quiet_start: Optional[str] = None, quiet_end: Optional[str] = None,
                     language: Optional[str] = None) -> dict[str, str]:
        cur = self.settings_for(user_id)
        qs, qe, lang = quiet_start or cur["quiet_start"], quiet_end or cur["quiet_end"], language or cur["language"]
        for v in (qs, qe):
            _parse_hhmm(v)
        if lang not in ("en", "hi"):
            raise ValueError("language is en or hi")
        self._exec("INSERT INTO notification_settings (user_id, quiet_start, quiet_end, language) VALUES (?,?,?,?) "
                   "ON CONFLICT(user_id) DO UPDATE SET quiet_start=excluded.quiet_start, "
                   "quiet_end=excluded.quiet_end, language=excluded.language", (user_id, qs, qe, lang))
        self._commit()
        return self.settings_for(user_id)

    def channel_enabled(self, user_id: str, kind: str, channel: str) -> bool:
        r = self._fetchone("SELECT enabled FROM notification_prefs WHERE user_id=? AND kind=? AND channel=?",
                           (user_id, kind, channel))
        if r is not None:
            return bool(r["enabled"])
        k = CATALOGUE[kind]
        return k.push if channel == "push" else k.email

    def set_channel(self, user_id: str, kind: str, channel: str, enabled: bool) -> None:
        if kind not in CATALOGUE:
            raise ValueError(f"unknown notification kind {kind!r}")
        if channel not in CHANNELS:
            raise ValueError("channel is push or email (the inbox always gets every notification)")
        self._exec("INSERT INTO notification_prefs (user_id, kind, channel, enabled) VALUES (?,?,?,?) "
                   "ON CONFLICT(user_id, kind, channel) DO UPDATE SET enabled=excluded.enabled",
                   (user_id, kind, channel, int(enabled)))
        self._commit()

    # ---------------- devices ----------------

    def register_device(self, user_id: str, token: str, platform: str) -> None:
        if platform not in ("android", "ios", "web"):
            raise ValueError("platform is android, ios or web")
        now = _now_iso()
        self._exec("INSERT INTO device_tokens (token, user_id, platform, created_at, last_seen) VALUES (?,?,?,?,?) "
                   "ON CONFLICT(token) DO UPDATE SET user_id=excluded.user_id, platform=excluded.platform, "
                   "last_seen=excluded.last_seen", (token, user_id, platform, now, now))
        self._commit()

    def remove_device(self, user_id: str, token: str) -> bool:
        with self._conn_lock:
            cur = self.conn.execute("DELETE FROM device_tokens WHERE token=? AND user_id=?", (token, user_id))
            n = cur.rowcount
        self._commit()
        return n > 0

    def devices_for(self, user_id: str) -> list[str]:
        return [r["token"] for r in self._fetchall("SELECT token FROM device_tokens WHERE user_id=?", (user_id,))]

    def _note_dead_tokens(self, tokens: list[str], dead: list[str], status: str, err: Optional[str]) -> None:
        """N-8-10: a token the push provider has called dead MAX_ATTEMPTS
        sends in a row is removed, so later notifications to that person stop
        failing on it. A transient failure neither counts nor clears; a send
        that went through clears the tokens that were not called dead."""
        with self._conn_lock:
            for token in dead:
                self.conn.execute("INSERT INTO device_token_failures (token, failures, last_error) VALUES (?,1,?) "
                                  "ON CONFLICT(token) DO UPDATE SET failures=failures+1, last_error=excluded.last_error",
                                  (token, (err or "")[:300]))
                r = self.conn.execute("SELECT failures FROM device_token_failures WHERE token=?", (token,)).fetchone()
                if r and r["failures"] >= MAX_ATTEMPTS:
                    self.conn.execute("DELETE FROM device_tokens WHERE token=?", (token,))
                    self.conn.execute("DELETE FROM device_token_failures WHERE token=?", (token,))
                    logger.info("removed a push token FCM called dead %d times: %s", MAX_ATTEMPTS, err)
            if status == "sent":
                self.conn.executemany("DELETE FROM device_token_failures WHERE token=?",
                                      [(t,) for t in tokens if t not in dead])

    # ---------------- notify and deliver ----------------

    def notify(self, *, school_id: str, user_ids: list[str], kind: str, params: dict[str, Any],
               link: Optional[str] = None, dedupe_key: Optional[str] = None,
               now: Optional[datetime] = None) -> list[Notification]:
        """Put a notification in each recipient's inbox and queue push/email
        per their preferences and quiet hours. A repeated `dedupe_key` for a
        user is ignored, so a retried request never notifies twice."""
        if kind not in CATALOGUE:
            raise ValueError(f"unknown notification kind {kind!r}")
        from ..curriculum.store import new_id
        now = now or datetime.now(timezone.utc)
        out = []
        with self._conn_lock:
            for uid in dict.fromkeys(user_ids):
                if not uid:
                    continue
                s = self.settings_for(uid)
                title, body = render(kind, params, s["language"])
                quiet = in_quiet_hours(now.astimezone(IST), s["quiet_start"], s["quiet_end"])
                held = "digest" if s.get("digest_at") and kind not in DIGEST_EXEMPT else None
                status = {ch: (held or ("deferred" if quiet else "pending")) if self.channel_enabled(uid, kind, ch)
                          else "none" for ch in CHANNELS}
                nid = new_id("ntf")
                try:
                    self._exec("INSERT INTO notifications (id, school_id, user_id, kind, title, body, link, data_json, "
                               "created_at, push_status, email_status, dedupe_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (nid, school_id, uid, kind, title, body, link, json.dumps(params, default=str),
                                now.isoformat(), status["push"], status["email"], dedupe_key))
                except Exception as exc:  # noqa: BLE001 - only the UNIQUE(user, dedupe) is expected here
                    if "UNIQUE" in str(exc):
                        continue
                    raise
                # SMS and WhatsApp, for those who consented (INT-2, messaging.py).
                self._queue_messages(nid, uid, kind, quiet=quiet, on_digest=bool(s.get("digest_at")))
                out.append(nid)
        self._commit()
        if self.deliver_inline:
            self.deliver_due(now=now)
        else:
            import threading
            threading.Thread(target=self._deliver_quietly, daemon=True).start()
        return [self.get_notification(i) for i in out]

    def _deliver_quietly(self) -> None:
        try:
            self.deliver_due()
        except Exception:  # noqa: BLE001 - a background send never breaks a request
            logger.warning("notification delivery failed", exc_info=True)

    def deliver_due(self, now: Optional[datetime] = None) -> int:
        """Send every pending item, every deferred item whose quiet hours
        have ended, and retry failures (up to MAX_ATTEMPTS).

        One pass at a time per store. A pass reads the due rows and writes
        each status only after sending, so two passes at once (every notify
        starts one, and so do the inbox and the scheduler) sent everything
        twice (N-8-1). A call that finds a pass running asks it to go round
        once more instead: it raises the flag before trying the lock, and the
        running pass reads the flag only after releasing it, so a row queued
        meanwhile is never left for a later call."""
        lock = self._delivery_lock()
        sent = 0
        self._deliver_again = True
        while self._deliver_again:
            if not lock.acquire(blocking=False):
                return sent
            try:
                self._deliver_again = False
                sent += self._deliver_pass(now or datetime.now(timezone.utc))
            finally:
                lock.release()
        return sent

    def _deliver_pass(self, now: datetime) -> int:
        sent = 0
        rows = self._fetchall("SELECT * FROM notifications WHERE (push_status IN ('pending','deferred','failed') "
                              "OR email_status IN ('pending','deferred','failed')) AND attempts < ?",
                              (MAX_ATTEMPTS,))
        for r in rows:
            n = self._notif(r)
            s = self.settings_for(n.user_id)
            if in_quiet_hours(now.astimezone(IST), s["quiet_start"], s["quiet_end"]):
                continue
            updates, errors = {}, []
            if n.push_status in ("pending", "deferred", "failed"):
                tokens = self.devices_for(n.user_id)
                status, err, *dead = send_push(tokens, n.title, n.body,
                                               {"kind": n.kind, "link": n.link or "", "id": n.id})
                self._note_dead_tokens(tokens, dead[0] if dead else [], status, err)
                updates["push_status"] = status
                if err:
                    errors.append(f"push: {err}")
            if n.email_status in ("pending", "deferred", "failed"):
                email = self.email_for(n.user_id) if self.email_for else None
                status, err = send_email(email, n.title, n.body)
                updates["email_status"] = status
                if err:
                    errors.append(f"email: {err}")
            failed = "failed" in updates.values()
            self._exec("UPDATE notifications SET push_status=COALESCE(?, push_status), "
                       "email_status=COALESCE(?, email_status), attempts=attempts+?, last_error=? WHERE id=?",
                       (updates.get("push_status"), updates.get("email_status"), 1 if failed else 0,
                        "; ".join(errors) or None, n.id))
            sent += sum(1 for v in updates.values() if v == "sent")
        sent += self._deliver_messages(now)
        self._commit()
        return sent

    def send_digests(self, now: Optional[datetime] = None) -> int:
        """For each person on a daily digest whose time has come today and
        who has not had today's: one push and one email listing what was
        held for them (SA-5). Returns how many digests went."""
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(IST)
        today, hhmm = local.date().isoformat(), local.strftime("%H:%M")
        sent = 0
        for d in self._fetchall("SELECT * FROM notification_digests WHERE digest_at<=? "
                                "AND (last_sent_on IS NULL OR last_sent_on<?)", (hhmm, today)):
            uid = d["user_id"]
            held = self._fetchall("SELECT * FROM notifications WHERE user_id=? AND (push_status='digest' "
                                  "OR email_status='digest') ORDER BY created_at", (uid,))
            if held:
                lang = self.settings_for(uid)["language"]
                count = len(held)
                title = (f"आज: {count} अपडेट" if lang == "hi"
                         else f"Today: {count} update" + ("" if count == 1 else "s"))
                short = "; ".join(r["title"] for r in held[:3]) + (" …" if count > 3 else "")
                if any(r["push_status"] == "digest" for r in held):
                    send_push(self.devices_for(uid), title, short, {"kind": "digest", "link": "/inbox", "id": ""})
                if any(r["email_status"] == "digest" for r in held):
                    email = self.email_for(uid) if self.email_for else None
                    send_email(email, title, "\n\n".join(r["title"] + "\n" + r["body"] for r in held))
                with self._conn_lock:
                    self.conn.execute("UPDATE notifications SET push_status=CASE WHEN push_status='digest' "
                                      "THEN 'digested' ELSE push_status END, email_status=CASE WHEN "
                                      "email_status='digest' THEN 'digested' ELSE email_status END WHERE user_id=?",
                                      (uid,))
                sent += 1
            self._exec("UPDATE notification_digests SET last_sent_on=? WHERE user_id=?", (today, uid))
        self._commit()
        return sent

    # ---------------- the inbox ----------------

    def get_notification(self, nid: str) -> Optional[Notification]:
        r = self._fetchone("SELECT * FROM notifications WHERE id=?", (nid,))
        return self._notif(r) if r else None

    def inbox(self, user_id: str, *, unread_only: bool = False, limit: int = 50) -> list[Notification]:
        sql = "SELECT * FROM notifications WHERE user_id=?" + (" AND read_at IS NULL" if unread_only else "")
        return [self._notif(r) for r in self._fetchall(sql + " ORDER BY created_at DESC LIMIT ?",
                                                        (user_id, int(limit)))]

    def unread_count(self, user_id: str) -> int:
        r = self._fetchone("SELECT COUNT(*) AS n FROM notifications WHERE user_id=? AND read_at IS NULL", (user_id,))
        return int(r["n"]) if r else 0

    def mark_read(self, user_id: str, nid: Optional[str] = None) -> int:
        with self._conn_lock:
            if nid is None:
                cur = self.conn.execute("UPDATE notifications SET read_at=? WHERE user_id=? AND read_at IS NULL",
                                        (_now_iso(), user_id))
            else:
                cur = self.conn.execute("UPDATE notifications SET read_at=? WHERE id=? AND user_id=? AND read_at IS NULL",
                                        (_now_iso(), nid, user_id))
            n = cur.rowcount
        self._commit()
        return n
