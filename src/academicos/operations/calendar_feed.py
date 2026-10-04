"""A private calendar feed (INT-2): one ICS URL per user for Google
Calendar, Outlook or a phone calendar -- the week's periods, substitution and
invigilation duties, exam papers, homework due dates and holidays.

A calendar app cannot send a login, so the URL carries a secret token of its
own; it is shown only to its owner and can be replaced at any time, which
kills the old URL. The feed holds only the owner's own timetable and dates:
no marks, no other student's data.
"""
from __future__ import annotations

import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from ..assessment.auth_routes import get_current_user
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel

router = APIRouter(prefix="/api/v1")


def store():
    """The operations store, looked up when called: operations/store.py
    imports this module for its table mixin, so importing routes here at load
    time would be circular."""
    from .routes import store as operations_store
    return operations_store()

FEEDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_feeds (
    user_id TEXT PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""


class FeedsMixin:
    """Added to OperationsStore."""

    def feed_token(self, user_id: str, *, reset: bool = False) -> str:
        row = self._fetchone("SELECT token FROM calendar_feeds WHERE user_id=?", (user_id,))
        if row and not reset:
            return row["token"]
        token = secrets.token_urlsafe(24)
        with self._conn_lock:
            self.conn.execute("INSERT INTO calendar_feeds (user_id, token, created_at) VALUES (?,?,?) ON CONFLICT(user_id)"
                              " DO UPDATE SET token=excluded.token, created_at=excluded.created_at",
                              (user_id, token, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self._commit()
        return token

    def feed_owner(self, token: str) -> Optional[str]:
        row = self._fetchone("SELECT user_id FROM calendar_feeds WHERE token=?", (token,))
        return row["user_id"] if row else None


class FeedResponse(Camel):
    url: str


def _feed_url(request: Request, token: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/v1/ics/{token}.ics"


@router.get("/calendar-feed", response_model=FeedResponse)
def calendar_feed(request: Request, current: User = Depends(get_current_user)) -> FeedResponse:
    """The caller's private calendar URL (made on first ask)."""
    if current.role not in ("teacher", "principal", "student"):
        raise HTTPException(403, "calendar feeds are for staff and students")
    return FeedResponse(url=_feed_url(request, store().feed_token(current.id)))


@router.post("/calendar-feed/reset", response_model=FeedResponse)
def reset_calendar_feed(request: Request, current: User = Depends(get_current_user)) -> FeedResponse:
    """A new URL; the old one stops working at once."""
    if current.role not in ("teacher", "principal", "student"):
        raise HTTPException(403, "calendar feeds are for staff and students")
    return FeedResponse(url=_feed_url(request, store().feed_token(current.id, reset=True)))


# ---------------- ICS ----------------

def _esc(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> list[str]:
    """RFC 5545 folding: lines of at most 75 octets, continued with a space."""
    out, cur = [], ""
    for ch in line:
        if len((cur + ch).encode("utf-8")) > 75:
            out.append(cur)
            cur = " " + ch
        else:
            cur += ch
    out.append(cur)
    return out


def _stamp(d: date, hhmm: str) -> str:
    return d.strftime("%Y%m%d") + "T" + hhmm.replace(":", "") + "00"


class _Cal:
    def __init__(self, name: str):
        self.lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AcademicOS//School calendar//EN",
                      "CALSCALE:GREGORIAN", f"X-WR-CALNAME:{_esc(name)}", "X-WR-TIMEZONE:Asia/Kolkata",
                      "BEGIN:VTIMEZONE", "TZID:Asia/Kolkata", "BEGIN:STANDARD", "DTSTART:19700101T000000",
                      "TZOFFSETFROM:+0530", "TZOFFSETTO:+0530", "TZNAME:IST", "END:STANDARD", "END:VTIMEZONE"]
        self.now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def event(self, uid: str, summary: str, *, day: date, start: Optional[str] = None, end: Optional[str] = None,
              until: Optional[date] = None, end_day: Optional[date] = None, description: str = "",
              except_days: tuple[date, ...] = ()) -> None:
        ev = ["BEGIN:VEVENT", f"UID:{uid}@academicos", f"DTSTAMP:{self.now}", f"SUMMARY:{_esc(summary)}"]
        if start and end:
            ev += [f"DTSTART;TZID=Asia/Kolkata:{_stamp(day, start)}", f"DTEND;TZID=Asia/Kolkata:{_stamp(day, end)}"]
        else:
            ev += [f"DTSTART;VALUE=DATE:{day.strftime('%Y%m%d')}",
                   f"DTEND;VALUE=DATE:{((end_day or day) + timedelta(days=1)).strftime('%Y%m%d')}"]
        if until is not None:
            ev.append(f"RRULE:FREQ=WEEKLY;UNTIL={until.strftime('%Y%m%d')}T235959Z")
        if except_days and start:
            ev.append("EXDATE;TZID=Asia/Kolkata:" + ",".join(_stamp(d, start) for d in except_days))
        if description:
            ev.append(f"DESCRIPTION:{_esc(description)}")
        ev.append("END:VEVENT")
        self.lines += ev

    def text(self) -> str:
        out = []
        for line in self.lines + ["END:VCALENDAR"]:
            out += _fold(line)
        return "\r\n".join(out) + "\r\n"


def _year(cs, school_id: str, today: date):
    years = [y for y in cs.academic_years_for_school(school_id)]
    return next((y for y in years if y.start_date <= today.isoformat() <= y.end_date),
                max(years, key=lambda y: y.start_date) if years else None)


@dataclass
class CalEvent:
    """One entry of a user's school calendar. `until` makes it weekly to that
    day (a timetabled period), except on `except_days` (a holiday, a half
    day's cut periods, an exam window: the period is not held then);
    `end_day` spans whole days (a long holiday)."""
    uid: str
    # period | holiday | half_day | exam_window | working_day | cover | exam | invigilation | homework
    kind: str
    title: str
    day: date
    start: Optional[str] = None
    end: Optional[str] = None
    until: Optional[date] = None
    end_day: Optional[date] = None
    except_days: tuple[date, ...] = ()


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _calendar_day_title(h) -> str:
    """What a calendar row says in a list (N-3-19: the apps showed none)."""
    from ..curriculum.calendar import EXAM_WINDOW, HALF_DAY, WORKING_DAY
    if h.kind == HALF_DAY:
        return f"Half day: {h.label} (periods 1-{h.last_period})"
    if h.kind == EXAM_WINDOW:
        classes = f" (class {', '.join(map(str, h.grades))})" if h.grades else ""
        return f"Exam window: {h.label}{classes}"
    if h.kind == WORKING_DAY and h.timetable_weekday is not None:
        return f"Working day: {h.label} ({_WEEKDAYS[h.timetable_weekday]}'s timetable)"
    return h.label


def calendar_events(user: User, today: date) -> list[CalEvent]:
    """Everything on the user's own school calendar: the ICS feed and the
    apps' "Coming up" list read the same events (TA-2, SA-1)."""
    cs = cr._require()
    ops = store()
    out: list[CalEvent] = []
    year = _year(cs, user.school_id, today)
    if year is None:
        return out
    ystart, yend = date.fromisoformat(year.start_date), date.fromisoformat(year.end_date)
    first_day = max(ystart, today - timedelta(days=today.weekday()))
    subjects: dict[str, str] = {}

    def subject(sid: str) -> str:
        if sid not in subjects:
            s = cs.get_subject(sid)
            subjects[sid] = s.name if s else "Class"
        return subjects[sid]

    def label(section_id: str) -> str:
        s = cs.get_section(section_id)
        return cs._section_label(s) if s else ""

    # the week, as weekly recurring periods at the bell's times, except on the
    # days a period is not held; a compensatory working day holds its
    # weekday's periods once (N-3-19)
    days = cs._school_days(year.id)
    grades: dict[str, Optional[int]] = {}

    def grade_of(sec) -> Optional[int]:
        if sec.id not in grades:
            g = cs.get_grade(sec.grade_id)
            grades[sec.id] = g.number if g else None
        return grades[sec.id]

    section = None
    if user.role == "student":
        e = cs.enrollment_for_student(user.id)
        section = cs.get_section(e.section_id) if e and e.section_id else None
        entries = cs.timetable_for_section(section.id) if section else []
    else:
        entries = cs.timetable_for_teacher(user.id, year.id)
    for t in entries:
        sec = section or cs.get_section(t.section_id)
        bell = cs.bell_for_section(sec) if sec else None
        # A bell's slots include breaks; a teaching slot carries its period number.
        slot = next((s for s in (bell.slots if bell else []) if getattr(s, "period", None) == t.period), None)
        if slot is None:
            continue
        start, end = slot.start, slot.end
        first = first_day + timedelta(days=(t.day_of_week - first_day.weekday()) % 7)
        what = subject(t.subject_id) if user.role == "student" else f"{subject(t.subject_id)} {label(t.section_id)}"
        g = grade_of(sec)
        weeks = [first + timedelta(days=7 * i) for i in range((yend - first).days // 7 + 1)]
        out.append(CalEvent(f"tt-{t.id}", "period", what, first, start, end, until=yend,
                            except_days=tuple(d for d in weeks if not days.holds(d.isoformat(), t.day_of_week,
                                                                                    t.period, g))))
        for on, weekday in sorted(days.runs_as.items()):
            if weekday == t.day_of_week and first_day.isoformat() <= on <= year.end_date \
                    and days.holds(on, weekday, t.period, g):
                out.append(CalEvent(f"tt-{t.id}-{on}", "period", what, date.fromisoformat(on), start, end))

    # holidays, and the days that run differently (N-3-19); an exam window
    # only for the classes it stops (all of them for the principal)
    from ..curriculum.calendar import DAY_KINDS, EXAM_WINDOW
    if user.role == "student":
        my_grades = {grade_of(section)} if section else set()
    else:
        my_grades = {grade_of(s) for s in (cs.get_section(sid) for sid in {t.section_id for t in entries}) if s}
    cal_row = cs.get_calendar_for_year(year.id)
    for h in (cs.holidays_for_calendar(cal_row.id) if cal_row else []):
        d = date.fromisoformat(h.date)
        if (date.fromisoformat(h.end_date) if h.end_date else d) < today - timedelta(days=7):
            continue
        if h.kind == EXAM_WINDOW and h.grades and user.role != "principal" and not my_grades & set(h.grades):
            continue
        out.append(CalEvent(f"hol-{h.id}", h.kind if h.kind in DAY_KINDS else "holiday", _calendar_day_title(h), d,
                            end_day=date.fromisoformat(h.end_date) if h.end_date else None))

    # substitution duties (staff)
    if user.role != "student":
        for s in cs.duties_for(user.id, today.isoformat(), (today + timedelta(days=60)).isoformat()):
            out.append(CalEvent(f"sub-{s.id}", "cover",
                                f"Cover: {label(s.section_id)} {subject(s.subject_id)} (period {s.period})",
                                date.fromisoformat(s.date)))

    # exams and homework
    for ex in ops.exams_for_school(user.school_id):
        if ex["status"] != "published":
            continue
        if user.role == "student" and section is not None:
            grade = cs.get_grade(section.grade_id).number
            for p in ops.exam_papers(ex["id"]):
                if p["grade"] == grade:
                    out.append(CalEvent(f"exam-{p['id']}", "exam", f"Exam: {p['subject_name']} ({ex['name']})",
                                        date.fromisoformat(p["date"]), p["start_time"], p["end_time"]))
        elif user.role != "student":
            for r in ops.roster(ex["id"]):
                if r["teacher_id"] == user.id:
                    out.append(CalEvent(f"inv-{r['exam_paper_id']}-{r['section_id']}", "invigilation",
                                        f"Invigilation: {label(r['section_id'])} {r['subject_name']}",
                                        date.fromisoformat(r["date"]), r["start_time"], r["end_time"]))
    if user.role == "student" and section is not None:
        for hw in ops.homework_for_section(section.id):
            if hw.status == "published" and hw.due_date >= (today - timedelta(days=7)).isoformat():
                out.append(CalEvent(f"hw-{hw.id}", "homework", f"Homework due: {hw.title} ({hw.subject_name})",
                                    date.fromisoformat(hw.due_date)))
    elif user.role != "student":
        for hw in ops.homework_for_school(user.school_id, status="published"):
            if hw.teacher_id == user.id and hw.due_date >= (today - timedelta(days=7)).isoformat():
                out.append(CalEvent(f"hw-{hw.id}", "homework", f"Homework due: {hw.title}",
                                    date.fromisoformat(hw.due_date)))
    return out


def build_calendar(user: User, today: date) -> str:
    cal = _Cal(f"School - {user.name}")
    for e in calendar_events(user, today):
        cal.event(e.uid, e.title, day=e.day, start=e.start, end=e.end, until=e.until, end_day=e.end_day,
                  except_days=e.except_days)
    return cal.text()


# ---------------- in the apps ----------------

class CalendarItem(Camel):
    kind: str
    title: str
    date: str
    end_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@router.get("/my-calendar", response_model=list[CalendarItem])
def my_calendar(days: int = Query(42, ge=1, le=60),
                current: User = Depends(get_current_user)) -> list[CalendarItem]:
    """The caller's dated school events from today: holidays, half days,
    exam windows and compensatory working days (N-3-19), cover and
    invigilation duties, exam papers and homework due. The ICS feed carries
    the same; the weekly periods are the timetable's, so they are left out."""
    if current.role not in ("teacher", "principal", "student"):
        raise HTTPException(403, "the school calendar is for staff and students; a parent opens their child's")
    return calendar_items(current, days)


def calendar_items(user: User, days: int) -> list[CalendarItem]:
    """`user`'s dated school events for the next `days` days: what
    /my-calendar shows them, and what a parent sees for their child
    (guardian_routes, SA-6)."""
    today = cr._school_today()
    last = today + timedelta(days=days)
    items = [e for e in calendar_events(user, today)
             if e.kind != "period" and (e.end_day or e.day) >= today and e.day <= last]
    items.sort(key=lambda e: (e.day, e.start or "", e.title))
    return [CalendarItem(kind=e.kind, title=e.title, date=e.day.isoformat(),
                         end_date=e.end_day.isoformat() if e.end_day else None,
                         start_time=e.start, end_time=e.end) for e in items]


class LoadRow(Camel):
    section_name: str
    subject_name: str
    periods_per_week: int


class MyLoadResponse(Camel):
    periods_per_week: int = 0
    max_in_one_day: int = 0
    # periods on each weekday, 0 = Monday
    days_taught: dict[int, int] = {}
    classes: list[LoadRow] = []
    covers_this_week: int = 0
    invigilations_ahead: int = 0


@router.get("/my-load", response_model=MyLoadResponse)
def my_load(current: User = Depends(get_current_user)) -> MyLoadResponse:
    """A teacher's own load (TA-2): timetabled periods a week by class and
    subject, the busiest day, cover duties this week and exam duties ahead.
    Only the caller's own numbers; the whole staff's load stays the
    principal's (ADM-5)."""
    if current.role not in ("teacher", "principal"):
        raise HTTPException(403, "a teaching load is for staff")
    cs = cr._require()
    today = cr._school_today()
    year = _year(cs, current.school_id, today)
    if year is None:
        return MyLoadResponse()
    entries = cs.timetable_for_teacher(current.id, year.id)
    per_day = Counter(e.day_of_week for e in entries)
    per_class = Counter((e.section_id, e.subject_id) for e in entries)

    def section_name(sid: str) -> str:
        s = cs.get_section(sid)
        return cs._section_label(s) if s else ""

    def subject_name(sid: str) -> str:
        s = cs.get_subject(sid)
        return s.name if s else ""

    monday = today - timedelta(days=today.weekday())
    covers = cs.duties_for(current.id, monday.isoformat(), (monday + timedelta(days=6)).isoformat())
    ops = store()
    ahead = sum(1 for ex in ops.exams_for_school(current.school_id) if ex["status"] == "published"
                for r in ops.roster(ex["id"]) if r["teacher_id"] == current.id and r["date"] >= today.isoformat())
    classes = [LoadRow(section_name=section_name(sec), subject_name=subject_name(sub), periods_per_week=n)
               for (sec, sub), n in per_class.items()]
    classes.sort(key=lambda r: (-r.periods_per_week, r.section_name, r.subject_name))
    return MyLoadResponse(periods_per_week=len(entries), max_in_one_day=max(per_day.values(), default=0),
                          days_taught=dict(sorted(per_day.items())), classes=classes,
                          covers_this_week=len(covers),
                          invigilations_ahead=ahead)


@router.get("/ics/{token}.ics")
def ics(token: str) -> Response:
    """The feed itself, for calendar apps. The token is the only key: an
    unknown or replaced token is a plain 404."""
    from .routes import _cfg
    if _cfg is None:
        raise HTTPException(503, "the calendar feed is not initialised")
    user_id = store().feed_owner(token)
    user = cr._require_users().get(user_id) if user_id else None
    if user is None:
        raise HTTPException(404, "Not Found")
    body = build_calendar(user, cr._school_today())
    return Response(content=body, media_type="text/calendar; charset=utf-8",
                    headers={"Cache-Control": "private, max-age=900"})
