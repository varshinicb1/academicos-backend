"""Scheduled automations (NTF-2): the reminders and digests nobody has to
remember to send.

Each job has a window on the school's clock (IST) and a period (a day or a
week). A run does every job whose window has opened and whose period has not
been done, and records it, so running every five minutes -- or on every
inbox poll -- sends each thing once. Notifications carry dedupe keys too, so
even two overlapping runs cannot double a message.

- homework_due_tomorrow (from 16:00): students who have not submitted
  homework due tomorrow.
- homework_overdue (from 09:00): the teacher of each homework that closed
  yesterday with students missing, once per homework.
- substitution_escalation (every run): the principal, when a substitution
  for today or tomorrow has no substitute, or its proposed substitute has not
  accepted within ESCALATE_AFTER_MINUTES.
- teacher_daily_digest (from 18:00): each teacher's periods and duties for
  tomorrow, when they have any.
- principal_weekly_digest (Monday from 07:00): last week's cover and homework.
- marking_due (from 15:00): teachers with submissions waiting over two days
  for their marks (TA-7).
- principal_alerts (from 08:00): sections behind plan, lost periods owed over a
  week, homework overdue in bulk, exams within a week without a full roster or
  with a paper not set (no approved question paper) (ADM-3).
- exam_reminders (from 17:00): students with a paper tomorrow, and invigilators
  with duties tomorrow, in a published exam (NTF-2, SA-5).
- term_report (from 15:00): on a term's last day, its principal hears the term
  report is ready (NTF-2 end-of-term reports, ADM-4).
- user_digests (every run): each person on a daily digest gets it once their
  chosen time has passed (SA-5).
- deliveries (every run): what quiet hours held overnight, and failed sends
  due a retry, go out without waiting for someone to open an inbox (N-8-2).

Triggered by POST /api/v1/automations/run (Cloud Scheduler, with the
operator's cron key) and, throttled, by ordinary traffic.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))
ESCALATE_AFTER_MINUTES = 30

AUTOMATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS automation_runs (
    job TEXT NOT NULL,
    period_key TEXT NOT NULL,
    ran_at TEXT NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, period_key)
);
-- What one run did for one school: a job runs for every school at once, and
-- a school's principal is shown only their own school's share (N-9-1).
CREATE TABLE IF NOT EXISTS automation_school_runs (
    job TEXT NOT NULL,
    period_key TEXT NOT NULL,
    school_id TEXT NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, period_key, school_id)
);
"""


class AutomationsMixin:
    """Added to OperationsStore."""

    def automation_done(self, job: str, period_key: str) -> bool:
        return self._fetchone("SELECT 1 AS x FROM automation_runs WHERE job=? AND period_key=?",
                              (job, period_key)) is not None

    def record_automation(self, job: str, period_key: str, notified: int,
                          per_school: Optional[dict[str, int]] = None) -> None:
        with self._conn_lock:
            self.conn.execute("INSERT OR REPLACE INTO automation_runs (job, period_key, ran_at, notified)"
                              " VALUES (?,?,?,?)",
                              (job, period_key, datetime.now(timezone.utc).isoformat(timespec="seconds"), notified))
            for school_id, n in (per_school or {}).items():
                self.conn.execute("INSERT OR REPLACE INTO automation_school_runs (job, period_key, school_id,"
                                  " notified) VALUES (?,?,?,?)", (job, period_key, school_id, n))
            self._commit()

    def automation_history(self, school_id: str, limit: int = 50) -> list[dict]:
        """The latest runs, with how many notifications each made for this
        school only; the runs themselves are the same for every school."""
        return self._fetchall(
            "SELECT r.job, r.period_key, r.ran_at, COALESCE(s.notified, 0) AS notified "
            "FROM automation_runs r LEFT JOIN automation_school_runs s "
            "ON s.job = r.job AND s.period_key = r.period_key AND s.school_id = ? "
            "ORDER BY r.ran_at DESC LIMIT ?", (school_id, limit))


def _schools(cs) -> list[str]:
    return [r["school_id"] for r in cs._fetchall("SELECT DISTINCT school_id FROM academic_years")]


def _year_on(cs, school_id: str, on: str):
    return next((y for y in cs.academic_years_for_school(school_id) if y.start_date <= on <= y.end_date), None)


def _label(cs, section_id: str) -> str:
    s = cs.get_section(section_id)
    return cs._section_label(s) if s else section_id


def _subject(cs, subject_id: str) -> str:
    s = cs.get_subject(subject_id)
    return s.name if s else subject_id


# ---------------- jobs: each returns how many were notified ----------------

def homework_due_tomorrow(ops, cs, users, notify, now: datetime) -> int:
    tomorrow = (now.astimezone(IST).date() + timedelta(days=1)).isoformat()
    n = 0
    for school in _schools(cs):
        for hw in ops.homework_for_school(school, status="published"):
            if hw.due_date != tomorrow:
                continue
            done = {s.student_id for s in ops.submissions_for(hw.id)}
            for sid in hw.section_ids:
                for e in cs.enrollments_for_section(sid):
                    if e.student_id in done:
                        continue
                    notify(school_id=school, user_ids=[e.student_id], kind="homework_due",
                           params={"title": hw.title, "due": "tomorrow", "due_hi": "कल"}, link=f"/my-homework/{hw.id}",
                           dedupe_key=f"hwdue:{hw.id}:{e.student_id}:{tomorrow}")
                    n += 1
    return n


def homework_overdue(ops, cs, users, notify, now: datetime) -> int:
    today = now.astimezone(IST).date().isoformat()
    n = 0
    for school in _schools(cs):
        for hw in ops.homework_for_school(school, status="published"):
            if hw.due_date >= today:
                continue
            done = {s.student_id for s in ops.submissions_for(hw.id)}
            missing = sum(1 for sid in hw.section_ids for e in cs.enrollments_for_section(sid)
                          if e.student_id not in done)
            if not missing:
                continue
            notify(school_id=school, user_ids=[hw.teacher_id], kind="homework_overdue",
                   params={"title": hw.title, "count": missing, "due": hw.due_date},
                   link=f"/homework/{hw.id}", dedupe_key=f"hwoverdue:{hw.id}")
            n += 1
    return n


def substitution_escalation(ops, cs, users, notify, now: datetime) -> int:
    today = now.astimezone(IST).date()
    horizon = (today + timedelta(days=1)).isoformat()
    cutoff = (now - timedelta(minutes=ESCALATE_AFTER_MINUTES)).isoformat()
    n = 0
    for school in _schools(cs):
        principals = [u.id for u in users.users_for_school(school, role="principal")]
        year = _year_on(cs, school, today.isoformat())
        if not principals or year is None:
            continue
        for s in cs.substitutions_between(year.id, today.isoformat(), horizon):
            stuck = s.status == "open" or (s.status in ("proposed", "declined") and s.updated_at <= cutoff)
            if not stuck:
                continue
            unfilled = s.status in ("open", "declined")
            what = "has no substitute" if unfilled else "is not accepted yet"
            label, subject = _label(cs, s.section_id), _subject(cs, s.subject_id)
            notify(school_id=school, user_ids=principals, kind="principal_alert",
                   params={"title": f"Cover needed: {label} period {s.period}",
                           "body": f"{subject} on {s.date} {what}.",
                           "title_hi": f"स्थानापन्न चाहिए: {label} पीरियड {s.period}",
                           "body_hi": (f"{s.date} को {subject} के लिए कोई स्थानापन्न नहीं है।" if unfilled
                                       else f"{s.date} को {subject} का स्थानापन्न अभी तक स्वीकार नहीं हुआ है।")},
                   link=f"/cover?date={s.date}", dedupe_key=f"subesc:{s.id}:{s.status}")
            n += 1
    return n


def staff_check_in(ops, cs, users, notify, now: datetime) -> int:
    """09:00 on a school day (staff_policy.py, check_in_reminder): a teacher
    who has not checked in and is not on leave is reminded, and the principal
    hears who they are, once. Nobody is marked absent by the machine: the
    register is the principal's."""
    from ..curriculum import calendar as calendar_mod
    today = now.astimezone(IST).date().isoformat()
    n = 0
    for school in _schools(cs):
        if not cs.staff_policy(school).check_in_reminder:
            continue
        year = _year_on(cs, school, today)
        if year is None:
            continue
        try:
            if today not in set(calendar_mod.school_days(cs, year.id).dates):
                continue                       # a weekly off, a holiday or a closure
        except ValueError:
            continue                           # no calendar yet: no school day to expect anyone on
        marks = cs.attendance_for_date(school, today)
        missing = []
        for u in users.users_for_school(school, role="teacher"):
            if u.id in marks:
                continue
            if any(l.status == "approved" and l.start_date <= today <= l.end_date for l in cs.leave_for_teacher(u.id)):
                continue
            missing.append(u)
        if not missing:
            continue
        for u in missing:
            notify(school_id=school, user_ids=[u.id], kind="check_in_reminder", params={},
                   link="/school/my-day", dedupe_key=f"checkin:{u.id}:{today}")
            n += 1
        principals = [u.id for u in users.users_for_school(school, role="principal")]
        names = sorted(u.name for u in missing)
        shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
        notify(school_id=school, user_ids=principals, kind="staff_not_checked_in",
               params={"count": f"{len(missing)} teacher{'s' if len(missing) != 1 else ''}", "names": shown},
               link="/school/cover", dedupe_key=f"notcheckedin:{school}:{today}")
        n += len(principals)
    return n


def teacher_daily_digest(ops, cs, users, notify, now: datetime) -> int:
    tomorrow = (now.astimezone(IST).date() + timedelta(days=1)).isoformat()
    n = 0
    for school in _schools(cs):
        year = _year_on(cs, school, tomorrow)
        if year is None:
            continue
        for u in users.users_for_school(school, role="teacher"):
            rows = [r for r in cs.day_view(year.id, tomorrow, teacher_id=u.id) if r["kind"] != "away"]
            if not rows:
                continue
            kinds = Counter(r["kind"] for r in rows)
            duties = kinds["substitute"] + kinds["supervised"] + kinds["combined"]
            first = min(rows, key=lambda r: r["period"])
            notify(school_id=school, user_ids=[u.id], kind="daily_digest",
                   params={"periods": len(rows) - duties, "duties": duties,
                           "first": f"period {first['period']}, {_label(cs, first['sectionId'])} "
                                    f"{_subject(cs, first['subjectId'])}",
                           "first_hi": f"पीरियड {first['period']}, {_label(cs, first['sectionId'])} "
                                       f"{_subject(cs, first['subjectId'])}"},
                   link=f"/my-day?date={tomorrow}", dedupe_key=f"digest:{u.id}:{tomorrow}")
            n += 1
    return n


PERIOD_NOTICE_MINUTES = 10


def period_starting(ops, cs, users, notify, now: datetime) -> int:
    """Each teacher is told of a period (their own or a cover) starting in
    the next ten minutes, once per period. Automations run every few minutes
    on the school day's traffic (routes._maybe_run_automations)."""
    local = now.astimezone(IST)
    today = local.date().isoformat()
    soon = (local + timedelta(minutes=PERIOD_NOTICE_MINUTES)).strftime("%H:%M")
    at = local.strftime("%H:%M")
    n = 0
    for school in _schools(cs):
        year = _year_on(cs, school, today)
        if year is None:
            continue
        for r in cs.day_view(year.id, today):
            teacher = r.get("teacherId")
            if not teacher or r["kind"] in ("away", "lost", "uncovered", "combined"):
                continue
            section = cs.get_section(r["sectionId"])
            bell = cs.bell_for_section(section) if section else None
            start = next((sl.start for sl in (bell.slots if bell else [])
                          if sl.kind == "teaching" and sl.period == r["period"]), None)
            if start is None or not (at <= start <= soon):
                continue
            room = cs.get_room(r["roomId"]) if r.get("roomId") else None
            notify(school_id=school, user_ids=[teacher], kind="period_starting",
                   params={"period": r["period"], "start": start, "section": _label(cs, r["sectionId"]),
                           "subject": _subject(cs, r["subjectId"]),
                           "room": f", {room.name}" if room else ""},
                   link=f"/my-day?date={today}", dedupe_key=f"period:{teacher}:{today}:{r['period']}")
            n += 1
    return n


def principal_weekly_digest(ops, cs, users, notify, now: datetime) -> int:
    today = now.astimezone(IST).date()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)
    n = 0
    for school in _schools(cs):
        principals = [u.id for u in users.users_for_school(school, role="principal")]
        year = _year_on(cs, school, last_sunday.isoformat()) or _year_on(cs, school, today.isoformat())
        if not principals or year is None:
            continue
        cover = cs.cover_summary(year.id, last_monday.isoformat(), last_sunday.isoformat())
        set_ = [hw for hw in ops.homework_for_school(school)
                if hw.published_at and last_monday.isoformat() <= hw.published_at[:10] <= last_sunday.isoformat()]
        notify(school_id=school, user_ids=principals, kind="weekly_digest",
               params={"week": last_monday.isoformat(), "substitutions": cover["substitutionsRequested"],
                       "unfilled": cover["unfilled"], "lost": cover["lost"], "homework": len(set_)},
               link="/dashboard", dedupe_key=f"weekly:{school}:{last_monday.isoformat()}")
        n += 1
    return n


MARKING_AFTER_DAYS = 2      # submissions waiting this long remind their teacher


def marking_due(ops, cs, users, notify, now: datetime) -> int:
    """TA-7, once an afternoon: each teacher with homework submissions
    waiting over MARKING_AFTER_DAYS days for their marks hears how many."""
    cutoff = (now - timedelta(days=MARKING_AFTER_DAYS)).isoformat()
    today = now.astimezone(IST).date().isoformat()
    n = 0
    for school in _schools(cs):
        waiting: dict[str, list[tuple[str, str]]] = {}
        for hw in ops.homework_for_school(school):
            if hw.status == "draft":
                continue
            for sub in ops.submissions_for(hw.id):
                if sub.status == "submitted" and sub.submitted_at <= cutoff:
                    waiting.setdefault(hw.teacher_id, []).append((sub.submitted_at, hw.title))
        for teacher, items in waiting.items():
            items.sort()
            notify(school_id=school, user_ids=[teacher], kind="marking_due",
                   params={"count": len(items), "days": MARKING_AFTER_DAYS, "title": items[0][1]},
                   link="/homework", dedupe_key=f"marking:{teacher}:{today}")
            n += 1
    return n


BEHIND_LESSONS = 5          # a section this many lessons behind its plan is "falling behind"
OWED_AFTER_DAYS = 7         # a lost period owed this long is flagged
# A generated question paper the principal has approved, or one already past
# approval (schemas.AssessmentStatus); "archived" is not a paper in use.
APPROVED_PAPER = frozenset({"principalApproved", "printed", "conducted", "scanning", "scanned", "evaluating",
                            "evaluated", "teacherReviewed", "reportsGenerated", "remediationSent"})


def _papers_not_set(ops, cs, ex: dict, papers: list[dict], today: str) -> Optional[list[tuple[int, str]]]:
    """N-67-9: the datesheet papers from today on with no approved question
    paper, as (class, subject). An exam paper row carries no link to a
    generated paper, so one is matched on what both record: the school, the
    class, the subject (without case, as paper_timing does), and when it was
    made (its scheduled date if set, else its creation) -- after the previous
    sitting of that class and subject in any of the school's exams (else from
    the start of the exam's year) and no later than this sitting. None when
    the question papers cannot be read, so nothing is claimed either way."""
    from ..assessment import routes as ar
    if ar._store is None:
        return None
    try:
        approved = [a for a in ar._store.list_by_school(ex["school_id"]) if a.status in APPROVED_PAPER]
    except Exception:  # noqa: BLE001 - an alert line is not worth failing the morning's alerts
        log.warning("could not read the question papers of %s", ex["school_id"], exc_info=True)
        return None
    sittings: dict[tuple[int, str], list[str]] = {}
    for other in ops.exams_for_school(ex["school_id"]):
        for p in ops.exam_papers(other["id"]):
            sittings.setdefault((p["grade"], p["subject_id"]), []).append(p["date"])
    year = cs.get_academic_year(ex["academic_year_id"])

    def made(a) -> str:
        at = a.scheduled_at or a.created_at
        return (at if at.tzinfo else at.replace(tzinfo=timezone.utc)).astimezone(IST).date().isoformat()

    missing = []
    for p in papers:
        if p["date"] < today:
            continue
        earlier = [d for d in sittings.get((p["grade"], p["subject_id"]), []) if d < p["date"]]
        subject = p["subject_name"].strip().casefold()
        if not any(a.grade == p["grade"] and a.subject.strip().casefold() == subject
                   and (made(a) > max(earlier) if earlier else made(a) >= (year.start_date if year else ""))
                   and made(a) <= p["date"] for a in approved):
            missing.append((p["grade"], p["subject_name"]))
    return sorted(set(missing))


def principal_alerts(ops, cs, users, notify, now: datetime) -> int:
    """ADM-3, once a morning: what needs the principal today, in one
    message -- sections falling behind their plan, lost periods not made up,
    homework overdue in bulk, exams within a week without a full roster or
    with a paper that has no approved question paper (_papers_not_set).
    Nothing to report sends nothing."""
    today = now.astimezone(IST).date()
    n = 0
    for school in _schools(cs):
        principals = [u.id for u in users.users_for_school(school, role="principal")]
        year = _year_on(cs, school, today.isoformat())
        if not principals or year is None:
            continue
        lines = []
        delayed = cs.get_delayed_topics(school_id=school, academic_year_id=year.id, as_of_date=today.isoformat())
        behind: dict[str, int] = {}
        for lesson in delayed["delayed_lessons"]:
            if lesson.get("section_id"):
                behind[lesson["section_id"]] = behind.get(lesson["section_id"], 0) + 1
        for sid, count in sorted(behind.items(), key=lambda kv: -kv[1]):
            if count >= BEHIND_LESSONS:
                lines.append((f"{_label(cs, sid)} is {count} lessons behind its plan",
                              f"{_label(cs, sid)} अपनी योजना से {count} पाठ पीछे है"))
        cutoff = (today - timedelta(days=OWED_AFTER_DAYS)).isoformat()
        owed = [l for l in cs.lost_periods_for_year(year.id, "owed") if l.date <= cutoff]
        if owed:
            lines.append((f"{len(owed)} lost periods not made up for over a week",
                          f"{len(owed)} छूटे पीरियड एक सप्ताह से अधिक समय से पूरे नहीं हुए"))
        bulk = 0
        for hw in ops.homework_for_school(school, status="published"):
            if hw.due_date >= today.isoformat():
                continue
            roster = sum(len(cs.enrollments_for_section(sid)) for sid in hw.section_ids)
            handed = len(ops.submissions_for(hw.id))
            if roster and handed < roster / 2:
                bulk += 1
        if bulk:
            lines.append((f"{bulk} homework past due with fewer than half handed in",
                          f"{bulk} गृहकार्य की अंतिम तिथि बीत गई और आधे से कम जमा हुए"))
        soon = (today + timedelta(days=7)).isoformat()
        for ex in ops.exams_for_school(school):
            if not (today.isoformat() <= ex["start_date"] <= soon):
                continue
            papers = ops.exam_papers(ex["id"])
            empty = sum(1 for r in ops.roster(ex["id"]) if not r["teacher_id"])
            if not papers:
                lines.append((f"{ex['name']} starts {ex['start_date']} with no datesheet",
                              f"{ex['name']} {ex['start_date']} से शुरू है, पर डेटशीट नहीं बनी"))
            elif empty or ex["status"] != "published":
                lines.append((f"{ex['name']} starts {ex['start_date']}: "
                              + (f"{empty} rooms without an invigilator" if empty else "not published yet"),
                              f"{ex['name']} {ex['start_date']} से शुरू है: "
                              + (f"{empty} कक्षों में निरीक्षक नहीं" if empty else "अभी प्रकाशित नहीं")))
            unset = _papers_not_set(ops, cs, ex, papers, today.isoformat()) if papers else None
            if unset:
                more = len(unset) - 5
                en = [f"class {g} {s}" for g, s in unset[:5]] + ([f"{more} more"] if more > 0 else [])
                hi = [f"कक्षा {g} {s}" for g, s in unset[:5]] + ([f"{more} और"] if more > 0 else [])
                lines.append((f"{ex['name']} starts {ex['start_date']}: no approved question paper for "
                              + ", ".join(en),
                              f"{ex['name']} {ex['start_date']} से शुरू है: " + ", ".join(hi)
                              + " का स्वीकृत प्रश्नपत्र नहीं"))
        if not lines:
            continue
        notify(school_id=school, user_ids=principals, kind="principal_alert",
               params={"title": f"Needs attention today ({len(lines)})",
                       "body": "; ".join(en for en, _ in lines) + ".",
                       "title_hi": f"आज ध्यान दें ({len(lines)})", "body_hi": "; ".join(hi for _, hi in lines) + "।"},
               link="/dashboard", dedupe_key=f"alerts:{school}:{today.isoformat()}")
        n += 1
    return n


def exam_reminders(ops, cs, users, notify, now: datetime) -> int:
    """The evening before: each student of a class sitting a paper tomorrow,
    and each invigilator with duties tomorrow, in a published exam."""
    tomorrow = (now.astimezone(IST).date() + timedelta(days=1)).isoformat()
    n = 0
    for school in _schools(cs):
        for ex in ops.exams_for_school(school):
            if ex["status"] != "published" or not (ex["start_date"] <= tomorrow <= ex["end_date"]):
                continue
            papers = [p for p in ops.exam_papers(ex["id"]) if p["date"] == tomorrow]
            if not papers:
                continue
            grade_ids = {g.number: g.id for g in cs.grades_for_year(ex["academic_year_id"])}
            for p in papers:
                students = [e.student_id for sec in (cs.sections_for_grade(grade_ids[p["grade"]])
                                                     if p["grade"] in grade_ids else [])
                            for e in cs.enrollments_for_section(sec.id)]
                if not students:
                    continue
                notify(school_id=school, user_ids=students, kind="exam_reminder",
                       params={"exam": ex["name"], "subject": p["subject_name"], "start": p["start_time"],
                               "end": p["end_time"]},
                       link="/my-exams", dedupe_key=f"examrem:{p['id']}")
                n += len(students)
            duties: dict[str, list[dict]] = {}
            for r in ops.roster(ex["id"]):
                if r["date"] == tomorrow and r["teacher_id"]:
                    duties.setdefault(r["teacher_id"], []).append(r)
            for teacher, rows in duties.items():
                first = min(rows, key=lambda r: r["start_time"])
                where = f"{_label(cs, first['section_id'])} {first['subject_name']}"
                notify(school_id=school, user_ids=[teacher], kind="duty_reminder",
                       params={"exam": ex["name"], "count": len(rows), "first": f"{where} at {first['start_time']}",
                               "first_hi": f"{where}, {first['start_time']} बजे"},
                       link="/my-exams", dedupe_key=f"dutyrem:{ex['id']}:{teacher}:{tomorrow}")
                n += 1
    return n


def term_report(ops, cs, users, notify, now: datetime) -> int:
    """On a term's last day, its school's principals hear the term report is
    ready (the report itself is built when opened, so it is never stale)."""
    today = now.astimezone(IST).date().isoformat()
    n = 0
    for school in _schools(cs):
        term = cs.term_for_date(school, today)
        if term is None or term.end_date != today:
            continue
        principals = [u.id for u in users.users_for_school(school, role="principal")]
        notify(school_id=school, user_ids=principals, kind="term_report_ready",
               params={"term": term.name, "end": term.end_date}, link="/school/term-report",
               dedupe_key=f"termreport:{term.id}")
        n += 1
    return n


def user_digests(ops, cs, users, notify, now: datetime) -> int:
    return ops.send_digests(now)


def deliveries(ops, cs, users, notify, now: datetime) -> int:
    return ops.deliver_due(now=now)


# (job, opens at IST hour, period: "day" | "week" | "run", weekday or None, function)
JOBS: list[tuple[str, int, str, Optional[int], Callable[..., int]]] = [
    ("homework_due_tomorrow", 16, "day", None, homework_due_tomorrow),
    ("homework_overdue", 9, "day", None, homework_overdue),
    ("substitution_escalation", 7, "run", None, substitution_escalation),
    ("teacher_daily_digest", 18, "day", None, teacher_daily_digest),
    ("principal_weekly_digest", 7, "week", 0, principal_weekly_digest),
    ("principal_alerts", 8, "day", None, principal_alerts),
    ("staff_check_in", 9, "day", None, staff_check_in),
    ("marking_due", 15, "day", None, marking_due),
    ("exam_reminders", 17, "day", None, exam_reminders),
    ("term_report", 15, "day", None, term_report),
    ("period_starting", 7, "run", None, period_starting),
    ("user_digests", 0, "run", None, user_digests),
    ("deliveries", 0, "run", None, deliveries),
]


def run_due(ops, cs, users, notify, now: Optional[datetime] = None) -> dict[str, Any]:
    """Run every job whose window is open and whose period is not done.
    A failing job is logged and reported, and does not stop the others."""
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(IST)
    out: dict[str, Any] = {}
    made: Counter = Counter()

    def counted(**kw: Any):
        # What each school actually got: a notification its dedupe key
        # suppressed is not in the list notify returns.
        result = notify(**kw)
        made[kw.get("school_id")] += len(result) if isinstance(result, list) else 0
        return result

    for job, hour, period, weekday, fn in JOBS:
        if local.hour < hour or (weekday is not None and local.weekday() != weekday):
            out[job] = "not yet"
            continue
        key = (local.date().isoformat() if period == "day"
               else f"{local.isocalendar().year}-W{local.isocalendar().week:02d}" if period == "week" else None)
        if key is not None and ops.automation_done(job, key):
            out[job] = "done"
            continue
        made.clear()
        try:
            n = fn(ops, cs, users, counted, now)
        except Exception as e:  # noqa: BLE001
            log.warning("automation %s failed", job, exc_info=True)
            out[job] = f"failed: {e}"
            continue
        if key is not None:
            ops.record_automation(job, key, n, {s: c for s, c in made.items() if s})
        out[job] = n
    return out
