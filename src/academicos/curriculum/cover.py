"""Leave, substitution and compensation -- keeping the timetable working when a
teacher is away or a day is lost (REQUIREMENTS SCH-5, SCH-6, SCH-7, SCH-8).

- A LeaveRequest (full day, first or second half, or named periods; planned or
  same-day) is approved or rejected by the principal (or a delegated admin).
- Approval finds every period the teacher misses: their own timetabled periods
  on those days (not one the class spends sitting an exam paper) and any
  substitution duty they had taken. Each gets a Substitution, and the engine
  proposes a substitute. The candidate must be free at that time by clock
  time (an invigilation duty is not free), not on leave then, and under the
  day's maximum. Candidates are ranked: the same subject and grade first, then the
  same subject, then someone who teaches that section, then the fewest
  substitutions this week, then the lightest day; staff with no class of
  their own come last. The principal confirms or picks another; the
  substitute accepts or declines.
- A period is LOST when nobody who teaches the subject takes it: nobody is
  free, the principal runs it as supervised study or combines sections, or a
  closure (a short-notice holiday, an event, an exam day) cancels the day. A
  lost period is a debt per section and subject. The engine proposes make-up
  periods (the section's free periods, when its subject teacher is free and
  under their day's maximum) and records the recovery. The section's lesson
  plan follows: the lesson of a lost period moves to the next free period,
  and a make-up period teaches the next lesson (periods_held).

Dates are the school's local dates (IST); times compare by the section's bell,
so a clash across two bells is still a clash.

CurriculumStore mixes this class in, like SchoolModelMixin.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

COVER_SCHEMA = """
CREATE TABLE IF NOT EXISTS leave_requests (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  teacher_id       TEXT NOT NULL,
  start_date       TEXT NOT NULL,
  end_date         TEXT NOT NULL,
  kind             TEXT NOT NULL,
  periods_json     TEXT NOT NULL DEFAULT '[]',
  reason           TEXT NOT NULL,
  handover_note    TEXT,
  status           TEXT NOT NULL,
  created_by       TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  decided_by       TEXT,
  decided_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_leave_teacher ON leave_requests(teacher_id, start_date);
CREATE INDEX IF NOT EXISTS idx_leave_year ON leave_requests(academic_year_id, status);

CREATE TABLE IF NOT EXISTS substitutions (
  id                TEXT PRIMARY KEY,
  school_id         TEXT NOT NULL,
  academic_year_id  TEXT NOT NULL,
  leave_id          TEXT,
  date              TEXT NOT NULL,
  section_id        TEXT NOT NULL,
  period            INTEGER NOT NULL,
  subject_id        TEXT NOT NULL,
  absent_teacher_id TEXT NOT NULL,
  substitute_id     TEXT,
  status            TEXT NOT NULL,
  mode              TEXT NOT NULL,
  note              TEXT,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL,
  UNIQUE(date, section_id, period)
);
CREATE INDEX IF NOT EXISTS idx_sub_date ON substitutions(academic_year_id, date);
CREATE INDEX IF NOT EXISTS idx_sub_substitute ON substitutions(substitute_id, date);

CREATE TABLE IF NOT EXISTS lost_periods (
  id                  TEXT PRIMARY KEY,
  school_id           TEXT NOT NULL,
  academic_year_id    TEXT NOT NULL,
  section_id          TEXT NOT NULL,
  subject_id          TEXT NOT NULL,
  date                TEXT NOT NULL,
  period              INTEGER NOT NULL,
  reason              TEXT NOT NULL,
  source_id           TEXT,
  status              TEXT NOT NULL,
  compensation_date   TEXT,
  compensation_period INTEGER,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL,
  UNIQUE(date, section_id, period)
);
CREATE INDEX IF NOT EXISTS idx_lost_year ON lost_periods(academic_year_id, status);
CREATE TABLE IF NOT EXISTS teacher_attendance (
  school_id   TEXT NOT NULL,
  date        TEXT NOT NULL,
  teacher_id  TEXT NOT NULL,
  status      TEXT NOT NULL,
  note        TEXT,
  leave_id    TEXT,
  marked_by   TEXT NOT NULL,
  marked_at   TEXT NOT NULL,
  PRIMARY KEY (date, teacher_id)
);
CREATE INDEX IF NOT EXISTS idx_attendance_school ON teacher_attendance(school_id, date);
"""

# SCH-9. An absence opens the same cover a same-day leave does; the half-day
# kinds miss only that half's periods.
ATTENDANCE_STATUSES = ("present", "late", "absent", "first_half_absent", "second_half_absent")
_ABSENCE_KIND = {"absent": "full_day", "first_half_absent": "first_half", "second_half_absent": "second_half"}

LEAVE_KINDS = ("full_day", "first_half", "second_half", "periods")
# open: needs a substitute; proposed: the engine or principal chose one, not yet
# accepted; accepted: the substitute took it; declined: they said no (open again);
# resolved: the principal ran it without a substitute (mode says how); cancelled:
# the leave was withdrawn.
SUB_OPEN = ("open", "declined")
SUB_TAKEN = ("proposed", "accepted")
MODES = ("substitute", "supervised", "combined", "lost")
MAX_PER_DAY = 7
_KEEP_NOTE = object()      # _propose: leave the substitution's note as it is


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    from .store import new_id
    return new_id(prefix)


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _dates(start: str, end: str):
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


@dataclass
class LeaveRequest:
    id: str
    school_id: str
    academic_year_id: str
    teacher_id: str
    start_date: str
    end_date: str
    kind: str
    periods: list[int]
    reason: str
    handover_note: Optional[str]
    status: str
    created_by: str
    created_at: str
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None


@dataclass
class Substitution:
    id: str
    school_id: str
    academic_year_id: str
    leave_id: Optional[str]
    date: str
    section_id: str
    period: int
    subject_id: str
    absent_teacher_id: str
    substitute_id: Optional[str]
    status: str
    mode: str
    note: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class LostPeriod:
    id: str
    school_id: str
    academic_year_id: str
    section_id: str
    subject_id: str
    date: str
    period: int
    reason: str
    source_id: Optional[str]
    status: str               # owed | compensated | waived
    compensation_date: Optional[str]
    compensation_period: Optional[int]
    created_at: str
    updated_at: str


@dataclass
class Candidate:
    teacher_id: str
    score: int
    qualified: bool           # teaches this subject (by name), any class
    reasons: list[str] = field(default_factory=list)


@dataclass
class ExamDay:
    """What one day's published exam papers hold (EX-7; v3 audit N-3-12):
    the clock spans each section sits a paper (with a line for the day
    view), and the spans each invigilator is on duty."""
    sitting: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    invigilating: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def sits(self, section_id: str, span: Optional[tuple[int, int]]) -> Optional[str]:
        """The paper `section_id` sits during the clock span `span`, or None."""
        if span is None:
            return None
        return next((label for s, e, label in self.sitting.get(section_id, ()) if s < span[1] and span[0] < e),
                    None)


class CoverError(ValueError):
    """The route's 422 (a bad request) -- see also KeyError (404)."""


class CoverMixin:

    # Two things the engine needs live in stores this one cannot read:
    # the school's teachers (user ids, from the user store) and its
    # published exam papers (operations, which imports this package).
    # curriculum.routes.init hands both over at boot. A bare store -- the
    # CLI, the store-level tests -- covers from the teaching allocations
    # alone and knows no exams.
    staff_source: Optional[Callable[[str], list[str]]] = None
    exam_source: Optional[Callable[[str, str], list[dict]]] = None   # (school, year) -> papers
    # Where a lesson moved by a lost or made-up period is recorded: the
    # assessment AuditLog, as PUSH writes it. Handed over at boot too; a bare
    # store has none, and its lost periods leave the plans where they are.
    plan_audit_log: Optional[Any] = None

    # ---------------- helpers ----------------

    def _exam_days(self, academic_year_id: str, on: Optional[str] = None) -> dict[str, ExamDay]:
        """{date: ExamDay} over the year's published exam papers (only
        `on`'s, when given). Every section of a paper's class sits it."""
        year = self.get_academic_year(academic_year_id) if self.exam_source else None
        if year is None:
            return {}
        papers = [p for p in self.exam_source(year.school_id, academic_year_id) if on is None or p["date"] == on]
        if not papers:
            return {}
        by_grade = {g.number: [s.id for s in self.sections_for_grade(g.id)]
                    for g in self.grades_for_year(academic_year_id)}
        days: dict[str, ExamDay] = defaultdict(ExamDay)
        for p in papers:
            span = (_minutes(p["start_time"]), _minutes(p["end_time"]))
            label = f"{p['exam_name']}: {p['subject_name']} paper, {p['start_time']}-{p['end_time']}"
            day = days[p["date"]]
            for sid in by_grade.get(p["grade"], []):
                day.sitting.setdefault(sid, []).append((*span, label))
            for _, teacher in p["duties"]:
                day.invigilating.setdefault(teacher, []).append(span)
        return dict(days)

    def _exam_day(self, academic_year_id: str, on: str) -> ExamDay:
        return self._exam_days(academic_year_id, on).get(on) or ExamDay()

    def _cover_times(self, section) -> dict[int, tuple[int, int]]:
        bell = self.bell_for_section(section)
        if bell is None:
            return {}
        return {s.period: (_minutes(s.start), _minutes(s.end)) for s in bell.slots if s.kind == "teaching"}

    def _working_dates(self, academic_year_id: str) -> set[str]:
        from .calendar import working_days_for_year
        try:
            return set(working_days_for_year(self, academic_year_id).dates)
        except ValueError:
            return set()

    def _leave_from_row(self, r: dict) -> LeaveRequest:
        r = dict(r)
        r["periods"] = json.loads(r.pop("periods_json") or "[]")
        return LeaveRequest(**r)

    def _half(self, section, period: int, half: str) -> bool:
        teaching = sorted(self._cover_times(section))
        if not teaching:
            return False
        cut = (len(teaching) + 1) // 2
        first = teaching[:cut]
        return (period in first) if half == "first_half" else (period not in first)

    def _leave_covers(self, leave: LeaveRequest, section, period: int) -> bool:
        if leave.kind == "full_day":
            return True
        if leave.kind == "periods":
            return period in leave.periods
        return self._half(section, period, leave.kind)

    # ---------------- leave (SCH-5) ----------------

    def apply_for_leave(self, *, school_id: str, academic_year_id: str, teacher_id: str,
                        start_date: str, end_date: str, kind: str, reason: str,
                        created_by: str, periods: Optional[list[int]] = None,
                        handover_note: Optional[str] = None) -> LeaveRequest:
        try:
            start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        except ValueError:
            raise CoverError("dates are YYYY-MM-DD")
        if end < start:
            raise CoverError("the leave ends before it starts")
        if kind not in LEAVE_KINDS:
            raise CoverError(f"kind is one of {', '.join(LEAVE_KINDS)}")
        periods = sorted(set(periods or []))
        if kind == "periods" and not periods:
            raise CoverError("name the periods the leave is for")
        if kind != "full_day" and start != end:
            raise CoverError("a half day or named periods are one day's leave")
        if not (reason or "").strip():
            raise CoverError("give a reason")
        year = self.get_academic_year(academic_year_id)
        if year is None:
            raise KeyError(academic_year_id)
        if start_date < year.start_date or end_date > year.end_date:
            raise CoverError("the leave is outside this academic year")
        with self._conn_lock:
            for other in self.leave_for_teacher(teacher_id):
                if other.status in ("pending", "approved") and not (
                        other.end_date < start_date or other.start_date > end_date):
                    if other.kind == "full_day" or kind == "full_day" or other.kind == kind:
                        raise CoverError(f"this overlaps leave already applied for "
                                         f"({other.start_date} to {other.end_date}, {other.status})")
            leave = LeaveRequest(id=_new_id("leave"), school_id=school_id, academic_year_id=academic_year_id,
                                 teacher_id=teacher_id, start_date=start_date, end_date=end_date, kind=kind,
                                 periods=periods, reason=reason.strip(), handover_note=handover_note,
                                 status="pending", created_by=created_by, created_at=_now())
            self._exec("INSERT INTO leave_requests (id, school_id, academic_year_id, teacher_id, start_date, "
                       "end_date, kind, periods_json, reason, handover_note, status, created_by, created_at) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (leave.id, leave.school_id, leave.academic_year_id, leave.teacher_id, leave.start_date,
                        leave.end_date, leave.kind, json.dumps(leave.periods), leave.reason,
                        leave.handover_note, leave.status, leave.created_by, leave.created_at))
        self._commit()
        return leave

    def get_leave(self, leave_id: str) -> Optional[LeaveRequest]:
        r = self._fetchone("SELECT * FROM leave_requests WHERE id=?", (leave_id,))
        return self._leave_from_row(r) if r else None

    def leave_for_teacher(self, teacher_id: str) -> list[LeaveRequest]:
        return [self._leave_from_row(r) for r in self._fetchall(
            "SELECT * FROM leave_requests WHERE teacher_id=? ORDER BY start_date", (teacher_id,))]

    def leave_for_year(self, academic_year_id: str, status: Optional[str] = None) -> list[LeaveRequest]:
        sql, params = "SELECT * FROM leave_requests WHERE academic_year_id=?", (academic_year_id,)
        if status:
            sql += " AND status=?"
            params += (status,)
        return [self._leave_from_row(r) for r in self._fetchall(sql + " ORDER BY start_date", params)]

    def _on_leave(self, teacher_id: str, on: str, section, period: int) -> bool:
        for leave in self.leave_for_teacher(teacher_id):
            if leave.status == "approved" and leave.start_date <= on <= leave.end_date \
                    and self._leave_covers(leave, section, period):
                return True
        return False

    def _other_teacher(self, e, teacher_id: str) -> Optional[str]:
        """The co-teaching partner of `teacher_id` in period `e` (SCH-8), or None."""
        pair = [t for t in (e.teacher_id, getattr(e, "co_teacher_id", None)) if t]
        others = [t for t in pair if t != teacher_id]
        return others[0] if teacher_id in pair and others else None

    def affected_periods(self, leave: LeaveRequest) -> list[tuple[str, Any, int, str]]:
        """(date, section, period, subject_id) that need someone because the
        teacher is away: their own timetabled periods on the leave's working
        days, and any substitution duty they had taken.

        A co-taught period (SCH-8) needs no one while the other teacher of it
        is there: they take the class. It needs a substitute only when both
        are away, whichever went on leave second. Nor does a period the class
        spends sitting an exam paper (v3 audit N-3-12)."""
        working = self._working_dates(leave.academic_year_id)
        own = [e for e in self.timetable_for_year(leave.academic_year_id)
               if leave.teacher_id in (e.teacher_id, e.co_teacher_id)]
        sections = {s.id: s for s in self.sections_for_year(leave.academic_year_id)}
        exam_days = self._exam_days(leave.academic_year_id) if own else {}
        out = []
        for d in _dates(leave.start_date, leave.end_date):
            if d not in working:
                continue
            wd = date.fromisoformat(d).weekday()
            exams = exam_days.get(d) or ExamDay()
            for e in own:
                sec = sections.get(e.section_id)
                if sec is None or e.day_of_week != wd or not self._leave_covers(leave, sec, e.period):
                    continue
                if exams.sits(sec.id, self._cover_times(sec).get(e.period)):
                    continue
                partner = self._other_teacher(e, leave.teacher_id)
                if partner and not self._on_leave(partner, d, sec, e.period):
                    continue
                out.append((d, sec, e.period, e.subject_id))
            for s in self._subs_where("substitute_id=? AND date=? AND status IN ('proposed','accepted')",
                                      (leave.teacher_id, d)):
                sec = sections.get(s.section_id)
                if sec is not None and self._leave_covers(leave, sec, s.period):
                    out.append((d, sec, s.period, s.subject_id))
        return out

    def decide_leave(self, leave_id: str, *, approve: bool, decided_by: str) -> tuple[LeaveRequest, list[Substitution]]:
        """Approve (and open a substitution for every missed period, each
        with the engine's proposed substitute) or reject."""
        with self._conn_lock:
            leave = self.get_leave(leave_id)
            if leave is None:
                raise KeyError(leave_id)
            if leave.status != "pending":
                raise CoverError(f"this leave is already {leave.status}")
            status = "approved" if approve else "rejected"
            self._exec("UPDATE leave_requests SET status=?, decided_by=?, decided_at=? WHERE id=?",
                       (status, decided_by, _now(), leave_id))
            leave.status = status
            subs: list[Substitution] = []
            if approve:
                for d, sec, period, subject_id in self.affected_periods(leave):
                    existing = self._subs_where("date=? AND section_id=? AND period=?", (d, sec.id, period))
                    if existing:
                        # A duty the teacher had taken: it needs someone else now.
                        sub = existing[0]
                        self._exec("UPDATE substitutions SET substitute_id=NULL, status='open', updated_at=?, "
                                   "note=? WHERE id=?",
                                   (_now(), "the substitute went on leave", sub.id))
                    else:
                        sub = Substitution(id=_new_id("sub"), school_id=leave.school_id,
                                           academic_year_id=leave.academic_year_id, leave_id=leave.id, date=d,
                                           section_id=sec.id, period=period, subject_id=subject_id,
                                           absent_teacher_id=leave.teacher_id, substitute_id=None,
                                           status="open", mode="substitute", note=None,
                                           created_at=_now(), updated_at=_now())
                        self._exec("INSERT INTO substitutions (id, school_id, academic_year_id, leave_id, date, "
                                   "section_id, period, subject_id, absent_teacher_id, substitute_id, status, mode, "
                                   "note, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (sub.id, sub.school_id, sub.academic_year_id, sub.leave_id, sub.date,
                                    sub.section_id, sub.period, sub.subject_id, sub.absent_teacher_id, None,
                                    sub.status, sub.mode, None, sub.created_at, sub.updated_at))
                    best = self.substitute_candidates(sub.id)
                    self._propose(sub, best[0] if best else None)
                    subs.append(self.get_substitution(sub.id))
        self._commit()
        return leave, subs

    def cancel_leave(self, leave_id: str, *, today: str) -> LeaveRequest:
        """Withdraw a leave: its substitutions from today on are cancelled
        (a period already taught or lost stays on the record)."""
        with self._conn_lock:
            leave = self.get_leave(leave_id)
            if leave is None:
                raise KeyError(leave_id)
            if leave.status not in ("pending", "approved"):
                raise CoverError(f"this leave is already {leave.status}")
            # A period someone was only supervising was owed (SCH-7); with
            # the teacher back it is taught after all, so the debt goes too.
            for sub in self._subs_where("leave_id=? AND date>=? AND status NOT IN ('resolved','cancelled')",
                                        (leave_id, today)):
                self._clear_lost(sub)
            self._exec("UPDATE leave_requests SET status='cancelled' WHERE id=?", (leave_id,))
            self._exec("UPDATE substitutions SET status='cancelled', updated_at=? WHERE leave_id=? AND date>=? "
                       "AND status<>'resolved'", (_now(), leave_id, today))
            leave.status = "cancelled"
            self._release_co_taught_cover(leave, today)
        self._commit()
        return leave

    def _release_co_taught_cover(self, leave: LeaveRequest, today: str) -> None:
        """SCH-8: a co-taught period got a substitute only because both of
        its teachers were away. When one of them is back (their leave
        cancelled, or marked present), the period has a teacher again: its
        substitution, opened under the other teacher's leave, is cancelled."""
        sections = {s.id: s for s in self.sections_for_year(leave.academic_year_id)}
        entries = {(e.section_id, e.day_of_week, e.period): e for e in self.timetable_for_year(leave.academic_year_id)
                   if e.co_teacher_id}
        for sub in self._subs_where("academic_year_id=? AND date>=? AND date>=? AND date<=? "
                                    "AND status IN ('open','proposed','accepted') AND leave_id<>?",
                                    (leave.academic_year_id, today, leave.start_date, leave.end_date, leave.id)):
            e = entries.get((sub.section_id, date.fromisoformat(sub.date).weekday(), sub.period))
            sec = sections.get(sub.section_id)
            if e is None or sec is None or leave.teacher_id not in (e.teacher_id, e.co_teacher_id):
                continue
            if self._on_leave(leave.teacher_id, sub.date, sec, sub.period):
                continue          # still away through another leave
            self._exec("UPDATE substitutions SET status='cancelled', note=?, updated_at=? WHERE id=?",
                       ("the co-teacher is back and takes the class", _now(), sub.id))
            self._clear_lost(sub)

    # ---------------- teacher attendance (SCH-9) ----------------

    def attendance_for_date(self, school_id: str, on: str) -> dict[str, dict]:
        return {r["teacher_id"]: r for r in self._fetchall(
            "SELECT * FROM teacher_attendance WHERE school_id=? AND date=?", (school_id, on))}

    def mark_attendance(self, *, school_id: str, academic_year_id: str, on: str, teacher_id: str, status: str,
                        marked_by: str, note: Optional[str] = None) -> tuple[dict, list["Substitution"]]:
        """Record one teacher's attendance for a day. Marking someone absent
        who has no leave for that day opens a same-day leave, approved at
        once, so every period they miss gets a substitution with a proposed
        substitute (decide_leave). Marking them present again cancels that
        leave -- only the one attendance opened -- and its substitutions.
        Returns (the row, the substitutions opened)."""
        if status not in ATTENDANCE_STATUSES:
            raise CoverError(f"status must be one of {', '.join(ATTENDANCE_STATUSES)}")
        before = self.attendance_for_date(school_id, on).get(teacher_id)
        leave_id = before["leave_id"] if before else None
        opened: list[Substitution] = []
        if leave_id:
            leave = self.get_leave(leave_id)
            if leave is None or status not in _ABSENCE_KIND or _ABSENCE_KIND[status] != leave.kind:
                if leave is not None and leave.status in ("pending", "approved"):
                    self.cancel_leave(leave_id, today=on)
                leave_id = None
        if status in _ABSENCE_KIND and leave_id is None:
            today_leave = [l for l in self.leave_for_teacher(teacher_id) if l.start_date <= on <= l.end_date]
            pending = next((l for l in today_leave if l.status == "pending"), None)
            if pending is not None:
                # They asked for it; being absent answers the request.
                _, opened = self.decide_leave(pending.id, approve=True, decided_by=marked_by)
            elif not any(l.status == "approved" for l in today_leave):
                leave = self.apply_for_leave(school_id=school_id, academic_year_id=academic_year_id,
                                             teacher_id=teacher_id, start_date=on, end_date=on,
                                             kind=_ABSENCE_KIND[status], reason="Marked absent at attendance",
                                             created_by=marked_by)
                leave, opened = self.decide_leave(leave.id, approve=True, decided_by=marked_by)
                leave_id = leave.id
        self._exec("INSERT INTO teacher_attendance (school_id, date, teacher_id, status, note, leave_id, marked_by,"
                   " marked_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(date, teacher_id) DO UPDATE SET"
                   " status=excluded.status, note=excluded.note, leave_id=excluded.leave_id,"
                   " marked_by=excluded.marked_by, marked_at=excluded.marked_at",
                   (school_id, on, teacher_id, status, note, leave_id, marked_by, _now()))
        self._commit()
        return self.attendance_for_date(school_id, on)[teacher_id], opened

    # ---------------- substitution (SCH-6) ----------------

    def _subs_where(self, where: str, params: tuple) -> list[Substitution]:
        return [Substitution(**r) for r in self._fetchall(
            f"SELECT * FROM substitutions WHERE {where} ORDER BY date, period", params)]

    def get_substitution(self, sub_id: str) -> Optional[Substitution]:
        rows = self._subs_where("id=?", (sub_id,))
        return rows[0] if rows else None

    def substitutions_between(self, academic_year_id: str, start: str, end: str) -> list[Substitution]:
        return self._subs_where("academic_year_id=? AND date>=? AND date<=?", (academic_year_id, start, end))

    def duties_for(self, teacher_id: str, start: str, end: str) -> list[Substitution]:
        return self._subs_where("substitute_id=? AND date>=? AND date<=? AND status IN ('proposed','accepted')",
                                (teacher_id, start, end))

    def _busy_spans(self, teacher_id: str, on: str, academic_year_id: str,
                    sections: dict, exclude_sub: Optional[str] = None,
                    exams: Optional[ExamDay] = None) -> list[tuple[int, int]]:
        """Clock spans a teacher is busy on a date: their own periods (if
        not on leave then, and not while that class sits an exam paper),
        substitution duties, make-up periods and invigilation duties. An
        invigilator was proposed as a substitute during their own paper
        (v3 audit N-3-12). `exams` is that day's, when the caller has it."""
        if exams is None:
            exams = self._exam_day(academic_year_id, on)
        wd = date.fromisoformat(on).weekday()
        spans = list(exams.invigilating.get(teacher_id, ()))
        for e in self.timetable_for_year(academic_year_id):
            if teacher_id not in (e.teacher_id, e.co_teacher_id) or e.day_of_week != wd:
                continue
            sec = sections.get(e.section_id)
            if sec is None or self._on_leave(teacher_id, on, sec, e.period):
                continue
            covered = self._subs_where("date=? AND section_id=? AND period=? AND status<>'cancelled'",
                                       (on, e.section_id, e.period))
            if covered and covered[0].absent_teacher_id == teacher_id:
                continue
            t = self._cover_times(sec).get(e.period)
            if t and not exams.sits(e.section_id, t):
                spans.append(t)
        for s in self._subs_where("substitute_id=? AND date=? AND status IN ('proposed','accepted')",
                                  (teacher_id, on)):
            if s.id == exclude_sub:
                continue
            t = self._cover_times(sections[s.section_id]).get(s.period) if s.section_id in sections else None
            if t:
                spans.append(t)
        for lp in self._lost_where("status='compensated' AND compensation_date=?", (on,)):
            alloc = self.allocation_for(lp.section_id, lp.subject_id)
            if alloc and alloc.teacher_id == teacher_id and lp.section_id in sections:
                t = self._cover_times(sections[lp.section_id]).get(lp.compensation_period)
                if t:
                    spans.append(t)
        return spans

    def substitute_candidates(self, sub_id: str) -> list[Candidate]:
        """Who can take this period, best first, with the reasons."""
        sub = self.get_substitution(sub_id)
        if sub is None:
            raise KeyError(sub_id)
        sections = {s.id: s for s in self.sections_for_year(sub.academic_year_id)}
        section = sections.get(sub.section_id)
        if section is None:
            return []
        slot = self._cover_times(section).get(sub.period)
        if slot is None:
            return []
        subject = self.get_subject(sub.subject_id)
        allocs = self.allocations_for_year(sub.academic_year_id)
        subject_names = {s.id: s.name for s in (self.get_subject(a.subject_id) for a in allocs) if s}
        teaches: dict[str, set[tuple[str, str]]] = defaultdict(set)     # teacher -> {(subject name, grade id)}
        in_section: set[str] = set()
        for a in allocs:
            sec = sections.get(a.section_id)
            for t in {a.teacher_id, a.co_teacher_id} - {None}:
                teaches[t].add((subject_names.get(a.subject_id, ""), sec.grade_id if sec else ""))
                if a.section_id == sub.section_id:
                    in_section.add(t)
        d = date.fromisoformat(sub.date)
        week_start = (d - timedelta(days=d.weekday())).isoformat()
        week_end = (d + timedelta(days=6 - d.weekday())).isoformat()
        this_week = Counter(s.substitute_id for s in self._subs_where(
            "academic_year_id=? AND date>=? AND date<=? AND status IN ('proposed','accepted')",
            (sub.academic_year_id, week_start, week_end)) if s.id != sub.id)
        # Staff with no class of their own (a librarian, a PT teacher, a new
        # joiner) are the school's usual cover pool, and were never offered
        # (v3 audit N-3-16). They can only supervise, so they come last: after
        # every teacher of the school, whatever the scores.
        no_class = [t for t in dict.fromkeys(self.staff_source(sub.school_id) if self.staff_source else [])
                    if t not in teaches]
        last_resort = set(no_class)
        exams = self._exam_day(sub.academic_year_id, sub.date)
        out: list[Candidate] = []
        for t in [*teaches, *no_class]:
            if t == sub.absent_teacher_id:
                continue
            if self._on_leave(t, sub.date, section, sub.period):
                continue
            spans = self._busy_spans(t, sub.date, sub.academic_year_id, sections, exclude_sub=sub.id, exams=exams)
            if any(s < slot[1] and slot[0] < e for s, e in spans):
                continue
            if len(spans) >= MAX_PER_DAY:
                continue
            name = subject.name if subject else ""
            taught = teaches.get(t, set())
            same_subject_grade = (name, section.grade_id) in taught
            same_subject = any(n == name for n, _ in taught)
            score, reasons = 0, []
            if t in last_resort:
                reasons.append("has no class of their own: only when no teacher is free")
            if same_subject_grade:
                score += 40
                reasons.append(f"teaches {name} to this class")
            elif same_subject:
                score += 30
                reasons.append(f"teaches {name}")
            if t in in_section:
                score += 10
                reasons.append("teaches this section")
            score -= 3 * this_week[t]
            reasons.append(f"{this_week[t]} substitution(s) this week")
            score -= len(spans)
            reasons.append(f"{len(spans)} period(s) that day")
            out.append(Candidate(teacher_id=t, score=score, qualified=same_subject, reasons=reasons))
        out.sort(key=lambda c: (c.teacher_id in last_resort, -c.score, c.teacher_id))
        return out

    def assign_substitute(self, sub_id: str, *, substitute_id: Optional[str], mode: str = "substitute",
                          note: Optional[str] = None) -> tuple[Substitution, Optional[LostPeriod]]:
        """The principal's decision for one period: a named substitute, or run
        it without one (supervised study, combined with another section, or
        lost). A period nobody who teaches the subject takes is a lost
        period for the section and subject (SCH-7)."""
        if mode not in MODES:
            raise CoverError(f"mode is one of {', '.join(MODES)}")
        with self._conn_lock:
            sub = self.get_substitution(sub_id)
            if sub is None:
                raise KeyError(sub_id)
            if sub.status in ("cancelled",):
                raise CoverError("this substitution was cancelled with its leave")
            lost = None
            if mode == "substitute":
                if substitute_id is None:
                    raise CoverError("name the substitute, or choose supervised, combined or lost")
                allowed = {c.teacher_id: c for c in self.substitute_candidates(sub_id)}
                if substitute_id not in allowed:
                    raise CoverError("that teacher is not free for this period (teaching, on leave, "
                                     "or at the day's maximum)")
                lost = self._propose(sub, allowed[substitute_id], note=note)
            else:
                self._exec("UPDATE substitutions SET substitute_id=?, status='resolved', mode=?, note=?, "
                           "updated_at=? WHERE id=?", (substitute_id if mode == "supervised" else None,
                                                       mode, note, _now(), sub_id))
                lost = self._record_lost(sub, reason={"supervised": "subject not taught (supervised)",
                                                      "combined": "combined with another section",
                                                      "lost": "no substitute"}[mode])
        self._commit()
        return self.get_substitution(sub_id), lost

    def respond_to_substitution(self, sub_id: str, *, teacher_id: str, accept: bool) -> Substitution:
        with self._conn_lock:
            sub = self.get_substitution(sub_id)
            if sub is None:
                raise KeyError(sub_id)
            if sub.substitute_id != teacher_id or sub.status != "proposed":
                raise CoverError("this is not a substitution waiting for your answer")
            if accept:
                self._exec("UPDATE substitutions SET status='accepted', updated_at=? WHERE id=?", (_now(), sub_id))
            else:
                self._exec("UPDATE substitutions SET status='declined', substitute_id=NULL, mode='substitute', "
                           "updated_at=? WHERE id=?", (_now(), sub_id))
                nxt = [c for c in self.substitute_candidates(sub_id) if c.teacher_id != teacher_id]
                self._propose(sub, nxt[0] if nxt else None)
        self._commit()
        return self.get_substitution(sub_id)

    def _propose(self, sub: Substitution, cand: Optional[Candidate], *,
                 note: Any = _KEEP_NOTE) -> Optional[LostPeriod]:
        """Put `cand` on the period, or leave it waiting for someone (open,
        or declined) when there is nobody. A teacher of the subject is a
        substitute. Anyone else only supervises: the subject is not taught,
        so the period is lost for the section and subject (SCH-7) and stays
        owed until it is made up. A period that gets a teacher of the
        subject again, or goes back to waiting, owes nothing (yet).

        One rule for every path. Until the v3 audit (N-3-9) only the
        principal's own pick was judged so: the engine's proposal on
        approving a leave, and the next candidate after a decline, were
        filed as 'substitute' whoever they were -- counted as filled, and
        never a lost period (8 of 24 in the audit's school)."""
        if cand is None:
            self._exec("UPDATE substitutions SET substitute_id=NULL, mode='substitute', updated_at=? WHERE id=?",
                       (_now(), sub.id))
            self._clear_lost(sub)
            return None
        mode = "substitute" if cand.qualified else "supervised"
        if note is _KEEP_NOTE:
            self._exec("UPDATE substitutions SET substitute_id=?, status='proposed', mode=?, updated_at=? "
                       "WHERE id=?", (cand.teacher_id, mode, _now(), sub.id))
        else:
            self._exec("UPDATE substitutions SET substitute_id=?, status='proposed', mode=?, note=?, updated_at=? "
                       "WHERE id=?", (cand.teacher_id, mode, note, _now(), sub.id))
        if cand.qualified:
            self._clear_lost(sub)
            return None
        return self._record_lost(sub, reason="subject not taught (supervised)")

    def settle_past_uncovered(self, academic_year_id: str, today: str) -> int:
        """A period whose day has passed with nobody covering it was lost."""
        n = 0
        with self._conn_lock:
            for sub in self._subs_where("academic_year_id=? AND date<? AND status IN ('open','declined')",
                                        (academic_year_id, today)):
                self._exec("UPDATE substitutions SET status='resolved', mode='lost', updated_at=? WHERE id=?",
                           (_now(), sub.id))
                if self._record_lost(sub, reason="no substitute") is not None:
                    n += 1
        if n:
            self._commit()
        return n

    # ---------------- lost periods and compensation (SCH-7) ----------------

    def _lost_where(self, where: str, params: tuple) -> list[LostPeriod]:
        return [LostPeriod(**r) for r in self._fetchall(
            f"SELECT * FROM lost_periods WHERE {where} ORDER BY date, period", params)]

    def _record_lost(self, sub: Substitution, *, reason: str) -> Optional[LostPeriod]:
        existing = self._lost_where("date=? AND section_id=? AND period=?", (sub.date, sub.section_id, sub.period))
        if existing:
            return existing[0]
        lp = self._insert_lost(sub.school_id, sub.academic_year_id, sub.section_id, sub.subject_id,
                               sub.date, sub.period, reason, sub.id)
        self._replan(lp.academic_year_id, lp.section_id, lp.subject_id, lp.date,
                     f"period {lp.period} on {lp.date} lost: {reason}")
        return lp

    def _insert_lost(self, school_id, academic_year_id, section_id, subject_id, on, period, reason,
                     source_id) -> LostPeriod:
        lp = LostPeriod(id=_new_id("lost"), school_id=school_id, academic_year_id=academic_year_id,
                        section_id=section_id, subject_id=subject_id, date=on, period=period, reason=reason,
                        source_id=source_id, status="owed", compensation_date=None,
                        compensation_period=None, created_at=_now(), updated_at=_now())
        self._exec("INSERT INTO lost_periods (id, school_id, academic_year_id, section_id, subject_id, date, "
                   "period, reason, source_id, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (lp.id, lp.school_id, lp.academic_year_id, lp.section_id, lp.subject_id, lp.date,
                    lp.period, lp.reason, lp.source_id, lp.status, lp.created_at, lp.updated_at))
        return lp

    def _clear_lost(self, sub: Substitution, *, replan: bool = True, keep_made_up: bool = False) -> None:
        """A period that now has a teacher of its subject is no longer lost,
        and the plan may teach in it again (`replan`). Its make-up goes with
        it: a teacher marked absent by mistake and then present kept the
        period "lost" and an extra make-up period booked (QA S-02), because
        only a period still owed was cleared. `keep_made_up` is for a period
        that stays lost for another reason (a closure), whose booked make-up
        still stands."""
        statuses = "('owed')" if keep_made_up else "('owed','compensated')"
        where = (f"date=? AND section_id=? AND period=? AND status IN {statuses} AND source_id=?",
                 (sub.date, sub.section_id, sub.period, sub.id))
        if not self._lost_where(*where):
            return
        self._exec(f"DELETE FROM lost_periods WHERE {where[0]}", where[1])
        if replan:
            self._replan(sub.academic_year_id, sub.section_id, sub.subject_id, sub.date,
                         f"period {sub.period} on {sub.date} is taught after all")

    def _replan(self, academic_year_id: str, section_id: str, subject_id: str, from_date: str,
                reason: str) -> int:
        """A period of the section's subject was lost, given back or made
        up: its plan is laid again on the periods that now happen (see
        periods_held), so a lesson whose period was lost is taught at the
        plan's next free period and a make-up period teaches the next one
        (v3 audit N-3-10). Each move is on the audit log, as a PUSH's is."""
        if self.plan_audit_log is None:
            return 0
        from .scheduling import replan_section
        return replan_section(self, self.plan_audit_log, academic_year_id=academic_year_id,
                              section_id=section_id, subject_id=subject_id, from_date=from_date,
                              reason=reason, changed_by="system")

    def lost_periods_for_year(self, academic_year_id: str, status: Optional[str] = None) -> list[LostPeriod]:
        if status:
            return self._lost_where("academic_year_id=? AND status=?", (academic_year_id, status))
        return self._lost_where("academic_year_id=?", (academic_year_id,))

    def get_lost_period(self, lost_id: str) -> Optional[LostPeriod]:
        rows = self._lost_where("id=?", (lost_id,))
        return rows[0] if rows else None

    def declare_closure(self, academic_year_id: str, on: str, *, reason: str,
                        section_ids: Optional[set[str]] = None) -> list[LostPeriod]:
        """A day that stops being taught at short notice (or after the fact):
        every section's timetabled periods that day are lost periods -- or
        only `section_ids`' (some classes out: a trip, an exam hall). The
        caller adds the holiday to the calendar and reflows the plans (or,
        asked not to, leaves them), so nothing here moves a plan."""
        year = self.get_academic_year(academic_year_id)
        if year is None:
            raise KeyError(academic_year_id)
        if not year.start_date <= on <= year.end_date:
            raise CoverError("the date is outside this academic year")
        wd = date.fromisoformat(on).weekday()
        out = []
        with self._conn_lock:
            # The day's cover is cancelled below. A period someone was only
            # supervising already owed itself; the closure owes it now, with
            # the closure's reason, like every other period of the day.
            for sub in self._subs_where("academic_year_id=? AND date=? AND status NOT IN ('resolved','cancelled')",
                                        (academic_year_id, on)):
                if section_ids is None or sub.section_id in section_ids:
                    self._clear_lost(sub, replan=False, keep_made_up=True)
            for e in self.timetable_for_year(academic_year_id):
                if e.day_of_week != wd:
                    continue
                if section_ids is not None and e.section_id not in section_ids:
                    continue
                if self._lost_where("date=? AND section_id=? AND period=?", (on, e.section_id, e.period)):
                    continue
                out.append(self._insert_lost(year.school_id, academic_year_id, e.section_id, e.subject_id,
                                             on, e.period, f"closure: {reason}", None))
            if section_ids is None:
                self._exec("UPDATE substitutions SET status='cancelled', updated_at=? WHERE academic_year_id=? "
                           "AND date=? AND status<>'resolved'", (_now(), academic_year_id, on))
            else:
                for sid in sorted(section_ids):
                    self._exec("UPDATE substitutions SET status='cancelled', updated_at=? WHERE "
                               "academic_year_id=? AND date=? AND section_id=? AND status<>'resolved'",
                               (_now(), academic_year_id, on, sid))
        self._commit()
        return out

    def compensation_options(self, lost_id: str, *, today: str, days_ahead: int = 21,
                             limit: int = 20) -> list[tuple[str, int]]:
        """(date, period) where the section is free (and not sitting an exam
        paper), its subject teacher is free and under the day's maximum,
        from today on."""
        lp = self.get_lost_period(lost_id)
        if lp is None:
            raise KeyError(lost_id)
        sections = {s.id: s for s in self.sections_for_year(lp.academic_year_id)}
        section = sections.get(lp.section_id)
        if section is None:
            return []
        times = self._cover_times(section)
        alloc = self.allocation_for(lp.section_id, lp.subject_id)
        teacher = alloc.teacher_id if alloc else None
        working = self._working_dates(lp.academic_year_id)
        start = max(date.fromisoformat(today), date.fromisoformat(lp.date) + timedelta(days=1))
        section_week = [e for e in self.timetable_for_section(lp.section_id)]
        exam_days = self._exam_days(lp.academic_year_id)
        out = []
        for i in range(days_ahead):
            d = (start + timedelta(days=i)).isoformat()
            if d not in working:
                continue
            wd = date.fromisoformat(d).weekday()
            exams = exam_days.get(d) or ExamDay()
            taken = {e.period for e in section_week if e.day_of_week == wd}
            taken |= {x.compensation_period for x in self._lost_where(
                "section_id=? AND compensation_date=? AND status='compensated'", (lp.section_id, d))}
            busy = self._busy_spans(teacher, d, lp.academic_year_id, sections, exams=exams) if teacher else []
            if teacher and len(busy) >= MAX_PER_DAY:
                continue
            for p, (s, e) in sorted(times.items()):
                if p in taken or exams.sits(section.id, (s, e)):
                    continue
                if teacher and (self._on_leave(teacher, d, section, p) or any(bs < e and s < be for bs, be in busy)):
                    continue
                out.append((d, p))
                if len(out) >= limit:
                    return out
        return out

    def compensate(self, lost_id: str, *, on: str, period: int, today: str) -> LostPeriod:
        with self._conn_lock:
            lp = self.get_lost_period(lost_id)
            if lp is None:
                raise KeyError(lost_id)
            if lp.status != "owed":
                raise CoverError(f"this lost period is already {lp.status}")
            if (on, period) not in self.compensation_options(lost_id, today=today, days_ahead=120, limit=10_000):
                raise CoverError("that period is not free for this section and its teacher")
            self._exec("UPDATE lost_periods SET status='compensated', compensation_date=?, compensation_period=?, "
                       "updated_at=? WHERE id=?", (on, period, _now(), lost_id))
            self._replan(lp.academic_year_id, lp.section_id, lp.subject_id, on,
                         f"make-up period {period} on {on} for {lp.date} period {lp.period}")
        self._commit()
        return self.get_lost_period(lost_id)

    def waive_lost(self, lost_id: str) -> LostPeriod:
        with self._conn_lock:
            lp = self.get_lost_period(lost_id)
            if lp is None:
                raise KeyError(lost_id)
            if lp.status != "owed":
                raise CoverError(f"this lost period is already {lp.status}")
            self._exec("UPDATE lost_periods SET status='waived', updated_at=? WHERE id=?", (_now(), lost_id))
        self._commit()
        return self.get_lost_period(lost_id)

    # ---------------- the plan's periods (SCH-4 with SCH-7, EX-7) ----------------

    def periods_held(self, academic_year_id: str, section_id: str,
                     subject_id: str) -> tuple[Counter, Counter]:
        """For one section's plan of one subject: per date, how many of its
        week's periods of the subject will not teach it, and how many extra
        periods will. calendar.teaching_slots_for_book takes the first off
        the plan's periods and adds the second, so a plan made, pushed or
        reflowed is laid only on periods that happen.

        Not taught: a lost period of the subject, whatever became of its
        debt, and a period the class spends sitting a published exam paper.
        Extra: a make-up period. Until the v3 audit a lost period left its
        lesson 'scheduled' on that day, a make-up taught nothing (N-3-10),
        and 10-A kept its Science lesson on the day of the Class 10 paper
        (N-3-12)."""
        section = self.get_section(section_id)
        if section is None:
            return Counter(), Counter()
        lost = self._lost_where("academic_year_id=? AND section_id=? AND subject_id=?",
                                (academic_year_id, section_id, subject_id))
        held = {(l.date, l.period) for l in lost}
        extra = Counter(l.compensation_date for l in lost if l.status == "compensated" and l.compensation_date)
        exam_days = self._exam_days(academic_year_id)
        if exam_days:
            times = self._cover_times(section)
            week = [e for e in self.timetable_for_section(section_id) if e.subject_id == subject_id]
            for d, exams in exam_days.items():
                wd = date.fromisoformat(d).weekday()
                held |= {(d, e.period) for e in week
                         if e.day_of_week == wd and exams.sits(section_id, times.get(e.period))}
        return Counter(d for d, _ in held), extra

    # ---------------- the day (what actually happens) ----------------

    def handover_for(self, sub: "Substitution") -> Optional[str]:
        """The absent teacher's note for whoever covers: the leave's handover
        note, else the substitution's own note."""
        leave = self.get_leave(sub.leave_id) if sub.leave_id else None
        return (leave.handover_note if leave and leave.handover_note else None) or sub.note

    def day_view(self, academic_year_id: str, on: str, *, section_id: Optional[str] = None,
                 teacher_id: Optional[str] = None) -> list[dict[str, Any]]:
        """One day as it will run: the week's periods, with that day's
        substitutions, lost periods and make-up periods applied. Filter by a
        section, or by a teacher (their periods, their duties, their make-ups;
        their periods on leave marked so).

        A period the class spends sitting a published exam paper is kind
        'exam', with no teacher: it is not taught. Until the v3 audit
        (N-3-12) grade 10's day showed regular teaching through its paper."""
        if on not in self._working_dates(academic_year_id):
            return []
        wd = date.fromisoformat(on).weekday()
        sections = {s.id: s for s in self.sections_for_year(academic_year_id)}
        subs = {(s.section_id, s.period): s for s in self._subs_where(
            "academic_year_id=? AND date=? AND status<>'cancelled'", (academic_year_id, on))}
        lost = {(l.section_id, l.period): l for l in self._lost_where("academic_year_id=? AND date=?",
                                                                      (academic_year_id, on))}
        exams = self._exam_day(academic_year_id, on)
        rows: list[dict[str, Any]] = []
        for e in self.timetable_for_year(academic_year_id):
            if e.day_of_week != wd:
                continue
            sec = sections.get(e.section_id)
            if sec is None:
                continue
            row = {"sectionId": e.section_id, "period": e.period, "subjectId": e.subject_id,
                   "teacherId": e.teacher_id, "coTeacherId": e.co_teacher_id, "roomId": e.room_id,
                   "kind": "regular", "substitutionId": None, "lostPeriodId": None, "note": None}
            s = subs.get((e.section_id, e.period))
            lp = lost.get((e.section_id, e.period))
            if s is None and lp is None and e.co_teacher_id:
                # Co-teaching (SCH-8): one of the two away is no substitution;
                # the other takes the class alone.
                main_away = e.teacher_id and self._on_leave(e.teacher_id, on, sec, e.period)
                co_away = self._on_leave(e.co_teacher_id, on, sec, e.period)
                if main_away and not co_away:
                    row.update(kind="co_teacher", teacherId=e.co_teacher_id, coTeacherId=None,
                               note="the co-teacher takes the class")
                elif co_away:
                    row["coTeacherId"] = None
            paper = exams.sits(e.section_id, self._cover_times(sec).get(e.period))
            if paper:
                row.update(kind="exam", teacherId=None, coTeacherId=None, note=paper,
                           substitutionId=s.id if s is not None else None)
            elif s is not None:
                row["substitutionId"] = s.id
                if s.status in SUB_TAKEN and s.mode == "substitute":
                    row.update(kind="substitute", teacherId=s.substitute_id)
                elif s.status in SUB_TAKEN or s.mode == "supervised":
                    row.update(kind="supervised", teacherId=s.substitute_id)
                elif s.mode in ("combined", "lost"):
                    row.update(kind=s.mode, teacherId=None)
                else:
                    row.update(kind="uncovered", teacherId=None)
                row["coTeacherId"] = None
            elif lp is not None:
                row.update(kind="lost", teacherId=None, note=lp.reason)
            if lp is not None:
                row["lostPeriodId"] = lp.id
            if section_id and e.section_id != section_id:
                continue
            if teacher_id:
                mine = teacher_id in (e.teacher_id, e.co_teacher_id)
                # Their class in the exam hall is shown as that, not as away.
                if mine and not paper and teacher_id not in (row["teacherId"], row["coTeacherId"]):
                    row = {**row, "kind": "away", "note": "on leave; see the handover" if s else row["note"]}
                elif not mine and row["teacherId"] != teacher_id:
                    continue
                elif not mine and s is not None:
                    # The cover teacher reads what the absent teacher left for
                    # them: the note was stored on the leave and reached no one
                    # (v3 audit N-4-3, N-3-13).
                    row = {**row, "note": self.handover_for(s) or row["note"]}
            rows.append(row)
        for lp in self._lost_where("academic_year_id=? AND compensation_date=? AND status='compensated'",
                                   (academic_year_id, on)):
            alloc = self.allocation_for(lp.section_id, lp.subject_id)
            teacher = alloc.teacher_id if alloc else None
            if section_id and lp.section_id != section_id:
                continue
            if teacher_id and teacher != teacher_id:
                continue
            rows.append({"sectionId": lp.section_id, "period": lp.compensation_period, "subjectId": lp.subject_id,
                         "teacherId": teacher, "coTeacherId": None, "roomId": None, "kind": "makeup",
                         "substitutionId": None,
                         "lostPeriodId": lp.id, "note": f"makes up {lp.date} period {lp.period}"})
        rows.sort(key=lambda r: (r["period"], r["sectionId"]))
        return rows

    # ---------------- the principal's summary (ADM-2) ----------------

    def cover_summary(self, academic_year_id: str, start: str, end: str) -> dict[str, Any]:
        subs = [s for s in self.substitutions_between(academic_year_id, start, end) if s.status != "cancelled"]
        lost = [l for l in self.lost_periods_for_year(academic_year_id) if start <= l.date <= end]
        debt: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for l in lost:
            debt[(l.section_id, l.subject_id)][l.status] += 1
        return {
            **count_cover(subs, lost),
            "debt": [{"sectionId": sid, "subjectId": subj, "owed": c["owed"], "compensated": c["compensated"],
                      "waived": c["waived"]} for (sid, subj), c in sorted(debt.items())],
        }


def count_cover(subs: list[Substitution], lost: list[LostPeriod]) -> dict[str, int]:
    """The cover figures the principal's summary and the term report both
    show (ADM-2), counted here once so the two cannot disagree. `subs`
    leaves out cancelled substitutions.

    Filled: a teacher of the subject has the period. Supervised: someone
    else sits with the class, so the subject is not taught and the period
    is also among the lost ones (and owed until made up). Which of the two
    a period is was decided when the person was put on it (_propose), so a
    supervisor never counts as filled (v3 audit N-3-9)."""
    return {
        "substitutionsRequested": len(subs),
        "filled": sum(1 for s in subs if s.status in SUB_TAKEN and s.mode == "substitute"),
        "supervised": sum(1 for s in subs if s.mode == "supervised"),
        "unfilled": sum(1 for s in subs if s.status in SUB_OPEN),
        "lost": len(lost),
        "compensated": sum(1 for l in lost if l.status == "compensated"),
        "owed": sum(1 for l in lost if l.status == "owed"),
    }
