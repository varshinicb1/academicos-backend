"""Leave, substitution and compensation -- keeping the timetable working when a
teacher is away or a day is lost (REQUIREMENTS SCH-5, SCH-6, SCH-7, SCH-8).

- A LeaveRequest (full day, first or second half, or named periods; planned or
  same-day) is approved or rejected by the principal (or a delegated admin).
- Approval finds every period the teacher misses: their own timetabled periods
  on those days and any substitution duty they had taken. Each gets a
  Substitution, and the engine proposes a substitute. The candidate must be
  free at that time by clock time, not on leave then, and under the day's
  maximum. Candidates are ranked: the same subject and grade first, then the
  same subject, then someone who teaches that section, then the fewest
  substitutions this week, then the lightest day. The principal confirms or
  picks another; the substitute accepts or declines.
- A period is LOST when nobody who teaches the subject takes it: nobody is
  free, the principal runs it as supervised study or combines sections, or a
  closure (a short-notice holiday, an event, an exam day) cancels the day. A
  lost period is a debt per section and subject. The engine proposes make-up
  periods (the section's free periods, when its subject teacher is free and
  under their day's maximum) and records the recovery.

Dates are the school's local dates (IST); times compare by the section's bell,
so a clash across two bells is still a clash.

CurriculumStore mixes this class in, like SchoolModelMixin.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

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
"""

LEAVE_KINDS = ("full_day", "first_half", "second_half", "periods")
# open: needs a substitute; proposed: the engine or principal chose one, not yet
# accepted; accepted: the substitute took it; declined: they said no (open again);
# resolved: the principal ran it without a substitute (mode says how); cancelled:
# the leave was withdrawn.
SUB_OPEN = ("open", "declined")
SUB_TAKEN = ("proposed", "accepted")
MODES = ("substitute", "supervised", "combined", "lost")
MAX_PER_DAY = 7


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


class CoverError(ValueError):
    """The route's 422 (a bad request) -- see also KeyError (404)."""


class CoverMixin:

    # ---------------- helpers ----------------

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

    def affected_periods(self, leave: LeaveRequest) -> list[tuple[str, Any, int, str]]:
        """(date, section, period, subject_id) the teacher would have taught:
        their own timetabled periods on the leave's working days, and any
        substitution duty they had taken."""
        working = self._working_dates(leave.academic_year_id)
        own = [e for e in self.timetable_for_year(leave.academic_year_id) if e.teacher_id == leave.teacher_id]
        sections = {s.id: s for s in self.sections_for_year(leave.academic_year_id)}
        out = []
        for d in _dates(leave.start_date, leave.end_date):
            if d not in working:
                continue
            wd = date.fromisoformat(d).weekday()
            for e in own:
                sec = sections.get(e.section_id)
                if sec is not None and e.day_of_week == wd and self._leave_covers(leave, sec, e.period):
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
                    if best:
                        self._exec("UPDATE substitutions SET substitute_id=?, status='proposed', mode='substitute', "
                                   "updated_at=? WHERE id=?", (best[0].teacher_id, _now(), sub.id))
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
            self._exec("UPDATE leave_requests SET status='cancelled' WHERE id=?", (leave_id,))
            self._exec("UPDATE substitutions SET status='cancelled', updated_at=? WHERE leave_id=? AND date>=? "
                       "AND status<>'resolved'", (_now(), leave_id, today))
            leave.status = "cancelled"
        self._commit()
        return leave

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
                    sections: dict, exclude_sub: Optional[str] = None) -> list[tuple[int, int]]:
        """Clock spans a teacher is teaching on a date: their own periods (if
        not on leave then), substitution duties and make-up periods."""
        wd = date.fromisoformat(on).weekday()
        spans = []
        for e in self.timetable_for_year(academic_year_id):
            if e.teacher_id != teacher_id or e.day_of_week != wd:
                continue
            sec = sections.get(e.section_id)
            if sec is None or self._on_leave(teacher_id, on, sec, e.period):
                continue
            covered = self._subs_where("date=? AND section_id=? AND period=? AND status<>'cancelled'",
                                       (on, e.section_id, e.period))
            if covered and covered[0].absent_teacher_id == teacher_id:
                continue
            t = self._cover_times(sec).get(e.period)
            if t:
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
            if not a.teacher_id:
                continue
            sec = sections.get(a.section_id)
            teaches[a.teacher_id].add((subject_names.get(a.subject_id, ""), sec.grade_id if sec else ""))
            if a.section_id == sub.section_id:
                in_section.add(a.teacher_id)
        d = date.fromisoformat(sub.date)
        week_start = (d - timedelta(days=d.weekday())).isoformat()
        week_end = (d + timedelta(days=6 - d.weekday())).isoformat()
        this_week = Counter(s.substitute_id for s in self._subs_where(
            "academic_year_id=? AND date>=? AND date<=? AND status IN ('proposed','accepted')",
            (sub.academic_year_id, week_start, week_end)) if s.id != sub.id)
        out: list[Candidate] = []
        for t in teaches:
            if t == sub.absent_teacher_id:
                continue
            if self._on_leave(t, sub.date, section, sub.period):
                continue
            spans = self._busy_spans(t, sub.date, sub.academic_year_id, sections, exclude_sub=sub.id)
            if any(s < slot[1] and slot[0] < e for s, e in spans):
                continue
            if len(spans) >= MAX_PER_DAY:
                continue
            name = subject.name if subject else ""
            same_subject_grade = (name, section.grade_id) in teaches[t]
            same_subject = any(n == name for n, _ in teaches[t])
            score, reasons = 0, []
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
        out.sort(key=lambda c: (-c.score, c.teacher_id))
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
                qualified = allowed[substitute_id].qualified
                self._exec("UPDATE substitutions SET substitute_id=?, status='proposed', mode=?, note=?, "
                           "updated_at=? WHERE id=?",
                           (substitute_id, "substitute" if qualified else "supervised", note, _now(), sub_id))
                self._clear_lost(sub)
                if not qualified:
                    lost = self._record_lost(sub, reason="subject not taught (supervised)")
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
                self._exec("UPDATE substitutions SET status='declined', substitute_id=NULL, updated_at=? "
                           "WHERE id=?", (_now(), sub_id))
                nxt = [c for c in self.substitute_candidates(sub_id) if c.teacher_id != teacher_id]
                if nxt:
                    self._exec("UPDATE substitutions SET substitute_id=?, status='proposed', updated_at=? "
                               "WHERE id=?", (nxt[0].teacher_id, _now(), sub_id))
        self._commit()
        return self.get_substitution(sub_id)

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
        return self._insert_lost(sub.school_id, sub.academic_year_id, sub.section_id, sub.subject_id,
                                 sub.date, sub.period, reason, sub.id)

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

    def _clear_lost(self, sub: Substitution) -> None:
        """A period that now has a teacher of its subject is no longer lost."""
        self._exec("DELETE FROM lost_periods WHERE date=? AND section_id=? AND period=? AND status='owed' "
                   "AND source_id=?", (sub.date, sub.section_id, sub.period, sub.id))

    def lost_periods_for_year(self, academic_year_id: str, status: Optional[str] = None) -> list[LostPeriod]:
        if status:
            return self._lost_where("academic_year_id=? AND status=?", (academic_year_id, status))
        return self._lost_where("academic_year_id=?", (academic_year_id,))

    def get_lost_period(self, lost_id: str) -> Optional[LostPeriod]:
        rows = self._lost_where("id=?", (lost_id,))
        return rows[0] if rows else None

    def declare_closure(self, academic_year_id: str, on: str, *, reason: str) -> list[LostPeriod]:
        """A day that stops being taught at short notice (or after the fact):
        every section's timetabled periods that day are lost periods. The
        caller adds the holiday to the calendar and reflows the plans."""
        year = self.get_academic_year(academic_year_id)
        if year is None:
            raise KeyError(academic_year_id)
        if not year.start_date <= on <= year.end_date:
            raise CoverError("the date is outside this academic year")
        wd = date.fromisoformat(on).weekday()
        out = []
        with self._conn_lock:
            for e in self.timetable_for_year(academic_year_id):
                if e.day_of_week != wd:
                    continue
                if self._lost_where("date=? AND section_id=? AND period=?", (on, e.section_id, e.period)):
                    continue
                out.append(self._insert_lost(year.school_id, academic_year_id, e.section_id, e.subject_id,
                                             on, e.period, f"closure: {reason}", None))
            self._exec("UPDATE substitutions SET status='cancelled', updated_at=? WHERE academic_year_id=? "
                       "AND date=? AND status<>'resolved'", (_now(), academic_year_id, on))
        self._commit()
        return out

    def compensation_options(self, lost_id: str, *, today: str, days_ahead: int = 21,
                             limit: int = 20) -> list[tuple[str, int]]:
        """(date, period) where the section is free, its subject teacher is
        free and under the day's maximum, from today on."""
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
        out = []
        for i in range(days_ahead):
            d = (start + timedelta(days=i)).isoformat()
            if d not in working:
                continue
            wd = date.fromisoformat(d).weekday()
            taken = {e.period for e in section_week if e.day_of_week == wd}
            taken |= {x.compensation_period for x in self._lost_where(
                "section_id=? AND compensation_date=? AND status='compensated'", (lp.section_id, d))}
            busy = self._busy_spans(teacher, d, lp.academic_year_id, sections) if teacher else []
            if teacher and len(busy) >= MAX_PER_DAY:
                continue
            for p, (s, e) in sorted(times.items()):
                if p in taken:
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

    # ---------------- the day (what actually happens) ----------------

    def day_view(self, academic_year_id: str, on: str, *, section_id: Optional[str] = None,
                 teacher_id: Optional[str] = None) -> list[dict[str, Any]]:
        """One day as it will run: the week's periods, with that day's
        substitutions, lost periods and make-up periods applied. Filter by a
        section, or by a teacher (their periods, their duties, their make-ups;
        their periods on leave marked so)."""
        if on not in self._working_dates(academic_year_id):
            return []
        wd = date.fromisoformat(on).weekday()
        sections = {s.id: s for s in self.sections_for_year(academic_year_id)}
        subs = {(s.section_id, s.period): s for s in self._subs_where(
            "academic_year_id=? AND date=? AND status<>'cancelled'", (academic_year_id, on))}
        lost = {(l.section_id, l.period): l for l in self._lost_where("academic_year_id=? AND date=?",
                                                                      (academic_year_id, on))}
        rows: list[dict[str, Any]] = []
        for e in self.timetable_for_year(academic_year_id):
            if e.day_of_week != wd:
                continue
            sec = sections.get(e.section_id)
            if sec is None:
                continue
            row = {"sectionId": e.section_id, "period": e.period, "subjectId": e.subject_id,
                   "teacherId": e.teacher_id, "roomId": e.room_id, "kind": "regular", "substitutionId": None,
                   "lostPeriodId": None, "note": None}
            s = subs.get((e.section_id, e.period))
            lp = lost.get((e.section_id, e.period))
            if s is not None:
                row["substitutionId"] = s.id
                if s.status in SUB_TAKEN and s.mode == "substitute":
                    row.update(kind="substitute", teacherId=s.substitute_id)
                elif s.status in SUB_TAKEN or s.mode == "supervised":
                    row.update(kind="supervised", teacherId=s.substitute_id)
                elif s.mode in ("combined", "lost"):
                    row.update(kind=s.mode, teacherId=None)
                else:
                    row.update(kind="uncovered", teacherId=None)
            elif lp is not None:
                row.update(kind="lost", teacherId=None, note=lp.reason)
            if lp is not None:
                row["lostPeriodId"] = lp.id
            if section_id and e.section_id != section_id:
                continue
            if teacher_id:
                if e.teacher_id == teacher_id and row["teacherId"] != teacher_id:
                    row = {**row, "kind": "away", "note": "on leave; see the handover" if s else row["note"]}
                elif row["teacherId"] != teacher_id:
                    continue
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
                         "teacherId": teacher, "roomId": None, "kind": "makeup", "substitutionId": None,
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
            "substitutionsRequested": len(subs),
            "filled": sum(1 for s in subs if s.status in SUB_TAKEN and s.mode == "substitute"),
            "supervised": sum(1 for s in subs if s.mode == "supervised"),
            "unfilled": sum(1 for s in subs if s.status in SUB_OPEN),
            "lost": len(lost),
            "compensated": sum(1 for l in lost if l.status == "compensated"),
            "owed": sum(1 for l in lost if l.status == "owed"),
            "debt": [{"sectionId": sid, "subjectId": subj, "owed": c["owed"], "compensated": c["compensated"],
                      "waived": c["waived"]} for (sid, subj), c in sorted(debt.items())],
        }
