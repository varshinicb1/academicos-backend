"""A school's staff rules, set once by the principal (owner, 2026-10-04:
"automate staff attendance and leave approvals ... and properly integrate
them into the workflows"; SCH-5 held for "no auto-approval rules and no
leave balances").

- **Leave allowance.** Days of leave a teacher has in an academic year. Used
  is the approved leave of the year, pending is what still waits for a
  decision; both count only the year's working days (a holiday inside a
  leave costs nothing), a half day as 0.5 and named periods as their share
  of the school's teaching day.
- **Automatic approval.** When the principal switches it on, a teacher's own
  leave is approved at once if it is no longer than `max_auto_days`, asked at
  least `min_notice_days` ahead, and within what is left of the allowance.
  Approval then does exactly what the principal's does: every missed period
  gets a substitution with a proposed substitute, the substitutes hear, and
  the principal is told (urgently when a period has no one free). Anything
  else waits for the principal, who is told it is waiting.
- **Check-in.** A teacher's own check-in after the first bell plus
  `check_in_grace_minutes` is recorded as late. With `check_in_reminder`, the
  morning automation reminds a teacher who has not checked in and tells the
  principal who is missing (operations/automations.py, staff_check_in).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Optional

POLICY_SCHEMA = """
CREATE TABLE IF NOT EXISTS staff_policies (
  school_id              TEXT PRIMARY KEY,
  auto_approve           INTEGER NOT NULL DEFAULT 0,
  max_auto_days          REAL NOT NULL DEFAULT 2,
  min_notice_days        INTEGER NOT NULL DEFAULT 1,
  annual_allowance_days  REAL NOT NULL DEFAULT 12,
  check_in_grace_minutes INTEGER NOT NULL DEFAULT 10,
  check_in_reminder      INTEGER NOT NULL DEFAULT 1,
  updated_by             TEXT,
  updated_at             TEXT
);
"""

HALF_DAY_KINDS = ("first_half", "second_half")


@dataclass
class StaffPolicy:
    auto_approve: bool = False
    max_auto_days: float = 2
    min_notice_days: int = 1
    annual_allowance_days: float = 12
    check_in_grace_minutes: int = 10
    check_in_reminder: bool = True
    updated_by: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class LeaveBalance:
    allowance: float
    used: float
    pending: float

    @property
    def left(self) -> float:
        return max(0.0, self.allowance - self.used - self.pending)


def auto_decision(policy: StaffPolicy, *, days: float, left_before: float, start_date: str,
                  today: str) -> tuple[bool, str]:
    """Whether a teacher's own leave is approved without the principal, and
    why not when it is not. `left_before` is the allowance left before this
    leave counts."""
    if not policy.auto_approve:
        return False, "waits for the principal"
    if days > policy.max_auto_days:
        return False, f"longer than {policy.max_auto_days:g} days, so it waits for the principal"
    notice = (date.fromisoformat(start_date) - date.fromisoformat(today)).days
    if notice < policy.min_notice_days:
        return False, (f"asked less than {policy.min_notice_days} day{'s' if policy.min_notice_days != 1 else ''} "
                       "ahead, so it waits for the principal")
    if days > left_before:
        return False, f"more than the {left_before:g} days of leave left, so it waits for the principal"
    return True, "approved automatically under the school's leave rule"


class StaffPolicyMixin:
    """Added to CurriculumStore."""

    def staff_policy(self, school_id: str) -> StaffPolicy:
        r = self._fetchone("SELECT * FROM staff_policies WHERE school_id=?", (school_id,))
        if r is None:
            return StaffPolicy()
        return StaffPolicy(auto_approve=bool(r["auto_approve"]), max_auto_days=float(r["max_auto_days"]),
                           min_notice_days=int(r["min_notice_days"]),
                           annual_allowance_days=float(r["annual_allowance_days"]),
                           check_in_grace_minutes=int(r["check_in_grace_minutes"]),
                           check_in_reminder=bool(r["check_in_reminder"]), updated_by=r["updated_by"],
                           updated_at=r["updated_at"])

    def set_staff_policy(self, school_id: str, policy: StaffPolicy, *, updated_by: str) -> StaffPolicy:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p = asdict(policy)
        with self._conn_lock:
            self._exec(
                "INSERT INTO staff_policies (school_id, auto_approve, max_auto_days, min_notice_days, "
                "annual_allowance_days, check_in_grace_minutes, check_in_reminder, updated_by, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(school_id) DO UPDATE SET "
                "auto_approve=excluded.auto_approve, max_auto_days=excluded.max_auto_days, "
                "min_notice_days=excluded.min_notice_days, annual_allowance_days=excluded.annual_allowance_days, "
                "check_in_grace_minutes=excluded.check_in_grace_minutes, "
                "check_in_reminder=excluded.check_in_reminder, updated_by=excluded.updated_by, "
                "updated_at=excluded.updated_at",
                (school_id, int(p["auto_approve"]), p["max_auto_days"], p["min_notice_days"],
                 p["annual_allowance_days"], p["check_in_grace_minutes"], int(p["check_in_reminder"]),
                 updated_by, now))
            self._commit()
        return self.staff_policy(school_id)

    def teaching_periods_a_day(self, academic_year_id: str) -> int:
        bells = self.bell_schedules_for_year(academic_year_id)
        bell = next((b for b in bells if b.is_default), bells[0] if bells else None)
        return (bell.teaching_periods if bell else 0) or 8

    def first_bell(self, academic_year_id: str) -> Optional[str]:
        """The earliest teaching period's start ("08:00") across the year's
        bells, or None with no bell."""
        starts = [s.start for b in self.bell_schedules_for_year(academic_year_id) for s in b.slots
                  if s.kind == "teaching" and s.start]
        return min(starts) if starts else None

    def leave_days(self, leave) -> float:
        """What a leave costs: its working days, a half day as 0.5 and named
        periods as their share of the teaching day."""
        working = set(self._school_days(leave.academic_year_id).working)
        days = sum(1 for d in _dates(leave.start_date, leave.end_date) if d in working)
        if leave.kind in HALF_DAY_KINDS:
            return days * 0.5
        if leave.kind == "periods":
            per_day = self.teaching_periods_a_day(leave.academic_year_id)
            return round(days * min(1.0, len(leave.periods) / per_day), 2)
        return float(days)

    def leave_balance(self, school_id: str, teacher_id: str, academic_year_id: str,
                      *, leaving_out: Optional[str] = None) -> LeaveBalance:
        """The teacher's allowance for the year, what approved leave used and
        what pending leave holds (`leaving_out`: a leave not to count, the one
        being decided)."""
        policy = self.staff_policy(school_id)
        used = pending = 0.0
        for leave in self.leave_for_teacher(teacher_id):
            if leave.academic_year_id != academic_year_id or leave.id == leaving_out:
                continue
            if leave.status == "approved":
                used += self.leave_days(leave)
            elif leave.status == "pending":
                pending += self.leave_days(leave)
        return LeaveBalance(allowance=policy.annual_allowance_days, used=round(used, 2), pending=round(pending, 2))


def _dates(start: str, end: str):
    from datetime import timedelta
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    while d <= last:
        yield d.isoformat()
        d += timedelta(days=1)
