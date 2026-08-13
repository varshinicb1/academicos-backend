"""Weekly pacing suggestions, derived from real CBSE marks-weightage.

CBSE's own curriculum document (Section 3.3, "Instructional Time") gives
marks-weightage per unit but explicitly leaves period-by-period timetabling to
each school -- there is no official "hours per unit" figure to consume. This
allocates a school's actual weekly periods proportionally to marks-weightage
(higher marks -> proportionally more class time), which is the same signal a
human HOD uses when building a pacing chart by hand. It is a deterministic,
explainable estimate, not a claim to official CBSE data.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cbse_syllabus import SyllabusDocument, load_syllabus


@dataclass(frozen=True)
class WeekSlot:
    week: int
    unit_name: str
    periods: int


@dataclass(frozen=True)
class UnitAllocation:
    unit_name: str
    marks: int
    suggested_periods: int


@dataclass(frozen=True)
class Timetable:
    subject: str
    grade: int
    periods_per_week: int
    weeks: int
    allocations: tuple[UnitAllocation, ...]
    schedule: tuple[WeekSlot, ...]


def generate_timetable(subject: str, grade: int, *, periods_per_week: int, weeks: int) -> Timetable | None:
    doc: SyllabusDocument | None = load_syllabus(subject, grade)
    if doc is None or periods_per_week <= 0 or weeks <= 0:
        return None

    total_periods = periods_per_week * weeks
    # Largest-remainder rounding so per-unit periods sum exactly to
    # total_periods instead of drifting from naive round()-per-unit.
    raw = [(u, total_periods * u.marks / doc.total_marks) for u in doc.units]
    floors = [(u, int(v)) for u, v in raw]
    allocated = sum(f for _, f in floors)
    remainder = total_periods - allocated
    fractional_order = sorted(range(len(raw)), key=lambda i: raw[i][1] - floors[i][1], reverse=True)
    periods_by_unit = {u.unit_no: p for u, p in floors}
    for i in fractional_order[:remainder]:
        u = raw[i][0]
        periods_by_unit[u.unit_no] += 1

    allocations = tuple(
        UnitAllocation(unit_name=u.name, marks=u.marks, suggested_periods=periods_by_unit[u.unit_no])
        for u in doc.units
    )

    schedule: list[WeekSlot] = []
    week = 1
    remaining_this_week = periods_per_week
    for u in doc.units:
        periods_left = periods_by_unit[u.unit_no]
        while periods_left > 0 and week <= weeks:
            take = min(periods_left, remaining_this_week)
            if take > 0:
                schedule.append(WeekSlot(week=week, unit_name=u.name, periods=take))
            periods_left -= take
            remaining_this_week -= take
            if remaining_this_week == 0:
                week += 1
                remaining_this_week = periods_per_week

    return Timetable(
        subject=subject, grade=grade, periods_per_week=periods_per_week, weeks=weeks,
        allocations=allocations, schedule=tuple(schedule),
    )
