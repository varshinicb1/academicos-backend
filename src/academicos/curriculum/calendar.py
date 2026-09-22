"""Academic calendar arithmetic -- Sec 10, 27-29 of the master build prompt:
Academic Year -> Working Days -> Holidays -> Period Duration -> Teaching-Time
Estimate.

curriculum/store.py already persists AcademicYear/Calendar/Holiday/
PeriodConfiguration (real CRUD, real tests) -- what was missing was turning
that config into an actual list of real teaching days, and then distributing
real instructional minutes across real Topics/Subtopics. Those two
computations are what this module adds.

Deliberately does NOT touch syllabus/timetable.py's existing
generate_timetable(): that function operates on cbse_syllabus.py's static
JSON (SyllabusDocument/UnitAllocation), a read-only reference dataset kept
separate from the operational curriculum_store by design (see
docs/ACADEMIC_DATA_MODEL.md). compute_teaching_time_estimates() below uses
the same real signal (a Unit's real CBSE marks-weightage) but reads it from
curriculum_store's own seeded Unit rows and writes real, persisted
TeachingTimeEstimate rows against real Subtopic ids -- the two
representations stay separate; only the marks-weightage idea is shared.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .models import TeachingTimeEstimate
from .store import CurriculumStore

_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Holiday kinds. Every kind the web admin offers (public / school / emergency,
# principal_admin_page.dart) closes the school; until 2026-09-22 only
# 'holiday' and 'unexpected_closure' were counted, so a Dussehra break
# entered from the UI was stored, listed with a 200 and removed zero
# teaching days. 'event' (Annual Day, a PTM) is the one kind that marks a
# day WITHOUT closing the school -- it stays a working day on purpose.
CLOSURE_KINDS = ("holiday", "public", "school", "emergency", "unexpected_closure")
NON_CLOSURE_KINDS = ("event",)
HOLIDAY_KINDS = CLOSURE_KINDS + NON_CLOSURE_KINDS

# Named alternate-Saturday rules, beside the digit form ("2nd,4th", "1,3").
# 'second_fourth' is what the web admin's dropdown sends; the digit-only
# parser used to match none of its tokens, so no Saturday ever came off.
_NAMED_SATURDAY_RULES: dict[str, frozenset[int]] = {
    "none": frozenset(),
    "all": frozenset({1, 2, 3, 4, 5}),
    "second_fourth": frozenset({2, 4}),
    "first_third": frozenset({1, 3}),
}
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd", 4: "th", 5: "th"}
ACCEPTED_SATURDAY_RULES_TEXT = (
    "'none', 'all', 'second_fourth', 'first_third', or Saturdays of the month as "
    "ordinals such as '2nd,4th'")


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _date_range(start: str, end: str) -> list[str]:
    """Every ISO date from start to end, inclusive of both ends."""
    s, e = _parse_date(start), _parse_date(end)
    out = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _nth_weekday_of_month(d: date) -> int:
    """1 for the first occurrence of d's weekday in its month, 2 for the
    second, etc. -- used to resolve "2nd,4th Saturday" against a real date."""
    return (d.day - 1) // 7 + 1


def normalize_weekly_off_days(days: list[str]) -> list[str]:
    """Lower-cased, de-duplicated weekday names; ValueError naming the
    accepted values for anything else. Validated up front because an
    unknown name ('Sun') used to be stored, then break working-days with
    'tuple.index(x): x not in tuple' -- and a calendar cannot be re-created."""
    out: list[str] = []
    for raw in days:
        name = str(raw).strip().lower()
        if name not in _WEEKDAY_NAMES:
            raise ValueError(
                f"unknown weekly off day {raw!r} -- use full weekday names: "
                + ", ".join(_WEEKDAY_NAMES))
        if name not in out:
            out.append(name)
    return out


def parse_alternate_saturday_rule(rule: Optional[str]) -> frozenset[int]:
    """Which Saturdays of the month (1st..5th) are off. Accepts a named rule
    or comma-separated ordinals ('2nd,4th', '2,4'). An unrecognised rule is
    a ValueError: silently reading it as "no Saturdays off" is how the web
    admin's 'second_fourth' went unnoticed."""
    text = (rule or "none").strip().lower()
    if not text:
        return frozenset()
    if text in _NAMED_SATURDAY_RULES:
        return _NAMED_SATURDAY_RULES[text]
    wanted: set[int] = set()
    for tok in text.split(","):
        tok = tok.strip()
        n = int(tok[0]) if tok[:1].isdigit() else 0
        if not 1 <= n <= 5 or tok not in (str(n), f"{n}{_ORDINAL_SUFFIX[n]}"):
            raise ValueError(
                f"unrecognised alternate Saturday rule {rule!r} -- use "
                + ACCEPTED_SATURDAY_RULES_TEXT)
        wanted.add(n)
    return frozenset(wanted)


def normalize_alternate_saturday_rule(rule: Optional[str]) -> str:
    """Validates the rule and returns it trimmed and lower-cased, in the
    caller's own form: the web admin reads it back into a dropdown whose
    values are the named forms, so 'second_fourth' is not rewritten."""
    parse_alternate_saturday_rule(rule)
    return (rule or "none").strip().lower() or "none"


def validate_holiday(*, date_: str, end_date: Optional[str], kind: str,
                     year_start: str, year_end: str) -> None:
    """ValueError for anything a holiday row must never hold: an unknown
    kind (it would be listed but never counted), an unparseable date
    ('next monday' was accepted), an end before the start (a 500 from the
    store before 2026-09-22), or a day outside the academic year
    (2030-01-01 was accepted for a 2026-27 year)."""
    if kind not in HOLIDAY_KINDS:
        raise ValueError(
            f"unknown holiday kind {kind!r} -- closures: {', '.join(CLOSURE_KINDS)}; "
            f"a marked day that still teaches: {', '.join(NON_CLOSURE_KINDS)}")
    parsed = {}
    for field_name, value in (("date", date_), ("end_date", end_date)):
        if value is None:
            continue
        try:
            parsed[field_name] = _parse_date(value)
        except (TypeError, ValueError):
            raise ValueError(f"holiday {field_name} {value!r} is not a YYYY-MM-DD date") from None
        # Python 3.11+ fromisoformat also takes '20261005' and '2026-W41-1'.
        # The raw string is what gets stored, and working_days_for_year matches
        # holidays against d.isoformat(), so such a holiday was listed but
        # closed zero days (261 working days before and after), and a mixed
        # '20261006' .. '2026-10-08' pair 500'd on the store's string compare.
        if parsed[field_name].isoformat() != value:
            raise ValueError(f"holiday {field_name} {value!r} is not a YYYY-MM-DD date")
    start = parsed["date"]
    end = parsed.get("end_date", start)
    if end < start:
        raise ValueError(f"holiday end_date {end_date} is before its date {date_}")
    if start < _parse_date(year_start) or end > _parse_date(year_end):
        raise ValueError(
            f"holiday {date_}{f' .. {end_date}' if end_date else ''} falls outside the "
            f"academic year ({year_start} .. {year_end})")


@dataclass(frozen=True)
class WorkingDaysResult:
    total_days: int
    working_days: int
    weekly_off_count: int
    alternate_saturday_off_count: int
    holiday_count: int
    dates: tuple[str, ...]   # ISO dates, real working days only


def compute_working_days(*, start_date: str, end_date: str, weekly_off_days: list[str],
                         alternate_saturday_rule: str, holiday_dates: set[str]) -> WorkingDaysResult:
    """Pure function, no store dependency -- the actual §10 arithmetic.
    Precedence per day: weekly-off > alternate-Saturday-off > holiday >
    working. A date can only be one of those, so double counting is
    impossible by construction."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    off_weekdays = {_WEEKDAY_NAMES.index(w) for w in normalize_weekly_off_days(weekly_off_days)}
    saturdays_off = parse_alternate_saturday_rule(alternate_saturday_rule)

    total_days = 0
    working: list[str] = []
    weekly_off_count = 0
    alt_sat_count = 0
    holiday_count = 0

    d = start
    while d <= end:
        total_days += 1
        if d.weekday() in off_weekdays:
            weekly_off_count += 1
        elif d.weekday() == 5 and _nth_weekday_of_month(d) in saturdays_off:
            alt_sat_count += 1
        elif d.isoformat() in holiday_dates:
            holiday_count += 1
        else:
            working.append(d.isoformat())
        d += timedelta(days=1)

    return WorkingDaysResult(
        total_days=total_days, working_days=len(working), weekly_off_count=weekly_off_count,
        alternate_saturday_off_count=alt_sat_count, holiday_count=holiday_count,
        dates=tuple(working),
    )


def working_days_for_year(store: CurriculumStore, academic_year_id: str) -> WorkingDaysResult:
    """Store-backed convenience: reads the real, persisted AcademicYear +
    Calendar + Holiday rows for a school and computes the real result."""
    year = store.get_academic_year(academic_year_id)
    if year is None:
        raise ValueError(f"no such academic year: {academic_year_id}")
    cal = store.get_calendar_for_year(academic_year_id)
    if cal is None:
        raise ValueError(
            f"academic year {academic_year_id} has no calendar configured yet -- "
            "create one first (POST .../calendar)")
    holidays = store.holidays_for_calendar(cal.id)
    # "event" markers (e.g. Annual Day) don't remove a teaching day; every
    # closure kind does (CLOSURE_KINDS above). A holiday with end_date set (a real multi-day block --
    # a 30-45 day summer break) expands to every date in the inclusive range,
    # not just its start date.
    holiday_dates: set[str] = set()
    for h in holidays:
        if h.kind not in CLOSURE_KINDS:
            continue
        if h.end_date is None:
            holiday_dates.add(h.date)
        else:
            holiday_dates.update(_date_range(h.date, h.end_date))
    return compute_working_days(
        start_date=year.start_date, end_date=year.end_date,
        weekly_off_days=cal.weekly_off_days, alternate_saturday_rule=cal.alternate_saturday_rule,
        holiday_dates=holiday_dates,
    )


@dataclass(frozen=True)
class TeachingTimeComputationResult:
    academic_year_id: str
    book_id: str
    periods_per_week: int
    period_minutes: int
    calendar_weeks: int
    total_subject_periods: int
    total_instructional_minutes: int
    units_skipped_no_subtopics: tuple[str, ...]
    estimates: tuple[TeachingTimeEstimate, ...]


def compute_teaching_time_estimates(
    store: CurriculumStore, *, academic_year_id: str, book_id: str, periods_per_week: int,
    approved_by: Optional[str] = None,
) -> TeachingTimeComputationResult:
    """Distributes a subject's real, calendar-grounded instructional time
    across its real Subtopics, proportional to each Unit's real CBSE
    marks-weightage (same signal a human HOD uses, and the same one
    syllabus/timetable.py already applies at Unit level -- applied here to
    curriculum_store's live Unit/Chapter/Topic/Subtopic hierarchy instead).

    `periods_per_week` is the subject's own weekly period count (e.g.
    "Science gets 6 periods a week") -- a real per-school scheduling
    decision this module doesn't yet persist as its own entity (no
    `SubjectPeriodAllocation` table exists), so it's taken as an explicit
    argument, same as syllabus/timetable.py's own `periods_per_week` param.
    Documented here rather than silently assumed.

    `calendar_weeks` (the multiplier) comes from the real academic year's
    date span (ceil(days/7)) rather than the raw working-day count: a
    school's stated "N periods per week" already presumes a normal week's
    shape, so the correct multiplier is how many such weeks the year spans,
    not the smaller raw teaching-day count once holidays are subtracted
    (that figure -- working_days_for_year() -- is the correct answer to a
    *different* question, "how many real teaching days exist this year",
    and is exposed separately rather than folded into this estimate through
    a shakier formula).

    Within a Unit, real CBSE curriculum only weights at unit granularity
    (confirmed: academicos-data/syllabus/*.json carries marks per unit, not
    per chapter/topic/subtopic) -- so a unit's periods are split evenly
    across its subtopics, the same "no finer real signal exists, don't
    fabricate one" stance seed_cbse10.py already takes for chapters without
    an explicit list.
    """
    if periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")

    year = store.get_academic_year(academic_year_id)
    if year is None:
        raise ValueError(f"no such academic year: {academic_year_id}")
    period_cfg = store.period_configuration_for_year(academic_year_id)
    if period_cfg is None:
        raise ValueError(
            f"academic year {academic_year_id} has no period configuration set yet -- "
            "create one first (POST .../period-configuration)")

    start = _parse_date(year.start_date)
    end = _parse_date(year.end_date)
    calendar_weeks = max(1, -(-(end - start).days // 7))  # ceil division

    total_subject_periods = periods_per_week * calendar_weeks
    total_instructional_minutes = total_subject_periods * period_cfg.period_minutes

    units = store.units_for_book(book_id)
    total_marks = sum(u.marks or 0 for u in units)
    if total_marks <= 0:
        raise ValueError(
            f"book {book_id} has no unit marks-weightage to distribute by -- "
            "seed real Unit.marks first")

    # Largest-remainder rounding, same shape as syllabus/timetable.py's
    # generate_timetable(), so unit minutes sum exactly to
    # total_instructional_minutes rather than drifting.
    raw = [(u, total_instructional_minutes * (u.marks or 0) / total_marks) for u in units]
    floors = [(u, int(v)) for u, v in raw]
    allocated = sum(v for _, v in floors)
    remainder = total_instructional_minutes - allocated
    fractional_order = sorted(range(len(raw)), key=lambda i: raw[i][1] - floors[i][1], reverse=True)
    minutes_by_unit = {u.id: v for u, v in floors}
    for i in fractional_order[:remainder]:
        u = raw[i][0]
        minutes_by_unit[u.id] += 1

    estimates: list[TeachingTimeEstimate] = []
    skipped: list[str] = []
    for unit in units:
        chapters = store.chapters_for_unit(unit.id)
        subtopic_ids: list[str] = []
        for ch in chapters:
            for topic in store.topics_for_chapter(ch.id):
                for st in store.subtopics_for_topic(topic.id):
                    subtopic_ids.append(st.id)
        if not subtopic_ids:
            skipped.append(unit.name)
            continue

        unit_minutes = minutes_by_unit[unit.id]
        base = unit_minutes // len(subtopic_ids)
        extra = unit_minutes - base * len(subtopic_ids)
        for i, subtopic_id in enumerate(subtopic_ids):
            minutes = base + (1 if i < extra else 0)
            periods = round(minutes / period_cfg.period_minutes) if period_cfg.period_minutes else None
            existing = store.teaching_time_estimate_for_subtopic(subtopic_id, academic_year_id)
            if existing is not None:
                continue  # idempotent -- a re-run doesn't duplicate/overwrite a prior estimate
            est = store.create_teaching_time_estimate(
                subtopic_id=subtopic_id, academic_year_id=academic_year_id,
                estimated_minutes=minutes, estimated_periods=periods,
                method="marks_weightage_proportional", approved_by=approved_by)
            estimates.append(est)

    return TeachingTimeComputationResult(
        academic_year_id=academic_year_id, book_id=book_id, periods_per_week=periods_per_week,
        period_minutes=period_cfg.period_minutes, calendar_weeks=calendar_weeks,
        total_subject_periods=total_subject_periods,
        total_instructional_minutes=total_instructional_minutes,
        units_skipped_no_subtopics=tuple(skipped), estimates=tuple(estimates),
    )
