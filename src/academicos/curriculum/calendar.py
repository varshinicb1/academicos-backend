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

from collections import Counter, defaultdict
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


# --------------------------------------------------------------------- #
# Teaching slots: which real periods of the year a subject gets. One rule,
# shared by the teaching-time budget below and by scheduling.py (placing,
# PUSH), so the budget can never promise periods the
# scheduler does not have. Until 2026-09-22 the two used different numbers:
# the budget was periods_per_week x calendar weeks (264 for a 6-period
# subject) while the scheduler had 220 real slots once alternate Saturdays
# and breaks came off -- Social Science ended the year with 35 of 193
# subtopics undated.
# --------------------------------------------------------------------- #

def subject_teaching_slots(working_days: list[date], periods_per_week: int,
                           periods_by_weekday: Optional[dict[int, int]] = None) -> list[date]:
    """One entry per real teaching period, in date order; a day with a
    double period appears twice.

    With `periods_by_weekday` (the school's SubjectTimetableSlot rows,
    counted per weekday: {Mon: 2, Tue: 1, ...}), every working day on a
    timetable weekday contributes that many periods -- the school's actual
    timetable. Until 2026-09-22 the timetable was reduced to a set of
    weekdays, so a Monday double period became one lesson and Mathematics'
    6 periods a week became 5.

    Without it (no timetable entered for this subject), the deterministic
    fallback: group the working days into ISO weeks and take the first
    `periods_per_week` days of each, one period a day. Still a documented
    simplification, not a claim to reproduce a school's real timetable."""
    if periods_by_weekday:
        return [d for d in working_days for _ in range(periods_by_weekday.get(d.weekday(), 0))]
    by_week: dict[tuple[int, int], list[date]] = defaultdict(list)
    for d in working_days:
        iso_year, iso_week, _ = d.isocalendar()
        by_week[(iso_year, iso_week)].append(d)
    selected: list[date] = []
    for key in sorted(by_week):
        selected.extend(sorted(by_week[key])[:periods_per_week])
    return selected


def timetable_periods_by_weekday(store: CurriculumStore, academic_year_id: str,
                                 book_id: str) -> dict[int, int]:
    """{weekday: periods} from this book's subject's timetable; empty when
    the school has not entered one."""
    subject = store.subject_name_for_book(book_id)
    if subject is None:
        return {}
    return dict(Counter(s.day_of_week for s in store.timetable_slots_for_subject(academic_year_id, subject)))


def teaching_slots_for_book(store: CurriculumStore, academic_year_id: str, book_id: str,
                            periods_per_week: int) -> list[date]:
    """subject_teaching_slots() over the year's real working days and this
    book's subject's timetable."""
    wd = working_days_for_year(store, academic_year_id)
    return subject_teaching_slots([date.fromisoformat(s) for s in wd.dates], periods_per_week,
                                  timetable_periods_by_weekday(store, academic_year_id, book_id) or None)


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Splits `total` into integers proportional to `weights` that sum to
    `total` exactly. Equal weights when every weight is zero."""
    if not weights or total <= 0:
        return [0] * len(weights)
    if sum(weights) <= 0:
        weights = [1.0] * len(weights)
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    out = [int(r) for r in raw]
    by_fraction = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in by_fraction[:total - sum(out)]:
        out[i] += 1
    return out


COMPUTED_ESTIMATE_METHOD = "marks_weightage_proportional"


@dataclass(frozen=True)
class TeachingTimeComputationResult:
    academic_year_id: str
    book_id: str
    periods_per_week: int
    period_minutes: int
    calendar_weeks: int
    total_subject_periods: int          # the budget: real teaching periods this year
    total_instructional_minutes: int
    units_skipped_no_subtopics: tuple[str, ...]
    estimates: tuple[TeachingTimeEstimate, ...]   # created (or recomputed) by this run
    periods_allocated: int = 0          # every estimate this book's subtopics now hold
    periods_short: int = 0              # allocated beyond the budget: the syllabus is bigger than the year

    @property
    def fits_in_year(self) -> bool:
        return self.periods_short == 0


def compute_teaching_time_estimates(
    store: CurriculumStore, *, academic_year_id: str, book_id: str, periods_per_week: int,
    approved_by: Optional[str] = None, recompute: bool = False,
) -> TeachingTimeComputationResult:
    """Distributes a subject's real teaching periods for the year across its
    real Subtopics, weighted by each Unit's real CBSE marks (same signal a
    human HOD uses, and the one syllabus/timetable.py applies at Unit level).

    The budget is the number of real teaching periods the subject has this
    year -- teaching_slots_for_book(): the calendar's working days (after
    weekly offs, alternate Saturdays and every closure) x the subject's
    periods on those days. It used to be periods_per_week x calendar weeks,
    which ignored holidays: 264 periods for a 6-period subject whose year
    really holds 220. `periods_per_week` still decides the no-timetable
    fallback; with a timetable the timetable decides, exactly as it does in
    scheduling.

    The split is exact. Every subtopic gets one period (it cannot be taught
    in none), and the rest of the budget goes to units by marks and evenly
    across a unit's subtopics, all by largest remainder -- so the stored
    periods sum to the budget, never above it. Until 2026-09-22 each
    subtopic's minutes were round()-ed to periods on their own: Social
    Science's estimates summed to 323 periods over a 288 budget. The one
    case the sum exceeds the budget is a syllabus with more subtopics than
    the year has periods; `periods_short` says by how much, rather than
    quietly dropping subtopics.

    Idempotent per subtopic: an existing estimate is kept and its periods
    come off the budget first (an admin's hand-set number is never
    clobbered by a re-run). `recompute=True` replaces this function's own
    earlier estimates (method 'marks_weightage_proportional') -- how a
    school fixes estimates computed before a calendar change, or by the
    pre-2026-09-22 formula -- and still keeps every other estimate.

    Within a Unit, CBSE weights only at unit granularity (academicos-data/
    syllabus/*.json carries marks per unit), so a unit's periods are split
    evenly across its subtopics -- "no finer real signal exists, don't
    fabricate one", as seed_cbse10.py does for chapters."""
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

    units = store.units_for_book(book_id)
    if sum(u.marks or 0 for u in units) <= 0:
        raise ValueError(
            f"book {book_id} has no unit marks-weightage to distribute by -- "
            "seed real Unit.marks first")

    start = _parse_date(year.start_date)
    end = _parse_date(year.end_date)
    calendar_weeks = max(1, -(-(end - start).days // 7))  # ceil division; reported, not the budget
    budget = len(teaching_slots_for_book(store, academic_year_id, book_id, periods_per_week))

    skipped: list[str] = []
    to_allocate: list[tuple[object, list[str]]] = []   # (unit, subtopic ids without a kept estimate)
    existing_by_subtopic: dict[str, TeachingTimeEstimate] = {}
    kept_periods = 0
    for unit in units:
        subtopic_ids = [st.id for ch in store.chapters_for_unit(unit.id)
                        for topic in store.topics_for_chapter(ch.id)
                        for st in store.subtopics_for_topic(topic.id)]
        if not subtopic_ids:
            skipped.append(unit.name)
            continue
        open_ids: list[str] = []
        for subtopic_id in subtopic_ids:
            est = store.teaching_time_estimate_for_subtopic(subtopic_id, academic_year_id)
            if est is not None and not (recompute and est.method == COMPUTED_ESTIMATE_METHOD):
                kept_periods += est.estimated_periods or 0
                continue
            if est is not None:
                existing_by_subtopic[subtopic_id] = est
            open_ids.append(subtopic_id)
        if open_ids:
            to_allocate.append((unit, open_ids))

    n_open = sum(len(ids) for _, ids in to_allocate)
    extra = max(0, budget - kept_periods - n_open)
    unit_extras = _largest_remainder(extra, [float(u.marks or 0) for u, _ in to_allocate])

    estimates: list[TeachingTimeEstimate] = []
    for (unit, open_ids), unit_extra in zip(to_allocate, unit_extras):
        base, rem = divmod(unit_extra, len(open_ids))
        for i, subtopic_id in enumerate(open_ids):
            periods = 1 + base + (1 if i < rem else 0)
            minutes = periods * period_cfg.period_minutes
            prior = existing_by_subtopic.get(subtopic_id)
            if prior is not None:
                est = store.update_teaching_time_estimate(
                    prior.id, estimated_minutes=minutes, estimated_periods=periods,
                    method=COMPUTED_ESTIMATE_METHOD, approved_by=approved_by)
            else:
                est = store.create_teaching_time_estimate(
                    subtopic_id=subtopic_id, academic_year_id=academic_year_id,
                    estimated_minutes=minutes, estimated_periods=periods,
                    method=COMPUTED_ESTIMATE_METHOD, approved_by=approved_by)
            estimates.append(est)

    allocated = kept_periods + sum(e.estimated_periods or 0 for e in estimates)
    return TeachingTimeComputationResult(
        academic_year_id=academic_year_id, book_id=book_id, periods_per_week=periods_per_week,
        period_minutes=period_cfg.period_minutes, calendar_weeks=calendar_weeks,
        total_subject_periods=budget,
        total_instructional_minutes=budget * period_cfg.period_minutes,
        units_skipped_no_subtopics=tuple(skipped), estimates=tuple(estimates),
        periods_allocated=allocated, periods_short=max(0, allocated - budget),
    )
