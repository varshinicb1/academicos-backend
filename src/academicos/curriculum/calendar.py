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
from dataclasses import dataclass, field
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

# Days that change how a working day runs (v3 audit N-3-19: the calendar knew
# only weekly offs, alternate Saturdays and closures). Each has a field of its
# own (models.Holiday):
# - half_day: only periods 1..last_period are held; the periods after it are
#   not lessons, need no cover and hold no part of a plan.
# - exam_window: teaching stops for the chosen classes (`grades`; none is
#   every class) from date to end_date. The days stay working days -- the
#   school is open and the classes sit exams -- so the cover engine treats a
#   window as it treats a published exam paper (cover.ExamDay), with or
#   without a datesheet.
# - working_day: a weekly off or an alternate Saturday off made a working day
#   (a compensatory Saturday), running the timetable of `timetable_weekday`.
HALF_DAY, EXAM_WINDOW, WORKING_DAY = "half_day", "exam_window", "working_day"
DAY_KINDS = (HALF_DAY, EXAM_WINDOW, WORKING_DAY)
MAX_PERIOD = 20

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


def is_off_day(d: date, weekly_off_days: list[str], alternate_saturday_rule: str) -> bool:
    """A weekly off, or an alternate Saturday off, by the calendar's rules."""
    if _WEEKDAY_NAMES[d.weekday()] in normalize_weekly_off_days(weekly_off_days):
        return True
    return d.weekday() == 5 and _nth_weekday_of_month(d) in parse_alternate_saturday_rule(alternate_saturday_rule)


def validate_holiday(*, date_: str, end_date: Optional[str], kind: str,
                     year_start: str, year_end: str, last_period: Optional[int] = None,
                     grades: Optional[list[int]] = None, timetable_weekday: Optional[int] = None,
                     calendar=None, year_grades: Optional[set[int]] = None) -> None:
    """ValueError for anything a holiday row must never hold: an unknown
    kind (it would be listed but never counted), an unparseable date
    ('next monday' was accepted), an end before the start (a 500 from the
    store before 2026-09-22), or a day outside the academic year
    (2030-01-01 was accepted for a 2026-27 year).

    A day kind (DAY_KINDS) needs its own field and no other's: a half day
    its last period, an exam window classes of this year (`year_grades`)
    or none for every class, and a working day the weekday whose timetable
    it runs -- on a day `calendar`'s rules make an off day."""
    if kind not in HOLIDAY_KINDS + DAY_KINDS:
        raise ValueError(
            f"unknown holiday kind {kind!r} -- closures: {', '.join(CLOSURE_KINDS)}; "
            f"a marked day that still teaches: {', '.join(NON_CLOSURE_KINDS)}; "
            f"a day that runs differently: {', '.join(DAY_KINDS)}")
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
    if kind == HALF_DAY:
        if last_period is None or not 1 <= last_period <= MAX_PERIOD:
            raise ValueError(f"a half day keeps periods 1 to its last period (lastPeriod, 1 to {MAX_PERIOD}); "
                             "the periods after it are not held")
    elif last_period is not None:
        raise ValueError("only a half day has a last period")
    if kind == EXAM_WINDOW:
        unknown = sorted(set(grades or ()) - year_grades) if year_grades is not None else []
        if unknown:
            raise ValueError(f"no class {', '.join(map(str, unknown))} this year")
    elif grades:
        raise ValueError("only an exam window names the classes it stops")
    if kind == WORKING_DAY:
        if timetable_weekday is None or not 0 <= timetable_weekday <= 6:
            raise ValueError("a working day runs one weekday's timetable: give timetableWeekday (0 = Monday)")
        if end != start:
            raise ValueError("declare a compensatory working day one day at a time")
        if calendar is not None:
            if not is_off_day(start, calendar.weekly_off_days, calendar.alternate_saturday_rule):
                raise ValueError(f"{date_} is already a working day: a compensatory working day is a weekly "
                                 "off or an alternate Saturday off made a working day")
            name = _WEEKDAY_NAMES[timetable_weekday]
            if name in normalize_weekly_off_days(calendar.weekly_off_days):
                raise ValueError(f"{name.title()} is a weekly off and has no timetable: choose the weekday "
                                 "whose timetable the day runs")
    elif timetable_weekday is not None:
        raise ValueError("only a working day runs another weekday's timetable")


@dataclass(frozen=True)
class WorkingDaysResult:
    total_days: int
    working_days: int
    weekly_off_count: int
    alternate_saturday_off_count: int
    holiday_count: int
    dates: tuple[str, ...]   # ISO dates, real working days only
    compensatory_count: int = 0   # off days made working days (counted in working_days)


def compute_working_days(*, start_date: str, end_date: str, weekly_off_days: list[str],
                         alternate_saturday_rule: str, holiday_dates: set[str],
                         extra_working_dates: frozenset[str] = frozenset()) -> WorkingDaysResult:
    """Pure function, no store dependency -- the actual §10 arithmetic.
    Precedence per day: weekly-off > alternate-Saturday-off > holiday >
    working. A date can only be one of those, so double counting is
    impossible by construction.

    `extra_working_dates` are off days made working days (a compensatory
    Saturday, N-3-19): neither off rule takes them, a closure still does."""
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
    compensatory = 0

    d = start
    while d <= end:
        total_days += 1
        made_working = d.isoformat() in extra_working_dates
        if d.weekday() in off_weekdays and not made_working:
            weekly_off_count += 1
        elif d.weekday() == 5 and _nth_weekday_of_month(d) in saturdays_off and not made_working:
            alt_sat_count += 1
        elif d.isoformat() in holiday_dates:
            holiday_count += 1
        else:
            working.append(d.isoformat())
            compensatory += made_working
        d += timedelta(days=1)

    return WorkingDaysResult(
        total_days=total_days, working_days=len(working), weekly_off_count=weekly_off_count,
        alternate_saturday_off_count=alt_sat_count, holiday_count=holiday_count,
        dates=tuple(working), compensatory_count=compensatory,
    )


def _calendar_rows(store: CurriculumStore, academic_year_id: str):
    year = store.get_academic_year(academic_year_id)
    if year is None:
        raise ValueError(f"no such academic year: {academic_year_id}")
    cal = store.get_calendar_for_year(academic_year_id)
    if cal is None:
        raise ValueError(
            f"academic year {academic_year_id} has no calendar configured yet -- "
            "create one first (POST .../calendar)")
    return year, cal, store.holidays_for_calendar(cal.id)


def _days_of(h) -> list[str]:
    return [h.date] if h.end_date is None else _date_range(h.date, h.end_date)


def _working_days(year, cal, holidays) -> WorkingDaysResult:
    # "event" markers (e.g. Annual Day) don't remove a teaching day; every
    # closure kind does (CLOSURE_KINDS above). A holiday with end_date set (a real multi-day block --
    # a 30-45 day summer break) expands to every date in the inclusive range,
    # not just its start date. A compensatory working day adds its day.
    holiday_dates: set[str] = set()
    for h in holidays:
        if h.kind in CLOSURE_KINDS:
            holiday_dates.update(_days_of(h))
    return compute_working_days(
        start_date=year.start_date, end_date=year.end_date,
        weekly_off_days=cal.weekly_off_days, alternate_saturday_rule=cal.alternate_saturday_rule,
        holiday_dates=holiday_dates,
        extra_working_dates=frozenset(h.date for h in holidays if h.kind == WORKING_DAY),
    )


def working_days_for_year(store: CurriculumStore, academic_year_id: str) -> WorkingDaysResult:
    """Store-backed convenience: reads the real, persisted AcademicYear +
    Calendar + Holiday rows for a school and computes the real result."""
    return _working_days(*_calendar_rows(store, academic_year_id))


@dataclass
class SchoolDays:
    """How each working day of the year runs (v3 audit N-3-19): which
    weekday's timetable, up to which period, and for which classes teaching
    stops. The one reading of the calendar that the day view, cover, the
    plans and the calendar feed share, so they cannot disagree."""
    dates: tuple[str, ...] = ()
    runs_as: dict[str, int] = field(default_factory=dict)        # a compensatory day -> its timetable's weekday
    last_period: dict[str, int] = field(default_factory=dict)    # a half day -> the last period held
    # date -> ((the classes whose teaching stops, None for every class), the window's label), ...
    exam_windows: dict[str, tuple[tuple[Optional[frozenset[int]], str], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.working = frozenset(self.dates)

    def weekday(self, on: str) -> int:
        """The weekday whose timetable runs on `on`."""
        return self.runs_as.get(on, date.fromisoformat(on).weekday())

    def keeps(self, on: str, period: int) -> bool:
        """False for a period a half day cuts."""
        cut = self.last_period.get(on)
        return cut is None or period <= cut

    def exam_window(self, on: str, grade: Optional[int]) -> Optional[str]:
        """The label of the exam window stopping `grade`'s teaching on `on`."""
        for grades, label in self.exam_windows.get(on, ()):
            if grades is None or grade in grades:
                return label
        return None

    def holds(self, on: str, weekday: int, period: int, grade: Optional[int] = None) -> bool:
        """A timetabled period (weekday, period) of a class of `grade` is
        taught on `on`."""
        return (on in self.working and self.weekday(on) == weekday and self.keeps(on, period)
                and self.exam_window(on, grade) is None)


def school_days(store: CurriculumStore, academic_year_id: str) -> SchoolDays:
    """SchoolDays from the year's calendar. ValueError as working_days_for_year."""
    year, cal, holidays = _calendar_rows(store, academic_year_id)
    wd = _working_days(year, cal, holidays)
    working = set(wd.dates)
    out = SchoolDays(dates=wd.dates)
    windows: dict[str, list] = defaultdict(list)
    for h in holidays:
        if h.kind == WORKING_DAY and h.date in working and h.timetable_weekday is not None:
            out.runs_as[h.date] = h.timetable_weekday
        elif h.kind == HALF_DAY and h.last_period:
            for d in _days_of(h):
                out.last_period[d] = min(h.last_period, out.last_period.get(d, h.last_period))
        elif h.kind == EXAM_WINDOW:
            for d in _days_of(h):
                windows[d].append((frozenset(h.grades) if h.grades else None, h.label))
    out.exam_windows = {d: tuple(v) for d, v in windows.items()}
    return out


def grade_number_for_book(store: CurriculumStore, book_id: str) -> Optional[int]:
    grade_id = store.grade_id_for_book(book_id)
    grade = store.get_grade(grade_id) if grade_id else None
    return grade.number if grade else None


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
                                 book_id: str, section_id: Optional[str] = None) -> dict[int, int]:
    """{weekday: periods} from this book's subject's timetable; empty when
    the school has not entered one.

    With a section (SCH-4) it is that section's own week for the book's
    subject (TimetableEntry, keyed by subject id). Without one it is the
    school-wide per-subject-NAME slots, which every class of the year shares
    -- audit D115, and why a plan per section exists."""
    if section_id is not None:
        book = store.get_book(book_id)
        if book is None:
            return {}
        return dict(Counter(e.day_of_week for e in store.timetable_for_section(section_id)
                            if e.subject_id == book.subject_id))
    subject = store.subject_name_for_book(book_id)
    if subject is None:
        return {}
    return dict(Counter(s.day_of_week for s in store.timetable_slots_for_subject(academic_year_id, subject)))


def _subject_week(store: CurriculumStore, academic_year_id: str, book_id: str,
                  section_id: Optional[str]) -> list[tuple[int, int]]:
    """(weekday, period) of each period of this book's subject in the week:
    the section's own (TimetableEntry) or the school-wide per-name slots,
    as timetable_periods_by_weekday() counts them."""
    if section_id is not None:
        book = store.get_book(book_id)
        if book is None:
            return []
        return [(e.day_of_week, e.period) for e in store.timetable_for_section(section_id)
                if e.subject_id == book.subject_id]
    subject = store.subject_name_for_book(book_id)
    if subject is None:
        return []
    return [(s.day_of_week, s.period_number) for s in store.timetable_slots_for_subject(academic_year_id, subject)]


def teaching_slots_for_book(store: CurriculumStore, academic_year_id: str, book_id: str,
                            periods_per_week: int, section_id: Optional[str] = None) -> list[date]:
    """subject_teaching_slots() over the year's real working days and this
    book's subject's timetable (the section's own week when given).

    The calendar's days that run differently count as they run (N-3-19): a
    compensatory working day holds the periods of the weekday whose
    timetable it runs, a half day only the periods up to its last, and an
    exam window of the book's class none.

    A section's plan is laid only on periods that happen: the periods
    store.periods_held() says will not teach the subject (a lost period, or
    the class sitting an exam paper) come off, and the extra ones it names
    (make-up periods) are added."""
    days = school_days(store, academic_year_id)
    grade = grade_number_for_book(store, book_id)
    working = [s for s in days.dates if days.exam_window(s, grade) is None]
    week = _subject_week(store, academic_year_id, book_id, section_id)
    if week:
        slots = [date.fromisoformat(s) for s in working for w, p in week
                 if days.weekday(s) == w and days.keeps(s, p)]
    else:
        slots = subject_teaching_slots([date.fromisoformat(s) for s in working], periods_per_week)
    book = store.get_book(book_id) if section_id is not None else None
    if book is None:
        return slots
    away, extra = store.periods_held(academic_year_id, section_id, book.subject_id)
    if not away and not extra:
        return slots
    out = []
    for d in slots:
        if away[d.isoformat()] > 0:
            away[d.isoformat()] -= 1
            continue
        out.append(d)
    teaching = set(working)
    out += [date.fromisoformat(d) for d, n in extra.items() if d in teaching for _ in range(n)]
    return sorted(out)


def sections_with_a_week_for_book(store: CurriculumStore, book_id: str) -> list[str]:
    """The sections of the book's class whose own timetable (TimetableEntry)
    holds the book's subject -- the weeks a section plan of this book is
    laid on."""
    book = store.get_book(book_id)
    subject = store.get_subject(book.subject_id) if book else None
    if subject is None:
        return []
    return [s.id for s in store.sections_for_grade(subject.grade_id)
            if any(e.subject_id == book.subject_id for e in store.timetable_for_section(s.id))]


def teaching_budget_for_book(store: CurriculumStore, academic_year_id: str, book_id: str,
                             periods_per_week: int, section_id: Optional[str] = None
                             ) -> tuple[int, Optional[str]]:
    """(the periods a book's estimates are sized to, the section whose week
    sized them).

    With a section, that section's periods of the subject on the year's
    working days. Without one, the book's class's sections that have their
    own week for the subject decide: the largest of their budgets, the same
    "most periods any section has" rule Plan every section times a book at.
    Only a class with no section week for the subject falls back to the
    school-wide per-subject-name slots. Until v3 audit N-3-8 the estimates
    always took that fallback, which with no named slots is one period per
    working day: every book's budget was 235, the year's working days, for
    a subject its sections teach 8-9 periods a week."""
    if section_id is None:
        weeks = {sid: len(teaching_slots_for_book(store, academic_year_id, book_id, periods_per_week, sid))
                 for sid in sections_with_a_week_for_book(store, book_id)}
        if weeks:
            section_id = max(weeks, key=lambda sid: (weeks[sid], sid))
            return weeks[section_id], section_id
    return len(teaching_slots_for_book(store, academic_year_id, book_id, periods_per_week, section_id)), section_id


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
    section_id: Optional[str] = None    # the section whose own week sized the budget; None: school-wide

    @property
    def fits_in_year(self) -> bool:
        return self.periods_short == 0


def compute_teaching_time_estimates(
    store: CurriculumStore, *, academic_year_id: str, book_id: str, periods_per_week: int,
    approved_by: Optional[str] = None, recompute: bool = False, section_id: Optional[str] = None,
) -> TeachingTimeComputationResult:
    """Distributes a subject's real teaching periods for the year across its
    real Subtopics, weighted by each Unit's real CBSE marks (same signal a
    human HOD uses, and the one syllabus/timetable.py applies at Unit level).

    The budget is the number of real teaching periods the subject has this
    year -- teaching_budget_for_book(): the calendar's working days (after
    weekly offs, alternate Saturdays and every closure) x the subject's
    periods on those days in the section's own week (`section_id`, or the
    class's sections' weeks when none is named). It used to be
    periods_per_week x calendar weeks, which ignored holidays: 264 periods
    for a 6-period subject whose year really holds 220. `periods_per_week`
    still decides the no-timetable fallback; with a timetable the timetable
    decides, exactly as it does in scheduling.

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
    budget, sized_by = teaching_budget_for_book(store, academic_year_id, book_id, periods_per_week,
                                                section_id)

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
        periods_allocated=allocated, periods_short=max(0, allocated - budget), section_id=sized_by,
    )
