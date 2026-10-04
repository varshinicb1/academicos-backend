"""Everything a new school needs to start, in one step (the "Set up your
school" wizard).

A school that has just signed up has no academic year, classes, calendar,
terms or bell schedule, and every screen that needs one of them answered with
the server's own refusal ("no academic year covers today; ..."). The pieces
could only be made one by one, in the right order, through five different
screens -- scripts/seed_test_school.py is that order written down. This makes
them in one call, with the defaults a CBSE school starts from:

  * the academic year: the Indian school year containing today, 1 April to
    31 March, labelled "2026-27" (the caller may give other dates);
  * the classes, 6 to 10 by default, each seeded with its CBSE subjects,
    books and chapters (seed_cbse10.seed_cbse_grade), and the sections asked
    for (section A by default);
  * the calendar: Sunday off, and the second and fourth Saturdays if asked;
  * two terms, April to September and October to March;
  * a bell schedule of 8 periods of 40 minutes from 08:00 with a break after
    the fourth.

It is idempotent. Each piece is looked up before it is made, so a second
call -- or a call after the principal set some of it up by hand -- adds only
what is missing and reports the rest as already there. Nothing that exists is
changed: a calendar, a term or a bell the school already has is kept as it is.

Classes 1-5 are not set up here: their syllabus comes from a book list this
release does not ship, so seeding them makes classes with no subjects
(project memory, 2026-10-01). 11 and 12 are out of this release.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from . import calendar as calendar_mod
from .models import DEFAULT_SECTION_NAME
from .school_model import WEEKDAY_NAMES
from .seed_cbse10 import seed_cbse_grade
from .store import CurriculumStore

# Classes 1-5 seed from their NCERT books' contents (ncert_books.json, N-3-2); until
# 2026-10-04 that file was on no branch and only 6-10 could be set up.
SETUP_GRADES = tuple(range(1, 11))
# With no classes given, the ones the question bank serves.
DEFAULT_GRADES = (6, 7, 8, 9, 10)
DEFAULT_SECTIONS = (DEFAULT_SECTION_NAME,)
MAX_SECTIONS_PER_CLASS = 20
BELL_NAME = "Main"

# Two clicks on "Set up" must not make two years: the look-ups and the writes
# of one school's setup run one at a time.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _school_lock(school_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(school_id, threading.Lock())


@dataclass
class ClassSpec:
    grade: int
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_SECTIONS))


@dataclass
class TermSpec:
    name: str
    start_date: str
    end_date: str


@dataclass
class BellSpec:
    periods: int = 8
    period_minutes: int = 40
    starts_at: str = "08:00"
    break_after_period: int = 4      # 0: no break
    break_minutes: int = 20


@dataclass
class ClassResult:
    grade: int
    grade_id: str
    sections: list[str]
    subjects: int
    chapters_added: int
    created: bool


@dataclass
class SetupResult:
    academic_year_id: str
    created: list[str] = field(default_factory=list)
    already_existed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    classes: list[ClassResult] = field(default_factory=list)
    calendar_id: Optional[str] = None
    term_ids: list[str] = field(default_factory=list)
    bell_schedule_id: Optional[str] = None


# ---------------------------------------------------------------- defaults

def school_year_for(today: date) -> tuple[str, str, str]:
    """The Indian school year containing `today`: (label, first day, last
    day), e.g. ("2026-27", "2026-04-01", "2027-03-31")."""
    first = today.year if today.month >= 4 else today.year - 1
    return year_label(first), f"{first}-04-01", f"{first + 1}-03-31"


def year_label(first_year: int) -> str:
    return f"{first_year}-{(first_year + 1) % 100:02d}"


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year, month = d.year + month // 12, month % 12 + 1
    for day in (d.day, 30, 29, 28):     # 31 August + 6 months is 28/29 February
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


def default_terms(start_date: str, end_date: str) -> list[TermSpec]:
    """Two halves: six months from the first day, then the rest -- for the
    usual year, 1 April to 30 September and 1 October to 31 March. A year of
    six months or less is one term."""
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    first_end = _add_months(start, 6) - timedelta(days=1)
    if first_end >= end:
        return [TermSpec("Term 1", start_date, end_date)]
    return [TermSpec("Term 1", start_date, first_end.isoformat()),
            TermSpec("Term 2", (first_end + timedelta(days=1)).isoformat(), end_date)]


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def bell_slots(spec: BellSpec) -> list[dict]:
    """The day's slots: `periods` teaching periods of `period_minutes` from
    `starts_at`, with one break after period `break_after_period`."""
    try:
        h, m = (int(x) for x in spec.starts_at.split(":"))
        if not (0 <= h < 24 and 0 <= m < 60) or len(spec.starts_at) != 5:
            raise ValueError
    except ValueError:
        raise ValueError(f"the day starts at a time like 08:00, not {spec.starts_at!r}") from None
    at = h * 60 + m
    slots: list[dict] = []
    for p in range(1, spec.periods + 1):
        slots.append({"start": _hhmm(at), "end": _hhmm(at + spec.period_minutes), "kind": "teaching"})
        at += spec.period_minutes
        if p == spec.break_after_period and p < spec.periods:
            slots.append({"start": _hhmm(at), "end": _hhmm(at + spec.break_minutes), "kind": "break"})
            at += spec.break_minutes
    if at > 23 * 60 + 59:
        raise ValueError("those periods would run past midnight; start earlier or have fewer or shorter periods")
    return slots


# ---------------------------------------------------------------- checks


def _iso(value: str, what: str) -> date:
    try:
        d = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"the {what} {value!r} is not a date like 2026-04-01") from None
    if d.isoformat() != value:
        raise ValueError(f"the {what} {value!r} is not a date like 2026-04-01")
    return d


def check_grades(classes: list[ClassSpec]) -> None:
    seen: set[int] = set()
    for c in classes:
        if c.grade not in SETUP_GRADES:
            raise ValueError(f"class {c.grade} cannot be set up here: choose classes "
                             f"{SETUP_GRADES[0]} to {SETUP_GRADES[-1]}")
        if c.grade in seen:
            raise ValueError(f"class {c.grade} is listed twice")
        seen.add(c.grade)


def _clean_sections(c: ClassSpec) -> list[str]:
    names: list[str] = []
    for raw in c.sections or list(DEFAULT_SECTIONS):
        name = CurriculumStore._clean_section_name(raw)
        if name.casefold() not in {n.casefold() for n in names}:
            names.append(name)
    if len(names) > MAX_SECTIONS_PER_CLASS:
        raise ValueError(f"class {c.grade} can have at most {MAX_SECTIONS_PER_CLASS} sections here")
    return names


# ---------------------------------------------------------------- the setup


def set_up_school(store: CurriculumStore, *, school_id: str, today: date,
                  year_label_: Optional[str] = None, start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  classes: Optional[list[ClassSpec]] = None,
                  weekly_off_days: Optional[list[str]] = None,
                  alternate_saturday_rule: str = "none",
                  terms: Optional[list[TermSpec]] = None,
                  bell: Optional[BellSpec] = None) -> SetupResult:
    """Makes what `school_id` is missing and reports what it already had.

    Everything the request says is checked before anything is written
    (ValueError, the route's 422), so a refused request leaves the school as
    it was. Pieces that exist are never changed."""
    classes = [ClassSpec(g) for g in DEFAULT_GRADES] if classes is None else classes
    if not classes:
        raise ValueError("choose at least one class")
    check_grades(classes)
    wanted_sections = {c.grade: _clean_sections(c) for c in classes}
    offs = calendar_mod.normalize_weekly_off_days(["sunday"] if weekly_off_days is None else weekly_off_days)
    if len(offs) >= 7:
        raise ValueError("a week needs at least one working day")
    saturday_rule = calendar_mod.normalize_alternate_saturday_rule(alternate_saturday_rule)
    bell_spec = bell or BellSpec()
    slots = bell_slots(bell_spec)
    teaching_days = [d for d, name in enumerate(WEEKDAY_NAMES) if name.lower() not in offs]

    # The year asked for, before looking at what exists.
    if start_date is None and end_date is None:
        label, start_date, end_date = school_year_for(today)
        label = year_label_ or label
    else:
        if start_date is None:
            raise ValueError("give the year's first day as well as its last")
        start = _iso(start_date, "first day")
        end = (_iso(end_date, "last day") if end_date is not None
               else _add_months(start, 12) - timedelta(days=1))
        end_date = end.isoformat()
        if end <= start:
            raise ValueError(f"the year ends ({end_date}) before it starts ({start_date})")
        if (end - start).days > 400:
            raise ValueError("a school year is at most about 13 months long")
        label = year_label_ or year_label(start.year)
    label = " ".join(label.split())
    if not label:
        raise ValueError("give the year a name, such as 2026-27")

    with _school_lock(school_id):
        result = SetupResult(academic_year_id="")

        # -- the academic year: the one with this name, else one that
        # overlaps these dates (two overlapping years would make "which term
        # is today" ambiguous), else a new one.
        years = store.academic_years_for_school(school_id)
        year = next((y for y in years if y.label == label), None) or next(
            (y for y in years if y.start_date <= end_date and start_date <= y.end_date), None)
        if year is None:
            year_is_new = True
            final_start, final_end = start_date, end_date
        else:
            year_is_new = False
            final_start, final_end = year.start_date, year.end_date

        # Terms are checked against the year they will be in, before any write.
        term_specs = default_terms(final_start, final_end) if terms is None else terms
        names: set[str] = set()
        for t in term_specs:
            if not t.name.strip():
                raise ValueError("every term needs a name")
            if t.name.strip().casefold() in names:
                raise ValueError(f"two terms are named {t.name.strip()!r}")
            names.add(t.name.strip().casefold())
            s, e = _iso(t.start_date, f"first day of {t.name}"), _iso(t.end_date, f"last day of {t.name}")
            if e < s:
                raise ValueError(f"{t.name} ends before it starts")
            if t.start_date < final_start or t.end_date > final_end:
                raise ValueError(f"{t.name} ({t.start_date} to {t.end_date}) is outside the year "
                                 f"({final_start} to {final_end})")

        if year_is_new:
            year = store.create_academic_year(school_id=school_id, label=label,
                                              start_date=start_date, end_date=end_date)
            result.created.append(f"Academic year {year.label} ({year.start_date} to {year.end_date})")
        else:
            note = "" if (year.start_date, year.end_date) == (start_date, end_date) else \
                f", kept as {year.start_date} to {year.end_date}"
            result.already_existed.append(f"Academic year {year.label}{note}")
        result.academic_year_id = year.id

        # -- classes, their CBSE subjects and chapters, and their sections.
        for c in sorted(classes, key=lambda c: c.grade):
            existed = store.get_grade_by_number(year.id, c.grade) is not None
            seeded = seed_cbse_grade(store, school_id=school_id, academic_year_label=year.label,
                                     start_date=year.start_date, end_date=year.end_date,
                                     grade_number=c.grade)
            wanted = wanted_sections[c.grade]
            have = store.sections_for_grade(seeded.grade_id)
            if (not existed and len(have) == 1 and have[0].name == DEFAULT_SECTION_NAME
                    and DEFAULT_SECTION_NAME.casefold() not in {w.casefold() for w in wanted}):
                # A new class comes with section A; the school asked for other
                # names, so its first section takes the first of them.
                store.update_section(have[0].id, name=wanted[0])
                have = store.sections_for_grade(seeded.grade_id)
            have_names = {s.name.casefold() for s in have}
            added = []
            for name in wanted:
                if name.casefold() not in have_names:
                    store.create_section(grade_id=seeded.grade_id, name=name)
                    added.append(name)
            sections = [s.name for s in store.sections_for_grade(seeded.grade_id)]
            subjects = len(store.subjects_for_grade(seeded.grade_id))
            result.classes.append(ClassResult(grade=c.grade, grade_id=seeded.grade_id, sections=sections,
                                              subjects=subjects, chapters_added=seeded.chapters_seeded,
                                              created=not existed))
            listed = ", ".join(f"{c.grade}-{s}" for s in sections)
            if not existed:
                result.created.append(f"Class {c.grade}: {subjects} subjects, "
                                      f"{seeded.chapters_seeded} chapters, sections {listed}")
            else:
                result.already_existed.append(f"Class {c.grade}")
                if added:
                    result.created.append(f"Class {c.grade}: sections "
                                          + ", ".join(f"{c.grade}-{s}" for s in added))
                if seeded.chapters_seeded:
                    result.created.append(f"Class {c.grade}: {seeded.chapters_seeded} missing chapters")
            if seeded.subjects_skipped:
                result.skipped.append(f"Class {c.grade}: no syllabus yet for "
                                      + ", ".join(seeded.subjects_skipped))

        # -- the calendar: weekly offs and the Saturday rule.
        cal = store.get_calendar_for_year(year.id)
        if cal is None:
            cal = store.create_calendar(academic_year_id=year.id, weekly_off_days=offs,
                                        alternate_saturday_rule=saturday_rule)
            result.created.append("Calendar: " + _calendar_words(offs, saturday_rule))
        else:
            result.already_existed.append("Calendar: " + _calendar_words(cal.weekly_off_days,
                                                                         cal.alternate_saturday_rule))
        result.calendar_id = cal.id

        # -- terms. The default two are only added to a year with no terms;
        # terms the caller names are each added unless one of that name exists.
        existing_terms = store.terms_for_year(year.id)
        if terms is None and existing_terms:
            result.already_existed.append("Terms: " + ", ".join(t.name for t in existing_terms))
        else:
            by_name = {t.name.casefold(): t for t in existing_terms}
            for t in term_specs:
                if t.name.strip().casefold() in by_name:
                    result.already_existed.append(f"Term {t.name.strip()}")
                    continue
                try:
                    made = store.create_term(academic_year_id=year.id, name=t.name,
                                             start_date=t.start_date, end_date=t.end_date)
                except ValueError as e:
                    result.skipped.append(f"Term {t.name.strip()}: {e}")
                    continue
                result.created.append(f"Term {made.name} ({made.start_date} to {made.end_date})")
        result.term_ids = [t.id for t in store.terms_for_year(year.id)]

        # -- the bell schedule.
        bells = store.bell_schedules_for_year(year.id)
        if bells:
            result.already_existed.append("Bell schedule " + ", ".join(b.name for b in bells))
            result.bell_schedule_id = next((b.id for b in bells if b.is_default), bells[0].id)
        else:
            b = store.create_bell_schedule(academic_year_id=year.id, name=BELL_NAME, slots=slots,
                                           days=teaching_days)
            result.created.append(f"Bell schedule: {b.teaching_periods} periods of "
                                  f"{bell_spec.period_minutes} minutes from {slots[0]['start']}")
            result.bell_schedule_id = b.id
        return result


def _calendar_words(offs: list[str], saturday_rule: str) -> str:
    days = ", ".join(d.capitalize() for d in offs) or "no day"
    rule = {"none": "", "all": "; every Saturday off",
            "second_fourth": "; second and fourth Saturdays off"}.get(
        saturday_rule, f"; Saturdays off: {saturday_rule}")
    return f"{days} off{rule}"
