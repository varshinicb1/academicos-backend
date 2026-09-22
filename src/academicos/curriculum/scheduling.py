"""Micro scheduling -- §11-14, §38-41: the module the transcript calls the
actual point of the project. Turns curriculum (real, approved Subtopics, in
their real delivery order) x teaching-time-estimate (real, calendar-grounded
periods per subtopic) x calendar (real working days) into a real, dated
`ScheduledLesson[]` -- never a week-numbered or subtopic-less approximation.

Deliberately consumes calendar.py's working_days_for_year() and
store.py's TeachingTimeEstimate rows rather than recomputing either --
scheduling is the last stage of a pipeline (calendar -> teaching-time
-> schedule), not a parallel path that could drift from the other two.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..assessment.audit_log import AuditLog
from .calendar import working_days_for_year
from .store import CurriculumStore


def _subject_teaching_days(working_days: list[date], periods_per_week: int,
                           weekday_slots: Optional[set[int]] = None) -> list[date]:
    """When `weekday_slots` is given (a school's real
    SubjectTimetableSlot rows for this subject/year -- "Science meets
    Monday, Wednesday, Friday"), returns every real working day landing on
    one of those weekdays: the school's actual timetable, not a guess.

    Without it (no real timetable entered yet for this subject), falls back
    to the previous deterministic simplification: groups the real working
    days into real calendar weeks (ISO Mon-Sun) and takes the first
    `periods_per_week` of each week, in date order. Still honestly
    documented as a fallback, not a claim to reproduce a real school's
    actual timetable -- the same "no finer real signal exists, don't
    fabricate one" stance seed_cbse10.py and calendar.py already take."""
    if weekday_slots:
        return [d for d in working_days if d.weekday() in weekday_slots]
    by_week: dict[tuple[int, int], list[date]] = defaultdict(list)
    for d in working_days:
        iso_year, iso_week, _ = d.isocalendar()
        by_week[(iso_year, iso_week)].append(d)
    selected: list[date] = []
    for key in sorted(by_week):
        days = sorted(by_week[key])
        selected.extend(days[:periods_per_week])
    return selected


def _timetable_slots_for_book(store: CurriculumStore, academic_year_id: str, book_id: str) -> list:
    subject = store.subject_name_for_book(book_id)
    if subject is None:
        return []
    return store.timetable_slots_for_subject(academic_year_id, subject)


def _teaching_days_for_book(store: CurriculumStore, academic_year_id: str, book_id: str,
                            periods_per_week: int) -> list[date]:
    """The one rule for "which real days does this book get taught on",
    shared by schedule_book() and push_lessons_after(). PUSH used to call
    _subject_teaching_days() without the subject's timetable slots, so a
    push fell back to "first N working days of the week" even when the
    school had entered a real timetable -- the 2026-09-21 audit watched it
    push Mon-Fri Mathematics lessons onto Saturdays."""
    wd = working_days_for_year(store, academic_year_id)
    slots = _timetable_slots_for_book(store, academic_year_id, book_id)
    weekday_slots = {s.day_of_week for s in slots} or None
    return _subject_teaching_days(
        [date.fromisoformat(s) for s in wd.dates], periods_per_week, weekday_slots)


@dataclass(frozen=True)
class ScheduleResult:
    academic_year_id: str
    book_id: str
    periods_per_week: int
    teaching_days_available: int
    lessons_created: int
    subtopics_scheduled: int
    subtopics_partially_scheduled: tuple[str, ...]   # got some but not all periods before days ran out
    subtopics_unscheduled: tuple[str, ...]            # got zero periods -- days ran out entirely
    subtopics_without_estimate: tuple[str, ...]       # skipped -- no real TeachingTimeEstimate yet
    first_scheduled_date: Optional[str]
    last_scheduled_date: Optional[str]


def schedule_book(
    store: CurriculumStore, *, school_id: str, academic_year_id: str, book_id: str,
    periods_per_week: int, force: bool = False,
) -> ScheduleResult:
    """Schedules every real, approved Subtopic in a book's real delivery
    order (Unit.seq -> Chapter.seq -> Topic.seq -> Subtopic.seq -- §13's
    independently-reorderable sequence, already respected by store.py's own
    ORDER BY) across the real teaching days this subject gets this year.

    A Subtopic without a real TeachingTimeEstimate for this academic year
    is skipped and reported, never given a guessed period count -- run
    calendar.py's compute_teaching_time_estimates() first.

    Not idempotent by default: re-scheduling an already-scheduled book
    raises unless `force=True`, which deletes the existing schedule first.
    A real schedule may already be visible to teachers/students (the next
    two milestones); silently regenerating it out from under them would be
    wrong. Deliberate rescheduling of individual lessons (§14, PUSH/ADJUST
    with an audit trail) is a separate, smaller, later operation -- this
    function only ever produces the initial full-book schedule.
    """
    if periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")

    existing = store.scheduled_lessons_for_book(academic_year_id, book_id)
    if existing:
        if not force:
            raise ValueError(
                f"book {book_id} already has a schedule for {academic_year_id} "
                f"({len(existing)} lessons) -- pass force=True to regenerate "
                "(this deletes the existing schedule first)")
        store.delete_scheduled_lessons_for_book(academic_year_id, book_id)

    teaching_days = _teaching_days_for_book(store, academic_year_id, book_id, periods_per_week)

    ordered: list[tuple[str, int]] = []   # (subtopic_id, periods_needed)
    without_estimate: list[str] = []
    for unit in store.units_for_book(book_id):
        for chapter in store.chapters_for_unit(unit.id):
            for topic in store.topics_for_chapter(chapter.id):
                for subtopic in store.subtopics_for_topic(topic.id):
                    est = store.teaching_time_estimate_for_subtopic(subtopic.id, academic_year_id)
                    if est is None or not est.estimated_periods:
                        without_estimate.append(subtopic.id)
                        continue
                    ordered.append((subtopic.id, max(1, est.estimated_periods)))

    lessons_created = 0
    fully_scheduled = 0
    partially_scheduled: list[str] = []
    unscheduled: list[str] = []
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    day_idx = 0

    for subtopic_id, periods_needed in ordered:
        placed = 0
        for _ in range(periods_needed):
            if day_idx >= len(teaching_days):
                break
            d = teaching_days[day_idx]
            store.create_scheduled_lesson(school_id=school_id, academic_year_id=academic_year_id,
                                          book_id=book_id, subtopic_id=subtopic_id, date=d.isoformat())
            lessons_created += 1
            first_date = first_date or d.isoformat()
            last_date = d.isoformat()
            day_idx += 1
            placed += 1
        if placed == 0:
            unscheduled.append(subtopic_id)
        elif placed < periods_needed:
            partially_scheduled.append(subtopic_id)
        else:
            fully_scheduled += 1

    return ScheduleResult(
        academic_year_id=academic_year_id, book_id=book_id, periods_per_week=periods_per_week,
        teaching_days_available=len(teaching_days), lessons_created=lessons_created,
        subtopics_scheduled=fully_scheduled,
        subtopics_partially_scheduled=tuple(partially_scheduled),
        subtopics_unscheduled=tuple(unscheduled),
        subtopics_without_estimate=tuple(without_estimate),
        first_scheduled_date=first_date, last_scheduled_date=last_date,
    )


# --------------------------------------------------------------------- #
# Rescheduling on disruption -- §14: PUSH / ADJUST, with a real audit
# trail (old date, new date, reason, changed-by, timestamp). Reuses the
# existing, real, general-purpose assessment audit log
# (assessment/audit_log.py) rather than a parallel mechanism -- exactly
# the reuse docs/TRANSCRIPT_REQUIREMENTS_MATRIX.md's §14 row called for.
# --------------------------------------------------------------------- #

RESCHEDULE_ACTION = "curriculum_lesson_rescheduled"


@dataclass(frozen=True)
class RescheduleResult:
    lesson_id: str
    subtopic_id: str
    old_date: str
    new_date: str
    reason: str
    mode: str   # "adjust" | "push"


def _log_reschedule(audit_log: AuditLog, *, lesson, new_date: str, reason: str, mode: str,
                    changed_by: str) -> RescheduleResult:
    old_date = lesson.date
    audit_log.append(
        RESCHEDULE_ACTION, actor=changed_by,
        details={
            "lesson_id": lesson.id, "book_id": lesson.book_id,
            "academic_year_id": lesson.academic_year_id, "subtopic_id": lesson.subtopic_id,
            "mode": mode, "old_date": old_date, "new_date": new_date, "reason": reason,
        },
    )
    return RescheduleResult(lesson_id=lesson.id, subtopic_id=lesson.subtopic_id, old_date=old_date,
                            new_date=new_date, reason=reason, mode=mode)


class RescheduleConflict(ValueError):
    """A reschedule refused because of existing state, not a malformed
    request. A ValueError subclass so existing callers that catch ValueError
    still refuse the move; routes.py maps it to 409 rather than 422."""


class RescheduleClash(RescheduleConflict):
    """The target date already holds as many of this book's lessons as the
    subject has periods that day."""


class LessonAlreadyRecorded(RescheduleConflict):
    """The lesson is 'completed' or 'skipped': its date is the historical
    record of when it was taught (or deliberately not), so no reschedule
    may change it."""


def _periods_on(slots: list, day: date) -> int:
    """How many lessons of this subject one real day can hold. With a
    timetable, that is the number of SubjectTimetableSlot periods on that
    weekday (two Monday periods -> two Monday lessons); a working day the
    timetable doesn't list, or no timetable at all, holds one -- the same
    one-lesson-per-teaching-day rule schedule_book() places lessons with."""
    return max(1, sum(1 for s in slots if s.day_of_week == day.weekday()))


def adjust_lesson(
    store: CurriculumStore, audit_log: AuditLog, *, lesson_id: str, new_date: str, reason: str,
    changed_by: str,
) -> RescheduleResult:
    """ADJUST: move exactly one lesson to a specific new real working day
    -- e.g. swapping two lessons' order, or fixing a one-off conflict.
    Never touches any other lesson. `new_date` must be a real working day
    for this lesson's academic year (never lets a lesson land on a real
    holiday/weekly-off/alternate-Saturday), and must have a free period
    for this book: until 2026-09-21 there was no clash check, and the audit
    moved a lesson onto 2026-09-24, which already held one, double-booking
    the day. Raises RescheduleClash naming the lessons already there.

    Only a still-to-teach ('scheduled') lesson moves. Until 2026-09-22 the
    PUSH fix left this path open: a review moved a completed lesson to a
    free working day and got the new date back with status still
    'completed'. Raises LessonAlreadyRecorded instead."""
    lesson = store.get_scheduled_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"no such scheduled lesson: {lesson_id}")
    if lesson.status != "scheduled":
        raise LessonAlreadyRecorded(
            f"lesson {lesson_id} is already marked {lesson.status!r} on {lesson.date}; "
            f"that date is the teaching record and cannot be rescheduled")
    wd = working_days_for_year(store, lesson.academic_year_id)
    if new_date not in wd.dates:
        raise ValueError(f"{new_date} is not a real working day for this academic year")
    occupants = [l for l in store.scheduled_lessons_for_book(lesson.academic_year_id, lesson.book_id)
                 if l.date == new_date and l.id != lesson_id]
    slots = _timetable_slots_for_book(store, lesson.academic_year_id, lesson.book_id)
    capacity = _periods_on(slots, date.fromisoformat(new_date))
    if len(occupants) >= capacity:
        raise RescheduleClash(
            f"{new_date} already has {len(occupants)} lesson(s) for this book "
            f"({', '.join(l.id for l in occupants)}) and this subject has {capacity} "
            f"period(s) that day -- move or push that lesson first")
    store.reschedule_lesson_date(lesson_id, new_date=new_date)
    return _log_reschedule(audit_log, lesson=lesson, new_date=new_date, reason=reason,
                           mode="adjust", changed_by=changed_by)


@dataclass(frozen=True)
class PushResult:
    academic_year_id: str
    book_id: str
    from_date: str
    periods_per_week: int   # lessons placed per week: distinct timetable weekdays when a timetable decided
    lessons_pushed: int
    lessons_dropped: tuple[str, ...]   # ran out of real teaching days before the year ends
    reschedules: tuple[RescheduleResult, ...]


def push_lessons_after(
    store: CurriculumStore, audit_log: AuditLog, *, academic_year_id: str, book_id: str,
    from_date: str, reason: str, changed_by: str, periods_per_week: Optional[int] = None,
) -> PushResult:
    """PUSH: real disruption handling -- "today just became a holiday" (or
    any other reason a slot at/after from_date is lost). Every lesson
    still to be taught (status 'scheduled') dated on or after from_date
    shifts one real teaching slot later, in order, using the same real
    calendar + timetable + weekday-selection rule schedule_book() used.

    A completed or skipped lesson never moves: its date is the historical
    record of when it was taught (or deliberately not). Until 2026-09-21
    PUSH moved them too -- the audit watched a lesson marked completed on
    2026-09-21 get re-dated 2026-09-22, still 'completed'. The days those
    lessons hold stay theirs; pushed lessons skip over them.

    With SubjectTimetableSlot rows the timetable alone decides the days,
    so no cadence is needed and `periods_per_week` in the result is the
    lessons the timetable places per week: one per distinct timetable
    weekday, because _subject_teaching_days returns each date once. (Until
    2026-09-22 it reported the slot count, so a timetable with two Monday
    periods said 6 while the pushed schedule held 5 lessons a week.) A
    supplied `periods_per_week` that disagrees with that is refused rather
    than dropped: until 2026-09-22 the push returned 200 and the Reschedule
    tab said "Schedule pushed" while the admin's number was never used.
    This departs from Task 102's "supplying it still overrides" for the
    timetable case only -- overriding would place lessons on weekdays the
    timetable does not have, the Saturday bug that task fixed.

    Without a timetable, `periods_per_week` defaults to the subject's
    persisted SubjectPeriodAllocation; passing it overrides. With neither,
    the push is refused rather than guessing a cadence that would select a
    different set of weekdays than the existing schedule used.

    Lessons that run past the real academic year's last teaching day are
    honestly reported in `lessons_dropped`, never silently discarded or
    placed past the calendar's real end date."""
    if periods_per_week is not None and periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")
    slots = _timetable_slots_for_book(store, academic_year_id, book_id)
    if slots:
        timetable_cadence = len({s.day_of_week for s in slots})
        if periods_per_week is not None and periods_per_week != timetable_cadence:
            raise ValueError(
                f"periodsPerWeek={periods_per_week} conflicts with this subject's timetable, which "
                f"places {timetable_cadence} lesson(s) a week (one per timetable weekday) -- the "
                f"timetable decides the days, so leave periodsPerWeek blank or change the timetable")
        periods_per_week = timetable_cadence
    elif periods_per_week is None:
        subject = store.subject_name_for_book(book_id)
        alloc = store.subject_period_allocation(academic_year_id, subject) if subject else None
        if alloc is None:
            raise ValueError(
                f"no periods_per_week supplied and no stored subject period allocation for "
                f"{subject or book_id!r} in this academic year -- set one, or pass periodsPerWeek")
        periods_per_week = alloc.periods_per_week
        if periods_per_week <= 0:
            raise ValueError("periods_per_week must be positive")

    on_or_after = [l for l in store.scheduled_lessons_for_book(academic_year_id, book_id)
                   if l.date >= from_date]
    affected = sorted((l for l in on_or_after if l.status == "scheduled"), key=lambda l: l.date)
    kept_dates = {l.date for l in on_or_after if l.status != "scheduled"}
    if not affected:
        return PushResult(academic_year_id=academic_year_id, book_id=book_id, from_date=from_date,
                          periods_per_week=periods_per_week, lessons_pushed=0, lessons_dropped=(),
                          reschedules=())

    all_days = _teaching_days_for_book(store, academic_year_id, book_id, periods_per_week)
    all_iso = [d.isoformat() for d in all_days]

    try:
        disruption_idx = next(i for i, d in enumerate(all_iso) if d >= from_date)
    except StopIteration:
        raise ValueError(
            f"{from_date} is at or after the end of this academic year's real teaching days")

    reschedules: list[RescheduleResult] = []
    dropped: list[str] = []
    slot = disruption_idx + 1   # skip one real slot for the disruption itself
    for lesson in affected:
        while slot < len(all_iso) and all_iso[slot] in kept_dates:
            slot += 1           # a taught/skipped lesson already holds this day
        if slot >= len(all_iso):
            dropped.append(lesson.id)
            continue
        new_date = all_iso[slot]
        store.reschedule_lesson_date(lesson.id, new_date=new_date)
        reschedules.append(_log_reschedule(audit_log, lesson=lesson, new_date=new_date, reason=reason,
                                           mode="push", changed_by=changed_by))
        slot += 1

    return PushResult(academic_year_id=academic_year_id, book_id=book_id, from_date=from_date,
                      periods_per_week=periods_per_week, lessons_pushed=len(reschedules),
                      lessons_dropped=tuple(dropped), reschedules=tuple(reschedules))


def reschedule_history_for_lesson(audit_log: AuditLog, lesson_id: str) -> list[dict]:
    """Every real PUSH/ADJUST ever applied to this lesson, newest first --
    reads through the same audit log the writes above went through, rather
    than a parallel history table. `for_action` is a full-table scan of
    this one action type (audit_log.py has no generic entity-scoped query
    today) -- fine at this scale, a real limitation to revisit if the
    audit log ever grows large enough to matter."""
    entries = audit_log.for_action(RESCHEDULE_ACTION)
    return [e for e in entries if e.get("details", {}).get("lesson_id") == lesson_id]
