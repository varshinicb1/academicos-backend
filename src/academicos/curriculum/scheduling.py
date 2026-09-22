"""Micro scheduling -- §11-14, §38-41: the module the transcript calls the
actual point of the project. Turns curriculum (real, approved Subtopics, in
their real delivery order) x teaching-time-estimate (real, calendar-grounded
periods per subtopic) x calendar (real working days) into a real, dated
`ScheduledLesson[]` -- never a week-numbered or subtopic-less approximation.

Deliberately consumes calendar.py's teaching_slots_for_book() and
store.py's TeachingTimeEstimate rows rather than recomputing either --
scheduling is the last stage of a pipeline (calendar -> teaching-time
-> schedule), not a parallel path that could drift from the other two.
The teaching-time budget is sized from the very same slot list, so a
syllabus that fits the budget fits the schedule.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..assessment.audit_log import AuditLog
from . import calendar as calendar_mod
from .calendar import working_days_for_year
from .models import RECORDED_STATUSES
from .store import CurriculumStore


def _resolve_cadence(store: CurriculumStore, academic_year_id: str, book_id: str,
                     periods_per_week: Optional[int]) -> int:
    """The periods a week this book's lessons are laid out at, for PUSH --
    the cadence the existing schedule was made with.

    With SubjectTimetableSlot rows the timetable decides: one lesson per
    timetable period, so a Monday double counts twice (until 2026-09-22 it
    counted once, and a 6-period timetable pushed at 5). Without one, the
    cadence schedule_book() recorded for this book; then (for a schedule
    made before that was recorded) the supplied value, then the subject's
    SubjectPeriodAllocation. A supplied value that contradicts the
    timetable or the recorded cadence is refused, not dropped: until
    2026-09-22 a book scheduled at 6 a week was pushed at the allocation's 3
    with a 200 "Schedule pushed"."""
    if periods_per_week is not None and periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")
    per_weekday = calendar_mod.timetable_periods_by_weekday(store, academic_year_id, book_id)
    if per_weekday:
        timetable_cadence = sum(per_weekday.values())
        if periods_per_week is not None and periods_per_week != timetable_cadence:
            raise ValueError(
                f"periodsPerWeek={periods_per_week} conflicts with this subject's timetable, which "
                f"places {timetable_cadence} lesson(s) a week (one per timetable period) -- the "
                f"timetable decides the days, so leave periodsPerWeek blank or change the timetable")
        return timetable_cadence
    recorded = store.book_schedule_cadence(academic_year_id, book_id)
    if recorded is not None:
        if periods_per_week is not None and periods_per_week != recorded:
            raise ValueError(
                f"periodsPerWeek={periods_per_week} conflicts with this book's schedule, which was "
                f"scheduled at {recorded} period(s) a week -- moving lessons at another cadence would "
                f"put them on days the schedule does not use; leave periodsPerWeek blank, or "
                f"regenerate the schedule at the new cadence")
        return recorded
    if periods_per_week is not None:
        return periods_per_week
    subject = store.subject_name_for_book(book_id)
    alloc = store.subject_period_allocation(academic_year_id, subject) if subject else None
    if alloc is None:
        raise ValueError(
            f"no periods_per_week supplied and no stored subject period allocation for "
            f"{subject or book_id!r} in this academic year -- set one, or pass periodsPerWeek")
    if alloc.periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")
    return alloc.periods_per_week


@dataclass(frozen=True)
class ScheduleResult:
    academic_year_id: str
    book_id: str
    periods_per_week: int             # lessons placed per week: timetable periods when a timetable decided
    teaching_days_available: int      # distinct days the new lessons could use
    lessons_created: int
    subtopics_scheduled: int
    subtopics_partially_scheduled: tuple[str, ...]   # got some but not all periods before days ran out
    subtopics_unscheduled: tuple[str, ...]            # got zero periods -- days ran out entirely
    subtopics_without_estimate: tuple[str, ...]       # skipped -- no real TeachingTimeEstimate yet
    first_scheduled_date: Optional[str]
    last_scheduled_date: Optional[str]
    teaching_periods_available: int = 0   # periods the new lessons could use (a double counts twice)
    lessons_kept: int = 0                 # completed/skipped lessons a force regenerate kept
    past_lessons_kept: int = 0            # unmarked lessons dated before from_date a regenerate kept
    warning: Optional[str] = None         # set whenever a subtopic has no (or too few) dated lessons

    @property
    def all_subtopics_scheduled(self) -> bool:
        return self.warning is None


def _shortfall_warning(*, total: int, unscheduled: int, partial: int, without_estimate: int,
                       periods_available: int, periods_needed: int,
                       from_date: Optional[str]) -> Optional[str]:
    """One sentence a principal cannot miss. The audit found 49 of 404 real
    subtopics with no date behind a plain 200 and a list of ids. The
    periods-left figure counts from the day the plan starts, so a mid-year
    regenerate reports what is really left."""
    parts = []
    if unscheduled:
        parts.append(f"{unscheduled} of {total} subtopics have no date")
    if partial:
        parts.append(f"{partial} of {total} subtopics got fewer periods than their estimate")
    if unscheduled or partial:
        since = f" from {from_date}" if from_date else ""
        parts.append(f"this subject has {periods_available} teaching periods left in the year{since} "
                     f"and the estimates need {periods_needed}")
    if without_estimate:
        parts.append(f"{without_estimate} of {total} subtopics have no teaching-time estimate yet "
                     f"(compute teaching-time estimates, then regenerate)")
    return "; ".join(parts) if parts else None


def schedule_book(
    store: CurriculumStore, *, school_id: str, academic_year_id: str, book_id: str,
    periods_per_week: int, force: bool = False, from_date: Optional[str] = None,
) -> ScheduleResult:
    """Schedules every real, approved Subtopic in a book's real delivery
    order (Unit.seq -> Chapter.seq -> Topic.seq -> Subtopic.seq -- §13's
    independently-reorderable sequence, already respected by store.py's own
    ORDER BY) across the real teaching periods this subject gets this year
    (calendar.teaching_slots_for_book: a Monday double period is two
    lessons that Monday).

    A Subtopic without a real TeachingTimeEstimate for this academic year
    is skipped and reported, never given a guessed period count -- run
    calendar.py's compute_teaching_time_estimates() first.

    Anything short of every subtopic fully dated sets `warning` (and so
    all_subtopics_scheduled=False): the unscheduled/partial lists alone went
    unnoticed behind a 200 when Social Science left 35 of 193 undated.

    Not idempotent by default: re-scheduling an already-scheduled book
    raises unless `force=True`. A regenerate replans only what is still to
    be taught: completed and skipped lessons are the teaching record and
    are kept exactly as they are, each counts toward its subtopic's
    estimate and holds its own period, and the replanned lessons fill every
    other period of the year, in delivery order. Until 2026-09-22 force
    deleted every lesson, completion marks included.

    `from_date` (ISO date; the route passes the school's today) is the first
    day a new lesson may use: a day that has passed can no longer be taught
    on. Until the review of 2026-09-22 a regenerate filled every free period
    from the year's first day, so a November regenerate (10 lessons marked
    in June) dated 100 of 210 replanned lessons before November and still
    said every subtopic was scheduled. Periods before `from_date` are simply
    not offered; kept lessons on or after it still hold their own periods,
    and the shortfall warning counts only the periods left. None means the
    year's first day (the pure planning case the tests build on).

    A 'scheduled' lesson dated before `from_date` is kept like a marked one
    (reported as `past_lessons_kept`): its day has passed, so it was taught
    but not ticked, or is overdue. Until the review of 9dd889d a regenerate
    on 2026-09-22 with nothing marked deleted the whole June-September plan,
    so the overdue record left delayed-topics and pace, and 9 of 21
    subtopics lost their dates because the first half was replanned into
    time the second half needed. Counting past lessons toward their
    subtopic's estimate means only the future is replanned.

    Records the cadence the lessons were laid out at, so PUSH moves them at
    the same cadence (Task 102 review)."""
    if periods_per_week <= 0:
        raise ValueError("periods_per_week must be positive")

    existing = store.scheduled_lessons_for_book(academic_year_id, book_id)
    recorded = [l for l in existing if l.status in RECORDED_STATUSES]
    past = [l for l in existing
            if l.status == "scheduled" and from_date is not None and l.date < from_date]
    kept = recorded + past
    if existing:
        if not force:
            raise ValueError(
                f"book {book_id} already has a schedule for {academic_year_id} "
                f"({len(existing)} lessons) -- pass force=True to regenerate (completed and skipped "
                "lessons are kept; only what is still to be taught is replanned)")
        store.delete_unrecorded_lessons_for_book(academic_year_id, book_id, from_date=from_date)

    # A kept lesson holds only its own period. Cutting the slot list after
    # the latest kept lesson (the first version of this fix) threw the whole
    # year before it away: a skip may be marked on a future lesson, and one
    # skip in March left the rest of the syllabus to fit after March.
    held = Counter(l.date for l in kept)
    slots = []
    for d in calendar_mod.teaching_slots_for_book(store, academic_year_id, book_id, periods_per_week):
        if from_date is not None and d.isoformat() < from_date:
            continue        # a day that has passed: nothing new can be taught on it
        if held[d.isoformat()] > 0:
            held[d.isoformat()] -= 1
            continue
        slots.append(d)
    # A skipped lesson keeps its period (it happened: assembly, exam duty) but
    # taught nothing, so it does not count toward its subtopic's estimate --
    # otherwise a subtopic whose lessons were all skipped comes back "fully
    # scheduled" with nothing left to teach (review of 2026-09-22).
    kept_per_subtopic = Counter(l.subtopic_id for l in kept if l.status != "skipped")

    ordered: list[tuple[str, int]] = []   # (subtopic_id, periods still needed)
    without_estimate: list[str] = []
    total_subtopics = 0
    already_covered = 0
    for unit in store.units_for_book(book_id):
        for chapter in store.chapters_for_unit(unit.id):
            for topic in store.topics_for_chapter(chapter.id):
                for subtopic in store.subtopics_for_topic(topic.id):
                    total_subtopics += 1
                    est = store.teaching_time_estimate_for_subtopic(subtopic.id, academic_year_id)
                    if est is None or not est.estimated_periods:
                        without_estimate.append(subtopic.id)
                        continue
                    needed = max(1, est.estimated_periods) - kept_per_subtopic[subtopic.id]
                    if needed <= 0:
                        already_covered += 1
                        continue
                    ordered.append((subtopic.id, needed))

    lessons_created = 0
    fully_scheduled = already_covered
    partially_scheduled: list[str] = []
    unscheduled: list[str] = []
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    slot_idx = 0

    for subtopic_id, periods_needed in ordered:
        placed = 0
        while placed < periods_needed and slot_idx < len(slots):
            d = slots[slot_idx].isoformat()
            store.create_scheduled_lesson(school_id=school_id, academic_year_id=academic_year_id,
                                          book_id=book_id, subtopic_id=subtopic_id, date=d)
            lessons_created += 1
            first_date = first_date or d
            last_date = d
            slot_idx += 1
            placed += 1
        if placed == 0:
            unscheduled.append(subtopic_id)
        elif placed < periods_needed:
            partially_scheduled.append(subtopic_id)
        else:
            fully_scheduled += 1

    per_weekday = calendar_mod.timetable_periods_by_weekday(store, academic_year_id, book_id)
    cadence = sum(per_weekday.values()) if per_weekday else periods_per_week
    store.set_book_schedule_cadence(academic_year_id, book_id, cadence)

    return ScheduleResult(
        academic_year_id=academic_year_id, book_id=book_id, periods_per_week=cadence,
        teaching_days_available=len(set(slots)), lessons_created=lessons_created,
        subtopics_scheduled=fully_scheduled,
        subtopics_partially_scheduled=tuple(partially_scheduled),
        subtopics_unscheduled=tuple(unscheduled),
        subtopics_without_estimate=tuple(without_estimate),
        first_scheduled_date=first_date, last_scheduled_date=last_date,
        teaching_periods_available=len(slots), lessons_kept=len(recorded),
        past_lessons_kept=len(past),
        warning=_shortfall_warning(
            total=total_subtopics, unscheduled=len(unscheduled), partial=len(partially_scheduled),
            without_estimate=len(without_estimate), periods_available=len(slots),
            periods_needed=sum(n for _, n in ordered), from_date=from_date),
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
    new_date: Optional[str]   # None only for mode "unscheduled"
    reason: str
    mode: str   # "adjust" | "push" | "unscheduled" (a PUSH could not fit it)


def _log_reschedule(audit_log: AuditLog, *, lesson, new_date: Optional[str], reason: str,
                    mode: str, changed_by: str) -> RescheduleResult:
    """`new_date` is None for mode 'unscheduled': the lesson lost its day
    and got no other."""
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


def _periods_on(per_weekday: dict[int, int], day: date) -> int:
    """How many lessons of this subject one real day can hold. With a
    timetable, that is the number of SubjectTimetableSlot periods on that
    weekday (two Monday periods -> two Monday lessons); a working day the
    timetable doesn't list, or no timetable at all, holds one."""
    return max(1, per_weekday.get(day.weekday(), 0))


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

    Only a still-to-teach lesson moves. Until 2026-09-22 the PUSH fix left
    this path open: a review moved a completed lesson to a free working day
    and got the new date back with status still 'completed'. Raises
    LessonAlreadyRecorded instead. An 'unscheduled' lesson (one a PUSH could
    not fit) is still to be taught: ADJUST is how it gets a day back, and it
    becomes 'scheduled' again."""
    lesson = store.get_scheduled_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"no such scheduled lesson: {lesson_id}")
    if lesson.status in RECORDED_STATUSES:
        raise LessonAlreadyRecorded(
            f"lesson {lesson_id} is already marked {lesson.status!r} on {lesson.date}; "
            f"that date is the teaching record and cannot be rescheduled")
    wd = working_days_for_year(store, lesson.academic_year_id)
    if new_date not in wd.dates:
        raise ValueError(f"{new_date} is not a real working day for this academic year")
    occupants = [l for l in store.scheduled_lessons_for_book(lesson.academic_year_id, lesson.book_id)
                 if l.date == new_date and l.id != lesson_id and l.status != "unscheduled"]
    per_weekday = calendar_mod.timetable_periods_by_weekday(store, lesson.academic_year_id,
                                                            lesson.book_id)
    capacity = _periods_on(per_weekday, date.fromisoformat(new_date))
    if len(occupants) >= capacity:
        raise RescheduleClash(
            f"{new_date} already has {len(occupants)} lesson(s) for this book "
            f"({', '.join(l.id for l in occupants)}) and this subject has {capacity} "
            f"period(s) that day -- move or push that lesson first")
    store.reschedule_lesson_date(lesson_id, new_date=new_date,
                                 status="scheduled" if lesson.status == "unscheduled" else None)
    return _log_reschedule(audit_log, lesson=lesson, new_date=new_date, reason=reason,
                           mode="adjust", changed_by=changed_by)


def _reflow(store: CurriculumStore, audit_log: AuditLog, *, academic_year_id: str, book_id: str,
            periods_per_week: int, from_date: str, skip_disruption_day: bool, reason: str,
            changed_by: str) -> tuple[list[RescheduleResult], list[str]]:
    """Lays every still-to-teach lesson dated on/after `from_date` onto the
    book's teaching periods from `from_date` on, in delivery order -- PUSH's
    placement rule.

    Completed/skipped lessons never move and keep their periods; an
    'unscheduled' lesson holds none. `skip_disruption_day` gives up every
    period of the first teaching day on/after `from_date` (PUSH: "today
    just became a holiday"; a double period there is lost too).

    A lesson with no period left before the year ends becomes
    'unscheduled' (returned as dropped). Until 2026-09-22 it stayed
    'scheduled' on its old date while the lesson before it was moved onto
    that same date -- a double-booked day. Losing its day is logged like a
    move (mode 'unscheduled', no new date), so the lesson's history says
    why it has none."""
    on_or_after = [l for l in store.scheduled_lessons_for_book(academic_year_id, book_id)
                   if l.date >= from_date]
    affected = [l for l in on_or_after if l.status == "scheduled"]   # date, then delivery order
    if not affected:
        return [], []
    held = Counter(l.date for l in on_or_after if l.status in RECORDED_STATUSES)

    slots = [d.isoformat() for d in
             calendar_mod.teaching_slots_for_book(store, academic_year_id, book_id, periods_per_week)]
    start = next((i for i, d in enumerate(slots) if d >= from_date), None)
    if start is None:
        raise ValueError(
            f"{from_date} is at or after the end of this academic year's real teaching days")
    if skip_disruption_day:
        lost_day = slots[start]
        while start < len(slots) and slots[start] == lost_day:
            start += 1

    free: list[str] = []
    for d in slots[start:]:
        if held[d] > 0:
            held[d] -= 1        # a taught/skipped lesson already holds this period
            continue
        free.append(d)

    moves: list[RescheduleResult] = []
    dropped: list[str] = []
    for i, lesson in enumerate(affected):
        if i >= len(free):
            store.mark_lesson_unscheduled(lesson.id)
            _log_reschedule(audit_log, lesson=lesson, new_date=None, reason=reason,
                            mode="unscheduled", changed_by=changed_by)
            dropped.append(lesson.id)
            continue
        if free[i] == lesson.date:
            continue
        store.reschedule_lesson_date(lesson.id, new_date=free[i])
        moves.append(_log_reschedule(audit_log, lesson=lesson, new_date=free[i], reason=reason,
                                     mode="push", changed_by=changed_by))
    return moves, dropped


@dataclass(frozen=True)
class PushResult:
    academic_year_id: str
    book_id: str
    from_date: str
    periods_per_week: int   # lessons placed per week: timetable periods when a timetable decided
    lessons_pushed: int
    lessons_dropped: tuple[str, ...]   # ran out of real teaching periods: now 'unscheduled'
    reschedules: tuple[RescheduleResult, ...]


def push_lessons_after(
    store: CurriculumStore, audit_log: AuditLog, *, academic_year_id: str, book_id: str,
    from_date: str, reason: str, changed_by: str, periods_per_week: Optional[int] = None,
) -> PushResult:
    """PUSH: real disruption handling -- "today just became a holiday" (or
    any other reason a day at/after from_date is lost). Every lesson still
    to be taught (status 'scheduled') dated on or after from_date moves
    past the lost day, in order, onto the book's own teaching periods (see
    _reflow).

    A completed or skipped lesson never moves: its date is the historical
    record of when it was taught (or deliberately not). Until 2026-09-21
    PUSH moved them too -- the audit watched a lesson marked completed on
    2026-09-21 get re-dated 2026-09-22, still 'completed'. The periods those
    lessons hold stay theirs; pushed lessons skip over them.

    The cadence is the one the schedule was made with (_resolve_cadence):
    the timetable when there is one, else the cadence schedule_book()
    recorded, else `periods_per_week`, else the subject allocation. A
    `periods_per_week` that contradicts the timetable or the recorded
    cadence is refused.

    Lessons that run past the real academic year's last teaching day are
    reported in `lessons_dropped` and marked 'unscheduled', never placed
    past the calendar's real end date or left holding a day."""
    cadence = _resolve_cadence(store, academic_year_id, book_id, periods_per_week)
    moves, dropped = _reflow(store, audit_log, academic_year_id=academic_year_id, book_id=book_id,
                             periods_per_week=cadence, from_date=from_date, skip_disruption_day=True,
                             reason=reason, changed_by=changed_by)
    return PushResult(academic_year_id=academic_year_id, book_id=book_id, from_date=from_date,
                      periods_per_week=cadence, lessons_pushed=len(moves),
                      lessons_dropped=tuple(dropped), reschedules=tuple(moves))


def reschedule_history_for_lesson(audit_log: AuditLog, lesson_id: str) -> list[dict]:
    """Every real PUSH/ADJUST ever applied to this lesson (and a PUSH that
    left it 'unscheduled'), newest first --
    reads through the same audit log the writes above went through, rather
    than a parallel history table. `for_action` is a full-table scan of
    this one action type (audit_log.py has no generic entity-scoped query
    today) -- fine at this scale, a real limitation to revisit if the
    audit log ever grows large enough to matter."""
    entries = audit_log.for_action(RESCHEDULE_ACTION)
    return [e for e in entries if e.get("details", {}).get("lesson_id") == lesson_id]
