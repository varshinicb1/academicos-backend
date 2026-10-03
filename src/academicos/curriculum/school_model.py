"""The school as it actually runs (M1.2-M1.3, docs/plans/m1-school-data-model.md):
who teaches what to which section and how often (TeachingAllocation), the rooms,
the bell (BellSchedule and its slots), and the week each section keeps
(TimetableEntry: section x day x period -> subject, teacher, room).

This is the model the scheduling engine reads (REQUIREMENTS SCH-2) and the one
that retires audit D115: until now periods per week and weekday slots were
stored once per subject NAME per year (`subject_period_allocations`,
`subject_timetable_slots`), so every class of a year shared one Science
timetable. Here everything is keyed by section and subject id, and a subject
belongs to one grade, so 6-A's Science and 10-B's Science are different rows.

The tables live in the curriculum database (sections, subjects and years are
there, and the whole file is snapshotted together). CurriculumStore mixes this
class in; the methods use its serialized `_exec`/`_fetch*`/`_commit` helpers and
its connection lock, so every check-then-write is one lock hold.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

SCHOOL_MODEL_SCHEMA = """
-- M1.2: who teaches which subject to which section, and how many periods a
-- week. One row per (section, subject); the teacher may be unset while the
-- principal is still filling the grid.
CREATE TABLE IF NOT EXISTS teaching_allocations (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  section_id       TEXT NOT NULL,
  subject_id       TEXT NOT NULL,
  teacher_id       TEXT,
  periods_per_week INTEGER NOT NULL,
  created_at       TEXT NOT NULL,
  UNIQUE(section_id, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_alloc_year ON teaching_allocations(academic_year_id);
CREATE INDEX IF NOT EXISTS idx_alloc_teacher ON teaching_allocations(teacher_id);

-- M1.3: the school's rooms, labs and halls.
CREATE TABLE IF NOT EXISTS rooms (
  id         TEXT PRIMARY KEY,
  school_id  TEXT NOT NULL,
  name       TEXT NOT NULL COLLATE NOCASE,
  kind       TEXT NOT NULL,
  capacity   INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(school_id, name)
);

-- M1.3: the bell. A schedule is a day's slots in order; teaching slots are
-- numbered 1..N and that number is a TimetableEntry's `period`.
CREATE TABLE IF NOT EXISTS bell_schedules (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  name             TEXT NOT NULL,
  days_json        TEXT NOT NULL,
  slots_json       TEXT NOT NULL,
  is_default       INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bell_year ON bell_schedules(academic_year_id);

-- SCH-2: the week a section keeps. `period` is the teaching period number of
-- the section's bell schedule. One entry per section/day/period; a teacher
-- and a room are checked for double booking across sections in code.
CREATE TABLE IF NOT EXISTS timetable_entries (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  section_id       TEXT NOT NULL,
  day_of_week      INTEGER NOT NULL,
  period           INTEGER NOT NULL,
  subject_id       TEXT NOT NULL,
  teacher_id       TEXT,
  room_id          TEXT,
  created_at       TEXT NOT NULL,
  UNIQUE(section_id, day_of_week, period)
);
CREATE INDEX IF NOT EXISTS idx_tt_year ON timetable_entries(academic_year_id);
CREATE INDEX IF NOT EXISTS idx_tt_teacher ON timetable_entries(teacher_id, day_of_week, period);

-- SCH-3: periods a teacher cannot teach (a day of the week and a period
-- number of the year's default bell); the solver leaves them free.
CREATE TABLE IF NOT EXISTS teacher_unavailability (
  academic_year_id TEXT NOT NULL,
  teacher_id       TEXT NOT NULL,
  day_of_week      INTEGER NOT NULL,
  period           INTEGER NOT NULL,
  PRIMARY KEY (academic_year_id, teacher_id, day_of_week, period)
);

-- SCH-8: periods a room cannot be used (a lab closed for its weekly
-- maintenance slot, a hall booked for assembly practice); the solver books
-- nothing there and a hand-made week that does is refused.
CREATE TABLE IF NOT EXISTS room_unavailability (
  academic_year_id TEXT NOT NULL,
  room_id          TEXT NOT NULL,
  day_of_week      INTEGER NOT NULL,
  period           INTEGER NOT NULL,
  PRIMARY KEY (academic_year_id, room_id, day_of_week, period)
);

-- SCH-3 (audit N-3-6): a generated week the principal previewed, kept so
-- that publishing writes exactly that week instead of solving again (the
-- solver is not deterministic). `inputs_hash` fingerprints everything the
-- week was made from; a publish after any of it changed is refused.
CREATE TABLE IF NOT EXISTS timetable_previews (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  section_ids_json TEXT NOT NULL,
  entries_json     TEXT NOT NULL,
  summary_json     TEXT NOT NULL,
  inputs_hash      TEXT NOT NULL,
  created_by       TEXT,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tt_preview_year ON timetable_previews(academic_year_id, created_at);

-- SCH-4: the cadence each section's plan of a book was placed with (the
-- school-wide plan's stays in book_schedule_cadences).
CREATE TABLE IF NOT EXISTS section_plan_cadences (
  academic_year_id TEXT NOT NULL,
  book_id          TEXT NOT NULL,
  section_id       TEXT NOT NULL,
  periods_per_week INTEGER NOT NULL,
  PRIMARY KEY (academic_year_id, book_id, section_id)
);
"""

ROOM_KINDS = ("classroom", "lab", "hall", "other")
SLOT_KINDS = ("teaching", "break", "assembly", "zero")
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DEFAULT_DAYS = (0, 1, 2, 3, 4, 5)          # Monday to Saturday, the usual CBSE week
MAX_PERIODS_PER_WEEK = 60
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    from .store import new_id
    return new_id(prefix)


class TimetableClash(ValueError):
    """A timetable change that would double-book a teacher, a room or a
    section, or give a subject more periods than it is allocated. `clashes`
    holds one sentence per problem; the route's 409 lists them all."""

    def __init__(self, clashes: list[str]):
        super().__init__("; ".join(clashes))
        self.clashes = clashes


class InUse(Exception):
    """A delete refused because something still points at the row (the 409)."""


class StalePreview(Exception):
    """A previewed week whose inputs changed after it was made (the 409):
    publishing it would write a week nobody checked against today's data."""


# Previews kept per year: the newest few, so two people previewing at once
# do not lose each other's, and the database does not fill with weeks.
PREVIEWS_KEPT = 3


@dataclass
class TeachingAllocation:
    id: str
    school_id: str
    academic_year_id: str
    section_id: str
    subject_id: str
    teacher_id: Optional[str]
    periods_per_week: int
    created_at: str = ""
    # SCH-8 co-teaching: a second teacher in the room for every period of the
    # cell. The period needs a substitute only when both are away.
    co_teacher_id: Optional[str] = None
    # SCH-2 labs: the kind of room every period of the cell needs ("lab");
    # None: the section's own room, which the timetable does not book.
    room_kind: Optional[str] = None
    # SCH-2 double periods: how many of the week's periods come as two in a
    # row with no break between them (a practical).
    double_periods: int = 0


@dataclass
class Room:
    id: str
    school_id: str
    name: str
    kind: str
    capacity: Optional[int] = None
    created_at: str = ""


@dataclass
class BellSlot:
    start: str            # "HH:MM", the school's local time
    end: str
    kind: str             # teaching | break | assembly | zero
    period: Optional[int] = None   # 1..N for teaching slots, else None


@dataclass
class BellSchedule:
    id: str
    school_id: str
    academic_year_id: str
    name: str
    days: list[int]
    slots: list[BellSlot]
    is_default: bool = False
    created_at: str = ""

    @property
    def teaching_periods(self) -> int:
        return sum(1 for s in self.slots if s.kind == "teaching")


@dataclass
class TimetableEntry:
    id: str
    school_id: str
    academic_year_id: str
    section_id: str
    day_of_week: int
    period: int
    subject_id: str
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    created_at: str = ""
    locked: int = 0          # SCH-3: the solver keeps a locked period where it is
    co_teacher_id: Optional[str] = None   # SCH-8: the allocation's co-teacher, in the room too


def clean_slots(raw: list[dict[str, Any]]) -> list[BellSlot]:
    """Validates a day's slots (ValueError, the route's 422) and numbers the
    teaching ones. Slots must be in order and must not overlap."""
    if not raw:
        raise ValueError("a bell schedule needs at least one slot")
    if len(raw) > 20:
        raise ValueError("a bell schedule has at most 20 slots")
    slots: list[BellSlot] = []
    period = 0
    prev_end = None
    for i, s in enumerate(raw, start=1):
        start, end, kind = str(s.get("start", "")), str(s.get("end", "")), str(s.get("kind", "teaching"))
        if not _HHMM.match(start) or not _HHMM.match(end):
            raise ValueError(f"slot {i}: times are HH:MM, 24-hour (got {start!r} to {end!r})")
        if start >= end:
            raise ValueError(f"slot {i}: {start} to {end} ends before it starts")
        if prev_end is not None and start < prev_end:
            raise ValueError(f"slot {i}: starts at {start}, before the previous slot ends at {prev_end}")
        if kind not in SLOT_KINDS:
            raise ValueError(f"slot {i}: kind is one of {', '.join(SLOT_KINDS)}")
        if kind == "teaching":
            period += 1
        slots.append(BellSlot(start=start, end=end, kind=kind, period=period if kind == "teaching" else None))
        prev_end = end
    if period == 0:
        raise ValueError("a bell schedule needs at least one teaching period")
    return slots


def adjacent_pairs(bell: "BellSchedule") -> list[tuple[int, int]]:
    """Teaching periods (p, p+1) that follow each other with nothing in
    between -- the only places a double period can go. A break, assembly or
    zero slot between two periods splits them."""
    pairs = []
    for a, b in zip(bell.slots, bell.slots[1:]):
        if a.kind == "teaching" and b.kind == "teaching":
            pairs.append((a.period, b.period))
    return pairs


def _clock_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class _Clock:
    """Each section's periods as clock time, from the section's own bell, so
    two sections on different bells are compared by when they actually
    teach (N-3-4). A span is (start, end, period); a section with no bell
    has no times and is compared by period number, as it always was."""

    def __init__(self, store, sections: dict):
        self._store, self._sections = store, sections
        self._times: dict[str, dict[int, tuple[int, int]]] = {}

    def span(self, section_id: str, period: int) -> tuple:
        if section_id not in self._times:
            section = self._sections.get(section_id) or self._store.get_section(section_id)
            bell = self._store.bell_for_section(section) if section is not None else None
            self._times[section_id] = {s.period: (_clock_minutes(s.start), _clock_minutes(s.end))
                                       for s in bell.slots if s.kind == "teaching"} if bell else {}
        start, end = self._times[section_id].get(period, (None, None))
        return start, end, period


def _overlap(a: tuple, b: tuple) -> bool:
    if a[0] is None or b[0] is None:
        return a[2] == b[2]
    return a[0] < b[1] and b[0] < a[1]


class _Busy:
    """When a teacher or a room is already taken, per day, by clock time."""

    def __init__(self, clock: _Clock):
        self._clock = clock
        self._at: dict[tuple[str, int], list] = defaultdict(list)

    def add(self, who: str, entry) -> None:
        self._at[(who, entry.day_of_week)].append((self._clock.span(entry.section_id, entry.period), entry))

    def find(self, who: str, day: int, span: tuple):
        """An entry of `who`'s that overlaps `span` on `day`, or None."""
        return next((e for other, e in self._at.get((who, day), ()) if _overlap(span, other)), None)


@dataclass
class _GroupPeriod:
    """A period of an elective or combined class (N-3-20), as the clash
    checks see it: on its first member section's bell, which all its
    members share."""
    section_id: str
    day_of_week: int
    period: int
    group_name: str


_UNCHANGED: Any = object()


def clean_days(days: Optional[list[int]]) -> list[int]:
    if days is None:
        return list(DEFAULT_DAYS)
    cleaned = sorted(set(int(d) for d in days))
    if not cleaned or any(d < 0 or d > 6 for d in cleaned):
        raise ValueError("days are weekday numbers 0 (Monday) to 6 (Sunday), at least one")
    return cleaned


class SchoolModelMixin:
    """CurriculumStore's methods for allocations, rooms, bell schedules and the
    timetable. Errors: ValueError is the route's 422, KeyError its 404,
    InUse and TimetableClash its 409."""

    # ---------------- teaching allocations (M1.2) ----------------

    def _section_and_subject(self, section_id: str, subject_id: str):
        section = self.get_section(section_id)
        if section is None:
            raise KeyError(section_id)
        subject = self.get_subject(subject_id)
        if subject is None:
            raise KeyError(subject_id)
        if subject.grade_id != section.grade_id:
            raise ValueError("that subject belongs to another class; choose one of this class's subjects")
        return section, subject

    def set_allocation(self, *, section_id: str, subject_id: str, teacher_id: Optional[str],
                       periods_per_week: int, co_teacher_id: Any = _UNCHANGED,
                       room_kind: Any = _UNCHANGED,
                       double_periods: Any = _UNCHANGED) -> tuple[Optional[TeachingAllocation], TeachingAllocation]:
        """Upsert the (section, subject) cell. Returns (before, after); before
        is None for a new cell. Lowering periods below what the timetable
        already gives the subject is refused: remove periods first.

        co_teacher_id, room_kind and double_periods left out keep what the
        cell has (none, for a new cell), so a caller that only sets the
        teacher and the periods never wipes them."""
        if not 1 <= periods_per_week <= MAX_PERIODS_PER_WEEK:
            raise ValueError(f"periods per week is 1 to {MAX_PERIODS_PER_WEEK}")
        with self._conn_lock:
            section, _ = self._section_and_subject(section_id, subject_id)
            before = self.allocation_for(section_id, subject_id)
            co = (before.co_teacher_id if before else None) if co_teacher_id is _UNCHANGED else co_teacher_id
            kind = (before.room_kind if before else None) if room_kind is _UNCHANGED else room_kind
            doubles = (before.double_periods if before else 0) if double_periods is _UNCHANGED \
                else int(double_periods or 0)
            if co is not None and co == teacher_id:
                raise ValueError("the co-teacher is the subject's teacher; choose someone else, or none")
            if co is not None and teacher_id is None:
                raise ValueError("choose the subject's teacher before a co-teacher")
            if kind is not None and kind not in ROOM_KINDS:
                raise ValueError(f"a room kind is one of {', '.join(ROOM_KINDS)}")
            if doubles < 0 or 2 * doubles > periods_per_week:
                raise ValueError(f"{doubles} double period(s) take {2 * doubles} periods; this subject "
                                 f"has {periods_per_week} a week")
            for g in self.groups_for_section(section_id):
                if any(lane.subject_id == subject_id for lane in g.lanes):
                    raise ValueError(f"this section is taught that subject in the group {g.name}; "
                                     "change the group instead")
            placed = len(self._entries_where("section_id=? AND subject_id=?", (section_id, subject_id)))
            if placed > periods_per_week:
                raise ValueError(f"the timetable already gives this subject {placed} periods a week; "
                                 f"remove {placed - periods_per_week} first, or allocate at least {placed}")
            # Every check runs before the first write: a refusal after an
            # _exec would leave the write on the shared connection for the
            # next commit to save.
            if before is not None and co is not None and before.co_teacher_id != co:
                self._check_co_teacher_free(co, section, subject_id)
            if before is None:
                after = TeachingAllocation(id=_new_id("alloc"), school_id=section.school_id,
                                           academic_year_id=section.academic_year_id,
                                           section_id=section_id, subject_id=subject_id,
                                           teacher_id=teacher_id, periods_per_week=periods_per_week,
                                           created_at=_now(), co_teacher_id=co, room_kind=kind,
                                           double_periods=doubles)
                self._exec(
                    "INSERT INTO teaching_allocations (id, school_id, academic_year_id, section_id, "
                    "subject_id, teacher_id, periods_per_week, created_at, co_teacher_id, room_kind, "
                    "double_periods) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (after.id, after.school_id, after.academic_year_id, after.section_id,
                     after.subject_id, after.teacher_id, after.periods_per_week, after.created_at,
                     after.co_teacher_id, after.room_kind, after.double_periods))
            else:
                after = TeachingAllocation(**{**before.__dict__, "teacher_id": teacher_id,
                                              "periods_per_week": periods_per_week, "co_teacher_id": co,
                                              "room_kind": kind, "double_periods": doubles})
                self._exec("UPDATE teaching_allocations SET teacher_id=?, periods_per_week=?, "
                           "co_teacher_id=?, room_kind=?, double_periods=? WHERE id=?",
                           (teacher_id, periods_per_week, co, kind, doubles, before.id))
                if before.teacher_id != teacher_id:
                    # The timetable's periods for this cell follow the new teacher,
                    # unless one was set by hand to someone else.
                    self._exec("UPDATE timetable_entries SET teacher_id=? WHERE section_id=? AND "
                               "subject_id=? AND (teacher_id IS ? OR teacher_id=?)",
                               (teacher_id, section_id, subject_id, before.teacher_id, before.teacher_id))
                if before.co_teacher_id != co:
                    # The co-teacher is in every period of the cell (checked
                    # free above, before anything was written).
                    self._exec("UPDATE timetable_entries SET co_teacher_id=? WHERE section_id=? AND "
                               "subject_id=?", (co, section_id, subject_id))
        self._commit()
        return before, after

    def _check_co_teacher_free(self, co_teacher_id: str, section, subject_id: str) -> None:
        """TimetableClash (the route's 422) naming every placed period of the
        cell in which the new co-teacher already teaches elsewhere -- by
        clock time, so a period on another bell counts when it overlaps."""
        mine = self._entries_where("section_id=? AND subject_id=?", (section.id, subject_id))
        sections = {s.id: s for s in self.sections_for_year(section.academic_year_id)}
        clock = _Clock(self, sections)
        busy = _Busy(clock)
        for e in self.timetable_for_teacher(co_teacher_id, section.academic_year_id):
            if e.section_id != section.id:
                busy.add(co_teacher_id, e)
        self._group_busy(section.academic_year_id, busy)
        clashes = []
        for e in mine:
            other = busy.find(co_teacher_id, e.day_of_week, clock.span(section.id, e.period))
            if other is not None:
                clashes.append(f"{WEEKDAY_NAMES[e.day_of_week]} period {e.period}: the co-teacher already teaches "
                               f"{self._clash_label(other, sections)} then")
        if clashes:
            raise TimetableClash(clashes)

    def _group_busy(self, academic_year_id: str, teacher_busy: "_Busy",
                    room_busy: Optional["_Busy"] = None) -> None:
        """Each lane's teacher and room are taken for every period of their
        elective or combined class (N-3-20)."""
        for g, d, p in self.group_periods_for_year(academic_year_id):
            if not g.section_ids:
                continue
            slot = _GroupPeriod(section_id=g.section_ids[0], day_of_week=d, period=p, group_name=g.name)
            for t in g.teacher_ids:
                teacher_busy.add(t, slot)
            for r in g.room_ids if room_busy is not None else ():
                room_busy.add(r, slot)

    def _clash_label(self, clash, sections: dict) -> str:
        """Who a clash is with: a group by its name, else the section."""
        if getattr(clash, "group_name", None):
            return f"the group {clash.group_name}"
        other = sections.get(clash.section_id)
        return self._section_label(other) if other else "another section"

    def allocation_for(self, section_id: str, subject_id: str) -> Optional[TeachingAllocation]:
        r = self._fetchone("SELECT * FROM teaching_allocations WHERE section_id=? AND subject_id=?",
                           (section_id, subject_id))
        return TeachingAllocation(**r) if r else None

    def allocations_for_year(self, academic_year_id: str) -> list[TeachingAllocation]:
        return [TeachingAllocation(**r) for r in self._fetchall(
            "SELECT * FROM teaching_allocations WHERE academic_year_id=? ORDER BY section_id, subject_id",
            (academic_year_id,))]

    def allocations_for_teacher(self, teacher_id: str,
                                academic_year_id: Optional[str] = None) -> list[TeachingAllocation]:
        sql = "SELECT * FROM teaching_allocations WHERE teacher_id=?"
        params: tuple = (teacher_id,)
        if academic_year_id:
            sql += " AND academic_year_id=?"
            params += (academic_year_id,)
        return [TeachingAllocation(**r) for r in self._fetchall(sql, params)]

    def co_taught_allocations(self, teacher_id: str,
                              academic_year_id: Optional[str] = None) -> list[TeachingAllocation]:
        """The cells a teacher co-teaches (SCH-8)."""
        sql = "SELECT * FROM teaching_allocations WHERE co_teacher_id=?"
        params: tuple = (teacher_id,)
        if academic_year_id:
            sql += " AND academic_year_id=?"
            params += (academic_year_id,)
        return [TeachingAllocation(**r) for r in self._fetchall(sql, params)]

    def delete_allocation(self, section_id: str, subject_id: str) -> TeachingAllocation:
        with self._conn_lock:
            alloc = self.allocation_for(section_id, subject_id)
            if alloc is None:
                raise KeyError(f"{section_id}/{subject_id}")
            placed = len(self._entries_where("section_id=? AND subject_id=?", (section_id, subject_id)))
            if placed:
                raise InUse(f"the timetable gives this subject {placed} period(s) a week; "
                            "remove them from the timetable first")
            self._exec("DELETE FROM teaching_allocations WHERE id=?", (alloc.id,))
        self._commit()
        return alloc

    def teacher_load(self, academic_year_id: str) -> list[dict[str, Any]]:
        """Per teacher: periods a week allocated and timetabled, and the most
        in one day -- the overload check the allocation grid shows."""
        allocated: Counter = Counter()
        cells: Counter = Counter()
        for a in self.allocations_for_year(academic_year_id):
            # A co-teacher is in the room for every period too (SCH-8).
            for t in {a.teacher_id, a.co_teacher_id} - {None}:
                allocated[t] += a.periods_per_week
                cells[t] += 1
        per_day: dict[str, Counter] = defaultdict(Counter)
        for e in self.timetable_for_year(academic_year_id):
            for t in {e.teacher_id, e.co_teacher_id} - {None}:
                per_day[t][e.day_of_week] += 1
        # A lane of an elective or a combined class is one class a teacher
        # takes, for the group's periods (N-3-20).
        for g in self.teaching_groups_for_year(academic_year_id):
            for t in g.teacher_ids:
                allocated[t] += g.periods_per_week
                cells[t] += 1
        for g, d, _ in self.group_periods_for_year(academic_year_id):
            for t in g.teacher_ids:
                per_day[t][d] += 1
        teachers = set(allocated) | set(per_day)
        return sorted(
            ({"teacherId": t, "allocatedPerWeek": allocated[t], "sectionSubjects": cells[t],
              "timetabledPerWeek": sum(per_day[t].values()),
              "maxInOneDay": max(per_day[t].values(), default=0)}
             for t in teachers),
            key=lambda row: (-row["allocatedPerWeek"], row["teacherId"]))

    # ---------------- rooms (M1.3) ----------------

    def create_room(self, *, school_id: str, name: str, kind: str = "classroom",
                    capacity: Optional[int] = None) -> Room:
        name = " ".join((name or "").split())
        if not name or len(name) > 40:
            raise ValueError("a room needs a name of up to 40 characters")
        if kind not in ROOM_KINDS:
            raise ValueError(f"a room's kind is one of {', '.join(ROOM_KINDS)}")
        if capacity is not None and not 1 <= capacity <= 1000:
            raise ValueError("capacity is 1 to 1000")
        with self._conn_lock:
            if self._fetchone("SELECT 1 FROM rooms WHERE school_id=? AND name=?", (school_id, name)):
                raise ValueError(f"there is already a room named {name!r}")
            room = Room(id=_new_id("room"), school_id=school_id, name=name, kind=kind,
                        capacity=capacity, created_at=_now())
            self._exec("INSERT INTO rooms (id, school_id, name, kind, capacity, created_at) "
                       "VALUES (?,?,?,?,?,?)",
                       (room.id, room.school_id, room.name, room.kind, room.capacity, room.created_at))
        self._commit()
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        r = self._fetchone("SELECT * FROM rooms WHERE id=?", (room_id,))
        return Room(**r) if r else None

    def rooms_for_school(self, school_id: str) -> list[Room]:
        return [Room(**r) for r in self._fetchall(
            "SELECT * FROM rooms WHERE school_id=? ORDER BY name COLLATE NOCASE", (school_id,))]

    def update_room(self, room_id: str, *, name: Optional[str] = None, kind: Optional[str] = None,
                    capacity: Any = ...) -> tuple[Room, Room]:
        with self._conn_lock:
            before = self.get_room(room_id)
            if before is None:
                raise KeyError(room_id)
            new_name = before.name if name is None else " ".join(name.split())
            if not new_name or len(new_name) > 40:
                raise ValueError("a room needs a name of up to 40 characters")
            if new_name.casefold() != before.name.casefold() and self._fetchone(
                    "SELECT 1 FROM rooms WHERE school_id=? AND name=? AND id<>?",
                    (before.school_id, new_name, room_id)):
                raise ValueError(f"there is already a room named {new_name!r}")
            new_kind = before.kind if kind is None else kind
            if new_kind not in ROOM_KINDS:
                raise ValueError(f"a room's kind is one of {', '.join(ROOM_KINDS)}")
            new_capacity = before.capacity if capacity is ... else capacity
            if new_capacity is not None and not 1 <= new_capacity <= 1000:
                raise ValueError("capacity is 1 to 1000")
            self._exec("UPDATE rooms SET name=?, kind=?, capacity=? WHERE id=?",
                       (new_name, new_kind, new_capacity, room_id))
        self._commit()
        return before, Room(**{**before.__dict__, "name": new_name, "kind": new_kind,
                               "capacity": new_capacity})

    def delete_room(self, room_id: str) -> Room:
        with self._conn_lock:
            room = self.get_room(room_id)
            if room is None:
                raise KeyError(room_id)
            used = self._entries_where("room_id=?", (room_id,))
            if used:
                raise InUse(f"the timetable uses this room for {len(used)} period(s); "
                            "move those periods first")
            self._exec("DELETE FROM rooms WHERE id=?", (room_id,))
            self._exec("DELETE FROM room_unavailability WHERE room_id=?", (room_id,))
        self._commit()
        return room

    # ---------------- bell schedules (M1.3) ----------------

    def _bell_from_row(self, r: dict) -> BellSchedule:
        return BellSchedule(id=r["id"], school_id=r["school_id"], academic_year_id=r["academic_year_id"],
                            name=r["name"], days=json.loads(r["days_json"]),
                            slots=[BellSlot(**s) for s in json.loads(r["slots_json"])],
                            is_default=bool(r["is_default"]), created_at=r["created_at"])

    def create_bell_schedule(self, *, academic_year_id: str, name: str, slots: list[dict[str, Any]],
                             days: Optional[list[int]] = None) -> BellSchedule:
        name = " ".join((name or "").split())
        if not name or len(name) > 40:
            raise ValueError("a bell schedule needs a name of up to 40 characters")
        cleaned, cleaned_days = clean_slots(slots), clean_days(days)
        with self._conn_lock:
            year = self.get_academic_year(academic_year_id)
            if year is None:
                raise KeyError(academic_year_id)
            first = not self.bell_schedules_for_year(academic_year_id)
            b = BellSchedule(id=_new_id("bell"), school_id=year.school_id,
                             academic_year_id=academic_year_id, name=name, days=cleaned_days,
                             slots=cleaned, is_default=first, created_at=_now())
            self._exec("INSERT INTO bell_schedules (id, school_id, academic_year_id, name, days_json, "
                       "slots_json, is_default, created_at) VALUES (?,?,?,?,?,?,?,?)",
                       (b.id, b.school_id, b.academic_year_id, b.name, json.dumps(b.days),
                        json.dumps([s.__dict__ for s in b.slots]), int(b.is_default), b.created_at))
        self._commit()
        return b

    def get_bell_schedule(self, bell_id: str) -> Optional[BellSchedule]:
        r = self._fetchone("SELECT * FROM bell_schedules WHERE id=?", (bell_id,))
        return self._bell_from_row(r) if r else None

    def bell_schedules_for_year(self, academic_year_id: str) -> list[BellSchedule]:
        return [self._bell_from_row(r) for r in self._fetchall(
            "SELECT * FROM bell_schedules WHERE academic_year_id=? ORDER BY is_default DESC, name",
            (academic_year_id,))]

    def update_bell_schedule(self, bell_id: str, *, name: Optional[str] = None,
                             slots: Optional[list[dict[str, Any]]] = None,
                             days: Optional[list[int]] = None,
                             make_default: bool = False) -> BellSchedule:
        """Refuses a change that would strand timetable periods: fewer
        teaching periods than a section on this schedule already uses, or a
        dropped day that has periods."""
        with self._conn_lock:
            b = self.get_bell_schedule(bell_id)
            if b is None:
                raise KeyError(bell_id)
            new_name = b.name if name is None else " ".join(name.split())
            if not new_name or len(new_name) > 40:
                raise ValueError("a bell schedule needs a name of up to 40 characters")
            new_slots = b.slots if slots is None else clean_slots(slots)
            new_days = b.days if days is None else clean_days(days)
            teaching = sum(1 for s in new_slots if s.kind == "teaching")
            for section in self._sections_on_bell(b):
                for e in self._entries_where("section_id=?", (section.id,)):
                    if e.period > teaching:
                        raise ValueError(f"{self._section_label(section)} has a period {e.period} "
                                         f"on {WEEKDAY_NAMES[e.day_of_week]}; this schedule would "
                                         f"have {teaching} -- move it first")
                    if e.day_of_week not in new_days:
                        raise ValueError(f"{self._section_label(section)} has periods on "
                                         f"{WEEKDAY_NAMES[e.day_of_week]}; move them before "
                                         "dropping the day")
            if make_default:
                self._exec("UPDATE bell_schedules SET is_default=0 WHERE academic_year_id=?",
                           (b.academic_year_id,))
            self._exec("UPDATE bell_schedules SET name=?, days_json=?, slots_json=?, is_default=? "
                       "WHERE id=?",
                       (new_name, json.dumps(new_days), json.dumps([s.__dict__ for s in new_slots]),
                        int(make_default or b.is_default), bell_id))
        self._commit()
        return self.get_bell_schedule(bell_id)

    def delete_bell_schedule(self, bell_id: str) -> BellSchedule:
        with self._conn_lock:
            b = self.get_bell_schedule(bell_id)
            if b is None:
                raise KeyError(bell_id)
            if b.is_default and len(self.bell_schedules_for_year(b.academic_year_id)) > 1:
                raise InUse("this is the year's default schedule; make another one the default first")
            users = [s for s in self._sections_on_bell(b) if s.bell_schedule_id == bell_id]
            if users:
                raise InUse(f"{len(users)} section(s) use this schedule; move them first")
            if b.is_default and self._entries_where("academic_year_id=?", (b.academic_year_id,)):
                raise InUse("the timetable uses this schedule's periods; clear it first")
            self._exec("DELETE FROM bell_schedules WHERE id=?", (bell_id,))
        self._commit()
        return b

    def bell_for_section(self, section) -> Optional[BellSchedule]:
        """The section's own schedule, else its year's default, else None."""
        if getattr(section, "bell_schedule_id", None):
            own = self.get_bell_schedule(section.bell_schedule_id)
            if own is not None:
                return own
        for b in self.bell_schedules_for_year(section.academic_year_id):
            if b.is_default:
                return b
        return None

    def set_section_bell(self, section_id: str, bell_id: Optional[str]) -> None:
        with self._conn_lock:
            section = self.get_section(section_id)
            if section is None:
                raise KeyError(section_id)
            if bell_id is not None:
                b = self.get_bell_schedule(bell_id)
                if b is None or b.academic_year_id != section.academic_year_id:
                    raise ValueError("that bell schedule is not this section's year's")
                teaching, days = b.teaching_periods, b.days
                for e in self._entries_where("section_id=?", (section_id,)):
                    if e.period > teaching or e.day_of_week not in days:
                        raise ValueError(f"the section has a period {e.period} on "
                                         f"{WEEKDAY_NAMES[e.day_of_week]} this schedule does not have")
            self._exec("UPDATE sections SET bell_schedule_id=? WHERE id=?", (bell_id, section_id))
        self._commit()

    def _sections_on_bell(self, b: BellSchedule) -> list:
        """The sections whose bell is `b`: their own, or the default's."""
        out = []
        for s in self.sections_for_year(b.academic_year_id):
            own = getattr(s, "bell_schedule_id", None)
            if own == b.id or (own is None and b.is_default):
                out.append(s)
        return out

    def _section_label(self, section) -> str:
        grade = self.get_grade(section.grade_id)
        return f"{grade.number if grade else '?'}-{section.name}"

    # ---------------- the timetable (SCH-2) ----------------

    def _entries_where(self, where: str, params: tuple) -> list[TimetableEntry]:
        return [TimetableEntry(**r) for r in self._fetchall(
            f"SELECT * FROM timetable_entries WHERE {where} ORDER BY day_of_week, period", params)]

    def timetable_for_year(self, academic_year_id: str) -> list[TimetableEntry]:
        return self._entries_where("academic_year_id=?", (academic_year_id,))

    def timetable_for_section(self, section_id: str) -> list[TimetableEntry]:
        return self._entries_where("section_id=?", (section_id,))

    def timetable_for_teacher(self, teacher_id: str, academic_year_id: str) -> list[TimetableEntry]:
        """The periods a teacher is in: their own, and those they co-teach."""
        return self._entries_where("(teacher_id=? OR co_teacher_id=?) AND academic_year_id=?",
                                   (teacher_id, teacher_id, academic_year_id))

    def replace_section_timetable(self, section_id: str,
                                  entries: list[dict[str, Any]]) -> list[TimetableEntry]:
        """The section's whole week at once: every entry is checked before
        anything is written, so a refused week leaves the old one intact.
        Each entry: day_of_week, period, subject_id, optional teacher_id
        (default: the allocation's teacher) and room_id. TimetableClash lists
        every problem."""
        with self._conn_lock:
            section = self.get_section(section_id)
            if section is None:
                raise KeyError(section_id)
            bell = self.bell_for_section(section)
            if bell is None and entries:
                raise ValueError("set up the year's bell schedule first: a period number means "
                                 "nothing until the school's periods are defined")
            label = self._section_label(section)
            problems: list[str] = []
            rows: list[TimetableEntry] = []
            seen: set[tuple[int, int]] = set()
            per_subject: Counter = Counter()
            allocs = {a.subject_id: a for a in self.allocations_for_year(section.academic_year_id)
                      if a.section_id == section_id}
            # Everyone else's week this year, to check teachers and rooms against.
            # A co-teacher is busy in a period just as its teacher is (SCH-8).
            # By clock time (N-3-4): another section on another bell clashes
            # when its period overlaps this one, whatever the two numbers.
            sections_by_id = {s.id: s for s in self.sections_for_year(section.academic_year_id)}
            clock = _Clock(self, sections_by_id)
            teacher_busy, room_busy = _Busy(clock), _Busy(clock)
            for e in self.timetable_for_year(section.academic_year_id):
                if e.section_id == section_id:
                    continue
                for t in (e.teacher_id, e.co_teacher_id):
                    if t:
                        teacher_busy.add(t, e)
                if e.room_id:
                    room_busy.add(e.room_id, e)
            # SCH-8 (N-3-20): this section's periods in an elective or a
            # combined class are the group's, and every lane's teacher and
            # room are taken in each of the group's periods.
            self._group_busy(section.academic_year_id, teacher_busy, room_busy)
            in_group = {(d, p): g for g, d, p in self.group_periods_for_section(section_id)}
            room_closed = self.room_unavailability_for_year(section.academic_year_id)
            for i, raw in enumerate(entries, start=1):
                day, period = int(raw["day_of_week"]), int(raw["period"])
                subject_id = raw["subject_id"]
                where = f"{WEEKDAY_NAMES[day] if 0 <= day <= 6 else f'day {day}'} period {period}"
                if day not in bell.days:
                    problems.append(f"{where}: the school does not teach on {WEEKDAY_NAMES[day] if 0 <= day <= 6 else f'day {day}'} "
                                    f"in {bell.name}")
                    continue
                if not 1 <= period <= bell.teaching_periods:
                    problems.append(f"{where}: {bell.name} has periods 1 to {bell.teaching_periods}")
                    continue
                if (day, period) in seen:
                    problems.append(f"{where}: given twice for {label}")
                    continue
                seen.add((day, period))
                if (day, period) in in_group:
                    problems.append(f"{where}: {label} is in {in_group[(day, period)].name} then; "
                                    "change the group, not the section")
                    continue
                alloc = allocs.get(subject_id)
                if alloc is None:
                    problems.append(f"{where}: {label} has no allocation for that subject; "
                                    "allocate it (teacher and periods a week) first")
                    continue
                teacher_id = raw.get("teacher_id") or alloc.teacher_id
                co_teacher_id = alloc.co_teacher_id if alloc.co_teacher_id != teacher_id else None
                room_id = raw.get("room_id")
                if room_id is not None:
                    room = self.get_room(room_id)
                    if room is None or room.school_id != section.school_id:
                        problems.append(f"{where}: that room is not this school's")
                        continue
                    if alloc.room_kind and room.kind != alloc.room_kind:
                        problems.append(f"{where}: this subject needs a {alloc.room_kind}; "
                                        f"{room.name} is a {room.kind}")
                    if (day, period) in room_closed.get(room_id, set()):
                        problems.append(f"{where}: {room.name} is not available then")
                elif alloc.room_kind:
                    problems.append(f"{where}: this subject needs a {alloc.room_kind}; choose one")
                span = clock.span(section_id, period)
                for who, t in (("teacher", teacher_id), ("co-teacher", co_teacher_id)):
                    clash = teacher_busy.find(t, day, span) if t else None
                    if clash is not None:
                        problems.append(f"{where}: the {who} already teaches "
                                        f"{self._clash_label(clash, sections_by_id)} then")
                clash = room_busy.find(room_id, day, span) if room_id else None
                if clash is not None:
                    problems.append(f"{where}: the room is already used by "
                                    f"{self._clash_label(clash, sections_by_id)} then")
                per_subject[subject_id] += 1
                rows.append(TimetableEntry(id=_new_id("tt"), school_id=section.school_id,
                                           academic_year_id=section.academic_year_id,
                                           section_id=section_id, day_of_week=day, period=period,
                                           subject_id=subject_id, teacher_id=teacher_id,
                                           room_id=room_id, created_at=_now(),
                                           locked=int(bool(raw.get("locked"))),
                                           co_teacher_id=co_teacher_id))
            for subject_id, n in per_subject.items():
                allowed = allocs[subject_id].periods_per_week
                if n > allowed:
                    subject = self.get_subject(subject_id)
                    problems.append(f"{subject.name if subject else subject_id} gets {n} periods a week "
                                    f"here but is allocated {allowed}")
            if problems:
                raise TimetableClash(problems)
            self._exec("DELETE FROM timetable_entries WHERE section_id=?", (section_id,))
            for e in rows:
                self._insert_entry(e)
        self._commit()
        return self.timetable_for_section(section_id)

    def _insert_entry(self, e: TimetableEntry) -> None:
        self._exec(
            "INSERT INTO timetable_entries (id, school_id, academic_year_id, section_id, "
            "day_of_week, period, subject_id, teacher_id, room_id, created_at, locked, co_teacher_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (e.id, e.school_id, e.academic_year_id, e.section_id, e.day_of_week, e.period,
             e.subject_id, e.teacher_id, e.room_id, e.created_at, e.locked, e.co_teacher_id))

    # ---------------- solver support (SCH-3) ----------------

    def locked_entries_for_year(self, academic_year_id: str) -> list[TimetableEntry]:
        return self._entries_where("academic_year_id=? AND locked=1", (academic_year_id,))

    def teacher_unavailability_for_year(self, academic_year_id: str) -> dict[str, set[tuple[int, int]]]:
        out: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for r in self._fetchall("SELECT teacher_id, day_of_week, period FROM teacher_unavailability "
                                "WHERE academic_year_id=?", (academic_year_id,)):
            out[r["teacher_id"]].add((r["day_of_week"], r["period"]))
        return dict(out)

    def set_teacher_unavailability(self, academic_year_id: str, teacher_id: str,
                                   slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Replace one teacher's unavailable periods for the year."""
        cleaned = sorted({(int(d), int(p)) for d, p in slots})
        if any(not 0 <= d <= 6 or not 1 <= p <= 20 for d, p in cleaned):
            raise ValueError("a period is a day 0 (Monday) to 6 and a period number 1 to 20")
        with self._conn_lock:
            self._exec("DELETE FROM teacher_unavailability WHERE academic_year_id=? AND teacher_id=?",
                       (academic_year_id, teacher_id))
            for d, p in cleaned:
                self._exec("INSERT INTO teacher_unavailability (academic_year_id, teacher_id, "
                           "day_of_week, period) VALUES (?,?,?,?)", (academic_year_id, teacher_id, d, p))
        self._commit()
        return cleaned

    def room_unavailability_for_year(self, academic_year_id: str) -> dict[str, set[tuple[int, int]]]:
        out: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for r in self._fetchall("SELECT room_id, day_of_week, period FROM room_unavailability "
                                "WHERE academic_year_id=?", (academic_year_id,)):
            out[r["room_id"]].add((r["day_of_week"], r["period"]))
        return dict(out)

    def set_room_unavailability(self, academic_year_id: str, room_id: str,
                                slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Replace one room's unavailable periods for the year (SCH-8: a lab
        out of use). Refused while the timetable books the room in one of
        them: move those periods first, or the week would silently hold a
        class in a closed lab."""
        cleaned = sorted({(int(d), int(p)) for d, p in slots})
        if any(not 0 <= d <= 6 or not 1 <= p <= 20 for d, p in cleaned):
            raise ValueError("a period is a day 0 (Monday) to 6 and a period number 1 to 20")
        with self._conn_lock:
            if self.get_room(room_id) is None:
                raise KeyError(room_id)
            booked = [e for e in self._entries_where("room_id=? AND academic_year_id=?",
                                                     (room_id, academic_year_id))
                      if (e.day_of_week, e.period) in set(cleaned)]
            if booked:
                sections = {s.id: s for s in self.sections_for_year(academic_year_id)}
                where = ", ".join(f"{WEEKDAY_NAMES[e.day_of_week]} period {e.period} "
                                  f"({self._section_label(sections[e.section_id]) if e.section_id in sections else '?'})"
                                  for e in booked[:5])
                raise InUse(f"the timetable uses this room then: {where}; move those periods first")
            self._exec("DELETE FROM room_unavailability WHERE academic_year_id=? AND room_id=?",
                       (academic_year_id, room_id))
            for d, p in cleaned:
                self._exec("INSERT INTO room_unavailability (academic_year_id, room_id, day_of_week, "
                           "period) VALUES (?,?,?,?)", (academic_year_id, room_id, d, p))
        self._commit()
        return cleaned

    def apply_solved_timetable(self, academic_year_id: str, entries: list,
                               section_ids: set[str], group_sessions: list = (),
                               group_ids: set[str] = frozenset()) -> int:
        """Write a solver's week for `section_ids` -- and the periods of the
        groups in `group_ids` (N-3-20) -- in one lock hold. An entry
        identical to a locked one stays locked. Returns entries written."""
        with self._conn_lock:
            self._replace_group_sessions(academic_year_id, set(group_ids), group_sessions)
            year = self.get_academic_year(academic_year_id)
            if year is None:
                raise KeyError(academic_year_id)
            was_locked = {(e.section_id, e.day_of_week, e.period, e.subject_id)
                          for e in self.locked_entries_for_year(academic_year_id)}
            for sid in section_ids:
                self._exec("DELETE FROM timetable_entries WHERE section_id=?", (sid,))
            written = 0
            for p in entries:
                if p.section_id not in section_ids:
                    continue
                self._insert_entry(TimetableEntry(
                    id=_new_id("tt"), school_id=year.school_id, academic_year_id=academic_year_id,
                    section_id=p.section_id, day_of_week=p.day_of_week, period=p.period,
                    subject_id=p.subject_id, teacher_id=p.teacher_id, room_id=p.room_id,
                    created_at=_now(),
                    locked=int((p.section_id, p.day_of_week, p.period, p.subject_id) in was_locked),
                    co_teacher_id=getattr(p, "co_teacher_id", None)))
                written += 1
        self._commit()
        return written

    # ---------------- previewed weeks (SCH-3, audit N-3-6) ----------------

    def timetable_inputs_hash(self, academic_year_id: str) -> str:
        """A fingerprint of everything a generated week is made from: the
        year's sections and their bells, allocations, bell schedules, the
        current timetable (locks included), unavailable periods and rooms.
        Equal fingerprints mean a previewed week still fits the school."""
        import hashlib
        with self._conn_lock:
            year = self.get_academic_year(academic_year_id)

            def rows(sql: str, params: tuple = (academic_year_id,)) -> list:
                return sorted(json.dumps({k: v for k, v in r.items() if k not in ("id", "created_at")},
                                         sort_keys=True, default=str)
                              for r in self._fetchall(sql, params))
            parts = {
                "sections": rows("SELECT id AS sid, bell_schedule_id FROM sections WHERE academic_year_id=?"),
                "allocations": rows("SELECT * FROM teaching_allocations WHERE academic_year_id=?"),
                "bells": rows("SELECT id AS bid, days_json, slots_json, is_default FROM bell_schedules "
                              "WHERE academic_year_id=?"),
                "timetable": rows("SELECT * FROM timetable_entries WHERE academic_year_id=?"),
                "teachers_away": rows("SELECT * FROM teacher_unavailability WHERE academic_year_id=?"),
                "rooms_closed": rows("SELECT * FROM room_unavailability WHERE academic_year_id=?"),
                "rooms": rows("SELECT id AS rid, kind FROM rooms WHERE school_id=?",
                              (year.school_id if year else "",)),
                "groups": rows("SELECT id AS gid, kind, periods_per_week, section_ids_json, lanes_json "
                               "FROM teaching_groups WHERE academic_year_id=?"),
                "group_periods": rows("SELECT * FROM group_sessions WHERE academic_year_id=?"),
            }
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def save_timetable_preview(self, *, academic_year_id: str, section_ids: set[str], entries: list,
                               summary: dict[str, Any], inputs_hash: str, created_by: Optional[str],
                               group_sessions: list = (), group_ids: set[str] = frozenset()) -> str:
        """Keep a generated week to publish later; returns its id. Only the
        year's newest PREVIEWS_KEPT are kept."""
        with self._conn_lock:
            year = self.get_academic_year(academic_year_id)
            if year is None:
                raise KeyError(academic_year_id)
            preview_id = _new_id("ttprev")
            self._exec("INSERT INTO timetable_previews (id, school_id, academic_year_id, section_ids_json, "
                       "entries_json, summary_json, inputs_hash, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (preview_id, year.school_id, academic_year_id, json.dumps(sorted(section_ids)),
                        json.dumps({"entries": [dict(e.__dict__) for e in entries],
                                    "groupSessions": [dict(s.__dict__) for s in group_sessions],
                                    "groupIds": sorted(group_ids)}),
                        json.dumps(summary), inputs_hash, created_by, _now()))
            self._exec("DELETE FROM timetable_previews WHERE academic_year_id=? AND id NOT IN "
                       "(SELECT id FROM timetable_previews WHERE academic_year_id=? "
                       "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                       (academic_year_id, academic_year_id, PREVIEWS_KEPT))
        self._commit()
        return preview_id

    def get_timetable_preview(self, preview_id: str) -> Optional[dict[str, Any]]:
        r = self._fetchone("SELECT * FROM timetable_previews WHERE id=?", (preview_id,))
        if r is None:
            return None
        week = json.loads(r["entries_json"])
        return {"id": r["id"], "school_id": r["school_id"], "academic_year_id": r["academic_year_id"],
                "section_ids": set(json.loads(r["section_ids_json"])), "entries": week["entries"],
                "group_sessions": week["groupSessions"], "group_ids": set(week["groupIds"]),
                "summary": json.loads(r["summary_json"]), "inputs_hash": r["inputs_hash"],
                "created_by": r["created_by"], "created_at": r["created_at"]}

    def publish_timetable_preview(self, preview_id: str, academic_year_id: str,
                                  make_entry) -> tuple[dict[str, Any], list[TimetableEntry], list]:
        """Write exactly the previewed week, checked and written in one lock
        hold. KeyError: no such preview for this year. StalePreview: what it
        was made from has changed since. `make_entry` turns a stored entry
        back into the solver's ProposedEntry. Returns (preview, the solved
        sections' week as it was before, and the solved groups' periods as
        they were: (group, day, period))."""
        with self._conn_lock:
            preview = self.get_timetable_preview(preview_id)
            if preview is None or preview["academic_year_id"] != academic_year_id:
                raise KeyError(preview_id)
            if self.timetable_inputs_hash(academic_year_id) != preview["inputs_hash"]:
                raise StalePreview(preview_id)
            before = [e for e in self.timetable_for_year(academic_year_id)
                      if e.section_id in preview["section_ids"]]
            groups_before = [(g, d, p) for g, d, p in self.group_periods_for_year(academic_year_id)
                             if g.id in preview["group_ids"]]
            from .teaching_groups import GroupSession
            self.apply_solved_timetable(academic_year_id, [make_entry(d) for d in preview["entries"]],
                                        preview["section_ids"],
                                        [GroupSession(**s) for s in preview["group_sessions"]],
                                        preview["group_ids"])
            # Every other preview of the year was made from the week just replaced.
            self._exec("DELETE FROM timetable_previews WHERE academic_year_id=?", (academic_year_id,))
        self._commit()
        return preview, before, groups_before

    def timetable_gaps(self, section_id: str) -> list[dict[str, Any]]:
        """Per allocated subject of the section: allocated vs timetabled
        periods a week (a shortfall is what the solver or the principal has
        still to place)."""
        placed = Counter(e.subject_id for e in self.timetable_for_section(section_id))
        return [{"subjectId": a.subject_id, "allocated": a.periods_per_week,
                 "timetabled": placed[a.subject_id]}
                for a in self.allocations_for_year(self.get_section(section_id).academic_year_id)
                if a.section_id == section_id]
