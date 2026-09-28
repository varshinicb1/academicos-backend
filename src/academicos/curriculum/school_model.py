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
                       periods_per_week: int) -> tuple[Optional[TeachingAllocation], TeachingAllocation]:
        """Upsert the (section, subject) cell. Returns (before, after); before
        is None for a new cell. Lowering periods below what the timetable
        already gives the subject is refused: remove periods first."""
        if not 1 <= periods_per_week <= MAX_PERIODS_PER_WEEK:
            raise ValueError(f"periods per week is 1 to {MAX_PERIODS_PER_WEEK}")
        with self._conn_lock:
            section, _ = self._section_and_subject(section_id, subject_id)
            before = self.allocation_for(section_id, subject_id)
            placed = len(self._entries_where("section_id=? AND subject_id=?", (section_id, subject_id)))
            if placed > periods_per_week:
                raise ValueError(f"the timetable already gives this subject {placed} periods a week; "
                                 f"remove {placed - periods_per_week} first, or allocate at least {placed}")
            if before is None:
                after = TeachingAllocation(id=_new_id("alloc"), school_id=section.school_id,
                                           academic_year_id=section.academic_year_id,
                                           section_id=section_id, subject_id=subject_id,
                                           teacher_id=teacher_id, periods_per_week=periods_per_week,
                                           created_at=_now())
                self._exec(
                    "INSERT INTO teaching_allocations (id, school_id, academic_year_id, section_id, "
                    "subject_id, teacher_id, periods_per_week, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (after.id, after.school_id, after.academic_year_id, after.section_id,
                     after.subject_id, after.teacher_id, after.periods_per_week, after.created_at))
            else:
                after = TeachingAllocation(**{**before.__dict__, "teacher_id": teacher_id,
                                              "periods_per_week": periods_per_week})
                self._exec("UPDATE teaching_allocations SET teacher_id=?, periods_per_week=? WHERE id=?",
                           (teacher_id, periods_per_week, before.id))
                if before.teacher_id != teacher_id:
                    # The timetable's periods for this cell follow the new teacher,
                    # unless one was set by hand to a co-teacher.
                    self._exec("UPDATE timetable_entries SET teacher_id=? WHERE section_id=? AND "
                               "subject_id=? AND (teacher_id IS ? OR teacher_id=?)",
                               (teacher_id, section_id, subject_id, before.teacher_id, before.teacher_id))
        self._commit()
        return before, after

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
            if a.teacher_id:
                allocated[a.teacher_id] += a.periods_per_week
                cells[a.teacher_id] += 1
        per_day: dict[str, Counter] = defaultdict(Counter)
        for e in self.timetable_for_year(academic_year_id):
            if e.teacher_id:
                per_day[e.teacher_id][e.day_of_week] += 1
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
        return self._entries_where("teacher_id=? AND academic_year_id=?", (teacher_id, academic_year_id))

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
            others = [e for e in self.timetable_for_year(section.academic_year_id)
                      if e.section_id != section_id]
            teacher_busy = {(e.teacher_id, e.day_of_week, e.period): e for e in others if e.teacher_id}
            room_busy = {(e.room_id, e.day_of_week, e.period): e for e in others if e.room_id}
            sections_by_id = {s.id: s for s in self.sections_for_year(section.academic_year_id)}
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
                alloc = allocs.get(subject_id)
                if alloc is None:
                    problems.append(f"{where}: {label} has no allocation for that subject; "
                                    "allocate it (teacher and periods a week) first")
                    continue
                teacher_id = raw.get("teacher_id") or alloc.teacher_id
                room_id = raw.get("room_id")
                if room_id is not None:
                    room = self.get_room(room_id)
                    if room is None or room.school_id != section.school_id:
                        problems.append(f"{where}: that room is not this school's")
                        continue
                if teacher_id and (teacher_id, day, period) in teacher_busy:
                    other = sections_by_id.get(teacher_busy[(teacher_id, day, period)].section_id)
                    problems.append(f"{where}: the teacher already teaches "
                                    f"{self._section_label(other) if other else 'another section'} then")
                if room_id and (room_id, day, period) in room_busy:
                    other = sections_by_id.get(room_busy[(room_id, day, period)].section_id)
                    problems.append(f"{where}: the room is already used by "
                                    f"{self._section_label(other) if other else 'another section'} then")
                per_subject[subject_id] += 1
                rows.append(TimetableEntry(id=_new_id("tt"), school_id=section.school_id,
                                           academic_year_id=section.academic_year_id,
                                           section_id=section_id, day_of_week=day, period=period,
                                           subject_id=subject_id, teacher_id=teacher_id,
                                           room_id=room_id, created_at=_now()))
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
                self._exec(
                    "INSERT INTO timetable_entries (id, school_id, academic_year_id, section_id, "
                    "day_of_week, period, subject_id, teacher_id, room_id, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (e.id, e.school_id, e.academic_year_id, e.section_id, e.day_of_week, e.period,
                     e.subject_id, e.teacher_id, e.room_id, e.created_at))
        self._commit()
        return self.timetable_for_section(section_id)

    def timetable_gaps(self, section_id: str) -> list[dict[str, Any]]:
        """Per allocated subject of the section: allocated vs timetabled
        periods a week (a shortfall is what the solver or the principal has
        still to place)."""
        placed = Counter(e.subject_id for e in self.timetable_for_section(section_id))
        return [{"subjectId": a.subject_id, "allocated": a.periods_per_week,
                 "timetabled": placed[a.subject_id]}
                for a in self.allocations_for_year(self.get_section(section_id).academic_year_id)
                if a.section_id == section_id]
