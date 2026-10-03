"""Teaching that spans sections (REQUIREMENTS SCH-2/SCH-8, v3 audit N-3-20).

- An ELECTIVE group: the students of one or more sections of a class split
  into parallel lanes -- Hindi with one teacher, Sanskrit with another,
  French with a third -- all taught at the same time.
- A COMBINED class: two or more sections of a class taught together, by one
  teacher, in one room.

A group is scheduled as one block: each of its periods (a "session") holds
every member section, and every lane's teacher and room, at the same time.
Sessions have their own table rather than rows in timetable_entries, which
holds one subject and teacher per section and period: an elective's period
is several teachers in one section's period. The solver, a hand-made
week's clash check, the timetable views, teacher load and the substitute's
busy check read them.

Also here, a section split or merged mid-year (SCH-8): a split moves some of
a section's students to a new section of the class, which gets the same
allocations, bell and groups so the generator can give it a week; a merge
moves every student of one section into another, and takes the emptied
section's week and allocations away with it.

CurriculumStore mixes this class in; errors follow SchoolModelMixin's:
ValueError is the route's 422, KeyError its 404.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

TEACHING_GROUPS_SCHEMA = """
-- SCH-2/SCH-8 (audit N-3-20): an elective group or a combined class across
-- sections. section_ids_json: the member sections, all of one class.
-- lanes_json: [{id, subjectId, teacherId, roomId, studentIds}] -- one lane
-- for a combined class, two or more parallel ones for an elective.
CREATE TABLE IF NOT EXISTS teaching_groups (
  id               TEXT PRIMARY KEY,
  school_id        TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  kind             TEXT NOT NULL,
  name             TEXT NOT NULL,
  periods_per_week INTEGER NOT NULL,
  section_ids_json TEXT NOT NULL,
  lanes_json       TEXT NOT NULL,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tgroup_year ON teaching_groups(academic_year_id);

-- A group's periods in the week: every member section and every lane's
-- teacher and room are taken then (the period numbers of the members' bell).
CREATE TABLE IF NOT EXISTS group_sessions (
  group_id         TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  day_of_week      INTEGER NOT NULL,
  period           INTEGER NOT NULL,
  PRIMARY KEY (group_id, day_of_week, period)
);
CREATE INDEX IF NOT EXISTS idx_gsession_year ON group_sessions(academic_year_id);
"""

GROUP_KINDS = ("elective", "combined")
GROUP_NAME_MAX = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GroupLane:
    id: str
    subject_id: str
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    # Which of the members' students take this lane (an elective's choice);
    # empty until the school records it.
    student_ids: list[str] = field(default_factory=list)


@dataclass
class TeachingGroup:
    id: str
    school_id: str
    academic_year_id: str
    kind: str                       # elective | combined
    name: str
    periods_per_week: int
    section_ids: list[str]
    lanes: list[GroupLane]
    created_at: str = ""

    @property
    def teacher_ids(self) -> list[str]:
        return [lane.teacher_id for lane in self.lanes if lane.teacher_id]

    @property
    def room_ids(self) -> list[str]:
        return [lane.room_id for lane in self.lanes if lane.room_id]

    def schedule_shape(self) -> tuple:
        """What placing the group depends on: a change to any of it means its
        periods must be placed again."""
        return (tuple(self.section_ids), self.periods_per_week,
                tuple((lane.teacher_id, lane.room_id) for lane in self.lanes))


@dataclass
class GroupSession:
    group_id: str
    day_of_week: int
    period: int


class TeachingGroupsMixin:
    """CurriculumStore's methods for teaching groups and for splitting and
    merging sections."""

    # ---------------- reads ----------------

    def _group_from_row(self, r: dict) -> TeachingGroup:
        lanes = [GroupLane(id=lane["id"], subject_id=lane["subjectId"], teacher_id=lane.get("teacherId"),
                           room_id=lane.get("roomId"), student_ids=list(lane.get("studentIds") or []))
                 for lane in json.loads(r["lanes_json"])]
        return TeachingGroup(id=r["id"], school_id=r["school_id"], academic_year_id=r["academic_year_id"],
                             kind=r["kind"], name=r["name"], periods_per_week=r["periods_per_week"],
                             section_ids=json.loads(r["section_ids_json"]), lanes=lanes,
                             created_at=r["created_at"])

    def teaching_groups_for_year(self, academic_year_id: str) -> list[TeachingGroup]:
        return [self._group_from_row(r) for r in self._fetchall(
            "SELECT * FROM teaching_groups WHERE academic_year_id=? ORDER BY name COLLATE NOCASE, id",
            (academic_year_id,))]

    def get_teaching_group(self, group_id: str) -> Optional[TeachingGroup]:
        r = self._fetchone("SELECT * FROM teaching_groups WHERE id=?", (group_id,))
        return self._group_from_row(r) if r else None

    def groups_for_section(self, section_id: str) -> list[TeachingGroup]:
        section = self.get_section(section_id)
        if section is None:
            return []
        return [g for g in self.teaching_groups_for_year(section.academic_year_id) if section_id in g.section_ids]

    def group_sessions_for_year(self, academic_year_id: str) -> list[GroupSession]:
        return [GroupSession(group_id=r["group_id"], day_of_week=r["day_of_week"], period=r["period"])
                for r in self._fetchall("SELECT group_id, day_of_week, period FROM group_sessions "
                                        "WHERE academic_year_id=? ORDER BY day_of_week, period, group_id",
                                        (academic_year_id,))]

    def group_periods_for_year(self, academic_year_id: str) -> list[tuple[TeachingGroup, int, int]]:
        """Every placed group period: (group, day, period)."""
        groups = {g.id: g for g in self.teaching_groups_for_year(academic_year_id)}
        return [(groups[s.group_id], s.day_of_week, s.period)
                for s in self.group_sessions_for_year(academic_year_id) if s.group_id in groups]

    def group_periods_for_section(self, section_id: str) -> list[tuple[TeachingGroup, int, int]]:
        section = self.get_section(section_id)
        if section is None:
            return []
        return [(g, d, p) for g, d, p in self.group_periods_for_year(section.academic_year_id)
                if section_id in g.section_ids]

    def group_periods_for_teacher(self, teacher_id: str,
                                  academic_year_id: str) -> list[tuple[TeachingGroup, GroupLane, int, int]]:
        """The lanes a teacher takes, at each of their group's periods."""
        return [(g, lane, d, p) for g, d, p in self.group_periods_for_year(academic_year_id)
                for lane in g.lanes if lane.teacher_id == teacher_id]

    # ---------------- writes ----------------

    def save_teaching_group(self, *, academic_year_id: str, kind: str, name: str, periods_per_week: int,
                            section_ids: list[str], lanes: list[dict[str, Any]],
                            group_id: Optional[str] = None) -> tuple[Optional[TeachingGroup], TeachingGroup]:
        """Create a group, or replace one's definition (group_id). Returns
        (before, after). Changing its sections, periods a week or a lane's
        teacher or room takes its placed periods away: the next generation
        places the group again. Every check runs before the first write."""
        from .school_model import MAX_PERIODS_PER_WEEK
        if kind not in GROUP_KINDS:
            raise ValueError("a group is an elective (parallel lanes) or a combined class")
        name = " ".join((name or "").split())
        if not name or len(name) > GROUP_NAME_MAX:
            raise ValueError(f"a group needs a name of up to {GROUP_NAME_MAX} characters")
        if not 1 <= periods_per_week <= MAX_PERIODS_PER_WEEK:
            raise ValueError(f"periods per week is 1 to {MAX_PERIODS_PER_WEEK}")
        ids = list(dict.fromkeys(section_ids))
        with self._conn_lock:
            year = self.get_academic_year(academic_year_id)
            if year is None:
                raise KeyError(academic_year_id)
            before = None
            if group_id is not None:
                before = self.get_teaching_group(group_id)
                if before is None or before.academic_year_id != academic_year_id:
                    raise KeyError(group_id)
            sections = []
            for sid in ids:
                section = self.get_section(sid)
                if section is None:
                    raise KeyError(sid)
                if section.academic_year_id != academic_year_id:
                    raise ValueError("a group's sections are this year's")
                sections.append(section)
            if not sections:
                raise ValueError("choose the sections the group takes students from")
            if len({s.grade_id for s in sections}) > 1:
                raise ValueError("a group's sections are all of one class: "
                                 + ", ".join(self._section_label(s) for s in sections) + " are not")
            if kind == "combined" and len(sections) < 2:
                raise ValueError("a combined class joins two sections or more")
            if kind == "combined" and len(lanes) != 1:
                raise ValueError("a combined class has one subject, one teacher and one room")
            if kind == "elective" and len(lanes) < 2:
                raise ValueError("an elective group has two parallel lanes or more (one per subject choice)")
            grade_id = sections[0].grade_id
            enrolled = {e.student_id for s in sections for e in self.enrollments_for_section(s.id)}
            teachers: set[str] = set()
            rooms: set[str] = set()
            taken: set[str] = set()
            clean: list[GroupLane] = []
            old_lanes = {lane.id: lane for lane in (before.lanes if before else [])}
            for i, raw in enumerate(lanes, start=1):
                subject = self.get_subject(raw.get("subject_id") or "")
                if subject is None:
                    raise KeyError(raw.get("subject_id"))
                if subject.grade_id != grade_id:
                    raise ValueError(f"lane {i}: {subject.name} is another class's subject")
                teacher = raw.get("teacher_id")
                if teacher and teacher in teachers:
                    raise ValueError(f"lane {i}: one teacher cannot take two lanes taught at the same time")
                room = raw.get("room_id")
                if room is not None:
                    found = self.get_room(room)
                    if found is None or found.school_id != year.school_id:
                        raise ValueError(f"lane {i}: that room is not this school's")
                    if room in rooms:
                        raise ValueError(f"lane {i}: two lanes taught at the same time cannot share a room")
                students = list(dict.fromkeys(raw.get("student_ids") or []))
                if kind == "combined" and students:
                    raise ValueError("a combined class is every student of its sections; it has no student list")
                strangers = [s for s in students if s not in enrolled]
                if strangers:
                    raise ValueError(f"lane {i}: {len(strangers)} student(s) are not in the group's sections")
                twice = [s for s in students if s in taken]
                if twice:
                    raise ValueError(f"lane {i}: {len(twice)} student(s) are already in another lane")
                for s in sections:
                    if self.allocation_for(s.id, subject.id) is not None:
                        raise ValueError(f"{self._section_label(s)} is already taught {subject.name} on its own: "
                                         "remove that allocation first, or leave the subject out of the group")
                teachers.update([teacher] if teacher else [])
                rooms.update([room] if room else [])
                taken.update(students)
                lane_id = raw.get("id") if raw.get("id") in old_lanes else f"lane_{i}_{subject.id}"
                clean.append(GroupLane(id=lane_id, subject_id=subject.id, teacher_id=teacher, room_id=room,
                                       student_ids=students))
            for other in self.teaching_groups_for_year(academic_year_id):
                if other.id == group_id or not set(other.section_ids) & set(ids):
                    continue
                shared = {lane.subject_id for lane in other.lanes} & {lane.subject_id for lane in clean}
                if shared:
                    subject = self.get_subject(next(iter(shared)))
                    raise ValueError(f"{subject.name if subject else 'a subject'} is already taught to these "
                                     f"sections in the group {other.name}")
            from .store import new_id
            after = TeachingGroup(id=before.id if before else new_id("tgroup"), school_id=year.school_id,
                                  academic_year_id=academic_year_id, kind=kind, name=name,
                                  periods_per_week=periods_per_week, section_ids=ids, lanes=clean,
                                  created_at=before.created_at if before else _now())
            lanes_json = json.dumps([{"id": lane.id, "subjectId": lane.subject_id, "teacherId": lane.teacher_id,
                                      "roomId": lane.room_id, "studentIds": lane.student_ids} for lane in clean])
            if before is None:
                self._exec("INSERT INTO teaching_groups (id, school_id, academic_year_id, kind, name, "
                           "periods_per_week, section_ids_json, lanes_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                           (after.id, after.school_id, academic_year_id, kind, name, periods_per_week,
                            json.dumps(ids), lanes_json, after.created_at))
            else:
                self._exec("UPDATE teaching_groups SET kind=?, name=?, periods_per_week=?, section_ids_json=?, "
                           "lanes_json=? WHERE id=?",
                           (kind, name, periods_per_week, json.dumps(ids), lanes_json, after.id))
                if before.kind != kind or before.schedule_shape() != after.schedule_shape():
                    self._exec("DELETE FROM group_sessions WHERE group_id=?", (after.id,))
        self._commit()
        return before, after

    def delete_teaching_group(self, group_id: str) -> TeachingGroup:
        with self._conn_lock:
            g = self.get_teaching_group(group_id)
            if g is None:
                raise KeyError(group_id)
            self._exec("DELETE FROM group_sessions WHERE group_id=?", (group_id,))
            self._exec("DELETE FROM teaching_groups WHERE id=?", (group_id,))
        self._commit()
        return g

    def _write_group(self, g: TeachingGroup) -> None:
        """Caller holds _conn_lock and commits."""
        self._exec("UPDATE teaching_groups SET section_ids_json=?, lanes_json=? WHERE id=?",
                   (json.dumps(g.section_ids),
                    json.dumps([{"id": lane.id, "subjectId": lane.subject_id, "teacherId": lane.teacher_id,
                                 "roomId": lane.room_id, "studentIds": lane.student_ids} for lane in g.lanes]),
                    g.id))

    def _replace_group_sessions(self, academic_year_id: str, group_ids, sessions) -> None:
        """The solver's periods for `group_ids` (caller holds _conn_lock and
        commits)."""
        for gid in group_ids:
            self._exec("DELETE FROM group_sessions WHERE group_id=?", (gid,))
        for s in sessions:
            if s.group_id in group_ids:
                self._exec("INSERT INTO group_sessions (group_id, academic_year_id, day_of_week, period) "
                           "VALUES (?,?,?,?)", (s.group_id, academic_year_id, s.day_of_week, s.period))

    # ---------------- splitting and merging sections (SCH-8) ----------------

    def split_section(self, section_id: str, *, name: str, student_ids: list[str]) -> dict[str, Any]:
        """Move `student_ids` of a section into a new section of its class,
        mid-year. The new section gets the old one's bell, a copy of each of
        its allocations (subject, teacher, periods, co-teacher, room kind,
        doubles) and its groups, so generating its week is all that is left;
        the old section keeps its week. Returns {"section", "moved",
        "allocations", "groups"}."""
        name = self._clean_section_name(name)
        students = list(dict.fromkeys(student_ids))
        if not students:
            raise ValueError("choose the students who move to the new section")
        with self._conn_lock:
            source = self.get_section(section_id)
            if source is None:
                raise KeyError(section_id)
            self._check_section_name_free(source.grade_id, name)
            enrolled = {e.student_id for e in self.enrollments_for_section(section_id)}
            strangers = [s for s in students if s not in enrolled]
            if strangers:
                raise ValueError(f"{len(strangers)} of those students are not in {self._section_label(source)}")
            if len(students) == len(enrolled):
                raise ValueError(f"leave some students in {self._section_label(source)}: moving all of them "
                                 "is a rename")
            new = self._insert_section(school_id=source.school_id, academic_year_id=source.academic_year_id,
                                       grade_id=source.grade_id, name=name)
            if getattr(source, "bell_schedule_id", None):
                self._exec("UPDATE sections SET bell_schedule_id=? WHERE id=?", (source.bell_schedule_id, new.id))
            for sid in students:
                self._exec("UPDATE student_enrollments SET section_id=? WHERE student_id=? AND section_id=?",
                           (new.id, sid, section_id))
            from .store import new_id
            allocations = [a for a in self.allocations_for_year(source.academic_year_id)
                           if a.section_id == section_id]
            for a in allocations:
                self._exec(
                    "INSERT INTO teaching_allocations (id, school_id, academic_year_id, section_id, subject_id, "
                    "teacher_id, periods_per_week, created_at, co_teacher_id, room_kind, double_periods) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (new_id("alloc"), a.school_id, a.academic_year_id, new.id, a.subject_id, a.teacher_id,
                     a.periods_per_week, _now(), a.co_teacher_id, a.room_kind, a.double_periods))
            groups = self.groups_for_section(section_id)
            for g in groups:
                g.section_ids.append(new.id)
                self._write_group(g)
        self._commit()
        return {"section": self.get_section(new.id), "moved": len(students), "allocations": len(allocations),
                "groups": [g.id for g in groups]}

    def merge_section(self, section_id: str, into_section_id: str) -> dict[str, Any]:
        """Move every student of a section into another section of its class,
        mid-year, and remove the emptied section with its week and its
        allocations: its students now keep the other section's week. Refused
        while the section is in a group the other is not (its students would
        lose those periods). Returns {"into", "moved", "periods",
        "allocations"}."""
        with self._conn_lock:
            gone, into = self.get_section(section_id), self.get_section(into_section_id)
            if gone is None:
                raise KeyError(section_id)
            if into is None:
                raise KeyError(into_section_id)
            if gone.id == into.id:
                raise ValueError("choose another section to merge into")
            if gone.grade_id != into.grade_id:
                raise ValueError("merge sections of one class: "
                                 f"{self._section_label(gone)} and {self._section_label(into)} are not")
            groups = self.groups_for_section(section_id)
            for g in groups:
                if into.id not in g.section_ids:
                    raise ValueError(f"{self._section_label(gone)} is in {g.name} and {self._section_label(into)} "
                                     f"is not: add {self._section_label(into)} to the group, or take "
                                     f"{self._section_label(gone)} out of it, first")
            moved = len(self.enrollments_for_section(section_id))
            periods = len(self._entries_where("section_id=?", (section_id,)))
            allocations = [a for a in self.allocations_for_year(gone.academic_year_id) if a.section_id == section_id]
            self._exec("UPDATE student_enrollments SET section_id=?, grade_id=? WHERE section_id=?",
                       (into.id, into.grade_id, section_id))
            self._exec("DELETE FROM timetable_entries WHERE section_id=?", (section_id,))
            self._exec("DELETE FROM teaching_allocations WHERE section_id=?", (section_id,))
            self._exec("DELETE FROM section_plan_cadences WHERE section_id=?", (section_id,))
            for g in groups:
                g.section_ids = [s for s in g.section_ids if s != section_id]
                self._write_group(g)
            self._exec("DELETE FROM sections WHERE id=?", (section_id,))
        self._commit()
        return {"into": into, "moved": moved, "periods": periods, "allocations": len(allocations)}
