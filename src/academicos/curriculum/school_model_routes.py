"""HTTP routes for the school model (M1.2-M1.3, curriculum/school_model.py):
teaching allocations, rooms, bell schedules and each section's timetable.

Reads are staff-only (a timetable names teachers); `/my-timetable` is any
signed-in user's own week. Writes are the principal's, school-scoped like every
other curriculum write, and each is written to the audit log with what it was
before (REQUIREMENTS ROLE-3). Teacher-level load is the principal's (ADM-5).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user, require_admin, require_principal, require_staff
from ..assessment.users import User
from . import routes as cr
from .schemas import Camel
from .school_model import InUse, StalePreview, TimetableClash

router = APIRouter(prefix="/api/v1/curriculum")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------- request / response shapes ----------------

class AllocationRequest(_Req):
    """coTeacherId, roomKind and doublePeriods left out keep what the cell
    has; null clears the co-teacher or the room kind (SCH-2, SCH-8)."""
    teacher_id: Optional[str] = None
    periods_per_week: int = Field(ge=1, le=60)
    co_teacher_id: Optional[str] = None
    room_kind: Optional[str] = None
    double_periods: int = Field(default=0, ge=0, le=30)


class AllocationResponse(Camel):
    id: str
    academic_year_id: str
    section_id: str
    subject_id: str
    teacher_id: Optional[str] = None
    periods_per_week: int
    co_teacher_id: Optional[str] = None
    room_kind: Optional[str] = None
    double_periods: int = 0


class TeacherLoadResponse(Camel):
    teacher_id: str
    allocated_per_week: int
    section_subjects: int
    timetabled_per_week: int
    max_in_one_day: int


class RoomRequest(_Req):
    name: str = Field(min_length=1, max_length=40)
    kind: str = "classroom"
    capacity: Optional[int] = Field(default=None, ge=1, le=1000)


class RoomUpdateRequest(_Req):
    name: Optional[str] = Field(default=None, min_length=1, max_length=40)
    kind: Optional[str] = None
    capacity: Optional[int] = Field(default=None, ge=1, le=1000)


class RoomResponse(Camel):
    id: str
    name: str
    kind: str
    capacity: Optional[int] = None


class BellSlotBody(_Req):
    start: str
    end: str
    kind: str = "teaching"


class BellSlotResponse(Camel):
    start: str
    end: str
    kind: str
    period: Optional[int] = None


class BellScheduleRequest(_Req):
    name: str = Field(min_length=1, max_length=40)
    days: Optional[list[int]] = None
    slots: list[BellSlotBody]


class BellScheduleUpdateRequest(_Req):
    name: Optional[str] = Field(default=None, min_length=1, max_length=40)
    days: Optional[list[int]] = None
    slots: Optional[list[BellSlotBody]] = None
    make_default: bool = False


class BellScheduleResponse(Camel):
    id: str
    academic_year_id: str
    name: str
    days: list[int]
    slots: list[BellSlotResponse]
    is_default: bool
    teaching_periods: int


class TimetableEntryBody(_Req):
    day_of_week: int = Field(ge=0, le=6)
    period: int = Field(ge=1, le=20)
    subject_id: str = Field(min_length=1)
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    locked: bool = False      # SCH-3: the solver keeps a locked period where it is


class SectionTimetableRequest(_Req):
    entries: list[TimetableEntryBody]


class SectionBellRequest(_Req):
    bell_schedule_id: Optional[str] = None


class TimetableEntryResponse(Camel):
    id: str
    section_id: str
    day_of_week: int
    period: int
    subject_id: str
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    locked: bool = False
    co_teacher_id: Optional[str] = None      # SCH-8: in the room too
    # Names, so a week grid reads "10-B Science" without more calls.
    subject_name: Optional[str] = None
    section_name: Optional[str] = None
    teacher_name: Optional[str] = None
    co_teacher_name: Optional[str] = None
    room_name: Optional[str] = None
    # The period's times on its OWN section's bell (N-3-15): a teacher's
    # week mixes sections on a junior and a main bell.
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    # A period of an elective group or a combined class (N-3-20): it is the
    # group's, not one section's; sectionName then names all its sections.
    group_id: Optional[str] = None
    group_name: Optional[str] = None


class TimetableGapResponse(Camel):
    subject_id: str
    allocated: int
    timetabled: int


class GroupLaneResponse(Camel):
    """One lane of a group: an elective's subject choice, or a combined
    class's one subject, with its teacher and room."""
    id: str
    subject_id: str
    subject_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    student_ids: list[str] = []


class GroupPeriodResponse(Camel):
    """A period a section spends in an elective group or a combined class
    (N-3-20), with every lane taught then."""
    group_id: str
    group_name: str
    kind: str
    day_of_week: int
    period: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    lanes: list[GroupLaneResponse] = []


class SectionTimetableResponse(Camel):
    section_id: str
    bell_schedule_id: Optional[str] = None
    entries: list[TimetableEntryResponse]
    gaps: list[TimetableGapResponse]
    # The section's periods in its groups: the group's to change, not the
    # section's own week (a PUT of the week cannot use them).
    group_periods: list[GroupPeriodResponse] = []


# ---------------- helpers ----------------

def _audit(action: str, principal: User, details: dict[str, Any]) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(
        action, actor=principal.id, details={"schoolId": principal.school_id, **details})


def _staff_member(user_id: Optional[str], principal: User) -> None:
    """A teacher or the principal of the principal's school, or no one."""
    if user_id is None:
        return
    from ..assessment.auth_routes import STAFF_ROLES
    target = cr._require_users().get(user_id)
    if target is None:
        raise HTTPException(404, "teacher not found")
    if target.school_id != principal.school_id:
        raise HTTPException(403, "that user belongs to a different school")
    if target.role not in STAFF_ROLES:
        raise HTTPException(422, "only a teacher or the principal can teach a subject")


def _alloc(a) -> AllocationResponse:
    return AllocationResponse(id=a.id, academic_year_id=a.academic_year_id, section_id=a.section_id,
                              subject_id=a.subject_id, teacher_id=a.teacher_id,
                              periods_per_week=a.periods_per_week, co_teacher_id=a.co_teacher_id,
                              room_kind=a.room_kind, double_periods=a.double_periods)


def _room(r) -> RoomResponse:
    return RoomResponse(id=r.id, name=r.name, kind=r.kind, capacity=r.capacity)


def _bell(b) -> BellScheduleResponse:
    return BellScheduleResponse(
        id=b.id, academic_year_id=b.academic_year_id, name=b.name, days=b.days,
        slots=[BellSlotResponse(start=s.start, end=s.end, kind=s.kind, period=s.period) for s in b.slots],
        is_default=b.is_default, teaching_periods=b.teaching_periods)


def _entry(e, when: Optional[tuple[str, str]] = None) -> TimetableEntryResponse:
    store = cr._require()
    subject = store.get_subject(e.subject_id)
    section = store.get_section(e.section_id)
    users = cr._require_users()
    teacher = users.get(e.teacher_id) if e.teacher_id else None
    co = users.get(e.co_teacher_id) if e.co_teacher_id else None
    room = store.get_room(e.room_id) if e.room_id else None
    return TimetableEntryResponse(id=e.id, section_id=e.section_id, day_of_week=e.day_of_week,
                                  period=e.period, subject_id=e.subject_id, teacher_id=e.teacher_id,
                                  room_id=e.room_id, locked=bool(e.locked), co_teacher_id=e.co_teacher_id,
                                  subject_name=subject.name if subject else None,
                                  section_name=store._section_label(section) if section else None,
                                  teacher_name=teacher.name if teacher else None,
                                  co_teacher_name=co.name if co else None,
                                  room_name=room.name if room else None,
                                  start_time=when[0] if when else None, end_time=when[1] if when else None)


def _section_bells(store, section_ids) -> dict[str, Any]:
    """Each section's own bell (its schedule, else the year's default), read
    once per section."""
    out: dict[str, Any] = {}
    for sid in section_ids:
        if sid not in out:
            section = store.get_section(sid)
            out[sid] = store.bell_for_section(section) if section is not None else None
    return out


def _entries(entries: list) -> list[TimetableEntryResponse]:
    """Entries with each period's times from its own section's bell (N-3-15)."""
    bells = _section_bells(cr._require(), [e.section_id for e in entries])
    times = {sid: {s.period: (s.start, s.end) for s in b.slots if s.kind == "teaching"} if b else {}
             for sid, b in bells.items()}
    return [_entry(e, times[e.section_id].get(e.period)) for e in entries]


def _lane(lane) -> GroupLaneResponse:
    store, users = cr._require(), cr._require_users()
    subject = store.get_subject(lane.subject_id)
    teacher = users.get(lane.teacher_id) if lane.teacher_id else None
    room = store.get_room(lane.room_id) if lane.room_id else None
    return GroupLaneResponse(id=lane.id, subject_id=lane.subject_id, subject_name=subject.name if subject else None,
                             teacher_id=lane.teacher_id, teacher_name=teacher.name if teacher else None,
                             room_id=lane.room_id, room_name=room.name if room else None,
                             student_ids=list(lane.student_ids))


def _group_label(store, g) -> str:
    """A group's sections as one label: "9-A, 9-B"."""
    return ", ".join(store._section_label(s) for s in (store.get_section(sid) for sid in g.section_ids) if s)


def _period_times(store, section_id: str) -> dict[int, tuple[str, str]]:
    bell = _section_bells(store, [section_id])[section_id]
    return {s.period: (s.start, s.end) for s in bell.slots if s.kind == "teaching"} if bell else {}


def _group_periods(store, section_id: str) -> list[GroupPeriodResponse]:
    times = _period_times(store, section_id)
    return [GroupPeriodResponse(group_id=g.id, group_name=g.name, kind=g.kind, day_of_week=d, period=p,
                                start_time=times.get(p, (None, None))[0], end_time=times.get(p, (None, None))[1],
                                lanes=[_lane(lane) for lane in g.lanes])
            for g, d, p in store.group_periods_for_section(section_id)]


def _group_entry(store, g, lane, d: int, p: int, *, subject_name: Optional[str] = None) -> TimetableEntryResponse:
    """A group period as a row of someone's own week (N-3-20): a lane's
    teacher sees their lane; a student their lane, or the whole group."""
    users = cr._require_users()
    times = _period_times(store, g.section_ids[0]) if g.section_ids else {}
    subject = store.get_subject(lane.subject_id) if lane else None
    teacher = users.get(lane.teacher_id) if lane and lane.teacher_id else None
    room = store.get_room(lane.room_id) if lane and lane.room_id else None
    return TimetableEntryResponse(
        id=f"{g.id}:{d}:{p}", section_id=g.section_ids[0] if g.section_ids else "", day_of_week=d, period=p,
        subject_id=lane.subject_id if lane else g.lanes[0].subject_id,
        teacher_id=lane.teacher_id if lane else None, room_id=lane.room_id if lane else None,
        subject_name=subject_name or (subject.name if subject else None), section_name=_group_label(store, g),
        teacher_name=teacher.name if teacher else None, room_name=room.name if room else None,
        start_time=times.get(p, (None, None))[0], end_time=times.get(p, (None, None))[1],
        group_id=g.id, group_name=g.name)


def _section_week(section) -> SectionTimetableResponse:
    store = cr._require()
    bell = store.bell_for_section(section)
    return SectionTimetableResponse(
        section_id=section.id, bell_schedule_id=bell.id if bell else None,
        entries=_entries(store.timetable_for_section(section.id)),
        gaps=[TimetableGapResponse(subject_id=g["subjectId"], allocated=g["allocated"],
                                   timetabled=g["timetabled"])
              for g in store.timetable_gaps(section.id)],
        group_periods=_group_periods(store, section.id))


def _owned_bell(bell_id: str, current: User):
    b = cr._require().get_bell_schedule(bell_id)
    if b is None:
        raise HTTPException(404, "bell schedule not found")
    if b.school_id != current.school_id:
        raise HTTPException(403, "this bell schedule belongs to a different school")
    return b


def _owned_room(room_id: str, current: User):
    r = cr._require().get_room(room_id)
    if r is None:
        raise HTTPException(404, "room not found")
    if r.school_id != current.school_id:
        raise HTTPException(403, "this room belongs to a different school")
    return r


# ---------------- teaching allocations (M1.2) ----------------

@router.get("/academic-years/{academic_year_id}/allocations", response_model=list[AllocationResponse])
def list_allocations(academic_year_id: str,
                     current: User = Depends(require_staff)) -> list[AllocationResponse]:
    """Every (section, subject) cell of the year: the teacher and periods a week."""
    cr._require_school_owns_academic_year(academic_year_id, current)
    return [_alloc(a) for a in cr._require().allocations_for_year(academic_year_id)]


@router.put("/sections/{section_id}/allocations/{subject_id}", response_model=AllocationResponse)
def set_allocation(section_id: str, subject_id: str, req: AllocationRequest,
                   principal: User = Depends(require_admin("timetable", scoped=True))) -> AllocationResponse:
    """Set one cell of the allocation grid: who teaches this subject to this
    section, and how many periods a week."""
    cr._require_school_owns_section(section_id, principal)
    cr.require_section_in_scope(section_id, principal, "timetable", subject_id=subject_id)
    _staff_member(req.teacher_id, principal)
    sent = req.model_fields_set
    extra: dict[str, Any] = {}
    if "co_teacher_id" in sent:
        _staff_member(req.co_teacher_id, principal)
        extra["co_teacher_id"] = req.co_teacher_id
    if "room_kind" in sent:
        extra["room_kind"] = req.room_kind
    if "double_periods" in sent:
        extra["double_periods"] = req.double_periods
    try:
        before, after = cr._require().set_allocation(section_id=section_id, subject_id=subject_id,
                                                     teacher_id=req.teacher_id,
                                                     periods_per_week=req.periods_per_week, **extra)
    except KeyError:
        raise HTTPException(404, "subject not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    if before is not None and before.teacher_id != after.teacher_id:
        # SCH-8: a teacher joins or leaves mid-year; the cover and the books follow.
        cr._require().follow_teacher_change(section_id, subject_id, before.teacher_id, after.teacher_id,
                                            today=cr._school_today().isoformat())
    # The SCH-2/SCH-8 fields are recorded when either side uses them, so a
    # plain cell's entry reads as it always has.
    special = any(x is not None and (x.co_teacher_id or x.room_kind or x.double_periods)
                  for x in (before, after))

    def fields(a):
        out = {"teacherId": a.teacher_id, "periodsPerWeek": a.periods_per_week}
        if special:
            out.update(coTeacherId=a.co_teacher_id, roomKind=a.room_kind, doublePeriods=a.double_periods)
        return out
    if before is None or fields(before) != fields(after):
        _audit("allocation_set", principal,
               {"sectionId": section_id, "subjectId": subject_id,
                "before": fields(before) if before else None, "after": fields(after)})
    return _alloc(after)


@router.delete("/sections/{section_id}/allocations/{subject_id}")
def delete_allocation(section_id: str, subject_id: str,
                      principal: User = Depends(require_admin("timetable", scoped=True))) -> dict:
    """409 while the timetable still gives the subject periods."""
    cr._require_school_owns_section(section_id, principal)
    cr.require_section_in_scope(section_id, principal, "timetable", subject_id=subject_id)
    try:
        alloc = cr._require().delete_allocation(section_id, subject_id)
    except KeyError:
        raise HTTPException(404, "allocation not found")
    except InUse as e:
        raise HTTPException(409, str(e))
    _audit("allocation_deleted", principal,
           {"sectionId": section_id, "subjectId": subject_id,
            "before": {"teacherId": alloc.teacher_id, "periodsPerWeek": alloc.periods_per_week}})
    return {"ok": True}


@router.get("/academic-years/{academic_year_id}/teacher-load", response_model=list[TeacherLoadResponse])
def teacher_load(academic_year_id: str,
                 principal: User = Depends(require_admin("timetable"))) -> list[TeacherLoadResponse]:
    """Per teacher: periods a week allocated and timetabled, and the most in
    one day. The principal's: teacher-level data (REQUIREMENTS ADM-5)."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    return [TeacherLoadResponse(teacher_id=r["teacherId"], allocated_per_week=r["allocatedPerWeek"],
                                section_subjects=r["sectionSubjects"],
                                timetabled_per_week=r["timetabledPerWeek"],
                                max_in_one_day=r["maxInOneDay"])
            for r in cr._require().teacher_load(academic_year_id)]


# ---------------- rooms (M1.3) ----------------

@router.get("/rooms", response_model=list[RoomResponse])
def list_rooms(current: User = Depends(require_staff)) -> list[RoomResponse]:
    return [_room(r) for r in cr._require().rooms_for_school(current.school_id)]


@router.post("/rooms", response_model=RoomResponse)
def create_room(req: RoomRequest, principal: User = Depends(require_admin("timetable"))) -> RoomResponse:
    try:
        room = cr._require().create_room(school_id=principal.school_id, name=req.name, kind=req.kind,
                                         capacity=req.capacity)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("room_created", principal, {"roomId": room.id, "after": {"name": room.name, "kind": room.kind,
                                                                    "capacity": room.capacity}})
    return _room(room)


@router.patch("/rooms/{room_id}", response_model=RoomResponse)
def update_room(room_id: str, req: RoomUpdateRequest,
                principal: User = Depends(require_admin("timetable"))) -> RoomResponse:
    _owned_room(room_id, principal)
    sent = req.model_fields_set
    try:
        before, after = cr._require().update_room(
            room_id, name=req.name if "name" in sent else None,
            kind=req.kind if "kind" in sent else None,
            capacity=req.capacity if "capacity" in sent else ...)
    except ValueError as e:
        raise HTTPException(422, str(e))
    f = lambda r: {"name": r.name, "kind": r.kind, "capacity": r.capacity}  # noqa: E731
    if f(before) != f(after):
        _audit("room_updated", principal, {"roomId": room_id, "before": f(before), "after": f(after)})
    return _room(after)


@router.delete("/rooms/{room_id}")
def delete_room(room_id: str, principal: User = Depends(require_admin("timetable"))) -> dict:
    """409 while the timetable uses the room."""
    _owned_room(room_id, principal)
    try:
        room = cr._require().delete_room(room_id)
    except InUse as e:
        raise HTTPException(409, str(e))
    _audit("room_deleted", principal, {"roomId": room_id, "before": {"name": room.name, "kind": room.kind}})
    return {"ok": True}


# ---------------- bell schedules (M1.3) ----------------

@router.get("/academic-years/{academic_year_id}/bell-schedules", response_model=list[BellScheduleResponse])
def list_bell_schedules(academic_year_id: str,
                        current: User = Depends(get_current_user)) -> list[BellScheduleResponse]:
    """The year's bells. Any signed-in user of the school: a student's day
    view needs the period times."""
    cr._require_school_owns_academic_year(academic_year_id, current)
    return [_bell(b) for b in cr._require().bell_schedules_for_year(academic_year_id)]


@router.post("/academic-years/{academic_year_id}/bell-schedules", response_model=BellScheduleResponse)
def create_bell_schedule(academic_year_id: str, req: BellScheduleRequest,
                         principal: User = Depends(require_admin("timetable"))) -> BellScheduleResponse:
    """The year's first schedule becomes its default."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    try:
        b = cr._require().create_bell_schedule(academic_year_id=academic_year_id, name=req.name,
                                               days=req.days,
                                               slots=[s.model_dump() for s in req.slots])
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("bell_schedule_created", principal, {"bellScheduleId": b.id, "name": b.name,
                                                 "teachingPeriods": b.teaching_periods})
    return _bell(b)


@router.put("/bell-schedules/{bell_schedule_id}", response_model=BellScheduleResponse)
def update_bell_schedule(bell_schedule_id: str, req: BellScheduleUpdateRequest,
                         principal: User = Depends(require_admin("timetable"))) -> BellScheduleResponse:
    before = _owned_bell(bell_schedule_id, principal)
    try:
        after = cr._require().update_bell_schedule(
            bell_schedule_id, name=req.name, days=req.days,
            slots=[s.model_dump() for s in req.slots] if req.slots is not None else None,
            make_default=req.make_default)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("bell_schedule_updated", principal,
           {"bellScheduleId": bell_schedule_id,
            "before": {"name": before.name, "days": before.days, "teachingPeriods": before.teaching_periods,
                       "isDefault": before.is_default},
            "after": {"name": after.name, "days": after.days, "teachingPeriods": after.teaching_periods,
                      "isDefault": after.is_default}})
    return _bell(after)


@router.delete("/bell-schedules/{bell_schedule_id}")
def delete_bell_schedule(bell_schedule_id: str, principal: User = Depends(require_admin("timetable"))) -> dict:
    _owned_bell(bell_schedule_id, principal)
    try:
        b = cr._require().delete_bell_schedule(bell_schedule_id)
    except InUse as e:
        raise HTTPException(409, str(e))
    _audit("bell_schedule_deleted", principal, {"bellScheduleId": bell_schedule_id, "name": b.name})
    return {"ok": True}


@router.put("/sections/{section_id}/bell-schedule", response_model=SectionTimetableResponse)
def set_section_bell(section_id: str, req: SectionBellRequest,
                     principal: User = Depends(require_admin("timetable", scoped=True))
                     ) -> SectionTimetableResponse:
    """Give a section its own bell (null: the year's default)."""
    section = cr._require_school_owns_section(section_id, principal)
    cr.require_section_in_scope(section_id, principal, "timetable")
    try:
        cr._require().set_section_bell(section_id, req.bell_schedule_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("section_bell_set", principal, {"sectionId": section_id, "before": section.bell_schedule_id,
                                            "after": req.bell_schedule_id})
    return _section_week(cr._require().get_section(section_id))


# ---------------- the timetable (SCH-2) ----------------

@router.get("/academic-years/{academic_year_id}/timetable", response_model=list[TimetableEntryResponse])
def year_timetable(academic_year_id: str, section_id: Optional[str] = Query(default=None, alias="sectionId"),
                   teacher_id: Optional[str] = Query(default=None, alias="teacherId"),
                   current: User = Depends(require_staff)) -> list[TimetableEntryResponse]:
    """The year's timetable, optionally one section's or one teacher's."""
    cr._require_school_owns_academic_year(academic_year_id, current)
    entries = cr._require().timetable_for_year(academic_year_id)
    if section_id:
        entries = [e for e in entries if e.section_id == section_id]
    if teacher_id:
        entries = [e for e in entries if teacher_id in (e.teacher_id, e.co_teacher_id)]
    return _entries(entries)


@router.get("/sections/{section_id}/timetable", response_model=SectionTimetableResponse)
def section_timetable(section_id: str, current: User = Depends(require_staff)) -> SectionTimetableResponse:
    """A section's week, its bell, and per subject the periods allocated
    against the periods placed."""
    return _section_week(cr._require_school_owns_section(section_id, current))


def _tell_timetable_change(principal: User, before: list, after: list, section_ids: set[str]) -> None:
    """N-8-5: a changed week tells the people it changes. Each section whose
    week differs tells its students; each teacher whose own periods in those
    sections differ is told. A week saved unchanged tells no one. Both the
    solver's publish and a hand-made week come through here; only the solver
    used to notify, and only teachers."""
    from ..operations.routes import notify_parents_safely, notify_safely

    def section_week(entries: list, sid: str) -> set:
        return {(e.day_of_week, e.period, e.subject_id, e.teacher_id, e.room_id, e.co_teacher_id)
                for e in entries if e.section_id == sid}

    changed = {sid for sid in section_ids if section_week(before, sid) != section_week(after, sid)}
    if not changed:
        return

    def teacher_week(entries: list, t: str) -> set:
        return {(e.section_id, e.day_of_week, e.period, e.subject_id, e.room_id)
                for e in entries if e.section_id in changed and t in (e.teacher_id, e.co_teacher_id)}

    teachers = sorted({t for e in before + after if e.section_id in changed for t in (e.teacher_id, e.co_teacher_id)
                       if t and teacher_week(before, t) != teacher_week(after, t)})
    store = cr._require()
    students = sorted({en.student_id for sid in changed for en in store.enrollments_for_section(sid)})
    for who in (teachers, students):
        if who:
            notify_safely(school_id=principal.school_id, user_ids=who, kind="timetable_published", params={},
                          link="/my-timetable")
    notify_parents_safely(school_id=principal.school_id, student_ids=students, kind="timetable_published",
                          params={}, exclude=teachers)


def _replan_changed_weeks(principal: User, before: list, after: list, section_ids: set[str]) -> int:
    """N-3-14: a section whose week changed has its plan re-laid on the new
    week. For each section and subject whose periods per weekday differ,
    every still-to-teach lesson from the school's today on moves onto the
    periods the section now has (scheduling.replan_section, the reflow a
    lost period uses); taught, skipped and past lessons stay. Each move is
    on the audit log with the reason. A subject with no period left in the
    new week is left as it is: there is nothing to lay it on until the week
    is filled in. Until now a mid-term change stranded lessons on weekdays
    the section no longer has the subject. Returns the plans that changed."""
    from collections import Counter

    from ..assessment.audit_log import get_audit_log
    from . import scheduling

    store = cr._require()
    today = cr._school_today().isoformat()
    log = get_audit_log(cr._cfg.data_root)
    changed = 0
    for sid in sorted(section_ids):
        old = Counter((e.subject_id, e.day_of_week) for e in before if e.section_id == sid)
        new = Counter((e.subject_id, e.day_of_week) for e in after if e.section_id == sid)
        section = store.get_section(sid)
        if section is None:
            continue
        still_taught = {subject_id for subject_id, _ in new}
        for subject_id in sorted({s for s, _ in (old - new) + (new - old)} & still_taught):
            changed += scheduling.replan_section(
                store, log, academic_year_id=section.academic_year_id, section_id=sid,
                subject_id=subject_id, from_date=today, changed_by=principal.id,
                reason=f"{store._section_label(section)}'s timetable changed")
    return changed


@router.put("/sections/{section_id}/timetable", response_model=SectionTimetableResponse)
def replace_section_timetable(section_id: str, req: SectionTimetableRequest,
                              principal: User = Depends(require_admin("timetable", scoped=True))
                              ) -> SectionTimetableResponse:
    """Replace a section's whole week. Checked before anything is written: a
    teacher or room already busy in another section, a period or day the
    bell does not have, a subject with no allocation or more periods than
    allocated. A 409 lists every clash; the old week is kept. A subject
    whose weekdays changed has its plan re-laid on the new week from today
    (N-3-14)."""
    section = cr._require_school_owns_section(section_id, principal)
    cr.require_section_in_scope(section_id, principal, "timetable")
    for e in req.entries:
        _staff_member(e.teacher_id, principal)
    store = cr._require()
    week_before = store.timetable_for_section(section_id)
    before = len(week_before)
    try:
        store.replace_section_timetable(section_id, [e.model_dump() for e in req.entries])
    except TimetableClash as e:
        raise HTTPException(409, {"message": "the timetable has clashes; nothing was changed",
                                  "clashes": e.clashes})
    except ValueError as e:
        raise HTTPException(422, str(e))
    week_after = store.timetable_for_section(section_id)
    replanned = _replan_changed_weeks(principal, week_before, week_after, {section_id})
    _audit("timetable_replaced", principal, {"sectionId": section_id, "before": before,
                                             "after": len(req.entries), "plansReplanned": replanned})
    _tell_timetable_change(principal, week_before, week_after, {section_id})
    return _section_week(store.get_section(section.id))


class MyTimetableResponse(Camel):
    role: str
    academic_year_id: Optional[str] = None
    section_id: Optional[str] = None
    # A student's class, so the app knows a young learner (SA-7, classes 1-5).
    grade: Optional[int] = None
    bell_schedule: Optional[BellScheduleResponse] = None
    entries: list[TimetableEntryResponse] = []


@router.get("/my-timetable", response_model=MyTimetableResponse)
def my_timetable(current: User = Depends(get_current_user)) -> MyTimetableResponse:
    """The caller's own week: a teacher's periods across sections, a
    student's section's periods. Empty (not an error) until the school has
    a timetable."""
    store = cr._require()
    years = store.academic_years_for_school(current.school_id)
    if not years:
        return MyTimetableResponse(role=current.role)
    year = sorted(years, key=lambda y: y.start_date)[-1]
    if current.role == "student":
        enrollment = store.enrollment_for_student(current.id)
        section = store.get_section(enrollment.section_id) if enrollment and enrollment.section_id else None
        if section is None:
            return MyTimetableResponse(role=current.role, academic_year_id=year.id)
        bell = store.bell_for_section(section)
        grade = store.get_grade(section.grade_id)
        # A group period is the student's lane when the school recorded
        # their choice, else the whole group ("Third language").
        groups = []
        for g, d, p in store.group_periods_for_section(section.id):
            lane = next((lane for lane in g.lanes if current.id in lane.student_ids), None)
            if lane is None and len(g.lanes) == 1:
                lane = g.lanes[0]
            groups.append(_group_entry(store, g, lane, d, p, subject_name=None if lane else g.name))
        return MyTimetableResponse(role=current.role, academic_year_id=section.academic_year_id,
                                   section_id=section.id, bell_schedule=_bell(bell) if bell else None,
                                   entries=_entries(store.timetable_for_section(section.id)) + groups,
                                   grade=grade.number if grade else None)
    # A teacher's periods can be on several bells (N-3-15): each entry has
    # its own section's times, and the bell is the one they all use -- the
    # year's default only when they use more than one, or none yet.
    mine = store.timetable_for_teacher(current.id, year.id)
    lanes = store.group_periods_for_teacher(current.id, year.id)
    used = {b.id: b for b in _section_bells(store, [e.section_id for e in mine]
                                            + [g.section_ids[0] for g, _, _, _ in lanes if g.section_ids]
                                            ).values() if b}
    bell = next(iter(used.values())) if len(used) == 1 else next(
        (b for b in store.bell_schedules_for_year(year.id) if b.is_default), None)
    return MyTimetableResponse(role=current.role, academic_year_id=year.id,
                               bell_schedule=_bell(bell) if bell else None,
                               entries=_entries(mine) + [_group_entry(store, g, lane, d, p)
                                                         for g, lane, d, p in lanes])


# ---------------- generation (SCH-3) ----------------

class SolveRequest(_Req):
    apply: bool = False
    keep_existing: bool = True
    max_per_day: int = Field(default=7, ge=1, le=12)
    max_consecutive: int = Field(default=4, ge=1, le=12)
    time_limit_seconds: int = Field(default=30, ge=1, le=60)
    section_ids: Optional[list[str]] = None


class ProposedEntryResponse(Camel):
    section_id: str
    day_of_week: int
    period: int
    subject_id: str
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    co_teacher_id: Optional[str] = None


class PeriodResponse(Camel):
    """What one period holds, with names to read it by. A group's period
    (N-3-20) carries the group; its subject is the group's first lane's."""
    subject_id: str
    subject_name: Optional[str] = None
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    co_teacher_id: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None


class GroupSessionResponse(Camel):
    """A period of an elective group or a combined class (N-3-20)."""
    group_id: str
    day_of_week: int
    period: int


class PeriodChangeResponse(Camel):
    """One period of one section the generated week changes (N-3-6):
    `before` -> `after`. No `before` is a period added; no `after`, one
    that becomes free."""
    section_id: str
    section_name: Optional[str] = None
    day_of_week: int
    period: int
    before: Optional[PeriodResponse] = None
    after: Optional[PeriodResponse] = None


class SolveResponse(Camel):
    status: str           # solved | infeasible | timeout | nothing_to_solve
    applied: bool
    problems: list[str]
    kept: int
    moved_or_added: int
    removed: int
    seconds: float
    entries: list[ProposedEntryResponse]
    # A proposal (apply: false) is kept under this id: publishing it writes
    # exactly this week, not a second solve (N-3-6).
    preview_id: Optional[str] = None
    # Every period that differs from the current week, section by section.
    changes: list[PeriodChangeResponse] = []
    # The proposed periods of each group placed (N-3-20).
    group_sessions: list[GroupSessionResponse] = []


def _changes(before: list, after: list, section_ids: set[str], groups_before: list = (),
             groups_after: list = ()) -> list[PeriodChangeResponse]:
    """The per-period diff of a week: each (section, day, period) whose
    subject, teacher, room or co-teacher differs -- or that moves into or out
    of a group (N-3-20) -- in section, day and period order. groups_*:
    (group, day, period)."""
    store, users = cr._require(), cr._require_users()
    names: dict[tuple[str, str], Optional[str]] = {}

    def name(kind: str, key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        if (kind, key) not in names:
            if kind == "subject":
                row = store.get_subject(key)
            elif kind == "room":
                row = store.get_room(key)
            else:
                row = users.get(key)
            names[(kind, key)] = row.name if row else None
        return names[(kind, key)]

    def holds(e) -> Optional[PeriodResponse]:
        if e is None:
            return None
        if isinstance(e, tuple):            # a group's period
            g = e[0]
            return PeriodResponse(subject_id=g.lanes[0].subject_id, subject_name=g.name, group_id=g.id,
                                  group_name=g.name)
        return PeriodResponse(subject_id=e.subject_id, subject_name=name("subject", e.subject_id),
                              teacher_id=e.teacher_id, teacher_name=name("user", e.teacher_id),
                              room_id=e.room_id, room_name=name("room", e.room_id),
                              co_teacher_id=getattr(e, "co_teacher_id", None))

    def keyed(entries: list, groups: list) -> dict:
        out: dict = {(e.section_id, e.day_of_week, e.period): e for e in entries if e.section_id in section_ids}
        for g, d, p in groups:
            for sid in g.section_ids:
                if sid in section_ids:
                    out[(sid, d, p)] = (g,)
        return out

    def same(a, b) -> bool:
        look = lambda e: ("group", e[0].id) if isinstance(e, tuple) else (  # noqa: E731
            e.subject_id, e.teacher_id, e.room_id, getattr(e, "co_teacher_id", None))
        return a is not None and b is not None and look(a) == look(b)

    old, new = keyed(before, groups_before), keyed(after, groups_after)
    labels = {s.id: store._section_label(s) for s in (store.get_section(sid) for sid in section_ids) if s}
    out = []
    for key in sorted(set(old) | set(new), key=lambda k: (labels.get(k[0], k[0]), k[1], k[2])):
        if not same(old.get(key), new.get(key)):
            out.append(PeriodChangeResponse(section_id=key[0], section_name=labels.get(key[0]),
                                            day_of_week=key[1], period=key[2],
                                            before=holds(old.get(key)), after=holds(new.get(key))))
    return out


def _groups_placed(store, academic_year_id: str, group_ids: set[str]) -> list:
    """The (group, day, period) now placed for `group_ids`."""
    return [(g, d, p) for g, d, p in store.group_periods_for_year(academic_year_id) if g.id in group_ids]


def _groups_proposed(store, academic_year_id: str, sessions: list) -> list:
    groups = {g.id: g for g in store.teaching_groups_for_year(academic_year_id)}
    return [(groups[s.group_id], s.day_of_week, s.period) for s in sessions if s.group_id in groups]


def _sessions(sessions: list) -> list[GroupSessionResponse]:
    return [GroupSessionResponse(group_id=s.group_id, day_of_week=s.day_of_week, period=s.period) for s in sessions]


def _proposed(entries: list) -> list[ProposedEntryResponse]:
    return [ProposedEntryResponse(section_id=e.section_id, day_of_week=e.day_of_week, period=e.period,
                                  subject_id=e.subject_id, teacher_id=e.teacher_id, room_id=e.room_id,
                                  co_teacher_id=e.co_teacher_id)
            for e in entries]


def _published(principal: User, academic_year_id: str, week_before: list, solving: set[str],
               summary: dict[str, Any], preview_id: Optional[str] = None) -> None:
    """The re-laid plans, the audit entry and the notices for a week just
    written -- by a one-step solve or by publishing a preview (N-3-6). A
    subject whose weekdays changed has its future lessons moved onto the new
    week (N-3-14)."""
    store = cr._require()
    week_after = [e for e in store.timetable_for_year(academic_year_id) if e.section_id in solving]
    replanned = _replan_changed_weeks(principal, week_before, week_after, solving)
    _audit("timetable_solved", principal,
           {"academicYearId": academic_year_id, "sections": len(solving), "kept": summary["kept"],
            "movedOrAdded": summary["movedOrAdded"], "removed": summary["removed"],
            "plansReplanned": replanned,
            **({"previewId": preview_id} if preview_id else {})})
    _tell_timetable_change(principal, week_before, week_after, solving)


@router.post("/academic-years/{academic_year_id}/timetable/solve", response_model=SolveResponse)
def solve_timetable(academic_year_id: str, req: SolveRequest,
                    principal: User = Depends(require_admin("timetable"))) -> SolveResponse:
    """Generate a clash-free week for the whole school (or `sectionIds`) from
    the allocations and bells: no teacher, room or section in two places,
    a teacher's maximum a day and in a row, subjects spread across the week,
    unavailable periods free, locked periods kept, and -- with keepExisting
    -- the fewest changes to the current week. Every changed period is
    listed in `changes`.

    `apply: false` writes nothing: it keeps the proposal under `previewId`,
    which `POST .../timetable/previews/{previewId}/publish` writes exactly
    (N-3-6). `apply: true` generates and publishes in one step: the week
    written is the one returned."""
    from .timetable_solver import SolveOptions, solve
    cr._require_school_owns_academic_year(academic_year_id, principal)
    store = cr._require()
    section_ids = None
    if req.section_ids:
        for sid in req.section_ids:
            cr._require_school_owns_section(sid, principal)
        section_ids = set(req.section_ids)
    # Fingerprinted before the solve reads anything: a change made while it
    # runs makes the preview stale, never silently part of it.
    inputs = store.timetable_inputs_hash(academic_year_id)
    result = solve(store, academic_year_id, SolveOptions(
        max_per_day=req.max_per_day, max_consecutive=req.max_consecutive,
        keep_existing=req.keep_existing, time_limit_seconds=req.time_limit_seconds,
        section_ids=section_ids))
    solving = section_ids or {s.id for s in store.sections_for_year(academic_year_id)}
    summary = {"kept": result.kept, "movedOrAdded": result.moved_or_added, "removed": result.removed,
               "seconds": round(result.seconds, 2)}
    week_before = [e for e in store.timetable_for_year(academic_year_id) if e.section_id in solving]
    groups_before = _groups_placed(store, academic_year_id, result.group_ids)
    groups_after = _groups_proposed(store, academic_year_id, result.group_sessions)
    applied, preview_id = False, None
    if result.status == "solved" and req.apply:
        store.apply_solved_timetable(academic_year_id, result.entries, solving, result.group_sessions,
                                     result.group_ids)
        applied = True
        _published(principal, academic_year_id, week_before, solving, summary)
    elif result.status == "solved":
        preview_id = store.save_timetable_preview(academic_year_id=academic_year_id, section_ids=solving,
                                                  entries=result.entries, summary=summary, inputs_hash=inputs,
                                                  created_by=principal.id, group_sessions=result.group_sessions,
                                                  group_ids=result.group_ids)
    return SolveResponse(
        status=result.status, applied=applied, problems=result.problems, kept=result.kept,
        moved_or_added=result.moved_or_added, removed=result.removed, seconds=round(result.seconds, 2),
        entries=_proposed(result.entries), preview_id=preview_id, group_sessions=_sessions(result.group_sessions),
        changes=_changes(week_before, result.entries, solving, groups_before, groups_after)
        if result.status == "solved" else [])


@router.post("/academic-years/{academic_year_id}/timetable/previews/{preview_id}/publish",
             response_model=SolveResponse)
def publish_timetable_preview(academic_year_id: str, preview_id: str,
                              principal: User = Depends(require_admin("timetable"))) -> SolveResponse:
    """Publish exactly the week a preview showed -- no second solve, so what
    the principal approved is what the school gets (N-3-6). 404: no such
    preview (a newer one or a publish replaced it). 409: the timetable,
    allocations, bells, rooms or availability changed after the preview was
    made; preview again."""
    from .teaching_groups import GroupSession
    from .timetable_solver import ProposedEntry
    cr._require_school_owns_academic_year(academic_year_id, principal)
    store = cr._require()
    try:
        preview, week_before, groups_before = store.publish_timetable_preview(preview_id, academic_year_id,
                                                                              lambda d: ProposedEntry(**d))
    except KeyError:
        raise HTTPException(404, "that preview is gone: a newer preview or a publish replaced it; preview again")
    except StalePreview:
        raise HTTPException(409, "the timetable, allocations, bells, rooms or availability changed after this "
                                 "preview was made; preview again to see the week that fits them now")
    solving, summary = preview["section_ids"], preview["summary"]
    entries = [ProposedEntry(**d) for d in preview["entries"]]
    sessions = [GroupSession(**s) for s in preview["group_sessions"]]
    _published(principal, academic_year_id, week_before, solving, summary, preview_id=preview_id)
    return SolveResponse(status="solved", applied=True, problems=[], kept=summary["kept"],
                         moved_or_added=summary["movedOrAdded"], removed=summary["removed"],
                         seconds=summary["seconds"], entries=_proposed(entries), preview_id=preview_id,
                         group_sessions=_sessions(sessions),
                         changes=_changes(week_before, entries, solving, groups_before,
                                          _groups_proposed(store, academic_year_id, sessions)))


class UnavailableSlot(_Req):
    day_of_week: int = Field(ge=0, le=6)
    period: int = Field(ge=1, le=20)


class TeacherUnavailabilityRequest(_Req):
    slots: list[UnavailableSlot]


class TeacherUnavailabilityResponse(Camel):
    teacher_id: str
    slots: list[UnavailableSlot]


@router.get("/academic-years/{academic_year_id}/teacher-unavailability",
            response_model=list[TeacherUnavailabilityResponse])
def list_teacher_unavailability(academic_year_id: str,
                                principal: User = Depends(require_admin("timetable"))
                                ) -> list[TeacherUnavailabilityResponse]:
    """Periods each teacher cannot teach (the default bell's numbering).
    Teacher-level data: the principal's (ADM-5)."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    rows = cr._require().teacher_unavailability_for_year(academic_year_id)
    return [TeacherUnavailabilityResponse(
        teacher_id=t, slots=[UnavailableSlot(day_of_week=d, period=p) for d, p in sorted(slots)])
        for t, slots in sorted(rows.items())]


@router.put("/academic-years/{academic_year_id}/teacher-unavailability/{teacher_id}",
            response_model=TeacherUnavailabilityResponse)
def set_teacher_unavailability(academic_year_id: str, teacher_id: str, req: TeacherUnavailabilityRequest,
                               principal: User = Depends(require_admin("timetable"))
                               ) -> TeacherUnavailabilityResponse:
    cr._require_school_owns_academic_year(academic_year_id, principal)
    _staff_member(teacher_id, principal)
    try:
        slots = cr._require().set_teacher_unavailability(
            academic_year_id, teacher_id, [(s.day_of_week, s.period) for s in req.slots])
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("teacher_unavailability_set", principal,
           {"academicYearId": academic_year_id, "teacherId": teacher_id, "slots": len(slots)})
    return TeacherUnavailabilityResponse(
        teacher_id=teacher_id, slots=[UnavailableSlot(day_of_week=d, period=p) for d, p in slots])



# ---------------- rooms out of use (SCH-8) ----------------

class RoomUnavailabilityResponse(Camel):
    room_id: str
    slots: list[UnavailableSlot]


@router.get("/academic-years/{academic_year_id}/room-unavailability",
            response_model=list[RoomUnavailabilityResponse])
def list_room_unavailability(academic_year_id: str,
                             current: User = Depends(require_staff)) -> list[RoomUnavailabilityResponse]:
    """Periods each room cannot be used this year (a lab closed for its
    weekly maintenance, say). The solver books nothing there."""
    cr._require_school_owns_academic_year(academic_year_id, current)
    rows = cr._require().room_unavailability_for_year(academic_year_id)
    return [RoomUnavailabilityResponse(
        room_id=r, slots=[UnavailableSlot(day_of_week=d, period=p) for d, p in sorted(slots)])
        for r, slots in sorted(rows.items())]


@router.put("/academic-years/{academic_year_id}/room-unavailability/{room_id}",
            response_model=RoomUnavailabilityResponse)
def set_room_unavailability(academic_year_id: str, room_id: str, req: TeacherUnavailabilityRequest,
                            principal: User = Depends(require_admin("timetable"))) -> RoomUnavailabilityResponse:
    """Replace one room's periods out of use. 409 while the timetable books
    the room in one of them: move those periods first."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    _owned_room(room_id, principal)
    try:
        slots = cr._require().set_room_unavailability(
            academic_year_id, room_id, [(s.day_of_week, s.period) for s in req.slots])
    except ValueError as e:
        raise HTTPException(422, str(e))
    except InUse as e:
        raise HTTPException(409, str(e))
    _audit("room_unavailability_set", principal,
           {"academicYearId": academic_year_id, "roomId": room_id, "slots": len(slots)})
    return RoomUnavailabilityResponse(
        room_id=room_id, slots=[UnavailableSlot(day_of_week=d, period=p) for d, p in slots])


# ---------------- the school's own name and logo (EX-5, audit D41) ----------------

class SchoolProfileRequest(_Req):
    name: str = Field(min_length=1, max_length=120)
    address: Optional[str] = Field(default=None, max_length=200)
    affiliation: Optional[str] = Field(default=None, max_length=120)


class SchoolProfileResponse(Camel):
    """`name` is null until the principal sets it: papers then print no
    school name rather than a placeholder."""
    name: Optional[str] = None
    address: Optional[str] = None
    affiliation: Optional[str] = None
    has_logo: bool = False
    updated_at: Optional[str] = None


def _profile_response(p) -> SchoolProfileResponse:
    if p is None:
        return SchoolProfileResponse()
    return SchoolProfileResponse(name=p.name, address=p.address, affiliation=p.affiliation,
                                 has_logo=p.has_logo, updated_at=p.updated_at)


@router.get("/school-profile", response_model=SchoolProfileResponse)
def get_school_profile(current: User = Depends(get_current_user)) -> SchoolProfileResponse:
    """The caller's school's name, address, affiliation line and whether it
    has a logo. Any signed-in member of the school: the apps show the name."""
    return _profile_response(cr._require().get_school_profile(current.school_id))


@router.put("/school-profile", response_model=SchoolProfileResponse)
def set_school_profile(req: SchoolProfileRequest,
                       principal: User = Depends(require_principal)) -> SchoolProfileResponse:
    """Set the name every paper, answer key and report prints at the top."""
    try:
        before, after = cr._require().set_school_profile(
            principal.school_id, name=req.name, address=req.address, affiliation=req.affiliation,
            updated_by=principal.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    f = lambda p: {"name": p.name, "address": p.address, "affiliation": p.affiliation}  # noqa: E731
    if before is None or f(before) != f(after):
        _audit("school_profile_set", principal, {"before": f(before) if before else None, "after": f(after)})
    return _profile_response(after)


@router.get("/school-profile/logo")
def get_school_logo(current: User = Depends(get_current_user)):
    from fastapi.responses import Response
    from .school_profile import local_logo
    profile = cr._require().get_school_profile(current.school_id)
    path = local_logo(cr._cfg.data_root, profile)
    if path is None:
        raise HTTPException(404, "the school has no logo")
    return Response(content=path.read_bytes(), media_type=profile.logo_type,
                    headers={"Cache-Control": "private, max-age=300"})


@router.post("/school-profile/logo", response_model=SchoolProfileResponse)
async def upload_school_logo(file: UploadFile = File(...),
                             principal: User = Depends(require_principal)) -> SchoolProfileResponse:
    """A PNG or JPEG of at most 1 MB, printed beside the school's name. The
    school's name must be set first."""
    from .school_profile import LOGO_TYPES, MAX_LOGO_BYTES, save_logo
    store = cr._require()
    profile = store.get_school_profile(principal.school_id)
    if profile is None:
        raise HTTPException(409, "set the school's name first; the logo prints beside it")
    if file.content_type not in LOGO_TYPES:
        raise HTTPException(415, "send the logo as a PNG or JPEG image")
    data = await file.read(MAX_LOGO_BYTES + 1)
    if len(data) > MAX_LOGO_BYTES:
        raise HTTPException(413, "a logo may be at most 1 MB")
    if not data:
        raise HTTPException(422, "the image is empty")
    if not _looks_like(file.content_type, data):
        raise HTTPException(415, "that file is not the PNG or JPEG image it says it is")
    from .school_profile import image_is_printable
    if not image_is_printable(data):
        raise HTTPException(422, "that image could not be read; open it, save it again as a PNG or JPEG, "
                                 "and upload the new file")
    key = save_logo(cr._cfg.data_root, profile, file.content_type, data)
    after = store.set_school_logo(principal.school_id, logo_type=file.content_type, blob_key=key,
                                  updated_by=principal.id)
    _audit("school_logo_set", principal, {"contentType": file.content_type, "bytes": len(data),
                                          "durable": key is not None})
    return _profile_response(after)


@router.delete("/school-profile/logo", response_model=SchoolProfileResponse)
def remove_school_logo(principal: User = Depends(require_principal)) -> SchoolProfileResponse:
    from .school_profile import remove_logo
    store = cr._require()
    if store.get_school_profile(principal.school_id) is None:
        raise HTTPException(404, "the school has no logo")
    remove_logo(cr._cfg.data_root, principal.school_id)
    after = store.set_school_logo(principal.school_id, logo_type=None, blob_key=None, updated_by=principal.id)
    _audit("school_logo_removed", principal, {})
    return _profile_response(after)


def _looks_like(content_type: str, data: bytes) -> bool:
    """The file's first bytes match the type it claims: a renamed file that
    is not an image would fail later inside the PDF renderer."""
    if content_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    return data.startswith(b"\xff\xd8\xff")
