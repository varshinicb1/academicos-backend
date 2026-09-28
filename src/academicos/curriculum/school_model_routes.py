"""HTTP routes for the school model (M1.2-M1.3, curriculum/school_model.py):
teaching allocations, rooms, bell schedules and each section's timetable.

Reads are staff-only (a timetable names teachers); `/my-timetable` is any
signed-in user's own week. Writes are the principal's, school-scoped like every
other curriculum write, and each is written to the audit log with what it was
before (REQUIREMENTS ROLE-3). Teacher-level load is the principal's (ADM-5).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import get_current_user, require_principal, require_staff
from ..assessment.users import User
from . import routes as cr
from .schemas import Camel
from .school_model import InUse, TimetableClash

router = APIRouter(prefix="/api/v1/curriculum")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------- request / response shapes ----------------

class AllocationRequest(_Req):
    teacher_id: Optional[str] = None
    periods_per_week: int = Field(ge=1, le=60)


class AllocationResponse(Camel):
    id: str
    academic_year_id: str
    section_id: str
    subject_id: str
    teacher_id: Optional[str] = None
    periods_per_week: int


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


class TimetableGapResponse(Camel):
    subject_id: str
    allocated: int
    timetabled: int


class SectionTimetableResponse(Camel):
    section_id: str
    bell_schedule_id: Optional[str] = None
    entries: list[TimetableEntryResponse]
    gaps: list[TimetableGapResponse]


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
                              periods_per_week=a.periods_per_week)


def _room(r) -> RoomResponse:
    return RoomResponse(id=r.id, name=r.name, kind=r.kind, capacity=r.capacity)


def _bell(b) -> BellScheduleResponse:
    return BellScheduleResponse(
        id=b.id, academic_year_id=b.academic_year_id, name=b.name, days=b.days,
        slots=[BellSlotResponse(start=s.start, end=s.end, kind=s.kind, period=s.period) for s in b.slots],
        is_default=b.is_default, teaching_periods=b.teaching_periods)


def _entry(e) -> TimetableEntryResponse:
    return TimetableEntryResponse(id=e.id, section_id=e.section_id, day_of_week=e.day_of_week,
                                  period=e.period, subject_id=e.subject_id, teacher_id=e.teacher_id,
                                  room_id=e.room_id)


def _section_week(section) -> SectionTimetableResponse:
    store = cr._require()
    bell = store.bell_for_section(section)
    return SectionTimetableResponse(
        section_id=section.id, bell_schedule_id=bell.id if bell else None,
        entries=[_entry(e) for e in store.timetable_for_section(section.id)],
        gaps=[TimetableGapResponse(subject_id=g["subjectId"], allocated=g["allocated"],
                                   timetabled=g["timetabled"])
              for g in store.timetable_gaps(section.id)])


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
                   principal: User = Depends(require_principal)) -> AllocationResponse:
    """Set one cell of the allocation grid: who teaches this subject to this
    section, and how many periods a week."""
    cr._require_school_owns_section(section_id, principal)
    _staff_member(req.teacher_id, principal)
    try:
        before, after = cr._require().set_allocation(section_id=section_id, subject_id=subject_id,
                                                     teacher_id=req.teacher_id,
                                                     periods_per_week=req.periods_per_week)
    except KeyError:
        raise HTTPException(404, "subject not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    fields = lambda a: {"teacherId": a.teacher_id, "periodsPerWeek": a.periods_per_week}  # noqa: E731
    if before is None or fields(before) != fields(after):
        _audit("allocation_set", principal,
               {"sectionId": section_id, "subjectId": subject_id,
                "before": fields(before) if before else None, "after": fields(after)})
    return _alloc(after)


@router.delete("/sections/{section_id}/allocations/{subject_id}")
def delete_allocation(section_id: str, subject_id: str,
                      principal: User = Depends(require_principal)) -> dict:
    """409 while the timetable still gives the subject periods."""
    cr._require_school_owns_section(section_id, principal)
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
                 principal: User = Depends(require_principal)) -> list[TeacherLoadResponse]:
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
def create_room(req: RoomRequest, principal: User = Depends(require_principal)) -> RoomResponse:
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
                principal: User = Depends(require_principal)) -> RoomResponse:
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
def delete_room(room_id: str, principal: User = Depends(require_principal)) -> dict:
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
                         principal: User = Depends(require_principal)) -> BellScheduleResponse:
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
                         principal: User = Depends(require_principal)) -> BellScheduleResponse:
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
def delete_bell_schedule(bell_schedule_id: str, principal: User = Depends(require_principal)) -> dict:
    _owned_bell(bell_schedule_id, principal)
    try:
        b = cr._require().delete_bell_schedule(bell_schedule_id)
    except InUse as e:
        raise HTTPException(409, str(e))
    _audit("bell_schedule_deleted", principal, {"bellScheduleId": bell_schedule_id, "name": b.name})
    return {"ok": True}


@router.put("/sections/{section_id}/bell-schedule", response_model=SectionTimetableResponse)
def set_section_bell(section_id: str, req: SectionBellRequest,
                     principal: User = Depends(require_principal)) -> SectionTimetableResponse:
    """Give a section its own bell (null: the year's default)."""
    section = cr._require_school_owns_section(section_id, principal)
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
        entries = [e for e in entries if e.teacher_id == teacher_id]
    return [_entry(e) for e in entries]


@router.get("/sections/{section_id}/timetable", response_model=SectionTimetableResponse)
def section_timetable(section_id: str, current: User = Depends(require_staff)) -> SectionTimetableResponse:
    """A section's week, its bell, and per subject the periods allocated
    against the periods placed."""
    return _section_week(cr._require_school_owns_section(section_id, current))


@router.put("/sections/{section_id}/timetable", response_model=SectionTimetableResponse)
def replace_section_timetable(section_id: str, req: SectionTimetableRequest,
                              principal: User = Depends(require_principal)) -> SectionTimetableResponse:
    """Replace a section's whole week. Checked before anything is written: a
    teacher or room already busy in another section, a period or day the
    bell does not have, a subject with no allocation or more periods than
    allocated. A 409 lists every clash; the old week is kept."""
    section = cr._require_school_owns_section(section_id, principal)
    for e in req.entries:
        _staff_member(e.teacher_id, principal)
    store = cr._require()
    before = len(store.timetable_for_section(section_id))
    try:
        store.replace_section_timetable(section_id, [e.model_dump() for e in req.entries])
    except TimetableClash as e:
        raise HTTPException(409, {"message": "the timetable has clashes; nothing was changed",
                                  "clashes": e.clashes})
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("timetable_replaced", principal, {"sectionId": section_id, "before": before,
                                             "after": len(req.entries)})
    return _section_week(store.get_section(section.id))


class MyTimetableResponse(Camel):
    role: str
    academic_year_id: Optional[str] = None
    section_id: Optional[str] = None
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
        return MyTimetableResponse(role=current.role, academic_year_id=section.academic_year_id,
                                   section_id=section.id, bell_schedule=_bell(bell) if bell else None,
                                   entries=[_entry(e) for e in store.timetable_for_section(section.id)])
    bells = store.bell_schedules_for_year(year.id)
    default = next((b for b in bells if b.is_default), None)
    return MyTimetableResponse(role=current.role, academic_year_id=year.id,
                               bell_schedule=_bell(default) if default else None,
                               entries=[_entry(e) for e in store.timetable_for_teacher(current.id, year.id)])
