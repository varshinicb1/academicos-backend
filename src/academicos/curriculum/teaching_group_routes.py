"""HTTP routes for teaching that spans sections (REQUIREMENTS SCH-2/SCH-8,
v3 audit N-3-20; curriculum/teaching_groups.py): elective groups and
combined classes, and splitting or merging a section mid-year.

A group's periods are placed by the timetable generator, as one block for
all its sections and lanes; creating or changing a group places nothing. A
split section gets its allocations and groups, and then its week from the
generator (`sectionIds: [the new section]`); a merged section's students
keep the other section's week at once.

Reads are staff-only. Writes are the timetable admin's, and each is written
to the audit log with what it was before (REQUIREMENTS ROLE-3).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from ..assessment.auth_routes import require_admin, require_staff
from ..assessment.users import User
from . import routes as cr
from .schemas import Camel, SectionResponse
from .school_model_routes import GroupLaneResponse, _audit, _lane, _owned_room, _Req, _staff_member

router = APIRouter(prefix="/api/v1/curriculum")


class GroupLaneBody(_Req):
    """One lane: its subject (a subject of the sections' class), teacher,
    room, and for an elective the students who chose it. `id` keeps a lane
    across an edit."""
    id: Optional[str] = None
    subject_id: str = Field(min_length=1)
    teacher_id: Optional[str] = None
    room_id: Optional[str] = None
    student_ids: list[str] = Field(default_factory=list, max_length=200)


class TeachingGroupRequest(_Req):
    """kind: "elective" (two or more parallel lanes, each its own subject
    and teacher) or "combined" (one lane: the sections taught together)."""
    kind: str
    name: str = Field(min_length=1, max_length=60)
    periods_per_week: int = Field(ge=1, le=60)
    section_ids: list[str] = Field(min_length=1, max_length=20)
    lanes: list[GroupLaneBody] = Field(min_length=1, max_length=12)


class GroupSlotResponse(Camel):
    day_of_week: int
    period: int


class TeachingGroupResponse(Camel):
    id: str
    academic_year_id: str
    kind: str
    name: str
    periods_per_week: int
    section_ids: list[str]
    section_names: list[str]
    lanes: list[GroupLaneResponse]
    # Where the generator placed it; empty until the timetable is generated
    # with the group (and again after its sections, periods or a lane's
    # teacher or room change).
    periods: list[GroupSlotResponse] = []


class SectionSplitRequest(_Req):
    name: str = Field(min_length=1, max_length=20)
    student_ids: list[str] = Field(min_length=1, max_length=200)


class SectionSplitResponse(Camel):
    """The new section, and what it took from the old one. Its week is
    generated next: POST .../timetable/solve with sectionIds [its id]."""
    section: SectionResponse
    moved_students: int
    allocations_copied: int
    groups_joined: int


class SectionMergeRequest(_Req):
    into_section_id: str = Field(min_length=1)


class SectionMergeResponse(Camel):
    into: SectionResponse
    moved_students: int
    periods_removed: int
    allocations_removed: int


def _group(g) -> TeachingGroupResponse:
    store = cr._require()
    sections = [store.get_section(sid) for sid in g.section_ids]
    return TeachingGroupResponse(
        id=g.id, academic_year_id=g.academic_year_id, kind=g.kind, name=g.name,
        periods_per_week=g.periods_per_week, section_ids=list(g.section_ids),
        section_names=[store._section_label(s) for s in sections if s], lanes=[_lane(lane) for lane in g.lanes],
        periods=[GroupSlotResponse(day_of_week=d, period=p) for gg, d, p in
                 store.group_periods_for_year(g.academic_year_id) if gg.id == g.id])


def _group_fields(g) -> dict[str, Any]:
    return {"kind": g.kind, "name": g.name, "periodsPerWeek": g.periods_per_week, "sectionIds": g.section_ids,
            "lanes": [{"subjectId": lane.subject_id, "teacherId": lane.teacher_id, "roomId": lane.room_id,
                       "students": len(lane.student_ids)} for lane in g.lanes]}


def _owned_group(group_id: str, current: User):
    g = cr._require().get_teaching_group(group_id)
    if g is None:
        raise HTTPException(404, "group not found")
    if g.school_id != current.school_id:
        raise HTTPException(403, "this group belongs to a different school")
    return g


def _save(req: TeachingGroupRequest, academic_year_id: str, principal: User,
          group_id: Optional[str] = None) -> TeachingGroupResponse:
    for sid in req.section_ids:
        cr._require_school_owns_section(sid, principal)
    for lane in req.lanes:
        _staff_member(lane.teacher_id, principal)
        if lane.room_id is not None:
            _owned_room(lane.room_id, principal)
    try:
        before, after = cr._require().save_teaching_group(
            academic_year_id=academic_year_id, kind=req.kind, name=req.name,
            periods_per_week=req.periods_per_week, section_ids=req.section_ids,
            lanes=[lane.model_dump() for lane in req.lanes], group_id=group_id)
    except KeyError:
        raise HTTPException(404, "a section, subject or the group was not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("teaching_group_saved", principal,
           {"groupId": after.id, "before": _group_fields(before) if before else None, "after": _group_fields(after)})
    return _group(after)


@router.get("/academic-years/{academic_year_id}/teaching-groups", response_model=list[TeachingGroupResponse])
def list_teaching_groups(academic_year_id: str, current: User = Depends(require_staff)) -> list[TeachingGroupResponse]:
    """The year's elective groups and combined classes, with their lanes and
    where they are placed."""
    cr._require_school_owns_academic_year(academic_year_id, current)
    return [_group(g) for g in cr._require().teaching_groups_for_year(academic_year_id)]


@router.post("/academic-years/{academic_year_id}/teaching-groups", response_model=TeachingGroupResponse)
def create_teaching_group(academic_year_id: str, req: TeachingGroupRequest,
                          principal: User = Depends(require_admin("timetable"))) -> TeachingGroupResponse:
    """An elective group (students of one or more sections of a class split
    into parallel lanes, taught at the same time by different teachers) or a
    combined class (two or more sections taught together by one teacher in
    one room). Generating the timetable places it as one block."""
    cr._require_school_owns_academic_year(academic_year_id, principal)
    return _save(req, academic_year_id, principal)


@router.put("/teaching-groups/{group_id}", response_model=TeachingGroupResponse)
def update_teaching_group(group_id: str, req: TeachingGroupRequest,
                          principal: User = Depends(require_admin("timetable"))) -> TeachingGroupResponse:
    """Replace a group's definition. Changing its sections, periods a week
    or a lane's teacher or room takes its placed periods away: generate the
    timetable again to place it."""
    g = _owned_group(group_id, principal)
    return _save(req, g.academic_year_id, principal, group_id=group_id)


@router.delete("/teaching-groups/{group_id}")
def delete_teaching_group(group_id: str, principal: User = Depends(require_admin("timetable"))) -> dict:
    """The group and its periods; its sections' periods then are free."""
    _owned_group(group_id, principal)
    g = cr._require().delete_teaching_group(group_id)
    _audit("teaching_group_deleted", principal, {"groupId": group_id, "before": _group_fields(g)})
    return {"ok": True}


def _section(sec) -> SectionResponse:
    store = cr._require()
    grade = store.get_grade(sec.grade_id)
    return cr._section_response(sec, grade.number if grade else 0, len(store.enrollments_for_section(sec.id)))


@router.post("/sections/{section_id}/split", response_model=SectionSplitResponse)
def split_section(section_id: str, req: SectionSplitRequest,
                  principal: User = Depends(require_admin("timetable", scoped=True))) -> SectionSplitResponse:
    """Split a section mid-year (SCH-8): `studentIds` move to a new section
    of the class named `name`, which gets the old section's bell, a copy of
    each of its allocations and its groups. The old section keeps its week;
    generate the new section's next (solve with sectionIds [its id])."""
    cr._require_school_owns_section(section_id, principal)
    cr.require_section_in_scope(section_id, principal, "timetable")
    store = cr._require()
    try:
        out = store.split_section(section_id, name=req.name, student_ids=req.student_ids)
    except KeyError:
        raise HTTPException(404, "section not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    new = out["section"]
    _audit("section_split", principal, {"sectionId": section_id, "newSectionId": new.id, "name": new.name,
                                        "movedStudents": out["moved"], "allocations": out["allocations"],
                                        "groups": out["groups"]})
    return SectionSplitResponse(section=_section(new), moved_students=out["moved"],
                                allocations_copied=out["allocations"], groups_joined=len(out["groups"]))


@router.post("/sections/{section_id}/merge", response_model=SectionMergeResponse)
def merge_section(section_id: str, req: SectionMergeRequest,
                  principal: User = Depends(require_admin("timetable", scoped=True))) -> SectionMergeResponse:
    """Merge a section into another of its class mid-year (SCH-8): every
    student moves and keeps the other section's week; the emptied section,
    its week and its allocations are removed. 422 while it is in a group
    the other section is not in. Its students and the teachers who lose
    periods are told."""
    from ..operations.routes import notify_parents_safely, notify_safely
    for sid in (section_id, req.into_section_id):
        cr._require_school_owns_section(sid, principal)
        cr.require_section_in_scope(sid, principal, "timetable")
    store = cr._require()
    students = [e.student_id for e in store.enrollments_for_section(section_id)]
    teachers = sorted({t for e in store.timetable_for_section(section_id)
                       for t in (e.teacher_id, e.co_teacher_id) if t})
    try:
        out = store.merge_section(section_id, req.into_section_id)
    except KeyError:
        raise HTTPException(404, "section not found")
    except ValueError as e:
        raise HTTPException(422, str(e))
    _audit("section_merged", principal, {"sectionId": section_id, "intoSectionId": req.into_section_id,
                                         "movedStudents": out["moved"], "periodsRemoved": out["periods"],
                                         "allocationsRemoved": out["allocations"]})
    for who in (teachers, students):
        if who:
            notify_safely(school_id=principal.school_id, user_ids=who, kind="timetable_published", params={},
                          link="/my-timetable")
    notify_parents_safely(school_id=principal.school_id, student_ids=students, kind="timetable_published",
                          params={}, exclude=teachers)
    return SectionMergeResponse(into=_section(out["into"]), moved_students=out["moved"],
                                periods_removed=out["periods"], allocations_removed=out["allocations"])
