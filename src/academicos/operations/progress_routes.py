"""A person's saved progress, and the school's setup level.

  GET  /api/v1/me/progress        what the caller has done (Guide jobs, the welcome, ...)
  PUT  /api/v1/me/progress/{key}  save one entry; last write wins
  GET  /api/v1/setup-status       the principal's school, step by step, and how far along

Setup level is computed from the school's own data on every read, never stored, so it
cannot drift from the truth: a step is done when the thing it names exists.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

from ..assessment import paper_timing
from ..assessment import routes as ar
from ..assessment.auth_routes import get_current_user, require_principal
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .progress import ProgressRefused
from .routes import store

router = APIRouter(prefix="/api/v1")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ProgressEntry(Camel):
    value: Any = None
    updated_at: str


class ProgressResponse(Camel):
    items: dict[str, ProgressEntry]


class ProgressWrite(_Req):
    value: Any = None


@router.get("/me/progress", response_model=ProgressResponse)
def my_progress(current: User = Depends(get_current_user)) -> ProgressResponse:
    return ProgressResponse(items={k: ProgressEntry(value=v["value"], updated_at=v["updatedAt"])
                                   for k, v in store().progress_of(current.id).items()})


@router.put("/me/progress/{key}", response_model=ProgressEntry)
def save_progress(key: str, req: ProgressWrite, current: User = Depends(get_current_user)) -> ProgressEntry:
    try:
        saved = store().set_progress(current.id, key, req.value)
    except ProgressRefused as e:
        raise HTTPException(422, str(e))
    return ProgressEntry(value=saved["value"], updated_at=saved["updatedAt"])


class SetupStep(Camel):
    key: str
    title: str
    done: bool
    detail: str
    path: str


class SetupStatus(Camel):
    steps: list[SetupStep]
    done: int
    total: int
    percent: int
    next_key: Optional[str] = None


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


@router.get("/setup-status", response_model=SetupStatus)
def setup_status(principal: User = Depends(require_principal)) -> SetupStatus:
    """How far this school is from making its first real paper, in the order a school does it."""
    school_id = principal.school_id
    cs, users = cr._require(), cr._require_users()

    profile = cs.get_school_profile(school_id)
    named = bool(profile and (profile.name or "").strip())
    years = sorted(cs.academic_years_for_school(school_id), key=lambda y: y.start_date)
    year = years[-1] if years else None
    grades = cs.grades_for_year(year.id) if year else []
    terms = cs.terms_for_year(year.id) if year else []
    allocations = cs.allocations_for_year(year.id) if year else []
    timetabled = bool(year and cs.timetable_for_year(year.id))
    teachers = users.users_for_school(school_id, role="teacher")
    waiting = [i for i in users.invites_for_school(school_id)
               if i.role == "teacher" and not i.used_by and not i.revoked_at]
    students = users.users_for_school(school_id, role="student")
    papers = paper_timing.kept_papers(ar._require()[1].list_by_school(school_id))
    staffed = [a for a in allocations if a.teacher_id]

    if teachers:
        teacher_detail = f"{len(teachers)} joined" + (f", {len(waiting)} invited" if waiting else "")
    elif waiting:
        teacher_detail = f"{_plural(len(waiting), 'invite', 'invites')} waiting to be used"
    else:
        teacher_detail = "No teacher invited yet"

    steps = [
        SetupStep(key="profile", title="School name and logo", done=named, path="/school/school-profile",
                  detail="Printed on every paper" if named else "Not set: papers print no school name"),
        SetupStep(key="classes", title="Classes and sections", done=bool(grades), path="/school/setup",
                  detail=_plural(len(grades), "class", "classes") if grades else "No classes yet"),
        SetupStep(key="calendar", title="Year, terms and bell", done=bool(terms), path="/school/setup",
                  detail=_plural(len(terms), "term", "terms") if terms else "No terms yet"),
        SetupStep(key="teachers", title="Teachers", done=bool(teachers), path="/admin", detail=teacher_detail),
        SetupStep(key="subjects", title="A teacher for each subject",
                  done=bool(allocations) and len(staffed) == len(allocations), path="/school/timetable",
                  detail=(f"{len(staffed)} of {len(allocations)} subject slots have a teacher"
                          if allocations else "No subjects set up yet")),
        SetupStep(key="timetable", title="Timetable", done=timetabled, path="/school/timetable",
                  detail="Published" if timetabled else "Not made yet"),
        SetupStep(key="students", title="Students", done=bool(students), path="/admin",
                  detail=_plural(len(students), "student", "students") if students else "None have joined yet"),
        SetupStep(key="paper", title="First question paper", done=bool(papers), path="/assessment",
                  detail=_plural(len(papers), "paper", "papers") if papers else "None made yet"),
    ]
    done = sum(1 for s in steps if s.done)
    nxt = next((s.key for s in steps if not s.done), None)
    return SetupStatus(steps=steps, done=done, total=len(steps), percent=round(100 * done / len(steps)),
                       next_key=nxt)
