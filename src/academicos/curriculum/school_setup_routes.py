"""POST /curriculum/school-setup: a new school's year, classes, calendar,
terms and bell schedule in one call (curriculum/school_setup.py).

Principal only, for the caller's own school (school_id comes from the
session, never the body). Idempotent: a second call adds only what is
missing. The answer lists what it made and what was already there, and the
whole call is one audit entry.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from ..assessment.auth_routes import require_principal
from ..assessment.users import User
from . import routes as cr
from . import school_setup as setup_mod
from .schemas import AcademicYearResponse, Camel, CamelRequest
from .school_model_routes import _audit

router = APIRouter(prefix="/api/v1/curriculum")


class SetupClassBody(CamelRequest):
    """One class and its sections. Classes 6-10 can be set up here."""
    grade: int = Field(ge=1, le=12)
    sections: list[str] = Field(default_factory=lambda: list(setup_mod.DEFAULT_SECTIONS),
                                min_length=1, max_length=setup_mod.MAX_SECTIONS_PER_CLASS)


class SetupTermBody(CamelRequest):
    name: str = Field(min_length=1, max_length=40)
    start_date: str
    end_date: str


class SetupBellBody(CamelRequest):
    """The default bell: `periods` periods of `periodMinutes` from `startsAt`,
    and one break of `breakMinutes` after period `breakAfterPeriod` (0: none)."""
    periods: int = Field(default=8, ge=1, le=12)
    period_minutes: int = Field(default=40, ge=20, le=90)
    starts_at: str = Field(default="08:00", min_length=5, max_length=5)
    break_after_period: int = Field(default=4, ge=0, le=12)
    break_minutes: int = Field(default=20, ge=5, le=90)


def _default_classes() -> list[SetupClassBody]:
    return [SetupClassBody(grade=g) for g in setup_mod.DEFAULT_GRADES]


class SchoolSetupRequest(CamelRequest):
    """Every field has a default: an empty body sets up the Indian year
    containing today, classes 6-10 with section A, Sunday off, two terms and
    an 8-period bell."""
    year_label: Optional[str] = Field(default=None, max_length=20)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    classes: list[SetupClassBody] = Field(default_factory=_default_classes, min_length=1, max_length=12)
    weekly_off_days: list[str] = Field(default_factory=lambda: ["sunday"], max_length=7)
    second_and_fourth_saturdays_off: bool = False
    terms: Optional[list[SetupTermBody]] = Field(default=None, max_length=4)
    bell: SetupBellBody = Field(default_factory=SetupBellBody)


class SetupClassResponse(Camel):
    grade: int
    grade_id: str
    sections: list[str]
    subjects: int
    chapters_added: int
    created: bool


class SchoolSetupResponse(Camel):
    """`created` and `alreadyExisted` are sentences for the person who
    pressed the button; `skipped` names what could not be made and why."""
    academic_year: AcademicYearResponse
    classes: list[SetupClassResponse]
    calendar_id: str
    term_ids: list[str]
    bell_schedule_id: str
    created: list[str]
    already_existed: list[str]
    skipped: list[str]


@router.post("/school-setup", response_model=SchoolSetupResponse)
def school_setup(req: SchoolSetupRequest,
                 principal: User = Depends(require_principal)) -> SchoolSetupResponse:
    """Sets up the caller's school to start: academic year, classes and
    sections with their CBSE subjects and chapters, calendar, terms and bell
    schedule. Safe to call again; it adds only what is missing."""
    store = cr._require()
    try:
        result = setup_mod.set_up_school(
            store, school_id=principal.school_id, today=cr._school_today(),
            year_label_=req.year_label, start_date=req.start_date, end_date=req.end_date,
            classes=[setup_mod.ClassSpec(c.grade, list(c.sections)) for c in req.classes],
            weekly_off_days=req.weekly_off_days,
            alternate_saturday_rule="second_fourth" if req.second_and_fourth_saturdays_off else "none",
            terms=None if req.terms is None else [setup_mod.TermSpec(t.name, t.start_date, t.end_date)
                                                  for t in req.terms],
            bell=setup_mod.BellSpec(**req.bell.model_dump()))
    except ValueError as e:
        raise HTTPException(422, str(e))
    if result.created:
        _audit("school_set_up", principal, {"academicYearId": result.academic_year_id,
                                            "created": result.created, "skipped": result.skipped})
    year = store.get_academic_year(result.academic_year_id)
    return SchoolSetupResponse(
        academic_year=AcademicYearResponse(id=year.id, school_id=year.school_id, label=year.label,
                                           start_date=year.start_date, end_date=year.end_date,
                                           status=year.status),
        classes=[SetupClassResponse(grade=c.grade, grade_id=c.grade_id, sections=c.sections,
                                    subjects=c.subjects, chapters_added=c.chapters_added, created=c.created)
                 for c in result.classes],
        calendar_id=result.calendar_id, term_ids=result.term_ids,
        bell_schedule_id=result.bell_schedule_id,
        created=result.created, already_existed=result.already_existed, skipped=result.skipped)
