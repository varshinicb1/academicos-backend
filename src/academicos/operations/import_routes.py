"""Bulk import and export (M1.6; INT-3, ROLE-2): CSV for sections, teachers,
students, parents, teaching allocations and holidays.

Every import is planned first: each row gets an outcome -- `ok` with the
action it will take, `skip` with why nothing is needed, or `error` with what
to fix. `dryRun` (the default) returns the plan and writes nothing. An apply
with any error row is refused whole, so a file is never half-imported
because of a typo on row 212. The apply is audited with its counts.

Accounts are never created here. A teacher, student or parent row becomes an
email-bound invite (its code is in the result for the school to hand out);
a student's invite enrols them in their section when they register, and a
parent's links them to their child as soon as both have joined
(guardian_routes.on_register).
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import require_admin, require_principal
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .guardians import RELATIONS
from .routes import store

router = APIRouter(prefix="/api/v1/admin")

COLUMNS: dict[str, list[str]] = {
    "sections": ["grade", "section", "class_teacher_email"],
    "teachers": ["name", "email"],
    "students": ["name", "email", "grade", "section"],
    "parents": ["name", "email", "student_email", "relation"],
    "allocations": ["grade", "section", "subject", "teacher_email", "periods_per_week"],
    "holidays": ["date", "end_date", "name", "kind"],
}
REQUIRED: dict[str, set[str]] = {
    "sections": {"grade", "section"},
    "teachers": {"email"},
    "students": {"email", "grade", "section"},
    "parents": {"email", "student_email"},
    "allocations": {"grade", "section", "subject", "periods_per_week"},
    "holidays": {"date", "name"},
}
EXAMPLES: dict[str, list[str]] = {
    "sections": ["10", "B", "anita@school.example"],
    "teachers": ["Anita Rao", "anita@school.example"],
    "students": ["Asha Kumar", "asha@school.example", "10", "B"],
    "parents": ["Meena Kumar", "meena@example.com", "asha@school.example", "mother"],
    "allocations": ["10", "B", "Science", "anita@school.example", "6"],
    "holidays": ["2026-10-20", "2026-10-24", "Diwali break", "holiday"],
}
MAX_BYTES = 1_000_000
MAX_ROWS = 2000
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class _Req(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ImportRequest(_Req):
    """The rows as CSV text, or as an Excel workbook (.xlsx, base64): its
    first sheet, in the same columns (INT-3)."""
    csv: Optional[str] = Field(default=None, max_length=MAX_BYTES)
    xlsx: Optional[str] = Field(default=None, max_length=MAX_BYTES * 2)
    academic_year_id: Optional[str] = None


class RowOutcome(Camel):
    row: int                        # the line in the file, counting the header as 1
    status: str                     # ok | skip | error
    message: str
    invite_code: Optional[str] = None


class ImportResponse(Camel):
    kind: str
    dry_run: bool
    applied: bool
    counts: dict[str, int]
    rows: list[RowOutcome]


@dataclass
class _Row:
    line: int
    status: str
    message: str
    apply: Optional[Callable[[], Optional[str]]] = None
    code: Optional[str] = None


@dataclass
class _Ctx:
    principal: User
    year: Any
    cs: Any
    users: Any
    ops: Any
    grades: dict[int, Any] = field(default_factory=dict)          # number -> Grade (existing or planned)
    planned: set = field(default_factory=set)                       # keys already planned in this file


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")


def _file_rows(req: ImportRequest) -> list[list[str]]:
    """The file's rows, from the CSV text or the Excel workbook."""
    if (req.csv is None) == (req.xlsx is None):
        raise HTTPException(422, "send the rows as csv or as xlsx, one of the two")
    if req.csv is not None:
        return list(csv.reader(io.StringIO(req.csv.lstrip("﻿"))))
    import base64
    import binascii
    from .xlsx import XlsxError, read_rows
    try:
        data = base64.b64decode(req.xlsx, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "the Excel file did not arrive whole; choose it again")
    try:
        return read_rows(data)
    except XlsxError as e:
        raise HTTPException(422, str(e))


def _parse(kind: str, rows_in: list[list[str]]) -> list[tuple[int, dict[str, str]]]:
    reader = iter(rows_in)
    try:
        header = [_norm(h) for h in next(reader)]
    except StopIteration:
        raise HTTPException(422, "the file is empty")
    missing = REQUIRED[kind] - set(header)
    if missing:
        raise HTTPException(422, f"the header is missing: {', '.join(sorted(missing))} "
                                 f"(expected columns: {', '.join(COLUMNS[kind])})")
    rows = []
    for i, cells in enumerate(reader, start=2):
        if not any(c.strip() for c in cells):
            continue
        rows.append((i, {h: (cells[j].strip() if j < len(cells) else "") for j, h in enumerate(header)}))
        if len(rows) > MAX_ROWS:
            raise HTTPException(422, f"at most {MAX_ROWS} rows per file")
    return rows


def _grade_number(v: str) -> Optional[int]:
    try:
        n = int(v)
    except ValueError:
        return None
    return n if 1 <= n <= 12 else None


def _section(ctx: _Ctx, grade: str, name: str):
    n = _grade_number(grade)
    g = ctx.grades.get(n) if n else None
    if g is None or isinstance(g, str):
        return None
    return next((s for s in ctx.cs.sections_for_grade(g.id) if s.name.lower() == name.lower()), None)


def _staff(ctx: _Ctx, email: str):
    u = ctx.users.get_by_email(email.lower())
    if u is None or u.school_id != ctx.principal.school_id or u.role not in ("teacher", "principal"):
        return None
    return u


def _open_invite(ctx: _Ctx, email: str, role: str):
    return next((i for i in ctx.users.invites_for_school(ctx.principal.school_id)
                 if i.email == email.lower() and i.role == role and i.status() == "open"), None)


def _invite(ctx: _Ctx, role: str, email: str) -> str:
    return ctx.users.create_invite(school_id=ctx.principal.school_id, role=role, created_by=ctx.principal.id,
                                   email=email).code


# ---------------- planners, one per kind ----------------

def _plan_sections(ctx: _Ctx, line: int, r: dict) -> _Row:
    n = _grade_number(r.get("grade", ""))
    name = r.get("section", "").strip()
    if n is None:
        return _Row(line, "error", f"grade must be a class from 1 to 12, not {r.get('grade')!r}")
    if not name or len(name) > 20:
        return _Row(line, "error", "section needs a name of at most 20 characters")
    key = ("section", n, name.lower())
    if key in ctx.planned:
        return _Row(line, "skip", f"{n}-{name} is already earlier in this file")
    ctx.planned.add(key)
    teacher = None
    if r.get("class_teacher_email"):
        teacher = _staff(ctx, r["class_teacher_email"])
        if teacher is None:
            return _Row(line, "error", f"no teacher account with email {r['class_teacher_email']} at this school")
    if _section(ctx, str(n), name) is not None:
        return _Row(line, "skip", f"{n}-{name} already exists")
    new_grade = n not in ctx.grades
    if new_grade:
        ctx.grades[n] = "planned"

    def apply() -> None:
        g = ctx.grades.get(n)
        if g == "planned":
            g = ctx.grades[n] = ctx.cs.create_grade(academic_year_id=ctx.year.id, number=n)
        existing = next((s for s in ctx.cs.sections_for_grade(g.id) if s.name.lower() == name.lower()), None)
        if existing is None:
            ctx.cs.create_section(grade_id=g.id, name=name, class_teacher_id=teacher.id if teacher else None)
    msg = (f"create class {n} (it starts with section A) and section {name}" if new_grade and name.upper() != "A"
           else f"create class {n}" if new_grade else f"create section {n}-{name}")
    return _Row(line, "ok", msg + (f", class teacher {teacher.name}" if teacher else ""), apply)


def _wrong_role(line: int, r: dict, kind: str, roles: tuple[str, ...]) -> Optional[_Row]:
    """A row that names its own role (the people export carries a role
    column) and names another: pasting the people export into the teachers
    import invited a role=student row as a teacher (QA S-08)."""
    role = (r.get("role") or "").strip().lower()
    if role and role not in roles:
        other = {"student": "students", "teacher": "teachers", "principal": "teachers"}.get(role)
        hint = f"; import it with Import {other}" if other and other != kind else ""
        return _Row(line, "error", f"this row's role is {role}, not {' or '.join(roles)}{hint}")
    return None


def _plan_teachers(ctx: _Ctx, line: int, r: dict) -> _Row:
    email = r.get("email", "").lower()
    if not _EMAIL.match(email):
        return _Row(line, "error", f"{r.get('email')!r} is not an email address")
    wrong = _wrong_role(line, r, "teachers", ("teacher", "principal"))
    if wrong is not None:
        return wrong
    if ("email", email) in ctx.planned:
        return _Row(line, "skip", "this email is already earlier in this file")
    ctx.planned.add(("email", email))
    u = ctx.users.get_by_email(email)
    if u is not None:
        if u.school_id == ctx.principal.school_id and u.role in ("teacher", "principal"):
            return _Row(line, "skip", f"{u.name} already has a staff account")
        if u.school_id == ctx.principal.school_id:
            return _Row(line, "error", f"{u.name} already has a {u.role} account at this school, "
                                       "not a staff one")
        return _Row(line, "error", "this email already has an account that is not this school's staff")
    if _open_invite(ctx, email, "teacher"):
        return _Row(line, "skip", "already invited; the invite is still open")
    return _Row(line, "ok", f"invite {r.get('name') or email} as a teacher", lambda: _invite(ctx, "teacher", email))


def _plan_students(ctx: _Ctx, line: int, r: dict) -> _Row:
    email = r.get("email", "").lower()
    if not _EMAIL.match(email):
        return _Row(line, "error", f"{r.get('email')!r} is not an email address")
    if ("email", email) in ctx.planned:
        return _Row(line, "skip", "this email is already earlier in this file")
    ctx.planned.add(("email", email))
    wrong = _wrong_role(line, r, "students", ("student",))
    if wrong is not None:
        return wrong
    section = _section(ctx, r.get("grade", ""), r.get("section", ""))
    if section is None:
        return _Row(line, "error", f"there is no section {r.get('grade')}-{r.get('section')} this year; "
                                   "import sections first")
    label = ctx.cs._section_label(section)
    u = ctx.users.get_by_email(email)
    if u is not None:
        if u.school_id != ctx.principal.school_id or u.role != "student":
            return _Row(line, "error", "this email already has an account that is not this school's student")
        e = ctx.cs.enrollment_for_student(u.id)
        if e is not None and e.section_id == section.id:
            return _Row(line, "skip", f"{u.name} is already in {label}")
        return _Row(line, "ok", f"move {u.name} into {label}" if e else f"enrol {u.name} in {label}",
                    lambda: ctx.cs.enroll_student(school_id=ctx.principal.school_id, student_id=u.id,
                                                  section_id=section.id) and None)
    if _open_invite(ctx, email, "student"):
        return _Row(line, "skip", "already invited; the invite is still open")

    def apply() -> str:
        code = _invite(ctx, "student", email)
        ctx.ops.remember_pending_enrollment(code=code, school_id=ctx.principal.school_id, section_id=section.id,
                                            created_by=ctx.principal.id)
        return code
    return _Row(line, "ok", f"invite {r.get('name') or email} as a student of {label}", apply)


def _plan_parents(ctx: _Ctx, line: int, r: dict) -> _Row:
    email = r.get("email", "").lower()
    child_email = r.get("student_email", "").lower()
    relation = (r.get("relation") or "guardian").strip().lower()
    if not _EMAIL.match(email) or not _EMAIL.match(child_email):
        return _Row(line, "error", "email and student_email must both be email addresses")
    if relation not in RELATIONS:
        return _Row(line, "error", f"relation must be one of {', '.join(RELATIONS)}")
    key = ("parent", email, child_email)
    if key in ctx.planned:
        return _Row(line, "skip", "this parent and child are already earlier in this file")
    ctx.planned.add(key)
    child = ctx.users.get_by_email(child_email)
    if child is not None and (child.school_id != ctx.principal.school_id or child.role != "student"):
        return _Row(line, "error", f"{child_email} is not a student of this school")
    if child is None and not _open_invite(ctx, child_email, "student") and ("email", child_email) not in ctx.planned:
        return _Row(line, "error", f"no student with email {child_email}; import students first")
    school = ctx.principal.school_id
    parent = ctx.users.get_by_email(email)
    if parent is not None:
        if parent.school_id != school or parent.role != "parent":
            return _Row(line, "error", "this email already has an account that is not a parent at this school")
        if child is not None:
            if ctx.ops.is_guardian(parent.id, child.id):
                return _Row(line, "skip", f"{parent.name} is already linked to {child.name}")
            return _Row(line, "ok", f"link {parent.name} to {child.name}",
                        lambda: ctx.ops.link_guardian(parent_id=parent.id, student_id=child.id, school_id=school,
                                                      relation=relation, created_by=ctx.principal.id) and None)
        return _Row(line, "ok", f"link {parent.name} to {child_email} once they join",
                    lambda: ctx.ops.remember_pending_link(parent_id=parent.id, school_id=school,
                                                          student_email=child_email, relation=relation,
                                                          created_by=ctx.principal.id))

    def apply() -> str:
        code = _invite(ctx, "parent", email)
        ctx.ops.remember_guardian_invite(code=code, school_id=school, relation=relation, created_by=ctx.principal.id,
                                         student_id=child.id if child else None,
                                         student_email=None if child else child_email)
        return code
    return _Row(line, "ok", f"invite {r.get('name') or email} as {relation} of {child.name if child else child_email}",
                apply)


def _plan_allocations(ctx: _Ctx, line: int, r: dict) -> _Row:
    section = _section(ctx, r.get("grade", ""), r.get("section", ""))
    if section is None:
        return _Row(line, "error", f"there is no section {r.get('grade')}-{r.get('section')} this year")
    label = ctx.cs._section_label(section)
    subject = next((s for s in ctx.cs.subjects_for_grade(section.grade_id)
                    if s.name.lower() == r.get("subject", "").strip().lower()), None)
    if subject is None:
        return _Row(line, "error", f"class {r.get('grade')} has no subject {r.get('subject')!r}")
    try:
        periods = int(r.get("periods_per_week", ""))
    except ValueError:
        periods = -1
    if not 1 <= periods <= 40:
        return _Row(line, "error", "periods_per_week must be a whole number from 1 to 40")
    teacher = None
    if r.get("teacher_email"):
        teacher = _staff(ctx, r["teacher_email"])
        if teacher is None:
            return _Row(line, "error", f"no teacher account with email {r['teacher_email']} at this school")
    key = ("allocation", section.id, subject.id)
    if key in ctx.planned:
        return _Row(line, "error", f"{subject.name} in {label} is in this file twice")
    ctx.planned.add(key)
    current = ctx.cs.allocation_for(section.id, subject.id)
    # The cell's co-teacher and double periods (set on the Timetable page)
    # are kept; a row that would break them is an error here, in the dry
    # run, not a refusal halfway through the apply.
    if current and current.co_teacher_id:
        if teacher is None:
            return _Row(line, "error", f"{subject.name} in {label} has a co-teacher; name its teacher too, "
                                       "or clear the co-teacher on the Timetable page first")
        if teacher.id == current.co_teacher_id:
            return _Row(line, "error", f"{teacher.name} is already the co-teacher of {subject.name} in {label}; "
                                       "clear the co-teacher on the Timetable page first")
    if current and 2 * current.double_periods > periods:
        return _Row(line, "error", f"{subject.name} in {label} has {current.double_periods} double period(s), "
                                   f"which take {2 * current.double_periods} periods a week")
    if current and current.teacher_id == (teacher.id if teacher else None) and current.periods_per_week == periods:
        return _Row(line, "skip", f"{subject.name} in {label} is already set that way")
    who = teacher.name if teacher else "no teacher yet"

    def apply() -> None:
        before, after = ctx.cs.set_allocation(section_id=section.id, subject_id=subject.id,
                                              teacher_id=teacher.id if teacher else None, periods_per_week=periods)
        if before is not None and before.teacher_id != after.teacher_id:
            # SCH-8: the cover and the books follow a mid-year change, as on the Timetable page.
            ctx.cs.follow_teacher_change(section.id, subject.id, before.teacher_id, after.teacher_id,
                                         today=cr._school_today().isoformat())
    return _Row(line, "ok", f"{subject.name} in {label}: {who}, {periods} periods a week", apply)


def _plan_holidays(ctx: _Ctx, line: int, r: dict) -> _Row:
    from ..curriculum.calendar import HOLIDAY_KINDS
    cal = ctx.cs.get_calendar_for_year(ctx.year.id)
    if cal is None:
        return _Row(line, "error", "this academic year has no calendar yet; set it up first")
    try:
        start = date.fromisoformat(r.get("date", "")).isoformat()
        end = date.fromisoformat(r["end_date"]).isoformat() if r.get("end_date") else None
    except ValueError:
        return _Row(line, "error", "dates must be YYYY-MM-DD")
    if end is not None and end < start:
        return _Row(line, "error", "end_date is before date")
    if not (ctx.year.start_date <= start <= ctx.year.end_date) or (end and end > ctx.year.end_date):
        return _Row(line, "error", f"outside the academic year {ctx.year.start_date} to {ctx.year.end_date}")
    kind = (r.get("kind") or "holiday").strip().lower()
    if kind not in HOLIDAY_KINDS:
        return _Row(line, "error", f"kind must be one of {', '.join(HOLIDAY_KINDS)}")
    name = r.get("name", "").strip()
    if any(h.date == start and h.label.lower() == name.lower() for h in ctx.cs.holidays_for_calendar(cal.id)) \
            or ("holiday", start, name.lower()) in ctx.planned:
        return _Row(line, "skip", f"{name} on {start} is already on the calendar")
    ctx.planned.add(("holiday", start, name.lower()))
    # The holiday route's own path: lessons on the days move, and who is affected hears (N-8-6).
    return _Row(line, "ok", f"{name}: {start}" + (f" to {end}" if end else ""),
                lambda: cr.declare_holiday(ctx.cs, ctx.year, cal, date=start, label=name, kind=kind,
                                           end_date=end, principal=ctx.principal) and None)


PLANNERS = {"sections": _plan_sections, "teachers": _plan_teachers, "students": _plan_students,
            "parents": _plan_parents, "allocations": _plan_allocations, "holidays": _plan_holidays}


def _year(academic_year_id: Optional[str], principal: User):
    cs = cr._require()
    if academic_year_id:
        return cr._require_school_owns_academic_year(academic_year_id, principal)
    years = cs.academic_years_for_school(principal.school_id)
    if not years:
        raise HTTPException(409, "set up an academic year first")
    return sorted(years, key=lambda y: y.start_date)[-1]


def _kind(kind: str) -> str:
    if kind not in COLUMNS:
        raise HTTPException(404, f"no import of {kind!r}; one of {', '.join(COLUMNS)}")
    return kind


@router.post("/import/{kind}", response_model=ImportResponse)
def bulk_import(kind: str, req: ImportRequest, dry_run: bool = Query(default=True, alias="dryRun"),
                principal: User = Depends(require_admin("users"))) -> ImportResponse:
    """Plan (and with dryRun=false, apply) an import from CSV or an Excel
    workbook. The year is the one named, else the school's latest."""
    from ..assessment.audit_log import get_audit_log
    _kind(kind)
    ctx = _Ctx(principal=principal, year=_year(req.academic_year_id, principal), cs=cr._require(),
               users=cr._require_users(), ops=store())
    ctx.grades = {g.number: g for g in ctx.cs.grades_for_year(ctx.year.id)}
    plan = [PLANNERS[kind](ctx, line, row) for line, row in _parse(kind, _file_rows(req))]
    counts = {s: sum(1 for p in plan if p.status == s) for s in ("ok", "skip", "error")}
    applied = False
    if not dry_run:
        if counts["error"]:
            raise HTTPException(422, {"message": f"{counts['error']} row(s) have errors; nothing was imported",
                                      "rows": [RowOutcome(row=p.line, status=p.status, message=p.message).model_dump(
                                          by_alias=True) for p in plan if p.status == "error"]})
        done = 0
        for p in plan:
            if p.apply is None:
                continue
            try:
                p.code = p.apply()
            except (ValueError, KeyError) as e:
                get_audit_log(cr._cfg.data_root).append(
                    "bulk_import", actor=principal.id,
                    details={"schoolId": principal.school_id, "kind": kind, "applied": done, "failedRow": p.line,
                             "error": str(e)})
                raise HTTPException(409, f"row {p.line} could not be applied ({e}); rows before it were "
                                         f"imported ({done}). Fix it and import the file again -- done rows "
                                         "will show as skipped.")
            done += 1
        applied = True
        get_audit_log(cr._cfg.data_root).append(
            "bulk_import", actor=principal.id,
            details={"schoolId": principal.school_id, "kind": kind, "academicYearId": ctx.year.id,
                     "applied": done, "skipped": counts["skip"]})
    return ImportResponse(kind=kind, dry_run=dry_run, applied=applied, counts=counts,
                          rows=[RowOutcome(row=p.line, status=p.status, message=p.message,
                                           invite_code=p.code if isinstance(p.code, str) else None) for p in plan])


def _csv(header: list[str], rows: list[list[Any]], filename: str, fmt: str = "csv") -> Response:
    if fmt == "xlsx":
        from .xlsx import workbook
        return Response(workbook([(filename.rsplit(".", 1)[0], [header] + rows)]),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{filename.rsplit(".", 1)[0]}.xlsx"'})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows([["" if v is None else v for v in r] for r in rows])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/import/{kind}/template", response_class=PlainTextResponse)
def import_template(kind: str, format: Literal["csv", "xlsx"] = "csv",
                    principal: User = Depends(require_admin("users"))) -> Response:
    """The header for an import, with one example row, as CSV or Excel."""
    return _csv(COLUMNS[_kind(kind)], [EXAMPLES[kind]], f"{kind}-template.csv", format)


EXPORTS = ("sections", "people", "allocations", "holidays", "timetable")


@router.get("/export/{kind}", response_class=PlainTextResponse)
def export(kind: str, academic_year_id: Optional[str] = Query(default=None, alias="academicYearId"),
           format: Literal["csv", "xlsx"] = "csv",
           principal: User = Depends(require_admin("users"))) -> Response:
    """The school's data as CSV or Excel, in the import's own columns where
    there is an import for it, so an export edited in a spreadsheet imports
    back."""
    if kind not in EXPORTS:
        raise HTTPException(404, f"no export of {kind!r}; one of {', '.join(EXPORTS)}")
    cs, users = cr._require(), cr._require_users()
    year = _year(academic_year_id, principal)
    email = {u.id: u.email for u in users.users_for_school(principal.school_id)}
    sections = {s.id: s for g in cs.grades_for_year(year.id) for s in cs.sections_for_grade(g.id)}
    grade_no = {g.id: g.number for g in cs.grades_for_year(year.id)}
    if kind == "sections":
        rows = [[grade_no[s.grade_id], s.name, email.get(getattr(s, "class_teacher_id", None) or "", "")]
                for s in sorted(sections.values(), key=lambda s: (grade_no[s.grade_id], s.name))]
        return _csv(COLUMNS["sections"], rows, "sections.csv", format)
    if kind == "people":
        rows = []
        for u in sorted(users.users_for_school(principal.school_id), key=lambda u: (u.role, u.name.lower())):
            e = cs.enrollment_for_student(u.id) if u.role == "student" else None
            s = sections.get(e.section_id) if e and e.section_id else None
            rows.append([u.name, u.email, u.role, grade_no.get(s.grade_id) if s else "", s.name if s else ""])
        return _csv(["name", "email", "role", "grade", "section"], rows, "people.csv", format)
    if kind == "allocations":
        subjects = {}
        rows = []
        for a in cs.allocations_for_year(year.id):
            s = sections.get(a.section_id)
            if s is None:
                continue
            subj = subjects.get(a.subject_id) or cs.get_subject(a.subject_id)
            subjects[a.subject_id] = subj
            rows.append([grade_no[s.grade_id], s.name, subj.name if subj else a.subject_id,
                         email.get(a.teacher_id or "", ""), a.periods_per_week])
        return _csv(COLUMNS["allocations"], sorted(rows, key=lambda r: (r[0], r[1], r[2])), "allocations.csv", format)
    if kind == "holidays":
        cal = cs.get_calendar_for_year(year.id)
        hs = cs.holidays_for_calendar(cal.id) if cal else []
        return _csv(COLUMNS["holidays"], [[h.date, h.end_date, h.label, h.kind] for h in sorted(hs, key=lambda h: h.date)],
                    "holidays.csv", format)
    rooms = {r.id: r.name for r in cs.rooms_for_school(principal.school_id)}
    names = {}
    rows = []
    for e in cs.timetable_for_year(year.id):
        s = sections.get(e.section_id)
        if s is None:
            continue
        subj = names.get(e.subject_id) or cs.get_subject(e.subject_id)
        names[e.subject_id] = subj
        rows.append([grade_no[s.grade_id], s.name, e.day_of_week + 1, e.period, subj.name if subj else e.subject_id,
                     email.get(e.teacher_id or "", ""), rooms.get(e.room_id or "", "")])
    return _csv(["grade", "section", "day", "period", "subject", "teacher_email", "room"],
                sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3])), "timetable.csv", format)
