"""HTTP routes for the school's staff rules and leave balances
(staff_policy.py). Staff read the rules (a teacher's leave form says what
will happen to a request); the principal, or a teacher with the leave
permission, sets them and sees every teacher's balance."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from ..assessment.auth_routes import require_admin, require_staff
from ..assessment.users import User
from . import routes as cr
from .schemas import Camel
from .staff_policy import StaffPolicy

router = APIRouter(prefix="/api/v1/curriculum")


class StaffPolicyBody(Camel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")
    auto_approve: bool = False
    max_auto_days: float = Field(default=2, ge=0.5, le=30)
    min_notice_days: int = Field(default=1, ge=0, le=30)
    annual_allowance_days: float = Field(default=12, ge=0, le=365)
    check_in_grace_minutes: int = Field(default=10, ge=0, le=180)
    check_in_reminder: bool = True


class StaffPolicyResponse(StaffPolicyBody):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    first_bell: Optional[str] = None
    updated_at: Optional[str] = None


class BalanceResponse(Camel):
    teacher_id: str
    name: str
    allowance: float
    used: float
    pending: float
    left: float


class MyBalanceResponse(BalanceResponse):
    # What will happen to a new request, for the leave form.
    auto_approve: bool
    max_auto_days: float
    min_notice_days: int


def _year(user: User):
    store = cr._require()
    today = cr._school_today().isoformat()
    year = next((y for y in store.academic_years_for_school(user.school_id)
                 if y.start_date <= today <= y.end_date), None)
    if year is None:
        raise HTTPException(404, "no academic year covers today; set up this year's classes first")
    return store, year


def _policy_response(store, school_id: str, year_id: Optional[str]) -> StaffPolicyResponse:
    p = store.staff_policy(school_id)
    return StaffPolicyResponse(auto_approve=p.auto_approve, max_auto_days=p.max_auto_days,
                               min_notice_days=p.min_notice_days, annual_allowance_days=p.annual_allowance_days,
                               check_in_grace_minutes=p.check_in_grace_minutes,
                               check_in_reminder=p.check_in_reminder,
                               first_bell=store.first_bell(year_id) if year_id else None, updated_at=p.updated_at)


@router.get("/staff-policy", response_model=StaffPolicyResponse)
def get_staff_policy(current: User = Depends(require_staff)) -> StaffPolicyResponse:
    """The school's leave and check-in rules."""
    store = cr._require()
    today = cr._school_today().isoformat()
    year = next((y for y in store.academic_years_for_school(current.school_id)
                 if y.start_date <= today <= y.end_date), None)
    return _policy_response(store, current.school_id, year.id if year else None)


@router.put("/staff-policy", response_model=StaffPolicyResponse)
def set_staff_policy(req: StaffPolicyBody, principal: User = Depends(require_admin("leave"))) -> StaffPolicyResponse:
    """Set the school's leave and check-in rules. Audited with before and after."""
    from ..assessment.audit_log import get_audit_log
    store = cr._require()
    before = store.staff_policy(principal.school_id)
    after = store.set_staff_policy(principal.school_id, StaffPolicy(**req.model_dump()), updated_by=principal.id)
    get_audit_log(cr._cfg.data_root).append(
        "staff_policy_set", actor=principal.id,
        details={"schoolId": principal.school_id, "before": {k: v for k, v in vars(before).items() if k not in (
            "updated_by", "updated_at")}, "after": req.model_dump()})
    today = cr._school_today().isoformat()
    year = next((y for y in store.academic_years_for_school(principal.school_id)
                 if y.start_date <= today <= y.end_date), None)
    return _policy_response(store, principal.school_id, year.id if year else None)


def _balance(store, user_id: str, name: str, school_id: str, year_id: str) -> BalanceResponse:
    b = store.leave_balance(school_id, user_id, year_id)
    return BalanceResponse(teacher_id=user_id, name=name, allowance=b.allowance, used=b.used, pending=b.pending,
                           left=b.left)


@router.get("/my-leave-balance", response_model=MyBalanceResponse)
def my_leave_balance(current: User = Depends(require_staff)) -> MyBalanceResponse:
    """The caller's leave this year, and what will happen to a new request."""
    store, year = _year(current)
    b = _balance(store, current.id, current.name, current.school_id, year.id)
    p = store.staff_policy(current.school_id)
    return MyBalanceResponse(**b.model_dump(), auto_approve=p.auto_approve, max_auto_days=p.max_auto_days,
                             min_notice_days=p.min_notice_days)


@router.get("/leave-balances", response_model=list[BalanceResponse])
def leave_balances(principal: User = Depends(require_admin("leave"))) -> list[BalanceResponse]:
    """Every teacher's leave this year, least left first."""
    store, year = _year(principal)
    staff = [u for u in cr._require_users().users_for_school(principal.school_id) if u.role == "teacher"]
    rows = [_balance(store, u.id, u.name, principal.school_id, year.id) for u in staff]
    return sorted(rows, key=lambda r: (r.left, r.name))
