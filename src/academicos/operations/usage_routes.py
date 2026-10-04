"""Per-school usage for the operator (NFR-7, "per-school usage metrics"),
2026-10-04: during a pilot, whether each school is actually using
AcademicOS -- its people by role, papers made, homework set, lessons marked
taught, when it last did anything, and whether it accepted the pilot
agreement.

Behind the operator's key (env ACOS_OPS_KEY, header X-Ops-Key), like
/ops/errors: 404 when no key is configured, 401 for a wrong one. Counts
only: no names, no emails, no student data. A school is listed once it has a
profile, an academic year or an accepted pilot agreement (the first thing a
new principal does).
"""
from __future__ import annotations

import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from ..assessment.schemas import Camel

router = APIRouter(prefix="/api/v1/ops")


class SchoolUsage(Camel):
    school_id: str
    name: Optional[str] = None
    principals: int
    teachers: int
    students: int
    parents: int
    papers_total: int
    papers_last_7_days: int
    papers_last_30_days: int
    homework_last_30_days: int
    lessons_marked_last_30_days: int
    last_activity: Optional[str] = None
    pilot_agreement_accepted_at: Optional[str] = None


def _check_key(given: Optional[str]) -> None:
    expected = os.environ.get("ACOS_OPS_KEY", "")
    if not expected:
        raise HTTPException(404, "Not Found")
    if not given or not hmac.compare_digest(given, expected):
        raise HTTPException(401, "a valid X-Ops-Key is required")


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


@router.get("/usage", response_model=list[SchoolUsage])
def usage(x_ops_key: Optional[str] = Header(default=None, alias="X-Ops-Key")) -> list[SchoolUsage]:
    """Each school's use of AcademicOS, most recently active first."""
    _check_key(x_ops_key)
    from ..assessment import routes as assessment_routes
    from ..curriculum import routes as cr
    from .routes import store

    cs, users, ops = cr._require(), cr._require_users(), store()
    _cfg, assessments = assessment_routes._require()
    now = datetime.now(timezone.utc)
    week, month = (now - timedelta(days=7)).isoformat(), (now - timedelta(days=30)).isoformat()

    schools = {r["school_id"] for r in cs._fetchall(
        "SELECT school_id FROM academic_years UNION SELECT school_id FROM school_profiles")}
    schools |= {r["school_id"] for r in ops._fetchall("SELECT school_id FROM pilot_agreements")}
    out = []
    for sid in sorted(s for s in schools if s):
        roles = [u.role for u in users.users_for_school(sid)]
        papers = [_iso(a.created_at) or "" for a in assessments.list_by_school(sid)]
        homework = ops._fetchone("SELECT COUNT(*) AS n, MAX(created_at) AS last FROM homework "
                                 "WHERE school_id=? AND created_at>=?", (sid, month)) or {}
        lessons = cs._fetchone("SELECT COUNT(*) AS n, MAX(completed_at) AS last FROM scheduled_lessons "
                               "WHERE school_id=? AND status='completed' AND completed_at>=?", (sid, month)) or {}
        profile = cs.get_school_profile(sid)
        pilot = ops.pilot_acceptance(sid)
        last = max([p for p in papers if p] + [x for x in (homework.get("last"), lessons.get("last"),
                                                           pilot and pilot["accepted_at"]) if x], default=None)
        out.append(SchoolUsage(
            school_id=sid, name=profile.name if profile else None,
            principals=roles.count("principal"), teachers=roles.count("teacher"),
            students=roles.count("student"), parents=roles.count("parent"),
            papers_total=len(papers), papers_last_7_days=sum(1 for p in papers if p >= week),
            papers_last_30_days=sum(1 for p in papers if p >= month), homework_last_30_days=int(homework.get("n") or 0),
            lessons_marked_last_30_days=int(lessons.get("n") or 0), last_activity=last,
            pilot_agreement_accepted_at=pilot["accepted_at"] if pilot else None))
    out.sort(key=lambda s: s.last_activity or "", reverse=True)
    return out
