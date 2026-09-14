"""Shared school-ownership checks for the assessment/pillar/mobile route
modules.

2026-09-15: extracted from pillar_routes.py's `_require_school_owns_assessment`
(the one place this pattern already existed) so routes.py and
mobile_routes.py can use the identical check instead of re-implementing it --
this was written the day 42 of 46 real-data routes across those three files
were found to have no authentication or ownership check at all (see
docs/deployment.md and AGENTS.md's security section). Every function here
follows the same convention already established by
`curriculum/routes.py`'s `_require_school_owns_unit` family and
`auth_routes.py`'s `get_current_user`/`require_principal`: 404 if the object
doesn't exist, 403 if it exists but belongs to a different school (never leak
existence vs. ownership through the status code chosen), 401 is handled
upstream by `Depends(get_current_user)` before any of these run.

Each function takes the relevant store as a plain parameter rather than
reaching for a module-level singleton -- routes.py, pillar_routes.py, and
mobile_routes.py each construct/own their own store instances (sometimes two
modules hold independent instances pointed at the same underlying file, e.g.
AssessmentStore in both routes.py and pillar_routes.py), so this stays a pure
function of its inputs instead of creating a new cross-module coupling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from .mobile_scan import ScanSession
    from .schemas import Assessment, GeneratedPaper
    from .store import AssessmentStore
    from .paper_store import PaperStore
    from .users import User, UserStore


def require_school_owns_assessment(
    assessment_store: "AssessmentStore", assessment_id: str, current: "User",
) -> "Assessment":
    a = assessment_store.get(assessment_id)
    if a is None:
        raise HTTPException(404, "assessment not found")
    if a.school_id != current.school_id:
        raise HTTPException(403, "this assessment belongs to a different school")
    return a


def require_school_owns_paper(
    paper_store: "PaperStore", assessment_store: "AssessmentStore",
    paper_id: str, current: "User",
) -> "GeneratedPaper":
    paper = paper_store.get(paper_id)
    if paper is None:
        raise HTTPException(404, "paper not found")
    # A paper's own school identity isn't stored on it directly (PaperStore
    # has no school_id column) -- it's resolved through the assessment it was
    # generated from, same as GradedStore/AuditLog resolve theirs through
    # assessment_id rather than carrying a redundant column.
    require_school_owns_assessment(assessment_store, paper.assessment_id, current)
    return paper


def require_school_owns_scan_session(
    session: "ScanSession", assessment_store: "AssessmentStore", current: "User",
) -> "ScanSession":
    """Takes an already-fetched ScanSession (mobile_scan.py's own get_session
    raises its own ScanError on a missing id, so callers fetch first and pass
    the result in here rather than this function doing a second, differently-
    shaped lookup)."""
    require_school_owns_assessment(assessment_store, session.assessment_id, current)
    return session


def require_school_owns_student(
    user_store: "UserStore", student_id: str, current: "User",
) -> "User":
    student = user_store.get(student_id)
    if student is None:
        raise HTTPException(404, "student not found")
    if student.school_id != current.school_id:
        raise HTTPException(403, "this student belongs to a different school")
    if current.role == "student" and current.id != student_id:
        raise HTTPException(403, "students can only access their own data")
    return student


def require_own_school(school_id: str, current: "User") -> None:
    """For routes keyed directly by a school_id path/query param rather than
    by an object that needs a lookup -- e.g. GET /schools/{school_id}/...
    No 404 case here (the school_id itself isn't a row to be missing), just
    403 if it isn't the caller's own school."""
    if school_id != current.school_id:
        raise HTTPException(403, "this action is scoped to a different school")
