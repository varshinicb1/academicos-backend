"""Shared school-ownership checks for the assessment/pillar/mobile route
modules, and for curriculum/routes.py where a check spans both packages.

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

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .consent import ConsentStore
    from ..curriculum.store import CurriculumStore
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


PAPER_CHANGE_DETAIL = ("only the paper's author, the principal or an exams admin can change this paper; "
                       "ask its author to make the change")
PAPER_READ_DETAIL = ("an exam paper and its answer key are downloaded only by the paper's author, its "
                     "reviewer, the principal or an exams admin")


def may_change_paper(assessment: "Assessment", current: "User") -> bool:
    """The paper's author, the principal, or a teacher holding the exams grant.

    Until 2026-10-01 the paper routes checked only that the caller was staff
    of the paper's school: QA found any teacher could swap or remove a
    colleague's questions and download the colleague's answer key before the
    exam (P-09)."""
    from .auth_routes import holds
    if assessment.teacher_id and assessment.teacher_id == current.id:
        return True
    try:
        grade: Optional[int] = int(assessment.grade)
    except (TypeError, ValueError):
        grade = None
    return holds(current, "exams", grade=grade)


def require_may_change_paper(assessment: "Assessment", current: "User") -> None:
    if not may_change_paper(assessment, current):
        raise HTTPException(403, PAPER_CHANGE_DETAIL)


# From here on the exam has been sat: a colleague marking a section's scripts
# needs the key, and there is nothing left to leak.
AFTER_EXAM = frozenset({"conducted", "scanning", "scanned", "evaluating", "evaluated",
                        "teacherReviewed", "reportsGenerated", "remediationSent", "archived"})


def exam_held(assessment: "Assessment") -> bool:
    """The paper's status says the exam was sat, or the date it was
    scheduled for (the planner's) has passed."""
    if assessment.status in AFTER_EXAM:
        return True
    when = assessment.scheduled_at
    if when is None:
        return False
    now = datetime.now(timezone.utc) if when.tzinfo else datetime.now()
    return when < now


def may_download_paper(assessment: "Assessment", current: "User") -> bool:
    """Who may take a paper or its answer key off the system: those who may
    change it, a colleague asked to review it (they read the files), and,
    once the exam has been held, any staff member of the school."""
    if may_change_paper(assessment, current) or exam_held(assessment):
        return True
    try:
        from ..operations import routes as ops
        return any(r["reviewer_id"] == current.id for r in ops.store().reviews_for(assessment.id))
    except HTTPException:
        return False


def require_may_download_paper(assessment: "Assessment", current: "User") -> None:
    if not may_download_paper(assessment, current):
        raise HTTPException(403, PAPER_READ_DETAIL)


def without_answer_key(paper: "GeneratedPaper") -> "GeneratedPaper":
    """The paper as a colleague may read it on screen: its questions, not its
    answers (GET /papers/{id} returned the key to every teacher, P-09)."""
    return paper.model_copy(update={
        "answer_key": {},
        "sets": [s.model_copy(update={"answer_key": {}}) for s in paper.sets],
    })


def papers_visible_to(assessments: "Iterable[Assessment]", current: "User") -> "list[Assessment]":
    """The school's papers a staff member has a reason to see: the principal
    sees all; a teacher, the ones they wrote, review, administer (the exams
    grant for its class) or teach the subject of in its class -- the last so
    a section teacher still finds a colleague's common paper to enter marks
    on and scan. Every teacher saw every colleague's papers (QA P-09)."""
    rows = list(assessments)
    if current.role == "principal":
        return rows
    try:
        from .marks_routes import teaching_cells
        cells = teaching_cells(current)
    except HTTPException:
        cells = set()
    try:
        from ..operations import routes as ops
        reviewing = {r["assessment_id"] for r in
                     ops.store().reviews_where(current.school_id, reviewer_id=current.id, state=None)}
    except HTTPException:
        reviewing = set()

    def teaches(a: "Assessment") -> bool:
        try:
            return (int(a.grade), str(a.subject).casefold()) in cells
        except (TypeError, ValueError):
            return False
    return [a for a in rows if a.id in reviewing or may_change_paper(a, current) or teaches(a)]


def paper_permissions(assessments: "Iterable[Assessment]", current: "User") -> "dict[str, dict[str, bool]]":
    """What `current` may do with each paper, by the rules the routes
    enforce, so a screen offers only those: the re-test found Download
    buttons and an editable marks grid that could only answer 403 (QA R-01,
    R-04). Reads the caller's teaching cells and reviews once for the lot.

    change -- swap, pick, remove, rename (may_change_paper);
    download -- the paper, the Word file, the key (may_download_paper);
    enterMarks -- the marks grid (marks_routes.may_enter_marks);
    delete -- its author or the principal (routes.delete_assessment)."""
    from .auth_routes import holds
    rows = list(assessments)
    principal = current.role == "principal"
    try:
        from .marks_routes import teaching_cells
        cells = set() if principal else teaching_cells(current)
    except HTTPException:
        cells = set()
    try:
        from ..operations import routes as ops
        reviewing = {r["assessment_id"] for r in
                     ops.store().reviews_where(current.school_id, reviewer_id=current.id, state=None)}
    except HTTPException:
        reviewing = set()
    out: dict[str, dict[str, bool]] = {}
    for a in rows:
        author = bool(a.teacher_id) and a.teacher_id == current.id
        try:
            grade: Optional[int] = int(a.grade)
        except (TypeError, ValueError):
            grade = None
        change = author or holds(current, "exams", grade=grade)
        teaches = grade is not None and (grade, str(a.subject).casefold()) in cells
        out[a.id] = {
            "change": change,
            "download": change or a.id in reviewing or exam_held(a),
            "enterMarks": change or teaches,
            "delete": author or principal,
        }
    return out


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
    if current.role not in ("teacher", "principal") and current.id != student_id:
        raise HTTPException(403, "students can only access their own data")
    return student


def require_own_subtopics(
    curriculum_store: "CurriculumStore", subtopic_ids: list[str], current: "User",
) -> None:
    """For the two routes that take a caller-picked list of curriculum
    subtopic ids and answer with the question ids tagged to them:
    curriculum/routes.py's POST /curriculum/questions/by-subtopics and
    routes.py's POST /questions/search (subtopic_ids). The second is the one
    the web client sends (assessment_create_page.dart ->
    QuestionSearchParams.subtopicIds); it had no school check until
    2026-09-21, so a school_2 caller holding a school_1 subtopic id got the
    question ids school_1 had tagged to it. One helper so the two cannot
    drift apart again.

    403 if any id belongs to another school, even mixed in with the caller's
    own -- dropping it silently would hand back a paper built from a
    different selection than the one the teacher made. An id that resolves
    to no school is not refused: it is a subtopic force-deleted since the
    picker loaded, its question links went with it, and it matches nothing.
    That is why there is no 404 case here, unlike the functions above."""
    owners = curriculum_store.school_ids_for_subtopics(subtopic_ids)
    if any(owner != current.school_id for owner in owners.values()):
        raise HTTPException(403, "this subtopic belongs to a different school")


def require_own_school(school_id: str, current: "User") -> None:
    """For routes keyed directly by a school_id path/query param rather than
    by an object that needs a lookup -- e.g. GET /schools/{school_id}/...
    No 404 case here (the school_id itself isn't a row to be missing), just
    403 if it isn't the caller's own school."""
    if school_id != current.school_id:
        raise HTTPException(403, "this action is scoped to a different school")


def student_names(student_ids: "Iterable[str]") -> "list[str]":
    """Each student's name, for a message a teacher reads: the refusal named
    internal ids ("user_94019f5d5c24"), so nobody could tell which child was
    missing consent (v3 audit N-4-5). An id stays when no account is found."""
    from . import auth_routes
    users = auth_routes._users
    out = []
    for sid in student_ids:
        u = users.get(sid) if users is not None else None
        out.append(u.name if u is not None else sid)
    return out


def _consent_refusal(student_ids: "list[str]") -> HTTPException:
    names = student_names(student_ids)
    who = f"student {names[0]}" if len(names) == 1 else "students " + ", ".join(names)
    return HTTPException(
        409, f"no recorded parental consent for {who}; record it under Classes "
             "before grading")


# What a student (or a parent submitting for them) reads instead: the staff
# wording told them to "record it under Classes", which they cannot do (N-5-1).
OWN_CONSENT_MESSAGE = ("Your school needs a parent's consent before your work can be saved and marked. "
                       "Ask your parent to give it in the app, or ask your class teacher.")


def require_own_consent(consent_store: "ConsentStore", school_id: str, student_id: str) -> None:
    """require_consent, worded for the student whose work it is."""
    if students_without_consent(consent_store, school_id, [student_id]):
        raise HTTPException(409, OWN_CONSENT_MESSAGE)


def students_without_consent(
    consent_store: "ConsentStore", school_id: str, student_ids: "Iterable[str]",
) -> list[str]:
    """The ids in `student_ids`, in order and without repeats, that have no
    ACTIVE consent on file for `school_id`. A revoked record is kept (the
    revocation is part of the trail) with status "revoked", and only
    "granted" reads as consent, so a withdrawal counts as none. A record
    whose status cannot be read raises ValueError in consent.py; that is
    counted as missing too -- "I cannot tell" is "no" (the same fail-closed
    reading consent.from_row makes) -- and LOGGED, because the teacher's 409
    then says "record it under Classes" for a student whose consent was
    recorded, and re-recording overwrites the bad row. The warning is the
    only evidence of a corrupt row, a partial remote payload or schema drift.

    One student is one keyed read. More than one reads the school's records
    once (ConsentStore.granted_student_ids): per-student reads with the
    durable table on are one remote select each, so GET /consent/missing
    with its allowed 500 ids was 500 sequential round trips in one request.
    The batch read fails closed per row, so the two paths give the same
    answer (tests/test_consent_enforced.py checks both on revoked, blank and
    other-school rows)."""
    ids = list(dict.fromkeys(student_ids))
    if len(ids) > 1:
        granted_ids = consent_store.granted_student_ids(school_id)
        return [sid for sid in ids if sid not in granted_ids]
    missing: list[str] = []
    for sid in ids:
        try:
            granted = consent_store.has_consent(school_id, sid)
        except (ValueError, KeyError):
            logger.warning("unreadable consent record for %s at %s; treating as missing",
                           sid, school_id, exc_info=True)
            granted = False
        if not granted:
            missing.append(sid)
    return missing


def require_consent(consent_store: "ConsentStore", school_id: str, student_id: str) -> None:
    """409 unless the student has active parental consent at `school_id`.

    DPDP Act 2023 s.9: a minor's personal data is processed only with
    verifiable parental consent, and every CBSE student is a minor. Until
    2026-09-22 consent was recorded (consent_routes.py) but nothing read it;
    the audit (item 8.5) found grading and scanning proceeded without it.
    Every route that processes a named student's work -- evaluating, reviewing,
    finalizing, awarding, scanning, practice -- calls this before it does, and
    after its own 401/403/404 checks, so a caller from another school still
    learns nothing about this school's consent records.

    409, not 403: the caller is allowed to grade; the student's record is in
    a state that forbids it until a staff member records consent, which is the
    usual meaning of Conflict and what the web client shows verbatim.

    Reads are not gated. A teacher looking at marks already stored is not
    processing new data, and withdrawal must not hide what was graded while
    consent was in force (docs/compliance.md)."""
    if students_without_consent(consent_store, school_id, [student_id]):
        raise _consent_refusal([student_id])


def require_consent_for_all(
    consent_store: "ConsentStore", school_id: str, student_ids: "Iterable[str]",
) -> None:
    """require_consent for a batch, refusing the whole batch if any one
    student lacks consent, and naming every such student. Used by the bulk
    award, which must never land half-applied."""
    missing = students_without_consent(consent_store, school_id, student_ids)
    if missing:
        raise _consent_refusal(missing)
