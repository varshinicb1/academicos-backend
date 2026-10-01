"""Erasing a student's or a parent's personal data on the school's request
(NFR-1; DPDP Act 2023; docs/dpa-template.md sections 6.2, 6.3 and 9).

Until this module no route deleted a person's data: closing an account
(N-67-8) stops the sign-in and keeps everything, and withdrawing consent only
stops processing. The DPA template promises deletion within 30 days of the
school's written request, "including the scanned answer sheet images,
OCR-extracted text, and derived marks/mastery data"; that was done by nobody.

Who: the principal, for a student or a parent of their own school. The school
is the data fiduciary and receives the request (from the parent, the student
or itself); the principal records whose request it was and the day it arrived,
and confirms by typing the person's name. A delegated users admin can close an
account but not erase one: erasure cannot be undone. Staff accounts are closed,
not erased: the papers they set, the marks they entered and the cover they
gave are the school's records.

How, in order:
  1. the account is closed (assessment/users.py), so the person is signed out
     and nothing new is recorded while the rest runs;
  2. each part below deletes what one store holds of the person, files before
     the rows that name them;
  3. the operations and curriculum snapshots are uploaded at once, so a
     restart cannot restore the deleted rows from the blob store;
  4. only if every part succeeded, the account row itself goes, its name and
     email with it.
A part that fails does not stop the others, except those that would delete
what a rerun of it needs to find (`Part.needs`): those are skipped. The
account stays (closed) and the route answers 503 naming what is left, so the
principal runs it again; every part deletes only what is still there.

GET previews the same parts with counts and changes nothing.

What stays, and why, is KEPT: chiefly the audit log, which is append-only and
hash-chained (assessment/audit_log.py) and kept under CERT-In's log-retention
direction; the erasure is itself recorded there, with counts and ids, never
the person's name or email. Database backups and blob versions keep a copy
until they expire (docs/compliance.md, "Erasure on request").
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException

from ..config import Config
from ..curriculum.schemas import Camel, CamelRequest
from .auth_routes import require_principal
from .users import CLOSED_PREFIX, User, is_closed

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth/users")

_cfg: Optional[Config] = None
_events = None  # the EventStore api/main.py opened (learner_events)

ERASABLE_ROLES = ("student", "parent")

KEPT = (
    ("Audit log", "Every entry about this person, including the erasure itself. The log is append-only and "
                  "hash-chained, and CERT-In's directions require it kept. Its entries name the person by id; "
                  "some kinds also carry a name or email: a consent recorded (the guardian's name), a parent "
                  "invited (their email), an account closed or reopened (the name)."),
    ("Parental consent a parent gave", "Erasing a parent keeps the consent they gave for a child: it is the "
                                       "child's record, and the basis for processing the child's work."),
    ("Backups", "Cloud SQL backups and blob versions keep a copy until they expire."),
    ("Files outside AcademicOS", "Papers, report cards and exports the school downloaded or printed."),
)


# Every Cloud SQL table (deploy/gcp/schema.sql) and what erasure does with it:
# the part that deletes the person's rows, or why it holds none of them.
# tests/test_erasure.py fails on a table in neither, as operations/forget.py's
# ERASED/KEPT do for the operations store's tables.
DURABLE_ERASED_BY = {
    "users": "account", "sessions": "account", "invites": "invites",
    "parental_consents": "consent", "graded_evaluations": "marked_sheets", "scan_sessions": "scans",
    "practice_sets": "remediation_practice", "learner_models": "learner_model",
    "learner_events": "learner_events",
}
DURABLE_KEPT = {
    "audit_log": "kept by law; see KEPT",
    "assessments": "a paper's record: its teacher and a count of students, no student's data",
    "papers": "generated question papers",
    "school_templates": "the school's paper templates",
}


def init(cfg: Config, *, events=None) -> None:
    global _cfg, _events
    _cfg = cfg
    _events = events


# ---------------- the parts ----------------


@dataclass
class Target:
    user: User
    role: str                         # the role it had, "left:" removed
    email: str                        # lower case
    invite_codes: list[str] = field(default_factory=list)


Count = Union[int, dict]


@dataclass
class Part:
    key: str
    label: str
    roles: tuple[str, ...]
    run: Callable[[Target, bool], Count]
    # Parts that must have finished first: this one deletes what a rerun of
    # them needs to find (the photo rows name the photo files; the invite
    # codes find the pending enrollments). If one failed, this part is
    # skipped and reported, so the rerun still has everything to finish with.
    needs: tuple[str, ...] = ()


def _users():
    from . import auth_routes
    return auth_routes._require()


def _ops():
    from ..operations import routes as ops
    return ops.store()


def _curriculum():
    from ..curriculum import routes as cr
    return cr._require()


def _pillar():
    from . import pillar_routes
    return pillar_routes


def _consent(t: Target, dry: bool) -> int:
    from .consent import get_consent_store
    return get_consent_store(_cfg.data_root).erase_student(t.user.school_id, t.user.id, dry_run=dry)


def _enrollment(t: Target, dry: bool) -> int:
    return _curriculum().erase_student_enrollment(t.user.id, dry_run=dry)


def _homework_photo_files(t: Target, dry: bool) -> int:
    """The photo files go before the rows that name them, so a failure here
    leaves the rows for the next run to find."""
    from ..operations import homework_photos as hp
    photos = _ops().photos_of_student(t.user.id)
    if dry:
        return len(photos)
    blobs = hp._blobs()
    media = Path(_cfg.data_root) / "homework-media"
    for p in photos:
        if p["blob_key"] and blobs.enabled:
            blobs.delete(p["blob_key"])
        (media / p["homework_id"] / t.user.id / f"{p['id']}.{hp.TYPES.get(p['content_type'], 'jpg')}").unlink(
            missing_ok=True)
    for folder in media.glob(f"*/{t.user.id}"):
        shutil.rmtree(folder, ignore_errors=True)
    return len(photos)


def _school_records(t: Target, dry: bool) -> dict:
    if t.role == "student":
        return _ops().forget_student(t.user.id, school_id=t.user.school_id, email=t.email, name=t.user.name,
                                     invite_codes=t.invite_codes, dry_run=dry)
    return _ops().forget_parent(t.user.id, email=t.email, invite_codes=t.invite_codes, dry_run=dry)


def _graded(t: Target, dry: bool) -> int:
    return _pillar()._require_graded().erase_student(t.user.id, dry_run=dry)


def _scans(t: Target, dry: bool) -> int:
    """Scanned answer sheets: the page photos, the text read from them, the
    PDFs made from them (in the blob store and on disk) and the session."""
    from . import mobile_scan as ms
    if ms._store is None:
        return 0
    sessions = ms._store.payloads_of_student(t.user.id)
    if dry:
        return len(sessions)
    exports = Path(_cfg.data_root) / "exports"
    for s in sessions:
        sid = s["id"]
        if ms._storage.enabled:
            keys = {k for k in (s.get("raw_pdf_storage_key"), s.get("corrected_pdf_storage_key")) if k}
            for page in s.get("pages") or []:
                keys.update(k for k in (page.get("raw_storage_key"), page.get("processed_storage_key")) if k)
            if hasattr(ms._storage, "keys"):   # GCS: also what no saved key names
                keys.update(ms._storage.keys(f"{sid}/"))
            for key in sorted(keys):
                ms._storage.delete(key)
        shutil.rmtree(ms._WORKDIR_ROOT / sid, ignore_errors=True)
        for name in (f"{sid}_raw.pdf", f"{sid}_corrected.pdf"):
            (exports / name).unlink(missing_ok=True)
        for path in (s.get("raw_pdf_path"), s.get("corrected_pdf_path")):
            if path:
                Path(path).unlink(missing_ok=True)
        ms._store.delete(sid)
        ms._sessions.pop(sid, None)
        ms._session_school.pop(sid, None)
    return len(sessions)


def _remediation(t: Target, dry: bool) -> int:
    practice = _pillar()._practice
    return practice.erase_student(t.user.id, dry_run=dry) if practice is not None else 0


def _learner_model(t: Target, dry: bool) -> int:
    _, knowledge, _ = _pillar()._require()
    return knowledge.erase(t.user.id, dry_run=dry)


def _learner_events(t: Target, dry: bool) -> int:
    return _events.erase_learner(t.user.id, dry_run=dry) if _events is not None else 0


def _report_files(t: Target, dry: bool) -> int:
    """Progress reports rendered to disk (report_pdf.py names them by id)."""
    safe = "".join(c for c in t.user.id if c.isalnum() or c in "-_")
    files = list((Path(_cfg.data_root) / "exports").glob(f"progress_{safe}_*.pdf")) if safe else []
    if not dry:
        for f in files:
            f.unlink(missing_ok=True)
    return len(files)


def _invites(t: Target, dry: bool) -> int:
    return len(t.invite_codes) if dry else _users().erase_invites(t.user)


S, P = ("student",), ("student", "parent")
PARTS = (
    Part("consent", "Parental consent record", S, _consent),
    Part("class", "Class enrollment", S, _enrollment),
    Part("homework_photo_files", "Photos of written homework", S, _homework_photo_files),
    Part("school_records", "Homework, marks, progress, practice, parent links, notifications and settings",
         P, _school_records, needs=("homework_photo_files",)),
    Part("marked_sheets", "Marked answer sheets", S, _graded),
    Part("scans", "Scanned answer sheets, the text read from them and their PDFs", S, _scans),
    Part("remediation_practice", "Practice made from marked work", S, _remediation),
    Part("learner_model", "Mastery model", S, _learner_model),
    Part("learner_events", "Learning event log", S, _learner_events),
    Part("report_files", "Progress report PDFs", S, _report_files),
    Part("invites", "Invites sent to the person", P, _invites, needs=("school_records",)),
)


def _target(user: User) -> Target:
    role = user.role[len(CLOSED_PREFIX):] if is_closed(user.role) else user.role
    t = Target(user=user, role=role, email=user.email.strip().lower())
    t.invite_codes = [i.code for i in _users().invites_of(user)]
    return t


def _parts_for(t: Target) -> list[Part]:
    return [p for p in PARTS if t.role in p.roles]


def _total(c: Count) -> int:
    return sum(c.values()) if isinstance(c, dict) else int(c)


# ---------------- routes ----------------


class ErasurePart(Camel):
    part: str
    label: str
    count: int
    tables: Optional[dict[str, int]] = None
    error: Optional[str] = None


class ErasureKept(Camel):
    what: str
    why: str


class ErasurePreview(Camel):
    user_id: str
    name: str
    role: str
    closed: bool
    parts: list[ErasurePart]
    kept: list[ErasureKept]


class ErasureRequest(CamelRequest):
    # Whose written request the school received, and the day it arrived.
    requested_by: Literal["parent", "student", "school"]
    request_received_on: date
    # The person's name as the school knows it: erasing the wrong id cannot be undone.
    confirm_name: str


class ErasureResult(Camel):
    user_id: str
    role: str
    erased_at: str
    parts: list[ErasurePart]
    kept: list[ErasureKept]


def _kept() -> list[ErasureKept]:
    return [ErasureKept(what=w, why=y) for w, y in KEPT]


def _erasable(user_id: str, principal: User) -> User:
    if _cfg is None:
        raise HTTPException(503, "erasure is not initialised")
    user = _users().get(user_id)
    if user is None or user.school_id != principal.school_id:
        raise HTTPException(404, "no such account in your school")
    role = user.role[len(CLOSED_PREFIX):] if is_closed(user.role) else user.role
    if role not in ERASABLE_ROLES:
        raise HTTPException(422, "only a student's or a parent's data is erased; a member of staff who left "
                                 "has their account closed, and the school keeps the records they made")
    return user


def _same_name(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


@router.get("/{user_id}/erasure", response_model=ErasurePreview)
def preview_erasure(user_id: str, principal: User = Depends(require_principal)) -> ErasurePreview:
    """What erasing this student or parent would delete, part by part, and
    what would stay. Changes nothing."""
    from .audit_log import get_audit_log, record_pii_read
    user = _erasable(user_id, principal)
    t = _target(user)
    # A read of the person's data (their name, what is held of them): logged
    # like every other (docs/compliance.md, the student-PII box).
    record_pii_read(get_audit_log(_cfg.data_root), actor=principal.id, what="erasure_preview",
                    student_id=user.id if t.role == "student" else None, userId=user.id)
    parts = []
    for p in _parts_for(t):
        c = p.run(t, True)
        parts.append(ErasurePart(part=p.key, label=p.label, count=_total(c),
                                 tables=c if isinstance(c, dict) else None))
    parts.append(ErasurePart(part="account", label="The account: name, email, password and sign-ins", count=1))
    return ErasurePreview(user_id=user.id, name=user.name, role=t.role, closed=is_closed(user.role),
                          parts=parts, kept=_kept())


@router.post("/{user_id}/erasure", response_model=ErasureResult)
def erase(user_id: str, body: ErasureRequest, principal: User = Depends(require_principal)) -> ErasureResult:
    """Erase a student's or a parent's personal data on the school's written
    request. Cannot be undone. See the module docstring for the order and
    for what stays."""
    from ..curriculum import routes as cr
    from .audit_log import get_audit_log
    user = _erasable(user_id, principal)
    if not _same_name(body.confirm_name, user.name):
        raise HTTPException(422, "type the person's name exactly as the school has it, to confirm whose data "
                                 "this is")
    if body.request_received_on > cr._school_today():
        raise HTTPException(422, "the request cannot have arrived after today")
    t = _target(user)
    if not is_closed(user.role):
        _users().close_account(user.id)
    parts: list[ErasurePart] = []
    unfinished: set[str] = set()
    for p in _parts_for(t):
        if any(n in unfinished for n in p.needs):
            unfinished.add(p.key)
            parts.append(ErasurePart(part=p.key, label=p.label, count=0, error="skipped"))
            continue
        try:
            c = p.run(t, False)
            parts.append(ErasurePart(part=p.key, label=p.label, count=_total(c),
                                     tables=c if isinstance(c, dict) else None))
        except Exception as exc:  # noqa: BLE001 - every other part still runs; this one is reported
            log.exception("erasure of %s: part %s failed", user.id, p.key)
            unfinished.add(p.key)
            parts.append(ErasurePart(part=p.key, label=p.label, count=0, error=type(exc).__name__))
    # Uploaded now: a restart before the debounced upload would restore the
    # deleted rows from the last snapshot.
    for name, store in (("operations", _ops), ("curriculum", _curriculum)):
        try:
            error = None if store().flush_snapshot() else "not uploaded yet"
        except Exception as exc:  # noqa: BLE001
            log.exception("erasure of %s: the %s snapshot did not upload", user.id, name)
            error = type(exc).__name__
        if error:
            parts.append(ErasurePart(part=f"snapshot_{name}", label=f"The {name} snapshot upload", count=0,
                                     error=error))
    failed = [p for p in parts if p.error]
    if not failed:
        try:
            _users().erase_account(user.id)
            parts.append(ErasurePart(part="account", label="The account: name, email, password and sign-ins",
                                     count=1))
        except Exception as exc:  # noqa: BLE001
            log.exception("erasure of %s: the account row was not deleted", user.id)
            failed.append(ErasurePart(part="account", label="The account", count=0, error=type(exc).__name__))
    erased_at = datetime.now(timezone.utc).isoformat()
    details = {"userId": user.id, "role": t.role, "requestedBy": body.requested_by,
               "requestReceivedOn": body.request_received_on.isoformat(),
               "erased": {p.part: p.count for p in parts if not p.error},
               "failed": sorted({p.part for p in failed})}
    get_audit_log(_cfg.data_root).append(
        "personal_data_erased" if not failed else "personal_data_erasure_incomplete",
        actor=principal.id, student_id=user.id if t.role == "student" else None, details=details)
    if failed:
        names = ", ".join(p.label.lower() for p in failed)
        raise HTTPException(503, f"the erasure is not finished: {names} could not be erased yet. Everything "
                                 "else was, and the account stays closed. Run the erasure again to finish it.")
    return ErasureResult(user_id=user.id, role=t.role, erased_at=erased_at, parts=parts, kept=_kept())
