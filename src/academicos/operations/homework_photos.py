"""Photos of written homework (SA-2): a student photographs their written
answers and attaches them to their submission; the teacher sees them beside
the typed answers and marks by hand.

A photo is kept on the container's disk and in the durable blob store
("homework-media", GCS on the GCP release), and read back from either.
JPEG, PNG or WebP, at most 5 MB each and 6 per submission. Uploading is
processing the student's work, so it needs parental consent; reading a
photo is logged as a read of student data when it is not the student's own.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from ..assessment.auth_routes import get_current_user, require_staff
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def store():
    """The operations store, looked up when called: operations/store.py
    imports this module for its table mixin, so importing routes here at load
    time would be circular."""
    from .routes import store as operations_store
    return operations_store()

PHOTOS_SCHEMA = """
CREATE TABLE IF NOT EXISTS homework_photos (
    id TEXT PRIMARY KEY,
    homework_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    blob_key TEXT,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_homework_photos ON homework_photos(homework_id, student_id);
"""
MAX_BYTES = 5 * 1024 * 1024
MAX_PHOTOS = 6
TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


class PhotosMixin:
    """Added to OperationsStore."""

    def add_photo(self, *, homework_id: str, student_id: str, content_type: str, size: int,
                  blob_key: Optional[str]) -> dict:
        pid = f"hwp_{uuid.uuid4().hex[:12]}"
        with self._conn_lock:
            self.conn.execute("INSERT INTO homework_photos (id, homework_id, student_id, content_type, size, blob_key,"
                              " uploaded_at) VALUES (?,?,?,?,?,?,?)",
                              (pid, homework_id, student_id, content_type, size, blob_key,
                               datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self._commit()
        return self.get_photo(pid)

    def get_photo(self, photo_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM homework_photos WHERE id=?", (photo_id,))

    def photos_for(self, homework_id: str, student_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM homework_photos WHERE homework_id=? AND student_id=? ORDER BY uploaded_at",
                              (homework_id, student_id))

    def remove_photo(self, photo_id: str) -> None:
        with self._conn_lock:
            self.conn.execute("DELETE FROM homework_photos WHERE id=?", (photo_id,))
            self._commit()


class PhotoResponse(Camel):
    id: str
    content_type: str
    size: int
    uploaded_at: str


def _dir(homework_id: str, student_id: str) -> Path:
    d = Path(cr._cfg.data_root) / "homework-media" / homework_id / student_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _blobs():
    from ..storage.blobs import durable_blob_store
    return durable_blob_store("homework-media")


def _bytes(photo: dict) -> bytes:
    local = _dir(photo["homework_id"], photo["student_id"]) / f"{photo['id']}.{TYPES[photo['content_type']]}"
    if local.exists():
        return local.read_bytes()
    if photo["blob_key"]:
        try:
            return _blobs().download(photo["blob_key"])
        except Exception:  # noqa: BLE001
            log.warning("homework photo %s is not in the blob store", photo["id"], exc_info=True)
    raise HTTPException(404, "this photo is no longer available")


def _out(p: dict) -> PhotoResponse:
    return PhotoResponse(id=p["id"], content_type=p["content_type"], size=p["size"], uploaded_at=p["uploaded_at"])


@router.post("/my-homework/{homework_id}/photos", response_model=PhotoResponse)
async def upload_photo(homework_id: str, file: UploadFile = File(...),
                       current: User = Depends(get_current_user)) -> PhotoResponse:
    """Attach a photo of written work to the caller's own submission. Allowed
    while the homework is open and the work is not yet marked."""
    from . import homework_routes as hr
    hw, _ = hr._my_homework(homework_id, current)
    if hw.status != "published":
        raise HTTPException(409, "this homework is closed")
    sub = store().get_submission(hw.id, current.id)
    if sub is not None and sub.status == "graded":
        raise HTTPException(409, "your homework has been marked; it cannot be changed now")
    if file.content_type not in TYPES:
        raise HTTPException(415, "send a JPEG, PNG or WebP photo")
    if len(store().photos_for(hw.id, current.id)) >= MAX_PHOTOS:
        raise HTTPException(409, f"at most {MAX_PHOTOS} photos per homework")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "a photo may be at most 5 MB")
    if not data:
        raise HTTPException(422, "the photo is empty")
    hr._consent(hw.school_id, current.id)
    blob_key = None
    photo = store().add_photo(homework_id=hw.id, student_id=current.id, content_type=file.content_type,
                              size=len(data), blob_key=None)
    (_dir(hw.id, current.id) / f"{photo['id']}.{TYPES[file.content_type]}").write_bytes(data)
    blobs = _blobs()
    if blobs.enabled:
        key = f"{hw.id}/{current.id}/{photo['id']}.{TYPES[file.content_type]}"
        try:
            blobs.upload(key, data, file.content_type)
            blob_key = key
        except Exception:  # noqa: BLE001
            from ..storage.blobs import record_upload_failure
            record_upload_failure("homework-media")
            log.warning("homework photo %s kept on container disk only", photo["id"], exc_info=True)
    if blob_key:
        with store()._conn_lock:
            store().conn.execute("UPDATE homework_photos SET blob_key=? WHERE id=?", (blob_key, photo["id"]))
            store()._commit()
    return _out(store().get_photo(photo["id"]))


@router.get("/my-homework/{homework_id}/photos", response_model=list[PhotoResponse])
def my_photos(homework_id: str, current: User = Depends(get_current_user)) -> list[PhotoResponse]:
    from . import homework_routes as hr
    hw, _ = hr._my_homework(homework_id, current)
    return [_out(p) for p in store().photos_for(hw.id, current.id)]


@router.get("/my-homework/{homework_id}/photos/{photo_id}")
def my_photo(homework_id: str, photo_id: str, current: User = Depends(get_current_user)) -> Response:
    from . import homework_routes as hr
    hw, _ = hr._my_homework(homework_id, current)
    p = store().get_photo(photo_id)
    if p is None or p["homework_id"] != hw.id or p["student_id"] != current.id:
        raise HTTPException(404, "photo not found")
    return Response(content=_bytes(p), media_type=p["content_type"], headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/my-homework/{homework_id}/photos/{photo_id}")
def delete_my_photo(homework_id: str, photo_id: str, current: User = Depends(get_current_user)) -> dict:
    """Remove a photo before the work is marked."""
    from . import homework_routes as hr
    hw, _ = hr._my_homework(homework_id, current)
    p = store().get_photo(photo_id)
    if p is None or p["homework_id"] != hw.id or p["student_id"] != current.id:
        raise HTTPException(404, "photo not found")
    sub = store().get_submission(hw.id, current.id)
    if sub is not None and sub.status == "graded":
        raise HTTPException(409, "your homework has been marked; it cannot be changed now")
    store().remove_photo(photo_id)
    local = _dir(hw.id, current.id) / f"{photo_id}.{TYPES[p['content_type']]}"
    local.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/homework/{homework_id}/submissions/{student_id}/photos/{photo_id}")
def student_photo(homework_id: str, student_id: str, photo_id: str,
                  current: User = Depends(require_staff)) -> Response:
    """For the teacher marking the work (or the principal); logged."""
    from ..assessment.audit_log import get_audit_log, record_pii_read
    from . import homework_routes as hr
    hw = hr._require_managed(homework_id, current)
    p = store().get_photo(photo_id)
    if p is None or p["homework_id"] != hw.id or p["student_id"] != student_id:
        raise HTTPException(404, "photo not found")
    record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what="homework_photo",
                    student_id=student_id, photo_id=photo_id)
    return Response(content=_bytes(p), media_type=p["content_type"], headers={"Cache-Control": "private, max-age=3600"})
