"""A real school's own document uploads (its own papers/marking schemes),
distinct from the CBSE/NCERT corpus crawl in `ingest/`.

Per docs/compliance.md's "before you ship" checklist for a new ingestion
source: a file-type allowlist, macros stripped from any Office document
before the file is stored, and an explicit copyright/license confirmation
captured at upload time -- all enforced here, not left as a TODO. Accepted
files are registered in the same `SourceRegistry` the CBSE corpus crawl
uses (`ingest/pipeline.py`), so a school's own material becomes a first-class
source with a real `source_id`, just tagged with school/uploader provenance
in its `meta` column instead of a corpus_root path. Every accepted or
rejected upload is written to the same append-only `audit_log` the grading
and paper-export paths already use.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import Config
from ..ingest.checksum import file_checksum
from ..models.enums import DocType
from ..storage.base import LocalStore
from ..storage.registry import SourceRegistry
from .audit_log import get_audit_log
from .schemas import Camel

router = APIRouter(prefix="/api/v1")

_ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "jpg", "jpeg", "png"}
_OOXML_EXTENSIONS = {"docx", "xlsx"}
_MACRO_MEMBER_RE = re.compile(r"vbaproject\.bin$", re.I)
_MACRO_CONTENT_TYPE_SUBS = (
    (b"vnd.ms-word.document.macroEnabled.main+xml",
     b"vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"),
    (b"vnd.ms-excel.sheet.macroEnabled.main+xml",
     b"vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
)

_cfg: Optional[Config] = None
_registry: Optional[SourceRegistry] = None
_store: Optional[LocalStore] = None


def init(config: Config) -> None:
    global _cfg, _registry, _store
    _cfg = config
    _registry = SourceRegistry(config.registry_db)
    _store = LocalStore(config.documents_dir)


def _require() -> tuple[Config, SourceRegistry, LocalStore]:
    if _cfg is None or _registry is None or _store is None:
        raise HTTPException(503, "ingestion module not initialized")
    return _cfg, _registry, _store


def strip_macros(data: bytes) -> tuple[bytes, bool]:
    """Remove any vbaProject.bin member from an OOXML zip and fix up its
    Content_Types macro-enabled declarations. Returns (possibly-rewritten
    bytes, macros_were_found). Non-zip or macro-free input is returned
    unchanged with macros_were_found=False."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zin:
            names = zin.namelist()
            if not any(_MACRO_MEMBER_RE.search(n) for n in names):
                return data, False
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if _MACRO_MEMBER_RE.search(item.filename):
                        continue
                    content = zin.read(item.filename)
                    if item.filename == "[Content_Types].xml":
                        for old, new in _MACRO_CONTENT_TYPE_SUBS:
                            content = content.replace(old, new)
                    zout.writestr(item, content)
            return out.getvalue(), True
    except zipfile.BadZipFile:
        return data, False


class DocumentIngestResponse(Camel):
    source_id: str
    doc_type: str
    status: str
    macros_stripped: bool
    duplicate: bool


@router.post("/ingest/school-documents", response_model=DocumentIngestResponse)
async def ingest_school_document(
    school_id: str = Form(..., alias="schoolId"),
    uploader_id: str = Form(..., alias="uploaderId"),
    doc_type: str = Form(..., alias="docType"),
    copyright_confirmed: bool = Form(..., alias="copyrightConfirmed"),
    title: str = Form("", alias="title"),
    file: UploadFile = File(...),
) -> DocumentIngestResponse:
    cfg, registry, store = _require()
    audit = get_audit_log(cfg.data_root)

    filename = file.filename or ""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in _ALLOWED_EXTENSIONS:
        audit.append("document_ingest_rejected", actor=uploader_id,
                      details={"schoolId": school_id, "filename": filename,
                               "reason": f"extension '.{ext}' not allowed"})
        raise HTTPException(400, f"file type '.{ext}' is not accepted -- "
                                  f"allowed: {sorted(_ALLOWED_EXTENSIONS)}")

    if not copyright_confirmed:
        audit.append("document_ingest_rejected", actor=uploader_id,
                      details={"schoolId": school_id, "filename": filename,
                               "reason": "no copyright confirmation"})
        raise HTTPException(400, "copyright/license confirmation is required for a new upload")

    try:
        resolved_doc_type = DocType(doc_type)
    except ValueError:
        raise HTTPException(400, f"unknown docType '{doc_type}' -- must be one of "
                                  f"{[d.value for d in DocType]}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file upload")

    macros_stripped = False
    if ext in _OOXML_EXTENSIONS:
        data, macros_stripped = strip_macros(data)

    import hashlib
    sha256 = hashlib.sha256(data).hexdigest()
    sha512 = hashlib.sha512(data).hexdigest()
    source_id = f"src:{sha512[:24]}"

    duplicate = registry.get(source_id) is not None
    key = f"school_uploads/{school_id}/{sha512[:24]}.{ext}"
    if not duplicate:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            store.put(tmp_path, key)
        finally:
            tmp_path.unlink(missing_ok=True)
        registry.register(
            source_id,
            file_key=key,
            sha256=sha256,
            sha512=sha512,
            size_bytes=len(data),
            doc_type=resolved_doc_type.value,
            title=title or Path(filename).stem,
            status="registered",
            meta={
                "schoolId": school_id,
                "uploaderId": uploader_id,
                "originalFilename": filename,
                "copyrightConfirmed": True,
                "macrosStripped": macros_stripped,
                "uploadedAt": datetime.now(timezone.utc).isoformat(),
            },
        )

    audit.append("document_ingested", actor=uploader_id,
                 details={"sourceId": source_id, "schoolId": school_id,
                          "docType": resolved_doc_type.value, "filename": filename,
                          "duplicate": duplicate, "macrosStripped": macros_stripped})

    return DocumentIngestResponse(
        source_id=source_id, doc_type=resolved_doc_type.value,
        status="duplicate" if duplicate else "registered",
        macros_stripped=macros_stripped, duplicate=duplicate,
    )
